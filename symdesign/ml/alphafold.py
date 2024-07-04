from __future__ import annotations

import logging
import random
import sys
import time
from collections.abc import MutableMapping, Sequence
from copy import deepcopy
from itertools import count
from typing import Any, Literal, Mapping, Optional, get_args

try:
    # import alphafold.data.feature_processing as af_feature_processing
    # import alphafold.data.parsers as af_data_parsers
    # import alphafold.data.msa_pairing as af_msa_pairing
    # import alphafold.data.pipeline as af_pipeline
    # import alphafold.data.pipeline_multimer as af_pipeline_multimer
    from alphafold.model import features
    from alphafold.model.model import RunModel as AlphaFoldRunModel, get_confidence_metrics
    from alphafold.common import protein, residue_constants
    from alphafold.data.feature_processing import pair_and_merge
    # from alphafold.data.parsers import parse_stockholm
    from alphafold.data.msa_pairing import MSA_FEATURES
    from alphafold.data.pipeline import make_msa_features, make_sequence_features
    from alphafold.data.pipeline_multimer import convert_monomer_features, int_id_to_str_id, pad_msa
    from alphafold.model.config import model_config, MODEL_PRESETS
    from alphafold.model.data import get_model_haiku_params
    from alphafold.notebooks.notebook_utils import empty_placeholder_template_features
    from . import monomer, multimer
    import haiku as hk
    import jax
    from jax import numpy as jnp
    from jaxlib import xla_extension as jax_xla
    import ml_collections
    import tree
    AVAILABLE = True
    print("AlphaFold not available")
except ModuleNotFoundError:
    AVAILABLE = False
    AlphaFoldRunModel = object
import numpy as np

from . import get_device_memory
from symdesign import utils
from symdesign.sequence import numerical_translation_alph1_unknown_gaped_bytes, protein_letters_alph1_unknown_gaped
from symdesign.structure import Entity, Pose
from symdesign.structure.sequence import sequences_to_numeric
from symdesign.structure.utils import chain_id_generator

putils = utils.path
logger = logging.getLogger(__name__)
FeatureDict = MutableMapping[str, np.ndarray]
# Alphafold variables and helpers
MULTIMER_RESIDUE_LIMIT = 4000
MONOMER_RESIDUE_LIMIT = 2500
# Todo enable compilation time savings by returning a precomputed model_factory. Padding the size of this helps greatly
# model_runners = ml.alphafold_model_factory.get()


def log_features(_features: FeatureDict) -> None:
    """Logs the shape and dtype of a FeatureDict"""
    for feat, values in _features.items():
        logger.debug(f'For feature {feat}, found .dtype={values.dtype} and .shape={values.shape}')


# Todo
#  Pose(ContainsEntities, SymmetryOpsMixin)
def get_pose_features(
        pose: Pose, symmetric: bool = False, multimer: bool = False, **kwargs
) -> FeatureDict:
    """Retrieve the required feature dictionary for this instance to use in Alphafold inference

    Args:
        pose: The pose to extract features from
        symmetric: Whether the symmetric version of the Pose should be used for feature production
        multimer: Whether to run as a multimer. If multimer is True while symmetric is False, the Pose will
            be processed according to the ASU

    Keyword Args:
        msas: Sequence - A sequence of multiple sequence alignments if they should be included in the features
        no_msa: bool = False - Whether multiple sequence alignments should be included in the features

    Returns:
        The alphafold FeatureDict which is essentially a dictionary with dict[str, np.ndarray]
    """
    # heteromer = heteromer or pose.number_of_entities > 1
    # symmetric = symmetric or heteromer or pose.number_of_chains > 1
    # if multimer:

    # Set up the ASU unless otherwise specified
    number_of_symmetry_mates = 1
    if pose.is_symmetric():
        if symmetric:
            number_of_symmetry_mates = pose.number_of_symmetry_mates
            multimer = True
    # else:
    #     logger.warning("Can't run as symmetric since Pose isn't symmetric")

    if pose.number_of_entities > 1:
        heteromer = multimer = True
    else:
        heteromer = False
        # if pose.number_of_chains > 1:
        #     multimer = True
        # raise ValueError(f"Can't run monomer when {pose.number_of_entities} entities are present in the "
        #                  f"{repr(pose)}. Use {get_entity_features.__name__}(Entity) on "
        #                  "each entity individually instead")

    chain_count = count(1)
    all_chain_features = {}
    available_chain_ids = list(chain_id_generator())[:pose.number_of_entities * number_of_symmetry_mates]
    available_chain_ids_iter = iter(available_chain_ids)
    for entity_idx, entity in enumerate(pose.entities):
        entity_features = get_entity_features(entity, heteromer=heteromer, **kwargs)  # no_msa=no_msa)
        # The above function creates most of the work for the adaptation
        # particular importance needs to be given to the MSA used.
        # Should fragments be utilized in the MSA? If so, naming them in some way to pair is required!
        # Follow the example in:
        #    make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict
        # to featurize

        if multimer:  # symmetric:
            # The chain_id passed to this function is used by the entity_features to (maybe) tie different chains
            # to this chain. In alphafold implementation, the chain_id passed should be oriented so that each
            # additional entity has the chain_id of the chain number within the entire system.
            # For example, for an A4B4 heteromer with C4 symmetry, the chain_id for entity idx 0 would be A and for
            # entity idx 1 would be E. This may not be important, but this is how symmetric is prepared
            chain_id = available_chain_ids[number_of_symmetry_mates * entity_idx]
            entity_features = convert_monomer_features(entity_features, chain_id=chain_id)

            entity_integer = entity_idx + 1
            entity_id = int_id_to_str_id(entity_integer)
            entity_residue_array = np.ones(entity.number_of_residues)

        # This entire section seems to be repeated from alphafold.get_entity_features(
        # for _ in range(pose.number_of_symmetry_mates):
        for sym_idx in range(1, 1 + number_of_symmetry_mates):
            # chain_id = next(available_chain_ids_iter)  # The mmCIF formatted chainID with 'AB' type notation
            this_entity_features = deepcopy(entity_features)
            # Where chain_id increments for each new chain instance i.e. A_1 is 1, A_2 is 2, ...
            # Where entity_id increments for each new Entity instance i.e. A_1 is 1, A_2 is 1, ...
            # Where sym_id increments for each new Entity instance regardless of chain i.e. A_1 is 1, A_2 is 2, ...,
            # B_1 is 1, B2 is 2
            if multimer:  # symmetric:
                this_entity_features.update({'asym_id': next(chain_count) * entity_residue_array,
                                             'sym_id': sym_idx * entity_residue_array,
                                             'entity_id': entity_integer * entity_residue_array})
                chain_name = f'{entity_id}_{sym_idx}'
            else:
                chain_name = next(available_chain_ids_iter)
            # Make the key '<seq_id>_<sym_id>' where seq_id is the chain name assigned to the Entity where
            # chain names increment according to reverse spreadsheet style i.e. A,B,...AA,BA,...
            # and sym_id increments from 1 to number_of_symmetry_mates
            all_chain_features[chain_name] = this_entity_features
            # all_chain_features[next(available_chain_ids_iter)] = this_entity_features
            # all_chain_features[next(available_chain_ids_iter)] = this_entity_features
            # NOT HERE chain_features = convert_monomer_features(copy.deepcopy(entity_features), chain_id=chain_id)
            # all_chain_features[chain_id] = chain_features

    # This v performed above during all_chain_features creation
    # all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)

    if multimer:  # symmetric:
        all_chain_features = pair_and_merge(all_chain_features=all_chain_features)
        # Pad MSA to avoid zero-sized extra_msa.
        all_chain_features = pad_msa(all_chain_features, 512)

    return all_chain_features


# Todo
#  Entity(SymmetryOpsMixin, StructuredGeneEntity(ContainsResidues))
def get_entity_features(entity: Entity, symmetric: bool = False, heteromer: bool = False, msas: Sequence = tuple(),
                        no_msa: bool = False, templates: bool = False, **kwargs) -> FeatureDict:
    # multimer: bool = False,
    """Retrieve the required feature dictionary for this instance to use in Alphafold inference

    Args:
        entity: The entity to extract features from
        symmetric: Whether the symmetric Entity should be used for feature production. If True, this function will
            fully process the FeatureDict in the symmetric form compatible with Alphafold multimer
        heteromer: Whether Alphafold should be run as a heteromer. Features directly used in
            Alphafold from this instance should never be used with heteromer=True
        msas: A sequence of multiple sequence alignments if they should be included in the features
        no_msa: Whether multiple sequence alignments should be included in the features
        templates: Whether the Entity should be returned with it's template features

    Returns:
        The Alphafold FeatureDict which is essentially a dictionary with dict[str, np.ndarray]
    """
    if heteromer:
        if symmetric:
            raise ValueError(
                f"Couldn't {get_entity_features.__name__} with both 'symmetric' and "
                f"'heteromer' True. Only run with {symmetric=} if {repr(entity)} "
                "is predicted as the only instance in a multimer")
    # if templates:
    #     if symmetric:
    #         # raise ValueError(f"Couldn't {get_entity_features.__name__} with both 'symmetric' and "
    #         logger.warning(f"Couldn't {get_entity_features.__name__} with both 'symmetric' and "
    #                        f"'templates' True. Templates not set up for multimer")
    # elif symmetric:
    #     # Set multimer True as need to make_msa_features_multimeric
    #     multimer = True

    # # IS THIS NECESSARY. DON'T THINK SO IF I HAVE MSA
    # chain_features = alphafold.alphafold.data.pipeline.DataPipeline.process(input_fasta_path=P, msa_output_dir=P)
    # This ^ runs
    number_of_residues = entity.number_of_residues
    sequence = entity.sequence
    sequence_features = make_sequence_features(
        sequence=sequence, description=entity.name, num_res=number_of_residues)
    # sequence_features = {
    #     'aatype': ,  # MAKE ONE HOT with X i.e.unknown are X
    #     'between_segment_residues': np.zeros((number_of_residues,), dtype=np.int32),
    #     'domain_name': np.array([input_description.encode('utf-8')], dtype=np.object_),
    #     'residue_index': np.arange(number_of_residues, dtype=np.int32),
    #     'seq_length': np.full(number_of_residues, number_of_residues, dtype=np.int32),
    #     'sequence': np.array([sequence.encode('utf-8')], dtype=np.object_)
    # }

    def _make_msa_features_multimeric(msa_feats: FeatureDict) -> FeatureDict:
        """Create the feature names for Alphafold heteromeric inputs run in multimer mode"""
        valid_feats = MSA_FEATURES + ('msa_species_identifiers',)
        return {f'{k}_all_seq': v for k, v in msa_feats.items() if k in valid_feats}

    # Multiple sequence alignment processing
    if msas:
        msa_features = make_msa_features(msas)
        # Can use a single one...
        # Other types from AlphaFold include: (uniref90_msa, bfd_msa, mgnify_msa)
        if heteromer:
            # Todo ensure that uniref90 runner was used...
            #  OR equivalent so that each sequence in a multimeric msa is paired
            # Stockholm format looks like
            # #=GF DE                          path/to/profiles/entity_id
            # #=GC RF                          AY--R...
            # 2gtr_1                           AY--R...
            # UniRef100_A0A455ABB#2            NC--R...
            # UniRef100_UPI00106966C#3         CI--L...
            raise NotImplementedError('No uniprot90 database hooked up...')

            uniref90_msa = parse_stockholm(uniref90_lines)
            msa_features = make_msa_features((uniref90_msa,))
            # msa_features.update(_make_msa_features_multimeric(msa_features))
    else:
        msa = entity.msa
        if no_msa or msa is None:
            # When no msa_used, construct our own
            num_sequences = 1
            deletion_matrix = np.zeros((num_sequences, number_of_residues), dtype=np.int32)
            species_ids = ['']  # Must include an empty '' as the first "reference" sequence
            msa_numeric = sequences_to_numeric(
                [sequence], translation_table=numerical_translation_alph1_unknown_gaped_bytes
            ).astype(dtype=np.int32)
        elif msa:
            deletion_matrix = msa.deletion_matrix.astype(np.int32)  # [:, msa.query_indices]
            num_sequences = msa.length
            species_ids = msa.sequence_identifiers
            # Set the msa.alphabet_type to ensure the numerical_alignment is embedded correctly
            msa.alphabet_type = protein_letters_alph1_unknown_gaped
            msa_numeric = msa.numerical_alignment[:, msa.query_indices]
            # logger.critical(f'Found {len(np.flatnonzero(msa.query_indices))} indices utilized in design')
        # Todo
        #  move to additional AlphaFold set up function...
        #  elif os.path.exists(msa_file):
        #      with open(msa_file, 'r') as f:
        #          uniclust_lines = f.read()
        #      file, extension = os.path.splitext(msa_file)
        #      if extension == '.sto':
        #          uniclust30_msa = parse_stockholm(uniclust_lines)
        #      else:
        #          raise ValueError(
        #              f"Currently, the multiple sequence alignment file type '{extension}' isn't supported\n"
        #              f"\tOffending file located at: {msa_file}")
        #      msas = (uniclust30_msa,)
        else:
            raise ValueError("Couldn't acquire AlphaFold msa features")

        logger.debug(f"Found the first 5 species_ids: {species_ids[:5]}")
        msa_features = {
            'deletion_matrix_int': deletion_matrix,
            # When not single sequence, GET THIS FROM THE MATRIX PROBABLY USING CODE IN COLLAPSE PROFILE cumcount...
            # 'msa': sequences_to_numeric([sequence], translation_table=HHBLITS_AA_TO_ID).astype(dtype=np.int32),
            'msa': msa_numeric,
            'num_alignments': np.full(number_of_residues, num_sequences, dtype=np.int32),
            # Fill by the number of residues how many sequences are in the MSA
            'msa_species_identifiers': np.array([id_.encode('utf-8') for id_ in species_ids], dtype=np.object_)
        }
        log_features(msa_features)

    if heteromer:
        # Make a deepcopy just incase this screws up something
        msa_features.update(_make_msa_features_multimeric(deepcopy(msa_features)))

    # Template processing
    if templates:
        template_features = entity.get_alphafold_template_features()  # symmetric=symmetric, heteromer=heteromer)
    else:
        template_features = empty_placeholder_template_features(num_templates=0, num_res=number_of_residues)

    log_features(template_features)

    entity_features = {
        **msa_features,
        **sequence_features,
        **template_features
    }
    if symmetric and entity.is_symmetric():
        # Hard code chain_id as a multimeric prediction done on the oligomeric version
        chain_id = 'A'
        entity_features = convert_monomer_features(entity_features, chain_id=chain_id)

        residue_array = np.ones(number_of_residues)
        chain_count = count(1)
        entity_integer = 1
        entity_id = int_id_to_str_id(entity_integer)
        all_chain_features = {}
        for sym_idx in range(1, 1 + entity.number_of_symmetry_mates):
            # chain_id = next(available_chain_ids_iter)  # The mmCIF formatted chainID with 'AB' type notation
            this_entity_features = deepcopy(entity_features)
            # Where chain_id increments for each new chain instance i.e. A_1 is 1, A_2 is 2, ...
            # Where entity_id increments for each new Entity instance i.e. A_1 is 1, A_2 is 1, ...
            # Where sym_id increments for each new Entity instance regardless of chain i.e. A_1 is 1, A_2 is 2, ...,
            # B_1 is 1, B2 is 2
            this_entity_features.update({'asym_id': next(chain_count) * residue_array,
                                         'sym_id': sym_idx * residue_array,
                                         'entity_id': entity_integer * residue_array})
            chain_name = f'{entity_id}_{sym_idx}'
            all_chain_features[chain_name] = this_entity_features

        # Alternative to pair_and_merge using hhblits a3m output
        # See PMID:36224222 "Structural predictions of dimeric and trimeric subcomponents" methods section
        # The first of the two MSAs is constructed by extracting the organism identifiers (OX) from the resulting
        # a3m file and pairing sequences using the top hit from each OX. The second is constructed by block
        # diagonalizing the resulting a3m file.
        all_chain_features = pair_and_merge(all_chain_features=all_chain_features)
        # Pad MSA to avoid zero-sized extra_msa.
        all_chain_features = pad_msa(all_chain_features, 512)

        return all_chain_features  # This is still a FeatureDict and could be named entity_features
    else:
        return entity_features


def jnp_to_np(jax_dict: dict[str, Any]) -> dict[str, Any]:
    """Recursively changes jax arrays to numpy arrays

    Args:
        jax_dict: A dictionary with the keys mapped to jax.numpy.array types

    Returns:
        The input dictionary modified with the keys mapped to np.array type
    """
    for k, v in jax_dict.items():
        if isinstance(v, dict):
            jax_dict[k] = jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            jax_dict[k] = np.array(v)
    return jax_dict


"""
The following code was modified from Alphafold to allow injection of prev_pos into model initialization
Key difference is the replacement of the Alphafold class with AlphafoldInitialGuess class, largely based on ideas from
the preprint "Improving de novo Protein Binder Design with Deep Learning" bioRxiv
-------------------------------------------------------------------------
Copyright 2021 DeepMind Technologies Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


class RunModel(AlphaFoldRunModel):
    """Container for JAX model."""

    def __init__(self,
                 config: ml_collections.ConfigDict,
                 params: Optional[Mapping[str, Mapping[str, jnp.ndarray]]] = None,
                 # SYMDESIGN
                 device: jax_xla.Device = None):
      """

      Args:
          config:
          params:
          device: The device the model should be compiled on
      """
      # SYMDESIGN
      self.config = config
      self.params = params
      self.multimer_mode = config.model.global_config.multimer_mode
      # SYMDESIGN
      self.parameter_map = {}
      # SYMDESIGN

      if self.multimer_mode:
        def _forward_fn(batch):
          # SYMDESIGN
          model = multimer.AlphaFoldInitialGuess(self.config.model)
          # SYMDESIGN
          return model(
              batch,
              is_training=False)
      else:
        def _forward_fn(batch):
          # SYMDESIGN
          model = monomer.AlphaFoldInitialGuess(self.config.model)
          # SYMDESIGN
          return model(
              batch,
              is_training=False,
              compute_loss=False,
              ensemble_representations=True)

      self.apply = jax.jit(hk.transform(_forward_fn).apply, device=device)
      self.init = jax.jit(hk.transform(_forward_fn).init, device=device)

    def predict(self,
                feat: features.FeatureDict,
                random_seed: int,
                ) -> Mapping[str, Any]:
      """Makes a prediction by inferencing the model on the provided features.

      Args:
        feat: A dictionary of NumPy feature arrays as output by
          RunModel.process_features.
        random_seed: The random seed to use when running the model. In the
          multimer model this controls the MSA sampling.

      Returns:
        A dictionary of model outputs.
      """
      self.init_params(feat)
      logging.info('Running predict with shape(feat) = %s',
                   tree.map_structure(lambda x: x.shape, feat))
      result = self.apply(self.params, jax.random.PRNGKey(random_seed), feat)

      # This block is to ensure benchmark timings are accurate. Some blocking is
      # already happening when computing get_confidence_metrics, and this ensures
      # all outputs are blocked on.
      jax.tree_map(lambda x: x.block_until_ready(), result)
      # SYMDESIGN
      result.update(get_confidence_metrics(result, multimer_mode=self.multimer_mode))
      # SYMDESIGN
      logging.info('Output shape was %s',
                   tree.map_structure(lambda x: x.shape, result))
      return result

    # SYMDESIGN
    def set_params(self, model_params: dict[str, Mapping[str, Mapping[str, jnp.ndarray]]]):
        """Set a collection of parameters that a single compiled model should run

        Args:
            model_params: A dictionary of model parameters

        Returns:
            None
        """
        self.parameter_map = model_params

    def predict_with_params(self, parameter_type: str,
                            feat: features.FeatureDict,
                            random_seed: int,
                            ) -> Mapping[str, Any]:
        """Makes a prediction by inferencing the model on the provided features.

        Args:
            parameter_type: The name of the parameter set to fetch
            feat: A dictionary of NumPy feature arrays as output by
                RunModel.process_features.
            random_seed: The random seed to use when running the model. In the
                multimer model this controls the MSA sampling.

        Returns:
            A dictionary of model outputs.
        """
        logging.info('Running predict with shape(feat) = %s',
                     tree.map_structure(lambda x: x.shape, feat))
        try:
            params = self.parameter_map[parameter_type]
        except KeyError:
            raise KeyError(
                f"The parameter_type='{parameter_type}' isn't available from the viable parameter sets\n"
                f"Currently available types include: {', '.join(self.parameter_map.keys())}")
        result = self.apply(params, jax.random.PRNGKey(random_seed), feat)

        # This block is to ensure benchmark timings are accurate. Some blocking is
        # already happening when computing get_confidence_metrics, and this ensures
        # all outputs are blocked on.
        jax.tree_map(lambda x: x.block_until_ready(), result)
        result.update(get_confidence_metrics(result, multimer_mode=self.multimer_mode))
        logging.info('Output shape was %s',
                     tree.map_structure(lambda x: x.shape, result))
        return result
    # SYMDESIGN


# Todo don't include this...
# Todo reflect the usage in eval_shape from AlphaFoldInitialGuess
# def calculate_alphafold_batch_length(device: jax_xla.Device, number_of_residues: int, element_memory: int = 4) -> int:
#     """
#
#     Args:
#         device: The ProteinMPNN model
#         number_of_residues: The number of residues used in the ProteinMPNN model
#         element_memory: Where each element is np.int64, np.float32, etc.
#
#     Returns:
#         The size of the batch that can be completed for the ProteinMPNN model given it's device
#     """
#     memory_constraint = get_device_memory(device)
#     import torch
#     from symdesign.ml import proteinmpnnbatch_divisor
#
#     number_of_elements_available = memory_constraint // element_memory
#     logger.debug(f'The number_of_elements_available is: {number_of_elements_available}')
#     number_of_model_parameter_elements = sum([math.prod(param.size()) for param in model.parameters()])
#     logger.debug(f'The number_of_model_parameter_elements is: {number_of_model_parameter_elements}')
#     model_elements = number_of_model_parameter_elements
#     # Todo use 5 as ideal CB is added by the model later with ca_only = False
#     num_model_residues = 5
#     model_elements += math.prod((number_of_residues, num_model_residues, 3))  # X,
#     model_elements += number_of_residues  # S.shape
#     model_elements += number_of_residues  # chain_mask.shape
#     model_elements += number_of_residues  # chain_encoding.shape
#     model_elements += number_of_residues  # residue_idx.shape
#     model_elements += number_of_residues  # mask.shape
#     model_elements += number_of_residues  # residue_mask.shape
#     model_elements += math.prod((number_of_residues, 21))  # omit_AA_mask.shape
#     model_elements += number_of_residues  # pssm_coef.shape
#     model_elements += math.prod((number_of_residues, 20))  # pssm_bias.shape
#     model_elements += math.prod((number_of_residues, 20))  # pssm_log_odds_mask.shape
#     model_elements += number_of_residues  # tied_beta.shape
#     model_elements += math.prod((number_of_residues, 21))  # bias_by_res.shape
#     logger.debug(f'The number of model_elements is: {model_elements}')
#
#     number_of_batches = number_of_elements_available // model_elements
#     batch_length = number_of_batches // batch_divisor
#     if batch_length == 0:
#         not_enough_proteinmpnn_memory = f"Can't find a device for {model} with enough memory to complete a single " \
#                                         f"batch of work with {number_of_residues} residues in the model"
#         if device.platform == 'cpu':
#             raise RuntimeError(not_enough_proteinmpnn_memory)
#
#         old_device = device
#         # This won't work. Try to put the model on a new device
#         max_memory = proteinmpnn.vanilla_model_memory
#         for device_int in range(torch.cuda.device_count()):
#             available_memory = get_device_memory(torch.device(device_int), free=True)
#             if available_memory > max_memory:
#                 max_memory = available_memory
#                 device_id = device_int
#         try:
#             device: torch.device = torch.device(device_id)
#         except UnboundLocalError:  # No device has memory greater than ProteinMPNN minimum required
#             device = jax.devices('cpu')[0]
#
#         if device == old_device:
#             # Solve using gpu is stuck
#             if device.type == 'cpu':
#                 # This hasn't been changed or device is cpu
#                 raise RuntimeError(not_enough_proteinmpnn_memory)
#             else:
#                 # Try one more time ensuring cpu. This will be caught above if still not enough memory
#                 device = torch.device('cpu')
#
#         # Recurse
#         return calculate_alphafold_batch_length(model, number_of_residues, element_memory)
#
#     return device


def get_jax_device_memory(device_int: int) -> int:  # jax_xla.Device
    """Based on the device number, use torch to get the device memory"""
    return get_device_memory(device_int)


def alphafold_required_memory(number_of_residues: int):
    """Get the bytes required for the number of residues in the model

    Args:
        number_of_residues: The number of residues in the model
    """
    if number_of_residues > 100:
        return 629_145_600  # '600 M'
    elif number_of_residues > 200:
        return 1_258_291_200  # '1200 M'
    elif number_of_residues > 400:
        return 2_516_582_400  # '2400 M'
    elif number_of_residues > 800:
        return 5_033_164_800  # '4800 M'
    elif number_of_residues > 1000:
        return 6_291_456_000  # '6000 M'
    else:  # Assume 2000+ residues
        return 12_582_912_000  # '12000 M'


def get_alphafold_model_device(number_of_residues: int) -> jax_xla.Device:
    """Get the GPU capable of performing the AlphaFold inference for the number of residues in the model

    Args:
        number_of_residues: The number of residues in the model

    Returns:
        The jax.Device to use with the number of residues present in the model
    """
    use_device = None
    max_memory = alphafold_required_memory(number_of_residues)
    for int, device in enumerate(jax.devices('gpu')):
        available_memory = get_jax_device_memory(int)
        if available_memory > max_memory:
            max_memory = available_memory
            use_device = device

    if use_device is None:
        raise RuntimeError(
            f"Couldn't find a usable device with the memory requirement {max_memory}")
    else:
        return use_device


AlphaFoldModelLiteral = Literal['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer']
af_model_types: tuple[str, ...] = get_args(AlphaFoldModelLiteral)
model_type_to_config_name = {
    'monomer': 'model_1',
    'monomer_casp14': 'model_1',
    'monomer_ptm': 'model_1_ptm',
    'multimer': 'model_1_multimer_v3',
}


def set_up_model_runners(
    model_type: AlphaFoldModelLiteral = 'monomer', number_of_residues: int = 1000, num_predictions_per_model: int = 1,
    num_ensemble: int = 1, development: bool = False
) -> dict[str, RunModel]:
    """Produce Alphafold RunModel class loaded with their training parameters

    Args:
        model_type: The type of model to load. Should be one of the viable Alphafold models including:
            'monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'
        number_of_residues: The number of residues in the model. Used only to calculate approximate memory needs during
            device allocation
        num_predictions_per_model: The number of predictions to make for each Alphafold model. Essentially duplicates
            the original models 'num_predictions_per_model' times
        num_ensemble: The number of model ensembles to make. Typically, 1 is sufficient, but during CASP14, 8 were used
        development: Whether a smaller subset of models should be used for increased testing performance

    Returns:
        A dictionary of the model name to the RunModel instance for each 'model_type'/'num_predictions_per_model'
            requested
    """
    # model_runners = {}
    # model_names = alphafold.MODEL_PRESETS[model_type]
    # for model_name in model_names:
    #     if development and model_name != 'model_2_multimer_v3':
    #         continue
    #     config = model_config(model_name)
    #     if config.model.global_config.multimer_mode:
    #         config.model.num_ensemble_eval = num_ensemble
    #     else:
    #         config.data.eval.num_ensemble = num_ensemble
    #     model_params = alphafold.get_model_haiku_params(model_name=model_name, data_dir=putils.alphafold_db_dir)
    #     # This is using prev_pos init
    #     model_runner = RunModel(config, model_params)
    #     # This should be used if the prediction is not for a design and we have an msa
    #     # model_runner = AlphaFoldRunModel(config, model_params)
    #
    #     for i in range(num_predictions_per_model):
    #         model_runners[f'{model_name}_pred_{i}'] = model_runner
    #
    # num_models = len(model_runners)
    # logger.info(f'Loaded {num_models} Alphafold models: {list(model_runners.keys())}')
    #
    # return model_runners

    # This routine is used to store each separate model parameters on one RunModel
    # Get model config
    config = model_config(model_type_to_config_name[model_type])
    if config.model.global_config.multimer_mode:
        config.model.num_ensemble_eval = num_ensemble
    else:
        config.data.eval.num_ensemble = num_ensemble
    # Set up model params
    model_params = {}
    for model_name in MODEL_PRESETS[model_type]:
        model_param = get_model_haiku_params(model_name=model_name, data_dir=str(putils.alphafold_db_dir))
        if 'model_1' in model_name:
            # Using the config for model_1 as it is most similar to other models
            #  model_1/2 includes template embeddings (monomer),
            #  while multimer model_1 is fairly similar to 2-5
            af_device = get_alphafold_model_device(number_of_residues)
            logger.info(f'Using device {af_device} for model calculations')
            # RunModel is using prev_pos init
            model_runner = RunModel(config, model_param, device=af_device)
            # # ?? Not sure why this would be the case -> if the prediction is not for a design and there is an msa
            # model_runner = AlphaFoldRunModel(config, model_params)
            if development:
                model_params[f'{model_name}_pred_{0}'] = model_param
                break

        for i in range(num_predictions_per_model):
            model_params[f'{model_name}_pred_{i}'] = model_param

    num_models = len(model_params)
    logger.info(f'Loaded {num_models} Alphafold models: {", ".join(model_params)}')

    model_runner.set_params(model_params)
    return {model_param_name: model_runner for model_param_name in model_params}


# Relax
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def amber_relax(prot: protein, gpu: bool = False):
    try:
        from alphafold.relax import amber_minimize
        from alphafold.relax.utils import assert_equal_nonterminal_atom_types
    except ModuleNotFoundError:
        logger.critical(f"\n\nIn order to run relaxation, the 'openmm' and 'pdbfixer' packages are required. "
                        "Retrieve these from conda/mamba then try again.\n\n")
        raise

    out = amber_minimize.run_pipeline(
        prot=prot,
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=gpu)
    # min_pos = out['pos']
    # start_pos = out['posinit']
    # rmsd = np.sqrt(np.sum((start_pos - min_pos) ** 2) / start_pos.shape[0])
    # debug_data = {
    #     'initial_energy': out['einit'],
    #     'final_energy': out['efinal'],
    #     'attempts': out['min_attempts'],
    #     'rmsd': rmsd
    # }
    min_pdb = out['min_pdb']
    # min_pdb = utils.overwrite_b_factors(min_pdb, prot.b_factors)
    assert_equal_nonterminal_atom_types(protein.from_pdb_string(min_pdb).atom_mask, prot.atom_mask)
    violations = out['structural_violations'][
        'total_per_residue_violations_mask'].tolist()

    # return min_pdb, debug_data, violations
    return min_pdb, violations


RelaxOptionsLiteral = Literal['all', 'best', 'none']
relax_options: tuple[str, ...] = get_args(RelaxOptionsLiteral)


def predict(features: FeatureDict, model_runners: dict[str, RunModel], gpu_relax: bool = False,
            models_to_relax: RelaxOptionsLiteral = None, random_seed: int = None,
            confidence_stop_threshold: float = 0.85) -> tuple[dict[str, dict[str, str]], dict[str, FeatureDict]]:
    """Run Alphafold to predict a structure from sequence/msa/template features

    Args:
        # length: The length of the desired output for prediction metrics
        features: The sequence/msa/template feature parameters to populate the jax model
        model_runners: The RunModel instances which should predict the structure
        gpu_relax: Whether predictions should be relaxed using a GPU (if one is available)
        models_to_relax: Specify which predictions should be relaxed
        random_seed: A random integer to seed the model. Could be provided to ensure consistency across runs
        confidence_stop_threshold: The confidence threshold to stop prediction if a prediction scores higher than it.
            Value provided should be between [0-1]. Will use mean plddt if the model is monomer, if model is multimer,
            will use 0.8*interface_predicted_template_modeling_score + 0.2*predicted_template_modeling_score

    Returns:
        The tuple of structure and score dictionaries. Where structures contains the keys 'relaxed' and
        'unrelaxed' mapped to the model name and the model PDB string and folding_scores contain the model name
        mapped to each of the score types 'predicted_aligned_error' (length, length), 'plddt' (length),
        'predicted_template_modeling_score' (1), and 'predicted_interface_template_modeling_score' (1)
    """
    num_models = len(model_runners)
    if random_seed is None:  # Make one
        random_seed = random.randrange(sys.maxsize // num_models)

    if confidence_stop_threshold > 1:
        raise ValueError(f"confidence_stop_threshold must be between 0 and 1. If using monomer models with plddt, "
                         f"will take the confidence metric as the percent plddt (out of a maximum of 100)")

    # # Set up folding_scores dictionary
    # scores = {
    #     'predicted_aligned_error': np.zeros((num_models, length, length), dtype=np.float32),
    #     'plddt': np.zeros((num_models, length), dtype=np.float32),
    #     'predicted_template_modeling_score': np.zeros(num_models, dtype=np.float32),
    #     'predicted_interface_template_modeling_score': np.zeros(num_models, dtype=np.float32)
    # }
    for model_name, model_runner in model_runners.items():
        if model_runner.multimer_mode:
            change_scores = [('iptm', 'predicted_interface_template_modeling_score'),
                             ('ptm', 'predicted_template_modeling_score')]
            # scores_ = {'predicted_template_modeling_score': [],
            #            'predicted_interface_template_modeling_score': []}
        elif 'ptm' in model_name:
            change_scores = [('ptm', 'predicted_template_modeling_score')]
            # scores_ = {'predicted_template_modeling_score': []}
        else:
            change_scores = []
            # raise NotImplementedError()
        break
    else:  # Can't run without model_runners...
        change_scores = []
    #     scores_ = {}
    # scores = {model_name: copy.deepcopy(scores_) for model_name in model_runners}

    unneeded_scores = [
        'distogram', 'experimentally_resolved', 'masked_msa', 'predicted_lddt', 'structure_module',
        'final_atom_positions', 'num_recycles', 'aligned_confidence_probs',
        'max_predicted_aligned_error',
        # 'ranking_confidence',
        # 'ptm', 'iptm', 'predicted_aligned_error', 'plddt',
    ]
    scores = {}
    ranking_confidences = {}
    unrelaxed_proteins = {}
    unrelaxed_pdbs_ = {}
    # Run the models.
    for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
        logger.info(f'Running JAX {model_name}')
        model_random_seed = model_index + random_seed*num_models
        processed_feature_dict = \
            model_runner.process_features(features, random_seed=model_random_seed)

        t_0 = time.time()
        # prediction_result = model_runner.predict(processed_feature_dict, random_seed=model_random_seed)
        prediction_result = \
            model_runner.predict_with_params(model_name, processed_feature_dict, random_seed=model_random_seed)
        logger.info(f'Prediction took {time.time() - t_0:.1f}s')
        # if this is the first go in the model_runner, then f'(includes compilation time)' would be accurate
        # Monomer?
        #  Should take about 96 secs on a 1000 residue protein using 3 recycles...

        # Remove jax dependency from results.
        np_prediction_result = jnp_to_np(dict(prediction_result))
        # logger.debug(f'Found prediction_results: {np_prediction_result}')
        # monomer
        # ['distogram', 'experimentally_resolved', 'masked_msa', 'predicted_lddt', 'structure_module', 'plddt',
        #  'ranking_confidence']
        # multimer
        # ['distogram', 'experimentally_resolved', 'masked_msa', 'predicted_lddt', 'structure_module', 'plddt',
        #  'ranking_confidence'
        #  'num_recycles', 'predicted_aligned_error', 'aligned_confidence_probs', 'max_predicted_aligned_error',
        #  'ptm', 'iptm']
        # logger.debug(f'Found the prediction_result shapes: {model_runner.eval_shape(np_prediction_result)}')
        # {'distogram': {'bin_edges': (63,), 'logits': (n_residues, n_residues, 64)},
        #  'experimentally_resolved': {'logits': (n_residues, atom_types)},
        #  'masked_msa': {'logits': (n_sequences, n_residues, n_amino_acid_types_gapped_unknown)},
        #  'predicted_aligned_error': (n_residues, n_residues),
        #  'predicted_lddt': {'logits': (n_residues, 50)},
        #  'structure_module': {'final_atom_mask': (n_residues, atom_types),
        #  'final_atom_positions': (n_residues, atom_types, 3)},
        #  'plddt': (n_residues,), 'aligned_confidence_probs': (n_residues, n_residues, 64),
        #  'max_predicted_aligned_error': (), 'ptm': (), 'iptm': (), 'ranking_confidence': (), 'num_recycles': (),
        #  }
        # Where ['predicted_lddt'] has the key ['logits'] which probably ?contains the raw logit values produced by
        # model heads? for the binned distogram rankings?

        # plddt = np_prediction_result['plddt']
        # scores[model_name]['plddt'] = plddt  # [:length]
        # if model_runner.multimer_mode:
        #     # This is a 2d array. Clean up to ASU at some point
        #     scores[model_name]['predicted_aligned_error'] = np_prediction_result['predicted_aligned_error']
        #     # scores['predicted_interface_template_modeling_score'][model_index] = np_prediction_result['iptm']
        #     scores[model_name]['predicted_interface_template_modeling_score'].append(np_prediction_result['iptm'])
        #     scores[model_name]['predicted_template_modeling_score'].append(np_prediction_result['ptm'])
        # elif 'ptm' in model_name:
        #     scores[model_name]['predicted_aligned_error'] = \
        #         np_prediction_result['predicted_aligned_error']  # [:length, :length]
        #     scores[model_name]['predicted_template_modeling_score'].append(np_prediction_result['ptm'])

        # Add the predicted LDDT in the b-factor column.
        plddt = np_prediction_result['plddt']
        # Note that higher predicted LDDT value means higher model confidence.
        plddt_b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=not model_runner.multimer_mode)
        unrelaxed_proteins[model_name] = unrelaxed_protein
        unrelaxed_pdbs_[model_name] = protein.to_pdb(unrelaxed_protein)

        ranking_confidences[model_name] = confidence_metric = np_prediction_result.pop('ranking_confidence')
        # Process incoming scores to be returned
        for old_score, new_score in change_scores:
            np_prediction_result[new_score] = np_prediction_result.pop(old_score)
        # Remove unnecessary scores
        for score in unneeded_scores:
            np_prediction_result.pop(score, None)
        scores[model_name] = np_prediction_result

        if model_runner.multimer_mode:
            pass
        else:  # monomer mode. Divide by 100 to get a percentage
            confidence_metric /= 100
        if confidence_metric > confidence_stop_threshold:
            # The prediction quality is already satisfactory
            break

    # Rank model names by model confidence.
    ranked_order = [design_model_name for design_model_name, confidence in
                    sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]
    # Sort the unrelaxed_pdbs accordingly
    unrelaxed_pdbs = {name: unrelaxed_pdbs_.pop(name) for name in ranked_order}

    # Relax predictions.
    relaxed_pdbs = {}
    # relax_metrics = {}
    if models_to_relax is None:
        pass
    else:
        if models_to_relax == 'best':
            to_relax = [ranked_order[0]]
        else:  # if models_to_relax == 'all':
            to_relax = ranked_order

        logger.info(f'Starting Amber relaxation')
        t_0 = time.time()
        for model_name in to_relax:
            logger.info(f'Relaxing {model_name}')
            # relaxed_pdb_str, _, violations = amber_relaxer.process(prot=unrelaxed_proteins[model_name])
            try:
                relaxed_pdb_str, violations = amber_relax(prot=unrelaxed_proteins[model_name], gpu=gpu_relax)
            except ValueError as error:  # Minimization failed after {max_iterations} attempts.
                logger.error(f'Ran into problem during Amber relax: {error}\nSkipping {model_name}')
                continue
            else:
                # relax_metrics[model_name] = {
                #     'remaining_violations': violations,
                #     'remaining_violations_count': sum(violations)
                # }
                relaxed_pdbs[model_name] = relaxed_pdb_str

        logger.info(f'Relaxation took {time.time() - t_0:.1f}s')

    return {'relaxed': relaxed_pdbs, 'unrelaxed': unrelaxed_pdbs}, scores
