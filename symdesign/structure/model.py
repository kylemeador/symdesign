from __future__ import annotations

import abc
import logging
import math
import os
import subprocess
import sys
import time
from collections import UserList, defaultdict
from collections.abc import Container, Generator, Iterable, Iterator, Sequence, MutableMapping
from copy import deepcopy
from itertools import combinations_with_replacement, combinations, product, count
from pathlib import Path
from random import random
from typing import Any, AnyStr, IO, TypedDict, Union
from typing_extensions import assert_never

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree, KDTree
# from sqlalchemy.ext.hybrid import hybrid_property

from .base import ContainsResidues, Residue, Structures, StructureBase, atom_or_residue_literal, \
    SymmetryBase, ContainsAtoms, read_pdb_file, read_mmcif_file
from .coordinates import Coordinates, superposition3d, superposition3d_quat, transform_coordinate_sets
from .fragment import find_fragment_overlap, create_fragment_info_from_pairs
from .fragment.db import alignment_types, alignment_types_literal, FragmentDatabase, FragmentInfo, FragmentObservation
from .fragment.metrics import fragment_metric_template
from .fragment.visuals import write_fragments_as_multimodel
from .sequence import GeneEntity, Profile, pssm_as_array, sequence_to_numeric, sequences_to_numeric, \
    sequence_to_one_hot, aa_counts_alph3, aa_nan_counts_alph3
from .utils import ConstructionError, chain_id_generator, coords_type_literal, default_clash_criteria, \
    default_clash_distance, DesignError, design_programs_literal, StructureException, SymmetryError
from symdesign import metrics, resources, utils
from symdesign.resources import ml, query, sql
from symdesign.sequence import default_substitution_matrix_array, default_substitution_matrix_translation_table, \
    generate_alignment, generate_mutations, get_equivalent_indices, numeric_to_sequence, \
    numerical_translation_alph1_unknown_gaped_bytes, numerical_translation_alph3_unknown_gaped_bytes, \
    protein_letters_3to1_extended, protein_letters_1to3_extended, profile_types, \
    protein_letters_alph1_unknown_gaped, get_lod, protein_letters_alph3
import alphafold.data.feature_processing as af_feature_processing
import alphafold.data.parsers as af_data_parsers
import alphafold.data.msa_pairing as af_msa_pairing
import alphafold.data.pipeline as af_pipeline
import alphafold.data.pipeline_multimer as af_pipeline_multimer
from alphafold.notebooks.notebook_utils import empty_placeholder_template_features
from symdesign.utils import types

FeatureDict = MutableMapping[str, np.ndarray]
BinaryTreeType = Union[BallTree, KDTree]
putils = utils.path

# Globals
logger = logging.getLogger(__name__)
ZERO_OFFSET = 1
SEQRES_LEN = 52
default_atom_contact_distance = 4.68
idx_slice = pd.IndexSlice


def softmax(x: np.ndarray) -> np.ndarray:
    """Take the softmax operation from an input array

    Args:
        x: The array to calculate softmax on. Uses the axis=-1

    Returns:
        The array with a softmax performed
    """
    input_exp = np.exp(x)
    return input_exp / input_exp.sum(axis=-1, keepdims=True)


def split_residue_pairs(interface_pairs: list[tuple[Residue, Residue]]) -> tuple[list[Residue], list[Residue]]:
    """Used to split Residue pairs, take the set, then sort by Residue.number, and return pairs separated by index"""
    if interface_pairs:
        residues1, residues2 = zip(*interface_pairs)
        return sorted(set(residues1), key=lambda residue: residue.number), \
            sorted(set(residues2), key=lambda residue: residue.number)
    else:
        return [], []


# def split_interface_numbers(interface_pairs) -> tuple[list[int], list[int]]:
#     """Used to split residue number pairs"""
#     if interface_pairs:
#         numbers1, numbers2 = zip(*interface_pairs)
#         return sorted(set(numbers1), key=int), sorted(set(numbers2), key=int)
#     else:
#         return [], []


def split_number_pairs_and_sort(pairs: list[tuple[int, int]]) -> tuple[list, list]:
    """Used to split integer pairs and sort, and return pairs separated by index"""
    if pairs:
        numbers1, numbers2 = zip(*pairs)
        return sorted(set(numbers1), key=int), sorted(set(numbers2), key=int)
    else:
        return [], []


def parse_cryst_record(cryst_record: str) -> tuple[list[float], str]:
    """Get the unit cell length, height, width, and angles alpha, beta, gamma and the space group

    Args:
        cryst_record: The CRYST1 record as found in .pdb file format
    """
    try:
        cryst, a, b, c, ang_a, ang_b, ang_c, *space_group = cryst_record.split()
        # a = [6:15], b = [15:24], c = [24:33], ang_a = [33:40], ang_b = [40:47], ang_c = [47:54]
    except ValueError:  # split() or unpacking went wrong
        a = b = c = ang_a = ang_b = ang_c = 0

    return list(map(float, [a, b, c, ang_a, ang_b, ang_c])), cryst_record[55:66].strip()


class MetricsMixin(abc.ABC):
    """Perform Metric evaluation for derived classes

    Subclasses of Metrics must implement _metrics_table property and calculate_metrics() method
    """
    _df: pd.Series
    _metrics: _metrics_table
    # _metrics_table: sql.Base
    _metrics_d: dict[str, Any]
    state_attributes: set[str] = {'_df', '_metrics', '_metrics_d'}

    @abc.abstractmethod
    def calculate_metrics(self, **kwargs) -> dict[str, Any]:
        """Perform Metric calculation for the Entity in question"""

    @property
    @abc.abstractmethod
    def _metrics_table(self) -> sql.Base:
        """The sqlalchemy Mapped class to associate the metrics with"""

    @property
    def _metrics_(self) -> dict[str, Any]:
        """Metrics as a dictionary. __init__: Retrieves all metrics with no arguments"""
        try:
            return self._metrics_d
        except AttributeError:  # Load metrics
            self._metrics_d = self.calculate_metrics()
        return self._metrics_d

    @property
    def metrics(self) -> _metrics_table:
        """Metrics as sqlalchemy Mapped class. __init__: Retrieves all metrics, loads sqlalchemy Mapped class"""
        try:
            return self._metrics
        except AttributeError:  # Load
            self._metrics = self._metrics_table(**self._metrics_)
        return self._metrics

    @property
    def df(self) -> pd.Series:
        """Metrics as a Series. __init__: Retrieves all metrics, loads pd.Series"""
        try:
            return self._df
        except AttributeError:  # Load
            self._df = pd.Series(self._metrics_)
        return self._df

    def clear_metrics(self) -> None:
        """Clear all Metrics.state_attributes for the Entity in question"""
        for attr in MetricsMixin.state_attributes:
            try:
                self.__delattr__(attr)
            except AttributeError:
                continue


class ParseStructureMixin(abc.ABC):

    @classmethod
    def from_file(cls, file: AnyStr, **kwargs):
        """Create a new Structure from a file with Atom records"""
        if '.pdb' in file:
            return cls.from_pdb(file, **kwargs)
        elif '.cif' in file:
            return cls.from_mmcif(file, **kwargs)
        else:
            raise NotImplementedError(
                f"{cls.__name__}: The file type {os.path.splitext(file)[-1]} isn't supported for parsing. Please use "
                f"the supported types '.pdb' or '.cif'. Alternatively use those constructors instead (ex: from_pdb(), "
                'from_mmcif()) if the file extension is nonsense, but the file format is respected.')

    @classmethod
    def from_pdb(cls, file: AnyStr, **kwargs):
        """Create a new Structure from a .pdb formatted file"""
        data = read_pdb_file(file, **kwargs)
        return cls._finish(cls(file_path=file, **data))

    @classmethod
    def from_pdb_lines(cls, pdb_lines: Iterable[str], **kwargs):
        """Create a new Structure from already parsed .pdb file lines"""
        data = read_pdb_file(pdb_lines=pdb_lines, **kwargs)
        return cls._finish(cls(**data))

    @classmethod
    def from_mmcif(cls, file: AnyStr, **kwargs):
        """Create a new Structure from a .cif formatted file"""
        data = read_mmcif_file(file, **kwargs)
        return cls._finish(cls(file_path=file, **data))

    @staticmethod
    def _finish(inst: ParseStructureMixin) -> ParseStructureMixin:  # Todo -> Self python 3.11
        if isinstance(inst, ContainsStructures):
            # Must set this after __init__() to populate .fragment_db in contained Structure instances
            inst.fragment_db = inst.fragment_db

        return inst


default_fragment_contribution = .5


class StructuredGeneEntity(ContainsResidues, GeneEntity):
    """Implements methods to map a Structure to a GeneEntity"""
    _disorder: dict[int, dict[str, str]]
    fragment_map: list[dict[int, set[FragmentObservation]]] | None
    """{1: {-2: {FragObservation(), ...}, 
            -1: {}, ...},
        2: {}, ...}
    Where the outer list indices match Residue.index, and each dictionary holds the various fragment indices
        (with fragment_length length) for that residue, where each index in the inner set can have multiple
        observations
    """

    def __init__(self, metadata: sql.ProteinMetadata = None, uniprot_ids: tuple[str, ...] = None,
                 thermophilicity: bool = None, reference_sequence: str = None, **kwargs):
        """Construct the instance

        Args:
            metadata: Unique database references
            uniprot_ids: The UniProtID(s) that describe this protein sequence
            thermophilicity: The extent to which the sequence is deemed thermophilic
            reference_sequence: The reference sequence (according to expression sequence or reference database)
        """
        super().__init__(**kwargs)  # StructuredGeneEntity
        self._alpha = default_fragment_contribution
        self.alpha = []
        self.fragment_map = None
        self.fragment_profile = None  # fragment specific scoring matrix

        self._api_data = None  # {chain: {'accession': 'Q96DC8', 'db': 'UniProt'}, ...}

        if metadata is None:
            if reference_sequence is not None:
                self._reference_sequence = reference_sequence

            self.thermophilicity = thermophilicity
            if uniprot_ids is not None:
                self.uniprot_ids = uniprot_ids
        else:
            if metadata.reference_sequence is not None:
                self._reference_sequence = metadata.reference_sequence

            self.thermophilicity = metadata.thermophilicity
            if metadata.uniprot_entities is not None:
                self.uniprot_ids = metadata.uniprot_ids

    def clear_api_data(self):
        """Removes any state information from the PDB API"""
        del self._reference_sequence
        self.uniprot_ids = (None,)
        self.thermophilicity = None

    def retrieve_api_metadata(self):
        """Try to set attributes from PDB API

        Sets:
            self._api_data: dict[str, Any]
                {'chains': [],
                 'dbref': {'accession': ('Q96DC8',), 'db': 'UniProt'},
                 'reference_sequence': 'MSLEHHHHHH...',
                 'thermophilicity': True
            self._uniprot_id: str | None
            self._reference_sequence: str
            self.thermophilicity: bool
        """
        entity_id = self.entity_id
        try:
            retrieve_api_info = resources.wrapapi.api_database_factory().pdb.retrieve_data
        except AttributeError:
            retrieve_api_info = query.pdb.query_pdb_by
        api_return = retrieve_api_info(entity_id=entity_id)
        """Get the data on it's own since retrieve_api_info returns
        {'EntityID':
           {'chains': ['A', 'B', ...],
            'dbref': {'accession': ('Q96DC8',), 'db': 'UniProt'},
            'reference_sequence': 'MSLEHHHHHH...',
            'thermophilicity': 1.0},
        ...}
        """
        if api_return:
            if entity_id.lower() in api_return:
                self.name = name = entity_id.lower()

            self._api_data = api_return.get(name, {})
        else:
            self._api_data = {}

        if self._api_data is not None:
            for data_type, data in self._api_data.items():
                # self.log.debug('Retrieving UNP ID for {self.name}\nAPI DATA for chain {chain}:\n{api_data}')
                if data_type == 'reference_sequence':
                    self._reference_sequence = data
                elif data_type == 'thermophilicity':
                    self.thermophilicity = data
                elif data_type == 'dbref':
                    if data.get('db') == query.pdb.UKB:
                        self.uniprot_ids = data.get('accession')
        else:
            self.log.warning(f'{repr(self)}: No information found from PDB API')

    # @hybrid_property
    @property
    def uniprot_ids(self) -> tuple[str | None, ...]:
        """The UniProtID(s) used for accessing external protein level features"""
        try:
            return self._uniprot_ids
        except AttributeError:
            # Set None but attempt to get from the API
            self._uniprot_ids = (None,)
            if self._api_data is None:
                self.retrieve_api_metadata()
        return self._uniprot_ids

    @uniprot_ids.setter
    def uniprot_ids(self, uniprot_ids: Iterable[str] | str):
        if isinstance(uniprot_ids, Iterable):
            self._uniprot_ids = tuple(uniprot_ids)
        elif isinstance(uniprot_ids, str):
            self._uniprot_ids = (uniprot_ids,)
        else:
            raise ValueError(
                f"Couldn't set {self.uniprot_ids.__name__}. Expected Iterable[str] or str, not "
                f"{type(uniprot_ids).__name__}")

    @property
    def reference_sequence(self) -> str:
        """Return the entire sequence, constituting all described residues, not just structurally modeled ones

        Returns:
            The sequence according to the Entity reference, or the Structure sequence if no reference available
        """
        try:
            return self._reference_sequence
        except AttributeError:
            if self._api_data is None:
                self.retrieve_api_metadata()
                try:
                    return self._reference_sequence
                except AttributeError:
                    pass

            self._reference_sequence = self._retrieve_reference_sequence_from_name(self.entity_id)
            if self._reference_sequence is None:
                self.log.info("The reference sequence couldn't be found. Using the Structure sequence instead")
                self._reference_sequence = self.sequence
            return self._reference_sequence

    def _retrieve_reference_sequence_from_name(self, name: str) -> str | None:
        """Using the EntityID, fetch information from the PDB API and set the instance .reference_sequence

        Args:
            name: The EntityID to search for a reference sequence

        Returns:
            The sequence (if located) from the PDB API, otherwise None
        """
        try:
            entry, entity_integer, *_ = name.split('_')
        except ValueError:  # Couldn't unpack correct number of values
            entity_id = self._retrieve_entity_id_from_sequence()
        else:
            if len(entry) == 4 and entity_integer.isdigit():
                entity_id = f'{entry}_{entity_integer}'
            else:
                self.log.debug(
                    f"{self._retrieve_reference_sequence_from_name.__name__}: The provided {name=} isn't the "
                    'correct format (1abc_1), and PDB API query will fail.')
                entity_id = self._retrieve_entity_id_from_sequence()

        if entity_id is None:
            return None
        else:
            self.log.debug(f'Querying {entity_id} reference sequence from PDB')
            return query.pdb.get_entity_reference_sequence(entity_id=entity_id)

    def _retrieve_entity_id_from_sequence(self) -> str | None:
        """Attempts to retrieve the EntityID from the sequence"""
        self.log.debug('Retrieving closest entity_id by PDB API structure sequence using the sequence similarity '
                       f'parameters: {", ".join(f"{k}: {v}" for k, v in query.pdb.default_sequence_values.items())}')
        return query.pdb.retrieve_entity_id_by_sequence(self.sequence)

    def _format_seqres(self, chain_id: str = None) -> str:
        """Format the reference sequence present in the SEQRES remark for writing to the output header

        Args:
            chain_id: The identifier used to name this instances reference sequences

        Returns:
            The .pdb formatted SEQRES record
        """
        _3letter_seq = ' '.join(protein_letters_1to3_extended.get(aa, 'XXX') for aa in self.reference_sequence)
        if chain_id is None:
            try:
                chain_id = getattr(self, 'chain_id')
            except AttributeError:
                chain_id = 'A'

        seq_length = len(self.reference_sequence)
        return '%s\n' \
            % '\n'.join(f'SEQRES{line_number:4d} {chain_id:1s}{seq_length:5d}  '
                        f'{_3letter_seq[SEQRES_LEN * (line_number - 1):SEQRES_LEN * line_number]}         '
                        for line_number in range(1, 1 + math.ceil(len(_3letter_seq) / SEQRES_LEN)))

    @property
    def offset_index(self) -> int:
        """The starting Residue index for the instance. Zero-indexed"""
        return self.residues[0].index

    def add_fragments_to_profile(self, fragments: Iterable[FragmentInfo],
                                 alignment_type: alignment_types_literal, **kwargs):
        """Distribute fragment information to self.fragment_map. Zero-indexed residue array

        Args:
            fragments: The fragment list to assign to the sequence profile with format
                [{'mapped': residue_index1 (int), 'paired': residue_index2 (int), 'cluster': tuple(int, int, int),
                  'match': match_score (float)}]
            alignment_type: Either 'mapped' or 'paired' indicating how the fragment observation was generated relative
                to this GeneEntity. Are the fragments mapped to the ContainsResidues or was it paired to it?

        Sets:
            self.fragment_map (list[list[dict[str, str | float]]]):
                [{-2: {FragObservation(), ...},
                  -1: {}, ...},
                 {}, ...]
                Where the outer list indices match Residue.index, and each dictionary holds the various fragment indices
                (with fragment_length length) for that residue, where each index in the inner set can have multiple
                observations
        """
        if alignment_type not in alignment_types:
            raise ValueError(
                f"Argument 'alignment_type' must be one of '{', '.join(alignment_types)}' not {alignment_type}")

        fragment_db = self.fragment_db
        fragment_map = self.fragment_map
        if fragment_map is None:
            # Create empty fragment_map to store information about each fragment observation in the profile
            self.fragment_map = fragment_map = [defaultdict(set) for _ in range(self.number_of_residues)]

        # Add frequency information to the fragment profile using parsed cluster information. Frequency information is
        # added in a fragment index dependent manner. If multiple fragment indices are present in a single residue, a
        # new observation is created for that fragment index.
        for fragment in fragments:
            # Offset the specified fragment index to the overall index in the ContainsStructures
            fragment_index = getattr(fragment, alignment_type)
            cluster = fragment.cluster
            match = fragment.match
            residue_index = fragment_index - self.offset_index
            # Retrieve the amino acid frequencies for this fragment cluster, for this alignment side
            aa_freq = getattr(fragment_db.info[cluster], alignment_type)
            for frag_idx, frequencies in aa_freq.items():  # (lower_bound - upper_bound), [freqs]
                _frequencies = frequencies.copy()
                _frag_info = FragmentObservation(source=alignment_type, cluster=cluster, match=match,
                                                 weight=_frequencies.pop('weight'), frequencies=_frequencies)
                fragment_map[residue_index + frag_idx][frag_idx].add(_frag_info)

    def simplify_fragment_profile(self, evo_fill: bool = False, **kwargs):
        """Take a multi-indexed, a multi-observation fragment_profile and flatten to single frequency for each residue.

        Weight the frequency of each observation by the fragment indexed, average observation weight, proportionally
        scaled by the match score between the fragment database and the observed fragment overlap

        From the self.fragment_map data, create a fragment profile and add to the GeneEntity

        Args:
            evo_fill: Whether to fill missing positions with evolutionary profile values

        Keyword Args:
            alpha: float = 0.5 - The maximum contribution of the fragment profile to use, bounded between (0, 1].
                0 means no use of fragments in the .profile, while 1 means only use fragments

        Sets:
            self.fragment_profile (Profile)
                [{'A': 0.23, 'C': 0.01, ..., stats': (1, 0.37)}, {...}, ...]
                list of profile_entry that combines all fragment information at a single residue using a weighted
                average. 'count' is number of fragment observations at each residue, and 'weight' is the total
                fragment weight over the entire residue
        """
        # keep_extras: Whether to keep values for all positions that are missing data
        fragment_db = self.fragment_db
        fragment_map = self.fragment_map
        if fragment_map is None:  # Need this for _calculate_alpha()
            raise RuntimeError(
                f'Must {self.add_fragments_to_profile.__name__}() before '
                f'{self.simplify_fragment_profile.__name__}(). No fragments were set')
        elif not fragment_db:
            raise AttributeError(
                f"{self.simplify_fragment_profile.__name__}: No '.fragment_db'. Can't calculate "
                'fragment contribution without one')

        database_bkgnd_aa_freq = fragment_db.aa_frequencies
        # Fragment profile is correct size for indexing all STRUCTURAL residues
        #  self.reference_sequence is not used for this. Instead, self.sequence is used in place since the use
        #  of a disorder indicator that removes any disordered residues from input evolutionary profiles is calculated
        #  on the full reference sequence. This ensures that the profile is the right length of the structure and
        #  captures disorder specific evolutionary signals that could be important in the calculation of profiles
        sequence = self.sequence
        no_design = []
        fragment_profile = [[{} for _ in range(fragment_db.fragment_length)]
                            for _ in range(self.number_of_residues)]
        indexed_observations: dict[int, set[FragmentObservation]]
        for residue_index, indexed_observations in enumerate(fragment_map):
            total_fragment_observations = total_fragment_weight_x_match = total_fragment_weight = 0

            # Sum the weight for each fragment observation
            for index, observations in indexed_observations.items():
                for observation in observations:
                    total_fragment_observations += 1
                    observation_weight = observation.weight
                    total_fragment_weight += observation_weight
                    total_fragment_weight_x_match += observation_weight * observation.match

            # New style, consolidated
            residue_frequencies = {'count': total_fragment_observations,
                                   'weight': total_fragment_weight,
                                   'info': 0.,
                                   'type': sequence[residue_index],
                                   }
            if total_fragment_weight_x_match > 0:
                # Combine all amino acid frequency distributions for all observations at each index
                residue_frequencies.update(**aa_counts_alph3)  # {'A': 0, 'R': 0, ...}
                for index, observations in indexed_observations.items():
                    for observation in observations:
                        # Multiply weight associated with observations by the match of the observation, then
                        # scale the observation weight by the total. If no weight, side chain isn't significant.
                        scaled_frag_weight = observation.weight * observation.match / total_fragment_weight_x_match
                        # Add all occurrences to summed frequencies list
                        for aa, frequency in observation.frequencies.items():
                            residue_frequencies[aa] += frequency * scaled_frag_weight

                residue_frequencies['lod'] = get_lod(residue_frequencies, database_bkgnd_aa_freq)
            else:  # Add to list for removal from the profile
                no_design.append(residue_index)
                # {'A': 0, 'R': 0, ...}
                residue_frequencies.update(lod=aa_nan_counts_alph3.copy(), **aa_nan_counts_alph3)

            # Add results to final fragment_profile residue position
            fragment_profile[residue_index] = residue_frequencies
            # Since self.evolutionary_profile is copied or removed, an empty dictionary is fine here
            # If this changes, maybe the == 0 condition needs an aa_counts_alph3.copy() instead of {}

        if evo_fill and self.evolutionary_profile:
            # If not an empty dictionary, add the corresponding value from evolution
            # For Rosetta, the packer palette is subtractive so the use of an overlapping evolution and
            # null fragment would result in nothing allowed during design...
            evolutionary_profile = self.evolutionary_profile
            for residue_index in no_design:
                fragment_profile[residue_index] = evolutionary_profile.get(residue_index + ZERO_OFFSET)

        # Format into fragment_profile Profile object
        self.fragment_profile = Profile(fragment_profile, dtype='fragment')

        self._calculate_alpha(**kwargs)

    def _calculate_alpha(self, alpha: float = default_fragment_contribution, **kwargs):
        """Find fragment contribution to design with a maximum contribution of alpha. Used subsequently to integrate
        fragment profile during combination with evolutionary profile in calculate_profile

        Takes self.fragment_profile (Profile)
            [{'A': 0.23, 'C': 0.01, ..., stats': [1, 0.37]}, {...}, ...]
        self.fragment_map (list[list[dict[str, str | float]]]):
            [{-2: {FragObservation(), ...},
              -1: {}, ...},
             {}, ...]
            Where the outer list indices match Residue.index, and each dictionary holds the various fragment indices
            (with fragment_length length) for that residue, where each index in the inner set can have multiple
            observations
        and self.fragment_db.statistics (dict)
            {cluster_id1 (str): [[mapped_index_average, paired_index_average,
                                 {max_weight_counts_mapped}, {_paired}],
                                 total_fragment_observations],
             cluster_id2: [], ...,
             frequencies: {'A': 0.11, ...}}
        To identify cluster_id and chain thus returning fragment contribution from the fragment database statistics

        Args:
            alpha: The maximum contribution of the fragment profile to use, bounded between (0, 1].
                0 means no use of fragments in the .profile, while 1 means only use fragments

        Sets:
            self.alpha: (list[float]) - [0.5, 0.321, ...]
        """
        fragment_db = self.fragment_db
        if not fragment_db:
            raise AttributeError(
                f"{self._calculate_alpha.__name__}: No fragment database connected. Can't calculate "
                f'fragment contribution without one')
        if alpha <= 0 or 1 <= alpha:
            raise ValueError(
                f'{self._calculate_alpha.__name__}: Alpha parameter must be bounded between 0 and 1')

        alignment_type_to_idx = {'mapped': 0, 'paired': 1}  # could move to class, but not used elsewhere
        match_score_default_value = 0.5  # When fragment pair RMSD is equal to the mean cluster size RMSD
        bounded_floor = 0.2
        default_match_score_floor = match_score_default_value - bounded_floor
        fragment_stats = fragment_db.statistics
        # self.alpha.clear()  # Reset the data
        alpha = [0 for _ in range(self.number_of_residues)]  # Reset the data
        for entry_idx, (data, fragment_map) in enumerate(zip(self.fragment_profile, self.fragment_map)):
            # Can't use the match count as the fragment index may have no useful residue information
            # Instead use number of fragments with SC interactions count from the frequency map
            frag_count = data.get('count', None)
            if not frag_count:  # When data is missing 'count' or count is 0
                continue  # Move on, this isn't a fragment observation, or no observed fragments
            else:  # Cast as a float
                frag_count = float(frag_count)

            # Match score 'match' is bounded between [0.2, 1]
            match_sum = sum(observation.match
                            for index_observations in fragment_map.values()
                            for observation in index_observations)

            # Find the match modifier which spans from 0 to 1
            average_match_score = match_sum / frag_count
            if average_match_score < match_score_default_value:
                # The match score is below the mean cluster RMSD match score. Subtract the floor to set a useful value
                match_modifier = (average_match_score - bounded_floor) / default_match_score_floor
            else:  # Set modifier to 1, the maximum bound
                match_modifier = 1.

            # Find the total contribution from a typical fragment of this type
            typical_fragment_weight_total = sum(
                fragment_stats[f'{observation.cluster[0]}_{observation.cluster[1]}_0'][0]
                [alignment_type_to_idx[observation.source]] for index_observations in fragment_map.values()
                for observation in index_observations)

            # Get the average contribution of each fragment type
            db_cluster_average = typical_fragment_weight_total / frag_count
            # Get the average fragment weight for this fragment_profile entry. Total weight/count
            frag_weight_average = data.get('weight') / frag_count

            # Find the weight modifier which spans from 0 to 1
            if db_cluster_average > frag_weight_average:
                # Scale the weight modifier by the difference between the cluster and the observations
                weight_modifier = frag_weight_average / db_cluster_average
            else:
                weight_modifier = 1.

            # Modify alpha proportionally to match_modifier in addition to cluster average weight
            alpha[entry_idx] = self._alpha * match_modifier * weight_modifier

        self.alpha = alpha

    def calculate_profile(self, favor_fragments: bool = False, boltzmann: bool = True, **kwargs):
        """Combine weights for profile PSSM and fragment SSM using fragment significance value to determine overlap

        Using self.evolutionary_profile
            (ProfileDict): HHblits - {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...},
                                                 'type': 'W', 'info': 0.00, 'weight': 0.00}, {...}}
                           PSIBLAST - {1: {'A': 0.13, 'R': 0.12, ..., 'lod': {'A': -5, 'R': 2, ...},
                                           'type': 'W', 'info': 3.20, 'weight': 0.73}, {...}}
        self.fragment_profile
            (dict[int, dict[str, float | list[float]]]):
                {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'count': 4, 'weight': 0.274}, 50: {...}, ...}
        self.alpha
            (list[float]): [0., 0., 0., 0.5, 0.321, ...]

        Args:
            favor_fragments: Whether to favor fragment profile in the lod score of the resulting profile
                Currently this routine is only used for Rosetta designs where the fragments should be favored by a
                particular weighting scheme. By default, the boltzmann weighting scheme is applied
            boltzmann: Whether to weight the fragment profile by a Boltzmann probability scaling using the formula
                lods = exp(lods[i]/kT)/Z, where Z = sum(exp(lods[i]/kT)), and kT is 1 by default.
                If False, residues are weighted by the residue local maximum lod score in a linear fashion
                All lods are scaled to a maximum provided in the Rosetta REF2015 per residue reference weight.

        Sets:
            self.profile: (ProfileDict)
                {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...},
                     'type': 'W', 'info': 0.00, 'weight': 0.00}, ...}, ...}
        """
        if self._alpha == 0:
            # Round up to avoid division error
            self.log.warning(f'{self.calculate_profile.__name__}: _alpha set with 1e-5 tolerance due to 0 value')
            self._alpha = 0.000001

        # Copy the evolutionary profile to self.profile (structure specific scoring matrix)
        self.profile = profile = deepcopy(self.evolutionary_profile)
        if sum(self.alpha) == 0:  # No fragments to combine
            return

        # Combine fragment and evolutionary probability profile according to alpha parameter
        fragment_profile = self.fragment_profile
        # log_string = []
        for entry_idx, weight in enumerate(self.alpha):
            # Weight will be 0 if the fragment_profile is empty
            if weight:
                # log_string.append(f'Residue {entry + 1:5d}: {weight * 100:.0f}% fragment weight')
                frag_profile_entry = fragment_profile[entry_idx]
                inverse_weight = 1 - weight
                _profile_entry = profile[entry_idx + ZERO_OFFSET]
                _profile_entry.update({aa: weight * frag_profile_entry[aa] + inverse_weight * _profile_entry[aa]
                                       for aa in protein_letters_alph3})
        # if log_string:
        #     # self.log.info(f'At {self.name}, combined evolutionary and fragment profiles into Design Profile with:'
        #     #               f'\n\t%s' % '\n\t'.join(log_string))
        #     pass

        if favor_fragments:
            fragment_db = self.fragment_db
            boltzman_energy = 1
            favor_seqprofile_score_modifier = 0.2 * utils.rosetta.reference_average_residue_weight
            if not fragment_db:
                raise AttributeError(
                    f"{self.calculate_profile.__name__}: No fragment database connected. Can't 'favor_fragments' "
                    'without one')
            database_bkgnd_aa_freq = fragment_db.aa_frequencies

            null_residue = get_lod(database_bkgnd_aa_freq, database_bkgnd_aa_freq, as_int=False)
            # This was needed in the case of domain errors with lod
            # null_residue = {aa: float(frequency) for aa, frequency in null_residue.items()}

            # Set all profile entries to a null entry first
            for entry, data in self.profile.items():
                data['lod'] = null_residue  # Caution, all reference same object

            alpha = self.alpha
            for entry, data in self.profile.items():
                data['lod'] = get_lod(fragment_profile[entry - ZERO_OFFSET], database_bkgnd_aa_freq, as_int=False)
                # Adjust scores with particular weighting scheme
                partition = 0.
                for aa, value in data['lod'].items():
                    if boltzmann:  # Boltzmann scaling, sum for the partition function
                        value = math.exp(value / boltzman_energy)
                        partition += value
                    else:  # if value < 0:
                        # With linear scaling, remove any lod penalty
                        value = max(0, value)

                    data['lod'][aa] = value

                # Find the maximum/residue (local) lod score
                max_lod = max(data['lod'].values())
                # Takes the percent of max alpha for each entry multiplied by the standard residue scaling factor
                modified_entry_alpha = (alpha[entry - ZERO_OFFSET] / self._alpha) * favor_seqprofile_score_modifier
                if boltzmann:
                    # lods = e ** odds[i]/Z, Z = sum(exp(odds[i]/kT))
                    modifier = partition
                    modified_entry_alpha /= (max_lod / partition)
                else:
                    modifier = max_lod

                # Weight the final lod score by the modifier and the scaling factor for the chosen method
                data['lod'] = {aa: value / modifier * modified_entry_alpha for aa, value in data['lod'].items()}
                # Get percent total (boltzman) or percent max (linear) and scale by alpha score modifier

    @property
    def disorder(self) -> dict[int, dict[str, str]]:
        """Return the Residue number keys where disordered residues are found by comparison of the reference sequence
        with the structure sequence

        Returns:
            Mutation index to mutations in the format of {1: {'from': 'A', 'to': 'K'}, ...}
        """
        try:
            return self._disorder
        except AttributeError:
            self._disorder = generate_mutations(self.reference_sequence, self.sequence, only_gaps=True)
            return self._disorder

    def format_missing_loops_for_design(
        self, max_loop_length: int = 12, exclude_n_term: bool = True, ignore_termini: bool = False, **kwargs
    ) -> tuple[list[tuple], dict[int, int], int]:
        """Process missing residue information to prepare for loop modeling files. Assumes residues in pose numbering!

        Args:
            max_loop_length: The max length for loop modeling.
                12 is the max for accurate KIC as of benchmarks from T. Kortemme, 2014
            exclude_n_term: Whether to exclude the N-termini from modeling due to Remodel Bug
            ignore_termini: Whether to ignore terminal loops in the loop file

        Returns:
            Pairs of indices where each loop starts and ends, adjacent indices (not all indices are disordered) mapped
                to their disordered residue indices, and the n-terminal residue index
        """
        disordered_residues = self.disorder
        # Formatted as {residue_number: {'from': aa, 'to': aa}, ...}
        reference_sequence_length = len(self.reference_sequence)
        loop_indices = []
        loop_to_disorder_indices = {}  # Holds the indices that should be inserted into the total residues to be modeled
        n_terminal_idx = 0  # Initialize as an impossible value
        excluded_disorder_len = 0  # Total residues excluded from loop modeling. Needed for pose numbering translation
        segment_length = 0  # Iterate each missing residue
        n_term = False
        loop_start = loop_end = None
        for idx, residue_number in enumerate(disordered_residues.keys(), 1):
            segment_length += 1
            if residue_number - 1 not in disordered_residues:  # indicate that this residue_number starts disorder
                # print('Residue number -1 not in loops', residue_number)
                loop_start = residue_number - 1 - excluded_disorder_len  # - 1 as loop modeling needs existing residue
                if loop_start < 1:
                    n_term = True

            if residue_number + 1 not in disordered_residues:
                # The segment has ended
                if residue_number != reference_sequence_length:
                    # Not the c-terminus
                    # logger.debug('f{residue_number=} +1 not in loops. Adding loop with {segment_length=}')
                    if segment_length <= max_loop_length:
                        # Modeling useful, add to loop_indices
                        if n_term and (ignore_termini or exclude_n_term):
                            # The n-terminus should be included
                            excluded_disorder_len += segment_length
                            n_term = False  # No more n_term considerations
                        else:  # Include the segment in the disorder_indices
                            loop_end = residue_number + 1 - excluded_disorder_len
                            loop_indices.append((loop_start, loop_end))
                            for it, residue_index in enumerate(range(loop_start + 1, loop_end), 1):
                                loop_to_disorder_indices[residue_index] = residue_number - (segment_length - it)
                            # Set the start and end indices as out of bounds numbers
                            loop_to_disorder_indices[loop_start], loop_to_disorder_indices[loop_end] = -1, -1
                            if n_term and idx != 1:  # If this is the n-termini and not start Met
                                n_terminal_idx = loop_end  # Save idx of last n-term insertion
                    else:  # Modeling not useful, sum the exclusion length
                        excluded_disorder_len += segment_length

                    # After handling disordered segment, reset increment and loop indices
                    segment_length = 0
                    loop_start = loop_end = None
                # Residue number is the c-terminal residue
                elif ignore_termini:
                    if segment_length <= max_loop_length:
                        # loop_end = loop_start + 1 + segment_length  # - excluded_disorder
                        loop_end = residue_number - excluded_disorder_len
                        loop_indices.append((loop_start, loop_end))
                        for it, residue_index in enumerate(range(loop_start + 1, loop_end), 1):
                            loop_to_disorder_indices[residue_index] = residue_number - (segment_length - it)
                        # Don't include start index in the loop_to_disorder map since c-terminal doesn't have attachment
                        loop_to_disorder_indices[loop_end] = -1

        return loop_indices, loop_to_disorder_indices, n_terminal_idx

    def make_loop_file(self, out_path: AnyStr = os.getcwd(), **kwargs) -> AnyStr | None:
        """Format a loops file according to Rosetta specifications. Assumes residues in pose numbering!

        The loop file format consists of one line for each specified loop with the format:

        LOOP 779 784 0 0 1

        Where LOOP specifies a loop line, start idx, end idx, cut site (0 lets Rosetta choose), skip rate, and extended

        All indices should refer to existing locations in the structure file so if a loop should be inserted into
        missing density, the density needs to be modeled first before the loop file would work to be modeled. You
        can't therefore specify that a loop should be between 779 and 780 if the loop is 12 residues long since there is
         no specification about how to insert those residues. This type of task requires a blueprint file.

        Args:
            out_path: The location the file should be written

        Keyword Args:
            max_loop_length=12 (int): The max length for loop modeling.
                12 is the max for accurate KIC as of benchmarks from T. Kortemme, 2014
            exclude_n_term=True (bool): Whether to exclude the N-termini from modeling due to Remodel Bug
            ignore_termini=False (bool): Whether to ignore terminal loops in the loop file

        Returns:
            The path of the file if one was written
        """
        loop_indices, _, _ = self.format_missing_loops_for_design(**kwargs)
        if not loop_indices:
            return None

        loop_file = os.path.join(out_path, f'{self.name}.loops')
        with open(loop_file, 'w') as f:
            f.write('%s\n' % '\n'.join(f'LOOP {start} {stop} 0 0 1' for start, stop in loop_indices))

        return loop_file

    def make_blueprint_file(self, out_path: AnyStr = os.getcwd(), **kwargs) -> AnyStr | None:
        """Format a blueprint file according to Rosetta specifications. Assumes residues in pose numbering!

        The blueprint file format is described nicely here:
            https://www.rosettacommons.org/docs/latest/application_documentation/design/rosettaremodel

        In a gist, a blueprint file consists of entries describing the type of design available at each position.

        Ex:
            1 x L PIKAA M   <- Extension

            1 x L PIKAA V   <- Extension

            1 V L PIKAA V   <- Attachment point

            2 D .

            3 K .

            4 I .

            5 L N PIKAA N   <- Attachment point

            0 x I NATAA     <- Insertion

            0 x I NATAA     <- Insertion

            6 N A PIKAA A   <- Attachment point

            7 G .

            0 X L PIKAA Y   <- Extension

            0 X L PIKAA P   <- Extension

        All structural indices must be specified in "pose numbering", i.e. starting with 1 ending with the last residue.
        If you have missing density in the middle, you should not specify those residues that are missing, but keep
        continuous numbering. You can specify an inclusion by specifying the entry index as 0 followed by the blueprint
        directive. For missing density at the n- or c-termini, the file should still start 1, however, the n-termini
        should be extended by prepending extra entries to the structurally defined n-termini entry 1. These blueprint
        entries should also have 1 as the residue index. For c-termini, extra entries should be appended with the
        indices as 0 like in insertions. For all unmodeled entries for which design should be performed, there should
        be flanking attachment points that are also capable of design. Designable entries are seen above with the PIKAA
        directive. Other directives are available. The only location this isn't required is at the c-terminal attachment
        point

        Args:
            out_path: The location the file should be written

        Keyword Args:
            max_loop_length=12 (int): The max length for loop modeling.
                12 is the max for accurate KIC as of benchmarks from T. Kortemme, 2014
            exclude_n_term=True (bool): Whether to exclude the N-termini from modeling due to Remodel Bug
            ignore_termini=False (bool): Whether to ignore terminal loops in the loop file

        Returns:
            The path of the file if one was written
        """
        disordered_residues = self.disorder
        # Formatted as {residue_number: {'from': aa, 'to': aa}, ...}
        # trying to remove tags at this stage runs into a serious indexing problem where tags need to be deleted from
        # disordered_residues and then all subsequent indices adjusted.

        # # look for existing tag to remove from sequence and save identity
        # available_tags = find_expression_tags(self.reference_sequence)
        # if available_tags:
        #     loop_sequences = ''.join(mutation['from'] for mutation in disordered_residues)
        #     remove_loop_pairs = []
        #     for tag in available_tags:
        #         tag_location = loop_sequences.find(tag['sequences'])
        #         if tag_location != -1:
        #             remove_loop_pairs.append((tag_location, len(tag['sequences'])))
        #     for tag_start, tag_length in remove_loop_pairs:
        #         for
        #
        #     # untagged_seq = remove_terminal_tags(loop_sequences, [tag['sequence'] for tag in available_tags])

        _, disorder_indices, start_idx = self.format_missing_loops_for_design(**kwargs)
        if not disorder_indices:
            return

        residues = self.residues
        # for residue_number in sorted(disorder_indices):  # ensure ascending order, insert dependent on prior inserts
        for residue_index, disordered_residue in disorder_indices.items():
            mutation = disordered_residues.get(disordered_residue)
            if mutation:  # add disordered residue to residues list if they exist
                residues.insert(residue_index - 1, mutation['from'])  # offset to match residues zero-index

        #                 index AA SS Choice AA
        # structure_str   = '%d %s %s'
        # loop_str        = '%d X %s PIKAA %s'
        blueprint_lines = []
        for idx, residue in enumerate(residues, 1):
            if isinstance(residue, Residue):  # use structure_str template
                residue_type = protein_letters_3to1_extended.get(residue.type)
                blueprint_lines.append(f'{residue.number} {residue_type} '
                                       f'{f"L PIKAA {residue_type}" if idx in disorder_indices else "."}')
            else:  # residue is the residue type from above insertion, use loop_str template
                blueprint_lines.append(f'{1 if idx < start_idx else 0} X {"L"} PIKAA {residue}')

        blueprint_file = os.path.join(out_path, f'{self.name}.blueprint')
        with open(blueprint_file, 'w') as f:
            f.write('%s\n' % '\n'.join(blueprint_lines))
        return blueprint_file


class Structure(StructuredGeneEntity, ParseStructureMixin):
    """The base class to handle structural manipulation of groups of Residue instances"""


class ContainsStructures(Structure):
    """Implements methods to interact with a Structure which contains other Structure instances"""
    structure_containers: list | list[str]

    def __init__(self, **kwargs):
        """Construct the instance

        Args:
            **kwargs:
        """
        super().__init__(**kwargs)  # ContainsStructures
        self.structure_containers = []

    @property
    def biological_assembly(self) -> str | None:
        """The integer which maps the structure to an assembly state from the PDB"""
        try:
            return self.metadata.biological_assembly
        except AttributeError:
            raise f"This {repr(self)} isn't the parent and has no metadata 'biological_assembly'"

    @property
    def file_path(self) -> str | None:
        """The integer which maps the structure to an assembly state from the PDB"""
        try:
            return self.metadata.file_path
        except AttributeError:
            raise f"This {repr(self)} isn't the parent and has no metadata 'file_path'"

    @property
    def resolution(self) -> float | None:
        """The integer which maps the structure to an assembly state from the PDB"""
        try:
            return self.metadata.resolution
        except AttributeError:
            raise f"This {repr(self)} isn't the parent and has no metadata 'resolution'"

    # def reset_and_reindex_structures(struct: ContainsResidues, *other_structs: Sequence[ContainsResidues]):
    @staticmethod
    def reset_and_reindex_structures(structs: Sequence[ContainsResidues] | Structures):
        """Given ContainsResidues instances, reset the states and renumber indices in the order passed"""
        struct: ContainsResidues
        other_structs: tuple[ContainsResidues]

        struct, *other_structs = structs
        struct.reset_state()
        struct._start_indices(at=0, dtype='atom')
        struct._start_indices(at=0, dtype='residue')
        prior_struct = struct
        for struct in other_structs:
            struct.reset_state()
            struct._start_indices(at=prior_struct.end_index + 1, dtype='atom')
            struct._start_indices(at=prior_struct.residue_indices[-1] + 1, dtype='residue')
            prior_struct = struct

    @ContainsResidues.fragment_db.setter
    def fragment_db(self, fragment_db: FragmentDatabase):
        """Set the Structure FragmentDatabase to assist with Fragment creation, manipulation, and profiles.
        Sets .fragment_db for each dependent Structure in 'structure_containers'
        """
        # Set this instance then set all dependents
        super(Structure, Structure).fragment_db.fset(self, fragment_db)
        _fragment_db = self._fragment_db
        if _fragment_db is not None:
            for structure_type in self.structure_containers:
                for structure in self.__getattribute__(structure_type):
                    structure.fragment_db = _fragment_db
        else:  # This is likely the RELOAD_DB token. Just return.
            return

    def format_header(self, **kwargs) -> str:
        """Returns any super().format_header() along with the SEQRES records

        Returns:
            The .pdb file header string
        """
        if self.is_parent() and isinstance(self.metadata.cryst_record, str):
            _header = self.metadata.cryst_record
        else:
            _header = ''

        return super().format_header(**kwargs) + self._format_seqres(**kwargs) + _header

    @abc.abstractmethod
    def _format_seqres(self, **kwargs):
        """"""

    def mutate_residue(
        self, residue: Residue = None, index: int = None, number: int = None, to: str = 'A', **kwargs
    ) -> list[int] | list:
        """Mutate a specific Residue to a new residue type. Type can be 1 or 3 letter format

        Args:
            residue: A Residue instance to mutate
            index: A Residue index to select the Residue instance of interest
            number: A Residue number to select the Residue instance of interest
            to: The type of amino acid to mutate to

        Returns:
            The indices of the Atoms being removed from the Structure
        """
        delete_indices = super().mutate_residue(residue=residue, index=index, number=number, to=to)
        if self.is_dependent() or not delete_indices:  # Probably an empty list, there are no indices to delete
            return delete_indices
        structure: Structure

        # Remove delete_indices from each Structure _atom_indices
        # If subsequent structures, update their _atom_indices accordingly
        delete_length = len(delete_indices)
        for structure_type in self.structure_containers:
            residue_found = False
            # Iterate over each Structure in each structure_container
            for structure in self.__getattribute__(structure_type):
                if residue_found:  # The Structure the Residue belongs to is already accounted for, just offset
                    structure._offset_indices(start_at=0, offset=-delete_length, dtype='atom')
                else:
                    try:
                        structure_atom_indices = structure.atom_indices
                        atom_delete_index = structure_atom_indices.index(delete_indices[0])
                    except ValueError:  # When delete_indices[0] isn't in structure_atom_indices
                        continue  # Haven't reached the correct Structure yet
                    else:
                        try:
                            for idx in iter(delete_indices):
                                structure_atom_indices.pop(atom_delete_index)
                        except IndexError:  # When atom_delete_index isn't in structure_atom_indices
                            raise IndexError(
                                f"{self.mutate_residue.__name__}: The index {idx} isn't in the {repr(self)}")
                            # structure._offset_indices(start_at=0, offset=-delete_indices.index(idx), dtype='atom')
                        else:
                            structure._offset_indices(start_at=atom_delete_index, offset=-delete_length, dtype='atom')
                            residue_found = True

                structure.reset_state()

        return delete_indices

    def delete_residues(self, residues: Iterable[Residue] | None = None, indices: Iterable[int] | None = None,
                        numbers: Container[int] | None = None, **kwargs) -> list[Residue] | list:
        """Deletes Residue instances from the Structure

        Args:
            residues: Residue instances to delete
            indices: Residue indices to select the Residue instances of interest
            numbers: Residue numbers to select the Residue instances of interest

        Returns:
            Each deleted Residue
        """
        residues = super().delete_residues(residues=residues, numbers=numbers, indices=indices)
        if self.is_dependent() or not residues:  # There are no Residue instances to delete
            return residues
        structure: Structure

        # The routine below assumes the Residue instances are sorted in ascending order
        atom_index_offset_amount = 0
        for residue_idx, residue in enumerate(residues):
            # Find the Residue, Atom indices to delete
            # Offset these indices if prior indices have already been removed
            atom_delete_indices = [idx - atom_index_offset_amount for idx in residue.atom_indices]
            delete_length = len(atom_delete_indices)
            residue_index = residue.index - residue_idx
            # Offset the next Residue Atom indices by the incrementing amount
            atom_index_offset_amount += delete_length
            for structure_type in self.structure_containers:
                residue_found = False
                # Iterate over each Structure in each structure_container
                for structure in self.__getattribute__(structure_type):
                    if residue_found:
                        # The Structure the Residue belongs to is already accounted for, just offset the indices
                        structure._offset_indices(start_at=0, offset=-delete_length, dtype='atom')
                        structure._offset_indices(start_at=0, offset=-1, dtype='residue')
                    # try:  # Remove atom_delete_indices, residue_indices from Structure
                    elif residue_index not in structure._residue_indices:
                        continue  # This structure is not the one of interest
                    else:  # Remove atom_delete_indices, residue_index from Structure
                        structure._delete_indices(atom_delete_indices, dtype='atom')
                        structure._delete_indices([residue_index], dtype='residue')
                        residue_found = True

                    structure.reset_state()

        return residues

    def insert_residue_type(self, index: int, residue_type: str, chain_id: str = None) -> Residue:
        """Insert a standard Residue type into the Structure based on Pose numbering (1 to N) at the origin.
        No structural alignment is performed.

        Args:
            index: The pose numbered location which a new Residue should be inserted into the Structure
            residue_type: Either the 1 or 3 letter amino acid code for the residue in question
            chain_id: The chain identifier to associate the new Residue with
        """
        new_residue = super().insert_residue_type(index, residue_type, chain_id=chain_id)
        if self.is_dependent():
            return new_residue

        new_residue_atom_indices = new_residue.atom_indices
        structure: Structure

        # Must update other Structures indices
        for structure_type in self.structure_containers:
            structures = self.__getattribute__(structure_type)
            idx = 0
            # Iterate over Structures in each structure_container
            for idx, structure in enumerate(structures, idx):
                try:  # Update each Structure _residue_indices and _atom_indices with additional indices
                    structure._insert_indices(
                        structure.residue_indices.index(index), [index], dtype='residue')
                    structure._insert_indices(
                        structure.atom_indices.index(new_residue.start_index), new_residue_atom_indices, dtype='atom')
                except (ValueError, IndexError):
                    # This should happen if the index isn't in the StructureBase.*_indices of interest
                    # Edge case where the index is being appended to the c-terminus
                    if index - 1 == structure.residue_indices[-1] and new_residue.chain_id == structure.chain_id:
                        structure._insert_indices(structure.number_of_residues, [index], dtype='residue')
                        structure._insert_indices(structure.number_of_atoms, new_residue_atom_indices, dtype='atom')
                    else:
                        continue

                # This was the Structure with insertion and insert_indices proceeded successfully
                structure.reset_state()
                break
            else:  # No matching structure found. The structure_type container should be empty...
                continue
            # For each subsequent structure in the structure container, update the indices with the last index from
            # the prior structure
            prior_structure = structure
            for structure in structures[idx + 1:]:
                structure._start_indices(at=prior_structure.atom_indices[-1] + 1, dtype='atom')
                structure._start_indices(at=prior_structure.residue_indices[-1] + 1, dtype='residue')
                structure.reset_state()
                prior_structure = structure

        return new_residue

    def insert_residues(self, index: int, new_residues: Iterable[Residue], chain_id: str = None) -> list[Residue]:
        """Insert Residue instances into the Structure at the origin. No structural alignment is performed!

        Args:
            index: The index to perform the insertion at
            new_residues: The Residue instances to insert
            chain_id: The chain identifier to associate the new Residue instances with

        Returns:
            The newly inserted Residue instances
        """
        new_residues = super().insert_residues(index, new_residues, chain_id=chain_id)
        if self.is_dependent():
            return new_residues

        number_new_residues = len(new_residues)
        first_new_residue = new_residues[0]
        new_residues_chain_id = first_new_residue.chain_id
        atom_start_index = first_new_residue.start_index
        new_residue_atom_indices = list(range(atom_start_index, new_residues[-1].end_index))
        new_residue_indices = list(range(index, index + number_new_residues))
        structure: Structure
        structures: Iterable[Structure]

        # Must update other Structures indices
        for structure_type in self.structure_containers:
            structures = self.__getattribute__(structure_type)
            idx = 0
            # Iterate over Structures in each structure_container
            for idx, structure in enumerate(structures, idx):
                try:  # Update each Structure _residue_indices and _atom_indices with additional indices
                    structure._insert_indices(
                        structure.residue_indices.index(index), [index], dtype='residue')
                    structure._insert_indices(
                        structure.atom_indices.index(atom_start_index), new_residue_atom_indices, dtype='atom')
                    break  # Move to the next container to update the indices by a set increment
                except (ValueError, IndexError):
                    # This should happen if the index isn't in the StructureBase.*_indices of interest
                    # Edge case where the index is being appended to the c-terminus
                    if index - 1 == structure.residue_indices[-1] and new_residues_chain_id == structure.chain_id:
                        structure._insert_indices(structure.number_of_residues, new_residue_indices, dtype='residue')
                        structure._insert_indices(structure.number_of_atoms, new_residue_atom_indices, dtype='atom')
                    else:
                        continue

                # This was the Structure with insertion and insert_indices proceeded successfully
                structure.reset_state()
                break
            else:  # The target structure wasn't found
                raise DesignError(
                    f"{self.insert_residues.__name__}: Couldn't locate the Structure to be modified by the inserted "
                    f"residues")
            # For each subsequent structure in the structure container, update the indices with the last index from
            # the prior structure
            prior_structure = structure
            for structure in structures[idx + 1:]:
                structure._start_indices(at=prior_structure.atom_indices[-1] + 1, dtype='atom')
                structure._start_indices(at=prior_structure.residue_indices[-1] + 1, dtype='residue')
                structure.reset_state()
                prior_structure = structure

        # self.log.debug(f'Deleted {number_new_residues} Residue instances')

        return new_residues

    def _update_structure_container_attributes(self, **kwargs):
        """Update attributes specified by keyword args for all ContainsResidues members"""
        for structure_type in self.structure_containers:
            for structure in self.__getattribute__(structure_type):
                for kwarg, value in kwargs.items():
                    setattr(structure, kwarg, value)

    def _copy_structure_containers(self):
        """Copy all contained Structure members"""
        # self.log.debug('In ContainsStructures copy_structure_containers()')
        for structure_type in self.structure_containers:
            # Get and copy the structure container
            new_structures = self.__getattribute__(structure_type).copy()
            for idx, structure in enumerate(new_structures):
                new_structures[idx] = structure.copy()
            # Set the copied and updated structure container
            self.__setattr__(structure_type, new_structures)

    def __copy__(self) -> ContainsStructures:  # Todo -> Self: in python 3.11
        other: ContainsStructures = super().__copy__()
        # Set the copying StructureBase attribute ".spawn" to indicate to dependents the "other" of this copy
        if other.is_parent():
            if self.is_parent():
                self.spawn = other
                other._copy_structure_containers()
                # Remove the attribute spawn after member containers are copied
                del self.spawn
            else:  # other is a new parent
                self._update_structure_container_attributes(_copier=True)
                other._copy_structure_containers()
                other._update_structure_container_attributes(_parent=other)
                self._update_structure_container_attributes(_copier=False)

        return other

    copy = __copy__  # Overwrites to use this instance __copy__


class ContainsChains(ContainsStructures):
    """Implements methods to interact with a Structure which contains Chain instances"""
    chain_ids: list[str]
    chains: list[Chain] | Structures
    original_chain_ids: list[str]

    @classmethod
    def from_chains(cls, chains: Sequence[Chain], **kwargs):
        """Create an instance from a Sequence of Chain objects. Automatically renames all chains"""
        return cls(chains=chains, rename_chains=True, **kwargs)

    def __init__(self, chains: bool | Sequence[Chain] = True, chain_ids: Iterable[str] = None,
                 rename_chains: bool = False, as_mates: bool = False, **kwargs):
        """Construct the instance

        Args:
            chain_ids: A list of identifiers to assign to each Chain instance
            chains: Whether to create Chain instances from passed Structure container instances, or existing Chain
                instances to create the Model with
            rename_chains: Whether to name each chain an incrementally new Alphabetical character
            as_mates: Whether Chain instances should be controlled by a captain (True), or be dependents
        """
        super().__init__(**kwargs)  # ContainsChains
        # Use the same list as default to save parsed chain ids
        self.original_chain_ids = self.chain_ids = []
        if chains:  # Populate chains
            self.structure_containers.append('_chains')
            if isinstance(chains, Sequence):
                # Set the chains accordingly, copying them to remove prior relationships
                self._chains = list(chains)
                self._copy_structure_containers()  # Copy each Chain in chains
                if as_mates:
                    if self.residues is None:
                        raise DesignError(
                            f"Couldn't initialize {self.__class__.__name__}.chains as it is missing '.residues' while "
                            f"{as_mates=}"
                        )
                else:  # Create the instance from existing chains
                    self.assign_residues_from_structures(chains)
                    # Reindex all residue and atom indices
                    self.reset_and_reindex_structures(self._chains)
                    # Set the parent attribute for all containers
                    self._update_structure_container_attributes(_parent=self)

                if chain_ids:
                    for chain, id_ in zip(self.chains, chain_ids):
                        chain.chain_id = id_
                # By using extend, self.original_chain_ids are set as well
                self.chain_ids.extend([chain.chain_id for chain in self.chains])
            else:  # Create Chain instances from Residues
                self._chains = []
                self._create_chains(chain_ids=chain_ids)
                if as_mates:
                    for chain in self.chains:
                        chain.make_parent()

            if rename_chains or not self.are_chain_ids_pdb_compatible():
                self.rename_chains()

            self.log.debug(f'Original chain_ids: {",".join(self.original_chain_ids)} | '
                           f'Loaded chain_ids: {",".join(self.chain_ids)}')
        else:
            self._chains = []

        if self.is_parent():
            reference_sequence = self.metadata.reference_sequence
            if isinstance(reference_sequence, dict):  # Was parsed from file
                self.set_reference_sequence_from_seqres(reference_sequence)

    @property
    def chains(self) -> list[Chain]:
        """Returns the Chain instances which are contained in the instance"""
        return self._chains

    def has_dependent_chains(self) -> bool:
        """Returns True if the .chains are dependents, otherwise Returns False if .chains are symmetry mates"""
        return '_chains' in self.structure_containers

    def is_parsed_multimodel(self) -> bool:
        """Returns True if parsing located multiple MODEL records, aka a 'multimodel' or multistate Structure"""
        return self.chain_ids != self.original_chain_ids

    def _create_chains(self, chain_ids: Iterable[str] = None):
        """For all the Residues in the Structure, create Chain objects which contain their member Residues

        Args:
            chain_ids: The desired chain_ids, used in order, for the new chains. Padded if shorter than Chain instances

        Sets:
            self.chain_ids (list[str])
            self.chains (list[Chain] | Structures)
            self.original_chain_ids (list[str])
        """
        residues = self.residues
        try:  # If there are no residues, there is an empty Model
            prior_residue, *other_residues = residues
        except TypeError:   # self.residues is None or contains no Residue instances. Cannot unpack
            return

        residue_idx_start, idx = 0, 1
        chain_residue_indices = []
        for idx, residue in enumerate(other_residues, idx):  # Start at the second index to avoid off by one
            if residue.number <= prior_residue.number or residue.chain_id != prior_residue.chain_id:
                # Less than or equal number should only happen with new chain. this SHOULD satisfy a malformed PDB
                chain_residue_indices.append(list(range(residue_idx_start, idx)))
                residue_idx_start = idx
            prior_residue = residue

        # Perform after iteration to get the final chain
        chain_residue_indices.append(list(range(residue_idx_start, idx + 1)))  # Increment as if next residue

        # To check for multimodel structures, use original_chain_ids.extend() as chain_ids is the same object
        self.original_chain_ids = original_chain_ids = \
            [residues[residue_indices[0]].chain_id for residue_indices in chain_residue_indices]
        unique_chain_ids = utils.remove_duplicates(original_chain_ids)
        if unique_chain_ids != original_chain_ids:
            self.log.debug(f'Multimodel file detected')

        if chain_ids is None:
            chain_ids = unique_chain_ids

        number_of_chain_ids = len(chain_ids)
        number_residue_groups = len(chain_residue_indices)
        if number_of_chain_ids != number_residue_groups:
            # Would be different if a multimodel or user provided differing length chain_ids
            available_chain_ids = chain_id_generator()
            new_chain_ids = []
            for chain_idx in range(number_residue_groups):
                if chain_idx < number_of_chain_ids:  # Use the chain_ids version
                    chain_id = chain_ids[chain_idx]
                else:
                    # Chose next available chain unless already taken, then try another
                    chain_id = next(available_chain_ids)
                    while chain_id in chain_ids:
                        chain_id = next(available_chain_ids)
                new_chain_ids.append(chain_id)

            self.chain_ids = new_chain_ids
        else:
            self.chain_ids = chain_ids

        self._chains.clear()
        for residue_indices, chain_id in zip(chain_residue_indices, self.chain_ids):
            self._chains.append(Chain(residue_indices=residue_indices, name=chain_id, parent=self))

    @property
    def number_of_chains(self) -> int:
        """Return the number of Chain instances in the Structure"""
        return len(self.chains)

    def are_chain_ids_pdb_compatible(self) -> bool:
        """Returns True if the chain_ids are compatible with legacy PDB format"""
        for chain_id in self.chain_ids:
            if len(chain_id) > 1:
                return False

        return True

    def rename_chains(self, exclude_chains: Sequence = None):
        """Renames each chain an incrementally new Alphabetical character using Structure.available_letters

        Args:
            exclude_chains: The chains which shouldn't be modified

        Sets:
            self.chain_ids (list[str])
        """
        if exclude_chains is None:
            exclude_chains = []

        # Update chain_ids, then each chain
        available_chain_ids = chain_id_generator()
        self.chain_ids = []
        for chain in self.chains:
            chain_id = next(available_chain_ids)
            while chain_id in exclude_chains:
                chain_id = next(available_chain_ids)
            chain.chain_id = chain_id
            self.chain_ids.append(chain_id)

    def renumber_residues_by_chain(self):
        """For each Chain instance, renumber Residue objects sequentially starting with 1"""
        for chain in self.chains:
            chain.renumber_residues()

    def get_chain(self, chain_id: str) -> Chain | None:
        """Return the Chain object specified by the passed ChainID from the Structure

        Args:
            chain_id: The name of the Chain to query

        Returns:
            The Chain if one was found
        """
        for idx, id_ in enumerate(self.chain_ids):
            if id_ == chain_id:
                try:
                    return self.chains[idx]
                except IndexError:
                    raise IndexError(
                        f'The number of chains in {repr(self)}, {self.number_of_chains} != {len(self.chain_ids)}, '
                        'the number of .chain_ids')
        return None

    def set_reference_sequence_from_seqres(self, reference_sequence: dict[str, str]):
        """If SEQRES was parsed, set the reference_sequence attribute from each parsed chain_id. Ensure that this is
        called after self._create_chains()
        """
        for original_chain, chain in zip(self.original_chain_ids, self.chains):
            try:
                chain._reference_sequence = reference_sequence[original_chain]
            except KeyError:  # original_chain not parsed in SEQRES
                pass

    def _format_seqres(self, **kwargs) -> str:
        """Format the reference sequence present in the SEQRES remark for writing to the output header

        Returns:
            The .pdb formatted SEQRES record
        """
        return ''.join(struct._format_seqres() for struct in self.chains)

    @staticmethod
    def chain_id_generator() -> Generator[str, None, None]:
        """Provide a generator which produces all combinations of chain ID strings

        Returns
            The generator producing a maximum 2 character string where single characters are exhausted,
                first in uppercase, then in lowercase
        """
        return chain_id_generator()

    @property
    def chain_breaks(self) -> list[int]:
        """Return the index where each of the Chain instances ends, i.e. at the c-terminal Residue"""
        return [structure.c_terminal_residue.index for structure in self.chains]
    #
    # @property
    # def atom_indices_per_chain(self) -> list[list[int]]:
    #     """Return the atom indices for each Chain in the Model"""
    #     return [structure.atom_indices for structure in self.chains]
    #
    # @property
    # def residue_indices_per_chain(self) -> list[list[int]]:
    #     """Return the residue indices for each Chain in the Model"""
    #     return [structure.residue_indices for structure in self.chains]

    def orient(self, symmetry: str = None):
        """Orient a symmetric Structure at the origin with symmetry axis set on canonical axes defined by symmetry file

        Sets the Structure with coordinates as described by a canonical orientation

        Args:
            symmetry: The symmetry of the Structure
        Raises:
            SymmetryError: When the specified symmetry is incompatible with the Structure
            StructureException: When the orient program fails
        """
        # These notes are obviated by the use of the below protocol with from_file() constructor
        # orient_oligomer.f program notes
        # C		Will not work in any of the infinite situations where a PDB file is f***ed up,
        # C		in ways such as but not limited to:
        # C     equivalent residues in different chains don't have the same numbering; different subunits
        # C		are all listed with the same chain ID (e.g. with incremental residue numbering) instead
        # C		of separate IDs; multiple conformations are written out for the same subunit structure
        # C		(as in an NMR ensemble), negative residue numbers, etc. etc.
        try:
            subunit_number = utils.symmetry.valid_subunit_number[symmetry]
        except KeyError:
            raise SymmetryError(
                f"{self.orient.__name__}: Symmetry {symmetry} isn't a valid symmetry. Please try one of: "
                f'{", ".join(utils.symmetry.valid_symmetries)}')

        number_of_subunits = self.number_of_chains
        multicomponent = False
        if symmetry == 'C1':
            self.log.debug("C1 symmetry doesn't have a canonical orientation. Translating to the origin")
            self.translate(-self.center_of_mass)
            return
        elif number_of_subunits > 1:
            if number_of_subunits != subunit_number:
                if number_of_subunits in utils.symmetry.multicomponent_valid_subunit_number.get(symmetry):
                    multicomponent = True
                else:
                    raise SymmetryError(
                        f"{self.name} couldn't be oriented: It has {number_of_subunits} subunits while a multiple of "
                        f'{subunit_number} are expected for symmetry={symmetry}')
        else:
            raise SymmetryError(
                f"{self.name}: Can't orient a Structure with only a single chain. No symmetry present")

        orient_input = Path(putils.orient_exe_dir, 'input.pdb')
        orient_output = Path(putils.orient_exe_dir, 'output.pdb')

        def clean_orient_input_output():
            orient_input.unlink(missing_ok=True)
            orient_output.unlink(missing_ok=True)

        clean_orient_input_output()
        orient_kwargs = {'out_path': str(orient_input)}
        if multicomponent:
            if not isinstance(self, ContainsEntities):
                raise SymmetryError(
                    f"Couldn't {repr(self)}.{self.orient.__name__} as the symmetry is {multicomponent=}, however, there"
                    f" are no .entities in the class {self.__class__.__name__}"
                )
            self.entities[0].write(assembly=True, **orient_kwargs)
        else:
            self.write(**orient_kwargs)

        name = self.name
        p = subprocess.Popen([putils.orient_exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=putils.orient_exe_dir)
        in_symm_file = os.path.join(putils.orient_exe_dir, 'symm_files', symmetry)
        stdout, stderr = p.communicate(input=in_symm_file.encode('utf-8'))
        self.log.debug(name + stdout.decode()[28:])
        self.log.debug(stderr.decode()) if stderr else None
        if not orient_output.exists() or orient_output.stat().st_size == 0:
            try:
                log_file = getattr(self.log.handlers[0], 'baseFilename', None)
            except IndexError:  # No handlers attached
                log_file = None
            log_message = f'. Check {log_file} for more information' if log_file else \
                f': {stderr.decode()}' if stderr else ''
            clean_orient_input_output()
            raise StructureException(
                f"{putils.orient_exe_path} couldn't orient {name}{log_message}")

        oriented_pdb = Model.from_file(str(orient_output), name=self.name, log=self.log)
        orient_fixed_struct = oriented_pdb.chains[0]
        if multicomponent:
            moving_struct = self.entities[0]
        else:
            moving_struct = self.chains[0]

        orient_fixed_seq = orient_fixed_struct.sequence
        moving_seq = moving_struct.sequence

        fixed_coords = orient_fixed_struct.ca_coords
        moving_coords = moving_struct.ca_coords
        if orient_fixed_seq != moving_seq:
            # Do an alignment, get selective indices, then follow with superposition
            self.log.debug(f'{self.orient.__name__}(): existing Chain {moving_struct.chain_id} and '
                           f'oriented Chain {orient_fixed_struct.chain_id} are being aligned for superposition')
            fixed_indices, moving_indices = get_equivalent_indices(orient_fixed_seq, moving_seq)
            fixed_coords = fixed_coords[fixed_indices]
            moving_coords = moving_coords[moving_indices]

        _, rot, tx = superposition3d(fixed_coords, moving_coords)

        self.transform(rotation=rot, translation=tx)
        clean_orient_input_output()


class ContainsEntities(ContainsChains):
    """Implements methods to interact with a Structure which contains Entity instances"""

    @classmethod
    def from_entities(cls, entities: list[Entity] | Structures, rename_chains: bool = True, **kwargs):
        """Construct a Structure instance from a container of Entity objects"""
        if not isinstance(entities, (list, Structures)):
            raise ValueError(
                f"{cls.__name__}.{cls.from_entities.__name__}() constructor received "
                f"'entities'={type(entities).__name__}. Expected list[Entity]"  # or Structures
            )
        return cls(entities=entities, chains=False, rename_chains=rename_chains, **kwargs)

    @property
    def entity_info(self) -> dict[str, dict[dict | list | str]] | dict:
        """Mapping of the Entity name to Metadata describing the Entity instance"""
        if not self.is_parent():
            raise NotImplementedError(
                f"'entity_info' isn't settable for the {repr(self)}"
            )
        return self.metadata.entity_info

    @entity_info.setter
    def entity_info(self, entity_info: dict[str, dict[dict | list | str]] | dict):
        """"""
        if not self.is_parent():
            raise NotImplementedError(
                f"'entity_info' isn't settable for the {repr(self)}"
            )
        self.metadata.entity_info = entity_info

    def __init__(self, entities: bool | Sequence[Entity] = True, entity_info: dict[str, dict[dict | list | str]] = None,
                 **kwargs):
        """Construct the instance

        Args:
            entities: Existing Entity instances used to construct the Structure, or evaluates False to skip creating
                Entity instances from the existing '.chains' Chain instances
            entity_info: Metadata describing the Entity instances
            **kwargs:
        """
        super().__init__(**kwargs)  # ContainsEntities

        self.entity_info = {} if entity_info is None else entity_info

        if entities:
            self.structure_containers.append('_entities')
            if isinstance(entities, Sequence):
                # Create the instance from existing entities
                self.assign_residues_from_structures(entities)
                # Set the entities accordingly, first copying, then resetting, and finally updating the parent
                self._entities = entities
                self._copy_structure_containers()
                self.reset_and_reindex_structures(self._entities)
                self._update_structure_container_attributes(_parent=self)
                rename_chains = kwargs.get('rename_chains')
                if rename_chains:  # Set each successive Entity to have an incrementally higher chain id
                    available_chain_ids = chain_id_generator()
                    for entity in self.entities:
                        entity.chain_id = next(available_chain_ids)
                        # If the full structure wanted contiguous chain_ids, this should be used
                        # for _ in range(entity.number_of_symmetry_mates):
                        #     # Discard ids
                        #     next(available_chain_ids)
                        # self.log.debug(f'Entity {entity.name} new chain identifier {entity.chain_id}')
                    # self.chain_ids.extend([entity.chain_id for entity in self.entities])
            else:  # Provided as True
                self._entities = []
                self._create_entities(**kwargs)

            if not self.chain_ids:
                # Set chain_ids according to self.entities as it wasn't set by self.chains (probably False)
                self.chain_ids.extend([entity.chain_id for entity in self.entities])
        else:
            self._entities = []

        # If any of the entities are symmetric, ensure the new Model is aware they are
        for entity in self.entities:
            if entity.is_symmetric():
                self.symmetric_dependents = '_entities'
                break

        # self.api_entry = None
        # """
        # {'entity': {1: {'A', 'B'}, ...}, 'res': resolution, 'dbref': {chain: {'accession': ID, 'db': UNP}, ...},
        #  'struct': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
        # """
        # Must be in class
        # self.biological_assembly

    @property
    def entities(self) -> list[Entity]:  # | Structures
        """Returns each of the Entity instances in the Structure"""
        return self._entities

    @property
    def number_of_entities(self) -> int:
        """Return the number of Entity instances in the Structure"""
        return len(self.entities)

    # def iterate_over_entity_chains(self) -> Generator[Chain, None, None]:
    #     for idx in range(max(entity.number_of_chains for entity in self.entities)):
    #         for entity in self.entities:
    #             try:
    #                 yield entity.chains[idx]
    #             except IndexError:
    #                 continue
    #
    # self._chains = iterate_over_entity_chains

    def format_header(self, **kwargs) -> str:
        """Return any super().format_header()

        Keyword Args:
            assembly: bool = False - Whether to write header details for the assembly

        Returns:
            The .pdb file header string
        """
        return super().format_header(**kwargs)

    def _format_seqres(self, assembly: bool = False, **kwargs) -> str:
        """Format the reference sequence present in the SEQRES remark for writing to the output header

        Args:
            assembly: Whether to write header details for the assembly

        Returns:
            The .pdb formatted SEQRES record
        """
        if assembly:
            structure_container = self.chains
        else:
            structure_container = self.entities

        return ''.join(struct._format_seqres() for struct in structure_container)

    def _get_entity_info_from_atoms(self, method: str = 'sequence', tolerance: float = 0.9,
                                    length_difference: float = None, **kwargs):
        """Find all unique Entities in the input .pdb file. These are unique sequence objects

        Args:
            method: The method used to extract information. One of 'sequence' or 'structure'
            tolerance: The acceptable difference between chains to consider them the same Entity.
                Alternatively, the use of a structural match should be used. For example, when each chain in an ASU is
                structurally deviating, but shares a sequence with an existing chain
            length_difference: A percentage expressing the maximum length difference acceptable for a matching entity.
                Where the difference in lengths is calculated by the magnitude difference between them divided by the
                larger. For example, 100 and 111 and 100 and 90 would both be ~10% length difference

        Sets:
            self.entity_info
        """
        if not 0 < tolerance <= 1:
            raise ValueError(
                f"{self._get_entity_info_from_atoms.__name__} tolerance={tolerance} isn't allowed. Must be bounded "
                "between (0-1]")
        if length_difference is None:
            length_difference = 1 - tolerance

        entity_start_idx = 1
        if self.entity_info:
            start_with_entity_info = True
            warn_parameters_msg = f"The parameters 'tolerance' and 'length_difference' aren't compatible with the " \
                                  f"chain.sequence and the soon to be Entity sequence.\n" \
                                  f"tolerance={tolerance}\tlength_difference={length_difference}"
            # Remove existing chain IDs
            for data in self.entity_info.values():
                data['chains'] = []
            entity_idx = count(entity_start_idx + len(self.entity_info))
        else:  # Assume that all chains are new_entities
            start_with_entity_info = False
            entity_idx = count(entity_start_idx)

        for chain in self.chains:
            # If the passed parameters don't cover the sequences provided
            warn_bad_parameters = False
            chain_id = chain.chain_id
            chain_sequence = chain.sequence
            numeric_chain_seq = sequence_to_numeric(
                chain_sequence, translation_table=default_substitution_matrix_translation_table)
            perfect_score = default_substitution_matrix_array[numeric_chain_seq, numeric_chain_seq].sum()
            best_score = -10000  # Setting arbitrarily bad
            self.log.debug(
                f'Searching for matching entities for Chain {chain_id} with perfect_score={perfect_score}')
            for entity_name, data in self.entity_info.items():
                # Check if the sequence associated with the Chain is in entity_info
                # Start with the Structure sequence as this is going to be a closer match
                entity_sequence = struct_sequence = data.get('sequence', False)
                if not struct_sequence:  # No Structure sequence in entity_info
                    entity_sequence = data['reference_sequence']
                #     reference_sequence = True
                # else:
                #     reference_sequence = False

                if chain_sequence == entity_sequence:
                    self.log.debug(f'Chain {chain_id} matches Entity {entity_name} sequence exactly')
                    data['chains'].append(chain_id)
                    break

                # Use which ever sequence is greater as the max
                len_seq, len_chain_seq = len(entity_sequence), len(chain_sequence)
                if len_seq >= len_chain_seq:
                    large_sequence_length, small_sequence_length = len_seq, len_chain_seq
                else:  # len_seq < len_chain_seq
                    small_sequence_length, large_sequence_length = len_seq, len_chain_seq

                # Align to find their match
                # generate_alignment('AAAAAAAAAA', 'PPPPPPPPPP').score -> -10
                alignment = generate_alignment(chain_sequence, entity_sequence)
                # score = alignment.score  # Grab score value
                # Find score and bound between 0-1. 1 should be max if it perfectly aligns
                match_score = alignment.score / perfect_score
                if match_score <= 0:  # Throw these away
                    continue
                length_proportion = (large_sequence_length - small_sequence_length) / large_sequence_length
                self.log.debug(f'Chain {chain_id} to Entity {entity_name} has {match_score:.2f} identity '
                               f'and {length_proportion:.2f} length difference with tolerance={tolerance} '
                               f'and length_difference={length_difference}')
                if match_score >= tolerance and length_proportion <= length_difference:
                    self.log.debug(f'Chain {chain_id} matches Entity {entity_name}')
                    # If number of sequence matches is > tolerance, and the length difference < tolerance
                    # the current chain is the same as the Entity, add to chains, and move on to the next chain
                    if start_with_entity_info:
                        # There is information, however, ensure it is the best info
                        # Especially if there isn't a structure_sequence, the reference is used to align
                        # so check all references then set a struct_sequence, i.e. data['sequence'] in the else clause
                        if struct_sequence:
                            # These should've matched perfectly unless there are mutations from model
                            data['chains'].append(chain_id)
                            break
                        else:  # ref_sequence. There is more lenience between the chain.sequence and entity sequence
                            if match_score > best_score:
                                # Set the best_score and best_entity
                                best_score = match_score
                                best_entity = entity_name
                            # Run again to ensure that the there isn't a better choice
                            continue
                    else:  # The entity isn't unique, add to the possible chain IDs
                        data['chains'].append(chain_id)
                        break
                elif start_with_entity_info and match_score > best_score:
                    # Set the best_score and best_entity
                    best_score = match_score
                    best_entity = entity_name
                    warn_bad_parameters = True
            else:  # No match, this is a new Entity
                if start_with_entity_info:
                    # self.log.warning(f"Couldn't find a matching Entity from those existing for Chain {chain_id}")
                    # Set the 'sequence' to the structure sequence
                    try:
                        data = self.entity_info[best_entity]
                    except UnboundLocalError:  # best_entity not set
                        raise DesignError(warn_parameters_msg)
                    data['sequence'] = struct_sequence
                    data['chains'].append(chain_id)
                    del best_entity
                    if warn_bad_parameters:
                        self.log.debug(warn_parameters_msg)
                else:  # No existing entity matches, add new entity
                    entity_name = f'{self.name}_{next(entity_idx)}'
                    self.log.debug(f'Chain {chain_id} is a new Entity "{entity_name}"')
                    self.entity_info[entity_name] = dict(chains=[chain_id], sequence=chain_sequence)

        self.log.debug(f'Entity information was solved by {method} match')

    def retrieve_metadata_from_pdb(self, biological_assembly: int = None) -> dict[str, Any] | dict:
        """Query the PDB API for information on the PDB code found at the Model.name attribute

        For each new instance, makes one call to the PDB API, plus an additional call for each Entity, and one more
        if biological_assembly is passed. If this has been loaded before, it uses the persistent wrapapi.APIDatabase

        Args:
            biological_assembly: The number of the biological assembly that is associated with this structural state

        Sets:
            self.api_entry (dict[str, dict[Any] | float] | dict):
                {'assembly': [['A', 'B'], ...],
                 'entity': {'EntityID':
                                {'chains': ['A', 'B', ...],
                                 'dbref': {'accession': ('Q96DC8',), 'db': 'UniProt'}
                                 'reference_sequence': 'MSLEHHHHHH...',
                                 'thermophilicity': 1.0},
                            ...},
                 'res': resolution,
                 'struct': {'space': space_group, 'a_b_c': (a, b, c),
                            'ang_a_b_c': (ang_a, ang_b, ang_c)}
                }
        """
        # api_entry = self.api_entry
        # if api_entry is not None:  # Already tried to solve this
        #     return

        # if self.api_db:
        try:
            # retrieve_api_info = self.api_db.pdb.retrieve_data
            retrieve_api_info = resources.wrapapi.api_database_factory().pdb.retrieve_data
        except AttributeError:
            retrieve_api_info = query.pdb.query_pdb_by

        # if self.name:  # Try to solve API details from name
        parsed_name = self.name
        splitter_iter = iter('_-')  # 'entity, assembly'
        idx = count(-1)
        extra = None
        while len(parsed_name) != 4:
            try:  # To parse the name using standard PDB API entry ID's
                parsed_name, *extra = parsed_name.split(next(splitter_iter))
            except StopIteration:
                # We didn't find an EntryID in parsed_name from splitting typical PDB formatted strings
                self.log.debug(f"The name '{self.name}' can't be coerced to PDB API format")
                # api_entry = {}
                return {}
            else:
                next(idx)
        # Set the index to the index that was stopped at
        idx = next(idx)

        # At some point, len(parsed_name) == 4
        if biological_assembly is not None:
            # query_args.update(assembly_integer=self.assembly)
            # # self.api_entry.update(_get_assembly_info(self.name))
            api_entry = retrieve_api_info(entry=parsed_name) or {}
            api_entry['assembly'] = retrieve_api_info(entry=parsed_name, assembly_integer=biological_assembly)
            # ^ returns [['A', 'A', 'A', ...], ...]
        elif extra:  # Extra not None or []
            # Try to parse any found extra to an integer denoting entity or assembly ID
            integer, *non_sense = extra
            if integer.isdigit() and not non_sense:
                integer = int(integer)
                if idx == 0:  # Entity integer, such as 1ABC_1.pdb
                    api_entry = dict(entity=retrieve_api_info(entry=parsed_name, entity_integer=integer))
                    # retrieve_api_info returns
                    # {'EntityID': {'chains': ['A', 'B', ...],
                    #               'dbref': {'accession': ('Q96DC8',), 'db': 'UniProt'}
                    #               'reference_sequence': 'MSLEHHHHHH...',
                    #               'thermophilicity': 1.0},
                    #  ...}
                    parsed_name = f'{parsed_name}_{integer}'
                else:  # Get entry alone. This is an assembly or unknown conjugation. Either way entry info is needed
                    api_entry = retrieve_api_info(entry=parsed_name) or {}

                    if idx == 1:  # This is an assembly integer, such as 1ABC-1.pdb
                        api_entry['assembly'] = retrieve_api_info(entry=parsed_name, assembly_integer=integer)
            else:  # This isn't an integer or there are extra characters
                # It's likely they are extra characters that won't be of help
                # Tod0, try to collect anyway?
                self.log.debug(
                    f"The name '{self.name}' contains extra info that can't be coerced to PDB API format")
                api_entry = {}
        elif extra is None:  # Nothing extra as it was correct length to begin with, just query entry
            api_entry = retrieve_api_info(entry=parsed_name)
        else:
            raise RuntimeError(
                f"This logic wasn't expected and shouldn't be allowed to persist: "
                f'self.name={self.name}, parse_name={parsed_name}, extra={extra}, idx={idx}')
        if api_entry:
            self.log.debug(f'Found PDB API information: '
                           f'{", ".join(f"{k}={v}" for k, v in api_entry.items())}')
            # Set the identified name to lowercase
            self.name = parsed_name.lower()
            for entity in self.entities:
                entity.name = entity.name.lower()

        return api_entry

    def _create_entities(self, entity_names: Sequence = None, query_by_sequence: bool = True, **kwargs):
        """Create all Entities in the PDB object searching for the required information if it was not found during
        file parsing. First, search the PDB API using an attached PDB entry_id, dependent on the presence of a
        biological assembly file and/or multimodel file. Finally, initialize them from the Residues in each Chain
        instance using a specified threshold of sequence homology

        Args:
            entity_names: Names explicitly passed for the Entity instances. Length must equal number of entities.
                Names will take precedence over query_by_sequence if passed
            query_by_sequence: Whether the PDB API should be queried for an Entity name by matching sequence. Only used
                if entity_names not provided
        """
        if self.is_parent():
            biological_assembly = self.metadata.biological_assembly
        else:
            biological_assembly = None

        if not self.entity_info:
            # We didn't get info from the file (probably not PDB), so we have to try and piece together.
            # The file is either from a program that has modified an original PDB file, or may be some sort of PDB
            # assembly. If it is a PDB assembly, the only way to know is that the file would have a final numeric suffix
            # after the .pdb extension (.pdb1). If not, it may be an assembly file from another source, in which case we
            # have to solve by using the atomic info

            # First, try to get api data for the structure in case it is a PDB EntryID (or likewise)
            # # self.retrieve_metadata_from_pdb()
            # # api_entry = self.api_entry
            api_entry = self.retrieve_metadata_from_pdb(biological_assembly=biological_assembly)
            if api_entry:  # Not an empty dict
                found_api_entry = True
                if biological_assembly:
                    # As API returns information on the asu, assembly may be different.
                    # We fetch API info for assembly, so we try to reconcile
                    multimodel = self.is_parsed_multimodel()
                    for entity_name, data in api_entry.get('entity', {}).items():
                        chains = data['chains']
                        assembly_data = api_entry.get('assembly', [])
                        for cluster_chains in assembly_data:
                            if not set(cluster_chains).difference(chains):  # nothing missing, correct cluster
                                self.entity_info[entity_name] = data
                                if multimodel:  # Ensure the renaming of chains is handled correctly
                                    self.entity_info[entity_name].update(
                                        {'chains': [
                                            new_chn for new_chn, old_chn in zip(self.chain_ids, self.original_chain_ids)
                                            if old_chn in chains
                                        ]})
                                # else:  # chain names should be the same as assembly API if file is sourced from PDB
                                #     self.entity_info[entity_name] = data
                                break  # we satisfied this cluster, move on
                        else:  # if we didn't satisfy a cluster, report and move to the next
                            self.log.error(f'Unable to find the chains corresponding from Entity {entity_name} to '
                                           f'assembly {biological_assembly} with data {assembly_data}')
                else:  # We can't be certain of the requested biological assembly
                    for entity_name, data in api_entry.get('entity', {}).items():
                        self.entity_info[entity_name] = data
            else:  # Still nothing, the API didn't work for self.name. Solve by atom information
                found_api_entry = False
                # Get rid of any information already acquired
                self.entity_info = {}
                self._get_entity_info_from_atoms(**kwargs)
                if query_by_sequence and entity_names is None:
                    # Copy self.entity_info data for iteration, then reset for re-addition
                    entity_names_to_data = list(self.entity_info.items())
                    self.entity_info = {}
                    for entity_name, data in entity_names_to_data:
                        # Using data['sequence'] here as there is no data['reference_sequence'] from PDB API
                        entity_sequence = data.pop('sequence', None)
                        pdb_api_entity_id = query.pdb.retrieve_entity_id_by_sequence(entity_sequence)
                        if pdb_api_entity_id:
                            new_name = pdb_api_entity_id.lower()
                            self.log.info(f'Entity {entity_name} now named "{new_name}", as found by PDB API '
                                          f'sequence search')
                        else:
                            self.log.info(f"Entity {entity_name} couldn't be located by PDB API sequence search")
                            # Set as the reference_sequence because we won't find one without another database...
                            data['reference_sequence'] = entity_sequence
                            new_name = entity_name
                        self.entity_info[new_name] = data
        else:
            # self.log.debug(f"_create_entities entity_info={self.entity_info}")
            # This is being set to True because there is API info in the self.entity_info (If passed correctly)
            found_api_entry = True

        if entity_names is not None:
            renamed_entity_info = {}
            for idx, (entity_name, data) in enumerate(self.entity_info.items()):
                try:
                    new_entity_name = entity_names[idx]
                except IndexError:
                    raise IndexError(
                        f'The number of indices in entity_names, {len(entity_names)} != {len(self.entity_info)}, the '
                        f'number of entities in the {self.__class__.__name__}')

                # Get any info already solved using the old name
                renamed_entity_info[new_entity_name] = data
                self.log.debug(f'Entity {entity_name} now named "{new_entity_name}", as supplied by entity_names')
            # Reset self.entity_info with the same order, but new names
            self.entity_info = renamed_entity_info

        # Check to see that the parsed entity_info is compatible with the chains already parsed
        if found_api_entry:
            # Set if self.retrieve_metadata_from_pdb was called above or entity_info provided to Model construction
            if self.nucleotides_present:
                self.log.warning(f"Integration of nucleotides hasn't been worked out yet, API information not useful")

            max_reference_sequence = 0
            for data in self.entity_info.values():
                reference_sequence_length = len(data['reference_sequence'])
                if reference_sequence_length > max_reference_sequence:
                    max_reference_sequence = reference_sequence_length
            # Find the minimum chain sequence length
            min_chain_sequence = sys.maxsize
            for chain in self.chains:
                chain_sequence_length = chain.number_of_residues
                if chain_sequence_length < min_chain_sequence:
                    min_chain_sequence = chain_sequence_length

            # Use the Structure attributes to get the entity info correct
            # Provide an expected tolerance and length_difference
            # Sequence could be highly mutated compared to the reference, so set the tolerance threshold to a percentage
            # of the maximum. This value seems to be close given Sanders and Sanders? 1994 (BLOSUM matrix publication)
            tolerance = .2  # .3 <- Some ProteinMPNN sequences failed this bar
            # Because the reference sequence could be much longer than the chain sequence,
            # find an 'expected' length proportion
            # length_proportion = (max_reference_sequence-min_chain_sequence) / max_reference_sequence
            length_proportion = min_chain_sequence / max_reference_sequence
            self._get_entity_info_from_atoms(tolerance=tolerance, length_difference=length_proportion, **kwargs)
        else:
            entity_api_entry = {}
            api_entry = {'entity': entity_api_entry}
            # entity_api_entry = self.api_entry.get('entity', {})
            # if not entity_api_entry:
            #     self.api_entry['entity'] = entity_api_entry
            # if self.api_db:
            try:
                # retrieve_api_info = self.api_db.pdb.retrieve_data
                retrieve_api_info = resources.wrapapi.api_database_factory().pdb.retrieve_data
            except AttributeError:
                retrieve_api_info = query.pdb.query_pdb_by

            for entity_name, data in self.entity_info.items():
                # update_entity_info_from_api(new_entity_name)
                # def update_entity_info_from_api(entity_name: str):
                entity_api_data: dict = retrieve_api_info(entity_id=entity_name)
                """entity_api_data takes the format:
                {'EntityID': 
                    {'chains': ['A', 'B', ...],
                     'dbref': {'accession': ('Q96DC8',), 'db': 'UniProt'},
                     'reference_sequence': 'MSLEHHHHHH...',
                     'thermophilicity': 1.0},
                 ...}
                This is the final format of each entry in the self.entity_info dictionary
                """
                # Set the parent self.api_entry['entity']
                # If the entity_name is already present, it's expected that self.entity_info is already solved
                if entity_api_data:  # and entity_name not in entity_api_entry:
                    entity_api_entry.update(entity_api_data)
                    # Respect already solved 'chains' info in self.entity_info
                    if data.get('chains', {}):
                        # Remove entity_api_data 'chains'
                        entity_api_data[entity_name].pop('chains')
                    # Update the entity_info with the entity_api_data
                    data.update(entity_api_data[entity_name])

        # Finish processing by cleaning data and preparing for Entity()
        for entity_name, data in list(self.entity_info.items()):
            # For each Entity, get matching Chain instances
            # Add any missing information to the individual data dictionary
            # dbref = data.get('dbref', None)
            # # if dbref is None:
            # #     dbref = {}
            # entity_data['dbref'] = data['dbref'] = dbref
            uniprot_ids = None
            # These aren't used anymore
            entity_sequence = data.pop('sequence', None)
            dbref = data.pop('dbref', None)
            if dbref is not None:
                db_source = dbref.get('db')
                if db_source == query.pdb.UKB:  # This is a protein
                    uniprot_ids = dbref['accession']
                elif db_source == query.pdb.GB:  # Nucleotide
                    self.log.critical(f'Found a PDB API database source of {db_source} for the Entity {entity_name}'
                                      f'This is currently not parsable')
                elif db_source == query.pdb.NOR:
                    self.log.critical(f'Found a PDB API database source of {db_source} for the Entity {entity_name}'
                                      f'This is currently not parsable')
            # else:
            #     data['dbref'] = None

            if 'reference_sequence' not in data:
                # This is only possible when self.entity_info was set during __init__()
                # This should not be possible with the passing of an explicit list[sql.ProteinMetadata class]
                # new_reference_sequence = data.get('sequence')
                # if new_reference_sequence is None:
                # Try to set using the entity_name
                new_reference_sequence = query.pdb.get_entity_reference_sequence(entity_id=entity_name)
                data['reference_sequence'] = new_reference_sequence

            # Set up a new dictionary with the modified keyword 'chains' which refers to Chain instances
            data_chains = utils.remove_duplicates(data.get('chains', []))
            chains = [self.get_chain(chain_id) if isinstance(chain_id, str) else chain_id
                      for chain_id in data_chains]
            entity_chains = [chain for chain in chains if chain]
            entity_data = {
                **data,  # Place the original data in the new dictionary
                'chains': entity_chains,  # Overwrite chains in data dictionary
                'uniprot_ids': uniprot_ids,
            }
            if len(entity_chains) == 0:
                if self.nucleotides_present:
                    self.log.warning(f'Nucleotide chain was already removed from Structure')
                else:
                    # This occurred when there were 2 entity records in the entity_info but only 1 in the Structure
                    self.log.debug(f'Missing associated chains for the Entity {entity_name} with data: '
                                   f"self.chain_ids={self.chain_ids}, entity_data['chains']={entity_chains}, "
                                   f"data['chains']={data_chains}, "
                                   f'{", ".join(f"{k}={v}" for k, v in data.items())}')
                    # Drop this section in the entity_info
                    self.log.warning(f'Dropping Entity {entity_name} from {self.__class__.__name__} as no Structure '
                                     f'information exists for it')
                    self.entity_info.pop(entity_name)
                    # raise DesignError(f"The Entity couldn't be processed as currently configured")
                continue
            #     raise utils.DesignError('Missing Chain object for %s %s! entity_info=%s, assembly=%s and '
            #                             'api_entry=%s, original_chain_ids=%s'
            #                             % (self.name, self._create_entities.__name__, self.entity_info,
            #                             self.biological_assembly, self.api_entry, self.original_chain_ids))
            # entity_data has attributes chains, dbref, and reference_sequence
            # entity_data.pop('dbref')  # This isn't used anymore
            entity = Entity.from_chains(**entity_data, name=entity_name, parent=self)
            for chain in entity_chains:
                chain._entity = entity

            self.entities.append(entity)

    @property
    def entity_breaks(self) -> list[int]:
        """Return the index where each of the Entity instances ends, i.e. at the c-terminal Residue"""
        return [entity.c_terminal_residue.index for entity in self.entities]

    def get_entity(self, entity_id: str) -> Entity | None:
        """Retrieve an Entity by name

        Args:
            entity_id: The name of the Entity to query

        Returns:
            The Entity if one was found
        """
        for entity in self.entities:
            if entity_id == entity.name:
                return entity
        return None

    def entity_from_chain(self, chain_id: str) -> Entity | None:
        """Returns the entity associated with a particular chain id"""
        for entity in self.entities:
            if chain_id == entity.chain_id:
                return entity
        return None

    def match_entity_by_seq(
        self, other_seq: str = None, force_closest: bool = True, tolerance: float = 0.7
    ) -> Entity | None:
        """From another sequence, returns the first matching chain from the corresponding Entity

        Uses a local alignment to produce the match score

        Args:
            other_seq: The sequence to query
            force_closest: Whether to force the search if a perfect match isn't identified
            tolerance: The acceptable difference between sequences to consider them the same Entity.
                Tuning this parameter is necessary if you have sequences which should be considered different entities,
                but are fairly similar

        Returns:
            The matching Entity if one was found
        """
        for entity in self.entities:
            if other_seq == entity.sequence:
                return entity

        # We didn't find an ideal match
        if force_closest:
            entity_alignment_scores = {}
            for entity in self.entities:
                alignment = generate_alignment(other_seq, entity.sequence, local=True)
                entity_alignment_scores[entity] = alignment.score

            max_score, max_score_entity = 0, None
            for entity, score in entity_alignment_scores.items():
                normalized_score = score / len(entity.sequence)
                if normalized_score > max_score:
                    max_score = normalized_score  # alignment_score_d[entity]
                    max_score_entity = entity

            if max_score > tolerance:
                return max_score_entity

        return None

    @property
    def sequence(self) -> str:
        """Return the sequence of structurally modeled residues for every Entity instance

        Returns:
            The concatenated sequence for all Entity instances combined
        """
        return ''.join(entity.sequence for entity in self.entities)

    @property
    def reference_sequence(self) -> str:
        """Return the sequence for every Entity instance, constituting all Residues, not just structurally modeled ones

        Returns:
            The concatenated reference sequences for all Entity instances combined
        """
        return ''.join(entity.reference_sequence for entity in self.entities)

    @property
    def atom_indices_per_entity(self) -> list[list[int]]:
        """Return the atom indices for each Entity"""
        return [entity.atom_indices for entity in self.entities]

    # @property
    # def residue_indices_per_entity(self) -> list[list[int]]:
    #     """Return the residue indices for each Entity"""
    #     return [entity.residue_indices for entity in self.entities]


class SymmetryOpsMixin(abc.ABC):
    """Implements methods to interact with symmetric Structure instances"""
    _asu_indices: slice  # list[int]
    _asu_model_idx: int
    _center_of_mass_symmetric_models: np.ndarray
    _cryst_record: str
    _no_reset: bool
    _number_of_symmetry_mates: int
    _sym_entry: utils.SymEntry.SymEntry | None
    _symmetric_coords: Coordinates  # np.ndarray
    _symmetric_coords_split: list[np.ndarray]
    _expand_matrices: np.ndarray | list[list[float]]
    _expand_translations: np.ndarray | list[float]
    symmetry_state_attributes: set[str] = SymmetryBase.symmetry_state_attributes | {
        '_asu_indices', '_asu_model_idx', '_center_of_mass_symmetric_models',
        '_number_of_symmetry_mates', '_symmetric_coords', '_symmetric_coords_split'
    }

    def __init__(self, sym_entry: utils.SymEntry.SymEntry | int = None, symmetry: str = None,
                 transformations: list[types.TransformationMapping] = None, uc_dimensions: list[float] = None,
                 symmetry_operators: np.ndarray | list = None, rotation_matrices: np.ndarray | list = None,
                 translation_matrices: np.ndarray | list = None, surrounding_uc: bool = True, **kwargs):
        """Construct the instance

        Args:
            sym_entry: The SymEntry which specifies all symmetry parameters
            symmetry: The name of a symmetry to be searched against compatible symmetries
            transformations: Transformation operations that reproduce the oligomeric/assembly for each Entity
            rotation_matrices: Rotation operations that create the symmetric state
            translation_matrices: Translation operations that create the symmetric state
            uc_dimensions: The unit cell dimensions for the crystalline symmetry
            symmetry_operators: A set of custom expansion matrices
            surrounding_uc: Whether the 3x3 layer group, or 3x3x3 space group should be generated
        """
        super().__init__(**kwargs)  # SymmetryOpsMixin
        self._expand_matrices = self._expand_translations = None
        self.set_symmetry(sym_entry=sym_entry, symmetry=symmetry, uc_dimensions=uc_dimensions,
                          operators=symmetry_operators, rotations=rotation_matrices, translations=translation_matrices,
                          transformations=transformations, surrounding_uc=surrounding_uc)

    def format_biomt(self, **kwargs) -> str:
        """Return the SymmetricModel expand_matrices as a BIOMT record

        Returns:
            The BIOMT REMARK 350 with PDB file formatting
        """
        if self.is_symmetric():
            if self.dimension < 2:
                return '%s\n' % '\n'.join(
                    'REMARK 350   BIOMT{:1d}{:4d}{:10.6f}{:10.6f}{:10.6f}{:15.5f}            '.format(
                        v_idx, m_idx, *vec, point)
                    for m_idx, (mat, tx) in enumerate(
                        zip(self.expand_matrices.tolist(), self.expand_translations.tolist()), 1)
                    for v_idx, (vec, point) in enumerate(zip(mat, tx), 1)
                )
        return ''

    def format_header(self, **kwargs) -> str:
        """Returns any super().format_header() along with the BIOMT record

        Returns:
            The .pdb file header string
        """
        return super().format_header(**kwargs) + self.format_biomt()

    def set_symmetry(self, sym_entry: utils.SymEntry.SymEntry | int = None, symmetry: str = None,
                     crystal: bool = False, cryst_record: str = None, uc_dimensions: list[float] = None,
                     **kwargs):
        """Set the model symmetry using the CRYST1 record, or the unit cell dimensions and the Hermann-Mauguin symmetry
        notation (in CRYST1 format, ex P432) for the Model assembly. If the assembly is a point group, only the symmetry
        notation is required

        Args:
            sym_entry: The SymEntry which specifies all symmetry parameters
            symmetry: The name of a symmetry to be searched against compatible symmetries
            crystal: Whether crystalline symmetry should be used
            cryst_record: If a CRYST1 record is known and should be used
            uc_dimensions: The unit cell dimensions for the crystalline symmetry
        """
        # Try to solve for symmetry as uc_dimensions are needed for cryst ops, if available
        crystal_symmetry = None
        if symmetry is not None:
            # Ensure conversion to Hermann–Mauguin notation. ex: P23 not P 2 3
            symmetry = ''.join(symmetry.split())

        if cryst_record:
            self.cryst_record = cryst_record
            crystal = True
        else:
            cryst_record = self.cryst_record  # Populated above or from file parsing
            if uc_dimensions and symmetry:
                crystal_symmetry = symmetry
                crystal = True

        if cryst_record:  # Populated above or from file parsing
            self.log.debug(f'Parsed record: {cryst_record.strip()}')
            if uc_dimensions is None and symmetry is None:  # Only if didn't provide either
                uc_dimensions, crystal_symmetry = parse_cryst_record(cryst_record)
                crystal_symmetry = ''.join(crystal_symmetry)
            self.log.debug(f'Found uc_dimensions={uc_dimensions}, symmetry={crystal_symmetry}')

        if crystal:  # CRYST in symmetry.upper():
            if sym_entry is None:
                sym_entry = utils.SymEntry.CrystRecord

        number_of_entities = self.number_of_entities
        if sym_entry is not None:
            if isinstance(sym_entry, utils.SymEntry.SymEntry):
                if sym_entry.needs_cryst_record():  # Replace with relevant info from the CRYST1 record
                    if sym_entry.is_token():
                        # Create a new SymEntry
                        sym_entry = utils.SymEntry.CrystSymEntry(
                            space_group=crystal_symmetry,
                            sym_map=[crystal_symmetry] + ['C1' for _ in range(number_of_entities)])
                        # Set the uc_dimensions as they must be parsed or provided
                        self.log.critical(f'Setting {self}.sym_entry to new crystalline symmetry {sym_entry}')
                    elif sym_entry.resulting_symmetry == crystal_symmetry:
                        # This is already the specified SymEntry, use the CRYST record to set cryst_record
                        self.log.critical(f'Setting {self}.sym_entry to {sym_entry}')
                    else:
                        raise SymmetryError(
                            f"The parsed CRYST record with symmetry '{crystal_symmetry}' doesn't match the symmetry "
                            f"'{sym_entry.resulting_symmetry}' specified by the provided {type(sym_entry).__name__}"
                        )
                    sym_entry.uc_dimensions = uc_dimensions
                    sym_entry.cryst_record = self.cryst_record
                # else:  # SymEntry is set up properly
                #     self.sym_entry = sym_entry
            else:  # Try to solve using sym_entry as integer and any info in symmetry.
                sym_entry = utils.SymEntry.parse_symmetry_to_sym_entry(
                    sym_entry_number=sym_entry, symmetry=symmetry)
        elif symmetry:  # Provided without uc_dimensions, crystal=True, or cryst_record. Assuming point group
            sym_entry = utils.SymEntry.parse_symmetry_to_sym_entry(
                symmetry=symmetry,
                # The below fails as most of the time entity.symmetry isn't set up at this point
                # sym_map=[symmetry] + [entity.symmetry for entity in self.entities]
            )
        else:  # No symmetry was provided
            # self.sym_entry/self.symmetry can be None
            return

        # Ensure the number of Entity instances matches the SymEntry groups
        n_groups = sym_entry.number_of_groups
        if number_of_entities != n_groups:
            if n_groups == 1:
                verb = 'was'
            else:
                verb = 'were'

            raise SymmetryError(
                f'The {self.__class__.__name__} has {number_of_entities} entities. '
                f'{n_groups} {verb} expected based on the {repr(sym_entry)} specified'
            )
        self.sym_entry = sym_entry

    # # Todo this seems to introduce errors during Pose.get_interface_residues()
    # def reset_state(self):
    def reset_symmetry_state(self):
        super().reset_symmetry_state()
        self.reset_mates()

    def reset_mates(self):
        """Remove oligomeric chains. They should be generated fresh"""
        self._chains.clear()

    @property
    def sym_entry(self) -> utils.SymEntry.SymEntry | None:
        """The SymEntry specifies the symmetric parameters for the utilized symmetry"""
        try:
            return self._sym_entry
        except AttributeError:
            self._sym_entry = None
            return self._sym_entry

    @sym_entry.setter
    def sym_entry(self, sym_entry: utils.SymEntry.SymEntry | int):
        if isinstance(sym_entry, utils.SymEntry.SymEntry):
            self._sym_entry = sym_entry
        else:  # Try to convert
            self._sym_entry = utils.SymEntry.symmetry_factory.get(sym_entry)

        self.symmetry = getattr(self.sym_entry, 'resulting_symmetry')

    @property
    def point_group_symmetry(self) -> str | None:
        """The point group underlying the resulting SymEntry"""
        return getattr(self.sym_entry, 'point_group_symmetry', None)

    @property
    def dimension(self) -> int | None:
        """The dimension of the symmetry from None, 0, 2, or 3"""
        return getattr(self.sym_entry, 'dimension', None)

    @property
    def uc_dimensions(self) -> tuple[float, float, float, float, float, float] | None:
        """The unit cell dimensions for the lattice specified by lengths a, b, c and angles alpha, beta, gamma

        Returns:
            length a, length b, length c, angle alpha, angle beta, angle gamma
        """
        return getattr(self.sym_entry, 'uc_dimensions', None)

    @property
    def cryst_record(self) -> str | None:
        """Return the symmetry parameters as a CRYST1 entry"""
        try:
            return self._cryst_record
        except AttributeError:  # As of now, don't use if the structure wasn't symmetric and no attribute was parsed
            self._cryst_record = None if not self.is_symmetric() or self.dimension == 0 \
                else utils.symmetry.generate_cryst1_record(self.uc_dimensions, self.symmetry)
            return self._cryst_record

    @cryst_record.setter
    def cryst_record(self, cryst_record: str | None):
        # if cryst_record[:6] != 'CRYST1':
        self._cryst_record = cryst_record

    @property
    def expand_matrices(self) -> np.ndarray:
        """The symmetry rotations to generate each of the symmetry mates"""
        try:
            return self._expand_matrices.swapaxes(-2, -1)
        except AttributeError:  # list has no attribute 'swapaxes'
            return np.array([])

    @property
    def expand_translations(self) -> np.ndarray:
        """The symmetry translations to generate each of the symmetry mates"""
        try:
            return self._expand_translations.squeeze()
        except AttributeError:  # list has no attribute 'squeeze'
            return np.array([])

    @property
    def chain_transforms(self) -> list[types.TransformationMapping]:
        """Returns the transformation operations for each of the symmetry mates (excluding the ASU)"""
        chain_transforms = []
        # Skip the first given the mechanism for symmetry mate creation
        for idx, (rot, tx) in enumerate(zip(list(self.expand_matrices[1:]), list(self.expand_translations[1:]))):
            chain_transforms.append(types.TransformationMapping(rotation=rot, translation=tx))

        return chain_transforms

    @property
    def number_of_symmetric_residues(self) -> int:
        """Describes the number of Residues when accounting for symmetry mates"""
        return self.number_of_symmetry_mates * len(self._residue_indices)  # <- Same as self.number_of_residues

    @property
    def number_of_symmetry_mates(self) -> int:
        """Describes the number of symmetric copies present in the coordinates"""
        try:
            return self._number_of_symmetry_mates
        except AttributeError:  # Set based on the symmetry, unless that fails then default to 1
            self._number_of_symmetry_mates = getattr(self.sym_entry, 'number_of_operations', 1)
            return self._number_of_symmetry_mates

    @property
    def number_of_uc_symmetry_mates(self) -> int:
        """Describes the number of symmetry mates present in the unit cell"""
        try:
            return utils.symmetry.space_group_number_operations[self.symmetry]
        except KeyError:
            raise SymmetryError(
                f"The symmetry '{self.symmetry}' isn't an available unit cell at this time. If this is a point group, "
                f"adjust your code, otherwise, help expand the code to include the symmetry operators for this symmetry"
                f' group')

    def is_surrounding_uc(self) -> bool:
        """Returns True if the current coordinates contains symmetry mates from the surrounding unit cells"""
        if self.dimension > 0:
            # This is True if self.number_of_symmetry_mates was set to a larger value
            return self.number_of_symmetry_mates > self.number_of_uc_symmetry_mates
        else:
            return False

    def make_indices_symmetric(self, indices: Iterable[int], dtype: atom_or_residue_literal = 'atom') -> list[int]:
        """Extend asymmetric indices using the symmetry state across atom or residue indices

        Args:
            indices: The asymmetric indices to symmetrize
            dtype: The type of indices to perform symmetrization with

        Returns:
            The symmetric indices of the asymmetric input
        """
        try:
            jump_size = getattr(self, f'number_of_{dtype}s')
        except AttributeError:
            raise AttributeError(
                f"The dtype 'number_of_{dtype}' wasn't found in the {self.__class__.__name__} object. "
                "'Possible values of dtype are 'atom' or 'residue'")

        model_jumps = [jump_size * model_num for model_num in range(self.number_of_symmetry_mates)]
        return [idx + model_jump for model_jump in model_jumps for idx in indices]

    def return_symmetric_copies(self, structure: StructureBase, **kwargs) -> list[StructureBase]:
        """Expand the provided Structure using self.symmetry for the symmetry specification

        Args:
            structure: A StructureBase instance containing .coords method/attribute

        Returns:
            The symmetric copies of the input structure
        """
        number_of_symmetry_mates = self.number_of_symmetry_mates
        sym_coords = self.return_symmetric_coords(structure.coords)

        sym_mates = []
        for coord_set in np.split(sym_coords, number_of_symmetry_mates):
            symmetry_mate = structure.copy()
            symmetry_mate.coords = coord_set
            sym_mates.append(symmetry_mate)

        return sym_mates

    def generate_symmetric_coords(self, surrounding_uc: bool = True):
        """Expand the asu using self.symmetry for the symmetry specification, and optional unit cell dimensions if
        self.dimension > 0. Expands assembly to complete point group, unit cell, or surrounding unit cells

        Args:
            surrounding_uc: Whether the 3x3 layer group, or 3x3x3 space group should be generated
        """
        self.log.debug('Generating symmetric coords')
        if surrounding_uc:
            if self.dimension > 0:
                if self.dimension == 3:
                    uc_number = 27
                elif self.dimension == 2:
                    uc_number = 9
                else:
                    assert_never(self.dimension)
                # Set the number_of_symmetry_mates to account for the unit cell number
                # This results in is_surrounding_uc() being True during return_symmetric_coords()
                self._number_of_symmetry_mates = self.number_of_uc_symmetry_mates * uc_number

        # Set the self.symmetric_coords property
        self._symmetric_coords = Coordinates(self.return_symmetric_coords(self.coords))

    def cart_to_frac(self, cart_coords: np.ndarray | Iterable | int | float) -> np.ndarray:
        """Return fractional coordinates from cartesian coordinates
        From http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm

        Args:
            cart_coords: The cartesian coordinates of a unit cell

        Returns:
            The fractional coordinates of a unit cell
        """
        if self.uc_dimensions is None:
            raise ValueError(
                "Can't manipulate the unit cell. No unit cell dimensions were passed")

        return np.matmul(cart_coords, np.transpose(self.sym_entry.deorthogonalization_matrix))

    def frac_to_cart(self, frac_coords: np.ndarray | Iterable | int | float) -> np.ndarray:
        """Return cartesian coordinates from fractional coordinates
        From http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm

        Args:
            frac_coords: The fractional coordinates of a unit cell

        Returns:
            The cartesian coordinates of a unit cell
        """
        if self.uc_dimensions is None:
            raise ValueError(
                "Can't manipulate the unit cell. No unit cell dimensions were passed")

        return np.matmul(frac_coords, np.transpose(self.sym_entry.orthogonalization_matrix))

    def return_symmetric_coords(self, coords: np.ndarray) -> np.ndarray:
        """Provided an input set of coordinates, return the symmetrized coordinates corresponding to the SymmetricModel

        Args:
            coords: The coordinates to symmetrize

        Returns:
            The symmetrized coordinates
        """
        # surrounding_uc: bool = True
        #   surrounding_uc: Whether the 3x3 layer group, or 3x3x3 space group should be generated
        if self.dimension > 0:
            if self.is_surrounding_uc():
                shift_3d = [0., 1., -1.]
                if self.dimension == 3:
                    z_shifts = shift_3d
                elif self.dimension == 2:
                    z_shifts = [0.]
                else:
                    assert_never(self.dimension)

                uc_frac_coords = self.return_unit_cell_coords(coords, fractional=True)
                surrounding_frac_coords = \
                    np.concatenate([uc_frac_coords + [x, y, z] for x in shift_3d for y in shift_3d for z in z_shifts])
                return self.frac_to_cart(surrounding_frac_coords)
            else:
                return self.return_unit_cell_coords(coords)
        else:  # self.dimension = 0 or None
            return (np.matmul(np.tile(coords, (self.number_of_symmetry_mates, 1, 1)),
                              self._expand_matrices) + self._expand_translations).reshape(-1, 3)

    def return_unit_cell_coords(self, coords: np.ndarray, fractional: bool = False) -> np.ndarray:
        """Return the unit cell coordinates from a set of coordinates for the specified SymmetricModel

        Args:
            coords: The cartesian coordinates to expand to the unit cell
            fractional: Whether to return coordinates in fractional or cartesian (False) unit cell frame

        Returns:
            All unit cell coordinates
        """
        model_coords = (np.matmul(np.tile(self.cart_to_frac(coords), (self.number_of_uc_symmetry_mates, 1, 1)),
                                  self._expand_matrices) + self._expand_translations).reshape(-1, 3)
        if fractional:
            return model_coords
        else:
            return self.frac_to_cart(model_coords)

    @property
    def asu_model_index(self) -> int:
        """The asu equivalent model in the SymmetricModel. Zero-indexed"""
        try:
            return self._asu_model_idx
        except AttributeError:
            template_residue = self.n_terminal_residue
            atom_ca_coord = template_residue.ca_coords
            atom_idx = template_residue.ca_atom_index
            number_of_atoms = self.number_of_atoms
            symmetric_coords = self.symmetric_coords
            for model_idx in range(self.number_of_symmetry_mates):
                if np.allclose(atom_ca_coord, symmetric_coords[model_idx*number_of_atoms + atom_idx]):
                    self._asu_model_idx = model_idx
                    return self._asu_model_idx

            self.log.error(f'FAILED to find {self.asu_model_index.__name__}')

    @property
    def asu_indices(self) -> slice:
        """Return the ASU indices"""
        try:
            return self._asu_indices
        except AttributeError:
            self._asu_indices = self.get_asu_atom_indices(as_slice=True)

            return self._asu_indices

    @property
    def symmetric_coords(self) -> np.ndarray:
        """Return a view of the symmetric Coordinates"""
        try:
            return self._symmetric_coords.coords
        except AttributeError:
            self.generate_symmetric_coords()
            return self._symmetric_coords.coords

    @property
    def symmetric_coords_split(self) -> list[np.ndarray]:
        """A view of the symmetric coordinates split by each symmetric model"""
        try:
            return self._symmetric_coords_split
        except AttributeError:
            self._symmetric_coords_split = np.split(self.symmetric_coords, self.number_of_symmetry_mates)
            return self._symmetric_coords_split

    @property
    def center_of_mass_symmetric(self) -> np.ndarray:
        """The center of mass for the symmetric system with shape (3,)"""
        # number_of_symmetry_atoms = len(self.symmetric_coords)
        # return np.matmul(np.full(number_of_symmetry_atoms, 1 / number_of_symmetry_atoms), self.symmetric_coords)
        # v since all symmetry by expand_matrix anyway
        return self.center_of_mass_symmetric_models.mean(axis=-2)

    @property
    def center_of_mass_symmetric_models(self) -> np.ndarray:
        """The center of mass points for each symmetry mate in the symmetric system with shape
        (number_of_symmetry_mates, 3)
        """
        # number_of_atoms = self.number_of_atoms
        # return np.matmul(np.full(number_of_atoms, 1 / number_of_atoms), self.symmetric_coords_split)
        try:
            return self._center_of_mass_symmetric_models
        except AttributeError:
            self._center_of_mass_symmetric_models = self.return_symmetric_coords(self.center_of_mass)
            return self._center_of_mass_symmetric_models

    def find_contacting_asu(self, distance: float = 8., **kwargs) -> list[Entity]:
        """Find the maximally contacting symmetry mate for each Entity and return the corresponding Entity instances

        Args:
            distance: The distance to check for contacts

        Returns:
            The minimal set of Entities containing the maximally touching configuration
        """
        entities = self.entities
        if not entities:
            # The SymmetricModel was probably set without them. Create them, then try to find the asu
            self._create_entities()
            entities = self.entities

        number_of_entities = len(entities)
        if number_of_entities != 1:
            idx = count()
            chain_combinations: list[tuple[Entity, Entity]] = []
            entity_combinations: list[tuple[Entity, Entity]] = []
            contact_count = \
                np.zeros(sum(map(math.prod, combinations((entity.number_of_symmetry_mates for entity in entities), 2))))
            for entity1, entity2 in combinations(entities, 2):
                for chain1 in entity1.chains:
                    chain_cb_coord_tree = BallTree(chain1.cb_coords)
                    for chain2 in entity2.chains:
                        entity_combinations.append((entity1, entity2))
                        chain_combinations.append((chain1, chain2))
                        contact_count[next(idx)] = \
                            chain_cb_coord_tree.two_point_correlation(chain2.cb_coords, [distance])[0]

            max_contact_idx = contact_count.argmax()
            additional_chains = []
            max_chains = list(chain_combinations[max_contact_idx])
            if len(max_chains) != number_of_entities:  # We found 2 entities at this point
                # find the indices where either of the maximally contacting chains are utilized
                selected_chain_indices = {idx for idx, chain_pair in enumerate(chain_combinations)
                                          if max_chains[0] in chain_pair or max_chains[1] in chain_pair}
                remaining_entities = set(entities).difference(entity_combinations[max_contact_idx])
                for entity in remaining_entities:  # get the max contacts and the associated entity and chain indices
                    # find the indices where the missing entity is utilized
                    remaining_indices = \
                        {idx for idx, entity_pair in enumerate(entity_combinations) if entity in entity_pair}
                    # pair_position = [0 if entity_pair[0] == entity else 1
                    #                  for idx, entity_pair in enumerate(entity_combinations) if entity in entity_pair]
                    # only use those where found asu chains already occur
                    viable_remaining_indices = list(remaining_indices.intersection(selected_chain_indices))
                    # out of the viable indices where the selected chains are matched with the missing entity,
                    # find the highest contact
                    max_idx = contact_count[viable_remaining_indices].argmax()
                    for entity_idx, entity_in_combo in enumerate(
                            entity_combinations[viable_remaining_indices[max_idx]]):
                        if entity == entity_in_combo:
                            additional_chains.append(chain_combinations[viable_remaining_indices[max_idx]][entity_idx])

            new_entities = max_chains + additional_chains
            # Rearrange the entities to have the same order as provided
            entities = [new_entity for entity in entities for new_entity in new_entities if entity == new_entity]

        return entities

    def get_contacting_asu(self, distance: float = 8., **kwargs) -> SymmetricModel:  # Todo -> Self python 3.11
        """Find the maximally contacting symmetry mate for each Entity and return the corresponding Entity instances as
         a new Pose

        If the chain IDs of the asu are the same, then chain IDs will automatically be renamed

        Args:
            distance: The distance to check for contacts

        Returns:
            A new Model with the minimal set of Entity instances. Will also be symmetric
        """
        if self.number_of_entities == 1:
            return self.copy()

        entities = self.find_contacting_asu(distance=distance, **kwargs)

        if len({entity.chain_id for entity in entities}) != len(entities):
            rename = True
        else:
            rename = False

        cls = type(self)
        # assert cls is Pose, f"Can't {self.get_contacting_asu.__name__} for the class={cls}. Only for Pose"
        return cls.from_entities(
            entities, name=f'{self.name}-asu', log=self.log, sym_entry=self.sym_entry, rename_chains=rename,
            cryst_record=self.cryst_record, **kwargs)  # , biomt_header=self.format_biomt(),

    def set_contacting_asu(self, from_assembly: bool = False, **kwargs):
        """Find the maximally contacting symmetry mate for each Entity, then set the Pose with this info

        Args:
            from_assembly: Whether the ASU should be set fresh from the entire assembly instances

        Keyword Args:
            distance: float = 8.0 - The distance to check for contacts

        Sets:
            self: To a SymmetricModel with the minimal set of Entities containing the maximally touching configuration
        """
        number_of_entities = self.number_of_entities
        if not self.is_symmetric():
            raise SymmetryError(
                f"Couldn't {self.set_contacting_asu.__name__}() with the asymmetric {repr(self)}"
            )
        elif number_of_entities == 1:
            return  # This can't be set any better

        # Check to see if the parsed Model is already represented symmetrically
        if from_assembly or self.has_dependent_chains():  # and self.is_asymmetric_mates() and not preserve_asymmetry:
            # If .from_chains() or .from_file(), ensure the SymmetricModel is an asu
            self.log.debug(f'Setting the {repr(self)} to an ASU from a symmetric representation. '
                           "This method hasn't been thoroughly debugged")
            # Essentially performs,
            # self.assign_residues_from_structures(self.entities)
            # however, without _assign_residues(), the containers are not updated.
            # Set base Structure attributes
            new_coords = []
            new_atoms = []
            new_residues = []
            for entity in enumerate(self.entities):
                new_coords.append(entity.coords)
                new_atoms.extend(entity.atoms)
                new_residues.extend(entity.residues)

            self._coords = Coordinates(np.concatenate(new_coords))
            self._atom_indices = list(range(len(new_atoms)))
            self._atoms.set(new_atoms)
            self._residue_indices = list(range(len(new_residues)))
            self._residues.set(new_residues)

            # Remove extra chains by creating fresh
            self._create_chains()
            # Update entities to reflect new indices
            self.reset_structures_states(self.entities)
            # Recurse this call to ensure that the entities are contacting
            self.set_contacting_asu(**kwargs)
            # else:
            #     raise SymmetryError(
            #         f"Couldn't {self.set_contacting_asu.__name__}() with the number of parsed chains, "
            #         f"{self.number_of_chains}. When symmetry={self.symmetry} and the number of entities is "
            #         f"{self.number_of_entities}, the number of symmetric chains should be "
            #         f'{number_of_entities * self.number_of_symmetry_mates}'
            #     )
        else:  # number_of_entities == number_of_chains:
            self.log.debug(f'Finding the ASU with the most contacting interface')
            entities = self.find_contacting_asu(**kwargs)

            # With perfect symmetry, v this is sufficient
            self._no_reset = True
            self.coords = np.concatenate([entity.coords for entity in entities])
            del self._no_reset
            # If imperfect symmetry, adapting below may provide some benefit
            # self.assign_residues_from_structures(entities)

    # def report_symmetric_coords(self, func_name: str):
    #     """Debug the symmetric coords for equality across instances"""
    #     self.log.debug(f'In {func_name}, coords and symmetric_coords (ASU) equal: '
    #                    f'{self.coords == self.symmetric_coords[:self.number_of_atoms]}\n'
    #                    f'coords {np.array_str(self.coords[:2], precision=2)}\n'
    #                    f'symmetric_coords {np.array_str(self.symmetric_coords[:2], precision=2)}')


class Chain(Structure, MetricsMixin):
    """A grouping of Atom, Coordinates, and Residue instances, typically from a connected polymer"""
    _chain_id: str
    _entity: Entity
    _metrics_table = None

    def __init__(self, chain_id: str = None, name: str = None, **kwargs):
        """Construct the instance

        Args:
            chain_id: The name of the Chain identifier to use for this instance
            name: The name of the Chain identifier to use for this instance. Typically used by Entity subclasses.
        """
        kwargs['name'] = name = name if name else chain_id
        super().__init__(**kwargs)  # Chain
        if type(self) is Chain and name is not None:
            # Only if this instance is a Chain, not Entity, set the chain_id
            self.chain_id = name

    @property
    def chain_id(self) -> str:
        """The Chain ID for the instance"""
        try:
            return self._chain_id
        except AttributeError:
            self._chain_id = self.residues[0].chain_id
            return self._chain_id

    @chain_id.setter
    def chain_id(self, chain_id: str):
        self.set_residues_attributes(chain_id=chain_id)
        self._chain_id = chain_id

    @property
    def entity(self) -> Entity | None:
        """The Entity associated with the instance"""
        try:
            return self._entity
        except AttributeError:
            return None

    @property
    def entity_id(self) -> str:
        """The Entity ID associated with the instance"""
        return getattr(self._entity, 'name', None)

    @property
    def reference_sequence(self) -> str:
        """Return the entire sequence, constituting all described residues, not just structurally modeled ones

        Returns:
            The sequence according to the Entity reference, or the Structure sequence if no reference available
        """
        try:
            return self.entity.reference_sequence
        except AttributeError:  # .entity isn't set
            self.log.debug(f"The reference sequence couldn't be found. Using the {repr(self)} sequence instead")
            return self.sequence

    def calculate_metrics(self, **kwargs) -> dict[str, Any]:
        """"""
        self.log.warning(f"{self.calculate_metrics.__name__} doesn't calculate anything yet...")
        return {
            # 'name': self.name,
            # 'n_terminal_helix': self.is_termini_helical(),
            # 'c_terminal_helix': self.is_termini_helical(termini='c'),
            # 'thermophile': thermophile
            # 'number_of_residues': self.number_of_residues,
        }


class Entity(SymmetryOpsMixin, ContainsChains, Chain):
    """Maps a biological instance of a Structure which ContainsChains(1-N) and is a Complex, to a single GeneProduct"""
    _captain: Entity | None
    """The specific transformation operators to generate all mate chains of the Oligomer"""
    _chains: list | list[Entity]
    _oligomer: Model
    _is_captain: bool
    # Metrics class attributes
    # _df: pd.Series  # Metrics
    # _metrics: sql.EntityMetrics  # Metrics
    _metrics_table = sql.EntityMetrics
    # _uniprot_id: str | None
    # _search_uniprot_id = False
    # """Whether the uniprot_id has been queried"""
    _api_data: dict[str, dict[str, str]] | None
    dihedral_chain: str | None
    _max_symmetry: int | None
    _max_symmetry_chain_idx: int
    mate_rotation_axes: list[dict[str, int | np.ndarray]] | list
    """Maps mate entities to their rotation matrix"""
    _uniprot_ids: tuple[str | None, ...]
    state_attributes = Chain.state_attributes | {'_oligomer'}  # '_chains' handled specifically

    @classmethod
    def from_chains(cls, chains: list[Chain] | Structures, residue_indices: list[int] = None, **kwargs):
        """Initialize an Entity from Chain instances

        Args:
            chains: A list of Chain instances that match the Entity
            residue_indices: The indices which the new Entity should contain
        """
        operators = kwargs.get('operators')
        if residue_indices is None:
            asymmetry = False
            if asymmetry:
                pass
            # No check
            else:
                representative, *additional_chains = chains

            residue_indices = representative.residue_indices

        return cls(chains=chains, residue_indices=residue_indices, operators=operators, **kwargs)

    # @classmethod
    # def from_metadata(cls, chains: list[Chain] | Structures, metadata: EntityData, **kwargs):
    #     """Initialize an Entity from a set of Chain objects and EntityData"""
    #     return cls(chains=chains, metadata=metadata, **kwargs)

    def __init__(self,
                 operators: tuple[np.ndarray | list[list[float]], np.ndarray | list[float]] | np.ndarray = None,
                 rotations: np.ndarray | list[list[float]] = None, translations: np.ndarray | list[float] = None,
                 # transformations: list[types.TransformationMapping] = None, surrounding_uc: bool = True,
                 **kwargs):
        """When init occurs chain_ids are set if chains were passed. If not, then they are auto generated

        Args:
            operators: A set of symmetry operations to designate how to apply symmetry
            rotations: A set of rotation matrices used to recapitulate the SymmetricModel from the asymmetric unit
            translations: A set of translation vectors used to recapitulate the SymmetricModel from the asymmetric unit
        """
        self._is_captain = True
        super().__init__(**kwargs,  # Entity
                         as_mates=True)  # <- needed when .from_file() w/ self.is_parent()
        self._captain = None
        self.dihedral_chain = None
        self.mate_rotation_axes = []

        chains = self.chains
        if not chains:
            raise DesignError(
                f"Can't construct {self.__class__.__name__} instance without 'chains'. "
                "Ensure that you didn't construct with chains=False | None"
            )

        representative, *additional_chains = chains
        if self.is_parent():
            # When this instance is the parent (.from_file(), .from_chains(parent=None))
            # Set attributes from representative now that _chains is parsed
            self._coords.set(representative.coords)
            self._assign_residues(representative.residues, atoms=representative.atoms)

        # Set up the chain copies
        number_of_symmetry_mates = len(chains)
        symmetry = None
        if number_of_symmetry_mates > 1:
            if self.is_dihedral():
                symmetry = f'D{int(number_of_symmetry_mates / 2)}'
            elif self.is_cyclic():
                symmetry = f'C{number_of_symmetry_mates}'
            else:  # Higher than D, probably T, O, I, or asymmetric
                try:
                    symmetry = utils.symmetry.subunit_number_to_symmetry[number_of_symmetry_mates]
                except KeyError:
                    self.log.warning(f"Couldn't find a compatible symmetry for the Entity with "
                                     f"{number_of_symmetry_mates} chain copies")
                    # symmetry = None
                    # self.symmetry = "Unknown-symmetry"
        self.set_symmetry(symmetry=symmetry)

        if not self.is_symmetric():
            # No symmetry keyword args were passed
            if operators is not None or rotations is not None or translations is not None:
                passed_args = []
                if operators:
                    passed_args.append('operators')
                if rotations:
                    passed_args.append('rotations')
                if translations:
                    passed_args.append('translations')
                raise ConstructionError(
                    f"Couldn't set_symmetry() using {', '.join(passed_args)} without explicitly passing "
                    "'symmetry' or 'sym_entry'"
                )
            return

        # Set rotations and translations to the correct symmetry operations
        # where operators, rotations, and translations are user provided from some sort of BIOMT (fiber, other)
        if operators is not None:
            symmetry_source_arg = "'operators' "

            num_operators = len(operators)
            if isinstance(operators, tuple) and num_operators == 2:
                self.log.warning("Providing custom symmetry 'operators' may result in improper symmetric "
                                 'configuration. Proceed with caution')
                rotations, translations = operators
            elif isinstance(operators, Sequence) and num_operators == number_of_symmetry_mates:
                rotations = []
                translations = []
                try:
                    for rot, tx in operators:
                        rotations.append(rot)
                        translations.append(tx)
                except TypeError:  # Unpack failed
                    raise ValueError(
                        f"Couldn't parse the 'operators'={repr(operators)}.\n\n"
                        "Expected a Sequence[rotation shape=(3,3). translation shape=(3,)] pairs."
                    )
            elif isinstance(operators, np.ndarray):
                if operators.shape[1:] == (3, 4):
                    # Parse from a single input of 3 row by 4 column style, like BIOMT
                    rotations = operators[:, :, :3]
                    translations = operators[:, :, 3:].squeeze()
                elif operators.shape[1:] == 3:  # Assume just rotations
                    rotations = operators
                    translations = np.tile(utils.symmetry.origin, len(rotations))
                else:
                    raise ConstructionError(
                        f"The 'operators' form {repr(operators)} isn't supported.")
            else:
                raise ConstructionError(
                    f"The 'operators' form {repr(operators)} isn't supported. Must provide a tuple of "
                    'array-like objects with the order (rotation matrices, translation vectors) or use the '
                    "'rotations' and 'translations' keyword args")
        else:
            symmetry_source_arg = ''

        # Now that symmetry is set, check if the Structure parsed all symmetric chains
        if len(chains) == self.number_of_entities * number_of_symmetry_mates:
            parsed_assembly = True
        else:
            parsed_assembly = False

        # Set the symmetry operations
        if rotations is not None and translations is not None:
            if not isinstance(rotations, np.ndarray):
                rotations = np.ndarray(rotations)
            if rotations.ndim == 3:
                # Assume operators were provided in a standard orientation and transpose for subsequent efficiency
                # Using .swapaxes(-2, -1) call here instead of .transpose() for safety
                self._expand_matrices = rotations.swapaxes(-2, -1)
            else:
                raise SymmetryError(
                    f"Expected {symmetry_source_arg}rotation matrices with 3 dimensions, not {rotations.ndim} "
                    "dimensions. Ensure the passed rotation matrices have a shape of (N symmetry operations, 3, 3)"
                )

            if not isinstance(translations, np.ndarray):
                translations = np.ndarray(translations)
            if translations.ndim == 2:
                # Assume operators were provided in a standard orientation each vector needs to be in own array on dim=2
                self._expand_translations = translations[:, None, :]
            else:
                raise SymmetryError(
                    f"Expected {symmetry_source_arg}translation vectors with 2 dimensions, not {translations.ndim} "
                    "dimensions. Ensure the passed translations have a shape of (N symmetry operations, 3)"
                )
        else:
            symmetry_source_arg = "'chains' "
            if self.dimension == 0:
                # The _expand_matrices rotation matrices are pre-transposed to avoid repetitive operations
                _expand_matrices = utils.symmetry.point_group_symmetry_operatorsT[self.symmetry]
                # The _expand_translations vectors are pre-sliced to enable numpy operations
                _expand_translations = \
                    np.tile(utils.symmetry.origin, (number_of_symmetry_mates, 1))[:, None, :]

                if parsed_assembly:
                    # The Structure should have symmetric chains
                    # This routine is essentially orient(). However, with one Entity, no extra need for orient
                    _expand_matrices = [utils.symmetry.identity_matrix]
                    _expand_translations = [utils.symmetry.origin]
                    self_seq = self.sequence
                    ca_coords = self.ca_coords
                    # Todo match this mechanism with the symmetric chain index
                    for chain in additional_chains:
                        chain_seq = chain.sequence
                        additional_chain_coords = chain.ca_coords
                        first_chain_coords = ca_coords
                        if chain_seq != self_seq:
                            # Get aligned indices, then follow with superposition
                            self.log.debug(f'{repr(chain)} and {repr(self)} require alignment to symmetrize')
                            fixed_indices, moving_indices = get_equivalent_indices(chain_seq, self_seq)
                            additional_chain_coords = additional_chain_coords[fixed_indices]
                            first_chain_coords = first_chain_coords[moving_indices]

                        _, rot, tx = superposition3d(additional_chain_coords, first_chain_coords)
                        _expand_matrices.append(rot)
                        _expand_translations.append(tx)

                    self._expand_matrices = np.array(_expand_matrices).swapaxes(-2, -1)
                    self._expand_translations = np.array(_expand_translations)[:, None, :]
                else:
                    self._expand_matrices = _expand_matrices
                    self._expand_translations = _expand_translations
            else:
                self._expand_matrices, self._expand_translations = \
                    utils.symmetry.space_group_symmetry_operatorsT[self.symmetry]

        # Removed parsed chain information
        self.reset_mates()

    @StructureBase.coords.setter
    def coords(self, coords: np.ndarray | list[list[float]]):
        """Set the Coords object while propagating changes to symmetric "mate" chains"""
        if self.is_symmetric() and self._is_captain:
            # **This routine handles imperfect symmetry**
            self.log.debug('Entity captain is updating coords')
            # Must do these before super().coords.fset()
            # Populate .chains (if not already) with current coords and transformation
            self_, *mate_chains = self.chains
            # Set current .ca_coords as prior_ca_coords
            prior_ca_coords = self.ca_coords.copy()

            # Set coords with new coords
            super(ContainsAtoms, ContainsAtoms).coords.fset(self, coords)
            if self.is_dependent():
                _parent = self.parent
                if _parent.is_symmetric() and not self._parent_is_updating:
                    _parent._dependent_is_updating = True
                    # Update the parent which propagates symmetric updates
                    _parent.coords = _parent.coords
                    _parent._dependent_is_updating = False

            # Find the transformation from the old coordinates to the new
            new_ca_coords = self.ca_coords
            _, new_rot, new_tx = superposition3d(new_ca_coords, prior_ca_coords)

            new_rot_t = np.transpose(new_rot)
            # Remove prior transforms by setting a fresh container
            _expand_matrices = [utils.symmetry.identity_matrix]
            _expand_translations = [utils.symmetry.origin]
            # Find the transform between the new coords and the current mate chain coords
            # for chain, transform in zip(mate_chains, current_chain_transforms):
            for chain in mate_chains:
                # self.log.debug(f'Updated transform of mate {chain.chain_id}')
                # In liu of using chain.coords as lengths might be different
                # Transform prior_coords to chain.coords position, then transform using new_rot and new_tx
                # new_chain_ca_coords = \
                #     np.matmul(np.matmul(prior_ca_coords,
                #                         np.transpose(transform['rotation'])) + transform['translation'],
                #               np.transpose(new_rot)) + new_tx
                new_chain_ca_coords = np.matmul(chain.ca_coords, new_rot_t) + new_tx
                # Find the transform from current coords and the new mate chain coords
                _, rot, tx = superposition3d(new_chain_ca_coords, new_ca_coords)
                # Save transform
                # self._chain_transforms.append(dict(rotation=rot, translation=tx))
                rot_t = np.transpose(rot)
                _expand_matrices.append(rot_t)
                _expand_translations.append(tx)
                # Transform existing mate chain
                chain.coords = np.matmul(coords, rot_t) + tx
                # self.log.debug(f'Setting coords on mate chain {chain.chain_id}')

            self._expand_matrices = np.array(_expand_matrices)
            self._expand_translations = np.array(_expand_translations)[:, None, :]
        else:  # Accept the new coords
            super(ContainsAtoms, ContainsAtoms).coords.fset(self, coords)

    def _format_seqres(self, assembly: bool = False, **kwargs) -> str:
        """Format the reference sequence present in the SEQRES remark for writing to the output header

        Args:
            assembly: Whether to write header details for the assembly

        Returns:
            The .pdb formatted SEQRES record
        """
        seqres = super(Chain, Chain)._format_seqres(self, **kwargs)
        if assembly:
            structure_container = self.chains[1:]
            chain_ids = self.chain_ids[1:]
        else:
            structure_container = chain_ids = []
            # structure_container = self.entities

        return seqres + ''.join(
            struct._format_seqres(chain_id=chain_id) for struct, chain_id in zip(structure_container, chain_ids))

    # @property
    # def sequence(self) -> str:
    #     """Return the sequence of structurally modeled residues for every Entity instance
    #
    #     Returns:
    #         The concatenated sequence for all Entity instances combined
    #     """
    #     # Due to the ContainsEntities MRO, need to call the ContainsResidues.sequence
    #     return super(Structure, Structure).sequence.fget(self)

    # @property
    # def reference_sequence(self) -> str:
    #     """Return the entire sequence, constituting all described residues, not just structurally modeled ones
    #
    #     Returns:
    #         The sequence according to the Entity reference, or the Structure sequence if no reference available
    #     """
    #     # Due to the ContainsEntities MRO, need to call the ContainsResidues.reference_sequence
    #     return super(Structure, Structure).reference_sequence.fget(self)

    # @property
    # def chain_id(self) -> str:
    #     """The Chain name for the Entity instance"""
    #     return self.residues[0].chain_id

    @Chain.chain_id.setter
    def chain_id(self, chain_id: str):
        super(Entity, Entity).chain_id.fset(self, chain_id)
        # # Same as Chain class property
        # self.set_residues_attributes(chain_id=chain_id)
        # self._chain_id = chain_id
        # Different from Chain class
        if self._is_captain:
            self._set_chain_ids()

    @property
    def entity_id(self) -> str:
        """The Entity ID associated with the instance"""
        return self.name

    @entity_id.setter
    def entity_id(self, entity_id: str):
        # self.set_residues_attributes(entity_id=entity_id)
        self.name = entity_id

    def _set_chain_ids(self):
        """From the Entity.chain_id set all mate Chains with an incrementally higher id

        Sets:
            self.chain_ids: list(str)
        """
        first_chain_id = self.chain_id
        chain_gen = chain_id_generator()
        # Iterate over the generator until the current chain_id is found
        discard = next(chain_gen)
        try:
            while discard != first_chain_id:
                discard = next(chain_gen)
        except StopIteration:
            self.log.warning(f"Couldn't find the self.chain_id '{self.chain_id}' in chain_id_generator()")
            # The end of the generator was reached without a success. Try to just use the first chain_ids returned
            chain_gen = chain_id_generator()

        additional_chain_ids = [next(chain_gen) for _ in range(self.number_of_symmetry_mates - 1)]
        # Use the existing chain_id and iterate over the generator for the mate chains
        if self.has_mates():
            # Must set .chain_ids because mate Chain chain_ids are set upon captain Entity .chains property init
            for chain, new_id in zip(self.chains[1:], additional_chain_ids):
                chain.chain_id = new_id

        self.chain_ids = [first_chain_id] + additional_chain_ids
        # # Alternative to above "Must set chain_ids first, then chains"
        # for chain in self.chains[1:]:
        #     chain.chain_id = next(chain_gen)
        #
        # self.chain_ids = [chain.chain_id for chain in self.chains]  # Use the existing chain_id

    # @property
    # def chain_ids(self) -> list:  # Also used in Model
    #     """The names of each Chain found in the Entity"""
    #     try:
    #         return self._chain_ids
    #     except AttributeError:  # This shouldn't be possible with the constructor available
    #         available_chain_ids = chain_id_generator()
    #         self._chain_ids = [self.chain_id]
    #         for _ in range(self.number_of_symmetry_mates - 1):
    #             next_chain = next(available_chain_ids)
    #             while next_chain in self._chain_ids:
    #                 next_chain = next(available_chain_ids)
    #
    #             self._chain_ids.append(next_chain)
    #
    #         return self._chain_ids
    #
    # @chain_ids.setter
    # def chain_ids(self, chain_ids: list[str]):
    #     self._chain_ids = chain_ids

    @SymmetryBase.symmetry.setter
    def symmetry(self, symmetry: str | None):
        super(StructureBase, StructureBase).symmetry.fset(self, symmetry)
        if self.is_symmetric() and self.is_dependent():
            # Set the parent StructureBase.symmetric_dependents to the 'entities' container
            self.parent.symmetric_dependents = 'entities'

    # @property
    # def center_of_mass_symmetric_models(self) -> np.ndarray:
    #     """The individual centers of mass for each model in the symmetric system"""
    #     # try:
    #     #     return self._center_of_mass_symmetric_models
    #     # except AttributeError:
    #     com = self.center_of_mass
    #     mate_coms = [com]
    #     for transform in self.chain_transforms:
    #         mate_coms.append(np.matmul(com, transform['rotation'].T) + transform['translation'])
    #
    #     # np.array makes the right shape while concatenate doesn't
    #     return np.array(mate_coms)

    def is_captain(self) -> bool:
        """Is the Entity instance the captain?"""
        return self._is_captain

    def is_mate(self) -> bool:
        """Is the Entity instance a mate?"""
        return not self._is_captain

    @property
    def number_of_entities(self) -> int:
        """Return the number of distinct entities (Gene/Protein products) found in the PoseMetadata"""
        return 1

    @property
    def entities(self) -> list[Entity]:  # Structures
        """Returns the Entity instance as a list"""
        return [self]

    def has_mates(self) -> bool:
        """Returns True if this Entity is a captain and has mates"""
        return len(self._chains) > 1

    @property
    def chains(self) -> list[Entity]:  # Todo python3.11 -> list[Self] | Structures
        """The mate Chain instances of the instance. If not created, returns transformed copies of the instance"""
        # Set in __init__() -> self._chains = [self]
        chain_transforms = self.chain_transforms
        chains = self._chains
        if self._is_captain and len(chains) == 1 and chain_transforms:
            # populate ._chains with Entity mates
            # These mates will be their own "parent", and will be under the control of this instance, i.e. the captain
            self.log.debug(f"Generating {len(chain_transforms)} {repr(self)} mate instances in '.chains' attribute")
            mate_entities = [self.get_transformed_mate(**transform) for transform in chain_transforms]

            # Set entity.chain_id which sets all residues
            for mate, chain_id in zip(mate_entities, self.chain_ids[1:]):
                mate.chain_id = chain_id
            chains.extend(mate_entities)

        return chains

    @property
    def assembly(self) -> Model:
        """Access the oligomeric Structure which is a copy of the Entity plus any additional symmetric mate chains

        Returns:
            Structures object with the underlying chains in the oligomer
        """
        try:
            return self._oligomer
        except AttributeError:
            self.log.debug(f'Constructing {repr(self)}.assembly')
            self._oligomer = Model.from_chains(self.chains, name=f'{self.name}-oligomer', log=self.log)
            return self._oligomer

    def remove_mate_chains(self):
        """Clear the Entity of all Chain and Oligomer information"""
        self._expand_matrices = self._expand_translations = []
        self.reset_mates()
        self._is_captain = False
        self.chain_ids = [self.chain_id]

    def _make_mate(self, captain: Entity):
        """Turn the Entity into a "mate" Entity

        Args:
            captain: The Entity to designate as the instance's captain Entity
        """
        # self.log.debug(f'Designating Entity as mate')
        self._captain = captain
        self._expand_matrices = self._expand_translations = []
        self._is_captain = False
        self.chain_ids = [captain.chain_id]  # Set for a length of 1, using the captain.chain_id

    def _make_captain(self):
        """Turn the Entity into a "captain" Entity if it isn't already"""
        if self._is_captain:
            return

        # self.log.debug(f'Promoting mate {repr(self)} to a captain')
        # Find and save the transforms between the self.coords and the prior captains mate chains
        current_ca_coords = self.ca_coords
        _expand_matrices = [utils.symmetry.identity_matrix]
        _expand_translations = [utils.symmetry.origin]
        for chain in self._captain.chains:
            # Find the transform from current coords and the new mate chain coords
            _, rot, tx = superposition3d(chain.ca_coords, current_ca_coords)
            if np.allclose(utils.symmetry.identity_matrix, rot):
                # This "chain" is the self instance and the identity transform is skipped
                # self.log.debug(f'Skipping identity transform')
                continue
            _expand_matrices.append(rot)
            _expand_translations.append(tx)

        self._expand_matrices = np.array(_expand_matrices).swapaxes(-2, -1)
        self._expand_translations = np.array(_expand_translations)[:, None, :]
        # # Alternative:
        # # Transform the transforms by finding transform from the old captain to the current coords
        # # Not sure about the algebraic requirements of the old translation. It may require rotation with offset_ro...
        # _, offset_rot, offset_tx = superposition3d(self.ca_coords, self._captain.ca_coords)
        # self._chain_transforms = []  # self._captain._chain_transforms.copy()
        # for idx, transform in enumerate(self._captain._chain_transforms):  # self._chain_transforms):
        #     # Rotate the captain oriented rotation matrix to the current coordinates
        #     new_rotation = np.matmul(transform['rotation'], offset_rot)
        #     new_transform = transform['translation'] + offset_tx
        #     self._chain_transforms.append()

        self._is_captain = True
        self.chain_id = self._captain.chain_id
        self._captain = None

    def make_oligomer(self, symmetry: str = None, rotation: list[list[float]] | np.ndarray = None,
                      translation: list[float] | np.ndarray = None, rotation2: list[list[float]] | np.ndarray = None,
                      translation2: list[float] | np.ndarray = None, **kwargs):
        """Given a symmetry and transformational mapping, generate oligomeric copies of the Entity

        Assumes that the symmetric system treats the canonical symmetric axis as the Z-axis, and if the Entity is not at
        the origin, that a transformation describing its current position relative to the origin is passed so that it
        can be moved to the origin. At the origin, makes the required oligomeric rotations, to generate an oligomer
        where symmetric copies are stored in the .chains attribute then reverses the operations back to original
        reference frame if any was provided

        Args:
            symmetry: The symmetry to set the Entity to
            rotation: The first rotation to apply, expected array shape (3, 3)
            translation: The first translation to apply, expected array shape (3,)
            rotation2: The second rotation to apply, expected array shape (3, 3)
            translation2: The second translation to apply, expected array shape (3,)

        Sets:
            self.symmetry (str)
            self.sym_entry (SymEntry)
            self.number_of_symmetry_mates (int)
            self._expand_matrices
            self._expand_translations
        """
        self.set_symmetry(symmetry=symmetry)
        if not self.is_symmetric():
            return

        symmetry = self.symmetry
        degeneracy_matrices = None
        if symmetry in utils.symmetry.cubic_point_groups:
            rotation_matrices = utils.symmetry.point_group_symmetry_operators[symmetry]
        elif 'D' in symmetry:  # Provide a 180-degree rotation along x (all D orient symmetries have axis here)
            rotation_matrices = \
                utils.SymEntry.get_rot_matrices(
                    utils.symmetry.rotation_range[symmetry.replace('D', 'C')],
                    'z', 360
                )
            degeneracy_matrices = [utils.symmetry.identity_matrix, utils.symmetry.flip_x_matrix]
        else:  # Symmetry is cyclic
            rotation_matrices = utils.SymEntry.get_rot_matrices(utils.symmetry.rotation_range[symmetry], 'z')

        degeneracy_rotation_matrices = utils.SymEntry.make_rotations_degenerate(
            rotation_matrices, degeneracy_matrices)

        assert self.number_of_symmetry_mates == len(degeneracy_rotation_matrices), \
            (f"The number of symmetry mates, {self.number_of_symmetry_mates} != {len(degeneracy_rotation_matrices)}, "
             "the number of operations")

        if rotation is None:
            rotation = inv_rotation = utils.symmetry.identity_matrix
        else:
            inv_rotation = np.linalg.inv(rotation)
        if translation is None:
            translation = utils.symmetry.origin

        if rotation2 is None:
            rotation2 = inv_rotation2 = utils.symmetry.identity_matrix
        else:
            inv_rotation2 = np.linalg.inv(rotation2)
        if translation2 is None:
            translation2 = utils.symmetry.origin
        # this is helpful for dihedral symmetry as entity must be transformed to origin to get canonical dihedral
        # entity_inv = entity.get_transformed_copy(rotation=inv_expand_matrix, rotation2=inv_set_matrix[group])
        # need to reverse any external transformation to the entity coords so rotation occurs at the origin...
        # and undo symmetry expansion matrices
        # centered_coords = transform_coordinate_sets(self.coords, translation=-translation2,
        # centered_coords = transform_coordinate_sets(self._coords.coords, translation=-translation2)
        cb_coords = self.cb_coords
        centered_coords = transform_coordinate_sets(cb_coords, translation=-translation2)

        centered_coords_inv = transform_coordinate_sets(centered_coords, rotation=inv_rotation2,
                                                        translation=-translation, rotation2=inv_rotation)
        _expand_matrices = [utils.symmetry.identity_matrix]
        _expand_translations = [utils.symmetry.origin]
        subunit_count = count()
        for rotation_matrix in degeneracy_rotation_matrices:
            if next(subunit_count) == 0 and np.all(rotation_matrix == utils.symmetry.identity_matrix):
                self.log.debug(f'Skipping {self.make_oligomer.__name__} transformation 1 as it is identity')
                continue
            rot_centered_coords = transform_coordinate_sets(centered_coords_inv, rotation=rotation_matrix)
            new_coords = transform_coordinate_sets(rot_centered_coords, rotation=rotation, translation=translation,
                                                   rotation2=rotation2, translation2=translation2)
            _, rot, tx = superposition3d(new_coords, cb_coords)
            _expand_matrices.append(rot)
            _expand_translations.append(tx)

        self._expand_matrices = np.array(_expand_matrices).swapaxes(-2, -1)
        self._expand_translations = np.array(_expand_translations)[:, None, :]

        # Set the new properties
        self.reset_mates()
        self._set_chain_ids()

    def get_transformed_mate(self, rotation: list[list[float]] | np.ndarray = None,
                             translation: list[float] | np.ndarray = None,
                             rotation2: list[list[float]] | np.ndarray = None,
                             translation2: list[float] | np.ndarray = None) -> Entity:
        """Make a semi-deep copy of the Entity, stripping any captain attributes, transforming the coordinates

        Transformation proceeds by matrix multiplication and vector addition with the order of operations as:
        rotation, translation, rotation2, translation2

        Args:
            rotation: The first rotation to apply, expected array shape (3, 3)
            translation: The first translation to apply, expected array shape (3,)
            rotation2: The second rotation to apply, expected array shape (3, 3)
            translation2: The second translation to apply, expected array shape (3,)

        Returns:
            A transformed copy of the original object
        """
        if rotation is not None:  # required for np.ndarray or None checks
            new_coords = np.matmul(self.coords, np.transpose(rotation))
        else:
            new_coords = self.coords

        if translation is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation)

        if rotation2 is not None:  # required for np.ndarray or None checks
            np.matmul(new_coords, np.transpose(rotation2), out=new_coords)

        if translation2 is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation2)

        new_structure = self.copy()
        new_structure._make_mate(self)
        new_structure.coords = new_coords

        return new_structure

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, header: str = None,
              assembly: bool = False, **kwargs) -> AnyStr | None:
        """Write Entity Structure to a file specified by out_path or with a passed file_handle

        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            header: A string that is desired at the top of the file
            assembly: Whether to write the oligomeric form of the Entity

        Keyword Args:
            increment_chains: bool = False - Whether to write each Structure with a new chain name, otherwise write as
                a new Model
            chain_id: str = None - The chain ID to use
            atom_offset: int = 0 - How much to offset the atom number by. Default returns one-indexed.
                Not used if assembly=True

        Returns:
            The name of the written file if out_path is used
        """
        self.log.debug(f'{Entity.__name__} is writing {repr(self)}')

        def _write(handle) -> None:
            if assembly:
                kwargs.pop('atom_offset', None)
                # if 'increment_chains' not in kwargs:
                #     kwargs['increment_chains'] = True
                # assembly_models = self._generate_assembly_models(**kwargs)
                assembly_models = Models(self.chains)
                assembly_models.write(file_handle=handle, multimodel=False, **kwargs)
            else:
                super(Structure, Structure).write(self, file_handle=handle, **kwargs)

        if file_handle:
            return _write(file_handle)
        else:  # out_path always has default argument current working directory
            # assembly=True implies all chains will be written, so asu=False to write each SEQRES record
            _header = self.format_header(assembly=assembly, **kwargs)
            if header is not None:
                if not isinstance(header, str):
                    header = str(header)
                _header += (header if header[-2:] == '\n' else f'{header}\n')

            with open(out_path, 'w') as outfile:
                outfile.write(_header)
                _write(outfile)

            return out_path

    def calculate_metrics(self, **kwargs) -> dict[str, Any]:
        """"""
        self.log.debug(f"{self.calculate_spatial_orientation_metrics.__name__} missing argument 'reference'")
        return self.calculate_spatial_orientation_metrics()

    def calculate_spatial_orientation_metrics(self, reference: np.ndarray = utils.symmetry.origin) -> dict[str, Any]:
        """Calculate metrics for the instance

        Args:
            reference: The reference where the point should be measured from

        Returns:
            {'radius'
             'min_radius'
             'max_radius'
             'n_terminal_orientation'
             'c_terminal_orientation'
            }
        """
        return {
            'radius': self.assembly.distance_from_reference(reference=reference),
            'min_radius': self.assembly.distance_from_reference(measure='min', reference=reference),
            'max_radius': self.assembly.distance_from_reference(measure='max', reference=reference),
            'n_terminal_orientation': self.termini_proximity_from_reference(reference=reference),
            'c_terminal_orientation': self.termini_proximity_from_reference(termini='c', reference=reference),
        }

    def get_alphafold_template_features(self, symmetric: bool = False, heteromer: bool = False, **kwargs) \
            -> FeatureDict:
        # if symmetric or heteromer:
        #     # raise NotImplementedError("Can't get multimeric features in "
        #     #                           f"{self.get_alphafold_template_features.__name__}")
        #     # Create a stack with 1, oligomeric.number_of_residues, feature dimension length
        #     # as found in HmmsearchHitFeaturizer(TemplateHitFeaturizer).get_templates()
        #     sequence = self.oligomer.sequence
        #     template_features = {
        #         'template_all_atom_positions': np.array([self.oligomer.alphafold_coords], dtype=np.float32),
        #         'template_all_atom_masks': np.array([self.oligomer.alphafold_atom_mask], dtype=np.int32),
        #         'template_sequence': np.array([sequence.encode()], dtype=object),
        #         'template_aatype': np.array([sequence_to_one_hot(sequence,
        #                                                          numerical_translation_alph3_unknown_gaped_bytes)],
        #                                     dtype=np.int32),
        #         'template_domain_names': np.array([self.name.encode()], dtype=object)
        #     }
        # else:
        # Create a stack with 1, number_of_residues, feature dimension length
        # as found in HmmsearchHitFeaturizer(TemplateHitFeaturizer).get_templates()
        template_features = {
            'template_all_atom_positions': np.array([self.alphafold_coords], dtype=np.float32),
            'template_all_atom_masks': np.array([self.alphafold_atom_mask], dtype=np.float32),
            'template_sequence': np.array([self.sequence.encode()], dtype=object),
            'template_aatype': np.array([sequence_to_one_hot(self.sequence,
                                                             numerical_translation_alph3_unknown_gaped_bytes)],
                                        dtype=np.float32),  # np.int32),
            'template_domain_names': np.array([self.name.encode()], dtype=object)
        }
        return template_features

    def get_alphafold_features(self, symmetric: bool = False, heteromer: bool = False, msas: Sequence = tuple(),
                               no_msa: bool = False, templates: bool = False, **kwargs) -> FeatureDict:
        # multimer: bool = False,
        """Retrieve the required feature dictionary for this instance to use in Alphafold inference

        Args:
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
                    f"Couldn't {self.get_alphafold_features.__name__} with both 'symmetric' and "
                    f"'heteromer' True. Only run with symmetric True if this {self.__class__.__name__} "
                    "instance alone should be predicted as a multimer")
        # if templates:
        #     if symmetric:
        #         # raise ValueError(f"Couldn't {self.get_alphafold_features.__name__} with both 'symmetric' and "
        #         logger.warning(f"Couldn't {self.get_alphafold_features.__name__} with both 'symmetric' and "
        #                        f"'templates' True. Templates not set up for multimer")
        # elif symmetric:
        #     # Set multimer True as we need to make_msa_features_multimeric
        #     multimer = True

        # # IS THIS NECESSARY. DON'T THINK SO IF I HAVE MSA
        # chain_features = alphafold.alphafold.data.pipeline.DataPipeline.process(input_fasta_path=P, msa_output_dir=P)
        # This ^ runs
        number_of_residues = self.number_of_residues
        sequence = self.sequence
        sequence_features = af_pipeline.make_sequence_features(
            sequence=sequence, description=self.name, num_res=number_of_residues)
        # sequence_features = {
        #     'aatype': ,  # MAKE ONE HOT with X i.e.unknown are X
        #     'between_segment_residues': np.zeros((number_of_residues,), dtype=np.int32),
        #     'domain_name': np.array([input_description.encode('utf-8')], dtype=np.object_),
        #     'residue_index': np.arange(number_of_residues, dtype=np.int32),
        #     'seq_length': np.full(number_of_residues, number_of_residues, dtype=np.int32),
        #     'sequence': np.array([sequence.encode('utf-8')], dtype=np.object_)
        # }

        def make_msa_features_multimeric(msa_feats: FeatureDict) -> FeatureDict:
            """Create the feature names for Alphafold heteromeric inputs run in multimer mode"""
            valid_feats = af_msa_pairing.MSA_FEATURES + ('msa_species_identifiers',)
            return {f'{k}_all_seq': v for k, v in msa_feats.items() if k in valid_feats}

        # Multiple sequence alignment processing
        if msas:
            msa_features = af_pipeline.make_msa_features(msas)
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
                # with open(self.msa_file, 'r') as f:
                #     uniref90_lines = f.read()

                uniref90_msa = af_data_parsers.parse_stockholm(uniref90_lines)
                msa_features = af_pipeline.make_msa_features((uniref90_msa,))
                msa_features.update(make_msa_features_multimeric(msa_features))
        else:
            msa = self.msa
            if no_msa or msa is None:  # or self.msa_file is None:
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
                # self.log.critical(f'982 Found {len(np.flatnonzero(msa.query_indices))} indices utilized in design')
            # Todo
            #  move to additional AlphaFold set up function...
            #  elif os.path.exists(self.msa_file):
            #      with open(self.msa_file, 'r') as f:
            #          uniclust_lines = f.read()
            #      file, extension = os.path.splitext(self.msa_file)
            #      if extension == '.sto':
            #          uniclust30_msa = af_data_parsers.parse_stockholm(uniclust_lines)
            #      else:
            #          raise ValueError(
            #              f"Currently, the multiple sequence alignment file type '{extension}' isn't supported\n"
            #              f"\tOffending file located at: {self.msa_file}")
            #      msas = (uniclust30_msa,)
            else:
                raise ValueError("Couldn't acquire AlphaFold msa features")

            self.log.debug(f"Found the first 5 species_ids: {species_ids[:5]}")
            msa_features = {
                'deletion_matrix_int': deletion_matrix,
                # When not single sequence, GET THIS FROM THE MATRIX PROBABLY USING CODE IN COLLAPSE PROFILE cumcount...
                # 'msa': sequences_to_numeric([sequence], translation_table=HHBLITS_AA_TO_ID).astype(dtype=np.int32),
                'msa': msa_numeric,
                'num_alignments': np.full(number_of_residues, num_sequences, dtype=np.int32),
                # Fill by the number of residues how many sequences are in the MSA
                'msa_species_identifiers': np.array([id_.encode('utf-8') for id_ in species_ids], dtype=np.object_)
            }
            # Debug features
            for feat, values in msa_features.items():
                self.log.debug(f'For feature {feat}, found shape {values.shape}')

            if heteromer:
                # Make a deepcopy just incase this screws up something
                msa_features.update(make_msa_features_multimeric(deepcopy(msa_features)))

        # Template processing
        if templates:
            template_features = self.get_alphafold_template_features()  # symmetric=symmetric, heteromer=heteromer)
        else:
            template_features = empty_placeholder_template_features(num_templates=0, num_res=number_of_residues)
        # Debug template features
        for feat, values in template_features.items():
            self.log.debug(f'For feature {feat}, found shape {values.shape}')

        entity_features = {
            **msa_features,
            **sequence_features,
            **template_features
        }
        if symmetric and self.is_symmetric():
            # Hard code in chain_id as we are using a multimeric predict on the oligomeric version
            chain_id = 'A'
            entity_features = af_pipeline_multimer.convert_monomer_features(entity_features, chain_id=chain_id)

            chain_count = count(1)
            entity_integer = 1
            entity_id = af_pipeline_multimer.int_id_to_str_id(entity_integer)
            all_chain_features = {}
            for sym_idx in range(1, 1 + self.number_of_symmetry_mates):
                # chain_id = next(available_chain_ids_iter)  # The mmCIF formatted chainID with 'AB' type notation
                this_entity_features = deepcopy(entity_features)
                # Where chain_id increments for each new chain instance i.e. A_1 is 1, A_2 is 2, ...
                # Where entity_id increments for each new Entity instance i.e. A_1 is 1, A_2 is 1, ...
                # Where sym_id increments for each new Entity instance regardless of chain i.e. A_1 is 1, A_2 is 2, ...,
                # B_1 is 1, B2 is 2
                this_entity_features.update({'asym_id': next(chain_count) * np.ones(number_of_residues),
                                             'sym_id': sym_idx * np.ones(number_of_residues),
                                             'entity_id': entity_integer * np.ones(number_of_residues)})
                chain_name = f'{entity_id}_{sym_idx}'
                all_chain_features[chain_name] = this_entity_features

            # Alternative to pair_and_merge using hhblits a3m output
            # See PMID:36224222 "Structural predictions of dimeric and trimeric subcomponents" methods section
            # The first of the two MSAs is constructed by extracting the organism identifiers (OX) from the resulting
            # a3m file and pairing sequences using the top hit from each OX. The second is constructed by block
            # diagonalizing the resulting a3m file.
            np_example = af_feature_processing.pair_and_merge(all_chain_features=all_chain_features)
            # Pad MSA to avoid zero-sized extra_msa.
            np_example = af_pipeline_multimer.pad_msa(np_example, 512)

            return np_example  # This is still a FeatureDict and could be named entity_features
        else:
            return entity_features

    def find_chain_symmetry(self):
        """Search for the chain symmetry by using quaternion geometry to solve the symmetric order of the rotations
         which superimpose chains on the Entity. Translates the Entity to the origin using center of mass, then the axis
        of rotation only needs to be translated to the center of mass to recapitulate the specific symmetry operation

        Requirements - all chains are the same length

        Sets:
            self.mate_rotation_axes (list[dict[str, int | np.ndarray]])
            self._max_symmetry (int)
            self.max_symmetry_chain_idx (int)

        Returns:
            The name of the file written for symmetry definition file creation
        """
        # Find the superposition from the Entity to every mate chain
        # center_of_mass = self.center_of_mass
        # symmetric_center_of_mass = self.center_of_mass_symmetric
        # self.log.debug(f'symmetric_center_of_mass={symmetric_center_of_mass}')
        self.mate_rotation_axes.clear()
        self.mate_rotation_axes.append({'sym': 1, 'axis': utils.symmetry.origin})
        self.log.debug(f'Reference chain is {self.chain_id}')
        if self.is_symmetric():

            def _get_equivalent_coords(
                self_ca_coords: np.ndarray, self_seq: str, chain_: Chain
            ) -> tuple[np.ndarray, np.ndarray]:
                return self_ca_coords, chain_.ca_coords
        else:

            def _get_equivalent_coords(
                self_ca_coords: np.ndarray, self_seq: str, chain_: Chain
            ) -> tuple[np.ndarray, np.ndarray]:
                chain_seq = chain_.sequence
                additional_chain_coords = chain_.ca_coords
                if chain_seq != self_seq:
                    # Get aligned indices, then follow with superposition
                    self.log.debug(f'{repr(chain_)} and {repr(self)} require alignment to symmetrize')
                    fixed_indices, moving_indices = get_equivalent_indices(chain_seq, self_seq)
                    additional_chain_coords = additional_chain_coords[fixed_indices]
                    self_ca_coords = self_ca_coords[moving_indices]

                return self_ca_coords, additional_chain_coords

        ca_coords = self.ca_coords
        sequence = self.sequence
        for chain in self.chains[1:]:
            self_coords, chain_coords = _get_equivalent_coords(ca_coords, sequence, chain)
            rmsd, quat, tx = superposition3d_quat(self_coords, chain_coords)
            # rmsd, quat, tx = superposition3d_quat(cb_coords-center_of_mass, chain.cb_coords-center_of_mass)
            self.log.debug(f'rmsd={rmsd} quaternion={quat} translation={tx}')
            w = abs(quat[3])
            omega = math.acos(w)
            try:
                symmetry_order = int(math.pi/omega + .5)  # Round to the nearest integer
            except ZeroDivisionError:  # w is 1, omega is 0
                # No axis of symmetry here
                symmetry_order = 1
                self.log.warning(f"Couldn't find any symmetry order for {self.name} mate Chain {chain.chain_id}. "
                                 f'Setting symmetry_order={symmetry_order}')
            self.log.debug(f'{chain.chain_id}:{symmetry_order}-fold axis')
            self.mate_rotation_axes.append({'sym': symmetry_order, 'axis': quat[:3]})

        # Find the highest order symmetry in the Structure
        max_sym = 0
        max_chain_idx = None
        for chain_idx, data in enumerate(self.mate_rotation_axes):
            if data['sym'] > max_sym:
                max_sym = data['sym']
                max_chain_idx = chain_idx

        self._max_symmetry = max_sym
        self._max_symmetry_chain_idx = max_chain_idx

    @property
    def max_symmetry_chain_idx(self) -> int:
        """The maximum symmetry order present"""
        try:
            return self._max_symmetry_chain_idx
        except AttributeError:
            self.find_chain_symmetry()
            return self._max_symmetry_chain_idx

    @property
    def max_symmetry(self) -> int:
        """The maximum symmetry order present"""
        try:
            return self._max_symmetry
        except AttributeError:
            self.find_chain_symmetry()
            return self._max_symmetry

    def is_cyclic(self) -> bool:
        """Report whether the symmetry is cyclic

        Returns:
            True if the Structure is cyclic, False if not
        """
        return self.number_of_symmetry_mates == self.max_symmetry

    def is_dihedral(self) -> bool:
        """Report whether the symmetry is dihedral

        Returns:
            True if the Structure is dihedral, False if not
        """
        return self.number_of_symmetry_mates / self.max_symmetry == 2

    def find_dihedral_chain(self) -> Entity | None:  # Todo python 3.11 self
        """From the symmetric system, find a dihedral chain and return the instance

        Returns:
            The dihedral mate Chain
        """
        if not self.is_dihedral():
            return None

        # Ensure if the structure is dihedral a selected dihedral_chain is orthogonal to the maximum symmetry axis
        max_symmetry_axis = self.mate_rotation_axes[self.max_symmetry_chain_idx]['axis']
        for chain_idx, data in enumerate(self.mate_rotation_axes):
            this_chain_axis = data['axis']
            if data['sym'] == 2:
                axis_dot_product = np.dot(max_symmetry_axis, this_chain_axis)
                if axis_dot_product < 0.01:
                    if np.allclose(this_chain_axis, [1, 0, 0]):
                        self.log.debug(f'The relation between {self.max_symmetry_chain_idx} and {chain_idx} would '
                                       'result in a malformed .sdf file')
                        pass  # This won't work in the 'make_symmdef.pl' script, should choose orthogonal y-axis
                    else:
                        return self.chains[chain_idx]
        return None

    def make_sdf(self, struct_file: AnyStr = None, out_path: AnyStr = os.getcwd(), **kwargs) -> AnyStr:
        """Uses the 'make_symmdef_file.pl' script from Rosetta to make a symmetry definition file on the Structure

        perl $ROSETTA/source/src/apps/public/symmetry/make_symmdef_file.pl -p filepath/to/pdb.pdb -i B -q

        Args:
            struct_file: The location of the input .pdb file
            out_path: The location the symmetry definition file should be written

        Keyword Args:
            modify_sym_energy_for_cryst: bool = False - Whether the symmetric energy in the file should be modified
            energy: int = 2 - Scalar to modify the Rosetta energy by

        Returns:
            Symmetry definition filename
        """
        out_file = os.path.join(out_path, f'{self.name}.sdf')
        if os.path.exists(out_file):
            return out_file

        if self.symmetry in utils.symmetry.cubic_point_groups:
            sdf_mode = 'PSEUDO'
            self.log.warning('Using experimental symmetry definition file generation, proceed with caution as Rosetta '
                             'runs may fail due to improper set up')
        else:
            sdf_mode = 'NCS'

        if not struct_file:
            struct_file = self.write(assembly=True, out_path=f'make_sdf_input-{self.name}-{random() * 100000:.0f}.pdb',
                                     increment_chains=True)

        # As increment_chains is used, get the chain name corresponding to the same index as incremental chain
        available_chain_ids = chain_id_generator()
        for _ in range(self.max_symmetry_chain_idx):
            next(available_chain_ids)
        chains = [next(available_chain_ids)]
        if self.is_dihedral():
            chains.append(self.find_dihedral_chain().chain_id)

        sdf_cmd = [
            'perl', str(putils.make_sdf_path), '-m', sdf_mode, '-q', '-p', struct_file, '-a', self.chain_ids[0], '-i'
        ] + chains
        self.log.info(f'Creating symmetry definition file: {subprocess.list2cmdline(sdf_cmd)}')
        # with open(out_file, 'w') as file:
        p = subprocess.Popen(sdf_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, err = p.communicate()

        if os.path.exists(struct_file):
            os.system(f'rm {struct_file}')
        if p.returncode != 0:
            raise DesignError(
                f'Symmetry definition file creation failed for {self.name}')

        self.format_sdf(out.decode('utf-8').split('\n')[:-1], to_file=out_file, **kwargs)
        #                 modify_sym_energy_for_cryst=False, energy=2)

        return out_file

    def format_sdf(self, lines: list, to_file: AnyStr = None, out_path: AnyStr = os.getcwd(),
                   modify_sym_energy_for_cryst: bool = False, energy: int = None) -> AnyStr:
        """Ensure proper sdf formatting before proceeding

        Args:
            lines: The symmetry definition file lines
            to_file: The name of the symmetry definition file
            out_path: The location the symmetry definition file should be written
            modify_sym_energy_for_cryst: Whether the symmetric energy should match crystallographic systems
            energy: Scalar to modify the Rosetta energy by

        Returns:
            The location the symmetry definition file was written
        """
        subunits, virtuals, jumps_com, jumps_subunit, trunk = [], [], [], [], []
        for idx, line in enumerate(lines, 1):
            if line.startswith('xyz'):
                virtual = line.split()[1]
                if virtual.endswith('_base'):
                    subunits.append(virtual)
                else:
                    virtuals.append(virtual.lstrip('VRT'))
                # last_vrt = line + 1
            elif line.startswith('connect_virtual'):
                jump = line.split()[1].lstrip('JUMP')
                if jump.endswith('_to_com'):
                    jumps_com.append(jump[:-7])
                elif jump.endswith('_to_subunit'):
                    jumps_subunit.append(jump[:-11])
                else:
                    trunk.append(jump)
                last_jump = idx  # index where the VRTs and connect_virtuals end. The "last jump"

        if set(trunk).difference(virtuals):
            raise SymmetryError(
                f"Symmetry Definition File VRTS are malformed. See '{to_file}'")
        if len(subunits) != self.number_of_symmetry_mates:
            raise SymmetryError(
                f"Symmetry Definition File VRTX_base are malformed. See '{to_file}'")

        if self.is_dihedral():  # Remove dihedral connecting (trunk) virtuals: VRT, VRT0, VRT1
            virtuals = [virtual for virtual in virtuals if len(virtual) > 1]  # subunit_
        else:
            try:
                virtuals.remove('')
            except ValueError:  # '' not present
                pass

        jumps_com_to_add = set(virtuals).difference(jumps_com)
        count_ = 0
        if jumps_com_to_add:
            for count_, jump_com in enumerate(jumps_com_to_add, count_):
                lines.insert(last_jump + count_,
                             f'connect_virtual JUMP{jump_com}_to_com VRT{jump_com} VRT{jump_com}_base')
            lines[-2] = lines[-2].strip() + (' JUMP%s_to_subunit' * len(jumps_com_to_add)) % tuple(jumps_com_to_add)

        jumps_subunit_to_add = set(virtuals).difference(jumps_subunit)
        if jumps_subunit_to_add:
            for count_, jump_subunit in enumerate(jumps_subunit_to_add, count_):
                lines.insert(last_jump + count_,
                             f'connect_virtual JUMP{jump_subunit}_to_subunit VRT{jump_subunit}_base SUBUNIT')
            lines[-1] = \
                lines[-1].strip() + (' JUMP%s_to_subunit' * len(jumps_subunit_to_add)) % tuple(jumps_subunit_to_add)

        if modify_sym_energy_for_cryst:
            # new energy should equal the energy multiplier times the scoring subunit plus additional complex subunits
            # where complex subunits = num_subunits - 1
            # new_energy = 'E = %d*%s + ' % (energy, subunits[0])  # assumes subunits are read in alphanumerical order
            # new_energy += ' + '.join('1*(%s:%s)' % t for t in zip(repeat(subunits[0]), subunits[1:]))
            lines[1] = f'E = 2*{subunits[0]}+{"+".join(f"1*({subunits[0]}:{pair})" for pair in subunits[1:])}'
        else:
            if not energy:
                energy = len(subunits)
            lines[1] = \
                f'E = {energy}*{subunits[0]}+{"+".join(f"{energy}*({subunits[0]}:{pair})" for pair in subunits[1:])}'

        if not to_file:
            to_file = os.path.join(out_path, f'{self.name}.sdf')

        with open(to_file, 'w') as f:
            f.write('%s\n' % '\n'.join(lines))
        if count_ != 0:
            self.log.info(f"Symmetry Definition File '{to_file}' was missing {count_} lines. A fix was attempted and "
                          'modeling may be affected')
        return to_file

    def reset_mates(self):
        """Remove oligomeric chains. They should be generated fresh"""
        self._chains.clear()
        self._chains.append(self)

    @ContainsResidues.fragment_db.setter
    def fragment_db(self, fragment_db: FragmentDatabase):
        """Set the Structure FragmentDatabase to assist with Fragment creation, manipulation, and profiles.
        Sets .fragment_db for each dependent Structure in 'structure_containers'

        Entity specific implementation to prevent recursion with [1:]
        """
        # Set this instance then set all dependents
        super(Structure, Structure).fragment_db.fset(self, fragment_db)
        _fragment_db = self._fragment_db
        if _fragment_db is not None:
            for structure_type in self.structure_containers:
                for structure in self.__getattribute__(structure_type)[1:]:
                    structure.fragment_db = _fragment_db
        else:  # This is likely the RELOAD_DB token. Just return.
            return

    def _update_structure_container_attributes(self, **kwargs):
        """Update attributes specified by keyword args for all Structure container members. Entity specific handling"""
        # As this causes error in keeping mate chains, mates, just return
        return

    def _copy_structure_containers(self):
        """Copy all contained Structure members. Entity specific handling of chains index 0"""
        # self.log.debug('In Entity copy_structure_containers()')
        for structure_type in self.structure_containers:
            # Get and copy the structure container
            new_structures = self.__getattribute__(structure_type).copy()
            for idx, structure in enumerate(new_structures[1:], 1):  # Only operate on [1:] since index 0 is different
                structure.entity_spawn = True
                new_structure = structure.copy()
                new_structure._captain = self
                new_structures[idx] = new_structure
            # Set the copied and updated structure container
            self.__setattr__(structure_type, new_structures)

    def __copy__(self) -> Entity:  # Todo -> Self: in python 3.11
        # self.log.debug('In Entity copy')
        # Save, then remove the _captain attribute before the copy
        captain = self._captain
        del self._captain
        try:
            _oligomer = self._oligomer
        except AttributeError:
            _oligomer = None
        else:
            del self._oligomer

        other: Entity = super().__copy__()
        # Mate Entity instances are a "parent", however, they are under the control of their captain instance
        if self._is_captain:  # If the copier is a captain
            other._captain = None  # Initialize the copy as a captain, i.e. _captain = None
            if self.is_dependent() and other.is_dependent():
                # A parent initiated the copy and the structure_containers weren't copied in super().copy().
                # They must be created new so copy here
                other._copy_structure_containers()
        else:
            # If the .entity_spawn attribute is set, the copy was initiated by the captain,
            # ._captain will be set after this __copy__ return in _copy_structure_containers()
            try:  # To delete entity_spawn attribute
                del self.entity_spawn
                del other.entity_spawn
            except AttributeError:
                # This isn't a captain and a captain didn't initiate the copy
                # First set other._captain to self._captain, then
                # _make_captain() extracts data, and finally, will set _captain as None
                other._captain = captain
                other._make_captain()  # other._captain -> None
                # Have to make it a captain -> None

        # Set the first chain as the object itself
        other._chains[0] = other
        # Reset the self state as before the copy. Reset _captain attribute on self
        self._captain = captain
        if _oligomer is not None:
            self._oligomer = _oligomer

        return other

    copy = __copy__  # Overwrites to use this instance __copy__

    # @property
    # def _key(self) -> tuple[str, int, ...]:
    #     return self.name, *self._residue_indices

    def __eq__(self, other: StructureBase) -> bool:
        if isinstance(other, Entity):
            # The first is True if this is a mate, the second is True if this is a captain
            return id(self._captain) == id(other) or id(other._captain) == id(self) or self._key == other._key
        elif isinstance(other, StructureBase):
            return self._key == other._key
        raise NotImplementedError(
            f"Can't compare {self.__class__.__name__} instance to {type(other).__name__} instance")

    # Must define __hash__ in all subclasses that define an __eq__
    def __hash__(self) -> int:
        return hash(self._key)


class Model(ContainsChains):
    """The main class for simple Structure manipulation, particularly containing multiple Chain instances

    Can initialize by passing a file, or passing Atom/Residue/Chain instances. If your Structure is symmetric, a
    SymmetricModel should be used instead. If you have multiple Model instances, use the MultiModel class.
    """


# class Models(Structures):
class Models(UserList):  # (Model):
    """Container for Model instances. Primarily used for writing [symmetric] multimodel-like Structure instances"""
    # _models_coords: Coords
    models: list[Model]  # Could be SymmetricModel/Pose
    # state_attributes: set[str] = Model.state_attributes | {'_models_coords'}

    @classmethod
    def from_models(cls, models: Iterable[Model], **kwargs):
        """Initialize from an iterable of Model instances"""
        return cls(models=models, **kwargs)

    def __init__(self, models: Iterable[ContainsEntities | Entity], name: str = None, **kwargs):
        super().__init__(initlist=models)  # Sets UserList.data to models

        for model in self:
            if not isinstance(model, (ContainsEntities, Entity)):
                raise TypeError(
                    f"Can't initialize {self.__class__.__name__} with a {type(model).__name__}. Must be an Iterable"
                    f' of {Model.__name__}')

        self.name = name if name else f'Nameless-{Models.__name__}'

    @property
    def number_of_models(self) -> int:
        """The number of unique models that are found in the Models object"""
        return len(self.data)

    # @property
    # def models_coords(self) -> np.ndarray | None:
    #     """Return a concatenated view of the Coords from all models"""
    #     try:
    #         return self._models_coords.coords
    #     except AttributeError:
    #         return None
    #
    # @models_coords.setter
    # def models_coords(self, coords: Coords | np.ndarray | list[list[float]]):
    #     if isinstance(coords, Coords):
    #         self._models_coords = coords
    #     else:
    #         self._models_coords = Coords(coords)

    # def append(self, model: Model):
    #     """Append an existing Model into the Models instance
    #
    #     Sets:
    #         self.models
    #     """
    #     self.models.append(model)

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, header: str = None,
              multimodel: bool = False, increment_chains: bool = False, **kwargs) -> AnyStr | None:
        """Write Models to a file specified by out_path or with a passed file_handle

        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            header: A string that is desired at the top of the file
            multimodel: Whether MODEL and ENDMDL records should be added at the end of each Model
            increment_chains: Whether to write each Chain with an incrementing chain ID,
                otherwise use the chain IDs present, repeating for each Model

        Keyword Args
            assembly: bool = False - Whether to write an assembly representation of each Model instance

        Returns:
            The name of the written file if out_path is used
        """
        logger.debug(f'{Models.__name__} is writing {repr(self)}')

        def _write(handle) -> None:
            if increment_chains:
                available_chain_ids = chain_id_generator()

                def _get_chain_id(struct: Chain) -> str:
                    return next(available_chain_ids)

            else:

                def _get_chain_id(struct: Chain) -> str:
                    return struct.chain_id

            chain: Chain
            offset = 0

            def _write_model(_model):
                nonlocal offset
                for chain in _model.entities:
                    chain_id = _get_chain_id(chain)
                    chain.write(file_handle=handle, chain_id=chain_id, atom_offset=offset, **kwargs)
                    c_term_residue: Residue = chain.c_terminal_residue
                    offset += chain.number_of_atoms
                    handle.write(f'TER   {offset + 1:>5d}      {c_term_residue.type:3s} '
                                 f'{chain_id:1s}{c_term_residue.number:>4d}\n')

            if multimodel:
                for model_number, model in enumerate(self, 1):
                    handle.write('{:9s}{:>4d}\n'.format('MODEL', model_number))
                    _write_model(model)
                    handle.write('ENDMDL\n')
            else:
                for model in self:
                    _write_model(model)

        if file_handle:
            return _write(file_handle)
        else:  # out_path always has default argument current working directory
            _header = ''  # self.format_header(**kwargs)
            if header is not None:
                if not isinstance(header, str):
                    header = str(header)
                _header += (header if header[-2:] == '\n' else f'{header}\n')

            with open(out_path, 'w') as outfile:
                outfile.write(_header)
                _write(outfile)
            return out_path

    def __iter__(self) -> ContainsChains:
        yield from self.data

    def __getitem__(self, idx: int) -> Model:
        return self.data[idx]


class SymmetricModel(SymmetryOpsMixin, ContainsEntities):
    _assembly: Model
    _assembly_minimally_contacting: Model
    _assembly_tree: BinaryTreeType
    """Stores a sklearn tree for coordinate searching"""
    _center_of_mass_symmetric_entities: list[list[np.ndarray]]  # list[np.ndarray]
    _oligomeric_model_indices: dict[Entity, list[int]]
    _symmetric_coords_by_entity: list[np.ndarray]
    _symmetric_coords_split_by_entity: list[list[np.ndarray]]
    _transformation: list[types.TransformationMapping] | list[dict]
    symmetry_state_attributes: set[str] = SymmetryOpsMixin.symmetry_state_attributes | {
        '_assembly', '_assembly_minimally_contacting', '_assembly_tree', '_center_of_mass_symmetric_entities',
        '_oligomeric_model_indices', '_symmetric_coords_by_entity', '_symmetric_coords_split_by_entity'
    }

    @classmethod
    def from_assembly(cls, assembly: Model, sym_entry: utils.SymEntry.SymEntry | int = None,
                      symmetry: str = None, **kwargs):
        """Initialize from a symmetric assembly"""
        if symmetry is None and sym_entry is None:
            raise ValueError(
                "Can't initialize without symmetry. Pass 'symmetry' or 'sym_entry' to "
                f'{cls.__name__}.{cls.from_assembly.__name__}() constructor')
        return cls(structure=assembly, sym_entry=sym_entry, symmetry=symmetry, **kwargs)

    @SymmetryBase.symmetry.setter
    def symmetry(self, symmetry):
        super(StructureBase, StructureBase).symmetry.fset(self, symmetry)
        if self.is_symmetric() and self.has_dependent_chains():
            self.structure_containers.remove('_chains')
            # self._chains will be dependent on self._entities now

    def set_symmetry(self, sym_entry: utils.SymEntry.SymEntry | int = None, symmetry: str = None,
                     crystal: bool = False, cryst_record: str = None, uc_dimensions: list[float] = None,
                     operators: tuple[np.ndarray | list[list[float]], np.ndarray | list[float]] | np.ndarray = None,
                     rotations: np.ndarray | list[list[float]] = None, translations: np.ndarray | list[float] = None,
                     transformations: list[types.TransformationMapping] = None, surrounding_uc: bool = True,
                     **kwargs):
        """Set the model symmetry using the CRYST1 record, or the unit cell dimensions and the Hermann-Mauguin symmetry
        notation (in CRYST1 format, ex P432) for the Model assembly. If the assembly is a point group, only the symmetry
        notation is required

        Args:
            sym_entry: The SymEntry which specifies all symmetry parameters
            symmetry: The name of a symmetry to be searched against compatible symmetries
            crystal: Whether crystalline symmetry should be used
            cryst_record: If a CRYST1 record is known and should be used
            uc_dimensions: The unit cell dimensions for the crystalline symmetry
            operators: A set of custom expansion matrices
            rotations: A set of rotation matrices used to recapitulate the SymmetricModel from the asymmetric unit
            translations: A set of translation vectors used to recapitulate the SymmetricModel from the asymmetric unit
            transformations: Transformation operations that reproduce the oligomeric state for each Entity
            surrounding_uc: Whether the 3x3 layer group, or 3x3x3 space group should be generated
            crystal: Whether crystalline symmetry should be used
            cryst_record: If a CRYST1 record is known and should be used
        """
        chains = self._chains
        super().set_symmetry(
            sym_entry=sym_entry, symmetry=symmetry,
            crystal=crystal, cryst_record=cryst_record, uc_dimensions=uc_dimensions,
        )
        number_of_symmetry_mates = self.number_of_symmetry_mates

        if not self.is_symmetric():
            # No symmetry keyword args were passed
            if operators is not None or rotations is not None or translations is not None:
                passed_args = []
                if operators:
                    passed_args.append('operators')
                if rotations:
                    passed_args.append('rotations')
                if translations:
                    passed_args.append('translations')
                raise ConstructionError(
                    f"Couldn't set_symmetry() using {', '.join(passed_args)} without explicitly passing "
                    "'symmetry' or 'sym_entry'"
                )
            return

        # Set rotations and translations to the correct symmetry operations
        # where operators, rotations, and translations are user provided from some sort of BIOMT (fiber, other)
        if operators is not None:
            symmetry_source_arg = "'operators' "

            num_operators = len(operators)
            if isinstance(operators, tuple) and num_operators == 2:
                self.log.warning("Providing custom symmetry 'operators' may result in improper symmetric "
                                 'configuration. Proceed with caution')
                rotations, translations = operators
            elif isinstance(operators, Sequence) and num_operators == number_of_symmetry_mates:
                rotations = []
                translations = []
                try:
                    for rot, tx in operators:
                        rotations.append(rot)
                        translations.append(tx)
                except TypeError:  # Unpack failed
                    raise ValueError(
                        f"Couldn't parse the 'operators'={repr(operators)}.\n\n"
                        "Expected a Sequence[rotation shape=(3,3). translation shape=(3,)] pairs."
                    )
            elif isinstance(operators, np.ndarray):
                if operators.shape[1:] == (3, 4):
                    # Parse from a single input of 3 row by 4 column style, like BIOMT
                    rotations = operators[:, :, :3]
                    translations = operators[:, :, 3:].squeeze()
                elif operators.shape[1:] == 3:  # Assume just rotations
                    rotations = operators
                    translations = np.tile(utils.symmetry.origin, len(rotations))
                else:
                    raise ConstructionError(
                        f"The 'operators' form {repr(operators)} isn't supported.")
            else:
                raise ConstructionError(
                    f"The 'operators' form {repr(operators)} isn't supported. Must provide a tuple of "
                    'array-like objects with the order (rotation matrices, translation vectors) or use the '
                    "'rotations' and 'translations' keyword args")
        else:
            symmetry_source_arg = ''

        # Now that symmetry is set, check if the Structure parsed all symmetric chains
        if len(chains) == self.number_of_entities * number_of_symmetry_mates:
            parsed_assembly = True
        else:
            parsed_assembly = False

        # Set the symmetry operations
        if rotations is not None and translations is not None:
            if not isinstance(rotations, np.ndarray):
                rotations = np.ndarray(rotations)
            if rotations.ndim == 3:
                # Assume operators were provided in a standard orientation and transpose for subsequent efficiency
                # Using .swapaxes(-2, -1) call here instead of .transpose() for safety
                self._expand_matrices = rotations.swapaxes(-2, -1)
            else:
                raise SymmetryError(
                    f"Expected {symmetry_source_arg}rotation matrices with 3 dimensions, not {rotations.ndim} "
                    "dimensions. Ensure the passed rotation matrices have a shape of (N symmetry operations, 3, 3)"
                )

            if not isinstance(translations, np.ndarray):
                translations = np.ndarray(translations)
            if translations.ndim == 2:
                # Assume operators were provided in a standard orientation each vector needs to be in own array on dim=2
                self._expand_translations = translations[:, None, :]
            else:
                raise SymmetryError(
                    f"Expected {symmetry_source_arg}translation vectors with 2 dimensions, not {translations.ndim} "
                    "dimensions. Ensure the passed translations have a shape of (N symmetry operations, 3)"
                )
        else:  # The symmetry operators must be possible to find or canonical.
            symmetry_source_arg = "'chains' "
            # Get canonical operators
            if self.dimension == 0:
                # The _expand_matrices rotation matrices are pre-transposed to avoid repetitive operations
                _expand_matrices = utils.symmetry.point_group_symmetry_operatorsT[self.symmetry]
                # The _expand_translations vectors are pre-sliced to enable numpy operations
                _expand_translations = \
                    np.tile(utils.symmetry.origin, (self.number_of_symmetry_mates, 1))[:, None, :]

                if parsed_assembly:
                    # The Structure should have symmetric chains
                    # Set up symmetry operations using orient

                    # Save the original position for subsequent reversion
                    ca_coords = self.ca_coords.copy()
                    # Transform to canonical orientation.
                    self.orient()
                    # Set the symmetry operations again as they are incorrect after orient()
                    self._expand_matrices = _expand_matrices
                    self._expand_translations = _expand_translations
                    # Next, transform back to original and carry the correctly situated symmetry operations along.
                    _, rot, tx = superposition3d(ca_coords, self.ca_coords)
                    self.transform(rotation=rot, translation=tx)
                else:
                    self._expand_matrices = _expand_matrices
                    self._expand_translations = _expand_translations
            else:
                self._expand_matrices, self._expand_translations = \
                    utils.symmetry.space_group_symmetry_operatorsT[self.symmetry]

        # Removed parsed chain information
        self.reset_mates()

        try:
            self._symmetric_coords.coords
        except AttributeError:
            self.generate_symmetric_coords(surrounding_uc=surrounding_uc)

        # Check if the oligomer is constructed for each entity
        for entity, subunit_number in zip(self.entities, self.sym_entry.group_subunit_numbers):
            if entity.number_of_symmetry_mates != subunit_number:
                # Generate oligomers for each entity
                self.make_oligomers(transformations=transformations)
                break

        # Once oligomers are specified, the minimal ASU can be set properly
        self.set_contacting_asu(from_assembly=parsed_assembly)

    @property
    def atom_indices_per_entity_symmetric(self):
        # alt solution may be quicker by performing the following addition then .flatten()
        # broadcast entity_indices ->
        # (np.arange(model_number) * number_of_atoms).T
        # |
        # v
        # number_of_atoms = self.number_of_atoms
        # number_of_atoms = len(self.coords)
        return [self.make_indices_symmetric(entity_indices) for entity_indices in self.atom_indices_per_entity]
        # return [[idx + (number_of_atoms * model_number) for model_number in range(self.number_of_symmetry_mates)
        #          for idx in entity_indices] for entity_indices in self.atom_indices_per_entity]

    @StructureBase.coords.setter
    def coords(self, coords: np.ndarray | list[list[float]]):
        self.log.debug(f'Setting {self.__class__.__name__} coords')
        if self.is_symmetric():  # Set the symmetric coords according to the ASU
            pass
        super(ContainsAtoms, ContainsAtoms).coords.fset(self, coords)

        if self.is_symmetric():  # Set the symmetric coords according to the ASU

            self.log.debug(f'Updating symmetric coords')
            self.generate_symmetric_coords(surrounding_uc=self.is_surrounding_uc())

        # Delete any saved attributes from the SymmetricModel (or Model)
        try:  # To see if setting doesn't require attribute reset in the case this instance is performing .coords set
            self._no_reset
        except AttributeError:
            self.reset_state()

    @property
    def symmetric_coords_split_by_entity(self) -> list[list[np.ndarray]]:
        """A view of the symmetric coordinates split for each symmetric model by the Pose Entity indices"""
        try:
            return self._symmetric_coords_split_by_entity
        except AttributeError:
            symmetric_coords_split = self.symmetric_coords_split
            self._symmetric_coords_split_by_entity = []
            for entity_indices in self.atom_indices_per_entity:
                # self._symmetric_coords_split_by_entity.append(symmetric_coords_split[:, entity_indices])
                self._symmetric_coords_split_by_entity.append(
                    [symmetric_split[entity_indices] for symmetric_split in symmetric_coords_split])

            return self._symmetric_coords_split_by_entity

    @property
    def symmetric_coords_by_entity(self) -> list[np.ndarray]:
        """A view of the symmetric coordinates for each Entity in order of the Pose Entity indices"""
        try:
            return self._symmetric_coords_by_entity
        except AttributeError:
            symmetric_coords = self.symmetric_coords
            self._symmetric_coords_by_entity = []
            for entity_indices in self.atom_indices_per_entity_symmetric:
                self._symmetric_coords_by_entity.append(symmetric_coords[entity_indices])

            return self._symmetric_coords_by_entity

    @property
    def center_of_mass_symmetric_entities(self) -> list[list[np.ndarray]]:
        """The center of mass position for each Entity instance in the symmetric system for each symmetry mate with
        shape [(number_of_symmetry_mates, 3), ... number_of_entities]
        """
        # if self.symmetry:
        # self._center_of_mass_symmetric_entities = []
        # for num_atoms, entity_coords in zip(self.number_of_atoms_per_entity, self.symmetric_coords_split_by_entity):
        #     self._center_of_mass_symmetric_entities.append(np.matmul(np.full(num_atoms, 1 / num_atoms),
        #                                                              entity_coords))
        # return self._center_of_mass_symmetric_entities
        # return [np.matmul(entity.center_of_mass, self._expand_matrices) for entity in self.entities]
        try:
            return self._center_of_mass_symmetric_entities
        except AttributeError:
            self._center_of_mass_symmetric_entities = [
                list(self.return_symmetric_coords(entity.center_of_mass)) for entity in self.entities]
            return self._center_of_mass_symmetric_entities

    def _generate_assembly_name(self, minimal: bool | int | None = False, surrounding_uc: bool = False) -> str:
        """Provide a name for a generated assembly depending on the symmetry

        Args:
            minimal: Whether to create the minimally contacting assembly model
            surrounding_uc: Whether to generate the surrounding unit cells as part of the assembly models

        Returns:
            The name of the assembly based on the symmetry
        """
        if minimal is None:
            # When the Pose is asymmetric
            name = f'{self.name}-copy'
        else:
            if self.dimension:
                if surrounding_uc:
                    symmetry_type_str = 'surrounding-uc-'
                else:
                    symmetry_type_str = 'uc-'
            else:
                symmetry_type_str = ''

            if minimal:  # Only return contacting
                name = f'{self.name}-minimal-assembly'
            else:
                name = f'{self.name}-{symmetry_type_str}assembly'

        return name

    @property
    def assembly(self) -> Model:
        """Provides a Structure instance containing all symmetric chains in the assembly unless the design is 2- or 3-D
        then the assembly only contains the contacting models
        """
        try:
            return self._assembly
        except AttributeError:
            # If the dimension is 0, then generate the full assembly, otherwise, generate partial
            self._assembly = self._generate_assembly(minimal=self.dimension)
            return self._assembly

    @property
    def assembly_minimally_contacting(self) -> Model:
        """Provides a Structure instance only containing the SymmetricModel instances contacting the ASU"""
        try:
            return self._assembly_minimally_contacting
        except AttributeError:
            if self.dimension is None:
                minimal = None
            else:
                minimal = True
            self._assembly_minimally_contacting = self._generate_assembly(minimal=minimal)
            return self._assembly_minimally_contacting

    def _generate_assembly_models(
            self, minimal: bool | int | None = False, surrounding_uc: bool = False, **kwargs  # <- Collect excess kwargs
    ) -> Models:
        """Create a group of Model copies for a specified number of the symmetry mates

        Args:
            minimal: Whether to create the minimally contacting assembly model. This is advantageous in crystalline
                symmetries to minimize the size of the "assembly"
            surrounding_uc: Whether to generate the surrounding unit cells as part of the assembly models

        Returns:
            A Model instance for each of the symmetric mates. These are transformed copies of the SymmetricModel
        """
        models = self._generate_models(minimal=minimal, surrounding_uc=surrounding_uc)
        name = self._generate_assembly_name(minimal=minimal, surrounding_uc=surrounding_uc)
        return Models(models, name=name)

    def _generate_models(
            self, minimal: bool | int | None = False, surrounding_uc: bool = False
    ) -> list[SymmetricModel]:  # Self Todo python 3.11
        """Create a group of instance copies for a specified symmetry specification

        Args:
            minimal: Whether to create the minimally contacting models. This is advantageous in crystalline
                symmetries to minimize the size of the "assembly"
            surrounding_uc: Whether to generate the surrounding unit cells as part of the models

        Returns:
            A "mate" instance for each of the specified symmetric mates. These are transformed copies of the instance
        """
        if minimal is None:
            # When the Pose is asymmetric
            coords_to_slice = self.coords
            model_indices = [0]
        else:  # Check for the surrounding_uc and minimal assembly flags
            # surrounding_uc needs to be made before get_asu_interaction_model_indices()
            if self.dimension:
                if surrounding_uc:
                    # When the surrounding_uc is requested, .symmetric_coords might need to be regenerated
                    if not self.is_surrounding_uc():
                        # The surrounding coords don't exist as the number of mates is equal to the unit cell
                        self.generate_symmetric_coords(surrounding_uc=surrounding_uc)
                    number_of_symmetry_mates = self.number_of_symmetry_mates
                else:  # Enforce number_of_uc_symmetry_mates is used
                    number_of_symmetry_mates = self.number_of_uc_symmetry_mates
            else:
                number_of_symmetry_mates = self.number_of_symmetry_mates

            if minimal:  # Only return contacting
                # Add the ASU index to the model first
                model_indices = [0] + self.get_asu_interaction_model_indices()
            else:
                model_indices = list(range(number_of_symmetry_mates))

            self.log.debug(f'Found selected models {model_indices} for assembly')
            coords_to_slice = self.symmetric_coords

        # Update all models with their coordinates
        number_of_atoms = self.number_of_atoms
        models = []
        for model_idx in model_indices:
            asu_copy = self.copy()
            asu_copy.coords = coords_to_slice[model_idx * number_of_atoms: (model_idx + 1) * number_of_atoms]
            models.append(asu_copy)

        return models

    def _generate_assembly(
            self, minimal: bool | int | None = False, surrounding_uc: bool = False
    ) -> Model:  # Self Todo python 3.11
        """Creates the Model with all chains from the SymmetricModel

        Args:
            minimal: Whether to create the minimally contacting assembly. This is advantageous in crystalline
                symmetries to minimize the size of the "assembly"
            surrounding_uc: Whether to generate the surrounding unit cells as part of the assembly

        Returns:
            A Model instance for each of the symmetric mates. These are transformed copies of the SymmetricModel
        """
        if not minimal and not surrounding_uc:
            # These are the defaults, so just use the chains attribute
            chains = self.chains
        else:
            chains = []
            models = self._generate_models(minimal=minimal, surrounding_uc=surrounding_uc)
            for model in models:
                chains.extend(model.entities)
        name = self._generate_assembly_name(minimal=minimal, surrounding_uc=surrounding_uc)

        return Model.from_chains(chains, name=name, log=self.log, entity_info=self.entity_info,
                                 cryst_record=self.cryst_record)

    @property
    def chains(self) -> list[Entity]:
        chains = self._chains
        if not chains:
            self.log.debug(f"{repr(self)} is generating .chains")
            models = self._generate_models()
            for model in models:
                chains.extend(model.entities)

        return chains

    @property
    def oligomeric_model_indices(self) -> dict[Entity, list[int]] | dict:
        try:
            return self._oligomeric_model_indices
        except AttributeError:
            self._oligomeric_model_indices = {}
            self._find_oligomeric_model_indices()
            return self._oligomeric_model_indices

    def _find_oligomeric_model_indices(self, epsilon: float = 0.5):
        """From an Entity's Chain members, find the SymmetricModel equivalent models using Chain center or mass
        compared to the symmetric model center of mass

        Args:
            epsilon: The distance measurement tolerance to find similar symmetric models to the oligomer
        """
        # number_of_atoms = self.number_of_atoms
        for entity, entity_symmetric_centers_of_mass in zip(self.entities, self.center_of_mass_symmetric_entities):
            if not entity.is_symmetric():
                self._oligomeric_model_indices[entity] = []
                continue
            # Need to slice through the specific Entity coords once we have the model
            # entity_indices = entity.atom_indices
            # entity_start, entity_end = entity_indices[0], entity_indices[-1]
            # entity_length = entity.number_of_atoms
            # entity_center_of_mass_divisor = np.full(entity_length, 1 / entity_length)
            equivalent_models = []
            for chain_idx, chain in enumerate(entity.chains):
                chain_center_of_mass = chain.center_of_mass
                for model_idx, sym_model_center_of_mass in enumerate(entity_symmetric_centers_of_mass):
                    # sym_model_center_of_mass = \
                    #     np.matmul(entity_center_of_mass_divisor,
                    #               self.symmetric_coords[model_idx*number_of_atoms + entity_start:
                    #                                     model_idx*number_of_atoms + entity_end + 1])
                    # #                                             have to add 1 for slice ^
                    com_norm = np.linalg.norm(chain_center_of_mass - sym_model_center_of_mass)
                    if com_norm < epsilon:
                        equivalent_models.append(model_idx)
                        self.log.debug(f'Entity.chain{chain_idx}/Sym-model{model_idx} center of mass overlap')
                        break
                    else:
                        self.log.debug(f'Entity.chain{chain_idx}/Sym-model{model_idx} center of mass distance: '
                                       f'{np.format_float_positional(com_norm, precision=2)}\n'
                                       # f'{np.array_str(com_norm, precision=2)}\n'
                                       f'\tEntity.chain com: {np.array_str(chain_center_of_mass, precision=2)} | '
                                       f'Sym-model com: {np.array_str(sym_model_center_of_mass, precision=2)}')

            if len(equivalent_models) != entity.number_of_chains:
                raise SymmetryError(
                    f'For Entity {entity.name}, the number of symmetry mates, {entity.number_of_chains} != '
                    f'{len(equivalent_models)}, the number of {self.__class__.__name__} equivalent symmetric models')

            self._oligomeric_model_indices[entity] = equivalent_models
            self.log.info(f'Masking {entity.name} coordinates from models '
                          f'{",".join(map(str, equivalent_models))} due to specified oligomer')

        # number_of_atoms = self.number_of_atoms
        # for entity in zip(self.entities):
        #     if not entity.is_symmetric():
        #         self._oligomeric_model_indices[entity] = []
        #         continue
        #
        #     # Prepare slice variables for Entity
        #     entity_indices = entity.atom_indices
        #     entity_start, entity_end = entity_indices[0], entity_indices[-1]
        #     entity_length = entity.number_of_atoms
        #     entity_center_of_mass_divisor = np.full(entity_length, 1 / entity_length)
        #     equivalent_models = []
        #     for chain in entity.chains:
        #         chain_center_of_mass = chain.center_of_mass
        #         # Slice through the symmetric coords for that coordinate representation of the Entity
        #         for model_idx in range(self.number_symmetry_mates):
        #             sym_model_center_of_mass = \
        #                 np.matmul(entity_center_of_mass_divisor,
        #                           self.symmetric_coords[model_idx*number_of_atoms + entity_start:
        #                                                 1 + model_idx*number_of_atoms + entity_end])
        #             #                                             have to add 1 for slice ^

    def get_asu_interaction_model_indices(self, calculate_contacts: bool = True, distance: float = 8., **kwargs) -> \
            list[int]:
        """From an ASU, find the symmetric models that immediately surround the ASU

        Args:
            calculate_contacts: Whether to calculate interacting models by atomic contacts
            distance: When calculate_contacts is True, the CB distance which nearby symmetric models should be found
                When calculate_contacts is False, uses the ASU radius plus the maximum Entity radius

        Returns:
            The indices of the models that contact the asu
        """
        if calculate_contacts:
            # DEBUG self.report_symmetric_coords(self.get_asu_interaction_model_indices.__name__)
            asu_query = self.assembly_tree.query_radius(self.coords[self.backbone_and_cb_indices], distance)
            # Combine each subarray of the asu_query and divide by the assembly_tree interval length -> len(asu_query)
            interacting_models = (
                                         np.array(list({asu_idx for asu_contacts in asu_query.tolist()
                                                        for asu_idx in asu_contacts.tolist()})
                                                  )
                                         // len(asu_query)
                                 ) + 1
            # The asu is missing from assembly_tree so add 1 to get the
            # correct model indices ^
            interacting_models = np.unique(interacting_models).tolist()
        else:
            # The furthest point from the asu COM + the max individual Entity radius
            distance = self.radius + max([entity.radius for entity in self.entities])
            self.log.debug(f'For ASU neighbor query, using the distance={distance}')
            center_of_mass = self.center_of_mass
            interacting_models = [idx for idx, sym_model_com in enumerate(self.center_of_mass_symmetric_models)
                                  if np.linalg.norm(center_of_mass - sym_model_com) <= distance]
            # Remove the first index from the interacting_models due to exclusion of asu from convention above
            interacting_models.pop(0)

        return interacting_models

    def get_asu_atom_indices(self, as_slice: bool = False) -> list[int] | slice:
        """Find the coordinate indices of the asu equivalent model in the SymmetricModel. Zero-indexed

        Returns:
            The indices in the SymmetricModel where the ASU is also located
        """
        asu_model_idx = self.asu_model_index
        number_of_atoms = self.number_of_atoms
        start_idx = number_of_atoms * asu_model_idx
        end_idx = number_of_atoms * (asu_model_idx + 1)

        if as_slice:
            return slice(start_idx, end_idx)
        else:
            return list(range(start_idx, end_idx))

    def get_oligomeric_atom_indices(self, entity: Entity) -> list[int]:
        """Find the coordinate indices of the intra-oligomeric equivalent models in the SymmetricModel. Zero-indexed

        Args:
            entity: The Entity with oligomeric chains to query for corresponding symmetry mates

        Returns:
            The indices in the SymmetricModel where the intra-oligomeric contacts are located
        """
        number_of_atoms = self.number_of_atoms
        oligomeric_atom_indices = []
        for model_number in self.oligomeric_model_indices.get(entity):
            oligomeric_atom_indices.extend(range(number_of_atoms * model_number,
                                                 number_of_atoms * (model_number + 1)))
        return oligomeric_atom_indices

    def get_asu_interaction_indices(self, **kwargs) -> list[int]:
        """Find the coordinate indices for the models in the SymmetricModel interacting with the asu. Zero-indexed

        Keyword Args:
            calculate_contacts: bool = True - Whether to calculate interacting models by atomic contacts
            distance: float = 8.0 - When calculate_contacts is True, the CB distance which nearby symmetric models
                 should be found. When calculate_contacts is False, uses the ASU radius plus the maximum Entity radius

        Returns:
            The indices in the SymmetricModel where the asu contacts other models
        """
        number_of_atoms = self.number_of_atoms
        interacting_indices = []
        for model_number in self.get_asu_interaction_model_indices(**kwargs):
            interacting_indices.extend(range(number_of_atoms * model_number,
                                             number_of_atoms * (model_number+1)))

        return interacting_indices

    @property
    def entity_transformations(self) -> list[types.TransformationMapping] | list:
        """The transformation parameters for each Entity in the SymmetricModel. Each entry has the
        TransformationMapping type
        """
        try:
            return self._transformation
        except AttributeError:
            self._transformation = self._assign_pose_transformation()
            return self._transformation
    #
    # @entity_transformations.setter
    # def entity_transformations(self, transformations: list[types.TransformationMapping]):
    #     self._transformation = transformations

    def _assign_pose_transformation(self) -> list[types.TransformationMapping] | list[dict]:
        """Using the symmetry entry and symmetric material, find the specific transformations necessary to establish the
        individual symmetric components in the global symmetry

        Sets:
            self.coords (np.ndarray) - To the most contacting asymmetric unit

        Returns:
            The entity_transformations dictionaries that places each Entity with a proper symmetry axis in the Pose
        """
        if not self.is_symmetric():
            raise SymmetryError(
                f'Must set a global symmetry to {self._assign_pose_transformation.__name__}')
        elif self.sym_entry.is_token():
            return [{} for entity in self.entities]

        self.log.debug(f'Searching for transformation parameters for the Pose {self.name}')

        # Get optimal external translation
        if self.dimension == 0:
            external_tx = [None for _ in self.sym_entry.groups]
        else:
            try:
                optimal_external_shifts = self.sym_entry.get_optimal_shift_from_uc_dimensions(*self.uc_dimensions)
            except AttributeError as error:
                self.log.error(f"\n\n\n{self._assign_pose_transformation.__name__}: Couldn't "
                               f'{utils.SymEntry.SymEntry.get_optimal_shift_from_uc_dimensions.__name__} with '
                               f'dimensions: {self.uc_dimensions}\nAnd sym_entry.unit_cell specification: '
                               f"{self.sym_entry.unit_cell}\nThis is likely because {self.symmetry} isn't a lattice "
                               "with parameterized external translations\n\n\n")
                raise error
            # external_tx1 = optimal_external_shifts[:, None] * self.sym_entry.external_dof1
            # external_tx2 = optimal_external_shifts[:, None] * self.sym_entry.external_dof2
            # external_tx = [external_tx1, external_tx2]
            self.log.warning('3D lattice local symmetry identification is under active development. '
                             'All outputs with crystalline symmetry should be inspected for '
                             'proper spatial parameters before proceeding')
            # Assuming that the point group symmetry of the lattice lies at the origin, no subtraction of external_tx
            # is required to solve...
            # Instead of using this, using the below to account for None situations
            # external_tx = \
            #     [(optimal_external_shifts[:, None] * getattr(self.sym_entry, f'external_dof{idx}')).sum(axis=-2)
            #      for idx, group in enumerate(self.sym_entry.groups, 1)]
            external_tx = []
            for idx, group in enumerate(self.sym_entry.groups, 1):
                if group == self.symmetry:
                    # The molecule should be oriented already and expand matrices handle oligomers
                    external_tx.append(None)
                elif group == 'C1':
                    # No oligomer possible
                    external_tx.append(None)
                else:
                    external_tx.append(
                        (optimal_external_shifts[:, None] * getattr(self.sym_entry, f'external_dof{idx}')).sum(axis=-2)
                    )
            # Given the lowest symmetry crystal system with 2 symmetric components possible is hexagonal,
            # by convention the length of C is placed on the z-axis. If there are systems such as rhobohedral, then
            # this wouldn't necessarily be correct. I believe according to the second setting convention,
            # b needs to go on z
            a, b, c, *angles = self.uc_dimensions
            a_half_length_coord = [a / 2, 0., 0.]
            b_half_length_coord = [0., b / 2, 0.]
            c_half_length_coord = [0., 0., c / 2]
            half_cell_dimensions_array = np.array([a_half_length_coord, b_half_length_coord, c_half_length_coord])

        # Solve for the transform solution for each symmetry group and the indices where the asymmetric unit
        entity_model_indices: list[int] | None
        internal_tx: np.ndarray | None
        setting_matrix: np.ndarray | None
        setting_matrices = utils.symmetry.setting_matrices
        inv_setting_matrices = utils.symmetry.inv_setting_matrices

        def set_solution():
            """Set the values for entity_model_indices, internal_tx, setting_matrix"""
            nonlocal entity_model_indices, internal_tx, setting_matrix
            entity_model_indices = possible_height_groups[centrally_disposed_group_height]
            internal_tx = temp_model_coms[entity_model_indices].mean(axis=-2)
            setting_matrix = setting_matrices[setting_matrix_idx]

        dimension = self.dimension
        symmetry = self.symmetry
        sym_entry = self.sym_entry
        center_of_mass_symmetric_entities = self.center_of_mass_symmetric_entities
        number_of_symmetry_mates = self.number_of_symmetry_mates
        point_group_symmetry = self.point_group_symmetry
        point_group_subsym_setting_matrices = utils.SymEntry.point_group_setting_matrix_members[point_group_symmetry]
        dihedral = False
        transform_solutions = []
        oligomeric_indices_groups: list[list[int]] = []
        """Contains Collection of indices that specify the model indices where the asymmetric unit could exist"""
        for group_idx, (entity_symmetric_centers_of_mass, sym_group) in \
                enumerate(zip(center_of_mass_symmetric_entities, sym_entry.groups)):
            # Find groups for which the oligomeric parameters do not apply or exist by nature of orientation [T, O, I]
            # Add every symmetric index to oligomeric_indices_groups
            if sym_group == symmetry:
                # The molecule should be oriented already and expand matrices handle oligomers
                transform_solutions.append(dict())  # rotation=rot, translation=tx
                oligomeric_indices_groups.append(list(range(number_of_symmetry_mates)))
                continue
            elif sym_group == 'C1':
                # No oligomer possible
                transform_solutions.append(dict())  # rotation=rot, translation=tx
                oligomeric_indices_groups.append(list(range(number_of_symmetry_mates)))
                continue
            elif sym_group[0] == 'D':
                dihedral = True
            # else:  # sym_group requires us to solve
            #     pass

            # Search through the sub_symmetry group setting matrices that make up the resulting point group symmetry
            # Apply setting matrix to the entity centers of mass indexed to the proper group number
            entity_model_indices = internal_tx = setting_matrix = None
            group_subunit_number = utils.symmetry.valid_subunit_number[sym_group]
            current_best_minimal_central_offset = float('inf')
            for setting_matrix_idx in point_group_subsym_setting_matrices.get(sym_group, []):
                # self.log.critical('Setting_matrix_idx = %d' % setting_matrix_idx)
                # Find groups of COMs with equal z heights
                possible_height_groups: dict[float, list[int]] = defaultdict(list)
                if dimension == 0:
                    temp_model_coms = np.matmul(entity_symmetric_centers_of_mass,
                                                np.transpose(inv_setting_matrices[setting_matrix_idx]))
                    # Rounding to 2 decimals may be required precision
                    for idx, com in enumerate(temp_model_coms.round(decimals=2).tolist()):
                        # z_coord = com[-1]
                        possible_height_groups[com[-1]].append(idx)
                else:
                    tp_inv_set_matrix = np.transpose(inv_setting_matrices[setting_matrix_idx])
                    temp_model_coms = np.matmul(entity_symmetric_centers_of_mass, tp_inv_set_matrix)
                    inverse_set_half_cell_dimension = np.matmul(half_cell_dimensions_array, tp_inv_set_matrix)
                    x_max, y_max, z_max = np.abs(inverse_set_half_cell_dimension).max(axis=0)
                    # Compare the absolute value of the transformed center of mass to unit cell parameters to only
                    # select COM positions that belong in the central unit cells along the Z-axis which is set
                    # equivalent to the c length of the cell
                    for idx, (abs_com, z_com) in enumerate(zip(np.abs(temp_model_coms).tolist(),
                                                               temp_model_coms.round(decimals=2)[:, -1].tolist())):
                        x_com, y_com, _ = abs_com
                        if x_com > x_max or y_com > y_max:
                            continue
                        possible_height_groups[z_com].append(idx)

                # For those height groups with the correct number of COMs in the group, find the most centrally
                # disposed, COM grouping where the COM are all most closely situated to the z-axis
                # This isn't necessarily positive, but we select the positive choice if there are equivalent
                # def get_group_offset(model_indices) -> float:
                #     # Get the first point from the group. Norms are equivalent as all are same height
                #     com_point = (temp_model_coms[model_indices] - [0., 0., z_height])[0]
                #     return np.sqrt(com_point.dot(com_point))  # np.abs()

                centrally_disposed_group_height = None
                additional_group_heights = []
                minimal_central_offset = float('inf')
                for z_height, model_indices in possible_height_groups.items():
                    if len(model_indices) == group_subunit_number:
                        # central_offset = get_group_offset(model_indices)
                        com_point = (temp_model_coms[model_indices] - [0., 0., z_height])[0]
                        central_offset = np.sqrt(com_point.dot(com_point)).round(decimals=3)
                        # self.log.debug(f'central_offset = {central_offset}')
                        if central_offset < minimal_central_offset:
                            minimal_central_offset = central_offset
                            centrally_disposed_group_height = z_height
                            self.log.debug(f'NEW group-{group_idx}, setting-matrix{setting_matrix_idx} '
                                           f'centrally_disposed_group_height = {centrally_disposed_group_height} at '
                                           f'central_offset = {central_offset}')
                        elif central_offset == minimal_central_offset:
                            if centrally_disposed_group_height < 0 < z_height or \
                                    centrally_disposed_group_height > z_height > 0:
                                # Select the most central group as the centrally disposed_group
                                centrally_disposed_group_height = z_height
                                self.log.debug(f'EQUAL group-{group_idx}, setting-matrix{setting_matrix_idx} '
                                               f'centrally_disposed_group_height = {centrally_disposed_group_height}'
                                               f' at central_offset = {central_offset}')
                        else:  # The central offset is larger
                            pass
                    elif dihedral and len(model_indices) == group_subunit_number / 2:
                        # This requires two central offsets are found, each with the same offset
                        com_point = (temp_model_coms[model_indices] - [0., 0., z_height])[0]
                        central_offset = np.sqrt(com_point.dot(com_point)).round(decimals=3)
                        # self.log.debug(f'central_offset = {central_offset}')
                        if central_offset < minimal_central_offset:
                            minimal_central_offset = central_offset
                            centrally_disposed_group_height = z_height
                            additional_group_heights = []  # [z_height]
                            self.log.debug(f'NEW group-{group_idx}, setting-matrix{setting_matrix_idx} '
                                           f'centrally_disposed_group_height = {centrally_disposed_group_height} at '
                                           f'central_offset = {central_offset}')
                        elif central_offset == minimal_central_offset:

                            # This is the same distance as previously located.
                            self.log.debug(f'EQUAL group-{group_idx}, setting-matrix{setting_matrix_idx} '
                                           f'centrally_disposed_group_height = {centrally_disposed_group_height}'
                                           f' at central_offset = {central_offset}')
                            additional_group_heights.append(z_height)
                        else:  # The central offset is larger
                            pass

                # Solve any equivalent groups as a result of dihedral symmetry
                best_additional = None
                minimal_difference = float('inf')
                for additional_group_height in additional_group_heights:
                    difference = abs(centrally_disposed_group_height - additional_group_height)
                    if difference < minimal_difference:
                        best_additional = additional_group_height

                if best_additional:
                    prior_group_height_indices = possible_height_groups[centrally_disposed_group_height]
                    centrally_disposed_group_height = (centrally_disposed_group_height+best_additional) / 2
                    self.log.debug(f'Dihedral centrally_disposed_group_height resulting from two collections of center'
                                   f' of mass sets with length {int(group_subunit_number / 2)} is equal to '
                                   f'{centrally_disposed_group_height}')
                    possible_height_groups[centrally_disposed_group_height] = \
                        prior_group_height_indices + possible_height_groups[best_additional]

                # If a viable group was found, save the group COM as an internal_tx and setting_matrix used to find it
                if centrally_disposed_group_height is not None:
                    if setting_matrix is None and internal_tx is None:
                        # These were not set yet
                        set_solution()
                        current_best_minimal_central_offset = minimal_central_offset
                    else:  # There is an alternative solution. Is it better? Or is it a degeneracy?
                        if minimal_central_offset < current_best_minimal_central_offset:
                            # The new one if it is less offset
                            set_solution()
                            current_best_minimal_central_offset = minimal_central_offset
                        elif minimal_central_offset == current_best_minimal_central_offset:
                            # Chose the positive one in the case that there are degeneracies (most likely)
                            self.log.debug('There are multiple pose transformation solutions for the symmetry group '
                                           f'{sym_group} (specified in position {group_idx + 1} of '
                                           f'{sym_entry.specification}). The solution with a positive translation '
                                           'was chosen by convention. This may result in inaccurate behavior')
                            # internal_tx will have been set already, check the z-axis value
                            internal_tx: np.ndarray
                            if internal_tx[-1] < 0 < centrally_disposed_group_height:
                                set_solution()
                #         else:  # The central offset is larger
                #             pass
                else:  # No viable group probably because the setting matrix was wrong. Continue with next
                    self.log.debug(f'No centrally_disposed_group_height found from the possible_height_groups\n\t:'
                                   f'{possible_height_groups}')

            if entity_model_indices is None:  # No solution
                raise SymmetryError(
                    f'For {repr(self)} with symmetry specified as {repr(sym_entry)}, there was no solution found for '
                    f'group #{group_idx + 1}->{sym_group}. If the Entity order is different than the specification, '
                    'please supply the correct order using the symmetry specification with format '
                    f"'{utils.SymEntry.symmetry_combination_format}' to the 'symmetry' argument. Another possibility "
                    f"is that the {self.__class__.__name__}.symmetry operations were generated improperly/imprecisely. "
                    f"Please ensure your input is symmetrically viable and if not, '.orient(symmetry={symmetry})'")
            else:
                if getattr(self.sym_entry, f'is_internal_tx{group_idx + 1}'):
                    translation = internal_tx
                else:
                    # self.log.debug('Group has NO internal_dof')
                    translation = None

                transform_solutions.append(
                    dict(translation=translation, rotation2=setting_matrix, translation2=external_tx[group_idx]))
                oligomeric_indices_groups.append(entity_model_indices)


        # def set_contacting_asu(from_com: bool = True):
        #     asu_coords = [symmetric_coords_split_by_entity[sym_idx]
        #                   for symmetric_coords_split_by_entity, sym_idx in
        #                   zip(self.symmetric_coords_split_by_entity, selected_asu_model_indices)]
        #     # self.log.critical('asu_coords: %s' % asu_coords)
        #     self.coords = np.concatenate(asu_coords)
        #     # self.asu_coords = Coords(np.concatenate(asu_coords))
        #     # for idx, entity in enumerate(self.entities):
        #     #     entity.make_oligomer(symmetry=self.sym_entry.groups[idx], **transform_solutions[idx])
        #
        # def find_minimal_com():
        """This routine uses the same logic as find_contacting_asu(), however, using the COM of the found
        transform_solutions to find the ASU entities. These COM are then used to make collection of entities assume a 
        globular nature. Therefore, the minimal com to com distance is our naive ASU coords
        """

        # The only input here is the oligomeric_indices_groups
        number_of_symmetry_groups = len(oligomeric_indices_groups)
        if number_of_symmetry_groups == 1:
            # We only have one Entity. Choice doesn't matter, grab the first
            selected_asu_model_indices: list[int] = [oligomeric_indices_groups[0][0]]
        else:
            oligomeric_com_groups = [[entity_symmetric_centers_of_mass[idx] for idx in indices]
                                     for entity_symmetric_centers_of_mass, indices in
                                     zip(center_of_mass_symmetric_entities, oligomeric_indices_groups)]
            # self.log.critical('oligomeric_com_groups: %s' % oligomeric_com_groups)

            offset_count = count()
            asu_indices_combinations = []
            entity_idx_pairs, asu_coms_index = [], []
            com_offsets = np.full(sum(map(math.prod,
                                          combinations((len(indices) for indices in oligomeric_indices_groups), 2))),
                                  np.inf)
            symmetric_group_indices = range(number_of_symmetry_groups)
            # Find the shortest distance between two center of mass points for any of the possible symmetric COMs
            for entity_idx1, entity_idx2 in combinations(symmetric_group_indices, 2):
                # for index1 in oligomeric_indices_groups[idx1]:
                for com_idx1, com1 in enumerate(oligomeric_com_groups[entity_idx1]):
                    for com_idx2, com2 in enumerate(oligomeric_com_groups[entity_idx2]):
                        asu_indices_combinations.append((entity_idx1, com_idx1, entity_idx2, com_idx2))
                        entity_idx_pairs.append((entity_idx1, entity_idx2))
                        asu_coms_index.append((com_idx1, com_idx2))
                        dist = com2 - com1
                        com_offsets[next(offset_count)] = np.sqrt(dist.dot(dist))

            # self.log.critical('com_offsets: %s' % com_offsets)
            minimal_com_distance_index = com_offsets.argmin()
            entity_idx1, com_idx1, entity_idx2, com_idx2 = asu_indices_combinations[minimal_com_distance_index]
            core_indices = [(entity_idx1, com_idx1), (entity_idx2, com_idx2)]
            minimal_entity_com_pair = (entity_idx1, entity_idx2)

            # Find any additional index pairs
            additional_indices: list[tuple[int, int]] = []
            if number_of_symmetry_groups != 2:  # We have to find more indices
                # Find indices where either of the minimally distanced COMs are utilized
                possible_entity_idx_pairs = {idx for idx, ent_idx_pair in enumerate(entity_idx_pairs)
                                             if entity_idx1 in ent_idx_pair or entity_idx2 in ent_idx_pair and
                                             ent_idx_pair != minimal_entity_com_pair}
                remaining_indices = set(symmetric_group_indices).difference(minimal_entity_com_pair)
                for additional_entity_idx in remaining_indices:
                    # Find the indices where the missing index is utilized
                    remaining_index_indices = {idx for idx, ent_idx_pair in enumerate(entity_idx_pairs)
                                               if additional_entity_idx in ent_idx_pair}
                    # Only use those where found asu indices already occur
                    viable_remaining_indices = list(remaining_index_indices.intersection(possible_entity_idx_pairs))
                    next_min_com_dist_idx = com_offsets[viable_remaining_indices].argmin()
                    new_ent_pair_idx = viable_remaining_indices[next_min_com_dist_idx]
                    for entity_idx, entity_com in zip(entity_idx_pairs[new_ent_pair_idx],
                                                      asu_coms_index[new_ent_pair_idx]):
                        if entity_idx == additional_entity_idx:
                            additional_indices.append((entity_idx, entity_com))
            new_asu_entity_com_pairs: list[tuple[int, int]] = core_indices + additional_indices

            selected_asu_model_indices = []
            for entity_idx in symmetric_group_indices:
                for possible_entity_idx, com_idx in new_asu_entity_com_pairs:
                    if entity_idx == possible_entity_idx:
                        selected_asu_model_indices.append(oligomeric_indices_groups[possible_entity_idx][com_idx])

        asu_coords = [symmetric_coords_split_by_entity[sym_idx]
                      for symmetric_coords_split_by_entity, sym_idx in
                      zip(self.symmetric_coords_split_by_entity, selected_asu_model_indices)]
        # self.log.critical('asu_coords: %s' % asu_coords)
        self.coords = np.concatenate(asu_coords)
        # self.asu_coords = Coords(np.concatenate(asu_coords))
        # for idx, entity in enumerate(self.entities):
        #     entity.make_oligomer(symmetry=self.sym_entry.groups[idx], **transform_solutions[idx])

        return transform_solutions

    def make_oligomers(self, transformations: list[types.TransformationMapping] = None):
        """Generate oligomers for each Entity in the SymmetricModel

        Args:
            transformations: The entity_transformations operations that reproduce the individual oligomers
        """
        self.log.debug(f'Initializing oligomeric symmetry')
        if transformations is None or not all(transformations):
            # If this fails then the symmetry is failed... It should never return an empty list as
            # .entity_transformations -> ._assign_pose_transformation() will raise SymmetryError
            transformations = self.entity_transformations

        for entity, subunit_number, symmetry, transformation in zip(
                self.entities, self.sym_entry.group_subunit_numbers, self.sym_entry.groups, transformations):
            if entity.number_of_symmetry_mates != subunit_number:
                entity.make_oligomer(symmetry=symmetry, **transformation)
            else:
                self.log.debug(f'{repr(entity)} is already the correct oligomer, skipping make_oligomer()')

    def symmetric_assembly_is_clash(self, measure: coords_type_literal = default_clash_criteria,
                                    distance: float = default_clash_distance, warn: bool = False) -> bool:
        """Returns True if the SymmetricModel presents any clashes at the specified distance

        Args:
            measure: The atom type to measure clashing by
            distance: The distance which clashes should be checked
            warn: Whether to emit warnings about identified clashes

        Returns:
            True if the symmetric assembly clashes with the asu, False otherwise
        """
        if not self.is_symmetric():
            self.log.warning("Can't check if the assembly is clashing as it has no symmetry")
            return False
        indices = self.__getattribute__(f'{measure}_indices')
        clashes = self.assembly_tree.two_point_correlation(self.coords[indices], [distance])
        if clashes[0] > 0:
            if warn:
                self.log.warning(
                    f"{self.name}: Found {clashes[0]} clashing sites. Pose isn't a viable symmetric assembly")
            return True  # Clash
        else:
            return False  # No clash

    @property
    def assembly_tree(self) -> BinaryTreeType:
        """Holds the tree structure of the backbone and cb symmetric_coords not including the asu coords"""
        try:
            return self._assembly_tree
        except AttributeError:
            # Select coords of interest from the model coords
            measure = 'backbone_and_cb'
            asu_bb_cb_indices = getattr(self, f'{measure}_indices')
            # All the indices from ASU must be multiplied by self.number_of_symmetry_mates to get all symmetric coords
            self._assembly_tree = \
                BallTree(self.symmetric_coords[
                             self.make_indices_symmetric(asu_bb_cb_indices, dtype='atom')[len(asu_bb_cb_indices):]])
            # Last, we take out those indices that are inclusive of the model_asu_indices like below
            return self._assembly_tree

    def orient(self, symmetry: str = None):
        if self.is_symmetric():
            super().orient(symmetry=self.symmetry)
        else:
            super().orient(symmetry=symmetry)
            self.set_symmetry(symmetry=symmetry)

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, header: str = None,
              assembly: bool = False, **kwargs) -> AnyStr | None:
        """Write SymmetricModel Atoms to a file specified by out_path or with a passed file_handle

        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            header: A string that is desired at the top of the file
            assembly: Whether to write the full assembly. Default writes only the ASU

        Keyword Args:
            increment_chains: bool = False - Whether to write each Structure with a new chain name, otherwise write as
                a new Model
            surrounding_uc: bool = False - Whether the 3x3 layer group, or 3x3x3 space group should be written when
                assembly is True and self.dimension > 1

        Returns:
            The name of the written file if out_path is used
        """
        self.log.debug(f'{SymmetricModel.__name__} is writing {repr(self)}')
        is_symmetric = self.is_symmetric()

        def _write(handle) -> None:
            if is_symmetric:
                if assembly:
                    models = self._generate_assembly_models(**kwargs)
                    models.write(file_handle=handle, **kwargs)
                else:  # Skip all models, write asu. Use biomt_record/cryst_record for symmetry
                    for entity in self.entities:
                        entity.write(file_handle=handle, **kwargs)
            else:  # Finish with a standard write
                super(Structure, Structure).write(self, file_handle=handle, **kwargs)

        if file_handle:
            return _write(file_handle)
        else:  # out_path default argument is current working directory
            # Write the header as an asu if no assembly requested or not symmetric
            assembly_header = (is_symmetric and assembly) or not is_symmetric
            _header = self.format_header(assembly=assembly_header, **kwargs)
            if header is not None:
                if not isinstance(header, str):
                    header = str(header)
                _header += (header if header[-2:] == '\n' else f'{header}\n')

            with open(out_path, 'w') as outfile:
                outfile.write(_header)
                _write(outfile)
            return out_path

    def __copy__(self) -> SymmetricModel:  # Todo -> Self: in python 3.11
        # self.log.debug('In SymmetricModel copy {repr(self)}')

        # Save, then remove the _assembly attributes
        try:
            _assembly = self._assembly
        except AttributeError:
            _assembly = None
        else:
            del self._assembly

        try:
            _assembly_minimally_contacting = self._assembly_minimally_contacting
        except AttributeError:
            _assembly_minimally_contacting = None
        else:
            del self._assembly_minimally_contacting

        other: SymmetricModel = super().__copy__()

        # Reset the self state as before the copy
        if _assembly_minimally_contacting is not None:
            self._assembly_minimally_contacting = _assembly_minimally_contacting
        if _assembly is not None:
            self._assembly = _assembly

        return other

    copy = __copy__


StructureSpecification = dict[str, Union[set[int], set[str], set, None]]


class PoseSpecification(TypedDict):
    mask: StructureSpecification
    required: StructureSpecification
    selection: StructureSpecification


class Pose(SymmetricModel, MetricsMixin):
    """A Pose is made of single or multiple Structure objects such as Entities, Chains, or other structures.
    All objects share a common feature such as the same symmetric system or the same general atom configuration in
    separate models across the Structure or sequence.
    """
    _interface_fragment_residue_indices: list[int]
    _design_residues: list[Residue]
    # Metrics class attributes
    # _df: pd.Series  # Metrics
    # _metrics: sql.PoseMetrics  # Metrics
    _metrics_table = sql.PoseMetrics
    _interface_neighbor_residues: list[Residue]
    _interface_residues: list[Residue]
    _design_selection_entity_names: set[str]
    _design_selector_atom_indices: set[int]
    _fragment_info_by_entity_pair: dict[tuple[str, str], list[FragmentInfo] | list]
    _interface_residue_indices_by_entity_name_pair: dict[tuple[str, str], tuple[list[int], list[int]]]
    _required_atom_indices: list[int]
    _interface_residue_indices_by_interface: dict[int, list[int]]
    _interface_residue_indices_by_interface_unique: dict[int, list[int]]
    split_interface_ss_elements: dict[int, list[int]]
    """Stores the interface number mapped to an index corresponding to the secondary structure type 
    Ex: {1: [0, 0, 1, 2, ...] , 2: [9, 9, 9, 13, ...]]}
    """
    ss_sequence_indices: list[int]
    """Index which indicates the Residue membership to the secondary structure type element sequence"""
    ss_type_sequence: list[str]
    """The ordered secondary structure type sequence which contains one character/secondary structure element"""
    # state_attributes = SymmetricModel.state_attributes | \
    #     {'ss_sequence_indices', 'ss_type_sequence',  # These should be .clear()
    #      'fragment_metrics', '_fragment_queries',
    #      'interface_design_residue_numbers', 'interface_residue_numbers', 'split_interface_ss_elements'
    #      # The below rely on indexing or getting during @property calls
    #      '_fragment_info_by_entity_pair', '_interface_residue_indices_by_entity_name_pair',
    #      '_interface_residue_indices_by_interface', '_interface_residue_indices_by_interface_unique',
    #      }

    def __init__(self, **kwargs):
        """Construct the instance

        Args:
            **kwargs:
        """
        super().__init__(**kwargs)  # Pose
        self._design_selection_entity_names = {entity.name for entity in self.entities}
        self._design_selector_atom_indices = set(self._atom_indices)
        self._required_atom_indices = []
        self._interface_residue_indices_by_entity_name_pair = {}
        self._interface_residue_indices_by_interface = {}
        self._interface_residue_indices_by_interface_unique = {}
        self._fragment_info_by_entity_pair = {}
        self.split_interface_ss_elements = {}
        self.ss_sequence_indices = []
        self.ss_type_sequence = []

    def calculate_metrics(self, **kwargs) -> dict[str, Any]:
        """Calculate metrics for the instance

        Returns:
            {
                'entity_max_radius_average_deviation',
                'entity_min_radius_average_deviation',
                'entity_radius_average_deviation',
                'interface_b_factor',
                'interface1_secondary_structure_fragment_topology',
                'interface1_secondary_structure_fragment_count',
                'interface1_secondary_structure_topology',
                'interface1_secondary_structure_count',
                'interface2_secondary_structure_fragment_topology',
                'interface2_secondary_structure_fragment_count',
                'interface2_secondary_structure_topology',
                'interface2_secondary_structure_count',
                'maximum_radius',
                'minimum_radius',
                'multiple_fragment_ratio',
                'nanohedra_score_normalized',
                'nanohedra_score_center_normalized',
                'nanohedra_score',
                'nanohedra_score_center',
                'number_residues_interface_fragment_total',
                'number_residues_interface_fragment_center',
                'number_fragments_interface',
                'number_residues_interface',
                'number_residues_interface_non_fragment',
                'percent_fragment_helix',
                'percent_fragment_strand',
                'percent_fragment_coil',
                'percent_residues_fragment_interface_total',
                'percent_residues_fragment_interface_center',
                'percent_residues_non_fragment_interface',
                'pose_length',
                'symmetric_interface'
            }
        """
        minimum_radius, maximum_radius = float('inf'), 0.
        entity_metrics = []
        reference_com = self.center_of_mass_symmetric
        for entity in self.entities:
            # _entity_metrics = entity.calculate_metrics()
            _entity_metrics = entity.calculate_spatial_orientation_metrics(reference=reference_com)

            if _entity_metrics['min_radius'] < minimum_radius:
                minimum_radius = _entity_metrics['min_radius']
            if _entity_metrics['max_radius'] > maximum_radius:
                maximum_radius = _entity_metrics['max_radius']
            entity_metrics.append(_entity_metrics)

        pose_metrics = {'minimum_radius': minimum_radius,
                        'maximum_radius': maximum_radius,
                        'pose_length': self.number_of_residues
                        # 'sequence': self.sequence
                        }
        radius_ratio_sum = min_ratio_sum = max_ratio_sum = 0.  # residue_ratio_sum
        counter = 1
        # index_combinations = combinations(range(1, 1 + len(entity_metrics)), 2)
        for counter, (metrics1, metrics2) in enumerate(combinations(entity_metrics, 2), counter):
            if metrics1['radius'] > metrics2['radius']:
                radius_ratio = metrics2['radius'] / metrics1['radius']
            else:
                radius_ratio = metrics1['radius'] / metrics2['radius']

            if metrics1['min_radius'] > metrics2['min_radius']:
                min_ratio = metrics2['min_radius'] / metrics1['min_radius']
            else:
                min_ratio = metrics1['min_radius'] / metrics2['min_radius']

            if metrics1['max_radius'] > metrics2['max_radius']:
                max_ratio = metrics2['max_radius'] / metrics1['max_radius']
            else:
                max_ratio = metrics1['max_radius'] / metrics2['max_radius']

            radius_ratio_sum += 1 - radius_ratio
            min_ratio_sum += 1 - min_ratio
            max_ratio_sum += 1 - max_ratio
            # residue_ratio_sum += abs(1 - residue_ratio)
            # entity_idx1, entity_idx2 = next(index_combinations)
            # pose_metrics.update({f'entity_radius_ratio_{entity_idx1}v{entity_idx2}': radius_ratio,
            #                      f'entity_min_radius_ratio_{entity_idx1}v{entity_idx2}': min_ratio,
            #                      f'entity_max_radius_ratio_{entity_idx1}v{entity_idx2}': max_ratio,
            #                      f'entity_number_of_residues_ratio_{entity_idx1}v{entity_idx2}': residue_ratio})

        pose_metrics.update({'entity_radius_average_deviation': radius_ratio_sum / counter,
                             'entity_min_radius_average_deviation': min_ratio_sum / counter,
                             'entity_max_radius_average_deviation': max_ratio_sum / counter,
                             # 'entity_number_of_residues_average_deviation': residue_ratio_sum / counter
                             })
        pose_metrics.update(**self.interface_metrics())

        return pose_metrics

    # @property
    # def df(self) -> pd.Series:
    #     """The Pose metrics. __init__: Retrieves all metrics, loads pd.Series if none loaded"""
    #     try:
    #         return self._df
    #     except AttributeError:  # Load metrics
    #         self._df = pd.Series(self.calculate_metrics())
    #     return self._df

    @property
    def active_entities(self) -> list[Entity]:
        """The Entity instances that are available for design calculations given a design selector"""
        return [self.get_entity(name) for name in self._design_selection_entity_names]

    @property
    def interface_residues(self) -> list[Residue]:
        """The Residue instances identified in interfaces in the Pose sorted based on index.
        Residue instances may be completely buried depending on interface distance
        """
        # try:
        #     return self._interface_residues
        # except AttributeError:
        if not self._interface_residue_indices_by_interface:
            self.find_and_split_interface()

        _interface_residues = []
        for number, residues in self.interface_residues_by_interface_unique.items():
            _interface_residues.extend(residues)
        _interface_residues = sorted(_interface_residues, key=lambda residue: residue.index)

        return _interface_residues

    @property
    def interface_residues_by_entity_pair(self) -> dict[tuple[Entity, Entity], tuple[list[Residue], list[Residue]]]:
        """The Residue instances identified between pairs of Entity instances"""
        residues = self.residues
        interface_residues_by_entity_pair = {}
        for (name1, name2), (indices1, indices2) in self._interface_residue_indices_by_entity_name_pair.items():
            interface_residues_by_entity_pair[(self.get_entity(name1), self.get_entity(name2))] = (
                [residues[ridx] for ridx in indices1],
                [residues[ridx] for ridx in indices2],
            )

        return interface_residues_by_entity_pair

    @property
    def interface_neighbor_residues(self) -> list[Residue]:
        """The Residue instances identified as neighbors to interfaces in the Pose. Assumes default distance of 8 A"""
        try:
            return self._interface_neighbor_residues
        except AttributeError:
            self._interface_neighbor_residues = []
            for residue in self.interface_residues:
                self._interface_neighbor_residues.extend(residue.get_neighbors())

            return self._interface_neighbor_residues

    # def interface_residues_accessible(self) -> list[Residue]:
    #     """Returns only those residues actively contributing to the interface"""
    #     try:
    #         return self._interface_residues_only
    #     except AttributeError:
    #         self._interface_residues_only = set()
    #         for entity in self.pose.entities:
    #             # entity.assembly.get_sasa()
    #             # Must get_residues by number as the Residue instance will be different in entity_oligomer
    #             for residue in entity.assembly.get_residues(self._interface_residues):
    #                 if residue.sasa > 0:
    #                     # Using set ensures that if we have repeats they won't be unique if Entity is symmetric
    #                     self._interface_residues_only.add(residue)
    #         # interface_residue_numbers = [residue.number for residue in self._interface_residues_only]
    #         # self.log.debug(f'Found interface residues: {", ".join(map(str, sorted(interface_residue_numbers)))}')
    #         return self._interface_residues_only

    @property
    def design_residues(self) -> list[Residue]:
        """The Residue instances identified for design in the Pose. Includes interface_residues"""
        try:
            return self._design_residues
        except AttributeError:
            # self.log.debug('The design_residues include interface_residues')
            self._design_residues = self.required_residues + self.interface_residues
            return self._design_residues

    @design_residues.setter
    def design_residues(self, residues: Iterable[Residue]):
        """The Residue instances identified for design in the Pose. Includes interface_residues"""
        self._design_residues = list(residues)

    def apply_design_selector(self, selection: StructureSpecification = None, mask: StructureSpecification = None,
                              required: StructureSpecification = None):
        """Set up a design selector for the Pose including selections, masks, and required Entities and Atoms

        Sets:
            self._design_selection_entity_names set[str]
            self._design_selector_atom_indices set[int]
            self._required_atom_indices Sequence[int]
        """

        def grab_indices(entities: set[str] = None, chains: set[str] = None, residues: set[int] = None) \
                -> tuple[set[str], set[int]]:
            # atoms: set[int] = None
            """Parses the residue selector to a set of entities and a set of atom indices

            Args:
                entities: The Entity identifiers to include in selection schemes
                chains: The Chain identifiers to include in selection schemes
                residues: The Residue identifiers to include in selection schemes

            Returns:
                A tuple with the names of Entity instances and the indices of the Atom/Coord instances that are parsed
            """
            # if start_with_none:
            #     set_function = set.union
            # else:  # Start with all indices and include those of interest
            #     set_function = set.intersection

            entity_atom_indices = []
            entities_of_interest = []
            # All selectors could be a set() or None.
            if entities:
                for entity_name in entities:
                    entity = self.get_entity(entity_name)
                    if entity is None:
                        raise NameError(
                            f"No entity named '{entity_name}'")
                    entities_of_interest.append(entity)
                    entity_atom_indices.extend(entity.atom_indices)

            if chains:
                for chain_id in chains:
                    chain = self.get_chain(chain_id)
                    if chain is None:
                        raise NameError(
                            f"No chain named '{chain_id}'")
                    entities_of_interest.append(chain.entity)
                    entity_atom_indices.extend(chain.atom_indices)

            # vv This is for the additive model
            # atom_indices.union(iter_chain.from_iterable(self.chain(chain_id).get_residue_atom_indices(numbers=residues)
            #                                     for chain_id in chains))
            # vv This is for the intersectional model
            # atom_indices = set_function(atom_indices, entity_atom_indices)
            # entity_set = set_function(entity_set, [ent.name for ent in entities_of_interest])
            atom_indices = set(entity_atom_indices)
            entity_set = {ent.name for ent in entities_of_interest}

            if residues:
                atom_indices = atom_indices.union(self.get_residue_atom_indices(numbers=residues))
            # if pdb_residues:
            #     atom_indices = set_function(atom_indices, self.get_residue_atom_indices(numbers=residues, pdb=True))
            # if atoms:
            #     atom_indices = set_function(atom_indices, [idx for idx in self._atom_indices if idx in atoms])

            return entity_set, atom_indices

        if selection:
            self.log.debug(f"The 'design_selector' {selection=}")
            entity_selection, atom_selection = grab_indices(**selection)
        else:  # Use existing entities and indices
            entity_selection = self._design_selection_entity_names
            atom_selection = self._design_selector_atom_indices

        if mask:
            self.log.debug(f"The 'design_selector' {mask=}")
            entity_mask, atom_mask = grab_indices(**mask)  # , start_with_none=True)
        else:
            entity_mask = set()
            atom_mask = set()

        entity_selection = entity_selection.difference(entity_mask)
        atom_selection = atom_selection.difference(atom_mask)

        if required:
            self.log.debug(f"The 'design_selector' {required=}")
            entity_required, required_atom_indices = grab_indices(**required)  # , start_with_none=True)
            self._required_atom_indices = list(required_atom_indices)
        else:
            entity_required = set()

        self._design_selection_entity_names = entity_selection.union(entity_required)
        self._design_selector_atom_indices = atom_selection.union(self._required_atom_indices)

        self.log.debug(f'Entities: {", ".join(entity.name for entity in self.entities)}')
        self.log.debug(f'Active Entities: {", ".join(name for name in self._design_selection_entity_names)}')

    @property
    def required_residues(self) -> list[Residue]:
        """Returns the Residue instances that are required according to DesignSelector"""
        return self.get_residues_by_atom_indices(self._required_atom_indices)

    def get_alphafold_features(
            self, symmetric: bool = False, multimer: bool = False, **kwargs
    ) -> FeatureDict:
        """Retrieve the required feature dictionary for this instance to use in Alphafold inference

        Args:
            symmetric: Whether the symmetric version of the Pose should be used for feature production
            multimer: Whether to run as a multimer. If multimer is True while symmetric is False, the Pose will
                be processed according to the ASU

        Keyword Args:
            msas: Sequence - A sequence of multiple sequence alignments if they should be included in the features
            no_msa: bool = False - Whether multiple sequence alignments should be included in the features

        Returns:
            The alphafold FeatureDict which is essentially a dictionary with dict[str, np.ndarray]
        """
        # heteromer = heteromer or self.number_of_entities > 1
        # symmetric = symmetric or heteromer or self.number_of_chains > 1
        # if multimer:

        # Set up the ASU unless otherwise specified
        number_of_symmetry_mates = 1
        if self.is_symmetric():
            if symmetric:
                number_of_symmetry_mates = self.number_of_symmetry_mates
                multimer = True
        # else:
        #     self.log.warning("Can't run as symmetric since Pose isn't symmetric")

        if self.number_of_entities > 1:
            # self.log.warning(f"Can't run with symmetric=True while multimer=False on a {self.__class__.__name__}. "
            #                  f"Setting heteromer=True")
            heteromer = multimer = True
        else:
            heteromer = False
            # if self.number_of_chains > 1:
            #     multimer = True
            # raise ValueError(f"Can't run monomer when {self.number_of_entities} entities are present in the "
            #                  f"{self.__class__.__name__}. Use Entity.{Entity.get_alphafold_features.__name__} on "
            #                  f"each entity individually instead")

        chain_count = count(1)
        all_chain_features = {}
        available_chain_ids = list(chain_id_generator())[:self.number_of_entities * self.number_of_symmetry_mates]
        available_chain_ids_iter = iter(available_chain_ids)
        for entity_idx, entity in enumerate(self.entities):
            entity_features = entity.get_alphafold_features(heteromer=heteromer, **kwargs)  # no_msa=no_msa)
            # The above function creates most of the work for the adaptation
            # particular importance needs to be given to the MSA used.
            # Should fragments be utilized in the MSA? If so, naming them in some way to pair is required!
            # Follow the example in:
            #    af_pipeline.make_msa_features(msas: Sequence[af_data_parsers.Msa]) -> FeatureDict
            # to featurize

            if multimer:  # symmetric:
                # The chain_id passed to this function is used by the entity_features to (maybe) tie different chains
                # to this chain. In alphafold implementation, the chain_id passed should be oriented so that each
                # additional entity has the chain_id of the chain number within the entire system.
                # For example, for an A4B4 heteromer with C4 symmetry, the chain_id for entity idx 0 would be A and for
                # entity idx 1 would be E. This may not be important, but this is how symmetric is prepared
                chain_id = available_chain_ids[self.number_of_symmetry_mates * entity_idx]
                entity_features = af_pipeline_multimer.convert_monomer_features(entity_features, chain_id=chain_id)

                entity_integer = entity_idx + 1
                entity_id = af_pipeline_multimer.int_id_to_str_id(entity_integer)

            entity_length = entity.number_of_residues
            # for _ in range(self.number_of_symmetry_mates):
            for sym_idx in range(1, 1 + number_of_symmetry_mates):
                # chain_id = next(available_chain_ids_iter)  # The mmCIF formatted chainID with 'AB' type notation
                this_entity_features = deepcopy(entity_features)
                # Where chain_id increments for each new chain instance i.e. A_1 is 1, A_2 is 2, ...
                # Where entity_id increments for each new Entity instance i.e. A_1 is 1, A_2 is 1, ...
                # Where sym_id increments for each new Entity instance regardless of chain i.e. A_1 is 1, A_2 is 2, ...,
                # B_1 is 1, B2 is 2
                if multimer:  # symmetric:
                    this_entity_features.update({'asym_id': next(chain_count) * np.ones(entity_length),
                                                 'sym_id': sym_idx * np.ones(entity_length),
                                                 'entity_id': entity_integer * np.ones(entity_length)})
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
        # all_chain_features = \
        #     af_pipeline_multimer.add_assembly_features(all_chain_features)

        if multimer:  # symmetric:
            all_chain_features = af_feature_processing.pair_and_merge(all_chain_features=all_chain_features)
            # Pad MSA to avoid zero-sized extra_msa.
            all_chain_features = af_pipeline_multimer.pad_msa(all_chain_features, 512)

        return all_chain_features

    def get_proteinmpnn_params(self, ca_only: bool = False, pssm_bias_flag: bool = False, pssm_multi: float = 0.,
                               bias_pssm_by_probabilities: bool = False, pssm_log_odds_flag: bool = False,
                               interface: bool = False, neighbors: bool = False, **kwargs) -> dict[str, np.ndarray]:
        # decode_core_first: bool = False
        """

        Args:
            ca_only: Whether a minimal CA variant of the protein should be used for design calculations
            pssm_bias_flag: Whether to use bias to modulate the residue probabilities designed
            pssm_multi: How much to skew the design probabilities towards the sequence profile.
                Bounded between [0, 1] where 0 is no sequence profile probability.
                Only used with pssm_bias_flag and modifies each coefficient in pssm_coef by the fractional amount
            bias_pssm_by_probabilities: Whether to produce bias by profile probabilities as opposed to lods
            pssm_log_odds_flag: Whether to use log_odds threshold (>0) to limit amino acid types of designed residues
                Creates pssm_log_odds_mask based on the threshold
            interface: Whether to design the interface
            neighbors: Whether to design interface neighbors

        Keyword Args:
            distance: float = 8. - The distance to measure Residues across an interface

        Returns:
            A mapping of the ProteinMPNN parameter names to their data, typically arrays
        """
        #   decode_core_first: Whether to decode the interface core first
        # Initialize pose data structures for design
        number_of_residues = self.number_of_residues
        if interface:
            self.find_and_split_interface(**kwargs)

            # Add all interface + required residues
            self.design_residues = self.required_residues + self.interface_residues
            if neighbors:
                self.design_residues += self.interface_neighbor_residues

            design_indices = [residue.index for residue in self.design_residues]
        else:
            self.design_residues = self.residues
            design_indices = list(range(number_of_residues))

        if ca_only:  # self.job.design.ca_only:
            coords_type = 'ca_coords'
            num_model_residues = 1
        else:
            coords_type = 'backbone_coords'
            num_model_residues = 4

        # Make masks for the sequence design task
        # Residue position mask denotes which residues should be designed. 1 - designed, 0 - known
        residue_mask = np.zeros(number_of_residues, dtype=np.int32)  # (number_of_residues,)
        residue_mask[design_indices] = 1

        omit_AAs_np = np.zeros(ml.mpnn_alphabet_length, dtype=np.int32)  # (alphabet_length,)
        bias_AAs_np = np.zeros_like(omit_AAs_np)  # (alphabet_length,)
        omit_AA_mask = np.zeros((number_of_residues, ml.mpnn_alphabet_length),
                                dtype=np.int32)  # (number_of_residues, alphabet_length)
        bias_by_res = np.zeros(omit_AA_mask.shape, dtype=np.float32)  # (number_of_residues, alphabet_length)

        # Get sequence profile to include for design bias
        pssm_threshold = 0.  # Must be a greater probability than wild-type
        pssm_log_odds = pssm_as_array(self.profile, lod=True)  # (number_of_residues, 20)
        pssm_log_odds_mask = np.where(pssm_log_odds >= pssm_threshold, 1., 0.)  # (number_of_residues, 20)
        pssm_coef = np.ones(residue_mask.shape, dtype=np.float32)  # (number_of_residues,)
        # shape (1, 21) where last index (20) is 1

        # Make the pssm_bias between 0 and 1 specifying how important position is where 1 is more important
        if bias_pssm_by_probabilities:
            pssm_probability = pssm_as_array(self.profile)
            pssm_bias = softmax(pssm_probability)  # (number_of_residues, 20)
        else:
            pssm_bias = softmax(pssm_log_odds)  # (number_of_residues, 20)

        if self.is_symmetric():
            number_of_symmetry_mates = self.number_of_symmetry_mates
            number_of_sym_residues = number_of_residues * number_of_symmetry_mates
            X = self.return_symmetric_coords(getattr(self, coords_type))
            # Should be N, CA, C, O for each residue
            #  v - Residue
            # [[[N  [x, y, z],
            #   [CA [x, y, z],
            #   [C  [x, y, z],
            #   [O  [x, y, z]],
            #  [[], ...      ]]
            # split the coordinates into those grouped by residues
            # X = np.array(.split(X, self.number_of_residues))
            X = X.reshape((number_of_sym_residues, num_model_residues, 3))  # (number_of_sym_residues, 4, 3)

            S = np.tile(self.sequence_numeric, number_of_symmetry_mates)  # (number_of_sym_residues,)
            # self.log.info(f'self.sequence_numeric: {self.sequence_numeric}')
            # self.log.info(f'Tiled sequence_numeric.shape: {S.shape}')
            # self.log.info(f'Tiled sequence_numeric start: {S[:5]}')
            # self.log.info(f'Tiled sequence_numeric chain_break: '
            #               f'{S[number_of_residues-5: number_of_residues+5]}')

            # Make masks for the sequence design task
            residue_mask = np.tile(residue_mask, number_of_symmetry_mates)  # (number_of_sym_residues,)
            mask = np.ones_like(residue_mask)  # (number_of_sym_residues,)
            # Chain mask denotes which chains should be designed. 1 - designed, 0 - known
            # For symmetric systems, treat each chain as designed as the logits are averaged during model.tied_sample()
            chain_mask = np.ones_like(residue_mask)  # (number_of_sym_residues,)
            # Set up a simple array where each residue index has the index of the chain starting with the index of 1
            chain_encoding = np.zeros_like(residue_mask)  # (number_of_residues,)
            # Set up an array where each residue index is incremented, however each chain break has an increment of 100
            residue_idx = np.arange(number_of_sym_residues, dtype=np.int32)  # (number_of_residues,)
            number_of_entities = self.number_of_entities
            for model_idx in range(number_of_symmetry_mates):
                model_offset = model_idx * number_of_residues
                model_entity_number = model_idx * number_of_entities
                for idx, entity in enumerate(self.entities, 1):
                    entity_number_of_residues = entity.number_of_residues
                    entity_start = entity.offset_index + model_offset
                    chain_encoding[entity_start:entity_start + entity_number_of_residues] = model_entity_number + idx
                    residue_idx[entity_start:entity_start + entity_number_of_residues] += \
                        (model_entity_number+idx) * 100

            # self.log.debug(f'Tiled chain_encoding chain_break: '
            #                f'{chain_encoding[number_of_residues-5: number_of_residues+5]}')
            # self.log.debug(f'Tiled residue_idx chain_break: '
            #                f'{residue_idx[number_of_residues-5: number_of_residues+5]}')

            pssm_coef = np.tile(pssm_coef, number_of_symmetry_mates)  # (number_of_sym_residues,)
            # Below have shape (number_of_sym_residues, alphabet_length)
            pssm_bias = np.tile(pssm_bias, (number_of_symmetry_mates, 1))
            pssm_log_odds_mask = np.tile(pssm_log_odds_mask, (number_of_symmetry_mates, 1))
            omit_AA_mask = np.tile(omit_AA_mask, (number_of_symmetry_mates, 1))
            bias_by_res = np.tile(bias_by_res, (number_of_symmetry_mates, 1))
            self.log.debug(f'Tiled bias_by_res start: {bias_by_res[:5]}')
            self.log.debug(f'Tiled bias_by_res: '
                           f'{bias_by_res[number_of_residues-5: number_of_residues+5]}')
            tied_beta = np.ones_like(residue_mask)  # (number_of_sym_residues,)
            tied_pos = [self.make_indices_symmetric([idx], dtype='residue') for idx in design_indices]
            # (design_residues, number_of_symmetry_mates)
        else:
            X = getattr(self, coords_type).reshape((number_of_residues, num_model_residues, 3))  # (residues, 4, 3)
            S = self.sequence_numeric  # (number_of_residues,)
            mask = np.ones_like(residue_mask)  # (number_of_residues,)
            chain_mask = np.ones_like(residue_mask)  # (number_of_residues,)
            # Set up a simple array where each residue index has the index of the chain starting with the index of 1
            chain_encoding = np.zeros_like(residue_mask)  # (number_of_residues,)
            # Set up an array where each residue index is incremented, however each chain break has an increment of 100
            residue_idx = np.arange(number_of_residues, dtype=np.int32)  # (number_of_residues,)
            for idx, entity in enumerate(self.entities):
                entity_number_of_residues = entity.number_of_residues
                entity_start = entity.offset_index
                chain_encoding[entity_start: entity_start + entity_number_of_residues] = idx + 1
                residue_idx[entity_start: entity_start + entity_number_of_residues] += idx * 100
            tied_beta = None  # np.ones_like(residue_mask)  # (number_of_sym_residues,)
            tied_pos = None  # [[]]

        return dict(X=X,
                    S=S,
                    chain_mask=chain_mask,
                    chain_encoding=chain_encoding,
                    residue_idx=residue_idx,
                    mask=mask,
                    omit_AAs_np=omit_AAs_np,
                    bias_AAs_np=bias_AAs_np,
                    chain_M_pos=residue_mask,
                    omit_AA_mask=omit_AA_mask,
                    pssm_coef=pssm_coef,
                    pssm_bias=pssm_bias,
                    pssm_multi=pssm_multi,
                    pssm_log_odds_flag=pssm_log_odds_flag,
                    pssm_log_odds_mask=pssm_log_odds_mask,
                    pssm_bias_flag=pssm_bias_flag,
                    tied_pos=tied_pos,
                    tied_beta=tied_beta,
                    bias_by_res=bias_by_res
                    )

    def generate_proteinmpnn_decode_order(self, to_device: str = None, core_first: bool = False, **kwargs) -> \
            torch.Tensor | np.ndarray:
        """Return the decoding order for ProteinMPNN. Currently just returns an array of random floats

        For original ProteinMPNN GitHub release, the decoding order is only dependent on first entry in batch for
        model.tied_sample() while it is dependent on the entire batch for model.sample()

        Args:
            to_device: Whether the decoding order should be transferred to the device that a ProteinMPNN model is on
            core_first: Whether the core residues (identified as fragment pairs) should be decoded first

        Returns:
            The decoding order to be used in ProteinMPNN graph decoding
        """
        pose_length = self.number_of_residues
        if self.is_symmetric():
            pose_length *= self.number_of_symmetry_mates

        if core_first:
            raise NotImplementedError("'core_first' isn't available yet")
        else:  # random decoding order
            randn = np.random.rand(pose_length)

        if to_device is None:
            return randn
        else:
            return torch.from_numpy(randn).to(dtype=torch.float32, device=to_device)

    def get_proteinmpnn_unbound_coords(self, ca_only: bool = False) -> np.ndarray:
        """Translate the coordinates along z in increments of 1000 to separate coordinates

        Args:
            ca_only: Whether a minimal CA variant of the protein should be used for design calculations

        Returns:
            The Pose coords where each Entity has been translated away from other entities
        """
        if ca_only:
            coords_type = 'ca_coords'
            num_model_residues = 1
        else:
            coords_type = 'backbone_coords'
            num_model_residues = 4

        unbound_transform = np.array([0., 0., 1000.])
        if self.is_symmetric():
            number_of_residues = self.number_of_symmetric_residues
            coord_func = self.return_symmetric_coords
        else:
            number_of_residues = self.number_of_residues
            def coord_func(coords): return coords

        # Caution this doesn't move the oligomeric unit, it moves the ASU entity.
        # "Unbound" measurements shouldn't modify the oligomeric unit
        entity_unbound_coords = []
        for idx, entity in enumerate(self.entities, 1):
            entity_unbound_coords.append(coord_func(getattr(entity, coords_type) + unbound_transform*idx))

        return np.concatenate(entity_unbound_coords).reshape((number_of_residues, num_model_residues, 3))

    @torch.no_grad()  # Ensure no gradients are produced
    def score_sequences(self, sequences: Sequence[str] | Sequence[Sequence[str]] | np.array,
                        method: design_programs_literal = putils.proteinmpnn,
                        measure_unbound: bool = True, ca_only: bool = False, **kwargs) -> dict[str, np.ndarray]:
        """Analyze the output of sequence design

        Args:
            sequences: The sequences to score
            method: Whether to score using ProteinMPNN or Rosetta
            measure_unbound: Whether the protein should be scored in the unbound state
            ca_only: Whether a minimal CA variant of the protein should be used for design calculations

        Keyword Args:
            model_name: The name of the model to use from ProteinMPNN taking the format v_X_Y,
                where X is neighbor distance and Y is noise
            backbone_noise: float = 0.0 - The amount of backbone noise to add to the pose during design
            pssm_multi: float = 0.0 - How much to skew the design probabilities towards the sequence profile.
                Bounded between [0, 1] where 0 is no sequence profile probability.
                Only used with pssm_bias_flag
            pssm_log_odds_flag: bool = False - Whether to use log_odds mask to limit the residues designed
            pssm_bias_flag: bool = False - Whether to use bias to modulate the residue probabilities designed
            bias_pssm_by_probabilities: Whether to produce bias by profile probabilities as opposed to profile lods
            decode_core_first: bool = False - Whether to decode identified fragments (constituting the protein core)
                first

        Returns:
            A mapping of the design score type name to the per-residue output data which is a ndarray with shape
            (number of sequences, pose_length).
            For proteinmpnn,
            these are the outputs: 'sequences', 'numeric_sequences', 'proteinmpnn_loss_complex', and
            'proteinmpnn_loss_unbound' mapped to their corresponding arrays with data types as np.ndarray

            For rosetta, this function isn't implemented
        """
        if method == putils.rosetta:
            sequences_and_scores = {}
            raise NotImplementedError(f"Can't score with Rosetta from this method yet...")
        elif method == putils.proteinmpnn:  # Design with vanilla version of ProteinMPNN
            # Convert the sequences to correct format
            # missing_alphabet = ''
            warn_alphabet = 'With passed sequences type of {}, ensure that the order of ' \
                            f'integers is of the default ProteinMPNN alphabet "{ml.mpnn_alphabet}"'

            def convert_and_check_sequence_type(sequences_) -> Sequence[Sequence[str | int]]:
                incorrect_input = ValueError(f'The passed sequences must be an Sequence[Sequence[Any]]')
                nesting_level = count()
                item = sequences_
                # print(item)
                while not isinstance(item, (int, str)):
                    next(nesting_level)
                    item = item[0]
                    # print(item)
                else:
                    final_level = next(nesting_level)
                    item_type = type(item)
                    # print(final_level)
                    # print(item_type)
                    if final_level == 1:
                        if item_type is str:
                            sequences_ = sequences_to_numeric(
                                sequences, translation_table=ml.proteinmpnn_default_translation_table)
                        else:
                            raise incorrect_input
                    elif final_level == 2:
                        if item_type is str:
                            for idx, sequence in enumerate(sequences_):
                                sequences[idx] = ''.join(sequence)

                            sequences_ = sequences_to_numeric(
                                sequences, translation_table=ml.proteinmpnn_default_translation_table)
                        else:
                            self.log.warning(warn_alphabet.format('int'))
                            sequences_ = np.array(sequences_)
                    else:
                        raise incorrect_input
                    # print('Final', sequences)
                return sequences_

            if isinstance(sequences, (torch.Tensor, np.ndarray)):
                if sequences.dtype in utils.np_torch_int_float_types:
                    # This is an integer sequence. An alphabet is required
                    self.log.warning(warn_alphabet.format(sequences.dtype))
                    numeric_sequences = sequences
                    sequences = numeric_to_sequence(sequences)
                    # raise ValueError(missing_alphabet)
                else:  # This is an AnyStr type?
                    numeric_sequences = sequences_to_numeric(sequences)
            else:  # Some sort of iterable
                numeric_sequences = convert_and_check_sequence_type(sequences)

            # pose_length = self.number_of_residues
            size, pose_length, *_ = numeric_sequences.shape
            batch_length = ml.PROTEINMPNN_SCORE_BATCH_LEN

            # Set up parameters and model sampling type based on symmetry
            if measure_unbound:
                parameters = {'X_unbound': self.get_proteinmpnn_unbound_coords(ca_only=ca_only)}
            else:
                parameters = {}

            # Set up return containers based on the asymmetric sequence
            per_residue_complex_sequence_loss = np.empty_like(numeric_sequences, dtype=np.float32)
            per_residue_unbound_sequence_loss = np.empty_like(per_residue_complex_sequence_loss)

            # Set up parameters for the scoring task, including scoring all positions
            parameters.update(**self.get_proteinmpnn_params(ca_only=ca_only, **kwargs))

            # Remove the precalculated sequence array to add our own
            parameters.pop('S')
            # # Insert the designed sequences inplace of the pose sequence
            # parameters['S'] = np.tile(numeric_sequences, (1, number_of_symmetry_mates))
            # Set up for symmetry
            numeric_sequences_ = np.tile(numeric_sequences, (1, self.number_of_symmetry_mates))
            # Solve decoding order
            # parameters['randn'] = self.generate_proteinmpnn_decode_order(**kwargs)  # to_device=device)
            # decoding_order = self.generate_proteinmpnn_decode_order(**kwargs)  # to_device=device)
            # Solve decoding order
            parameters['randn'] = self.generate_proteinmpnn_decode_order(**kwargs)  # to_device=device)

            @ml.batch_calculation(size=size, batch_length=batch_length,
                                  setup=ml.setup_pose_batch_for_proteinmpnn,
                                  compute_failure_exceptions=(RuntimeError,
                                                              np.core._exceptions._ArrayMemoryError))
            def _proteinmpnn_batch_score(*args, **_kwargs):
                return ml.proteinmpnn_batch_score(*args, **_kwargs)

            # Set up the model with the desired weights
            proteinmpnn_model = ml.proteinmpnn_factory(ca_only=ca_only, **kwargs)
            device = proteinmpnn_model.device

            # Send the numpy array to torch.tensor and the device
            # Pass sequences as 'S' parameter to _proteinmpnn_batch_score instead of as setup_kwargs
            # unique_parameters = ml.proteinmpnn_to_device(device, S=sequences, decoding_order=decoding_order)
            unique_parameters = ml.proteinmpnn_to_device(device, S=numeric_sequences_)
            # score_start = time.time()
            scores = \
                _proteinmpnn_batch_score(proteinmpnn_model, **unique_parameters,  # S=sequences,
                                         pose_length=pose_length,  # decoding_order=decoding_order,
                                         setup_args=(device,),
                                         setup_kwargs=parameters,
                                         return_containers={
                                             'proteinmpnn_loss_complex': per_residue_complex_sequence_loss,
                                             'proteinmpnn_loss_unbound': per_residue_unbound_sequence_loss})
            sequences_and_scores = {'sequences': sequences,
                                    'numeric_sequences': numeric_sequences,
                                    **scores}
        else:
            sequences_and_scores = {}
            raise ValueError(
                f"The method '{method}' isn't a viable scoring protocol")

        return sequences_and_scores

    @torch.no_grad()  # Ensure no gradients are produced
    def design_sequences(self, method: design_programs_literal = putils.proteinmpnn, number: int = 10,
                         temperatures: Sequence[float] = (0.1,), interface: bool = False, neighbors: bool = False,
                         measure_unbound: bool = True, ca_only: bool = False, **kwargs) -> dict[str, np.ndarray]:
        """Perform sequence design on the Pose

        Args:
            method: Whether to design using ProteinMPNN or Rosetta
            number: The number of sequences to design
            temperatures: The temperatures to perform design at
            interface: Whether to design the interface
            neighbors: Whether to design interface neighbors
            measure_unbound: Whether the protein should be designed with concern for the unbound state
            ca_only: Whether a minimal CA variant of the protein should be used for design calculations

        Keyword Args:
            neighbors: bool = False - Whether to design interface neighbors
            model_name: The name of the model to use from ProteinMPNN taking the format v_X_Y,
                where X is neighbor distance and Y is noise
            backbone_noise: float = 0.0 - The amount of backbone noise to add to the pose during design
            pssm_multi: float = 0.0 - How much to skew the design probabilities towards the sequence profile.
                Bounded between [0, 1] where 0 is no sequence profile probability.
                Only used with pssm_bias_flag
            pssm_log_odds_flag: bool = False - Whether to use log_odds mask to limit the residues designed
            pssm_bias_flag: bool = False - Whether to use bias to modulate the residue probabilites designed
            bias_pssm_by_probabilities: Whether to produce bias by profile probabilities as opposed to profile lods
            decode_core_first: bool = False - Whether to decode identified fragments (constituting the protein core)
                first

        Returns:
            A mapping of the design score type to the per-residue output data which is a ndarray with shape
            (number*temperatures, pose_length).
            For proteinmpnn,
            these are the outputs 'sequences', 'numeric_sequences', 'design_indices', 'proteinmpnn_loss_complex', and
            'proteinmpnn_loss_unbound' mapped to their corresponding score types. For each return array, the return
            varies such as: [temp1/number1, temp1/number2, ...,
            tempN/number1, ...] where designs are sorted by temperature

            For rosetta, this function isn't implemented
        """
        # rosetta: Whether to design using Rosetta energy functions
        if method == putils.rosetta:
            sequences_and_scores = {}
            raise NotImplementedError(f"Can't design with Rosetta from this method yet...")
        elif method == putils.proteinmpnn:  # Design with vanilla version of ProteinMPNN
            pose_length = self.number_of_residues
            # Set up parameters and model sampling type based on symmetry
            if self.is_symmetric():
                # number_of_symmetry_mates = pose.number_of_symmetry_mates
                # mpnn_sample = proteinmpnn_model.tied_sample
                number_of_residues = pose_length * self.number_of_symmetry_mates
            else:
                # mpnn_sample = proteinmpnn_model.sample
                number_of_residues = pose_length

            if measure_unbound:
                parameters = {'X_unbound': self.get_proteinmpnn_unbound_coords(ca_only=ca_only)}
            else:
                parameters = {}

            # Set up the inference task
            parameters.update(**self.get_proteinmpnn_params(interface=interface, neighbors=neighbors,
                                                            ca_only=ca_only, **kwargs))
            # Solve decoding order
            parameters['randn'] = self.generate_proteinmpnn_decode_order(**kwargs)  # to_device=device)

            # Set up the model with the desired weights
            size = number
            proteinmpnn_model = ml.proteinmpnn_factory(ca_only=ca_only, **kwargs)
            device = proteinmpnn_model.device
            batch_length = ml.calculate_proteinmpnn_batch_length(proteinmpnn_model, number_of_residues)
            # batch_length = ml.PROTEINMPNN_DESIGN_BATCH_LEN
            logger.info(f'Found ProteinMPNN batch_length={batch_length}')

            generated_sequences = np.empty((size, len(temperatures), pose_length), dtype=np.int64)
            per_residue_complex_sequence_loss = np.empty_like(generated_sequences, dtype=np.float32)
            per_residue_unbound_sequence_loss = np.empty_like(per_residue_complex_sequence_loss)
            design_indices = np.zeros((size, pose_length), dtype=bool)

            @ml.batch_calculation(size=size, batch_length=batch_length,
                                  setup=ml.setup_pose_batch_for_proteinmpnn,
                                  compute_failure_exceptions=(RuntimeError,
                                                              np.core._exceptions._ArrayMemoryError))
            def _proteinmpnn_batch_design(*args, **_kwargs):
                return ml.proteinmpnn_batch_design(*args, **_kwargs)

            # Data has shape (batch_length, number_of_temperatures, pose_length)
            number_of_temps = len(temperatures)
            # design_start = time.time()
            sequences_and_scores = \
                _proteinmpnn_batch_design(proteinmpnn_model, temperatures=temperatures, pose_length=pose_length,
                                          setup_args=(device,),
                                          setup_kwargs=parameters,
                                          return_containers={
                                              'sequences': generated_sequences,
                                              'proteinmpnn_loss_complex': per_residue_complex_sequence_loss,
                                              'proteinmpnn_loss_unbound': per_residue_unbound_sequence_loss,
                                              'design_indices': design_indices})
            # self.log.debug(f"Took {time.time() - design_start:8f}s for _proteinmpnn_batch_design")

            sequences_and_scores['numeric_sequences'] = sequences_and_scores.pop('sequences')
            sequences_and_scores['sequences'] = numeric_to_sequence(sequences_and_scores['numeric_sequences'])

            # Format returns to have shape (temperatures*size, pose_length) where the temperatures vary slower
            # Ex: [temp1/pose1, temp1/pose2, ..., tempN/pose1, ...] This groups the designs by temperature first
            for data_type, data in sequences_and_scores.items():
                if data_type == 'design_indices':
                    # These must vary by temperature
                    sequences_and_scores['design_indices'] = np.tile(data, (number_of_temps, 1))
                    # self.log.debug(f'Found design_indices with shape: {sequences_and_scores["design_indices"].shape}')
                    continue
                # print(f"{data_type} has shape {data.shape}")
                sequences_and_scores[data_type] = np.concatenate(data, axis=1).reshape(-1, pose_length)

            # self.log.debug(f'Found sequences with shape {sequences_and_scores["sequences"].shape}')
            # self.log.debug(f'Found proteinmpnn_loss_complex with shape {sequences_and_scores["proteinmpnn_loss_complex"].shape}')
            # self.log.debug(f'Found proteinmpnn_loss_unbound with shape {sequences_and_scores["proteinmpnn_loss_unbound"].shape}')
        else:
            sequences_and_scores = {}
            raise ValueError(
                f"The method '{method}' isn't a viable design protocol")

        return sequences_and_scores

    def get_termini_accessibility(self, entity: Entity = None, report_if_helix: bool = False) -> dict[str, bool]:
        """Returns a dictionary indicating which termini are not buried. Coarsely locates termini which face outward

        Args:
            entity: The Structure to query which originates in the pose
            report_if_helix: Whether the query should additionally report on the helicity of the termini

        Returns:
            A dictionary with the mapping from termini to True if the termini is exposed
                Ex: {'n': True, 'c': False}
        """
        assembly = self.assembly
        if not assembly.sasa:
            assembly.get_sasa()

        # Find the chain that matches the Entity
        for chain in assembly.chains:
            if chain.entity_id == entity.name:
                entity_chain = chain
                break
        else:
            raise DesignError(
                f"Couldn't find a corresponding {repr(assembly)}.chain for the passed entity={repr(entity)}")

        n_term = c_term = False
        entity_reference = None
        if self.is_symmetric():
            if self.dimension > 0:
                self.log.critical("Locating termini accessibility for lattice symmetries hasn't been tested")
                for _entity, entity_transformation in zip(self.entities, self.entity_transformations):
                    if entity == _entity:
                        entity_reference = entity_transformation.get('translation2', None)
                        break
                else:
                    raise ValueError(
                        f"Can't measure point of reference for entity={repr(entity)} as a matching instance wasn't "
                        f"found in the {repr(self)}")

        if entity.termini_proximity_from_reference(reference=entity_reference) == 1:  # if outward
            if entity_chain.n_terminal_residue.relative_sasa > metrics.default_sasa_burial_threshold:
                n_term = True

        if entity.termini_proximity_from_reference(termini='c', reference=entity_reference) == 1:  # if outward
            if entity_chain.c_terminal_residue.relative_sasa > metrics.default_sasa_burial_threshold:
                c_term = True

        if report_if_helix:
            try:
                retrieve_stride_info = resources.structure_db.structure_database_factory().stride.retrieve_data
            except AttributeError:
                pass
            else:
                parsed_secondary_structure = retrieve_stride_info(name=entity.name)
                if parsed_secondary_structure:
                    entity.secondary_structure = parsed_secondary_structure

            n_term = True if n_term and entity.is_termini_helical() else False
            c_term = True if c_term and entity.is_termini_helical(termini='c') else False

        return dict(n=n_term, c=c_term)

    def per_residue_contact_order(self, oligomeric_interfaces: bool = False, **kwargs) -> dict[str, np.ndarray]:
        """Calculate the contact order separating calculation for chain breaks as would be expected for 3 state folding

        Args:
            oligomeric_interfaces: Whether to query oligomeric interfaces

        Returns:
            The dictionary of {'contact_order': array of shape (number_of_residues,)}
        """
        if oligomeric_interfaces:
            raise NotImplementedError(
                "Need to perform 'oligomeric_interfaces' calculation on the Entity.oligomer")

        contact_order = []
        for idx, entity in enumerate(self.entities):
            contact_order.append(entity.contact_order)

        return {'contact_order': np.concatenate(contact_order)}

    def get_folding_metrics(
        self, profile_type: profile_types = 'evolutionary', **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate metrics relating to the Pose folding, separating calculation for chain breaks. These include
        contact_order, hydrophobic_collapse, and hydrophobic_collapse_profile (each Entity MUST have a .*_profile
        attribute to return the hydrophobic collapse profile!)

        Args:
            profile_type: The type of profile to use to calculate the hydrophobic collapse profile

        Keyword Args:
            hydrophobicity: str = 'standard' – The hydrophobicity scale to consider. Either 'standard' (FILV),
                'expanded' (FMILYVW), or provide one with 'custom' keyword argument
            custom: mapping[str, float | int] = None – A user defined mapping of amino acid type, hydrophobicity value
                pairs
            alphabet_type: alphabet_types = None – The amino acid alphabet if the sequence consists of integer
                characters
            lower_window: int = 3 – The smallest window used to measure
            upper_window: int = 9 – The largest window used to measure

        Returns:
            The per-residue contact_order_z_score (number_of_residues),
                a per-residue hydrophobic_collapse (number_of_residues),
                and the hydrophobic_collapse profile (number_of_residues) based on Entity.evolutionary_profile instances
        """
        #       and the hydrophobic_collapse profile (msa.length, msa.number_of_residues) based on Entity.msa instances
        # Measure the wild type (reference) entity versus modified entity(ies) to find the hci delta
        # Calculate Reference sequence statistics
        contact_order_z, hydrophobic_collapse, hydrophobic_collapse_profile = [], [], []
        missing = []
        msa_metrics = True
        for idx, entity in enumerate(self.entities):
            contact_order = entity.contact_order
            # This calculation shouldn't depend on oligomers... Only assumes unfolded -> folded
            # contact_order = entity_oligomer.contact_order[:entity.number_of_residues]
            entity_residue_contact_order_z = metrics.z_score(contact_order, contact_order.mean(), contact_order.std())
            contact_order_z.append(entity_residue_contact_order_z)
            # inverse_residue_contact_order_z.append(entity_residue_contact_order_z * -1)
            hydrophobic_collapse.append(entity.hydrophobic_collapse(**kwargs))

            # Set the entity.msa which makes a copy and adjusts for any disordered residues
            # This method is more accurate as it uses full sequences from MSA. However,
            # more time-consuming and may not matter much
            # if chain.msa and msa_metrics:
            #     hydrophobic_collapse_profile.append(chain.collapse_profile(**kwargs))
            #     collapse = chain.collapse_profile()
            #     entity_collapse_mean.append(collapse.mean(axis=-2))
            #     entity_collapse_std.append(collapse.std(axis=-2))
            #     reference_collapse_z_score.append(utils.z_score(reference_collapse, entity_collapse_mean[idx],
            #                                                     entity_collapse_std[idx]))
            if msa_metrics and entity.evolutionary_profile:
                try:
                    profile = getattr(entity, f'{profile_type}_profile')
                    if profile_type == 'fragment':
                        profile_array = profile.as_array()
                    else:
                        profile_array = pssm_as_array(profile)
                except AttributeError:  # No profile from getattr()
                    raise ValueError(
                        f"The profile_type '{profile_type}' isn't available on {type(entity).__name__} {entity.name}")

                hydrophobic_collapse_profile.append(
                    metrics.hydrophobic_collapse_index(profile_array, alphabet_type='protein_letters_alph1', **kwargs))
            else:
                missing.append(1)
                msa_metrics = False

        contact_order_z = np.concatenate(contact_order_z)
        hydrophobic_collapse = np.concatenate(hydrophobic_collapse)
        if sum(missing):  # Need to sum as it could be empty from no .entities and then wouldn't collect either
            hydrophobic_collapse_profile = np.empty(0)
            self.log.warning(f'There were missing .evolutionary_profile attributes for {sum(missing)} Entity instances.'
                             f' The collapse_profile will not be captured for the entire {self.__class__.__name__}')
        #     self.log.warning(f'There were missing MultipleSequenceAlignment objects on {sum(missing)} Entity '
        #                      'instances. The collapse_profile will not be captured for the entire '
        #                      f'{self.__class__.__name__}.')
        # else:
        #     # We have to concatenate where the values will be different
        #     # axis=1 is along the residues, so the result should be the length of the pose
        #     # axis=0 will be different for each individual entity, so we pad to the maximum for lesser ones
        #     array_sizes = [array.shape[0] for array in hydrophobic_collapse_profile]
        #     axis0_max_length = max(array_sizes)
        #     # full_hydrophobic_collapse_profile = \
        #     #     np.full((axis0_max_length, self.number_of_residues), np.nan)  # , dtype=np.float32)
        #     for idx, array in enumerate(hydrophobic_collapse_profile):
        #         hydrophobic_collapse_profile[idx] = \
        #             np.pad(array, ((0, axis0_max_length - array_sizes[idx]), (0, 0)), constant_values=np.nan)
        #
        #     hydrophobic_collapse_profile = np.concatenate(hydrophobic_collapse_profile, axis=1)
        else:
            hydrophobic_collapse_profile = np.concatenate(hydrophobic_collapse_profile)

        return contact_order_z, hydrophobic_collapse, hydrophobic_collapse_profile

    def interface_metrics(self) -> dict[str, Any]:
        """Gather all metrics relating to the Pose and the interfaces within the Pose

        Calls self.get_fragment_metrics(), self.calculate_secondary_structure()

        Returns:
            Metrics measured as: {
                'entity_max_radius_average_deviation',
                'entity_min_radius_average_deviation',
                'entity_radius_average_deviation',
                'interface_b_factor',
                'interface1_secondary_structure_fragment_topology',
                'interface1_secondary_structure_fragment_count',
                'interface1_secondary_structure_topology',
                'interface1_secondary_structure_count',
                'interface2_secondary_structure_fragment_topology',
                'interface2_secondary_structure_fragment_count',
                'interface2_secondary_structure_topology',
                'interface2_secondary_structure_count',
                'maximum_radius',
                'minimum_radius',
                'multiple_fragment_ratio',
                'nanohedra_score_normalized',
                'nanohedra_score_center_normalized',
                'nanohedra_score',
                'nanohedra_score_center',
                'number_residues_interface_fragment_total',
                'number_residues_interface_fragment_center',
                'number_fragments_interface',
                'number_residues_interface',
                'number_residues_interface_non_fragment',
                'percent_fragment_helix',
                'percent_fragment_strand',
                'percent_fragment_coil',
                'percent_residues_fragment_interface_total',
                'percent_residues_fragment_interface_center',
                'percent_residues_non_fragment_interface',
                'pose_length',
                'symmetric_interface'
            }
                # 'entity_radius_ratio_#v#',
                # 'entity_min_radius_ratio_#v#',
                # 'entity_max_radius_ratio_#v#',
                # 'entity_number_of_residues_ratio_#v#',
                # 'entity_number_of_residues_average_deviation,
        """
        pose_metrics = self.get_fragment_metrics(total_interface=True)
        # Remove *_indices from further analysis
        interface_fragment_residue_indices = self.interface_fragment_residue_indices = (
            pose_metrics.pop('center_indices', []))
        pose_metrics.pop('total_indices')
        number_residues_fragment_center = pose_metrics.pop('number_residues_fragment_center')
        number_residues_fragment_total = pose_metrics.pop('number_residues_fragment_total')

        interface_residues = self.interface_residues
        number_residues_interface = len(interface_residues)
        number_residues_interface_non_fragment = number_residues_interface - number_residues_fragment_total
        # if number_residues_interface_non_fragment < 0:
        #     raise ValueError(f'Fragment metrics are broken due to "number_residues_interface_non_fragment" > 1')
        # Interface B Factor
        int_b_factor = sum(residue.b_factor for residue in interface_residues)
        try:  # If interface_distance is different from interface query and fragment generation these can be < 0 or > 1
            percent_residues_fragment_interface_center = number_residues_fragment_center / number_residues_interface
            # This value can be more than 1 so not a percent...
            # percent_residues_fragment_interface_total = number_residues_fragment_total / number_residues_interface
            percent_residues_fragment_interface_total = \
                min(number_residues_fragment_total / number_residues_interface, 1)
            percent_interface_residues_non_fragment = number_residues_interface_non_fragment / number_residues_interface
            # if percent_residues_fragment_interface_center > 1:
            #     raise ValueError(f'Fragment metrics are broken due to "percent_residues_fragment_interface_center">1')
            # if percent_interface_residues_non_fragment > 1:
            #     raise ValueError(f'Fragment metrics are broken due to "percent_interface_residues_non_fragment" > 1')
            ave_b_factor = int_b_factor / number_residues_interface
        except ZeroDivisionError:
            self.log.warning(f'{self.name}: No interface residues were found. Is there an interface in your design?')
            ave_b_factor = percent_residues_fragment_interface_center = percent_residues_fragment_interface_total = \
                percent_interface_residues_non_fragment = 0.

        pose_metrics.update({
            'interface_b_factor': ave_b_factor,
            'number_residues_interface': number_residues_interface,
            'number_residues_interface_fragment_center': number_residues_fragment_center,
            'number_residues_interface_fragment_total': number_residues_fragment_total,
            'number_residues_interface_non_fragment': number_residues_interface_non_fragment,
            'percent_residues_non_fragment_interface': percent_interface_residues_non_fragment,
            'percent_residues_fragment_interface_total': percent_residues_fragment_interface_total,
            'percent_residues_fragment_interface_center': percent_residues_fragment_interface_center})

        interface_residues_by_interface_unique = self.interface_residues_by_interface_unique
        if not self.ss_sequence_indices or not self.ss_type_sequence:
            self.calculate_secondary_structure()

        # interface_ss_topology = {}  # {1: 'HHLH', 2: 'HSH'}
        # interface_ss_fragment_topology = {}  # {1: 'HHH', 2: 'HH'}
        ss_type_array = self.ss_type_sequence
        total_interface_ss_topology = total_interface_ss_fragment_topology = ''
        for number, elements in self.split_interface_ss_elements.items():
            # Use unique as 2-fold interfaces 'interface_residues_by_interface_unique' duplicate fragment_elements
            fragment_elements = {
                element for residue, element in zip(interface_residues_by_interface_unique[number], elements)
                if residue.index in interface_fragment_residue_indices}
            # Take the set of elements as there are element repeats if SS is continuous over residues
            interface_ss_topology = ''.join(ss_type_array[element] for element in set(elements))
            interface_ss_fragment_topology = ''.join(ss_type_array[element] for element in fragment_elements)

            pose_metrics[f'interface{number}_secondary_structure_topology'] = interface_ss_topology
            pose_metrics[f'interface{number}_secondary_structure_count'] = len(interface_ss_topology)
            total_interface_ss_topology += interface_ss_topology
            pose_metrics[f'interface{number}_secondary_structure_fragment_topology'] = interface_ss_fragment_topology
            pose_metrics[f'interface{number}_secondary_structure_fragment_count'] = len(interface_ss_fragment_topology)
            total_interface_ss_fragment_topology += interface_ss_fragment_topology

        pose_metrics['interface_secondary_structure_fragment_topology'] = total_interface_ss_fragment_topology
        pose_metrics['interface_secondary_structure_fragment_count'] = len(total_interface_ss_fragment_topology)
        pose_metrics['interface_secondary_structure_topology'] = total_interface_ss_topology
        pose_metrics['interface_secondary_structure_count'] = len(total_interface_ss_topology)

        # Calculate secondary structure percent for the entire interface
        helical_designations = ['H', 'G', 'I']
        strand_designations = ['E', 'B']
        coil_designations = ['C', 'T']
        number_helical_residues = number_strand_residues = number_loop_residues = 0
        for residue in interface_residues:
            if residue.secondary_structure in helical_designations:
                number_helical_residues += 1
            elif residue.secondary_structure in strand_designations:
                number_strand_residues += 1
            elif residue.secondary_structure in coil_designations:
                number_loop_residues += 1

        pose_metrics['percent_interface_helix'] = number_helical_residues / number_residues_interface
        pose_metrics['percent_interface_strand'] = number_strand_residues / number_residues_interface
        pose_metrics['percent_interface_coil'] = number_loop_residues / number_residues_interface
        if self.interface_residues_by_interface == interface_residues_by_interface_unique:
            pose_metrics['symmetric_interface'] = False
        else:
            pose_metrics['symmetric_interface'] = True

        return pose_metrics

    def per_residue_errat(self) -> dict[str, list[float]]:
        """Return per-residue metrics for the interface surface area

        Returns:
            The dictionary of errat metrics {errat_deviation, } mapped to arrays of shape (number_of_residues,)
        """
        per_residue_data = {}
        # pose_length = self.number_of_residues
        assembly_minimally_contacting = self.assembly_minimally_contacting
        self.log.debug(f'Starting {repr(self)} Errat')
        errat_start = time.time()
        _, per_residue_errat = assembly_minimally_contacting.errat(out_path=os.devnull)
        self.log.debug(f'Finished Errat, took {time.time() - errat_start:6f} s')
        per_residue_data['errat_deviation'] = per_residue_errat[:self.number_of_residues].tolist()

        return per_residue_data

    def _find_interface_residues_by_buried_surface_area(self):
        """Find interface_residues_by_entity_pair using buried surface area"""
        per_residue_bsa = self.per_residue_buried_surface_area()
        atom_contact_distance = default_atom_contact_distance
        interface_residue_bool = per_residue_bsa > metrics.bsa_tolerance
        entity_residue_coords = []
        coords_indexed_residues_by_entity = []
        entities = self.entities
        for entity in entities:
            entity_offset = entity.offset_index
            interface_residue_indices = np.flatnonzero(
                interface_residue_bool[entity_offset:entity_offset + entity.number_of_residues])
            interface_residues = entity.get_residues(indices=interface_residue_indices)
            # interface_coords = []
            # for residue in interface_residues:
            #     interface_coords.extend(residue.coords)
            interface_coords = np.concatenate([residue.coords for residue in interface_residues])
            entity_residue_coords.append(interface_coords)
            coords_indexed_interface_residues = [residue for residue in interface_residues for _ in residue.range]
            coords_indexed_residues_by_entity.append(coords_indexed_interface_residues)

        is_symmetric = self.is_symmetric()
        if is_symmetric:
            query_entity_residue_coords = [self.return_symmetric_coords(entity_interface_coords)
                                           for entity_interface_coords in entity_residue_coords]
        else:
            query_entity_residue_coords = entity_residue_coords

        found_entity_pairs = []
        for entity_idx, coords in enumerate(entity_residue_coords):
            number_of_coords = len(coords)
            interface_tree = BallTree(coords)
            for query_idx, entity_interface_coords in enumerate(query_entity_residue_coords):
                this_entity_index_pair = (entity_idx, query_idx)
                if this_entity_index_pair in found_entity_pairs:
                    continue
                else:  # Add both forward and reverse index versions
                    found_entity_pairs.extend((this_entity_index_pair, (query_idx, entity_idx)))

                entity2 = entities[query_idx]
                exclude_oligomeric_coord_indices = []
                if entity_idx == query_idx:
                    if not is_symmetric:
                        # Don't measure coords with itself if not symmetric
                        continue

                    if entity2.is_symmetric():  # and not oligomeric_interfaces:
                        # Remove oligomeric protomers (contains asu)
                        for model_number in self.oligomeric_model_indices.get(entity2):
                            exclude_oligomeric_coord_indices.extend(range(number_of_coords * model_number,
                                                                          number_of_coords * (model_number+1)))
                    else:  # Just remove the asu
                        exclude_oligomeric_coord_indices.extend(range(number_of_coords))
                # Create a mask for only coords of interest
                use_query_coords = np.ones(len(entity_interface_coords), dtype=bool)
                use_query_coords[exclude_oligomeric_coord_indices] = 0

                entity_coords_indexed_residues = coords_indexed_residues_by_entity[entity_idx]
                query_coords_indexed_residues = coords_indexed_residues_by_entity[query_idx]
                number_of_query_coords = len(query_coords_indexed_residues)
                interface_query = interface_tree.query_radius(
                    entity_interface_coords[use_query_coords], atom_contact_distance)
                # Divide by the asymmetric number of coords for the query to find the correct Residue instance
                contacting_pairs = [(entity_coords_indexed_residues[_entity_idx],
                                     query_coords_indexed_residues[query_idx % number_of_query_coords])
                                    for query_idx, entity_contacts in enumerate(interface_query.tolist())
                                    for _entity_idx in entity_contacts.tolist()]
                if entity_idx == query_idx:
                    asymmetric_contacting_pairs, found_pairs = [], set()
                    for pair1, pair2 in contacting_pairs:
                        # Only add to contacting pair if pair has never been observed
                        if (pair1, pair2) not in found_pairs:
                            asymmetric_contacting_pairs.append((pair1, pair2))
                        # Add both pair orientations, (1, 2) and (2, 1), regardless
                        found_pairs.update([(pair1, pair2), (pair2, pair1)])

                    contacting_pairs = asymmetric_contacting_pairs

                entity1_residues, entity2_residues = split_residue_pairs(contacting_pairs)

                if entity_idx == query_idx:
                    # Is the interface across a dimeric interface?
                    for residue in entity2_residues:  # entity2 usually has fewer residues, this might be quickest
                        if residue in entity1_residues:  # The interface is dimeric
                            # Include all residues found to only one side and move on
                            entity1_residues = sorted(set(entity1_residues).union(entity2_residues),
                                                      key=lambda res: res.number)
                            entity2_residues = []
                            break

                entity1 = entities[entity_idx]
                self.log.info(f'At Entity {entity1.name} | Entity {entity2.name} interface:'
                              f'\n\t{entity1.name} found residue numbers: '
                              f'{", ".join(str(r.number) for r in entity1_residues)}'
                              f'\n\t{entity2.name} found residue numbers: '
                              f'{", ".join(str(r.number) for r in entity2_residues)}')

                self._interface_residue_indices_by_entity_name_pair[(entity1.name, entity2.name)] = (
                    [r.index for r in entity1_residues],
                    [r.index for r in entity2_residues])

    def per_residue_interface_surface_area(self) -> dict[str, list[float]]:
        """Return per-residue metrics for the interface surface area

        Returns:
            The dictionary of metrics mapped to arrays of values with shape (number_of_residues,)
                Metrics include sasa_hydrophobic_complex, sasa_polar_complex, sasa_relative_complex,
                sasa_hydrophobic_bound, sasa_polar_bound, sasa_relative_bound
        """
        per_residue_data = {}
        pose_length = self.number_of_residues
        assembly_minimally_contacting = self.assembly_minimally_contacting

        # Perform SASA measurements
        if not assembly_minimally_contacting.sasa:
            assembly_minimally_contacting.get_sasa()
        assembly_asu_residues = assembly_minimally_contacting.residues[:pose_length]
        per_residue_data['sasa_hydrophobic_complex'] = [residue.sasa_apolar for residue in assembly_asu_residues]
        per_residue_data['sasa_polar_complex'] = [residue.sasa_polar for residue in assembly_asu_residues]
        per_residue_data['sasa_relative_complex'] = [residue.relative_sasa for residue in assembly_asu_residues]
        per_residue_sasa_unbound_apolar, per_residue_sasa_unbound_polar, per_residue_sasa_unbound_relative = [], [], []
        for entity in self.entities:
            if not entity.assembly.sasa:
                entity.assembly.get_sasa()
            oligomer_asu_residues = entity.assembly.residues[:entity.number_of_residues]
            per_residue_sasa_unbound_apolar.extend([residue.sasa_apolar for residue in oligomer_asu_residues])
            per_residue_sasa_unbound_polar.extend([residue.sasa_polar for residue in oligomer_asu_residues])
            per_residue_sasa_unbound_relative.extend([residue.relative_sasa for residue in oligomer_asu_residues])

        per_residue_data['sasa_hydrophobic_bound'] = per_residue_sasa_unbound_apolar
        per_residue_data['sasa_polar_bound'] = per_residue_sasa_unbound_polar
        per_residue_data['sasa_relative_bound'] = per_residue_sasa_unbound_relative

        return per_residue_data

    def _per_residue_interface_classification(
            self, relative_sasa_thresh: float = metrics.default_sasa_burial_threshold, **kwargs) -> pd.DataFrame:
        """Perform residue classification on the interface residues

        Args:
            relative_sasa_thresh: The relative area threshold that the Residue should fall below before it is considered
                'support'. Default cutoff percent is based on Levy, E. 2010

        Keyword Args:
            atom: bool = True - Whether the output should be generated for each atom.
                If False, will be generated for each Residue
            probe_radius: float = 1.4 - The radius which surface area should be generated

        Returns:
            The per-residue DataFrame with columns containing residue indices in level=0 and metrics in level=1
        """
        if not self.sasa:
            self.get_sasa(**kwargs)
        residue_bsa_df = {
            'bsa_total': self.per_residue_buried_surface_area(),
            **self.per_residue_relative_surface_area()
        }
        per_residue_df = pd.DataFrame(residue_bsa_df, index=pd.RangeIndex(self.number_of_residues)) \
            .unstack().to_frame().T.swaplevel(0, 1, axis=1)
        # Format ^ for classify_interface_residues
        return metrics.classify_interface_residues(per_residue_df, relative_sasa_thresh=relative_sasa_thresh)

    @property
    def core_residues(
            self, relative_sasa_thresh: float = metrics.default_sasa_burial_threshold, **kwargs
    ) -> list[Residue]:
        """Get the Residue instances that reside in the core of the interfaces

        Args:
            relative_sasa_thresh: The relative area threshold that the Residue should fall below before it is considered
                'core'. Default cutoff percent is based on Levy, E. 2010

        Keyword Args:
            atom: bool = True - Whether the output should be generated for each atom.
                If False, will be generated for each Residue
            probe_radius: float = 1.4 - The radius which surface area should be generated

        Returns:
            The core Residue instances
        """
        per_residue_df = self._per_residue_interface_classification(relative_sasa_thresh=relative_sasa_thresh, **kwargs)
        _residue_df = per_residue_df.loc[:, idx_slice[:, 'core']]

        return self.get_residues(indices=np.flatnonzero(_residue_df.values))

    @property
    def rim_residues(
            self, relative_sasa_thresh: float = metrics.default_sasa_burial_threshold, **kwargs
    ) -> list[Residue]:
        """Get the Residue instances that reside in the rim of the interface

        Args:
            relative_sasa_thresh: The relative area threshold that the Residue should fall below before it is considered
                'rim'. Default cutoff percent is based on Levy, E. 2010

        Keyword Args:
            atom: bool = True - Whether the output should be generated for each atom.
                If False, will be generated for each Residue
            probe_radius: float = 1.4 - The radius which surface area should be generated

        Returns:
            The rim Residue instances
        """
        per_residue_df = self._per_residue_interface_classification(relative_sasa_thresh=relative_sasa_thresh, **kwargs)
        _residue_df = per_residue_df.loc[:, idx_slice[:, 'rim']]

        return self.get_residues(indices=np.flatnonzero(_residue_df.values))

    @property
    def support_residues(
            self, relative_sasa_thresh: float = metrics.default_sasa_burial_threshold, **kwargs
    ) -> list[Residue]:
        """Get the Residue instances that support the interface

        Args:
            relative_sasa_thresh: The relative area threshold that the Residue should fall below before it is considered
                'support'. Default cutoff percent is based on Levy, E. 2010

        Keyword Args:
            atom: bool = True - Whether the output should be generated for each atom.
                If False, will be generated for each Residue
            probe_radius: float = 1.4 - The radius which surface area should be generated

        Returns:
            The support Residue instances
        """
        per_residue_df = self._per_residue_interface_classification(relative_sasa_thresh=relative_sasa_thresh, **kwargs)
        _residue_df = per_residue_df.loc[:, idx_slice[:, 'support']]

        return self.get_residues(indices=np.flatnonzero(_residue_df.values))

    def per_residue_relative_surface_area(self) -> dict[str, np.ndarray]:
        """Returns the relative solvent accessible surface area (SASA), per residue type compared to ideal three residue
        peptides, in both the bound (but not repacked) and complex states of the constituent Entity instances

        Returns:
            Mapping of 'sasa_relative_complex' and 'sasa_relative_bound' to array with the per-residue relative SASA
        """
        pose_length = self.number_of_residues
        assembly_minimally_contacting = self.assembly_minimally_contacting

        # Perform SASA measurements
        if not assembly_minimally_contacting.sasa:
            assembly_minimally_contacting.get_sasa()
        assembly_asu_residues = assembly_minimally_contacting.residues[:pose_length]
        per_residue_relative_sasa_complex = np.array([residue.relative_sasa for residue in assembly_asu_residues])
        per_residue_relative_sasa_unbound = []
        for entity in self.entities:
            if not entity.assembly.sasa:
                entity.assembly.get_sasa()
            oligomer_asu_residues = entity.assembly.residues[:entity.number_of_residues]
            per_residue_relative_sasa_unbound.extend([residue.relative_sasa for residue in oligomer_asu_residues])

        per_residue_relative_sasa_unbound = np.array(per_residue_relative_sasa_unbound)

        return {'sasa_relative_complex': per_residue_relative_sasa_complex,
                'sasa_relative_bound': per_residue_relative_sasa_unbound}

    def per_residue_buried_surface_area(self) -> np.ndarray:
        """Returns the buried surface area (BSA) as a result of all interfaces between Entity instances

        Returns:
            An array with the per-residue unbound solvent accessible surface area (SASA) minus the SASA of the complex
        """
        pose_length = self.number_of_residues
        assembly_minimally_contacting = self.assembly_minimally_contacting

        # Perform SASA measurements
        if not assembly_minimally_contacting.sasa:
            assembly_minimally_contacting.get_sasa()
        assembly_asu_residues = assembly_minimally_contacting.residues[:pose_length]
        per_residue_sasa_complex = np.array([residue.sasa for residue in assembly_asu_residues])
        per_residue_sasa_unbound = []
        for entity in self.entities:
            if not entity.assembly.sasa:
                entity.assembly.get_sasa()
            oligomer_asu_residues = entity.assembly.residues[:entity.number_of_residues]
            per_residue_sasa_unbound.extend([residue.sasa for residue in oligomer_asu_residues])

        per_residue_sasa_unbound = np.array(per_residue_sasa_unbound)

        return per_residue_sasa_unbound - per_residue_sasa_complex

    def per_residue_spatial_aggregation_propensity(self, distance: float = 5.0) -> dict[str, list[float]]:
        """Return per-residue spatial_aggregation for the complexed and unbound states. Positive values are more
        aggregation prone, while negative values are less prone

        Args:
            distance: The distance in angstroms to measure Atom instances in contact

        Returns:
            The dictionary of metrics mapped to arrays of values with shape (number_of_residues,)
                Metrics include 'spatial_aggregation_propensity' and 'spatial_aggregation_propensity_unbound'
        """
        per_residue_sap = {}
        pose_length = self.number_of_residues
        assembly_minimally_contacting = self.assembly_minimally_contacting
        assembly_minimally_contacting.spatial_aggregation_propensity_per_residue(distance=distance)
        assembly_asu_residues = assembly_minimally_contacting.residues[:pose_length]
        per_residue_sap['spatial_aggregation_propensity'] = \
            [residue.spatial_aggregation_propensity for residue in assembly_asu_residues]
        # Calculate sap for each Entity
        per_residue_sap_unbound = []
        for entity in self.entities:
            entity.assembly.spatial_aggregation_propensity_per_residue(distance=distance)
            oligomer_asu_residues = entity.assembly.residues[:entity.number_of_residues]
            # per_residue_sap_unbound.extend(entity.assembly.spatial_aggregation_propensity[:entity.number_of_residues])
            per_residue_sap_unbound.extend(
                [residue.spatial_aggregation_propensity for residue in oligomer_asu_residues])

        per_residue_sap['spatial_aggregation_propensity_unbound'] = per_residue_sap_unbound

        return per_residue_sap

    def get_interface(self, distance: float = 8.) -> Model:
        """Provide a view of the Pose interface by generating a Model containing only interface Residues

        Args:
            distance: The distance across the interface to query for Residue contacts

        Returns:
            The Structure containing only the Residues in the interface
        """
        if not self.is_symmetric():
            raise NotImplementedError('This function has not been properly converted to deal with non-symmetric poses')

        # interface_residues = []
        # interface_core_coords = []
        # for residues1, residues2 in self.interface_residues_by_entity_pair.values():
        #     if not residues1 and not residues2:  # no interface
        #         continue
        #     elif residues1 and not residues2:  # symmetric case
        #         interface_residues.extend(residues1)
        #         # This was useful when not doing the symmetrization below...
        #         # symmetric_residues = []
        #         # for _ in range(number_of_models):
        #         #     symmetric_residues.extend(residues1)
        #         # residues1_coords = np.concatenate([residue.coords for residue in residues1])
        #         # # Add the number of symmetric observed structures to a single new Structure
        #         # symmetric_residue_structure = Chain.from_residues(residues=symmetric_residues)
        #         # # symmetric_residues2_coords = self.return_symmetric_coords(residues1_coords)
        #         # symmetric_residue_structure.replace_coords(self.return_symmetric_coords(residues1_coords))
        #         # # use a single instance of the residue coords to perform a distance query against symmetric coords
        #         # residues_tree = BallTree(residues1_coords)
        #         # symmetric_query = residues_tree.query_radius(symmetric_residue_structure.coords, distance)
        #         # # symmetric_indices = [symmetry_idx for symmetry_idx, asu_contacts in enumerate(symmetric_query)
        #         # #                      if asu_contacts.any()]
        #         # # finally, add all correctly located, asu interface indexed symmetrical residues to the interface
        #         # coords_indexed_residues = symmetric_residue_structure.coords_indexed_residues
        #         # interface_residues.extend(set(coords_indexed_residues[sym_idx]
        #         #                               for sym_idx, asu_contacts in enumerate(symmetric_query)
        #         #                               if asu_contacts.any()))
        #     else:  # non-symmetric case
        #         interface_core_coords.extend([residue.cb_coords for residue in residues1])
        #         interface_core_coords.extend([residue.cb_coords for residue in residues2])
        #         interface_residues.extend(residues1), interface_residues.extend(residues2)

        # return Chain.from_residues(residues=sorted(interface_residues, key=lambda residue: residue.number))
        # interface_symmetry_mates = self.return_symmetric_copies(interface_asu_structure)
        # interface_coords = interface_asu_structure.coords
        # interface_cb_indices = interface_asu_structure.cb_indices
        # print('NUMBER of RESIDUES:', interface_asu_structure.number_of_residues,
        #       '\nNUMBER of CB INDICES', len(interface_cb_indices))
        # residue_number = interface_asu_structure.number_of_residues
        # [interface_asu_structure.cb_indices + (residue_number * model) for model in self.number_of_symmetry_mates]
        # symmetric_cb_indices = np.array([idx + (coords_length * model_num) for model_num in range(number_of_models)
        #                                  for idx in interface_asu_structure.cb_indices])
        # print('Number sym CB INDICES:\n', len(symmetric_cb_indices))
        # From the interface core, find the mean position to seed clustering
        entities_asu_com = self.center_of_mass
        # initial_interface_coords = self.return_symmetric_coords(entities_asu_com)
        initial_interface_coords = self.center_of_mass_symmetric_models
        # initial_interface_coords = self.return_symmetric_coords(np.array(interface_core_coords).mean(axis=0))

        # Get all interface residues and sort to ensure the asu is ordered and entity breaks can be found
        sorted_residues = sorted(self.interface_residues, key=lambda residue: residue.index)
        interface_asu_structure = Structure.from_residues(residues=sorted_residues)  # , chains=False, entities=False)
        # symmetric_cb_indices = self.make_indices_symmetric(interface_asu_structure.cb_indices)
        number_of_models = self.number_of_symmetry_mates
        coords_length = interface_asu_structure.number_of_atoms
        jump_sizes = [coords_length * model_num for model_num in range(number_of_models)]
        symmetric_cb_indices = [idx + jump_size for jump_size in jump_sizes
                                for idx in interface_asu_structure.cb_indices]
        symmetric_interface_coords = self.return_symmetric_coords(interface_asu_structure.coords)
        # index_cluster_labels = KMeans(n_clusters=self.number_of_symmetry_mates).fit_predict(symmetric_interface_coords)
        # symmetric_interface_cb_coords = symmetric_interface_coords[symmetric_cb_indices]
        # print('Number sym CB COORDS:\n', len(symmetric_interface_cb_coords))
        # initial_cluster_indices = [interface_cb_indices[0] + (coords_length * model_number)
        #                            for model_number in range(self.number_of_symmetry_mates)]
        # Fit a KMeans model to the symmetric interface cb coords
        kmeans_cluster_model = \
            KMeans(n_clusters=number_of_models, init=initial_interface_coords, n_init=1) \
            .fit(symmetric_interface_coords[symmetric_cb_indices])
        # kmeans_cluster_model = \
        #     KMeans(n_clusters=self.number_of_symmetry_mates, init=symmetric_interface_coords[initial_cluster_indices],
        #            n_init=1).fit(symmetric_interface_cb_coords)
        # Find the label where the asu is nearest too
        asu_label = kmeans_cluster_model.predict(entities_asu_com[None, :])  # <- add new first axis
        # asu_interface_labels = kmeans_cluster_model.predict(interface_asu_structure.cb_coords)

        # closest_interface_indices = np.where(index_cluster_labels == 0, True, False)
        # [False, False, False, True, True, True, True, True, True, False, False, False, False, False, ...]
        # symmetric_residues = interface_asu_structure.residues * self.number_of_symmetry_mates
        # [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, ...]
        # asu_index = np.median(asu_interface_labels)
        # grab the symmetric indices for a single interface cluster, matching spatial proximity to the asu_index
        # closest_asu_sym_cb_indices = symmetric_cb_indices[index_cluster_labels == asu_index]
        # index_cluster_labels = kmeans_cluster_model.labels_
        closest_asu_sym_cb_indices = \
            np.where(kmeans_cluster_model.labels_ == asu_label, symmetric_cb_indices, 0)
        # # find the cb indices of the closest interface asu
        # closest_asu_cb_indices = closest_asu_sym_cb_indices % coords_length
        # interface_asu_structure.coords_indexed_residues
        # find the model indices of the closest interface asu
        # print('Normal sym CB INDICES\n:', closest_asu_sym_cb_indices)

        # Create an array where each model is along axis=0 and each cb_index is along axis=1
        # The values in the array correspond to the symmetric cb_index if the cb_index is in the ASU
        # or 0 if the index isn't in the ASU
        # In this example the numbers aren't actually cb atom indices, they are cb residue indices...
        # [[ 0,  0, 2, 3, 4,  0, ...],
        #  [10,  0, 0, 0, 0,  0, ...],
        #  [ 0, 21, 0, 0, 0, 25, ...]].sum(axis=0) ->
        # [ 10, 21, 2, 3, 4, 25, ...]
        sym_cb_index_per_cb = closest_asu_sym_cb_indices.reshape((number_of_models, -1)).sum(axis=0)
        # print('FLATTENED CB INDICES to get MODEL\n:', sym_cb_index_per_cb)
        # Now divide that array by the number of atoms to get the model index
        sym_model_indices_per_cb = list(sym_cb_index_per_cb // coords_length)
        # print('FLOORED sym CB INDICES to get MODEL\n:', sym_model_indices_per_cb)
        symmetry_mate_index_symmetric_coords = symmetric_interface_coords.reshape((number_of_models, -1, 3))
        # print('RESHAPED SYMMETRIC COORDS SHAPE:', symmetry_mate_index_symmetric_coords.shape,
        #       '\nCOORDS length:', coords_length)
        # Get the cb coords from the symmetric coords that correspond to the asu indices
        closest_interface_coords = \
            np.concatenate([symmetry_mate_index_symmetric_coords[model_idx, residue.atom_indices]
                            for model_idx, residue in zip(sym_model_indices_per_cb, interface_asu_structure.residues)])
        # closest_symmetric_coords = \
        #     np.where(index_cluster_labels[:, None] == asu_index, symmetric_interface_coords, np.array([0.0, 0.0, 0.0]))
        # closest_interface_coords = \
        #     closest_symmetric_coords.reshape((self.number_of_symmetry_mates, interface_coords.shape[0], -1)).sum(axis=0)

        interface_asu_structure.coords = closest_interface_coords

        # Correct structure attributes
        # Get the new indices where there are different Entity instances
        entity_breaks = iter(self.entity_breaks)
        next_entity_break = next(entity_breaks)
        interface_residue_entity_breaks = []  # 0]
        for idx, residue in enumerate(sorted_residues):
            if residue.index > next_entity_break:
                interface_residue_entity_breaks.append(idx)
                try:
                    next_entity_break = next(entity_breaks)
                except StopIteration:
                    raise StopIteration(f'Reached the end of self.entity_breaks before sorted_residues ran out')
        else:  # Add the final idx
            interface_residue_entity_breaks.append(idx + 1)

        # For each residue, rename the chain_id according to the model it belongs to
        number_entities = self.number_of_entities
        model_chain_ids = list(chain_id_generator())[:number_of_models * number_entities]
        entity_idx = 0
        next_interface_asu_entity_break = interface_residue_entity_breaks[entity_idx]
        for model_idx, residue in zip(sym_model_indices_per_cb, interface_asu_structure.residues):
            if residue.index >= next_interface_asu_entity_break:  # Increment the entity_idx
                entity_idx += 1
                next_interface_asu_entity_break = interface_residue_entity_breaks[entity_idx]
            residue.chain_id = model_chain_ids[(model_idx * number_entities) + entity_idx]

        return interface_asu_structure

    def _find_interface_residue_pairs(self, entity1: Entity = None, entity2: Entity = None, distance: float = 8.,
                                      oligomeric_interfaces: bool = False) -> list[tuple[Residue, Residue]] | None:
        """Get pairs of Residues that have CB Atoms within a distance between two Entities

        Splits symmetric interfaces between self queries according to the unique set of asymmetric contacts

        Caution: Pose must have Coords representing all atoms! Residue pairs are found using CB indices from all atoms
        Symmetry aware. If symmetry is used, by default all atomic coordinates for entity2 are symmeterized.
        design_selector aware. Will remove interface residues if not active under the design selector

        Args:
            entity1: First entity to measure interface between
            entity2: Second entity to measure interface between
            distance: The distance to query the interface in Angstroms
            oligomeric_interfaces: Whether to query oligomeric interfaces

        Returns:
            A list of interface residue numbers across the interface
        """
        entity1_name = entity1.name
        self.log.debug(f'Entity {entity1_name} | Entity {entity2.name} interface query')
        # Get CB Atom Coordinates including CA coordinates for Gly residues
        entity1_indices = entity1.cb_indices
        entity2_indices = entity2.cb_indices

        if self._design_selector_atom_indices:  # Subtract the masked atom indices from the entity indices
            before = len(entity1_indices) + len(entity2_indices)
            entity1_indices = list(set(entity1_indices).intersection(self._design_selector_atom_indices))
            entity2_indices = list(set(entity2_indices).intersection(self._design_selector_atom_indices))
            self.log.debug('Applied design selection to interface identification. Number of indices before '
                           f'selection = {before}. Number after = {len(entity1_indices) + len(entity2_indices)}')

        if not entity1_indices or not entity2_indices:
            return None

        coords = self.coords
        if self.is_symmetric():  # Get the symmetric indices of interest
            entity2_indices = self.make_indices_symmetric(entity2_indices)
            # Solve for entity2_indices to query
            if entity1 == entity2:
                # Remove interactions with the asu model or intra-oligomeric models
                if entity1.is_symmetric() and not oligomeric_interfaces:
                    # Remove oligomeric protomers (contains asu)
                    remove_indices = self.get_oligomeric_atom_indices(entity1)
                    self.log.debug(f'Removing {len(remove_indices)} indices from symmetric query due to oligomer')
                else:  # Just remove asu
                    remove_indices = self.get_asu_atom_indices()
                # self.log.debug(f'Number of indices before removal of "self" indices: {len(entity2_indices)}')
                entity2_indices = list(set(entity2_indices).difference(remove_indices))
                # self.log.debug(f'Final indices remaining after removing "self": {len(entity2_indices)}')
            entity2_coords = self.symmetric_coords[entity2_indices]  # Get the symmetric indices from Entity 2
            sym_string = 'symmetric '
        elif entity1 == entity2:
            # Without symmetry, this can't be measured, unless intra-oligomeric contacts are desired
            self.log.warning(
                "Entities are the same, but not symmetric. The interface between them won't be detected")
            raise NotImplementedError(
                f"These entities shouldn't necessarily be equal. Did you mean to have symmetry={self.symmetry}? "
                f'If so, this issue needs to be addressed by expanding {entity1.__class__.__name__}.__eq__() method to '
                'more accurately reflect Structure object equality programmatically')
        else:
            sym_string = ''
            entity2_coords = coords[entity2_indices]  # Only get specified coordinate indices

        # Ensure the array is not empty
        if len(entity2_coords) == 0:
            return None

        # Construct CB tree for entity1 and query entity2 CBs for a distance less than a threshold
        entity1_coords = coords[entity1_indices]  # Only get specified coordinate indices
        entity1_tree = BallTree(entity1_coords)

        self.log.debug(f'Querying {len(entity1_indices)} CB residues in Entity {entity1.name} versus, '
                       f'{len(entity2_indices)} CB residues in {sym_string}Entity {entity2.name}')
        entity2_query = entity1_tree.query_radius(entity2_coords, distance)

        # Return pairs of Residue instances
        coords_indexed_residues = self.coords_indexed_residues
        # Get the modulus of the number_of_atoms to account for symmetry if used
        number_of_atoms = self.number_of_atoms
        # # Trying to make gains on the computations done below seems to result in trade offs in the flexibility of
        # # this function. The iteration over the entity2_query is an immmediate spot (i.e. .tolist() before iteration.
        # # Additional benefit could come from only indexing the Residue instances for pairs that are being returned
        # # after pairs are solved for
        # cb_coords_indexed_residues = self.cb_coords_indexed_residues
        # number_of_cb_atoms = len(cb_coords_indexed_residues)
        # cb_contacting_pairs_idx = [(entity1_idx,
        #                             entity2_idx % number_of_cb_atoms)
        #                            for entity2_idx, entity1_contacts in enumerate(entity2_query.tolist())
        #                            for entity1_idx in entity1_contacts.tolist()]
        # cb_contacting_pairs = [(cb_coords_indexed_residues[entity1_idx],
        #                         cb_coords_indexed_residues[entity2_idx % number_of_cb_atoms])
        #                        for entity2_idx, entity1_contacts in enumerate(entity2_query.tolist())
        #                        for entity1_idx in entity1_contacts.tolist()]
        # print(f"cb_contacting_pairs len={len(cb_contacting_pairs)}, {cb_contacting_pairs_idx[:5]}")
        # print(f"Found the cb_contacting_pairs residues of "
        #       f"{[(res1.index, res2.index) for res1, res2 in cb_contacting_pairs[:5]]}")
        # contacting_pairs_idx = [(entity1_indices[entity1_idx],
        #                          entity2_indices[entity2_idx] % number_of_atoms)
        #                         for entity2_idx, entity1_contacts in enumerate(entity2_query.tolist())
        #                         for entity1_idx in entity1_contacts.tolist()]
        contacting_pairs = [(coords_indexed_residues[entity1_indices[entity1_idx]],
                             coords_indexed_residues[entity2_indices[entity2_idx] % number_of_atoms])
                            for entity2_idx, entity1_contacts in enumerate(entity2_query.tolist())
                            for entity1_idx in entity1_contacts.tolist()]
        # print(f"Found the contacting_pairs residues of "
        #       f"{[(res1.index, res2.index) for res1, res2 in contacting_pairs[:5]]}")
        # print(f"contacting_pairs len={len(contacting_pairs)}, {contacting_pairs_idx[:5]}")
        if entity1 == entity2:  # Solve symmetric results for asymmetric contacts
            asymmetric_contacting_pairs, found_pairs = [], set()
            for pair1, pair2 in contacting_pairs:
                # Only add to contacting pair if pair has never been observed
                if (pair1, pair2) not in found_pairs:
                    asymmetric_contacting_pairs.append((pair1, pair2))
                # Add both pair orientations, (1, 2) and (2, 1), regardless
                found_pairs.update([(pair1, pair2), (pair2, pair1)])

            return asymmetric_contacting_pairs
        else:
            return contacting_pairs

    def get_interface_residues(
            self, entity1: Entity = None, entity2: Entity = None, **kwargs
    ) -> tuple[list[Residue] | list, list[Residue] | list]:
        """Get unique Residues across an interface between two Entities

        If the interface occurs between the same Entity which is non-symmetrically defined, but happens to occur along a
        dimeric axis of symmetry (evaluates to True when the same Residue is found on each side of the interface), then
        the residues are returned belonging to only one side of the interface

        Args:
            entity1: First Entity to measure interface between
            entity2: Second Entity to measure interface between

        Keyword Args:
            oligomeric_interfaces: bool = False - Whether to query oligomeric interfaces
            distance: float = 8. - The distance to measure Residues across an interface

        Returns:
            The Entity1 and Entity2 interface Residue instances
        """
        entity1_residues, entity2_residues = \
            split_residue_pairs(self._find_interface_residue_pairs(entity1=entity1, entity2=entity2, **kwargs))

        if not entity1_residues or not entity2_residues:
            self.log.info(f'Interface search at {entity1.name} | {entity2.name} found no interface residues')
            return [], []

        if entity1 == entity2:  # If symmetric query
            # Is the interface across a dimeric interface?
            for residue in entity2_residues:  # entity2 usually has fewer residues, this might be quickest
                if residue in entity1_residues:  # The interface is dimeric
                    # Include all residues found to only one side and move on
                    entity1_residues = sorted(set(entity1_residues).union(entity2_residues), key=lambda res: res.number)
                    entity2_residues = []
                    break
        self.log.info(f'At Entity {entity1.name} | Entity {entity2.name} interface:'
                      f'\n\t{entity1.name} found residue numbers: {", ".join(str(r.number) for r in entity1_residues)}'
                      f'\n\t{entity2.name} found residue numbers: {", ".join(str(r.number) for r in entity2_residues)}')

        return entity1_residues, entity2_residues

    def _find_interface_atom_pairs(self, entity1: Entity = None, entity2: Entity = None,
                                   distance: float = default_atom_contact_distance, residue_distance: float = None,
                                   **kwargs) -> list[tuple[int, int]] | list:
        """Get pairs of heavy atom indices that are within a distance at the interface between two Entities

        Caution: Pose must have Coordinates representing all atoms! Residue pairs are found using CB indices from all atoms

        Symmetry aware. If symmetry is used, by default all atomic coordinates for entity2 are symmeterized

        Args:
            entity1: First Entity to measure interface between
            entity2: Second Entity to measure interface between
            distance: The distance to measure contacts between atoms

        Keyword Args:
            by_distance: bool = False - Whether interface Residue instances should be found by inter-residue Cb distance
            residue_distance: float = 8. - The distance to measure Residues across an interface
            oligomeric_interfaces: bool = False - Whether to query oligomeric interfaces

        Sets:
            self._interface_residues_by_entity_pair (dict[tuple[str, str], tuple[list[int], list[int]]]):
                The Entity1/Entity2 interface mapped to the interface Residues

        Returns:
            The pairs of Atom indices for the interface
        """
        try:
            residues1, residues2 = self.interface_residues_by_entity_pair[(entity1, entity2)]
        except KeyError:  # When interface_residues haven't been set
            if residue_distance is not None:
                kwargs['distance'] = residue_distance
            self.find_and_split_interface(**kwargs)
            try:
                residues1, residues2 = self.interface_residues_by_entity_pair[(entity1, entity2)]
            except KeyError:
                raise DesignError(
                    f"{self._find_interface_atom_pairs.__name__} can't access 'interface_residues' as the Entity pair "
                    f"{entity1.name}, {entity2.name} hasn't located any 'interface_residues'")

        if not residues1:
            return []
        entity1_atom_indices: list[int] = []
        for residue in residues1:
            entity1_atom_indices.extend(residue.heavy_indices)

        if not residues2:  # check if the interface is a symmetric self dimer and all residues are in residues1
            # residues2 = residues1
            entity2_atom_indices = entity1_atom_indices
        else:
            entity2_atom_indices: list[int] = []
            for residue in residues2:
                entity2_atom_indices.extend(residue.heavy_indices)

        if self.is_symmetric():  # get all symmetric indices for entity2
            entity2_atom_indices = self.make_indices_symmetric(entity2_atom_indices)
            # No need to remove oligomeric indices as this procedure was done for residues
            query_coords = self.symmetric_coords[entity2_atom_indices]
        else:
            query_coords = self.coords[entity2_atom_indices]

        interface_atom_tree = BallTree(self.coords[entity1_atom_indices])
        atom_query = interface_atom_tree.query_radius(query_coords, distance)
        contacting_pairs = [(entity1_atom_indices[entity1_idx], entity2_atom_indices[entity2_idx])
                            for entity2_idx, entity1_contacts in enumerate(atom_query.tolist())
                            for entity1_idx in entity1_contacts.tolist()]
        return contacting_pairs

    def local_density_interface(self, distance: float = 12., atom_distance: float = None, **kwargs) -> float:
        """Returns the density of heavy Atoms neighbors within 'distance' Angstroms to Atoms in the Pose interface

        Args:
            distance: The cutoff distance with which Atoms should be included in local density
            atom_distance: The distance to measure contacts between atoms. By default, uses default_atom_count_distance

        Keyword Args:
            residue_distance: float = 8. - The distance to residue contacts in the interface. Uses the default if None
            oligomeric_interfaces: bool = False - Whether to query oligomeric interfaces

        Returns:
            The local atom density around the interface
        """
        if atom_distance:
            kwargs['distance'] = atom_distance

        interface_indices1, interface_indices2 = [], []
        for entity1, entity2 in self.interface_residues_by_entity_pair:
            atoms_indices1, atoms_indices2 = \
                split_number_pairs_and_sort(self._find_interface_atom_pairs(entity1=entity1, entity2=entity2, **kwargs))
            interface_indices1.extend(atoms_indices1)
            interface_indices2.extend(atoms_indices2)

        if not interface_indices1 and not interface_indices2:
            self.log.warning(f'No interface atoms located during {self.local_density_interface.__name__}')
            return 0.

        interface_indices = list(set(interface_indices1).union(interface_indices2))
        if self.is_symmetric():
            interface_coords = self.symmetric_coords[interface_indices]
        else:
            interface_coords = self.coords[interface_indices]

        interface_tree = BallTree(interface_coords)
        interface_counts = interface_tree.query_radius(interface_coords, distance, count_only=True)

        return interface_counts.mean()

    @property
    def fragment_info_by_entity_pair(self) -> dict[tuple[Entity, Entity], list[FragmentInfo]]:
        """Returns the FragmentInfo present as the result of structural overlap between pairs of Entity instances"""
        fragment_queries = {}
        for (entity1_name, entity2_name), fragment_info in self._fragment_info_by_entity_pair.items():
            entity1 = self.get_entity(entity1_name)
            entity2 = self.get_entity(entity2_name)
            fragment_queries[(entity1, entity2)] = fragment_info

        return fragment_queries

    def query_entity_pair_for_fragments(self, entity1: Entity = None, entity2: Entity = None,
                                        oligomeric_interfaces: bool = False, **kwargs):
        """For all found interface residues in an Entity/Entity interface, search for corresponding fragment pairs

        Args:
            entity1: The first Entity to measure for interface fragments
            entity2: The second Entity to measure for interface fragments
            oligomeric_interfaces: Whether to query oligomeric interfaces

        Keyword Args:
            by_distance: bool = False - Whether interface Residue instances should be found by inter-residue Cb distance
            distance: float = 8. - The distance to measure Residues across an interface

        Sets:
            self._fragment_info_by_entity_pair (dict[tuple[str, str], list[FragmentInfo]])
        """
        if (entity1.name, entity2.name) in self._fragment_info_by_entity_pair:
            # Due to asymmetry in fragment generation, (2, 1) isn't checked
            return

        try:
            residues1, residues2 = self.interface_residues_by_entity_pair[(entity1, entity2)]
        except KeyError:  # When interface_residues haven't been set
            self.find_and_split_interface(**kwargs)
            try:
                residues1, residues2 = self.interface_residues_by_entity_pair[(entity1, entity2)]
            except KeyError:
                raise DesignError(
                    f"{self.query_entity_pair_for_fragments.__name__} can't access 'interface_residues' as the Entity "
                    f"pair {entity1.name}, {entity2.name} hasn't located any 'interface_residues'")

        # Because of the way self.interface_residues_by_entity_pair is set, when there isn't an interface, a check on
        # residues1 is sufficient, however residues2 is empty with an interface present across a
        # non-oligomeric dimeric 2-fold
        if not residues1:
            self.log.debug(f'No residues at the {entity1.name} | {entity2.name} interface. Fragments not available')
            self._fragment_info_by_entity_pair[(entity1.name, entity2.name)] = []
            return

        frag_residues1 = entity1.get_fragment_residues(residues=residues1, fragment_db=self.fragment_db)
        if not residues2:  # entity1 == entity2 and not residues2:
            frag_residues2 = frag_residues1.copy()
        else:
            frag_residues2 = entity2.get_fragment_residues(residues=residues2, fragment_db=self.fragment_db)

        if not frag_residues1 or not frag_residues2:
            self.log.info(f'No fragments found at the {entity1.name} | {entity2.name} interface')
            self._fragment_info_by_entity_pair[(entity1.name, entity2.name)] = []
            return
        else:
            self.log.debug(f'At Entity {entity1.name} | Entity {entity2.name} interface:\n\t'
                           f'{entity1.name} has {len(frag_residues1)} interface fragments at residues: '
                           f'{",".join(map(str, [res.number for res in frag_residues1]))}\n\t'
                           f'{entity2.name} has {len(frag_residues2)} interface fragments at residues: '
                           f'{",".join(map(str, [res.number for res in frag_residues2]))}')

        if self.is_symmetric():
            # Even if entity1 == entity2, only need to expand the entity2 fragments due to surface/ghost frag mechanics
            skip_models = []
            if entity1 == entity2:
                if oligomeric_interfaces:  # Intra-oligomeric contacts are desired
                    self.log.info(f'Including oligomeric models: '
                                  f'{", ".join(map(str, self.oligomeric_model_indices.get(entity1)))}')
                elif entity1.is_symmetric():
                    # Remove interactions with the intra-oligomeric contacts (contains asu)
                    skip_models = self.oligomeric_model_indices.get(entity1)
                    self.log.info(f'Skipping oligomeric models: {", ".join(map(str, skip_models))}')
                # else:  # Probably a C1

            symmetric_frags2 = [self.return_symmetric_copies(residue) for residue in frag_residues2]
            frag_residues2.clear()
            for frag_mates in symmetric_frags2:
                frag_residues2.extend([frag for sym_idx, frag in enumerate(frag_mates) if sym_idx not in skip_models])
            self.log.debug(f'Entity {entity2.name} has {len(frag_residues2)} symmetric fragments')

        # For clash check, only the backbone and Cb are desired
        entity1_coords = entity1.backbone_and_cb_coords
        fragment_time_start = time.time()
        ghostfrag_surfacefrag_pairs = \
            find_fragment_overlap(frag_residues1, frag_residues2, clash_coords=entity1_coords)
        self.log.info(f'Found {len(ghostfrag_surfacefrag_pairs)} overlapping fragment pairs at the {entity1.name} | '
                      f'{entity2.name} interface')
        self.log.debug(f'Took {time.time() - fragment_time_start:.8f}s')

        self._fragment_info_by_entity_pair[(entity1.name, entity2.name)] = \
            create_fragment_info_from_pairs(ghostfrag_surfacefrag_pairs)

    @property
    def interface_fragment_residue_indices(self) -> list[int]:
        """The Residue indices where Fragment occurrences are observed"""
        try:
            return self._interface_fragment_residue_indices
        except AttributeError:
            interface_fragment_residue_indices = set()
            for frag_info in self.get_fragment_observations():
                interface_fragment_residue_indices.update((frag_info.mapped, frag_info.paired))

            self._interface_fragment_residue_indices = sorted(interface_fragment_residue_indices)

            return self._interface_fragment_residue_indices

    @interface_fragment_residue_indices.setter
    def interface_fragment_residue_indices(self, indices: Iterable[int]):
        self._interface_fragment_residue_indices = sorted(indices)

    def find_and_split_interface(self, by_distance: bool = False, **kwargs):
        """Locate interfaces regions for the designable entities and split into two contiguous interface residues sets

        Args:
            by_distance: Whether interface Residue instances should be found by inter-residue Cb distance

        Keyword Args:
            distance: float = 8. - The distance to measure Residues across an interface
            oligomeric_interfaces: bool = False - Whether to query oligomeric interfaces

        Sets:
            self._interface_residues_by_entity_pair (dict[tuple[str, str], tuple[list[int], list[int]]]):
                The Entity1/Entity2 interface mapped to the interface Residues
            self._interface_residues_by_interface (dict[int, list[int]]): Residue instances separated by
                interface topology
            self._interface_residues_by_interface_unique (dict[int, list[int]]): Residue instances separated by
                interface topology removing any dimeric duplicates
        """
        if self._interface_residue_indices_by_interface:
            self.log.debug("Interface residues weren't set as they're already set. If they've been changed, the "
                           "attribute 'interface_residues_by_interface' should be reset")
            return

        active_entities = self.active_entities
        self.log.debug('Find and split interface using active_entities: '
                       f'{", ".join(entity.name for entity in active_entities)}')

        if by_distance:
            if self.is_symmetric():
                entity_combinations = combinations_with_replacement(active_entities, 2)
            else:
                entity_combinations = combinations(active_entities, 2)

            entity_pair: tuple[Entity, Entity]
            for entity1, entity2 in entity_combinations:
                residues1, residues2 = self.get_interface_residues(entity1, entity2, **kwargs)
                self._interface_residue_indices_by_entity_name_pair[(entity1.name, entity2.name)] = (
                    [res.index for res in residues1],
                    [res.index for res in residues2]
                )
        else:
            self._find_interface_residues_by_buried_surface_area()

        self.check_interface_topology()

    def check_interface_topology(self):
        """From each pair of entities that share an interface, split the identified residues into two distinct groups.
        If an interface can't be composed into two distinct groups, raise DesignError

        Sets:
            self._interface_residues_by_interface (dict[int, list[int]]): Residue instances separated by
                interface topology
            self._interface_residues_by_interface_unique (dict[int, list[int]]): Residue instances separated by
                interface topology removing any dimeric duplicates
        """
        first_side, second_side = 0, 1
        first_interface_side = defaultdict(list)  # {}
        second_interface_side = defaultdict(list)  # {}
        # interface = {first_side: {}, second_side: {}}
        # Assume no symmetric contacts to start, i.e. [False, False]
        self_indication = [False, False]
        """Set to True if the interface side (1 or 2) contains self-symmetric contacts"""
        symmetric_dimer = {entity: False for entity in self.entities}
        terminate = False
        for (entity1, entity2), (residues1, residues2) in self.interface_residues_by_entity_pair.items():
            # if not entity_residues:
            if not residues1:  # No residues were found at this interface
                continue
            # Partition residues from each entity to the correct interface side
            # Check for any existing symmetry
            if entity1 == entity2:  # The query is with itself. Record as a self interaction
                _self = True
                if not residues2:  # The interface is a symmetric dimer and residues were removed from interface 2
                    symmetric_dimer[entity1] = True
                    # Set residues2 as residues1
                    residues2 = residues1  # .copy()  # Add residues1 to residues2
            else:
                _self = False

            if not first_interface_side:  # This is first interface observation
                # Add the pair to the dictionary in their indexed order
                first_interface_side[entity1].extend(residues1)  # residues1.copy()
                second_interface_side[entity2].extend(residues2)  # residues2.copy()
                # Indicate whether the interface is a self symmetric interface by marking side 2 with _self
                self_indication[second_side] = _self
            else:  # Interface already assigned, so interface observation >= 2
                # Need to check if either Entity is in either side before adding correctly
                if entity1 in first_interface_side:  # is Entity1 on the interface side 1?
                    # Is Entity1 in interface1 here as a result of self symmetric interaction?
                    if self_indication[first_side]:
                        # Yes. Ex4 - self Entity was added to index 0 while ASU added to index 1
                        # Add Entity1 to interface side 2
                        second_interface_side[entity1].extend(residues1)
                        # Add new Entity2 to interface side 1
                        first_interface_side[entity2].extend(residues2)  # residues2.copy()
                    else:  # Entities are properly indexed
                        # Add Entity1 to the first
                        first_interface_side[entity1].extend(residues1)
                        # Because of combinations with replacement Entity search, the second Entity isn't in
                        # interface side 2, UNLESS the Entity self interaction is on interface 1 (above if check)
                        # Therefore, add Entity2 to the second without checking for overwrite
                        second_interface_side[entity2].extend(residues2)  # residues2.copy()
                        # This can't happen, it would VIOLATE RULES
                        # if _self:
                        #     self_indication[second_side] = _self
                # Entity1 isn't in the first index. It may be in the second, it may not
                elif entity1 in second_interface_side:
                    # Yes. Ex5, add to interface side 2
                    second_interface_side[entity1].extend(residues1)
                    # Also, add Entity2 to the first side
                    # Entity 2 can't be in interface side 1 due to combinations with replacement check
                    first_interface_side[entity2].extend(residues2)  # residues2.copy()
                    if _self:  # Only modify if self is True, don't want to overwrite an existing True value
                        self_indication[first_side] = _self
                # If Entity1 is missing, check Entity2 to see if it has been identified yet
                # This is more likely from combinations with replacement
                elif entity2 in second_interface_side:
                    # Possible in an iteration Ex: (A:D) (C:D)
                    second_interface_side[entity2].extend(residues2)
                    # Entity 1 was not in first interface (from if #1), therefore we can set directly
                    first_interface_side[entity1].extend(residues1)  # residues1.copy()
                    # Ex3
                    if _self:  # Only modify if self is True, don't want to overwrite an existing True value
                        self_indication[first_side] = _self
                elif entity2 in first_interface_side:
                    # The first Entity wasn't found in either interface, but both interfaces are already set,
                    # therefore Entity pair isn't self, so the only way this works is if entity1 is further in the
                    # iterative process which is an impossible topology given combinations with replacement.
                    # Violates interface separation rules
                    second_interface_side[entity1] = False
                    terminate = True
                    break
                # Neither of our Entities were found, thus we would add 2 entities to each interface side, violation
                else:
                    first_interface_side[entity1] = second_interface_side[entity2] = False
                    terminate = True
                    break

            if len(first_interface_side) == 2 and len(second_interface_side) == 2 and all(self_indication):
                pass
            elif len(first_interface_side) == 1 or len(second_interface_side) == 1:
                pass
            else:
                terminate = True
                break

        if terminate:
            self.log.critical('The set of interfaces found during interface search generated a topologically '
                              'disallowed combination.\n\t %s\n This cannot be modeled by a simple split for residues '
                              'on either side while respecting the requirements of polymeric Entities. '
                              '%sPlease correct your design_selectors to reduce the number of Entities you are '
                              'attempting to design'
                              % (' | '.join(':'.join(entity.name for entity in interface_entities)
                                            for interface_entities in (first_interface_side, second_interface_side)),
                                 'Symmetry was set which may have influenced this unfeasible topology, you can try to '
                                 'set it False. ' if self.is_symmetric() else ''))
            raise DesignError('The specified interfaces generated a topologically disallowed combination')

        if not first_interface_side:
            # raise utils.DesignError('Interface was unable to be split because no residues were found on one side of '
            self.log.warning("The interface wasn't able to be split because no residues were found on one side. "
                             "Check that your input has an interface or your flags aren't too stringent")
            return

        for interface_number, entity_residues in enumerate((first_interface_side, second_interface_side), 1):
            _residue_indices = [residue.index for _, residues in entity_residues.items() for residue in residues]
            self._interface_residue_indices_by_interface[interface_number] = sorted(set(_residue_indices))

        if any(symmetric_dimer.values()):
            # For each entity, find the maximum residue observations on each interface side
            entity_observations = {entity: [0, 0] for entity, dimer in symmetric_dimer.items() if dimer}
            for interface_index, entity_residues in enumerate((first_interface_side, second_interface_side)):
                for entity, observations in entity_observations.items():
                    observations[interface_index] += len(entity_residues[entity])

            # Remove non-unique occurrences by entity if there are fewer observations
            for entity, observations in entity_observations.items():
                interface_obs1, interface_obs2 = entity_observations[entity]
                if interface_obs1 > interface_obs2:
                    second_interface_side.pop(entity)
                elif interface_obs1 < interface_obs2:
                    first_interface_side.pop(entity)
                elif len(entity_observations) == 1:
                    # This is a homo-dimer, by convention, get rid of the second side
                    second_interface_side.pop(entity)
                else:
                    raise SymmetryError(
                        f"Couldn't separate {entity.name} dimeric interface into unique residues. The number of "
                        f"residues in each interface is equal: {interface_obs1} == {interface_obs2}")

            # Perform the sort as without dimeric constraints
            for interface_number, entity_residues in enumerate((first_interface_side, second_interface_side), 1):
                _residue_indices = [residue.index for _, residues in entity_residues.items() for residue in residues]
                self._interface_residue_indices_by_interface_unique[interface_number] = sorted(set(_residue_indices))

        else:  # Just make a copy of the internal list... Avoids unforeseen issues if these are ever modified
            self._interface_residue_indices_by_interface_unique = \
                {number: residue_indices.copy()
                 for number, residue_indices in self._interface_residue_indices_by_interface.items()}

    @property
    def interface_residues_by_interface_unique(self) -> dict[int, list[Residue]]:
        """Keeps the Residue instances grouped by membership to each side of the interface.
        Residues are unique to one side of the interface
            Ex: {1: [Residue, ...], 2: [Residue, ...]}
        """
        interface_residues_by_interface_unique = {}
        residues = self.residues
        for number, residue_indices in self._interface_residue_indices_by_interface_unique.items():
            interface_residues_by_interface_unique[number] = [residues[idx] for idx in residue_indices]

        # self.log.debug('The unique interface (no dimeric duplication) is split as:'
        #                '\n\tInterface 1: %s\n\tInterface 2: %s'
        #                % tuple(','.join(f'{residue.number}{residue.chain_id}' for residue in residues)
        #                        for residues in interface_residues_by_interface_unique.values()))
        return interface_residues_by_interface_unique

    @property
    def interface_residues_by_interface(self) -> dict[int, list[Residue]]:
        """Keeps the Residue instances grouped by membership to each side of the interface.
        Residues can be duplicated on each side when interface contains a 2-fold axis of symmetry
            Ex: {1: [Residue, ...], 2: [Residue, ...]}
        """
        interface_residues_by_interface = {}
        residues = self.residues
        for number, residue_indices in self._interface_residue_indices_by_interface.items():
            interface_residues_by_interface[number] = [residues[idx] for idx in residue_indices]

        # self.log.debug('The interface is split as:\n\tInterface 1: %s\n\tInterface 2: %s'
        #                % tuple(','.join(f'{residue.number}{residue.chain_id}' for residue in residues)
        #                        for residues in self.interface_residues_by_interface.values()))
        return interface_residues_by_interface

    def calculate_secondary_structure(self):
        """Curate the secondary structure topology for each Entity

        Sets:
            self.ss_sequence_indices (list[int]):
                Index which indicates the Residue membership to the secondary structure type element sequence
            self.ss_type_sequence (list[str]):
                The ordered secondary structure type sequence which contains one character/secondary structure element
            self.split_interface_ss_elements (dict[int, list[int]]):
                Maps the interface number to a list of indices corresponding to the secondary structure type
                Ex: {1: [0, 0, 1, 2, ...] , 2: [9, 9, 9, 13, ...]]}
        """
        pose_secondary_structure = ''
        for entity in self.entities:  # self.active_entities:
            pose_secondary_structure += entity.secondary_structure

        # Increment a secondary structure index which changes with every secondary structure transition
        # Simultaneously, map the secondary structure type to an array of pose length
        ss_increment_index = 0
        ss_sequence_indices = [ss_increment_index]
        ss_type_sequence = [pose_secondary_structure[0]]
        for prior_ss_type, ss_type in zip(pose_secondary_structure[:-1], pose_secondary_structure[1:]):
            if prior_ss_type != ss_type:
                ss_increment_index += 1
                ss_type_sequence.append(ss_type)
            ss_sequence_indices.append(ss_increment_index)

        # Clear any information if it exists
        self.ss_sequence_indices.clear(), self.ss_type_sequence.clear()
        self.ss_sequence_indices.extend(ss_sequence_indices)
        self.ss_type_sequence.extend(ss_type_sequence)

        for number, residue_indices in self._interface_residue_indices_by_interface_unique.items():
            self.split_interface_ss_elements[number] = [ss_sequence_indices[idx] for idx in residue_indices]
        self.log.debug(f'Found interface secondary structure: {self.split_interface_ss_elements}')
        self.secondary_structure = pose_secondary_structure

    def calculate_fragment_profile(self, **kwargs):
        """Take the fragment_profile from each member Entity and combine

        Keyword Args:
            evo_fill: bool = False - Whether to fill missing positions with evolutionary profile values
            alpha: float = 0.5 - The maximum contribution of the fragment profile to use, bounded between (0, 1].
                0 means no use of fragments in the .profile, while 1 means only use fragments
        """
        #   keep_extras: bool = True - Whether to keep values for all that are missing data
        for (entity1, entity2), fragment_info in self.fragment_info_by_entity_pair.items():
            if fragment_info:
                self.log.debug(f'Query Pair: {entity1.name}, {entity2.name}'
                               f'\n\tFragment Info:{fragment_info}')
                entity1.add_fragments_to_profile(fragments=fragment_info, alignment_type='mapped')
                entity2.add_fragments_to_profile(fragments=fragment_info, alignment_type='paired')

        # The order of this and below could be switched by combining self.fragment_map too
        # Also, need to extract the entity.fragment_map to process_fragment_profile()
        fragments_available = False
        entities = self.entities
        for entity in entities:
            if entity.fragment_map:
                entity.simplify_fragment_profile(**kwargs)
                fragments_available = True
            else:
                entity.fragment_profile = Profile(
                    list(entity.create_null_profile(nan=True, zero_index=True).values()), dtype='fragment')

        if fragments_available:
            # # This assumes all values are present. What if they are not?
            # self.fragment_profile = concatenate_profile([entity.fragment_profile for entity in self.entities],
            #                                             start_at=0)
            # self.alpha = concatenate_profile([entity.alpha for entity in self.entities])
            fragment_profile = []
            alpha = []
            for entity in entities:
                fragment_profile.extend(entity.fragment_profile)
                alpha.extend(entity.alpha)
            fragment_profile = Profile(fragment_profile, dtype='fragment')
            self._alpha = entity._alpha  # Logic enforces entity is always referenced here
        else:
            alpha = [0 for _ in range(self.number_of_residues)]  # Reset the data
            fragment_profile = Profile(
                list(self.create_null_profile(nan=True, zero_index=True).values()), dtype='fragment')
        self.alpha = alpha
        self.fragment_profile = fragment_profile

    def get_fragment_observations(self, interface: bool = True) -> list[FragmentInfo] | list:
        """Return the fragment observations identified on the Pose for various types of tertiary structure interactions

        Args:
            interface: Whether to return fragment observations from only the Pose interface

        Returns:
            The fragment observations
        """
        # Ensure fragments are generated if they aren't already
        if interface:
            self.generate_interface_fragments()
        else:
            self.generate_fragments()

        interface_residues_by_entity_pair = self.interface_residues_by_entity_pair
        observations = []
        # {(ent1, ent2): [{mapped: res_num1, paired: res_num2, cluster: (int, int, int), match: score}, ...], ...}
        for entity_pair, fragment_info in self.fragment_info_by_entity_pair.items():
            if interface:
                if entity_pair not in interface_residues_by_entity_pair:
                    continue

            observations.extend(fragment_info)

        return observations

    @property
    def fragment_metrics_by_entity_pair(self) -> dict[tuple[Entity, Entity], dict[str, Any]]:
        """Returns the metrics from structural overlapping Fragment observations between pairs of Entity instances"""
        fragment_metrics = {}
        fragment_db = self.fragment_db
        for entity_pair, fragment_info in self.fragment_info_by_entity_pair.items():
            fragment_metrics[entity_pair] = fragment_db.calculate_match_metrics(fragment_info)

        return fragment_metrics

    def get_fragment_metrics(self, fragments: list[FragmentInfo] = None, total_interface: bool = True,
                             by_interface: bool = False, by_entity: bool = False,
                             entity1: Entity = None, entity2: Entity = None, **kwargs) -> dict[str, Any]:
        """Return fragment metrics from the Pose. Returns the entire Pose unless by_interface or by_entity is True

        Uses data from self.fragment_queries unless fragments are passed

        Args:
            fragments: A list of fragment observations
            total_interface: Return all fragment metrics for every interface found in the Pose
            by_interface: Return fragment metrics for each particular interface between Chain instances in the Pose
            by_entity: Return fragment metrics for each Entity found in the Pose
            entity1: The first Entity object to identify the interface if per_interface=True
            entity2: The second Entity object to identify the interface if per_interface=True

        Keyword Args:
            distance: float = 8. - The distance to measure Residues across an interface
            oligomeric_interfaces: bool = False - Whether to query oligomeric interfaces

        Returns:
            A mapping of the following metrics for the requested structural region -
                {'center_indices',
                 'total_indices',
                 'nanohedra_score',
                 'nanohedra_score_center',
                 'nanohedra_score_normalized',
                 'nanohedra_score_center_normalized',
                 'number_residues_fragment_total',
                 'number_residues_fragment_center',
                 'multiple_fragment_ratio',
                 'number_fragments_interface'
                 'percent_fragment_helix'
                 'percent_fragment_strand'
                 'percent_fragment_coil'
                 }
            Will include a single mapping if total_interface, a mapping for each interface if by_interface, and a
            mapping for each Entity if by_entity
        """

        if fragments is not None:
            fragment_db = self.fragment_db
            return fragment_db.format_fragment_metrics(fragment_db.calculate_match_metrics(fragments))

        fragment_metrics = self.fragment_metrics_by_entity_pair

        if by_interface:
            fragment_info = self.fragment_info_by_entity_pair
            # Check for either orientation as the final interface score will be the same
            if (entity1, entity2) not in fragment_info or (entity2, entity1) not in fragment_info:
                self.query_entity_pair_for_fragments(entity1=entity1, entity2=entity2, **kwargs)

            metric_d = deepcopy(fragment_metric_template)
            for query_pair, _metrics in fragment_metrics.items():
                # Check either orientation as the function query could vary from self.fragment_metrics
                if (entity1, entity2) in query_pair or (entity2, entity1) in query_pair:
                    if _metrics:
                        metric_d = self.fragment_db.format_fragment_metrics(_metrics)
                        break
            else:
                self.log.warning(f"Couldn't locate query metrics for Entity pair {entity1.name}, {entity2.name}")
        elif by_entity:
            metric_d = {}
            for query_pair, _metrics in fragment_metrics.items():
                if not _metrics:
                    continue
                for align_type, entity in zip(alignment_types, query_pair):
                    if entity not in metric_d:
                        metric_d[entity] = deepcopy(fragment_metric_template)

                    metric_d[entity]['center_indices'].update(_metrics[align_type]['center']['indices'])
                    metric_d[entity]['total_indices'].update(_metrics[align_type]['total']['indices'])
                    metric_d[entity]['nanohedra_score'] += _metrics[align_type]['total']['score']
                    metric_d[entity]['nanohedra_score_center'] += _metrics[align_type]['center']['score']
                    metric_d[entity]['multiple_fragment_ratio'] += _metrics[align_type]['multiple_ratio']
                    metric_d[entity]['number_fragments_interface'] += _metrics['total']['observations']
                    metric_d[entity]['percent_fragment_helix'] += _metrics[align_type]['index_count'][1]
                    metric_d[entity]['percent_fragment_strand'] += _metrics[align_type]['index_count'][2]
                    metric_d[entity]['percent_fragment_coil'] += _metrics[align_type]['index_count'][3] \
                        + _metrics[align_type]['index_count'][4] + _metrics[align_type]['index_count'][5]

            # Finally, tabulate based on the total for each Entity
            for entity, _metrics in metric_d.items():
                _metrics['number_residues_fragment_total'] = len(_metrics['total_indices'])
                _metrics['number_residues_fragment_center'] = len(_metrics['center_indices'])
                number_fragments_interface = _metrics['number_fragments_interface']
                _metrics['percent_fragment_helix'] /= number_fragments_interface
                _metrics['percent_fragment_strand'] /= number_fragments_interface
                _metrics['percent_fragment_coil'] /= number_fragments_interface
                try:
                    _metrics['nanohedra_score_normalized'] = \
                        _metrics['nanohedra_score'] / _metrics['number_residues_fragment_total']
                    _metrics['nanohedra_score_center_normalized'] = \
                        _metrics['nanohedra_score_center'] / _metrics['number_residues_fragment_center']
                except ZeroDivisionError:
                    self.log.warning(f'{self.name}: No interface residues were found. Is there an interface in your '
                                     f'design?')
                    _metrics['nanohedra_score_normalized'] = _metrics['nanohedra_score_center_normalized'] = 0.

        elif total_interface:  # For the entire interface
            metric_d = deepcopy(fragment_metric_template)
            for query_pair, _metrics in fragment_metrics.items():
                if not _metrics:
                    continue
                metric_d['center_indices'].update(
                    _metrics['mapped']['center']['indices'].union(_metrics['paired']['center']['indices']))
                metric_d['total_indices'].update(
                    _metrics['mapped']['total']['indices'].union(_metrics['paired']['total']['indices']))
                metric_d['nanohedra_score'] += _metrics['total']['total']['score']
                metric_d['nanohedra_score_center'] += _metrics['total']['center']['score']
                metric_d['multiple_fragment_ratio'] += _metrics['total']['multiple_ratio']
                metric_d['number_fragments_interface'] += _metrics['total']['observations']
                metric_d['percent_fragment_helix'] += _metrics['total']['index_count'][1]
                metric_d['percent_fragment_strand'] += _metrics['total']['index_count'][2]
                metric_d['percent_fragment_coil'] += _metrics['total']['index_count'][3] \
                    + _metrics['total']['index_count'][4] + _metrics['total']['index_count'][5]

            # Finally, tabulate based on the total
            metric_d['number_residues_fragment_total'] = len(metric_d['total_indices'])
            metric_d['number_residues_fragment_center'] = len(metric_d['center_indices'])
            total_observations = metric_d['number_fragments_interface'] * 2  # 2x observations in ['total']['index_count']
            try:
                metric_d['percent_fragment_helix'] /= total_observations
                metric_d['percent_fragment_strand'] /= total_observations
                metric_d['percent_fragment_coil'] /= total_observations
            except ZeroDivisionError:
                metric_d['percent_fragment_helix'] = metric_d['percent_fragment_strand'] = \
                    metric_d['percent_fragment_coil'] = 0.
            try:
                metric_d['nanohedra_score_normalized'] = \
                    metric_d['nanohedra_score'] / metric_d['number_residues_fragment_total']
                metric_d['nanohedra_score_center_normalized'] = \
                    metric_d['nanohedra_score_center']/metric_d['number_residues_fragment_center']
            except ZeroDivisionError:
                self.log.debug(f'{self.name}: No fragment residues were found')
                metric_d['nanohedra_score_normalized'] = metric_d['nanohedra_score_center_normalized'] = 0.

        else:  # For the entire Pose?
            raise NotImplementedError("There isn't a mechanism to return fragments for the mode specified")

        return metric_d

    def residue_processing(
        self, design_scores: dict[str, dict[str, float | str]], columns: list[str]
    ) -> dict[str, dict[int, dict[str, float | list]]]:
        """Process Residue Metrics from Rosetta score dictionary (One-indexed residues)

        Args:
            design_scores: {'001': {'buns': 2.0, 'res_energy_complex_15A': -2.71, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection_complex': '15A,21A,26A,35A,...'}, ...}
            columns: ['per_res_energy_complex_5', 'per_res_energy_1_unbound_5', ...]

        Returns:
            {'001': {15: {'type': 'T', 'energy': {'complex': -2.71, 'unbound': [-1.9, 0]}, 'fsp': 0., 'cst': 0.}, ...},
             ...}
        """
        # energy_template = {'complex': 0., 'unbound': 0., 'fsp': 0., 'cst': 0.}
        residue_template = {'energy': {'complex': 0., 'unbound': [0. for ent in self.entities], 'fsp': 0., 'cst': 0.}}
        pose_length = self.number_of_residues
        # adjust the energy based on pose specifics
        pose_energy_multiplier = self.number_of_symmetry_mates  # Will be 1 if not self.is_symmetric()
        entity_energy_multiplier = [entity.number_of_symmetry_mates for entity in self.entities]

        warn = False
        parsed_design_residues = {}
        for design, scores in design_scores.items():
            residue_data = {}
            for column in columns:
                if column not in scores:
                    continue
                metadata = column.strip('_').split('_')
                # remove chain_id in rosetta_numbering="False"
                # if we have enough chains, weird chain characters appear "per_res_energy_complex_19_" which mess up
                # split. Also numbers appear, "per_res_energy_complex_1161" which may indicate chain "1" or residue 1161
                residue_number = int(metadata[-1].translate(utils.digit_keeper()))
                if residue_number > pose_length:
                    if not warn:
                        warn = True
                        logger.warning(
                            'Encountered %s which has residue number > the pose length (%d). If this system is '
                            'NOT a large symmetric system and output_as_pdb_nums="true" was used in Rosetta '
                            'PerResidue SimpleMetrics, there is an error in processing that requires your '
                            'debugging. Otherwise, this is likely a numerical chain and will be treated under '
                            'that assumption. Always ensure that output_as_pdb_nums="true" is set'
                            % (column, pose_length))
                    residue_number = residue_number[:-1]
                if residue_number not in residue_data:
                    residue_data[residue_number] = deepcopy(residue_template)  # deepcopy(energy_template)

                metric = metadata[2]  # energy [or sasa]
                if metric != 'energy':
                    continue
                pose_state = metadata[-2]  # unbound or complex [or fsp (favor_sequence_profile) or cst (constraint)]
                entity_or_complex = metadata[3]  # 1,2,3,... or complex

                # use += because instances of symmetric residues from symmetry related chains are summed
                try:  # to convert to int. Will succeed if we have an entity value, ex: 1,2,3,...
                    entity = int(entity_or_complex) - ZERO_OFFSET
                    residue_data[residue_number][metric][pose_state][entity] += \
                        (scores.get(column, 0) / entity_energy_multiplier[entity])
                except ValueError:  # complex is the value, use the pose state
                    residue_data[residue_number][metric][pose_state] += (scores.get(column, 0) / pose_energy_multiplier)
            parsed_design_residues[design] = residue_data

        return parsed_design_residues

    def process_rosetta_residue_scores(self, design_scores: dict[str, dict[str, float | str]]) -> \
            dict[str, dict[int, dict[str, float | list]]]:
        """Process Residue Metrics from Rosetta score dictionary (One-indexed residues) accounting for symmetric energy

        Args:
            design_scores: {'001': {'buns': 2.0, 'res_energy_complex_15A': -2.71, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection_complex': '15A,21A,26A,35A,...'}, ...}

        Returns:
            The parsed design information where the outer key is the design alias, and the next key is the Residue.index
            where the corresponding information belongs. Only returns Residue metrics for those positions where metrics
            were taken. Example:
                {'001':
                    {15: {'complex': -2.71, 'bound': [-1.9, 0], 'unbound': [-1.9, 0],
                          'solv_complex': -2.71, 'solv_bound': [-1.9, 0], 'solv_unbound': [-1.9, 0],
                          'fsp': 0., 'cst': 0.},
                     ...},
                 ...}
        """
        res_slice = slice(0, 4)
        pose_length = self.number_of_residues
        # Adjust the energy based on pose specifics
        pose_energy_multiplier = self.number_of_symmetry_mates  # Will be 1 if not self.is_symmetric()
        entity_energy_multiplier = [entity.number_of_symmetry_mates for entity in self.entities]

        def get_template(): return deepcopy({
            'complex': 0., 'bound': [0. for _ in self.entities], 'unbound': [0. for _ in self.entities],
            'solv_complex': 0., 'solv_bound': [0. for _ in self.entities],
            'solv_unbound': [0. for _ in self.entities], 'fsp': 0., 'cst': 0.})

        warn_additional = True
        parsed_design_residues = {}
        for design, scores in design_scores.items():
            residue_data = defaultdict(get_template)
            for key, value in scores.items():
                if key[res_slice] != 'res_':
                    continue
                # key contains: 'per_res_energysolv_complex_15W' or 'per_res_energysolv_2_bound_415B'
                res, metric, entity_or_complex, *_ = metadata = key.strip('_').split('_')
                # metadata[1] is energy [or sasa]
                # metadata[2] is entity designation or complex such as '1','2','3',... or 'complex'

                # Take the "possibly" symmetric Rosetta residue index (one-indexed) and convert to python,
                # then take the modulus for the pose numbering
                try:
                    residue_index = (int(metadata[-1])-1) % pose_length
                except ValueError:
                    continue  # This is a residual metric
                # remove chain_id in rosetta_numbering="False"
                if metric == 'energysolv':
                    metric_str = f'solv_{metadata[-2]}'  # pose_state - unbound, bound, complex
                elif metric == 'energy':
                    metric_str = metadata[-2]  # pose_state - unbound, bound, complex
                else:  # Other residual such as sasa or something else old/new
                    if warn_additional:
                        warn_additional = False
                        logger.warning(f"Found additional metrics that aren't being processed. Ex {key}={value}")
                    continue

                # Use += because instances of symmetric residues from symmetry related chains are summed
                try:  # To convert to int. Will succeed if we have an entity as a string integer, ex: 1,2,3,...
                    entity = int(entity_or_complex) - ZERO_OFFSET
                except ValueError:  # 'complex' is the value, use the pose state
                    residue_data[residue_index][metric_str] += (value / pose_energy_multiplier)
                else:
                    residue_data[residue_index][metric_str][entity] += (value / entity_energy_multiplier[entity])

            parsed_design_residues[design] = residue_data

        return parsed_design_residues

    def rosetta_hbond_processing(self, design_scores: dict[str, dict]) -> dict[str, set[int]]:
        """Process Hydrogen bond Metrics from Rosetta score dictionary

        if rosetta_numbering="true" in .xml then use offset, otherwise, hbonds are PDB numbering

        Args:
            design_scores: {'001': {'buns': 2.0, 'per_res_energy_complex_15A': -2.71, ...,
                                    'yhh_planarity':0.885, 'hbonds_res_selection_complex': '15A,21A,26A,35A,...',
                                    'hbonds_res_selection_1_bound': '26A'}, ...}

        Returns:
            {'001': {34, 54, 67, 68, 106, 178}, ...}
        """
        pose_length = self.number_of_residues
        hbonds = {}
        for design, scores in design_scores.items():
            unbound_bonds, complex_bonds = set(), set()
            for column, value in scores.items():
                if 'hbonds_res_' not in column:  # if not column.startswith('hbonds_res_selection'):
                    continue
                meta_data = column.split('_')  # ['hbonds', 'res', 'selection', 'complex/interface_number', '[unbound]']
                # Offset rosetta numbering to python index and make asu index using the modulus
                parsed_hbond_indices = set((int(hbond)-1) % pose_length
                                           for hbond in value.split(',') if hbond != '')  # '' in case no hbonds
                # if meta_data[-1] == 'bound' and offset:  # find offset according to chain
                #     res_offset = offset[meta_data[-2]]
                #     parsed_hbonds = set(residue + res_offset for residue in parsed_hbonds)
                if meta_data[3] == 'complex':
                    complex_bonds = parsed_hbond_indices
                else:  # From another state
                    unbound_bonds = unbound_bonds.union(parsed_hbond_indices)
            if complex_bonds:  # 'complex', '1', '2'
                hbonds[design] = complex_bonds.difference(unbound_bonds)
            else:  # No hbonds were found in the complex
                hbonds[design] = complex_bonds

        return hbonds

    def generate_interface_fragments(self, oligomeric_interfaces: bool = False, **kwargs):
        """Generate fragments between the Pose interface(s). Finds interface(s) if not already available

        Args:
            oligomeric_interfaces: Whether to query oligomeric interfaces

        Keyword Args:
            by_distance: bool = False - Whether interface Residue instances should be found by inter-residue Cb distance
            distance: float = 8. - The distance to measure Residues across an interface
        """
        if not self._interface_residue_indices_by_interface:
            self.find_and_split_interface(oligomeric_interfaces=oligomeric_interfaces, **kwargs)

        if self.is_symmetric():
            entity_combinations = combinations_with_replacement(self.active_entities, 2)
        else:
            entity_combinations = combinations(self.active_entities, 2)

        entity_pair: Iterable[Entity]
        for entity_pair in entity_combinations:
            self.log.debug(f'Querying Entity pair: {", ".join(entity.name for entity in entity_pair)} '
                           f'for interface fragments')
            self.query_entity_pair_for_fragments(*entity_pair, oligomeric_interfaces=oligomeric_interfaces, **kwargs)

    def generate_fragments(self, **kwargs):
        """Generate fragments pairs between every possible Residue instance in the Pose

        Keyword Args:
            distance: float = 8. - The distance to query for neighboring fragments
        """
        for entity in self.active_entities:
            self.log.info(f'Querying Entity: {entity} for internal fragments')
            search_start_time = time.time()
            ghostfrag_surfacefrag_pairs = entity.find_fragments(**kwargs)
            self.log.info(f'Internal fragment search took {time.time() - search_start_time:8f}s')
            self._fragment_info_by_entity_pair[(entity.name, entity.name)] = \
                create_fragment_info_from_pairs(ghostfrag_surfacefrag_pairs)

    def write_fragment_pairs(self, out_path: AnyStr = os.getcwd(), multimodel: bool = False) -> AnyStr | None:
        """Write the fragments associated with the pose to disk

        Args:
            out_path: The path to the directory to output files to
            multimodel: Whether to write all fragments as a multimodel file. File written to "'out_path'/all_frags.pdb"

        Returns:
            The path to the written file if one was written
        """
        residues = self.residues
        ghost_frags = []
        clusters = []
        for entity_pair, fragment_info in self.fragment_info_by_entity_pair.items():
            for info in fragment_info:
                ijk = info.cluster
                clusters.append(ijk)
                # match_score = info.match
                aligned_residue = residues[info.mapped]
                ghost_frag = aligned_residue.ghost_fragments[ijk]
                ghost_frags.append(ghost_frag)

        putils.make_path(out_path)
        file_path = None
        if multimodel:
            file_path = os.path.join(out_path, 'all_frags.pdb')
            write_fragments_as_multimodel(ghost_frags, file_path)
        else:
            match_count = count(1)
            for ghost_frag, ijk in zip(ghost_frags, clusters):
                file_path = os.path.join(
                    out_path, '{}_{}_{}_fragment_match_{}.pdb'.format(*ijk, next(match_count)))
                ghost_frag.representative.write(out_path=file_path)

        # frag_file = Path(out_path, putils.frag_text_file)
        # frag_file.unlink(missing_ok=True)  # Ensure old file is removed before new write
        # for match_count, (ghost_frag, surface_frag, match_score) in enumerate(ghost_mono_frag_pairs, 1):
        #     ijk = ghost_frag.ijk
        #     fragment_pdb, _ = ghost_frag.fragment_db.paired_frags[ijk]
        #     trnsfmd_fragment = fragment_pdb.get_transformed_copy(*ghost_frag.transformation)
        # write_frag_match_info_file(ghost_frag=ghost_frag, matched_frag=surface_frag,
        #                                     overlap_error=z_value_from_match_score(match_score),
        #                                     match_number=match_count, out_path=out_path)
        return file_path

    def debug_pose(self, out_dir: AnyStr = os.getcwd(), tag: str = None):
        """Write out all Structure objects for the Pose PDB"""
        entity_debug_path = os.path.join(out_dir, f'{f"{tag}_" if tag else ""}POSE_DEBUG_Entities_{self.name}.pdb')
        with open(entity_debug_path, 'w') as f:
            available_chain_ids = chain_id_generator()
            for entity_idx, entity in enumerate(self.entities, 1):
                # f.write(f'REMARK 999   Entity {entity_idx} - ID {entity.name}\n')
                # entity.write(file_handle=f, chain_id=next(available_chain_ids))
                for chain_idx, chain in enumerate(entity.chains, 1):
                    f.write(f'REMARK 999   Entity {entity_idx} - ID {entity.name}   '
                            f'Chain {chain_idx} - ID {chain.chain_id}\n')
                    chain.write(file_handle=f, chain_id=next(available_chain_ids))

        debug_path = os.path.join(out_dir, f'{f"{tag}_" if tag else ""}POSE_DEBUG_{self.name}.pdb')
        assembly_debug_path = os.path.join(out_dir, f'{f"{tag}_" if tag else ""}POSE_DEBUG_Assembly_{self.name}.pdb')

        self.log.critical(f'Wrote debugging Pose Entities to: {entity_debug_path}')
        self.write(out_path=debug_path)
        self.log.critical(f'Wrote debugging Pose to: {debug_path}')
        self.write(assembly=True, out_path=assembly_debug_path)
        self.log.critical(f'Wrote debugging Pose assembly to: {assembly_debug_path}')
