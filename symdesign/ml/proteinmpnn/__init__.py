from __future__ import annotations

import logging
import math
import os
import time
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Container, Iterable, Sequence

import numpy as np
import torch

from .model import ProteinMPNN
from symdesign import utils
from .. import get_device_memory, sequence_nllloss
from symdesign.sequence import numerical_translation_alph1_unknown_bytes

proteinmpnn = Path(__file__)
logger = logging.getLogger(__name__)
default_translation_table = numerical_translation_alph1_unknown_bytes
mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'  # structure.utils.protein_letters_alph1_unknown
mpnn_alphabet_length = len(mpnn_alphabet)
MPNN_NULL_IDX = 20
# These seem to be the same, but when loading a second model, the amount is much less, 6291456 bytes or 6 M
vanilla_model_memory = ca_model_memory = 14_680_064
# This heuristic was decided based on successful runs and the batch_length at which they succeeded
batch_divisor = 400
# 6 - Works for 24 GiB mem with 6264 residue T input, 7 is too much
# 1 - Works for 5.61 GiB mem with 6264 residue T input, 2 is too much
# (5.66 GiB - 2.74 GiB) for sample() with 6264 residue T input
# (2.74 GiB - 1.25 GiB) for log_probs() with 6264 residue T input


def calculate_batch_length(model: ProteinMPNN, number_of_residues: int, element_memory: int = 4) -> int:
    """

    Args:
        model: The ProteinMPNN model
        number_of_residues: The number of residues used in the ProteinMPNN model
        element_memory: Where each element is np.int64, np.float32, etc.

    Returns:
        The size of the batch that can be completed for the ProteinMPNN model given its device
    """
    memory_constraint = get_device_memory(model.device)

    number_of_elements_available = memory_constraint // element_memory
    logger.debug(f'The number_of_elements_available is: {number_of_elements_available}')
    number_of_model_parameter_elements = sum([math.prod(param.size()) for param in model.parameters()])
    logger.debug(f'The number_of_model_parameter_elements is: {number_of_model_parameter_elements}')
    model_elements = number_of_model_parameter_elements
    # Use 5 as ideal CB is added by the model later with ca_only = False
    num_model_atoms = 5
    cartesian_space = 3
    model_elements += math.prod((number_of_residues, num_model_atoms, cartesian_space))  # X,
    model_elements += number_of_residues  # S.shape
    model_elements += number_of_residues  # chain_mask.shape
    model_elements += number_of_residues  # chain_encoding.shape
    model_elements += number_of_residues  # residue_idx.shape
    model_elements += number_of_residues  # mask.shape
    model_elements += number_of_residues  # residue_mask.shape
    model_elements += math.prod((number_of_residues, mpnn_alphabet_length))  # omit_AA_mask.shape
    model_elements += number_of_residues  # pssm_coef.shape
    model_elements += math.prod((number_of_residues, 20))  # pssm_bias.shape
    model_elements += math.prod((number_of_residues, 20))  # pssm_log_odds_mask.shape
    model_elements += number_of_residues  # tied_beta.shape
    model_elements += math.prod((number_of_residues, mpnn_alphabet_length))  # bias_by_res.shape
    logger.debug(f'The number of model_elements is: {model_elements}')

    number_of_batches = number_of_elements_available // model_elements
    batch_length = number_of_batches // batch_divisor
    if batch_length == 0:
        not_enough_memory = f"Can't find a device for {model} with enough memory to complete a single " \
                            f"batch of work with {number_of_residues} residues in the model"
        if model.device.type == 'cpu':
            raise RuntimeError(not_enough_memory)

        old_device = model.device
        # This won't work. Try to put the model on a new device
        max_memory = vanilla_model_memory
        for device_int in range(torch.cuda.device_count()):
            available_memory = get_device_memory(torch.device(device_int), free=True)
            if available_memory > max_memory:
                max_memory = available_memory
                device_id = device_int
        try:
            device: torch.device = torch.device(device_id)
        except UnboundLocalError:  # No device has memory greater than ProteinMPNN minimum required
            device = torch.device('cpu')

        if device == old_device:
            # Solve using gpu is stuck
            if device.type == 'cpu':
                # This hasn't been changed or device is cpu
                raise RuntimeError(not_enough_memory)
            else:
                # Try one more time ensuring cpu. This will be caught above if still not enough memory
                device = torch.device('cpu')

        # Set the device parameters
        model.to(device)
        model.device = device
        # Recurse
        return calculate_batch_length(model, number_of_residues, element_memory)

    return batch_length


PROTEINMPNN_DESIGN_BATCH_LEN = 6
PROTEINMPNN_SCORE_BATCH_LEN = 6
# # Use these options to bring down GPU memory when using Tensor instances with fixed size to a model
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
# pip install GPUtil
# from GPUtil import showUtilization as gpu_usage


class ProteinMPNNFactory(utils.factory.FactoryABC):
    """Return a ProteinMPNN instance by calling the Factory instance with the ProteinMPNN model name

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the named ProteinMPNN model
    """

    def __call__(
        self, model_name: str = 'v_48_020', backbone_noise: float = 0., ca_only: bool = False, **kwargs
    ) -> ProteinMPNN:
        """Returns the specified ProteinMPNN singleton

        Args:
            model_name: The name of the model to use from ProteinMPNN taking the format v_X_Y,
                where X is neighbor distance and Y is noise
            backbone_noise: The amount of backbone noise to add to the pose during design
            ca_only: Whether a minimal CA variant of the protein should be used for design calculations

        Returns:
            The instance of the initialized ProteinMPNN model
        """
        if ca_only:
            ca = '_ca'
            if model_name == 'v_48_030':
                logger.error(f"No such 'ca_only' model 'v_48_030'. Loading ca_only model 'v_48_020' (highest "
                             "backbone noise 'ca_only' model) instead")
                model_name = 'v_48_020'
            weights_dir = files(proteinmpnn) / 'ca_model_weights'
            required_memory = ca_model_memory
        else:
            ca = ''
            soluble = False  # Todo
            if soluble:
                weights_dir = files(proteinmpnn) / 'soluble_model_weights'
            else:
                weights_dir = files(proteinmpnn) / 'vanilla_model_weights'
            required_memory = vanilla_model_memory

        model_name_key = f'{model_name}{ca}_{backbone_noise}'
        model = self._instances.get(model_name_key)
        if model:
            return model
        else:  # Create a new ProteinMPNN model instance
            # if not self._models:  # Nothing initialized
            # Acquire an adequate computing device
            if torch.cuda.is_available():
                max_memory = required_memory
                for device_int in range(torch.cuda.device_count()):
                    available_memory = get_device_memory(torch.device(device_int), free=True)
                    if available_memory > max_memory:
                        max_memory = available_memory
                        device_id = device_int
                try:
                    device: torch.device = torch.device(device_id)
                except UnboundLocalError:  # No device has memory greater than ProteinMPNN minimum required
                    device = torch.device('cpu')
                else:
                    # Set the environment to use memory efficient cuda management
                    max_split = 1000
                    pytorch_conf = f'max_split_size_mb:{max_split},' \
                                   f'roundup_power2_divisions:4,' \
                                   f'garbage_collection_threshold:0.7'
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = pytorch_conf
                    logger.debug(f'Setting pytorch configuration:\n{pytorch_conf}\n'
                                 f'Result:{os.getenv("PYTORCH_CUDA_ALLOC_CONF")}')
            else:
                device = torch.device('cpu')

            checkpoint = torch.load(weights_dir / f'{model_name}.pt', map_location=device)
            hidden_dim = 128
            num_layers = 3
            with torch.no_grad():
                model = ProteinMPNN(num_letters=mpnn_alphabet_length,
                                    node_features=hidden_dim,
                                    edge_features=hidden_dim,
                                    hidden_dim=hidden_dim,
                                    num_encoder_layers=num_layers,
                                    num_decoder_layers=num_layers,
                                    augment_eps=backbone_noise,
                                    k_neighbors=checkpoint['num_edges'],
                                    ca_only=ca_only)
                model.to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                model.device = device
                model.model_name = model_name_key

            model.log.info(f"'{repr(model)}' on device '{device}' has "
                           f'{checkpoint["num_edges"]} edges, {checkpoint["noise_level"]} Angstroms of training '
                           f'noise, and {backbone_noise} Angstroms of backbone noise')
            # number_of_mpnn_model_parameters = sum([math.prod(param.size()) for param in model.parameters()])
            # logger.debug(f'The number of proteinmpnn model parameters is: {number_of_mpnn_model_parameters}')

            self._instances[model_name_key] = model

        return model


factory: Annotated[ProteinMPNNFactory,
                   'Calling this factory method returns the single instance of the ProteinMPNN class '
                   'located at the "source" keyword argument'] = \
    ProteinMPNNFactory()
"""Calling this factory method returns the single instance of the Database class located at the "source" keyword 
argument
"""
# kwargs = [X, S, mask, chain_M_pos, chain_mask, chain_encoding, residue_idx, omit_AA_mask,
#           tied_beta, pssm_coef, pssm_bias, pssm_log_odds_mask, bias_by_res]
# torch.int = torch.int32
# torch.long = torch.int64
# torch.float = torch.float32
# torch.double = torch.float64
dtype_map = dict(
    X=torch.float32,  # X,
    S=torch.long,  # S,
    randn=torch.float32,
    decoding_order=torch.long,
    chain_mask=torch.float32,  # residue_idx,
    chain_encoding=torch.long,  # mask,
    residue_idx=torch.long,  # chain_M_pos,  # residue_mask,
    mask=torch.float32,  # chain_mask,
    chain_M_pos=torch.float32,  # chain_encoding,
    omit_AA_mask=torch.float32,  # omit_AA_mask,
    pssm_coef=torch.float32,  # pssm_coef,
    pssm_bias=torch.float32,  # pssm_bias,
    pssm_log_odds_mask=torch.float32,  # pssm_log_odds_mask,
    tied_beta=torch.float32,  # tied_beta,
    bias_by_res=torch.float32,  # bias_by_res
    X_unbound=torch.float32,  # Add the special parameter X_unbound for measuring interfaces...
)
batch_params = list(dtype_map.keys())
# Remove tied_beta as this parameter is not "batched" in ProteinMPNN.tied_sample()
batch_params.pop(batch_params.index('tied_beta'))
gb_divisor = 1e9


# Used in batches
# X: np.ndarray = None,
# S: np.ndarray = None,
# chain_mask: np.ndarray = None,
# chain_encoding: np.ndarray = None,
# residue_idx: np.ndarray = None,
# mask: np.ndarray = None,
# chain_M_pos: np.ndarray = None,  # residue_mask
# omit_AA_mask: np.ndarray = None,
# pssm_coef: np.ndarray = None,
# pssm_bias: np.ndarray = None,
# pssm_log_odds_mask: np.ndarray = None,
# bias_by_res: np.ndarray = None,
# # tied_beta: np.ndarray = None,
# Not used in batches
# omit_AAs_np: np.ndarray = None, #
# bias_AAs_np: np.ndarray = None, #
# pssm_multi: np.ndarray = None, #
# pssm_log_odds_flag: np.ndarray = None #
# pssm_bias_flag: np.ndarray = None, #
# tied_pos: np.ndarray = None, #
# bias_by_res: np.ndarray = None, #
def batch_input(size: int = None, **kwargs) -> dict[str, np.ndarray]:
    """Set up all data for batches of proteinmpnn design

    Args:
        size: The number of inputs to use. If left blank, the size will be inferred from axis=0 of the X array

    Keyword Args:
        X: numpy.ndarray = None - The array specifying the parameter X
        X_unbound: numpy.ndarray = None - The array specifying the parameter X_unbound
        S: numpy.ndarray = None - The array specifying the parameter S
        randn: numpy.ndarray = None - The array specifying the parameter randn
        chain_mask: numpy.ndarray = None - The array specifying the parameter chain_mask
        chain_encoding: numpy.ndarray = None - The array specifying the parameter chain_encoding
        residue_idx: numpy.ndarray = None - The array specifying the parameter residue_idx
        mask: numpy.ndarray = None - The array specifying the parameter mask
        chain_M_pos: numpy.ndarray = None - The array specifying the parameter chain_M_pos (residue_mask)
        omit_AA_mask: numpy.ndarray = None - The array specifying the parameter omit_AA_mask
        pssm_coef: numpy.ndarray = None - The array specifying the parameter pssm_coef
        pssm_bias: numpy.ndarray = None - The array specifying the parameter pssm_bias
        pssm_log_odds_mask: numpy.ndarray = None - The array specifying the parameter pssm_log_odds_mask
        bias_by_res: numpy.ndarray = None - The array specifying the parameter bias_by_res
        tied_beta: numpy.ndarray = None - The array specifying the parameter tied_beta

    Returns:
        A dictionary with each of the ProteinMPNN parameters formatted in a batch
    """
    # This is my preferred name for the chain_M_pos...
    # residue_mask: (numpy.ndarray) = None - The array specifying the parameter residue_mask of ProteinMPNN
    if size is None:  # Use X as is
        X = kwargs.get('X')
        if X is None:
            raise ValueError(
                f"{batch_input.__name__} must pass keyword argument 'X' if argument 'size' is None")
        size = len(X)
    # else:
    #     X = np.tile(X, (size,) + (1,)*X.ndim)

    # Stack ProteinMPNN sequence design task in "batches"
    device_kwargs = {}
    for key in batch_params:
        param = kwargs.pop(key, None)
        if param is not None:
            device_kwargs[key] = np.tile(param, (size,) + (1,)*param.ndim)

    # Add all kwargs that were not accessed back to the return dictionary
    device_kwargs.update(**kwargs)
    return device_kwargs


# Used on device
# X: np.ndarray = None,
# S: np.ndarray = None,
# chain_mask: np.ndarray = None,
# chain_encoding: np.ndarray = None,
# residue_idx: np.ndarray = None,
# mask: np.ndarray = None,
# chain_M_pos: np.ndarray = None,  # residue_mask
# omit_AA_mask: np.ndarray = None,
# pssm_coef: np.ndarray = None,
# pssm_bias: np.ndarray = None,
# pssm_log_odds_mask: np.ndarray = None,
# bias_by_res: np.ndarray = None,
# tied_beta: np.ndarray = None,
# Not used on device
# omit_AAs_np = kwargs.get('omit_AAs_np', None)
# bias_AAs_np = kwargs.get('bias_AAs_np', None)
# pssm_multi = kwargs.get('pssm_multi', None)
# pssm_log_odds_flag = kwargs.get('pssm_log_odds_flag', None)
# pssm_bias_flag = kwargs.get('pssm_bias_flag', None)
# tied_pos = kwargs.get('tied_pos', None)
# bias_by_res = kwargs.get('bias_by_res', None)
def to_device(device: torch.device | int | str, **kwargs) -> dict[str, torch.Tensor]:
    """Set up all data to torch.Tensors for ProteinMPNN design

    Args:
        device: The device to load tensors to

    Keyword Args:
        X: numpy.ndarray = None - The array specifying the parameter X
        X_unbound: numpy.ndarray = None - The array specifying the parameter X_unbound
        S: numpy.ndarray = None - The array specifying the parameter S
        randn: numpy.ndarray = None - The array specifying the parameter randn
        chain_mask: numpy.ndarray = None - The array specifying the parameter chain_mask
        chain_encoding: numpy.ndarray = None - The array specifying the parameter chain_encoding
        residue_idx: numpy.ndarray = None - The array specifying the parameter residue_idx
        mask: numpy.ndarray = None - The array specifying the parameter mask
        chain_M_pos: numpy.ndarray = None - The array specifying the parameter chain_M_pos (residue_mask)
        omit_AA_mask: numpy.ndarray = None - The array specifying the parameter omit_AA_mask
        pssm_coef: numpy.ndarray = None - The array specifying the parameter pssm_coef
        pssm_bias: numpy.ndarray = None - The array specifying the parameter pssm_bias
        pssm_log_odds_mask: numpy.ndarray = None - The array specifying the parameter pssm_log_odds_mask
        bias_by_res: numpy.ndarray = None - The array specifying the parameter bias_by_res
        tied_beta: numpy.ndarray = None - The array specifying the parameter tied_beta

    Returns:
        The torch.Tensor ProteinMPNN parameters
    """
    if device is None:
        raise ValueError('Must provide the desired device to load proteinmpnn')
    logger.debug(f'Loading ProteinMPNN parameters to device: {device}')

    # Convert all numpy arrays to pytorch
    device_kwargs = {}
    for key, dtype in dtype_map.items():
        param = kwargs.pop(key, None)
        if param is not None:
            device_kwargs[key] = torch.from_numpy(param).to(dtype=dtype, device=device)

    # Add all kwargs that were not accessed back to the return dictionary
    device_kwargs.update(**kwargs)
    return device_kwargs


@torch.no_grad()  # Ensure no gradients are produced
def setup_pose_batch(batch_length: int, device: torch.device | int | str, **parameters) \
        -> dict[str, np.ndarray | torch.Tensor]:
    """

    Args:
        batch_length: The length the batch to set up
        device: The device used for batch calculations

    Returns:
        A mapping of ProteinMPNN inference containers formatted as a batch and loaded on the specified device
    """
    # batch_length = batch_slice.stop - batch_slice.start
    # Create batch_length fixed parameter data which are the same across poses
    batch_parameters: dict[str, np.ndarray | torch.Tensor] = batch_input(size=batch_length, **parameters)
    # Move fixed data structures to the model device
    # Update parameters as some are not transferred to the identified device
    batch_parameters.update(to_device(device, **batch_parameters))

    return batch_parameters


# @batch_calculation(size=size, batch_length=batch_length,
#                    setup=setup_pose_batch,
#                    compute_failure_exceptions=(RuntimeError, np.core._exceptions._ArrayMemoryError))
def batch_design(
    batch_slice: slice,
    model: ProteinMPNN,
    X: torch.Tensor = None,
    randn: torch.Tensor = None,
    S: torch.Tensor = None,
    chain_mask: torch.Tensor = None,
    chain_encoding: torch.Tensor = None,
    residue_idx: torch.Tensor = None,
    mask: torch.Tensor = None,
    temperatures: Sequence[float] = (0.1,),
    pose_length: int = None,
    bias_by_res: torch.Tensor = None,
    tied_pos: Iterable[Container] = None,
    X_unbound: torch.Tensor = None,
    **batch_parameters
) -> dict[str, np.ndarray]:
    """Perform ProteinMPNN design tasks on input that is split into batches

    Args:
        batch_slice:
        model:
        X:
        randn:
        S:
        chain_mask:
        chain_encoding:
        residue_idx:
        mask:
        temperatures:
        pose_length:
        bias_by_res:
        tied_pos:
        X_unbound:

    Returns:
        A mapping of the key describing a metric to the array of corresponding values,
            i.e. sequences, complex_sequence_loss, and unbound_sequence_loss
    """
    # X = batch_parameters.pop('X', None)
    # S = batch_parameters.pop('S', None)
    # chain_mask = batch_parameters.pop('chain_mask', None)
    # chain_encoding = batch_parameters.pop('chain_encoding', None)
    # residue_idx = batch_parameters.pop('residue_idx', None)
    # mask = batch_parameters.pop('mask', None)
    # randn = batch_parameters.pop('randn', None)
    # # omit_AAs_np = batch_parameters.get('omit_AAs_np', None)
    # # bias_AAs_np = batch_parameters.get('bias_AAs_np', None)
    residue_mask = batch_parameters.pop('chain_M_pos', None)  # name change makes more sense
    # # omit_AA_mask = batch_parameters.get('omit_AA_mask', None)
    # # pssm_coef = batch_parameters.get('pssm_coef', None)
    # # pssm_bias = batch_parameters.get('pssm_bias', None)
    # # pssm_multi = batch_parameters.get('pssm_multi', None)
    # # pssm_log_odds_flag = batch_parameters.get('pssm_log_odds_flag', None)
    # # pssm_log_odds_mask = batch_parameters.get('pssm_log_odds_mask', None)
    # # pssm_bias_flag = batch_parameters.get('pssm_bias_flag', None)
    # tied_pos = batch_parameters.pop('tied_pos', None)
    # # tied_beta = batch_parameters.pop('tied_beta', None)
    # # bias_by_res = batch_parameters.get('bias_by_res', None)

    actual_batch_length = batch_slice.stop - batch_slice.start
    # # Clone the data from the sequence tensor so that it can be set with the null token below
    # S_design = S.detach().clone()
    if pose_length is None:
        batch_length, pose_length, *_ = S.shape
    else:
        batch_length, *_ = S.shape

    if actual_batch_length != batch_length:
        # Slice these for the last iteration
        X = X[:actual_batch_length]  # , None)
        chain_mask = chain_mask[:actual_batch_length]  # , None)
        chain_encoding = chain_encoding[:actual_batch_length]  # , None)
        residue_idx = residue_idx[:actual_batch_length]  # , None)
        mask = mask[:actual_batch_length]  # , None)
        bias_by_res = bias_by_res[:actual_batch_length]  # , None)
        randn = randn[:actual_batch_length]
        residue_mask = residue_mask[:actual_batch_length]
        S = S[:actual_batch_length]  # , None)
        # S_design = S_design[:actual_batch_length]  # , None)
        # Unpack, unpacked keyword args
        omit_AA_mask = batch_parameters.get('omit_AA_mask')
        pssm_coef = batch_parameters.get('pssm_coef')
        pssm_bias = batch_parameters.get('pssm_bias')
        pssm_log_odds_mask = batch_parameters.get('pssm_log_odds_mask')
        # Set keyword args
        batch_parameters['omit_AA_mask'] = omit_AA_mask[:actual_batch_length]
        batch_parameters['pssm_coef'] = pssm_coef[:actual_batch_length]
        batch_parameters['pssm_bias'] = pssm_bias[:actual_batch_length]
        batch_parameters['pssm_log_odds_mask'] = pssm_log_odds_mask[:actual_batch_length]
        try:
            X_unbound = X_unbound[:actual_batch_length]  # , None)
        except TypeError:  # Can't slice NoneType
            pass

    # # Use the sequence as an unknown token then guess the probabilities given the remaining
    # # information, i.e. the sequence and the backbone
    # S_design_null[residue_mask.type(torch.bool)] = MPNN_NULL_IDX
    chain_residue_mask = chain_mask * residue_mask

    batch_sequences = []
    _per_residue_complex_sequence_loss = []
    _per_residue_unbound_sequence_loss = []
    number_of_temps = len(temperatures)
    for temp_idx, temperature in enumerate(temperatures):
        sample_start_time = time.time()
        if tied_pos is None:
            sample_dict = model.sample(X, randn, S, chain_mask, chain_encoding, residue_idx, mask,
                                       chain_M_pos=residue_mask, temperature=temperature, bias_by_res=bias_by_res,
                                       **batch_parameters)
        else:
            sample_dict = model.tied_sample(X, randn, S, chain_mask, chain_encoding, residue_idx,
                                            mask, chain_M_pos=residue_mask, temperature=temperature,
                                            bias_by_res=bias_by_res, tied_pos=tied_pos, **batch_parameters)
        model.log.info(f'Sample calculation took {time.time() - sample_start_time:8f}s')

        # Format outputs - All have at lease shape (batch_length, model_length,)
        S_sample = sample_dict['S']
        _batch_sequences = S_sample[:, :pose_length]
        # Check for null sequence output
        null_seq = _batch_sequences == 20
        # null_indices = np.argwhere(null_seq == 1)
        # if null_indices.nelement():  # Not an empty tensor...
        # Find the indices that are null on each axis
        null_design_indices, null_sequence_indices = torch.nonzero(null_seq == 1, as_tuple=True)
        if null_design_indices.nelement():  # Not an empty tensor...
            model.log.warning(f'Found null sequence output... Resampling selected positions')
            model.log.debug(f'At sequence position(s): {null_sequence_indices}')
            null_seq = (False,)
            sampled_probs = sample_dict['probs'].cpu()
            while not all(null_seq):  # null_indices.nelement():  # Not an empty tensor...
                # _decoding_order = decoding_order.cpu().numpy()[:, :pose_length] / 12  # Hard coded symmetry divisor
                # # (batch_length, pose_length)
                # print(f'Shape of null_seq: {null_seq.shape}')
                # print(f'Shape of _decoding_order: {_decoding_order.shape}')
                # print(f'Shape of _batch_sequences: {_batch_sequences.shape}')
                # print(f'Found the decoding sites with a null output: {_decoding_order[null_seq]}')
                # print(f'Found the probabilities with a null output: {_probabilities[null_seq]}')
                # print(_batch_sequences.numpy()[_decoding_order])
                # _probabilities = sample_dict['probs']  # .cpu().numpy()[:, :pose_length]
                # _probabilities with shape (batch_length, model_length, mpnn_alphabet_length)
                new_amino_acid_types = \
                    torch.multinomial(sampled_probs[null_design_indices, null_sequence_indices],
                                      1).squeeze(-1)
                # _batch_sequences[null_indices] = new_amino_acid_type
                # new_amino_acid_type = torch.multinomial(sample_dict['probs'][null_seq], 1)
                # _batch_sequences[null_seq] = new_amino_acid_type
                null_seq = new_amino_acid_types != 20
                # null_seq = _batch_sequences == 20
                # null_indices = np.argwhere(null_seq == 1)
            else:
                # Set the
                _batch_sequences[null_design_indices, null_sequence_indices] = new_amino_acid_types
            # model.log.debug('Fixed null sequence elements')

        decoding_order = sample_dict['decoding_order']
        # decoding_order_out = decoding_order  # When using the same decoding order for all
        log_probs_start_time = time.time()
        if X_unbound is not None:
            # unbound_log_prob_start_time = time.time()
            # logger.critical(f'Starting unbound calc: '
            #                 f'available memory={get_device_memory(model.device)/gb_divisor}')
            unbound_log_probs = \
                model(X_unbound, S_sample, mask, chain_residue_mask, residue_idx, chain_encoding,
                      None,  # This argument is provided but with below args, is not used
                      use_input_decoding_order=True, decoding_order=decoding_order)
            # logger.critical(f'After unbound calc: '
            #                 f'available memory={get_device_memory(model.device)/gb_divisor}')
            _per_residue_unbound_sequence_loss.append(
                sequence_nllloss(_batch_sequences, unbound_log_probs[:, :pose_length]).cpu().numpy())
            # logger.debug(f'Unbound log probabilities calculation took '
            #              f'{time.time() - unbound_log_prob_start_time:8f}s')

        # logger.critical(f'Starting bound calc: '
        #                 f'available memory={get_device_memory(model.device) / gb_divisor}')
        complex_log_probs = \
            model(X, S_sample, mask, chain_residue_mask, residue_idx, chain_encoding,
                  None,  # This argument is provided but with below args, is not used
                  use_input_decoding_order=True, decoding_order=decoding_order)
        # logger.critical(f'After bound calc: '
        #                 f'available memory={get_device_memory(model.device) / gb_divisor}')
        # complex_log_probs is
        # tensor([[[-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772],
        #          [-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772],
        #          [-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772],
        #          ...,
        #          [-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772],
        #          [-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772],
        #          [-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772]],
        #         [[-2.6934, -4.0610, -2.6506, ..., -4.2404, -3.4620, -4.8641],
        #          [-2.8753, -4.3959, -2.4042,  ..., -4.4922, -3.5962, -5.1403],
        #          [-2.5235, -4.0181, -2.7738,  ..., -4.2454, -3.4768, -4.8088],
        #          ...,
        #          [-3.4500, -4.4373, -3.7814,  ..., -5.1637, -4.6107, -5.2295],
        #          [-0.9690, -4.9492, -3.9373,  ..., -2.0154, -2.2262, -4.3334],
        #          [-3.1118, -4.3809, -3.8763,  ..., -4.7145, -4.1524, -5.3076]]])
        # Score the redesigned structure-sequence
        # mask_for_loss = chain_mask_and_mask*residue_mask
        # batch_scores = sequence_nllloss(S_sample, complex_log_probs, mask_for_loss, per_residue=False)
        # batch_scores is
        # tensor([2.1039, 2.0618, 2.0802, 2.0538, 2.0114, 2.0002], device='cuda:0')
        _per_residue_complex_sequence_loss.append(
            sequence_nllloss(_batch_sequences, complex_log_probs[:, :pose_length]).cpu().numpy())
        model.log.info(f'Log probabilities score calculation took {time.time() - log_probs_start_time:8f}s')
        batch_sequences.append(_batch_sequences.cpu())

    # Reshape data structures to have shape (batch_length, number_of_temperatures, pose_length)
    _residue_indices_of_interest = residue_mask[:, :pose_length].cpu().numpy().astype(bool)
    sequences = np.concatenate(batch_sequences, axis=1).reshape(actual_batch_length, number_of_temps, pose_length)
    complex_sequence_loss =\
        np.concatenate(_per_residue_complex_sequence_loss, axis=1) \
        .reshape(actual_batch_length, number_of_temps, pose_length)
    if X_unbound is not None:
        unbound_sequence_loss = \
            np.concatenate(_per_residue_unbound_sequence_loss, axis=1) \
            .reshape(actual_batch_length, number_of_temps, pose_length)
    else:
        unbound_sequence_loss = np.empty_like(complex_sequence_loss)

    return {'sequences': sequences,
            'proteinmpnn_loss_complex': complex_sequence_loss,
            'proteinmpnn_loss_unbound': unbound_sequence_loss,
            'design_indices': _residue_indices_of_interest}


def batch_score(
    batch_slice: slice,
    model: ProteinMPNN,
    X: torch.Tensor = None,
    S: torch.Tensor = None,
    chain_mask: torch.Tensor = None,
    chain_encoding: torch.Tensor = None,
    residue_idx: torch.Tensor = None,
    mask: torch.Tensor = None,
    pose_length: int = None,
    X_unbound: torch.Tensor = None,
    chain_M_pos: torch.Tensor = None,
    residue_mask: torch.Tensor = None,
    randn: torch.Tensor = None,
    decoding_order: torch.Tensor = None,
    **batch_parameters
) -> dict[str, np.ndarray]:
    """Perform ProteinMPNN design tasks on input that is split into batches

    Args:
        batch_slice:
        model:
        X:
        S:
        chain_mask:
        chain_encoding:
        residue_idx:
        mask:
        pose_length:
        X_unbound:
        chain_M_pos:
        residue_mask:
        randn:
        decoding_order:

    Returns:
        A mapping of the key describing to the corresponding value, i.e. sequences, complex_sequence_loss, and
            unbound_sequence_loss
    """
    if chain_M_pos is not None:
        residue_mask = chain_M_pos  # Name change makes more sense
    elif residue_mask is not None:
        pass
    else:
        raise ValueError("Must pass either 'residue_mask' or 'chain_M_pos'")

    if pose_length is None:
        batch_length, pose_length, *_ = X.shape
    else:
        batch_length, *_ = X.shape

    actual_batch_length = batch_slice.stop - batch_slice.start

    # Slice the sequence according to those that are currently batched for scoring
    S = S[batch_slice]  # , None)
    if actual_batch_length != batch_length:
        # Slice these for the last iteration
        X = X[:actual_batch_length]  # , None)
        chain_mask = chain_mask[:actual_batch_length]  # , None)
        chain_encoding = chain_encoding[:actual_batch_length]  # , None)
        residue_idx = residue_idx[:actual_batch_length]  # , None)
        mask = mask[:actual_batch_length]  # , None)
        # randn = randn[:actual_batch_length]
        residue_mask = residue_mask[:actual_batch_length]
        try:
            X_unbound = X_unbound[:actual_batch_length]  # , None)
        except TypeError:  # Can't slice NoneType
            pass

    # logger.debug(f'S shape: {S.shape}')
    # logger.debug(f'X shape: {X.shape}')
    # # logger.debug(f'chain_mask shape: {chain_mask.shape}')
    # logger.debug(f'chain_encoding shape: {chain_encoding.shape}')
    # logger.debug(f'residue_idx shape: {residue_idx.shape}')
    # logger.debug(f'mask shape: {mask.shape}')
    # # logger.debug(f'residue_mask shape: {residue_mask.shape}')

    chain_residue_mask = chain_mask * residue_mask
    # logger.debug(f'chain_residue_mask shape: {chain_residue_mask.shape}')

    # Score and format outputs - All have at lease shape (batch_length, model_length,)
    if decoding_order is not None:
        # logger.debug(f'decoding_order shape: {decoding_order.shape}, type: {decoding_order.dtype}')
        decoding_order = decoding_order[:actual_batch_length]
        provided_decoding_order = True
        randn = None
    elif randn is not None:
        # logger.debug(f'decoding_order shape: {randn.shape}, type: {randn.dtype}')
        randn = randn[:actual_batch_length]
        decoding_order = None
        provided_decoding_order = False
    else:
        # Todo generate a randn fresh?
        raise ValueError("Missing required argument 'randn' or 'decoding_order'")

    # decoding_order_out = decoding_order  # When using the same decoding order for all
    log_probs_start_time = time.time()

    # Todo debug the input Tensor. Most likely the sequence must be (batch, pose, aa?)
    # RuntimeError: Index tensor must have the same number of dimensions as input tensor
    complex_log_probs = \
        model(X, S, mask, chain_residue_mask, residue_idx, chain_encoding, randn,
              use_input_decoding_order=provided_decoding_order, decoding_order=decoding_order)
    per_residue_complex_sequence_loss = \
        sequence_nllloss(S[:, :pose_length], complex_log_probs[:, :pose_length]).cpu().numpy()

    # Reshape data structures to have shape (batch_length, number_of_temperatures, pose_length)
    # _residue_indices_of_interest = residue_mask[:, :pose_length].cpu().numpy().astype(bool)
    # sequences = np.concatenate(batch_sequences, axis=1).reshape(actual_batch_length, number_of_temps, pose_length)
    # complex_sequence_loss = \
    #     np.concatenate(per_residue_complex_sequence_loss, axis=1)\
    #     .reshape(actual_batch_length, number_of_temps, pose_length)
    # if X_unbound is not None:
    #     unbound_sequence_loss = \
    #         np.concatenate(per_residue_unbound_sequence_loss, axis=1)\
    #         .reshape(actual_batch_length, number_of_temps, pose_length)
    # else:
    #     unbound_sequence_loss = np.empty_like(complex_sequence_loss)
    if X_unbound is not None:
        # unbound_log_prob_start_time = time.time()
        unbound_log_probs = \
            model(X_unbound, S, mask, chain_residue_mask, residue_idx, chain_encoding, randn,
                  use_input_decoding_order=provided_decoding_order, decoding_order=decoding_order)
        per_residue_unbound_sequence_loss = \
            sequence_nllloss(S[:, :pose_length], unbound_log_probs[:, :pose_length]).cpu().numpy()
        # logger.debug(f'Unbound log probabilities calculation took '
        #              f'{time.time() - unbound_log_prob_start_time:8f}s')
    else:
        per_residue_unbound_sequence_loss = np.empty_like(complex_log_probs)
    model.log.info(f'Log probabilities score calculation took {time.time() - log_probs_start_time:8f}s')

    return {'proteinmpnn_loss_complex': per_residue_complex_sequence_loss,
            'proteinmpnn_loss_unbound': per_residue_unbound_sequence_loss}
