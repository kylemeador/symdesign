from __future__ import annotations

import copy
import functools
import logging
import math
import os
import random
import sys
import time
import traceback
import warnings
from math import ceil
from typing import Annotated, Iterable, Container, Literal, Type, Callable, Sequence, Any

import psutil
from Bio import BiopythonDeprecationWarning
import jax.numpy as jnp
import numpy as np
import torch

# import symdesign.third_party.alphafold.alphafold as af
from symdesign.third_party.alphafold.alphafold.model import config as afconfig, data as afdata
from symdesign.third_party.alphafold.alphafold.common import protein as afprotein, residue_constants
with warnings.catch_warnings():
    # Cause all warnings to always be ignored
    warnings.simplefilter('ignore', category=BiopythonDeprecationWarning)
    from symdesign.third_party.alphafold.alphafold.data.pipeline import FeatureDict
from symdesign.third_party.alphafold.alphafold.relax import amber_minimize, utils as af_relax_utils
from .config import relax_options_literal
from symdesign.third_party.ProteinMPNN.protein_mpnn_utils import ProteinMPNN
from symdesign.sequence import numerical_translation_alph1_unknown_bytes
from symdesign import utils
putils = utils.path

logger = logging.getLogger(__name__)
proteinmpnn_default_translation_table = numerical_translation_alph1_unknown_bytes
mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'  # structure.utils.protein_letters_alph1_unknown
mpnn_alphabet_length = len(mpnn_alphabet)
MPNN_NULL_IDX = 20


def get_device_memory(device: str | int | None) -> int:
    if device.type == 'cpu':
        memory_constraint = psutil.virtual_memory().available
        logger.debug(f'The available cpu memory is: {memory_constraint}')
    else:
        free_memory, gpu_memory_total = torch.cuda.mem_get_info()
        logger.debug(f'The available gpu memory is: {free_memory}')
        memory_reserved = torch.cuda.memory_reserved()
        logger.debug(f'The reserved gpu memory is: {memory_reserved}')
        memory_constraint = free_memory + memory_reserved
    return memory_constraint


def calculate_proteinmpnn_batch_length(model: ProteinMPNN, number_of_residues: int, element_memory: int = 4) -> int:
    """

    Args:
        model: The ProteinMPNN model
        number_of_residues: The number of residues used in the ProteinMPNN model
        element_memory: Where each element is np.int64, np.float32, etc.
    Returns:

    """
    memory_constraint = get_device_memory(model.device)

    number_of_elements_available = memory_constraint // element_memory
    logger.debug(f'The number_of_elements_available is: {number_of_elements_available}')
    number_of_model_parameter_elements = sum([math.prod(param.size()) for param in model.parameters()])
    logger.debug(f'The number_of_model_parameter_elements is: {number_of_model_parameter_elements}')
    model_elements = number_of_model_parameter_elements
    # Todo use 5 as ideal CB is added by the model later with ca_only = False
    num_model_residues = 5
    model_elements += math.prod((number_of_residues, num_model_residues, 3))  # X,
    model_elements += number_of_residues  # S.shape
    model_elements += number_of_residues  # chain_mask.shape
    model_elements += number_of_residues  # chain_encoding.shape
    model_elements += number_of_residues  # residue_idx.shape
    model_elements += number_of_residues  # mask.shape
    model_elements += number_of_residues  # residue_mask.shape
    model_elements += math.prod((number_of_residues, 21))  # omit_AA_mask.shape
    model_elements += number_of_residues  # pssm_coef.shape
    model_elements += math.prod((number_of_residues, 20))  # pssm_bias.shape
    model_elements += math.prod((number_of_residues, 20))  # pssm_log_odds_mask.shape
    model_elements += number_of_residues  # tied_beta.shape
    model_elements += math.prod((number_of_residues, 21))  # bias_by_res.shape
    logger.debug(f'The number of model_elements is: {model_elements}')

    number_of_batches = number_of_elements_available // model_elements
    return number_of_batches // proteinmpnn_batch_divisor


# This heuristic was decided based on successful runs and the batch_length at which they succeeded
proteinmpnn_batch_divisor = 400
# 6 - Works for 24 GiB mem with 6264 residue T input, 7 is too much
# 1 - Works for 5.61 GiB mem with 6264 residue T input, 2 is too much
# (5.66 GiB - 2.74 GiB) for sample() with 6264 residue T input
# (2.74 GiB - 1.25 GiB) for log_probs() with 6264 residue T input
#
PROTEINMPNN_DESIGN_BATCH_LEN = 6
PROTEINMPNN_SCORE_BATCH_LEN = 6
# # Use these options to bring down GPU memory when using Tensor instances with fixed size to a model
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
# pip install GPUtil
# from GPUtil import showUtilization as gpu_usage


# def batch_calculation(size: int, batch_length: int, function_args: Iterable = tuple(),  # function: Callable,
#                                 function_kwargs: dict = None, function_return_containers: tuple = tuple(),
#                                 compute_failure_exceptions: tuple[Type[Exception]] =
#                                 (np.core._exceptions._ArrayMemoryError,), setup: Callable = None) -> tuple:
#     """Execute a function in batches over an input that is too large for the available memory
#
#     Args:
#         size: The total number of units of work to be done
#         batch_length: The starting length of a batch. This should be chosen empirically
#         function: The callable to iteratively execute
#         function_args: The arguments to pass to the function
#         function_kwargs: The keyword arguments to pass to the function
#         function_return_containers: The returns that should be populated by the function
#         compute_failure_exceptions: A tuple of possible failure modes to restart the calculation upon error
#         setup: A function which should be called before the batches are executed to produce data that is passed to the
#             function
#     Returns:
#         The populated function_return_containers
#     """


def batch_calculation(size: int, batch_length: int, setup: Callable = None,
                      compute_failure_exceptions: tuple[Type[Exception], ...] = (Exception,)) -> Callable:
    """Use as a decorator to execute a function in batches over an input that is too large for available computational
    resources, typically memory

    Produces the variables actual_batch_length and batch_slice that can be used inside the decorated function

    Args:
        size: The total number of units of work to be done
        batch_length: The starting length of a batch. This should be chosen empirically
        setup: A Callable which should be called before the batches are executed to produce data that is passed to the
            function. The first argument of this Callable should be batch_length
        compute_failure_exceptions: A tuple of possible exceptions which upon raising should be allowed to restart
    Decorated Callable Args:
        args: The arguments to pass to the function
        kwargs: Keyword Arguments to pass to the decorated Callable
        setup_args: Arguments to pass to the setup Callable
        setup_kwargs: Keyword Arguments to pass to the setup Callable
        return_containers: dict - The key and SupportsIndex value to store decorated Callable returns inside
    Returns:
        The populated function_return_containers
    """
    def wrapper(func: Callable) -> Callable[[tuple[Any, ...], dict | None, tuple, dict | None, dict[str, Any]], dict]:
        if setup is None:
            def setup_(*_args, **_kwargs) -> dict:
                return {}
        else:
            setup_ = setup

        @functools.wraps(func)
        def wrapped(*args, return_containers: dict = None,
                    setup_args: tuple = tuple(), setup_kwargs: dict = None, **kwargs) -> dict:

            if return_containers is None:
                return_containers = {}

            if setup_kwargs is None:
                setup_kwargs = {}

            _batch_length = batch_length
            # finished = False
            _error = last_error = None
            while True:  # not finished:
                logger.debug(f'The batch_length is: {_batch_length}')
                try:  # The next batch_length
                    # The number_of_batches indicates how many iterations are needed to exhaust all models
                    try:
                        number_of_batches = int(ceil(size/_batch_length) or 1)  # Select at least 1
                    except ZeroDivisionError:  # We hit the minimal batch size. Report the previous error
                        if last_error is not None:  # This exited from the compute_failure_exceptions except
                            break  # break out and raise the _error
                        else:
                            raise ValueError(f'The batch_length ({batch_length}) must be greater than 0')
                    # Perform any setup operations
                    # logger.critical(f'Before SETUP\nmemory_allocated: {torch.cuda.memory_allocated()}'
                    #                 f'\nmemory_reserved: {torch.cuda.memory_reserved()}')
                    setup_start = time.time()
                    setup_returns = setup_(_batch_length, *setup_args, **setup_kwargs)
                    logger.debug(f'{batch_calculation.__name__} setup function took {time.time() - setup_start:8f}s')
                    # logger.critical(f'After SETUP\nmemory_allocated: {torch.cuda.memory_allocated()}'
                    #                 f'\nmemory_reserved: {torch.cuda.memory_reserved()}')
                    batch_start = time.time()
                    for batch in range(number_of_batches):
                        # Find the upper slice limit
                        batch_slice = slice(batch * _batch_length, min((batch+1) * _batch_length, size))
                        # Perform the function, batch_slice must be used inside the func
                        logger.debug(f'Calculating batch {batch + 1}')
                        function_returns = func(batch_slice, *args, **kwargs, **setup_returns)
                        # Set the returned values in the order they were received to the precalculated return_container
                        for return_key, return_value in list(function_returns.items()):
                            try:  # To access the return_container_key in the function
                                return_containers[return_key][batch_slice] = return_value
                            except KeyError:  # If it doesn't exist
                                raise KeyError(f"Couldn't return the data specified by {return_key} to the "
                                               f"return_container with keys:{', '.join(return_containers.keys())}")
                            except ValueError as error:  # Arrays are incorrectly sized
                                raise ValueError(f"Couldn't return the data specified by {return_key} from "
                                                 f"{func.__name__} due to: {error}")
                        # for return_container_key, return_container in list(return_containers.items()):
                        #     try:  # To access the return_container_key in the function
                        #         return_container[batch_slice] = function_returns[return_container_key]
                        #     except KeyError:  # If it doesn't exist
                        #         # Remove the data from the return_containers
                        #         return_containers.pop(return_container_key)

                    # Report success
                    logger.debug(f'Successful execution with batch_length of {_batch_length}. '
                                 f'Took {time.time() - batch_start:8f}s')
                    last_error = None
                    break  # finished = True
                except compute_failure_exceptions as error:
                    # del setup_returns
                    # logger.critical(f'After ERROR\nmemory_allocated: {torch.cuda.memory_allocated()}'
                    #                 f'\nmemory_reserved: {torch.cuda.memory_reserved()}')
                    # gc.collect()
                    # logger.critical(f'After GC\nmemory_allocated: {torch.cuda.memory_allocated()}'
                    #                 f'\nmemory_reserved: {torch.cuda.memory_reserved()}')
                    if _error is None:  # Set the error the first time
                        # _error = last_error = error
                        _error = last_error = traceback.format_exc()  # .format_exception(error)
                    else:
                        # last_error = error
                        last_error = traceback.format_exc()  # .format_exception(error)
                    _batch_length -= 1

            if last_error is not None:  # This exited from the ZeroDivisionError except
                # try:
                logger.critical(f'{batch_calculation.__name__} exited with the following exceptions:\n\nThe first '
                                f'exception in the traceback was the result of the first iteration, while the '
                                f'most recent exception in the traceback is last\n')
                # raise _error
                print(''.join(_error))
                # except compute_failure_exceptions:
                #     raise last_error
                print(''.join(last_error))
                raise RuntimeError(f"{func.__name__} wasn't able to be executed. See the above traceback")

            return return_containers
        return wrapped
    return wrapper


def create_decoding_order(randn: torch.Tensor, chain_mask: torch.Tensor, tied_pos: Iterable[Container] = None,
                          to_device: str = None, **kwargs) -> torch.Tensor:
    """

    Args:
        randn:
        chain_mask:
        tied_pos:
        to_device:

    Returns:

    """
    if to_device is None:
        to_device = randn.device
    # Numbers are smaller for places where chain_mask = 0.0 and higher for places where chain_mask = 1.0
    decoding_order = torch.argsort((chain_mask+0.0001) * (torch.abs(randn)))

    if tied_pos is not None:
        # Calculate the tied decoding order according to ProteinMPNN.tied_sample()
        new_decoding_order: list[list[int]] = []
        found_decoding_indices = []
        for t_dec in list(decoding_order[0].cpu().numpy()):
            if t_dec not in found_decoding_indices:
                for item in tied_pos:
                    if t_dec in item:
                        break
                else:
                    item = [t_dec]
                # Keep list of lists format
                new_decoding_order.append(item)
                # Add all found decoding_indices
                found_decoding_indices.extend(item)

        decoding_order = torch.tensor(found_decoding_indices, device=to_device)[None].repeat(len(randn), 1)

    return decoding_order


class _ProteinMPNN(ProteinMPNN):
    """Implemented to instruct logging outputs"""
    log = logging.getLogger(f'{__name__}.ProteinMPNN')
    pass


class ProteinMPNNFactory:
    """Return a ProteinMPNN instance by calling the Factory instance with the ProteinMPNN model name

        Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
        allocating a shared pointer to the named ProteinMPNN model
    """
    def __init__(self, **kwargs):
        self._models = {}
        # self._models = None

    def __call__(self, model_name: str = 'v_48_020', backbone_noise: float = 0., ca_only: bool = False, **kwargs) \
            -> ProteinMPNN:
        """Return the specified ProteinMPNN object singleton

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
                logger.error(f"No such ca_only model 'v_48_030'. Loading ca_only model 'v_48_020' (highest "
                             f"backbone noise ca_only model) instead")
                model_name = 'v_48_020'
        else:
            ca = ''
        model_name_key = f'{model_name}{ca}_{backbone_noise}'
        model = self._models.get(model_name_key)
        if model:
            return model
        else:  # Create a new ProteinMPNN model instance
            if not self._models:  # Nothing initialized
                # Acquire a adequate computing device
                if torch.cuda.is_available():
                    self.device = torch.device('cuda:0')
                    # Set the environment to use memory efficient cuda management
                    max_split = 1000
                    pytorch_conf = f'max_split_size_mb:{max_split},' \
                                   f'roundup_power2_divisions:4,' \
                                   f'garbage_collection_threshold:0.7'
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = pytorch_conf
                    logger.debug(f'Setting pytorch configuration:\n{pytorch_conf}\n'
                                 f'Result:{os.getenv("PYTORCH_CUDA_ALLOC_CONF")}')
                else:
                    self.device = torch.device('cpu')
                logger.info(f'Loading ProteinMPNN model "{model_name_key}" to device: {self.device}')

            if ca_only:
                weights_dir = utils.path.protein_mpnn_ca_weights_dir
            else:
                weights_dir = utils.path.protein_mpnn_weights_dir
            checkpoint = torch.load(os.path.join(weights_dir, f'{model_name}.pt'), map_location=self.device)
            hidden_dim = 128
            num_layers = 3
            with torch.no_grad():
                model = _ProteinMPNN(num_letters=mpnn_alphabet_length,
                                     node_features=hidden_dim,
                                     edge_features=hidden_dim,
                                     hidden_dim=hidden_dim,
                                     num_encoder_layers=num_layers,
                                     num_decoder_layers=num_layers,
                                     augment_eps=backbone_noise,
                                     k_neighbors=checkpoint['num_edges'])
                model.to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                model.device = self.device

            model.log.info(f'Number of edges: {checkpoint["num_edges"]}')
            model.log.info(f'Training noise level: {checkpoint["noise_level"]} Angstroms')

            number_of_mpnn_model_parameters = sum([math.prod(param.size()) for param in model.parameters()])
            logger.debug(f'The number of proteinmpnn model parameters is: {number_of_mpnn_model_parameters}')

            self._models[model_name_key] = model

        return model

    def get(self, **kwargs) -> ProteinMPNN:
        """Return the specified ProteinMPNN object singleton

        Keyword Args:
            model_name - str = 'v_48_020' - The name of the model to use from ProteinMPNN.
                v_X_Y where X is neighbor distance, and Y is noise
            backbone_noise - float = 0.0 - The amount of backbone noise to add to the pose during design
        Returns:
            The instance of the initialized ProteinMPNN model
        """
        return self.__call__(**kwargs)


proteinmpnn_factory: Annotated[ProteinMPNNFactory,
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


def batch_proteinmpnn_input(size: int = None, **kwargs) -> dict[str, np.ndarray]:
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
            raise ValueError(f'{batch_proteinmpnn_input.__name__} must pass keyword argument "X" if argument "size" '
                             f'is None')
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


def proteinmpnn_to_device(device: str = None, **kwargs) -> dict[str, torch.Tensor]:
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
def setup_pose_batch_for_proteinmpnn(batch_length: int, device, **parameters) -> dict[str, np.ndarray | torch.Tensor]:
    """

    Args:
        batch_length: The length the batch to set up
        device: The device used for batch calculations
    Returns:
        A mapping of necessary containers for ProteinMPNN inference in batches and loaded to the device
    """
    # batch_length = batch_slice.stop - batch_slice.start
    # Create batch_length fixed parameter data which are the same across poses
    batch_parameters: dict[str, np.ndarray | torch.Tensor] = \
        batch_proteinmpnn_input(size=batch_length, **parameters)
    # Move fixed data structures to the model device
    # Update parameters as some are not transferred to the identified device
    batch_parameters.update(proteinmpnn_to_device(device, **batch_parameters))

    return batch_parameters


gb_divisor = 1e9


# @batch_calculation(size=size, batch_length=batch_length,
#                    setup=setup_pose_batch_for_proteinmpnn,
#                    compute_failure_exceptions=(RuntimeError, np.core._exceptions._ArrayMemoryError))
def proteinmpnn_batch_design(batch_slice: slice, proteinmpnn: ProteinMPNN,
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
        proteinmpnn:
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
        A mapping of the key describing to the corresponding value, i.e. sequences, complex_sequence_loss, and
            unbound_sequence_loss
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
            sample_dict = proteinmpnn.sample(X, randn, S, chain_mask, chain_encoding, residue_idx, mask,
                                             chain_M_pos=residue_mask, temperature=temperature, bias_by_res=bias_by_res,
                                             **batch_parameters)
        else:
            sample_dict = proteinmpnn.tied_sample(X, randn, S, chain_mask, chain_encoding, residue_idx,
                                                  mask, chain_M_pos=residue_mask, temperature=temperature,
                                                  bias_by_res=bias_by_res, tied_pos=tied_pos, **batch_parameters)
        proteinmpnn.log.info(f'Sample calculation took {time.time() - sample_start_time:8f}s')

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
            proteinmpnn.log.warning(f'Found null sequence output... Resampling selected positions')
            proteinmpnn.log.debug(f'At sequence position(s): {null_sequence_indices}')
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
            # proteinmpnn.log.debug('Fixed null sequence elements')

        decoding_order = sample_dict['decoding_order']
        # decoding_order_out = decoding_order  # When using the same decoding order for all
        log_probs_start_time = time.time()
        if X_unbound is not None:
            # unbound_log_prob_start_time = time.time()
            # logger.critical(f'Starting unbound calc: '
            #                 f'available memory={get_device_memory(proteinmpnn.device)/gb_divisor}')
            unbound_log_probs = \
                proteinmpnn(X_unbound, S_sample, mask, chain_residue_mask, residue_idx, chain_encoding,
                            None,  # This argument is provided but with below args, is not used
                            use_input_decoding_order=True, decoding_order=decoding_order)
            # logger.critical(f'After unbound calc: '
            #                 f'available memory={get_device_memory(proteinmpnn.device)/gb_divisor}')
            _per_residue_unbound_sequence_loss.append(
                sequence_nllloss(_batch_sequences, unbound_log_probs[:, :pose_length]).cpu().numpy())
            # logger.debug(f'Unbound log probabilities calculation took '
            #              f'{time.time() - unbound_log_prob_start_time:8f}s')

        # logger.critical(f'Starting bound calc: '
        #                 f'available memory={get_device_memory(proteinmpnn.device) / gb_divisor}')
        complex_log_probs = \
            proteinmpnn(X, S_sample, mask, chain_residue_mask, residue_idx, chain_encoding,
                        None,  # This argument is provided but with below args, is not used
                        use_input_decoding_order=True, decoding_order=decoding_order)
        # logger.critical(f'After bound calc: '
        #                 f'available memory={get_device_memory(proteinmpnn.device) / gb_divisor}')
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
        proteinmpnn.log.info(f'Log probabilities score calculation took {time.time() - log_probs_start_time:8f}s')
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


def proteinmpnn_batch_score(batch_slice: slice, proteinmpnn: ProteinMPNN,
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
        proteinmpnn:
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
        raise ValueError(f'Must pass either "residue_mask" or "chain_M_pos"')

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
        raise ValueError('Missing required argument "randn" or "decoding_order"')

    # decoding_order_out = decoding_order  # When using the same decoding order for all
    log_probs_start_time = time.time()

    # Todo debug the input Tensor. Most likely the sequence must be (batch, pose, aa?)
    # RuntimeError: Index tensor must have the same number of dimensions as input tensor
    complex_log_probs = \
        proteinmpnn(X, S, mask, chain_residue_mask, residue_idx, chain_encoding, randn,
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
            proteinmpnn(X_unbound, S, mask, chain_residue_mask, residue_idx, chain_encoding, randn,
                        use_input_decoding_order=provided_decoding_order, decoding_order=decoding_order)
        per_residue_unbound_sequence_loss = \
            sequence_nllloss(S[:, :pose_length], unbound_log_probs[:, :pose_length]).cpu().numpy()
        # logger.debug(f'Unbound log probabilities calculation took '
        #              f'{time.time() - unbound_log_prob_start_time:8f}s')
    else:
        per_residue_unbound_sequence_loss = np.empty_like(complex_log_probs)
    proteinmpnn.log.info(f'Log probabilities score calculation took {time.time() - log_probs_start_time:8f}s')

    return {'proteinmpnn_loss_complex': per_residue_complex_sequence_loss,
            'proteinmpnn_loss_unbound': per_residue_unbound_sequence_loss}


def sequence_nllloss(sequence: torch.Tensor, log_probs: torch.Tensor,
                     mask: torch.Tensor = None, per_residue: bool = True) -> torch.Tensor:
    """Score designed sequences using the Negative log likelihood loss function

    Args:
        sequence: The sequence tensor
        log_probs: The logarithmic probabilities at each residue for every amino acid.
            This may be found by an evolutionary profile or a forward pass through ProteinMPNN
        mask: Any positions that are masked in the design task
        per_residue: Whether to return scores per residue
    Returns:
        The loss calculated over the log probabilities compared to the sequence tensor.
            If per_residue=True, the returned Tensor is the same shape as sequence (i.e. (batch, length)),
            otherwise, it is just the length of sequence as calculated by the average loss over every residue
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    # Measure log_probs loss with respect to the sequence. Make each sequence and log probs stacked along axis=0
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        sequence.contiguous().view(-1)
    ).view(sequence.size())  # Revert the shape to the original sequence shape
    # Take the average over every designed position and return the single score
    if per_residue:
        return loss
    elif mask is None:
        return torch.sum(loss, dim=-1)
    else:
        return torch.sum(loss*mask, dim=-1) / torch.sum(mask, dim=-1)


# Alphafold variables and helpers
af_model_literal = Literal['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer']
MULTIMER_RESIDUE_LIMIT = 4000
MONOMER_RESIDUE_LIMIT = 2500
# Relax
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


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


# The following code was modified from Alphafold to allow injection of prev_pos into model initialization
# Key difference is the replacement of the Alphafold class with AlphafoldInitialGuess class, largely based on ideas from
# the preprint "Improving de novo Protein Binder Design with Deep Learning" bioRxiv
# -------------------------------------------------------------------------
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for constructing the model."""
from typing import Any, Mapping, Optional, Union

from absl import logging
import haiku as hk
import jax
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf
import tree

# from symdesign.third_party.alphafold.alphafold.common import confidence
from symdesign.third_party.alphafold.alphafold.model import features, model as afmodel
from . import monomer, multimer


class RunModel(afmodel.RunModel):
    """Container for JAX model."""

    def __init__(self,
                 config: ml_collections.ConfigDict,
                 params: Optional[Mapping[str, Mapping[str, jnp.ndarray]]] = None):
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

      self.apply = jax.jit(hk.transform(_forward_fn).apply)
      self.init = jax.jit(hk.transform(_forward_fn).init)

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
      result.update(
          afmodel.get_confidence_metrics(result, multimer_mode=self.multimer_mode))
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
            raise KeyError(f"The parameter_type='{parameter_type}' isn't available from the viable parameter "
                           f"sets\nCurrently available types include: {', '.join(self.parameter_map.keys())}")
        result = self.apply(params, jax.random.PRNGKey(random_seed), feat)

        # This block is to ensure benchmark timings are accurate. Some blocking is
        # already happening when computing get_confidence_metrics, and this ensures
        # all outputs are blocked on.
        jax.tree_map(lambda x: x.block_until_ready(), result)
        result.update(
            afmodel.get_confidence_metrics(result, multimer_mode=self.multimer_mode))
        logging.info('Output shape was %s',
                     tree.map_structure(lambda x: x.shape, result))
        return result
    # SYMDESIGN


model_type_to_config_name = {
    'multimer': 'model_1_multimer_v3',
    'monomer_ptm': 'model_1_ptm',
    'monomer': 'model_1',
    'monomer_casp14': 'model_1',
}


def set_up_model_runners(model_type: af_model_literal = 'monomer', num_predictions_per_model: int = 1,
                         num_ensemble: int = 1, development: bool = False) -> dict[str, RunModel]:
    """Produce Alphafold RunModel class loaded with their training parameters

    Args:
        model_type: The type of model to load. Should be one of the viable Alphafold models including:
            'monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'
        num_predictions_per_model: The number of predictions to make for each Alphafold model. Essentially duplicates
            the original models 'num_predictions_per_model' times
        num_ensemble: The number of model ensembles to make. Typically, 1 is sufficient, but during CASP14, 8 were used
        development: Whether a smaller subset of models should be used for increased testing performance
    Returns:
        A dictionary of the model name to the RunModel instance for each 'model_type'/'num_predictions_per_model'
            requested
    """
    # model_runners = {}
    # model_names = afconfig.MODEL_PRESETS[model_type]
    # for model_name in model_names:
    #     if development and model_name != 'model_2_multimer_v3':
    #         continue
    #     model_config = afconfig.model_config(model_name)
    #     if model_config.model.global_config.multimer_mode:
    #         model_config.model.num_ensemble_eval = num_ensemble
    #     else:
    #         model_config.data.eval.num_ensemble = num_ensemble
    #     model_params = afdata.get_model_haiku_params(model_name=model_name, data_dir=putils.alphafold_db_dir)
    #     # This is using prev_pos init
    #     model_runner = RunModel(model_config, model_params)
    #     # This should be used if the prediction is not for a design and we have an msa
    #     # model_runner = afmodel.RunModel(model_config, model_params)
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
    model_config = afconfig.model_config(model_type_to_config_name[model_type])
    if model_config.model.global_config.multimer_mode:
        model_config.model.num_ensemble_eval = num_ensemble
    else:
        model_config.data.eval.num_ensemble = num_ensemble
    # Set up model params
    model_params = {}
    for model_name in afconfig.MODEL_PRESETS[model_type]:
        model_param = afdata.get_model_haiku_params(model_name=model_name, data_dir=putils.alphafold_db_dir)
        if 'model_1' in model_name:
            # Using the config for model_1 as it is most similar to other models
            #  model_1/2 includes template embeddings (monomer),
            #  while multimer model_1 is fairly similar to 2-5
            # RunModel is using prev_pos init
            model_runner = RunModel(model_config, model_param)
            # # ?? Not sure why this would be the case -> if the prediction is not for a design and there is an msa
            # model_runner = afmodel.RunModel(model_config, model_params)
            if development:
                model_params[f'{model_name}_pred_{0}'] = model_param
                break

        for i in range(num_predictions_per_model):
            model_params[f'{model_name}_pred_{i}'] = model_param

    num_models = len(model_params)
    logger.info(f'Loaded {num_models} Alphafold models: {", ".join(model_params)}')

    model_runner.set_params(model_params)
    return {model_param_name: model_runner for model_param_name in model_params}


def amber_relax(prot: afprotein, gpu: bool = False):
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
    af_relax_utils.assert_equal_nonterminal_atom_types(
        afprotein.from_pdb_string(min_pdb).atom_mask, prot.atom_mask)
    violations = out['structural_violations'][
        'total_per_residue_violations_mask'].tolist()

    # return min_pdb, debug_data, violations
    return min_pdb, violations


def af_predict(features: FeatureDict, model_runners: dict[str, RunModel],
               gpu_relax: bool = False, models_to_relax: relax_options_literal = None, random_seed: int = None) \
        -> tuple[dict[str, dict[str, str]], dict[str, FeatureDict]]:
    """Run Alphafold to predict a structure from sequence/msa/template features

    Args:
        # length: The length of the desired output for prediction metrics
        features: The sequence/msa/template feature parameters to populate the jax model
        model_runners: The RunModel instances which should predict the structure
        gpu_relax: Whether predictions should be relaxed using a GPU (if one is available)
        models_to_relax: Specify which predictions should be relaxed
        random_seed: A random integer to seed the model. Could be provided to ensure consistency across runs
    Returns:
        The tuple of structure and score dictionaries. Where structures contains the keys 'relaxed' and
        'unrelaxed' mapped to the model name and the model PDB string and folding_scores contain the model name
        mapped to each of the score types 'predicted_aligned_error' (length, length), 'plddt' (length),
        'predicted_template_modeling_score' (1), and 'predicted_interface_template_modeling_score' (1)
    """
    num_models = len(model_runners)
    if random_seed is None:  # Make one
        random_seed = random.randrange(sys.maxsize // num_models)

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
        'final_atom_positions', 'ranking_confidence', 'num_recycles', 'aligned_confidence_probs',
        'max_predicted_aligned_error',
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
        unrelaxed_protein = afprotein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=not model_runner.multimer_mode)
        unrelaxed_proteins[model_name] = unrelaxed_protein
        unrelaxed_pdbs_[model_name] = afprotein.to_pdb(unrelaxed_protein)

        ranking_confidences[model_name] = np_prediction_result['ranking_confidence']
        # Process incoming scores to be returned
        for old_score, new_score in change_scores:
            np_prediction_result[new_score] = np_prediction_result.pop(old_score)
        # Remove unnecessary scores
        for score in unneeded_scores:
            np_prediction_result.pop(score, None)
        scores[model_name] = np_prediction_result

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
