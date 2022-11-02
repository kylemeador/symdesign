from __future__ import annotations

import functools
import os
import time
from math import ceil
from typing import Annotated, Iterable, Container, Type, Callable, Sequence, Any

import numpy as np
import torch

from ProteinMPNN.protein_mpnn_utils import ProteinMPNN
from symdesign import utils

logger = utils.start_log(name=__name__)
mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'  # structure.utils.protein_letters_alph1_unknown
mpnn_alphabet_length = len(mpnn_alphabet)
MPNN_NULL_IDX = 20
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
            def _setup(*_args, **_kwargs) -> dict:
                return {}
        else:
            _setup = setup

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
                    setup_returns = _setup(_batch_length, *setup_args, **setup_kwargs)
                    for batch in range(number_of_batches):
                        # Find the upper slice limit
                        batch_slice = slice(batch * _batch_length, min((batch+1) * _batch_length, size))
                        # Perform the function, batch_slice must be used inside the func
                        function_returns = func(batch_slice, *args, **kwargs, **setup_returns)
                        # Set the returned values in the order they were received to the precalculated return_container
                        for return_container_key, return_container in return_containers.items():
                            return_container[batch_slice] = function_returns[return_container_key]

                    # Report success
                    logger.debug(f'Successful execution with batch_length of {_batch_length}')
                    last_error = None
                    break  # finished = True
                except compute_failure_exceptions as error:
                    if _error is None:  # Set the error the first time
                        _error = last_error = error
                    else:
                        # raise _error
                        last_error = error
                    logger.debug(f'{batch_calculation.__name__}: encountered error during {func.__name__} execution:'
                                 f'\n{error}')
                    _batch_length -= 1

            if last_error is not None:  # This exited from the ZeroDivisionError except
                try:
                    logger.critical(f'{batch_calculation.__name__} exited with the following exceptions:\n\nThe first '
                                    f'exception in the traceback was the result of the first iteration, while the '
                                    f'most recent exception in the traceback is last\n')
                    raise _error
                except compute_failure_exceptions:
                    raise last_error

            return return_containers
        return wrapped
    return wrapper


def create_decoding_order(randn: torch.Tensor, chain_mask: torch.Tensor, tied_pos: Iterable[Container] = None,
                          to_device: str = None, **kwargs) \
        -> torch.Tensor:
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
    # numbers are smaller for places where chain_mask = 0.0 and higher for places where chain_mask = 1.0
    decoding_order = torch.argsort((chain_mask+0.0001) * (torch.abs(randn)))

    if tied_pos is not None:
        # Calculate the tied decoding order according to ProteinMPNN.tied_sample()
        # return decoding_order
    # else:
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

        decoding_order = torch.tensor(found_decoding_indices, device=to_device)[None].repeat(randn.shape[0], 1)

    return decoding_order


class ProteinMPNNFactory:
    """Return a ProteinMPNN instance by calling the Factory instance with the ProteinMPNN model name

        Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
        allocating a shared pointer to the named ProteinMPNN model
    """
    def __init__(self, **kwargs):
        self._models = {}
        # self._models = None

    def __call__(self, model_name: str = 'v_48_020', backbone_noise: float = 0., **kwargs) -> ProteinMPNN:
        """Return the specified ProteinMPNN object singleton

        Args:
            model_name: The name of the model to use from ProteinMPNN. v_X_Y where X is neighbor distance, and Y is noise
            backbone_noise: The amount of backbone noise to add to the pose during design
        Returns:
            The instance of the initialized ProteinMPNN model
        """
        model_name_key = f'{model_name}_{backbone_noise}'
        model = self._models.get(model_name_key)
        if model:
            return model
        else:  # Create a new ProteinMPNN model instance
            # Acquire a adequate computing device
            device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
            # device = torch.device('cpu')
            checkpoint = torch.load(os.path.join(utils.path.protein_mpnn_weights_dir, f'{model_name}.pt'),
                                    map_location=device)
            logger.info(f'Number of edges: {checkpoint["num_edges"]}')
            logger.info(f'Training noise level: {checkpoint["noise_level"]} Angstroms')
            hidden_dim = 128
            num_layers = 3
            model = ProteinMPNN(num_letters=mpnn_alphabet_length,
                                node_features=hidden_dim,
                                edge_features=hidden_dim,
                                hidden_dim=hidden_dim,
                                num_encoder_layers=num_layers,
                                num_decoder_layers=num_layers,
                                augment_eps=backbone_noise,
                                k_neighbors=checkpoint['num_edges'])
            model.to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.device = device
            self._models[model_name_key] = model

        return model

    def get(self, **kwargs) -> ProteinMPNN:
        """Return the specified ProteinMPNN object singleton

        Keyword Args:
            model_name - (str) = 'v_48_020' - The name of the model to use from ProteinMPNN.
                v_X_Y where X is neighbor distance, and Y is noise
            backbone_noise - (float) = 0.0 - The amount of backbone noise to add to the pose during design
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
        size = X.shape[0]
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
    # Clone the data from the sequence tensor so that it can be set with the null token below
    S_design_null = S.detach().clone()
    if pose_length is None:
        batch_length, pose_length, *_ = S.shape
    else:
        batch_length, *_ = S.shape
    # X_unbound = batch_parameters.get('X_unbound')
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
        S_design_null = S_design_null[:actual_batch_length]  # , None)
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

    # Use the sequence as an unknown token then guess the probabilities given the remaining
    # information, i.e. the sequence and the backbone
    S_design_null[residue_mask.type(torch.bool)] = MPNN_NULL_IDX
    chain_residue_mask = chain_mask * residue_mask

    batch_sequences = []
    _per_residue_complex_sequence_loss = []
    _per_residue_unbound_sequence_loss = []
    number_of_temps = len(temperatures)
    for temp_idx, temperature in enumerate(temperatures):
        sample_start_time = time.time()
        if tied_pos is None:
            sample_dict = proteinmpnn.sample(X, randn, S_design_null, chain_mask, chain_encoding, residue_idx, mask,
                                             chain_M_pos=residue_mask, temperature=temperature, bias_by_res=bias_by_res,
                                             **batch_parameters)
        else:
            sample_dict = proteinmpnn.tied_sample(X, randn, S_design_null, chain_mask, chain_encoding, residue_idx,
                                                  mask, chain_M_pos=residue_mask, temperature=temperature,
                                                  bias_by_res=bias_by_res, tied_pos=tied_pos, **batch_parameters)
        logger.info(f'Sample calculation took {time.time() - sample_start_time:8f}s')
        S_sample = sample_dict['S']
        # Format outputs
        _batch_sequences = S_sample.cpu()[:, :pose_length]
        batch_sequences.append(_batch_sequences)
        decoding_order = sample_dict['decoding_order']
        # decoding_order_out = decoding_order  # When using the same decoding order for all
        if X_unbound is not None:
            unbound_log_prob_start_time = time.time()
            unbound_log_probs = \
                proteinmpnn(X_unbound, S_sample, mask, chain_residue_mask, residue_idx, chain_encoding,
                            None,  # This argument is provided but with below args, is not used
                            use_input_decoding_order=True, decoding_order=decoding_order).cpu()
            _per_residue_unbound_sequence_loss.append(
                sequence_nllloss(_batch_sequences, unbound_log_probs[:, :pose_length]).numpy())
            logger.debug(f'Unbound log probabilities calculation took '
                         f'{time.time() - unbound_log_prob_start_time:8f}s')

        log_probs_start_time = time.time()
        complex_log_probs = \
            proteinmpnn(X, S_sample, mask, chain_residue_mask, residue_idx, chain_encoding,
                        None,  # This argument is provided but with below args, is not used
                        use_input_decoding_order=True, decoding_order=decoding_order).cpu()
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
            sequence_nllloss(_batch_sequences, complex_log_probs[:, :pose_length]).numpy())
        logger.debug(f'Log probabilities calculation took {time.time() - log_probs_start_time:8f}s')

    # Reshape data structures to have shape (batch_length, number_of_temperatures, pose_length)
    sequences = np.concatenate(batch_sequences, axis=1).reshape(actual_batch_length, number_of_temps, pose_length)
    complex_sequence_loss =\
        np.concatenate(_per_residue_complex_sequence_loss, axis=1).reshape(actual_batch_length,
                                                                           number_of_temps,
                                                                           pose_length)
    if X_unbound is not None:
        unbound_sequence_loss = \
            np.concatenate(_per_residue_unbound_sequence_loss, axis=1).reshape(actual_batch_length,
                                                                               number_of_temps,
                                                                               pose_length)
    else:
        unbound_sequence_loss = np.empty_like(complex_sequence_loss)

    return {'sequences': sequences,
            'complex_sequence_loss': complex_sequence_loss,
            'unbound_sequence_loss': unbound_sequence_loss}


def sequence_nllloss(sequence: torch.Tensor, log_probs: torch.Tensor,
                     mask: torch.Tensor = None, per_residue: bool = True) -> torch.Tensor:
    """Score ProteinMPNN sequences using the Negative log likelihood loss function

    Args:
        sequence: The sequence tensor
        log_probs: The logarithmic probabilities as found by a forward pass through ProteinMPNN
        mask: Any positions that are masked in the design task
        per_residue: Whether to return scores per residue
    Returns:
        The loss calculated over the log probabilities compared to the sequence tensor.
            If per_residue=True, the returned Tensor is the same shape as S, otherwise, it is the length of S
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
