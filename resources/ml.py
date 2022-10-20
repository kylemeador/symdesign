import functools
import os
from math import ceil
from typing import Annotated, Iterable, Container, Type, Callable

import numpy as np
import torch

from ProteinMPNN.protein_mpnn_utils import ProteinMPNN
from utils import start_log
from utils.path import protein_mpnn_weights_dir

logger = start_log(name=__name__)
mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'  # structure.utils.protein_letters_alph1_unknown
mpnn_alphabet_length = len(mpnn_alphabet)
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
                      compute_failure_exceptions: tuple[Type[Exception]] = (Exception,)) -> Callable:  # tuple
    """Use as a decorator to execute a function in batches over an input that is too large for available computational
    resources, typically memory

    Produces the variables actual_batch_length and batch_slice that can be used inside the decorated function

    Args:
        size: The total number of units of work to be done
        batch_length: The starting length of a batch. This should be chosen empirically
        setup: A function which should be called before the batches are executed to produce data that is passed to the
            function
        compute_failure_exceptions: A tuple of possible exceptions which upon raising should be allowed to restart
    Returns:
        The populated function_return_containers
    """
    def wrapper(func: Callable) -> tuple:
        if setup is None:
            def _setup(*_args, **_kwargs) -> dict:
                return {}
        else:
            _setup = setup

        # def wrapped(function_args: Iterable = tuple(), function_kwargs: dict = None,
        #             function_return_containers: tuple = tuple(), setup: Callable = None) -> tuple:
        @functools.wraps(func)
        def wrapped(*args, function_return_containers: dict = None,
                    setup_args: tuple = tuple(), setup_kwargs: dict = None, **kwargs) -> tuple:

            if function_return_containers is None:
                function_return_containers = {}

            if setup_kwargs is None:
                setup_kwargs = {}

            _batch_length = batch_length
            # finished = False
            while True:  # not finished:
                logger.debug(f'The batch_length is: {_batch_length}')
                try:  # The next batch_length
                    # The number_of_batches indicates how many iterations are needed to exhaust all models
                    number_of_batches = int(ceil(size/_batch_length) or 1)  # Select at least 1
                    # Perform any setup operations
                    setup_returns = _setup(_batch_length, *setup_args, **setup_kwargs)
                    for batch in range(number_of_batches):
                        # Find the upper slice limit
                        batch_slice = slice(batch * _batch_length, (batch+1) * _batch_length)
                        # Perform the function, batch_slice must be used inside the func
                        function_returns = func(batch_slice, *args, **kwargs, **setup_returns)
                        # Set the returned values in the order they were received to the precalculated return_container
                        for return_container_key, return_container in function_return_containers.items():
                            return_container[batch_slice] = function_returns[return_container_key]

                    # Report success
                    logger.debug(f'Successful execution with batch_length of {_batch_length}')
                    break  # finished = True
                except compute_failure_exceptions:
                    _batch_length -= 1

            return function_return_containers

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
            checkpoint = torch.load(os.path.join(protein_mpnn_weights_dir, f'{model_name}.pt'),
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
dtype_map = dict(
    X=torch.float32,  # X,
    S=torch.long,  # S,
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
batch_params.pop(batch_params.index('tied_beta'))


def batch_proteinmpnn_input(size: int = None,
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
                            **kwargs) -> dict[str, np.ndarray]:
    # omit_AAs_np: np.ndarray = None, #
    # bias_AAs_np: np.ndarray = None, #
    # pssm_multi: np.ndarray = None, #
    # pssm_log_odds_flag: np.ndarray = None #
    # pssm_bias_flag: np.ndarray = None, #
    # tied_pos: np.ndarray = None, #
    # bias_by_res: np.ndarray = None, #
    """Set up all data for batches of proteinmpnn design

    Args:
        size: The number of inputs to use. If left blank, the size will be inferred from axis=0 of the X array
    Keyword Args:
        X: (numpy.ndarray) = None - The array specifying the parameter X of ProteinMPNN
        S: (numpy.ndarray) = None - The array specifying the parameter S of ProteinMPNN
        chain_mask: (numpy.ndarray) = None - The array specifying the parameter chain_mask of ProteinMPNN
        chain_encoding: (numpy.ndarray) = None - The array specifying the parameter chain_encoding of ProteinMPNN
        residue_idx: (numpy.ndarray) = None - The array specifying the parameter residue_idx of ProteinMPNN
        mask: (numpy.ndarray) = None - The array specifying the parameter mask of ProteinMPNN
        chain_M_pos: (numpy.ndarray) = None - The array specifying the parameter residue_mask of ProteinMPNN
        # residue_mask: (numpy.ndarray) = None - The array specifying the parameter residue_mask of ProteinMPNN
        omit_AA_mask: (numpy.ndarray) = None - The array specifying the parameter omit_AA_mask of ProteinMPNN
        pssm_coef: (numpy.ndarray) = None - The array specifying the parameter pssm_coef of ProteinMPNN
        pssm_bias: (numpy.ndarray) = None - The array specifying the parameter pssm_bias of ProteinMPNN
        pssm_log_odds_mask: (numpy.ndarray) = None - The array specifying the parameter pssm_log_odds_mask of ProteinMPNN
        bias_by_res: (numpy.ndarray) = None - The array specifying the parameter bias_by_res of ProteinMPNN
        tied_beta: (numpy.ndarray) = None - The array specifying the parameter tied_beta of ProteinMPNN
    Returns:
        A dictionary with each of the proteinmpnn parameters formatted in a batch
    """
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
        param = kwargs.get(key)
        if param is not None:
            device_kwargs[key] = np.tile(param, (size,) + (1,)*param.ndim)

    return device_kwargs

    # S = np.tile(S, (size,) + (1,)*S.ndim)
    # chain_mask = np.tile(chain_mask, (size,) + (1,)*chain_mask.ndim)
    # chain_encoding = np.tile(chain_encoding, (size,) + (1,)*chain_encoding.ndim)
    # residue_idx = np.tile(residue_idx, (size,) + (1,)*residue_idx.ndim)
    # mask = np.tile(mask, (size,) + (1,)*mask.ndim)
    # chain_M_pos = np.tile(chain_M_pos, (size,) + (1,)*chain_M_pos.ndim)  # residue_mask
    # omit_AA_mask = np.tile(omit_AA_mask, (size,) + (1,)*omit_AA_mask.ndim)
    # pssm_coef = np.tile(pssm_coef, (size,) + (1,)*pssm_coef.ndim)
    # pssm_bias = np.tile(pssm_bias, (size,) + (1,)*pssm_bias.ndim)
    # pssm_log_odds_mask = np.tile(pssm_log_odds_mask, (size,) + (1,)*pssm_log_odds_mask.ndim)
    # bias_by_res = np.tile(bias_by_res, (size,) + (1,)*bias_by_res.ndim)
    # # tied_beta = np.tile(tied_beta, (size,) + (1,)*tied_beta.ndim)
    #
    # return dict(X=X,
    #             S=S,
    #             chain_mask=chain_mask,
    #             chain_encoding=chain_encoding,
    #             residue_idx=residue_idx,
    #             mask=mask,
    #             chain_M_pos=residue_mask,
    #             omit_AA_mask=omit_AA_mask,
    #             pssm_coef=pssm_coef,
    #             pssm_bias=pssm_bias,
    #             pssm_log_odds_mask=pssm_log_odds_mask,
    #             bias_by_res=bias_by_res,
    #             # tied_beta=tied_beta
    #             )


def proteinmpnn_to_device(device: str = None,
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
                          **kwargs) -> dict[str, torch.Tensor]:
    # omit_AAs_np = kwargs.get('omit_AAs_np', None)
    # bias_AAs_np = kwargs.get('bias_AAs_np', None)
    # pssm_multi = kwargs.get('pssm_multi', None)
    # pssm_log_odds_flag = kwargs.get('pssm_log_odds_flag', None)
    # pssm_bias_flag = kwargs.get('pssm_bias_flag', None)
    # tied_pos = kwargs.get('tied_pos', None)
    # bias_by_res = kwargs.get('bias_by_res', None)
    """Set up all data to torch.Tensors for proteinmpnn design

    Args:
        device: The device to load tensors to
    Keyword Args:
        X: (numpy.ndarray) = None - The array specifying the parameter X of ProteinMPNN
        S: (numpy.ndarray) = None - The array specifying the parameter S of ProteinMPNN
        chain_mask: (numpy.ndarray) = None - The array specifying the parameter chain_mask of ProteinMPNN
        chain_encoding: (numpy.ndarray) = None - The array specifying the parameter chain_encoding of ProteinMPNN
        residue_idx: (numpy.ndarray) = None - The array specifying the parameter residue_idx of ProteinMPNN
        mask: (numpy.ndarray) = None - The array specifying the parameter mask of ProteinMPNN
        chain_M_pos: (numpy.ndarray) = None - The array specifying the parameter residue_mask of ProteinMPNN
        # residue_mask: (numpy.ndarray) = None - The array specifying the parameter residue_mask of ProteinMPNN
        omit_AA_mask: (numpy.ndarray) = None - The array specifying the parameter omit_AA_mask of ProteinMPNN
        pssm_coef: (numpy.ndarray) = None - The array specifying the parameter pssm_coef of ProteinMPNN
        pssm_bias: (numpy.ndarray) = None - The array specifying the parameter pssm_bias of ProteinMPNN
        pssm_log_odds_mask: (numpy.ndarray) = None - The array specifying the parameter pssm_log_odds_mask of ProteinMPNN
        bias_by_res: (numpy.ndarray) = None - The array specifying the parameter bias_by_res of ProteinMPNN
        tied_beta: (numpy.ndarray) = None - The array specifying the parameter tied_beta of ProteinMPNN
    Returns:
        The torch.Tensor proteinmpnn parameters
    """
    if device is None:
        raise ValueError('Must provide the desired device to load proteinmpnn')

    # Convert all numpy arrays to pytorch
    device_kwargs = {}
    for item, dtype in dtype_map.items():
        param = kwargs.get(item)
        if param is not None:
            device_kwargs[item] = torch.from_numpy(param).to(dtype=dtype, device=device)

    return device_kwargs
    # X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    # S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    # mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    # chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)                # residue_mask
    # chain_mask = torch.from_numpy(chain_mask).to(dtype=torch.float32, device=device)
    # chain_encoding = torch.from_numpy(chain_encoding).to(dtype=torch.long, device=device)
    # residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    # omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    # pssm_coef = torch.from_numpy(pssm_coef).to(dtype=torch.float32, device=device)
    # pssm_bias = torch.from_numpy(pssm_bias).to(dtype=torch.float32, device=device)
    # pssm_log_odds_mask = torch.from_numpy(pssm_log_odds_mask).to(dtype=torch.float32, device=device)
    # tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)
    # bias_by_res = torch.from_numpy(bias_by_res).to(dtype=torch.float32, device=device)
    # # omit_aas = torch.from_numpy(omit_aas).to(dtype=torch.float32, device=device)
    #
    # return dict(X=X,
    #             S=S,
    #             chain_mask=chain_mask,
    #             chain_encoding=chain_encoding,
    #             residue_idx=residue_idx,
    #             mask=mask,
    #             chain_M_pos=chain_M_pos,  # residue_mask,
    #             omit_AA_mask=omit_AA_mask,
    #             pssm_coef=pssm_coef,
    #             pssm_bias=pssm_bias,
    #             pssm_log_odds_mask=pssm_log_odds_mask,
    #             tied_beta=tied_beta,
    #             bias_by_res=bias_by_res
    #             )


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
