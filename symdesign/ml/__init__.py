from __future__ import annotations

import functools
import logging
import time
import traceback
from math import ceil
from typing import Any, Callable, MutableMapping, Type

import numpy as np
import torch

from symdesign import utils

logger = logging.getLogger(__name__)
FeatureDict = MutableMapping[str, np.ndarray]


def get_device_memory(device: torch.device | int | str | None, free: bool = False) -> int:
    """Get the memory available for a requested device to calculate computational constraints

    Args:
        device: The current device of the pytorch model in question
        free: Whether to return the free memory if the device is a cuda GPU, otherwise return all pytorch memory

    Returns:
        The bytes of memory available
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)

    if device.type == 'cpu':  # device is None or
        memory_constraint = utils.get_available_memory()
        logger.debug(f'The available cpu memory is: {memory_constraint}')
    else:
        free_memory, gpu_memory_total = torch.cuda.mem_get_info(device)
        logger.debug(f'The available gpu memory is: {free_memory}')
        memory_reserved = torch.cuda.memory_reserved(device)
        logger.debug(f'The reserved gpu memory is: {memory_reserved}')
        if free:
            memory_constraint = free_memory
        else:
            memory_constraint = free_memory + memory_reserved

    return memory_constraint


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
#
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
                            raise ValueError(
                                f'The batch_length ({batch_length}) must be greater than 0')
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
                                raise KeyError(
                                    f"Couldn't return the data specified by {return_key} to the return_container with "
                                    f"keys:{', '.join(return_containers.keys())}")
                            except ValueError as error:  # Arrays are incorrectly sized
                                raise ValueError(
                                    f"Couldn't return the data specified by {return_key} from {func.__name__} due to: "
                                    f"{error}")
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
                raise utils.SymDesignException(
                    f"{func.__name__}() wasn't able to be executed. See the above traceback")

            return return_containers
        return wrapped
    return wrapper


def sequence_nllloss(sequence: torch.Tensor, log_probs: torch.Tensor,
                     mask: torch.Tensor = None, per_residue: bool = True) -> torch.Tensor:
    """Score designed sequences using the Negative log likelihood loss function

    Args:
        sequence: The sequence tensor
        log_probs: The logarithmic probabilities at each residue for every amino acid.
            This may be the result of a 'profile' such as that generated by a forward pass through ProteinMPNN
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


# Todo resolve this circular import error...
# from . import alphafold, proteinmpnn
from . import proteinmpnn
