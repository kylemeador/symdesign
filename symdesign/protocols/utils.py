from __future__ import annotations

import functools
import logging
from typing import Callable, Type, Any

from symdesign.utils import path as putils

logger = logging.getLogger(__name__)
variance = 0.8
warn_missing_symmetry = \
    f'Cannot %s without providing symmetry! Provide symmetry with "--symmetry" or "--{putils.sym_entry}"'


def close_logs(func: Callable):
    """Wrap a function/method to close the functions first arguments .log attribute FileHandlers after use"""
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        func_return = func(self, *args, **kwargs)
        # adapted from https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile
        for handler in self.log.handlers:
            handler.close()
        return func_return
    return wrapped


def remove_structure_memory(func):
    """Decorator to remove large memory attributes from the instance after processing is complete"""
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        func_return = func(self, *args, **kwargs)
        if self.job.reduce_memory:
            self.pose = None
            self.entities.clear()
        return func_return
    return wrapped


def handle_design_errors(errors: tuple[Type[Exception], ...] = (Exception,)) -> Callable:
    """Wrap a function/method with try: except errors: and log exceptions to the functions first argument .log attribute

    This argument is typically self and is in a class with .log attribute

    Args:
        errors: A tuple of exceptions to monitor. Must be a tuple even if single exception
    Returns:
        Function return upon proper execution, else is error if exception raised, else None
    """
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs) -> Any:
            try:
                return func(self, *args, **kwargs)
            except errors as error:
                self.log.error(error)  # Allows exception reporting using self.log
                return error
        return wrapped
    return wrapper
