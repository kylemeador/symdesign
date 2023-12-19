from __future__ import annotations

import functools
import logging
import os
import traceback
from subprocess import list2cmdline
from typing import Callable, Type, Any

from symdesign import flags
from symdesign.resources import distribute
from symdesign.utils import SymDesignException, ReportException, path as putils, starttime

# Globals
logger = logging.getLogger(__name__)
warn_missing_symmetry = \
    (f"Cannot %s without providing symmetry. Provide symmetry with '{flags.format_args(flags.symmetry_args)}' "
     f"or '{flags.format_args(flags.sym_entry_args)}'")


def close_logs(func: Callable):
    """Wrap a function/method to close the functions first arguments .log attribute FileHandlers after use"""
    @functools.wraps(func)
    def wrapped(job, *args, **kwargs):
        func_return = func(job, *args, **kwargs)
        # Adapted from https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile
        for handler in job.log.handlers:
            handler.close()
        return func_return
    return wrapped


def remove_structure_memory(func):
    """Decorator to remove large memory attributes from the instance after processing is complete"""
    @functools.wraps(func)
    def wrapped(job, *args, **kwargs):
        func_return = func(job, *args, **kwargs)
        if job.job.reduce_memory:
            job.clear_state()
        return func_return
    return wrapped


def handle_design_errors(errors: tuple[Type[Exception], ...] = (SymDesignException,)) -> Callable:
    """Wrap a function/method with try: except errors: and log exceptions to the functions first argument .log attribute

    This argument is typically self and is in a class with .log attribute

    Args:
        errors: A tuple of exceptions to monitor. Must be a tuple even if single exception
    Returns:
        Function return upon proper execution, else is error if exception raised, else None
    """
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapped(job, *args, **kwargs) -> Any:
            try:
                return func(job, *args, **kwargs)
            except errors as error:
                # Perform exception reporting using self.log
                job.report_exception(context=func.__name__)
                return ReportException(str(error))
        return wrapped
    return wrapper


def handle_job_errors(errors: tuple[Type[Exception], ...] = (SymDesignException,)) -> Callable:
    """Wrap a function/method with try/except `errors`

    Args:
        errors: A tuple of exceptions to monitor. Must be a tuple even if single exception
    Returns:
        Function return upon proper execution, else is error if exception raised, else None
    """
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapped(*args, **kwargs) -> list[Any]:
            try:
                return func(*args, **kwargs)
            except errors as error:
                # Perform exception reporting
                return [ReportException(str(error))]
        return wrapped
    return wrapper


def protocol_decorator(errors: tuple[Type[Exception], ...] = (SymDesignException,)) -> Callable:
    """Wrap a function/method with try: except errors: and log exceptions to the functions first argument .log attribute

    This argument is typically self and is in a class with .log attribute

    Args:
        errors: A tuple of exceptions to monitor. Must be a tuple even if single exception
    Returns:
        Function return upon proper execution, else is error if exception raised, else None
    """
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapped(job, *args, **kwargs) -> Any:
            # Todo
            #  Ensure that the below setting doesn't conflict with PoseJob inherent setting
            #  job.protocol = job.job.module
            # distribute_protocol()
            if job.job.distribute_work:
                # Skip any execution, instead create the command and add as job.current_script attribute
                base_cmd = list(putils.program_command_tuple) + job.job.get_parsed_arguments()
                base_cmd += ['--single', job.pose_directory]
                # cmd, *additional_cmds = getattr(job, f'get_cmd_{job.protocol}')()
                os.makedirs(job.scripts_path, exist_ok=True)
                job.current_script = distribute.write_script(
                    list2cmdline(base_cmd), name=f'{starttime}_{job.job.module}.sh', out_path=job.scripts_path,
                    # additional=[list2cmdline(_cmd) for _cmd in additional_cmds]
                )
                return None

            logger.info(f'Processing {func.__name__}({repr(job)})')
            # handle_design_errors()
            try:
                func_return = func(job, *args, **kwargs)
            except errors as error:
                # Perform exception reporting using job.log
                job.log.error(error)
                job.log.info(''.join(traceback.format_exc()))
                func_return = ReportException(str(error))
            # remove_structure_memory()
            if job.job.reduce_memory:
                job.clear_state()
            job.protocol = None
            # close_logs()
            # Adapted from https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile
            for handler in job.log.handlers:
                handler.close()

            return func_return
        return wrapped
    return wrapper
