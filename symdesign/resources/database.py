from __future__ import annotations

import logging
import os
from glob import glob
from logging import Logger
from typing import Any, Callable, AnyStr

from symdesign import utils

logger = logging.getLogger(__name__)
# for checking out the options to read and write Rosetta runs to a relational DB such as MySQL
# https://new.rosettacommons.org/docs/latest/rosetta_basics/options/Database-options


def read_file(file, **kwargs) -> list[AnyStr]:
    """The simplest form of parsing a file encoded in ASCII characters"""
    with open(file, 'r') as f:
        return f.readlines()


def write_str_to_file(string, file_name, **kwargs) -> AnyStr:
    """Use standard file IO to write a string to a file

    Args:
        string: The string to write
        file_name: The location of the file to write
    Returns:
        The name of the written file
    """
    with open(file_name, 'w') as f_save:
        f_save.write(f'{string}\n')

    return file_name


def write_list_to_file(_list, file_name, **kwargs) -> AnyStr:
    """Use standard file IO to write a string to a file

    Args:
        _list: The string to write
        file_name: The location of the file to write
    Returns:
        The name of the written file
    """
    with open(file_name, 'w') as f_save:
        lines = '\n'.join(map(str, _list))
        f_save.write(f'{lines}\n')

    return file_name


def not_implemented(data, file_name):
    raise NotImplemented(f'For save_file with {os.path.splitext(file_name)[-1]}DataStore method not available')
# lambda data, file_name: (_ for _ in ()).throw(NotImplemented(f'For save_file with {os.path.splitext(file_name)[-1]}'
#                                                              f'DataStore method not available'))


class DataStore:
    """

    Args:
        location: The location to store/retrieve data if directories are used
        extension: The extension of files to use during file handling. If extension is other than .txt or .json, the
            arguments load_file/save_file must be provided to handle storage
        load_file: Callable taking the file_name as first argument
        save_file: Callable taking the object to save as first argument and file_name as second argument
        sql: The database to use if the storage is based on a SQL database
        log: The Logger to handle operation reporting
    """
    location: str
    extension: str
    sql: None
    log: Logger
    load_file: Callable
    save_file: Callable

    def __init__(self, location: str = None, extension: str = '.txt', load_file: Callable = None,
                 save_file: Callable = None, sql=None, log: Logger = logger):
        self.log = log
        if sql is not None:
            self.sql = sql
        else:
            self.sql = sql
            self.location = location
            self.extension = extension

            if '.txt' in extension:  # '.txt' read the file and return the lines
                self.load_file = read_file
                self.save_file = write_list_to_file
            elif '.json' in extension:
                self.load_file = utils.read_json
                self.save_file = utils.write_json
            else:
                self.load_file = load_file if load_file else not_implemented
                self.save_file = save_file if save_file else not_implemented

    def make_path(self, condition: bool = True):
        """Make all required directories in specified path if it doesn't exist, and optional condition is True

        Args:
            condition: A condition to check before the path production is executed
        """
        if condition:
            os.makedirs(self.location, exist_ok=True)

    def path_to(self, name: str = '*') -> AnyStr:
        """Return the path_to of the storage location given an entity name"""
        return os.path.join(self.location, f'{name}{self.extension}')

    def retrieve_file(self, name: str) -> AnyStr | None:
        """Returns the actual location by combining the requested name with the stored .location"""
        path = self.path_to(name)
        files = sorted(glob(path))
        if files:
            file = files[0]
            if len(files) > 1:
                self.log.warning(f'Found more than one file at "{path}". Grabbing the first one: {file}')
            return file
        else:
            self.log.info(f'No file found for "{path}"')
            return None

    def retrieve_files(self) -> list:
        """Returns the actual location of all files in the stored .location"""
        path = self.path_to()
        files = sorted(glob(path))
        if not files:
            self.log.info(f'No files found in "{path}"')
        return files

    def retrieve_names(self) -> list[str]:
        """Returns the names of all objects in the stored .location"""
        path = self.path_to()
        names = list(map(os.path.basename, [os.path.splitext(file)[0] for file in sorted(glob(path))]))
        if not names:
            self.log.warning(f'No data found with names at "{path}"')
        return names

    def store_data(self, data: Any, name: str, **kwargs):
        """Store the data specfied by data and name to the DataStore. Saves the data as well

        Args:
            data: The data object to be stored with name
            name: The name of the data to be used
        Sets:
            self.name = data
        """
        setattr(self, name, data)
        self._save_data(data, name, **kwargs)

    def retrieve_data(self, name: str = None) -> object | None:
        """Return the data requested by name. Otherwise, load into the Database from a specified location

        Args:
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        data = getattr(self, name, None)
        if data:
            self.log.debug(f'{name}{self.extension} was retrieved from {type(self).__name__}')
        else:
            data = self._load_data(name, log=None)  # Attempt to retrieve the new data
            if data:
                setattr(self, name, data)  # Attempt to store the new data as an attribute
                self.log.debug(f'The file {name}{self.extension} was loaded into the {type(self).__name__}')
            else:
                self.log.debug(f'Failed to load file {name}{self.extension} into the {type(self).__name__}')

        return data

    def _save_data(self, data: Any, name: str, **kwargs) -> AnyStr | None:
        """Return the data located in a particular entry specified by name

        Args:
            data: The data object to be stored with name
            name: The name of the data to be used
        Returns:
            The name of the saved data if there was one or the return from the Database insertion
        """
        if self.sql:
            # dummy = True
            pass
            # return None
        else:
            if data:
                return self.save_file(data, self.path_to(name=name), **kwargs)  # Could also use self.retrieve_data()
        return None

    def _load_data(self, name: str, **kwargs) -> Any | None:
        """Return the data located in a particular entry specified by name

        Args:
            name: The name of the data to be used
        Returns:
            The data found at the requested name if any
        """
        if self.sql:
            dummy = True
        else:
            file = self.retrieve_file(name)
            if file:
                return self.load_file(file, **kwargs)
        return None

    def load_all_data(self, **kwargs):
        """Loads all data located in the particular DataStore storage location"""
        if self.sql:
            dummy = True
        else:
            for file in sorted(glob(os.path.join(self.location, f'*{self.extension}'))):
                # self.log.debug('Fetching %s' % file)
                setattr(self, os.path.splitext(os.path.basename(file))[0], self.load_file(file))


class Database:  # Todo ensure that the single object is completely loaded before multiprocessing... Queues and whatnot
    sources: list[DataStore]

    def __init__(self, sql=None, log: Logger = logger):  # sql: sqlite = None
        super().__init__()  # object
        if sql:
            raise NotImplementedError('SQL set up has not been completed!')
            self.sql = sql
        else:
            self.sql = sql

        self.log = log

    def load_all_data(self):
        """For every resource, acquire all existing data in memory"""
        for source in self.sources:
            try:
                source.load_all_data()
            except ValueError:
                raise ValueError(f'Issue loading data from source {source}')

    def source(self, name: str) -> DataStore:
        """Return on of the various DataStores supported by the Database

        Args:
            name: The name of the data source to use
        """
        try:
            return getattr(self, name)
        except AttributeError:
            raise AttributeError(f'There is no source named "{name}" found in the {type(self).__name__}. '
                                 f'Possible sources are: {", ".join(self.__dict__)}')

    def retrieve_data(self, source: str = None, name: str = None) -> object | None:
        """Return the data requested by name from the specified source. Otherwise, load into the Database from a
        specified location

        Args:
            source: The name of the data source to use
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        data = self.source(source).retrieve_data(name)
        return data

    def retrieve_file(self, source: str = None, name: str = None) -> AnyStr | None:
        """Retrieve the file specified by the source and identifier name

        Args:
            source: The name of the data source to use
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If the file is available, it will be returned, else None
        """
        return self.source(source).retrieve_file(name)
