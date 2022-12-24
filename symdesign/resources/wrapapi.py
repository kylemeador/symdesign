from __future__ import annotations


import logging
import os
from pathlib import Path
from typing import Annotated, AnyStr

from symdesign.structure.sequence import MultipleSequenceAlignment, parse_hhblits_pssm
from ..sequence import read_fasta_file, write_sequence_to_fasta
from symdesign.structure.base import parse_stride
from symdesign.resources.database import Database, DataStore
from .query.pdb import query_entity_id, query_assembly_id, parse_entities_json, parse_assembly_json, query_entry_id, \
    parse_entry_json, _is_entity_thermophilic
from .query.uniprot import query_uniprot
from symdesign.utils import path as putils
# import dependencies.bmdca as bmdca

# Globals
logger = logging.getLogger(__name__)


class APIDatabase(Database):
    def __init__(self, stride: AnyStr | Path = None, sequences: AnyStr | Path = None,
                 hhblits_profiles: AnyStr | Path = None, pdb: AnyStr | Path = None,
                 uniprot: AnyStr | Path = None, **kwargs):
        # passed to Database
        # sql: sqlite = None, log: Logger = logger
        super().__init__(**kwargs)  # Database

        self.stride = DataStore(location=stride, extension='.stride', sql=self.sql, log=self.log,
                                load_file=parse_stride)
        self.sequences = DataStore(location=sequences, extension='.fasta', sql=self.sql, log=self.log,
                                   load_file=read_fasta_file, save_file=write_sequence_to_fasta)
        # elif extension == '.fasta' and msa:  # Todo if msa is in fasta format
        #  load_file = MultipleSequenceAlignment.from_fasta
        self.alignments = DataStore(location=hhblits_profiles, extension='.sto', sql=self.sql, log=self.log,
                                    load_file=MultipleSequenceAlignment.from_stockholm)
        # if extension == '.pssm':  # Todo for psiblast
        #  load_file = parse_pssm
        self.hhblits_profiles = DataStore(location=hhblits_profiles, extension='.hmm', sql=self.sql, log=self.log,
                                          load_file=parse_hhblits_pssm)
        self.pdb = PDBDataStore(location=pdb, extension='.json', sql=self.sql, log=self.log)
        self.uniprot = UniProtDataStore(location=uniprot, extension='.json', sql=self.sql, log=self.log)
        # self.bmdca_fields = \
        #     DataStore(location=hhblits_profiles, extension='_bmDCA%sparameters_h_final.bin' % os.sep,
        #     sql=self.sql, log=self.log)
        #  elif extension == f'_bmDCA{os.sep}parameters_h_final.bin':
        #      self.load_file = bmdca.load_fields
        #      self.save_file = not_implemented
        # self.bmdca_couplings = \
        #     DataStore(location=hhblits_profiles, extension='_bmDCA%sparameters_J_final.bin' % os.sep,
        #     sql=self.sql, log=self.log)
        #  elif extension == f'_bmDCA{os.sep}sparameters_J_final.bin':
        #      self.load_file = bmdca.load_couplings
        #      self.save_file = not_implemented
        self.sources = [self.stride, self.sequences, self.alignments, self.hhblits_profiles, self.pdb,
                        self.uniprot]


class APIDatabaseFactory:
    """Return a APIDatabase instance by calling the Factory instance with the APIDatabase source name

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the named APIDatabase
    """

    def __init__(self, **kwargs):
        # self._databases = {}
        self._database = None

    def __call__(self, source: str = os.path.join(os.getcwd(), f'{putils.program_name}{putils.data.title()}'),
                 sql: bool = False, **kwargs) -> APIDatabase:
        """Return the specified APIDatabase object singleton

        Args:
            source: The APIDatabase source path, or name if SQL database
            sql: Whether the APIDatabase is a SQL database
        Returns:
            The instance of the specified Database
        """
        # Todo potentially configure, however, we really only want a single database
        # database = self._databases.get(source)
        # if database:
        #     return database
        if self._database:
            return self._database
        elif sql:
            raise NotImplementedError('SQL set up has not been completed!')
        else:
            structure_info_dir = os.path.join(source, putils.structure_info)
            sequence_info_dir = os.path.join(source, putils.sequence_info)
            external_db = os.path.join(source, 'ExternalDatabases')
            # stride directory
            stride_dir = os.path.join(structure_info_dir, 'stride')
            # Todo only make paths if they are needed...
            putils.make_path(stride_dir)
            # sequence_info subdirectories
            sequences = os.path.join(sequence_info_dir, 'sequences')
            profiles = os.path.join(sequence_info_dir, 'profiles')
            putils.make_path(sequences)
            putils.make_path(profiles)
            # external database subdirectories
            pdb = os.path.join(external_db, 'pdb')
            putils.make_path(pdb)
            # pdb_entity_api = os.path.join(external_db, 'pdb_entity')
            # pdb_assembly_api = os.path.join(external_db, 'pdb_assembly')
            uniprot = os.path.join(external_db, 'uniprot')
            putils.make_path(uniprot)
            # self._databases[source] = APIDatabase(stride_dir, sequences, profiles, pdb, uniprot, sql=None)
            self._database = APIDatabase(stride_dir, sequences, profiles, pdb, uniprot, sql=None)

        # return self._databases[source]
        return self._database

    def get(self, **kwargs) -> APIDatabase:
        """Return the specified APIDatabase object singleton

        Keyword Args:
            source: str = 'current_working_directory/Data' - The APIDatabase source path, or name if SQL database
            sql: bool = False - Whether the Database is a SQL database
        Returns:
            The instance of the specified Database
        """
        return self.__call__(**kwargs)


api_database_factory: Annotated[APIDatabaseFactory,
                                'Calling this factory method returns the single instance of the Database class located'
                                ' at the "source" keyword argument'] = \
    APIDatabaseFactory()
"""Calling this factory method returns the single instance of the Database class located at the "source" keyword 
argument
"""


# class EntityDataStore(DataStore):
#     def retrieve_data(self, name: str = None, **kwargs) -> dict | None:
#         """Return data requested by PDB EntityID. Loads into the Database or queries the PDB API
#
#         Args:
#             name: The name of the data to be retrieved. Will be found with location and extension attributes
#         Returns:
#             If the data is available, the object requested will be returned, else None
#         """
#         data = super().retrieve_data(name=name)
#         #         data = getattr(self, name, None)
#         #         if data:
#         #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
#         #         else:
#         #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
#         #             if data:
#         #                 setattr(self, name, data)  # attempt to store the new data as an attribute
#         #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
#         #
#         #         return data
#         if not data:
#             request = query_entity_id(entity_id=name)
#             if not request:
#                 logger.warning(f'PDB API found no matching results for {name}')
#             else:
#                 data = request.json()
#                 # setattr(self, name, data)
#                 self.store_data(data, name=name)
#
#         return data
#
#
# class AssemblyDataStore(DataStore):
#     def retrieve_data(self, name: str = None, **kwargs) -> dict | None:
#         """Return data requested by PDB AssemblyID. Loads into the Database or queries the PDB API
#
#         Args:
#             name: The name of the data to be retrieved. Will be found with location and extension attributes
#         Returns:
#             If the data is available, the object requested will be returned, else None
#         """
#         data = super().retrieve_data(name=name)
#         #         data = getattr(self, name, None)
#         #         if data:
#         #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
#         #         else:
#         #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
#         #             if data:
#         #                 setattr(self, name, data)  # attempt to store the new data as an attribute
#         #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
#         #
#         #         return data
#         if not data:
#             request = query_assembly_id(assembly_id=name)
#             if not request:
#                 logger.warning(f'PDB API found no matching results for {name}')
#             else:
#                 data = request.json()
#                 # setattr(self, name, data)
#                 self.store_data(data, name=name)
#
#         return data


class PDBDataStore(DataStore):
    def __init__(self, location: str = None, extension: str = '.json', sql=None, log: logging.Logger = logger):
        super().__init__(location=location, extension=extension, sql=sql, log=log)
        # pdb_entity_api: AnyStr | Path = os.path.join(self.location, 'pdb_entity')
        # pdb_assembly_api: AnyStr | Path = os.path.join(self.location, 'pdb_assembly')
        # self.entity_api = EntityDataStore(location=pdb_entity_api, extension='.json', sql=self.sql, log=self.log)
        # self.assembly_api = AssemblyDataStore(location=pdb_assembly_api, extension='.json', sql=self.sql, log=self.log)
        # putils.make_path(pdb_entity_api)
        # putils.make_path(pdb_assembly_api)

    def is_thermophilic(self, name: str = None, **kwargs) -> bool:
        """Return whether the entity json entry in question is thermophilic. If no data is found, also returns False"""
        data = self.retrieve_entity_data(name=name)
        if data is None:
            return False
        return _is_entity_thermophilic(data)

    def retrieve_entity_data(self, name: str = None, **kwargs) -> dict | None:
        """Return data requested by PDB EntityID. Loads into the Database or queries the PDB API

        Args:
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        data = super().retrieve_data(name=name)
        #         data = getattr(self, name, None)
        #         if data:
        #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
        #         else:
        #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
        #             if data:
        #                 setattr(self, name, data)  # attempt to store the new data as an attribute
        #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
        #
        #         return data
        if not data:
            request = query_entity_id(entity_id=name)
            if not request:
                logger.warning(f'PDB API found no matching results for {name}')
            else:
                data = request.json()
                # setattr(self, name, data)
                self.store_data(data, name=name)

        return data

    def retrieve_assembly_data(self, name: str = None, **kwargs) -> dict | None:
        """Return data requested by PDB AssemblyID. Loads into the Database or queries the PDB API

        Args:
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        data = super().retrieve_data(name=name)
        #         data = getattr(self, name, None)
        #         if data:
        #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
        #         else:
        #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
        #             if data:
        #                 setattr(self, name, data)  # attempt to store the new data as an attribute
        #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
        #
        #         return data
        if not data:
            request = query_assembly_id(assembly_id=name)
            if not request:
                logger.warning(f'PDB API found no matching results for {name}')
            else:
                data = request.json()
                # setattr(self, name, data)
                self.store_data(data, name=name)

        return data

    def retrieve_data(self, entry: str = None, assembly_id: str = None, assembly_integer: int | str = None,
                      entity_id: str = None, entity_integer: int | str = None, chain: str = None, **kwargs) -> \
            dict | list[list[str]] | None:
        """Return data requested by PDB identifier. Loads into the Database or queries the PDB API

        Args:
            entry: The 4 character PDB EntryID of interest
            assembly_id: The AssemblyID to query with format (1ABC-1)
            assembly_integer: The particular assembly integer to query. Must include entry as well
            entity_id: The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
            entity_integer: The entity integer from the EntryID of interest
            chain: The polymer "chain" identifier otherwise known as the "asym_id" from the PDB EntryID of interest
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        if entry is not None:
            if len(entry) == 4:
                if entity_integer is not None:
                    # logger.debug(f'Querying PDB API with {entry}_{entity_integer}')
                    # data = self.entity_api.retrieve_data(name=f'{entry}_{entity_integer}')
                    # return parse_entities_json([self.entity_api.retrieve_data(name=f'{entry}_{entity_integer}')])
                    return parse_entities_json([self.retrieve_entity_data(name=f'{entry}_{entity_integer}')])
                elif assembly_integer is not None:
                    # logger.debug(f'Querying PDB API with {entry}-{assembly_integer}')
                    # data = self.assembly_api.retrieve_data(name=f'{entry}_{assembly_integer}')
                    # return parse_assembly_json(self.assembly_api.retrieve_data(name=f'{entry}-{assembly_integer}'))
                    return parse_assembly_json(self.retrieve_assembly_data(name=f'{entry}-{assembly_integer}'))
                else:
                    # logger.debug(f'Querying PDB API with {entry}')
                    # perform the normal DataStore routine with super(), however, finish with API call if no data found
                    data = super().retrieve_data(name=entry)
                    #         data = getattr(self, name, None)
                    #         if data:
                    #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
                    #         else:
                    #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
                    #             if data:
                    #                 setattr(self, name, data)  # attempt to store the new data as an attribute
                    #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
                    #
                    #         return data
                    if not data:
                        entry_request = query_entry_id(entry)
                        if not entry_request:
                            logger.warning(f'PDB API found no matching results for {entry}')
                        else:
                            data = entry_request.json()
                            # setattr(self, entry, data)
                            self.store_data(data, name=entry)

                    data = dict(entity=parse_entities_json([self.retrieve_entity_data(name=f'{entry}_{integer}')
                                                            for integer in range(1, int(data['rcsb_entry_info']
                                                                                        ['polymer_entity_count']) + 1)
                                                            ]),
                                **parse_entry_json(data))
                    if chain is not None:
                        integer = None
                        for entity_idx, chains in data.get('entity').items():
                            if chain in chains:
                                integer = entity_idx
                                break
                        if integer:
                            # logger.debug(f'Querying PDB API with {entry}_{integer}')
                            return self.retrieve_entity_data(name=f'{entry}_{integer}')
                        else:
                            raise KeyError(f'No chain "{chain}" found in PDB ID {entry}. Possible chains '
                                           f'{", ".join(ch for chns in data.get("entity", {}).items() for ch in chns)}')
                    else:  # provide the formatted PDB API Entry ID information
                        return data
            else:
                logger.warning(
                    f'EntryID "{entry}" is not of the required format and will not be found with the PDB API')
        elif assembly_id is not None:
            entry, assembly_integer, *extra = assembly_id.split('-')
            if not extra and len(entry) == 4:
                # logger.debug(f'Querying PDB API with {entry}-{assembly_integer}')
                # data = self.assembly_api.retrieve_data(name=f'{entry}-{assembly_integer}')
                return parse_assembly_json(self.retrieve_assembly_data(name=f'{entry}-{assembly_integer}'))

            logger.warning(
                f'AssemblyID "{entry}-{assembly_integer}" is not of the required format and will not be found '
                f'with the PDB API')

        elif entity_id is not None:
            entry, entity_integer, *extra = entity_id.split('_')
            if not extra and len(entry) == 4:
                # logger.debug(f'Querying PDB API with {entry}_{entity_integer}')
                # data = self.entity_api.retrieve_data(name=f'{entry}_{entity_integer}')
                return parse_entities_json([self.retrieve_entity_data(name=f'{entry}_{entity_integer}')])

            logger.warning(f'EntityID "{entry}_{entity_integer}" is not of the required format and will not be found '
                           'with the PDB API')

        else:  # this could've been passed as name=. This case would need to be solved with some parsing of the splitter
            raise RuntimeError(f'No valid arguments passed to {self.retrieve_data.__name__}. Valid arguments include: '
                               f'entry, assembly_id, assembly_integer, entity_id, entity_integer, chain')

        return None


class UniProtDataStore(DataStore):
    def __init__(self, location: str = None, extension: str = '.json', sql=None, log: logging.Logger = logger):
        super().__init__(location=location, extension=extension, sql=sql, log=log)

    def retrieve_data(self, name: str = None, **kwargs) -> dict | None:
        """Return data requested by UniProtID. Loads into the Database or queries the UniProt API

        Args:
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        if name is None:
            return None
        data = super().retrieve_data(name=name)
        #         data = getattr(self, name, None)
        #         if data:
        #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
        #         else:
        #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
        #             if data:
        #                 setattr(self, name, data)  # attempt to store the new data as an attribute
        #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
        #
        #         return data
        if not data:
            response = query_uniprot(uniprot_id=name)
            if not response:
                logger.warning(f'UniprotKB API found no matching results for {name}')
            else:
                data = response.json()
                self.store_data(data, name=name)

        return data

    def is_thermophilic(self, uniprot_id: str) -> int:
        """Query if a UniProtID is thermophilic

        Args:
            uniprot_id: The formatted UniProtID which consists of either a 6 or 10 character code
        Returns:
            1 if the UniProtID of interest has an organism lineage from a thermophilic taxa, else 0
        """
        data = self.retrieve_data(name=uniprot_id)
        for element in data.get('organism', {}).get('lineage', []):
            if 'thermo' in element.lower():
                return 1  # True

        return 0  # False
