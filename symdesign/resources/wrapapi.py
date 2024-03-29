from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, AnyStr

from sqlalchemy import Column, String
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import synonym, relationship

from . import sql
from .database import Database, DataStore
from .query.pdb import query_entity_id, query_assembly_id, parse_entities_json, parse_assembly_json, query_entry_id, \
    parse_entry_json, thermophilic_taxonomy_ids, thermophilicity_from_entity_json
from .query.uniprot import query_uniprot
from symdesign.sequence import MultipleSequenceAlignment, parse_hhblits_pssm, read_fasta_file, write_sequence_to_fasta
from symdesign.utils import path as putils
# import dependencies.bmdca as bmdca

# Globals
logger = logging.getLogger(__name__)


class APIDatabase(Database):
    """A Database which stores general API queries"""
    def __init__(self, sequences: AnyStr | Path = None,
                 hhblits_profiles: AnyStr | Path = None, pdb: AnyStr | Path = None,
                 uniprot: AnyStr | Path = None, **kwargs):
        """Construct the instance

        Args:
            sequences: The path the data stored for these particular queries
            hhblits_profiles: The path the data stored for these particular queries
            pdb: The path the data stored for these particular queries
            uniprot: The path the data stored for these particular queries
            **kwargs:
        """
        # passed to Database
        # sql: sqlite = None, log: Logger = logger
        super().__init__(**kwargs)  # Database

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

        self.sources = [self.sequences, self.alignments, self.hhblits_profiles, self.pdb, self.uniprot]


class APIDatabaseFactory:
    """Return a APIDatabase instance by calling the Factory instance with the APIDatabase source name

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the named APIDatabase
    """

    def __init__(self, **kwargs):
        self._database = None

    def destruct(self, **kwargs):
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
            sequence_info_dir = os.path.join(source, 'SequenceInfo')
            external_db = os.path.join(source, 'ExternalDatabases')
            # sequence_info subdirectories
            sequences = os.path.join(sequence_info_dir, 'sequences')
            profiles = os.path.join(sequence_info_dir, 'profiles')
            putils.make_path(sequences)
            putils.make_path(profiles)
            # external database subdirectories
            pdb = os.path.join(external_db, 'pdb')
            putils.make_path(pdb)
            uniprot = os.path.join(external_db, 'uniprot')
            putils.make_path(uniprot)
            # self._databases[source] = APIDatabase(sequences, profiles, pdb, uniprot, sql=None)
            self._database = APIDatabase(sequences, profiles, pdb, uniprot, sql=None)

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
#         #             data = self.load_data(name, log=None)  # attempt to retrieve the new data
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
#         #             data = self.load_data(name, log=None)  # attempt to retrieve the new data
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

    def entity_thermophilicity(self, name: str = None, **kwargs) -> float:  # bool:
        """Return the extent to which the EntityID in question is thermophilic

        Args:
            name: The EntityID
        Returns:
            Value ranging from 0-1 where 1 is completely thermophilic
        """
        # Todo make possible for retrieve_entry_data(name=name)
        data = self.retrieve_entity_data(name=name)
        if data is None:
            return False
        return thermophilicity_from_entity_json(data)

    def retrieve_entity_data(self, name: str = None, **kwargs) -> dict | None:
        """Return data requested by PDB EntityID. If in the Database, load, otherwise, query the PDB API and store

        Args:
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If data is available, the JSON object from PDB API will be returned, else None
        """
        data = super().retrieve_data(name=name)
        if not data:
            request = query_entity_id(entity_id=name)
            if not request:
                logger.warning(f'PDB API found no matching results for {name}')
            else:
                data = request.json()
                self.store_data(data, name=name)

        return data

    def retrieve_assembly_data(self, name: str = None, **kwargs) -> dict | None:
        """Return data requested by PDB AssemblyID. If in the Database, load, otherwise, query the PDB API and store

        Args:
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If data is available, the JSON object from PDB API will be returned, else None
        """
        data = super().retrieve_data(name=name)
        if not data:
            request = query_assembly_id(assembly_id=name)
            if not request:
                logger.warning(f'PDB API found no matching results for {name}')
            else:
                data = request.json()
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
            The possible return formats include:
            If entry
            {'entity':
                {'EntityID':
                    {'chains': ['A', 'B', ...],
                     'dbref': {'accession': ('Q96DC8',), 'db': 'UniProt'},
                     'reference_sequence': 'MSLEHHHHHH...',
                     'thermophilicity': 1.0},
                 ...}
             'method': xray,
             'res': resolution,
             'struct': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
             }
            If entity_id OR entry AND entity_integer
            {'EntityID':
                {'chains': ['A', 'B', ...],
                 'dbref': {'accession': ('Q96DC8',), 'db': 'UniProt'},
                 'reference_sequence': 'MSLEHHHHHH...',
                 'thermophilicity': 1.0},
             ...}
            If assembly_id OR entry AND assembly_integer
            [['A', 'A', 'A', ...], ...]
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
                    # Perform the normal DataStore routine with super(), however, finish with API call if no data found
                    data = super().retrieve_data(name=entry)
                    if not data:
                        entry_request = query_entry_id(entry)
                        if not entry_request:
                            logger.warning(f'PDB API found no matching results for {entry}')
                            return None
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
                    else:  # Provide the formatted PDB API Entry ID information
                        return data
            else:
                logger.debug(f"EntryID '{entry}' isn't the required format and will not be found with the PDB API")
        elif assembly_id is not None:
            try:
                entry, assembly_integer, *extra = assembly_id.split('-')
            except ValueError:  # Not enough values to unpack
                pass
            else:
                if not extra and len(entry) == 4:
                    # logger.debug(f'Querying PDB API with {entry}-{assembly_integer}')
                    return parse_assembly_json(self.retrieve_assembly_data(name=f'{entry}-{assembly_integer}'))

            logger.debug(f"AssemblyID '{assembly_id}' isn't the required format and will not be found with the PDB API")
        elif entity_id is not None:
            try:
                entry, entity_integer, *extra = entity_id.split('_')
            except ValueError:  # Not enough values to unpack
                pass
            else:
                if not extra and len(entry) == 4:
                    # logger.debug(f'Querying PDB API with {entry}_{entity_integer}')
                    return parse_entities_json([self.retrieve_entity_data(name=f'{entry}_{entity_integer}')])

            logger.debug(f"EntityID '{entity_id}' isn't the required format and will not be found with the PDB API")
        else:  # This could've been passed as name=. This case would need to be solved with some parsing of the splitter
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
        data = super().retrieve_data(name=name)
        if not data:
            response = query_uniprot(uniprot_id=name)
            if not response:
                logger.warning(f'UniprotKB API found no matching results for {name}')
            else:
                data = response.json()
                self.store_data(data, name=name)

        return data

    def thermophilicity(self, uniprot_id: str) -> float:
        """Query if a UniProtID is thermophilic

        Args:
            uniprot_id: The formatted UniProtID which consists of either a 6 or 10 character code
        Returns:
            1 if the UniProtID of interest is a thermophilic organism according to taxonomic classification, else 0
        """
        data = self.retrieve_data(name=uniprot_id)

        # Exact - parsing the taxonomic ID and cross-reference
        taxonomic_id = int(data.get('organism', {}).get('taxonId', -1))
        if taxonomic_id in thermophilic_taxonomy_ids:
            return 1.0

        # # Coarse - parsing the taxonomy for 'thermo'
        # for element in data.get('organism', {}).get('lineage', []):
        #     if 'thermo' in element.lower():
        #         return 1  # True

        return 0.0  # False


uniprot_accession_length = 10


class UniProtEntity(sql.Base):
    __tablename__ = 'uniprot_entity'
    id = Column(String(uniprot_accession_length), primary_key=True, autoincrement=False)
    """The UniProtID"""
    uniprot_id = synonym('id')
    # _uniprot_id = Column('uniprot_id', String)
    # entity_id = Column(String, nullable=False, index=True)  # entity_name is used in config.metrics
    # """This is a stand in for the StructureBase.name attribute"""

    # # Set up one-to-many relationship with entity_data table
    # entities = relationship('EntityData', back_populates='entity')
    # Set up many-to-many relationship with protein_metadata table
    # protein_metadata = relationship('ProteinMetadata', secondary='uniprot_protein_association',
    #                                 back_populates='uniprot_entities')
    protein_metadata = association_proxy('_protein_metadata', 'protein')
    _protein_metadata = relationship('UniProtProteinAssociation',
                                     back_populates='uniprot')
    _reference_sequence: str

    @property
    def reference_sequence(self) -> str:
        """Get the sequence from the UniProtID"""
        try:
            return self._reference_sequence
        except AttributeError:
            api_db = api_database_factory()
            response_json = api_db.uniprot.retrieve_data(name=self.id)
            if response_json is not None:
                sequence = response_json.get('sequence')
                if sequence:
                    self._reference_sequence = sequence['value']
                    return self._reference_sequence
                else:
                    logger.error(f"Couldn't find UniProt data for {self.id}")

            # else:  # uniprot_id found no data from UniProt API
            # Todo this isn't correct due to many-to-many association
            max_seq_len = 0
            for data in self.protein_metadata:
                seq_len = len(data.reference_sequence)
                if seq_len > max_seq_len:
                    max_seq_len = seq_len
                    _reference_sequence = data.reference_sequence
            self._reference_sequence = _reference_sequence
            return self._reference_sequence
