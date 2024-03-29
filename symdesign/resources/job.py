from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from copy import deepcopy
from itertools import repeat
from typing import Annotated, Any, AnyStr

import jax
import psutil
import tensorflow as tf
from tqdm.auto import tqdm
from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from . import config, distribute, sql, structure_db, wrapapi
from symdesign import flags, sequence, structure, utils
from symdesign.sequence import hhblits
from symdesign.utils import SymEntry, path as putils

logger = logging.getLogger(__name__)
gb_divisior = 1_000_000_000


def generate_sequence_mask(fasta_file: AnyStr) -> list[int]:
    """From a sequence with a design_selector, grab the residue indices that should be designed in the target
    structural calculation

    Args:
        fasta_file: The path to a file with fasta information

    Returns:
        The residue numbers (in pose format) that should be ignored in design
    """
    sequence_and_mask = list(sequence.read_fasta_file(fasta_file))
    # sequences = sequence_and_mask
    _sequence, mask, *_ = sequence_and_mask
    if not len(_sequence) == len(mask):
        raise ValueError(
            'The sequence and design_selector are different lengths. Please correct the alignment before proceeding')

    return [idx for idx, aa in enumerate(mask, 1) if aa != '-']


def generate_chain_mask(chains: str) -> set[str]:
    """From a string with a design_selection, format the chains provided

    Args:
        chains: The specified chains separated by commas to split
    Returns:
        The provided chain ids in pose format
    """
    return set(utils.clean_comma_separated_string(chains))


def parse_design_selector_flags(
        # select_designable_residues_by_sequence: str = None,
        # select_designable_residues_by_pdb_number: str = None,
        # select_designable_residues_by_pose_number: str = None,
        # select_designable_chains: str = None,
        # mask_designable_residues_by_sequence: str = None,
        # mask_designable_residues_by_pdb_number: str = None,
        # mask_designable_residues_by_pose_number: str = None,
        # mask_designable_chains: str = None,
        # require_design_by_pdb_number: str = None,
        # require_design_by_pose_number: str = None,
        # require_design_by_chain: str = None,
        design_entities: str = None,
        design_chains: str = None,
        design_residues: str = None,
        mask_entities: str = None,
        mask_chains: str = None,
        mask_residues: str = None,
        require_entities: str = None,
        require_chains: str = None,
        require_residues: str = None,
        **kwargs: dict[str, Any]) -> structure.model.PoseSpecification:

    # Todo move to a verify design_selectors function inside of Pose? Own flags module?
    #  Pull mask_design_using_sequence out of flags
    # -------------------
    # if select_designable_residues_by_sequence is not None:
    #     residue_select = residue_select.union(generate_sequence_mask(select_designable_residues_by_sequence))
    # if select_designable_residues_by_pose_number is not None:
    #     residue_select = residue_select.union(utils.format_index_string(select_designable_residues_by_pose_number))

    if design_entities is not None:
        entity_select = set(design_entities)
    else:
        entity_select = set()

    if design_residues is not None:
        residue_select = set(utils.format_index_string(design_residues))
    else:
        residue_select = set()

    if design_chains is not None:
        chain_select = set(generate_chain_mask(design_chains))
    else:
        chain_select = set()
    # -------------------
    # if mask_designable_residues_by_sequence is not None:
    #     residue_mask = residue_mask.union(generate_sequence_mask(mask_designable_residues_by_sequence))
    if mask_entities is not None:
        entity_mask = set(mask_entities)
    else:
        entity_mask = set()

    if mask_residues is not None:
        residue_mask = set(utils.format_index_string(mask_residues))
    else:
        residue_mask = set()

    if mask_chains is not None:
        chain_mask = set(generate_chain_mask(mask_chains))
    else:
        chain_mask = set()
    # -------------------
    if require_entities is not None:
        entity_req = set(require_entities)
    else:
        entity_req = set()

    if require_residues is not None:
        residues_req = set(utils.format_index_string(require_residues))
    else:
        residues_req = set()

    if require_chains is not None:
        chain_req = set(generate_chain_mask(require_chains))
    else:
        chain_req = set()

    return structure.model.PoseSpecification(
        selection=dict(entities=entity_select, chains=chain_select, residues=residue_select),
        mask=dict(entities=entity_mask, chains=chain_mask, residues=residue_mask),
        required=dict(entities=entity_req, chains=chain_req, residues=residues_req))


def format_args_for_namespace(args_: dict[str, Any], namespace: str, flags_: Iterable[str]) -> dict[str, Any]:
    namespace_args = {}
    for flag in flags_:
        try:
            arg_ = args_[flag]  # .pop(flag)  # , None)  # get(flag)
        except KeyError:  # No flag is here
            continue
        else:  # logger.debug('flag', flag, 'arg', arg_)
        # if arg_ is not None:
            # Replace the arg destination with the plain flag, no namespace prefix
            namespace_args[flag.replace(f'{namespace}_', '')] = arg_

    return namespace_args


@dataclasses.dataclass
class FlagsBase:
    namespace: str

    @classmethod
    def from_flags(cls, **kwargs):
        return cls(**format_args_for_namespace(kwargs, cls.namespace, flags.namespaces[cls.namespace]))
        # cls_parameters = inspect.signature(cls).parameters
        # return cls(**{key: value for key, value in kwargs.items()
        #               if key in cls_parameters})
# def from_flags(cls, **kwargs):
#     return cls(**{key: value for key, value in kwargs.items()
#                   if key in inspect.signature(cls).parameters})


# These dataclasses help simplify the use of flags into namespaces
# Due to an error evaluating the singleton eval(type(None).__name__), need to pass a globals argument
nonetype_map = {'NoneType': None}
Cluster = dataclasses.make_dataclass(
    'Cluster',
    [(flag, eval(type(default).__name__, nonetype_map), dataclasses.field(default=default))
     for flag, default in flags.cluster_defaults.items()],
    bases=(FlagsBase,))
#     frozen=True)
Design = dataclasses.make_dataclass(
    'Design',
    [(flag, eval(type(default).__name__, nonetype_map), dataclasses.field(default=default))
     for flag, default in flags.design_defaults.items()],
    bases=(FlagsBase,))
#     frozen=True)
Dock = dataclasses.make_dataclass(
    'Dock',
    [(flag, eval(type(default).__name__, nonetype_map), dataclasses.field(default=default))
     for flag, default in flags.dock_defaults.items()],
    bases=(FlagsBase,))
#     frozen=True)
Init = dataclasses.make_dataclass(
    'Init',
    [(flag, eval(type(default).__name__, nonetype_map), dataclasses.field(default=default))
     for flag, default in flags.init_defaults.items()],
    bases=(FlagsBase,))
#     frozen=True)
Predict = dataclasses.make_dataclass(
    'Predict',
    [(flag, eval(type(default).__name__, nonetype_map), dataclasses.field(default=default))
     for flag, default in flags.predict_defaults.items()],
    bases=(FlagsBase,))
#     frozen=True)


class DBInfo:
    def __init__(self, location: AnyStr, echo: bool = False):
        self.location = location
        self.engine: Engine = create_engine(self.location, echo=echo, future=True)
        self.session: sessionmaker = sessionmaker(self.engine, future=True)

        if 'sqlite' in self.location:
            # The below functions are recommended to help overcome issues with SQLite transaction scope
            # See: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#pysqlite-serializable
            @event.listens_for(self.engine, 'connect')
            def do_connect(dbapi_connection, connection_record):
                """Disable pysqlite's emitting of the BEGIN statement entirely.
                Also stops it from emitting COMMIT before any DDL
                """
                dbapi_connection.isolation_level = None

            @event.listens_for(self.engine, 'begin')
            def do_begin(conn):
                """Emit our own BEGIN"""
                conn.exec_driver_sql('BEGIN')


class JobResources:
    """The intention of JobResources is to serve as a singular source of design info which is common across all
    jobs. This includes common paths, databases, and design flags which should only be set once in program operation,
    then shared across all member designs"""
    _construct_pose: bool
    _input_source: str | list[str] | None
    _location: str | None
    _modules: list[str]
    _output_directory: AnyStr | None
    _session: Session | None
    db: DBInfo | None
    reduce_memory: bool = False

    def __init__(self, program_root: AnyStr = None, arguments: argparse.Namespace = None, initial: bool = False,
                 **kwargs):
        """Parse the program operation location, ensure paths to these resources are available, and parse arguments

        Args:
            program_root: The root location of program operation
            arguments: The argparse.Namespace object with associated program flags
            initial: Whether this is the first instance of the particular program output
        """
        try:
            if os.path.exists(program_root):
                self.program_root = program_root
            else:
                raise FileNotFoundError(
                    f"Path doesn't exist\n\t{program_root}")
        except TypeError:
            raise TypeError(
                f"Can't initialize {JobResources.__name__} without parameter 'program_root'")

        # Format argparse.Namespace arguments
        if arguments is not None:
            kwargs.update(deepcopy(vars(arguments)))

        # Set the module for the current job. This will always be a '-' separated string when more than one name
        self.module: str = kwargs.get(flags.module)
        # Ensure that the protocol is viable
        if self.module == flags.protocol:
            self.protocol_module = True
            self.modules = kwargs.get(flags.modules)
        else:
            self.protocol_module = False
            # Instead of setting this, let self.module be used dynamically with property
            # self.modules = [self.module]

        # Computing environment and development Flags
        # self.command_only: bool = kwargs.get('command_only', False)
        # """Whether to reissue commands, only if distribute_work=False"""
        self.log_level: bool = kwargs.get(flags.log_level._)
        self.debug: bool = True if self.log_level == logging.DEBUG else False
        self.force: bool = kwargs.get(flags.force._)
        self.development: bool = kwargs.get(flags.development._)
        self.profile_memory: bool = kwargs.get(flags.profile_memory._)
        if self.profile_memory and not self.development:
            logger.warning(f"{flags.profile_memory.long} was set but {flags.development.long} wasn't")

        self.mpi: int = kwargs.get(flags.mpi)
        if self.mpi is None:
            self.mpi = 0
            self.distribute_work: bool = kwargs.get(flags.distribute_work._)
            # # Todo implement, see symdesign.utils and CommandDistributor
            # # extras = ' mpi {CommmandDistributer.mpi}'
            # number_mpi_processes = CommmandDistributer.mpi - 1
            # logger.info('Setting job up for submission to MPI capable computer. Pose trajectories run in parallel, '
            #             f'{number_mpi_processes} at a time. This will speed up processing ~
            #             f'{job.design.number / number_mpi_processes:2f}-fold.')
        else:  # self.mpi > 0
            self.distribute_work = True
            raise NotImplementedError(
                f"Can't compute the number of resources to allocate using {flags.mpi.long} yet...")

        self.multi_processing: int = kwargs.get(flags.multi_processing._)
        if self.multi_processing:
            # Calculate the number of cores to use depending on computer resources
            self.cores = utils.calculate_mp_cores(cores=kwargs.get(flags.cores))  # Todo mpi=self.mpi
        else:
            self.cores: int = 1
        self.threads = self.cores * 2
        self.gpu_available = False

        # Input parameters
        self.project_name = kwargs.get(flags.project_name._)
        # program_root subdirectories
        self.data = os.path.join(self.program_root, putils.data.title())
        self.projects = os.path.join(self.program_root, putils.projects)
        self.job_paths = os.path.join(self.program_root, 'JobPaths')
        self.sbatch_scripts = os.path.join(self.program_root, 'Scripts')
        self.all_scores = os.path.join(self.program_root, 'AllScores')

        self.api_db = wrapapi.api_database_factory.get(source=self.data)
        self.sequences = self.api_db.sequences.location
        self.profiles = self.api_db.hhblits_profiles.location
        self.pdb_api = self.api_db.pdb.location
        self.uniprot_api = self.api_db.uniprot.location
        # data subdirectories
        self.clustered_poses = os.path.join(self.data, 'ClusteredPoses')
        self.structure_db = structure_db.structure_database_factory.get(
            source=os.path.join(self.data, 'StructureInfo'))
        self.pdb_dir = self.structure_db.models.location  # Used to store downloaded PDB's
        # pdb_dir subdirectories
        self.orient_dir = self.structure_db.oriented.location
        self.orient_asu_dir = self.structure_db.oriented_asu.location
        self.refine_dir = self.structure_db.refined.location
        self.full_model_dir = self.structure_db.full_models.location
        self.stride_db_dir = self.structure_db.stride.location

        # Set the job instance on these db objects
        self.api_db.job = self
        self.structure_db.job = self
        self.fragment_source = kwargs.get(flags.fragment_source._)
        self.fragment_db: structure.fragment.db.FragmentDatabase | None = None

        default_db = f'sqlite:///{os.path.join(self.data, f"{putils.program_name}.db")}'
        self.db_config = os.path.join(self.data, 'db.cfg')
        database_url = kwargs.get(flags.database_url._)
        if initial:
            if database_url is None:
                database_url = default_db
            db_cfg = {'url': database_url}
            with open(self.db_config, 'w') as f:
                json.dump(db_cfg, f)
        else:
            if os.path.exists(self.db_config):
                with open(self.db_config, 'r') as f:
                    db_cfg = json.load(f)
                if database_url is not None:
                    raise utils.InputError(
                        f"The {flags.database_url.long} '{database_url}' can't be used as this {putils.program_output} "
                        f"was already initialized with the url='{db_cfg.get('url')}")
                else:
                    database_url = db_cfg.get('url')
            else:  # This should always exist
                database_url = default_db

        self.database_url = database_url
        self.debug_db = kwargs.get('debug_db')
        self.db: DBInfo = DBInfo(self.database_url, echo=self.debug_db)
        if initial:  # if not os.path.exists(self.internal_db):
            # Emit CREATE TABLE DDL
            sql.Base.metadata.create_all(self.db.engine)
        self.load_to_db = kwargs.get(flags.load_to_db._)
        self.reset_db = kwargs.get(flags.reset_db._)
        if self.reset_db:
            response = input(f"All database information will be wiped if you proceed. Enter 'YES' to proceed"
                             f"{utils.query.input_string}")
            if response == 'YES':
                logger.warning(f'Dropping all tables and data from DB')
                # All tables are deleted
                sql.Base.metadata.drop_all(self.db.engine)
                # Emit CREATE TABLE DDL
                sql.Base.metadata.create_all(self.db.engine)
            else:
                logger.info(f'Skipping {flags.format_args(flags.reset_db)}')
                pass
        # else:  # When --no-database is provided as a flag
        #     self.db = None

        # PoseJob initialization flags
        self.init = Init.from_flags(**kwargs)
        self.specify_entities = kwargs.get(flags.specify_entities._)
        # self.init.pre_refined
        # self.init.pre_loop_modeled
        # self.init.refine_input
        # self.init.loop_model_input

        # self.preprocessed = kwargs.get(flags.preprocessed)
        # if self.init.pre_loop_modeled or self.init.pre_refined:
        #     self.preprocessed = True
        # else:
        #     self.preprocessed = False
        self.range = kwargs.get(flags.range_._)
        if self.range is not None:
            try:
                self.low, self.high = map(float, self.range.split('-'))
            except ValueError:  # Didn't unpack correctly
                raise ValueError(
                    f'The {flags.format_args(flags.range_args)} flag must take the form "LOWER-UPPER"')
        else:
            self.low = self.high = None
        # Program flags
        # self.consensus: bool = kwargs.get(consensus, False)  # Whether to run consensus
        self.background_profile: str = kwargs.get(flags.background_profile._)
        """The type of position specific profile (per-residue amino acid frequencies) to utilize as the design 
        background profile. 
        Choices include putils.design_profile, putils.evolutionary_profile, and putils.fragment_profile
        """
        # Process design_selector
        self.design_selector: PoseSpecification = parse_design_selector_flags(**kwargs)

        self.update_metadata = kwargs.get(flags.update_metadata._)
        self.component1 = kwargs.get(flags.component1._)
        self.query_codes = kwargs.get(flags.query_codes._)
        pdb_codes = kwargs.get(flags.pdb_code._, kwargs.get('target_pdb_code'))
        if pdb_codes:
            # Collect all provided codes required for component 1 processing
            codes = []
            for code_or_file in pdb_codes:
                codes.extend(utils.to_iterable(code_or_file))
            self.pdb_codes = utils.remove_duplicates(codes)
        else:
            self.pdb_codes = None

        self.component2 = kwargs.get(flags.component2._)
        self.query_codes2 = kwargs.get('query_codes2')
        pdb_codes2 = kwargs.get('pdb_code2', kwargs.get('aligned_pdb_code'))
        if pdb_codes2:
            # Collect all provided codes required for component 1 processing
            codes = []
            for code_or_file in pdb_codes2:
                codes.extend(utils.to_iterable(code_or_file))
            self.pdb_codes2 = utils.remove_duplicates(codes)
        else:
            self.pdb_codes2 = None

        # Docking flags
        self.dock = Dock.from_flags(**kwargs)
        if self.development:
            self.dock.quick = True
        if self.dock.perturb_dof or self.dock.perturb_dof_rot or self.dock.perturb_dof_tx:
            # Check if no other values were set and set them if so
            if not self.dock.perturb_dof_rot and not self.dock.perturb_dof_tx:
                # Set all perturb_dof on and set to the provided default
                self.dock.perturb_dof_rot = self.dock.perturb_dof_tx = True
                if self.dock.perturb_dof_steps is None:
                    self.dock.perturb_dof_steps_rot = self.dock.perturb_dof_steps_tx = flags.default_perturbation_steps
                else:
                    self.dock.perturb_dof_steps_rot = self.dock.perturb_dof_steps_tx = self.dock.perturb_dof_steps
            else:  # Parse the provided values
                self.dock.perturb_dof = True
                if self.dock.perturb_dof_rot:
                    if self.dock.perturb_dof_steps_rot is None:
                        self.dock.perturb_dof_steps_rot = flags.default_perturbation_steps
                else:
                    self.dock.perturb_dof_steps_rot = 1

                if self.dock.perturb_dof_tx:
                    if self.dock.perturb_dof_steps_tx is None:
                        self.dock.perturb_dof_steps_tx = flags.default_perturbation_steps
                else:
                    self.dock.perturb_dof_steps_tx = 1
        else:  # None provided, set the unavailable dof to 1 step and warn if one was provided
            if self.dock.perturb_dof_steps is not None:
                logger.warning(f"Couldn't use the flag {flags.perturb_dof_steps.long} as {flags.perturb_dof.long}"
                               f" wasn't set")
            if self.dock.perturb_dof_steps_rot is not None:
                logger.warning(f"Couldn't use the flag {flags.perturb_dof_steps_rot.long} as "
                               f"{flags.perturb_dof_rot.long} wasn't set")
            if self.dock.perturb_dof_steps_tx is not None:
                logger.warning(f"Couldn't use the flag {flags.perturb_dof_steps_tx.long} as {flags.perturb_dof_tx.long}"
                               f" wasn't set")
            self.dock.perturb_dof_steps = self.dock.perturb_dof_steps_rot = self.dock.perturb_dof_steps_tx = 1

        # dock_weight = kwargs.get('weight')
        # dock_weight_file = kwargs.get('weight_file')
        if self.dock.weight or self.dock.weight_file is not None:
            self.dock.weight = flags.parse_weights(self.dock.weight, file=self.dock.weight_file)
        # No option to get filters on the fly...
        # elif self.dock.weight is not None:  # --dock-weight was provided, but as a boolean-esq. Query the user
        #     self.dock.weight = []
        else:
            self.dock.weight = None
        if self.dock.filter or self.dock.filter_file is not None:
            self.dock.filter = flags.parse_filters(self.dock.filter, file=self.dock.filter_file)
        # No option to get filters on the fly...
        # elif self.dock.weight is not None:  # --dock-weight was provided, but as a boolean-esq. Query the user
        #     self.dock.weight = []
        else:
            self.dock.filter = None
        # self.proteinmpnn_score: bool = kwargs.get('proteinmpnn_score', False)
        # self.contiguous_ghosts: bool = kwargs.get('contiguous_ghosts', False)

        # self.rotation_step1: bool = kwargs.get('rotation_step1', False)
        # self.rotation_step2: bool = kwargs.get('rotation_step2', False)
        # self.min_matched: bool = kwargs.get('min_matched', False)
        # self.match_value: bool = kwargs.get('match_value', False)
        # self.initial_z_value: bool = kwargs.get('initial_z_value', False)

        self.fuse_chains: list[tuple[str]] = [tuple(pair.split(':')) for pair in kwargs.get(flags.fuse_chains._, [])]

        self.interface_distance = kwargs.get(flags.interface_distance._)
        self.interface = kwargs.get(flags.interface._)
        self.interface_only = kwargs.get(flags.interface_only._)
        self.oligomeric_interfaces = kwargs.get(flags.oligomeric_interfaces._)
        self.use_proteinmpnn = kwargs.get(flags.use_proteinmpnn._)
        self.use_evolution = kwargs.get(flags.use_evolution._)
        # Explicitly set to false if not designing or predicting
        use_evolution_modules = [
            flags.nanohedra, flags.initialize_building_blocks, flags.refine, flags.interface_metrics,
            flags.process_rosetta_metrics, flags.analysis, flags.predict_structure, flags.design
        ]
        if self.use_evolution and not any([module in use_evolution_modules for module in self.modules]):
            logger.info(f'Setting {flags.format_args(flags.use_evolution_args)} to False as no module '
                        'requesting evolutionary information is utilized')
            self.use_evolution = False

        # Design flags
        self.design = Design.from_flags(**kwargs)
        if self.design.ignore_clashes:
            self.design.ignore_pose_clashes = self.design.ignore_symmetric_clashes = True
        # Handle protocol specific flags
        if self.module == flags.interface_design:  # or self.design.neighbors:
            # Handle interface-design module alias
            self.module = flags.design
            self.design.interface = True
        if self.design.method == putils.consensus:
            self.design.term_constraint = True
        if self.design.term_constraint:
            self.generate_fragments: bool = True
        else:
            self.generate_fragments = False

        if self.design.structure_background:
            self.design.evolution_constraint = False
            self.design.hbnet = False
            self.design.scout = False
            self.design.term_constraint = False

        # if self.design.evolution_constraint and flags.design not in self.modules:
        #     logger.debug(f'Setting {flags.format_args(flags.evolution_constraint_args)} to False as the no module '
        #                  f'requesting evolutionary information is utilized')
        #     self.design.evolution_constraint = False

        # self.dock_only: bool = kwargs.get('dock_only')
        # if self.dock_only:
        #     self.design.sequences = self.design.structures = False
        self.only_write_frag_info: bool = kwargs.get(flags.only_write_frag_info._)
        self.increment_chains: bool = kwargs.get(flags.increment_chains._)
        self.interface_to_alanine: bool = kwargs.get(flags.interface_to_alanine._)
        self.metrics: bool = kwargs.get(flags.metrics._)
        self.measure_pose: str = kwargs.get(flags.measure_pose._)
        self.specific_protocol: str = kwargs.get(flags.specific_protocol._)
        # Process symmetry
        sym_entry_number = kwargs.get(flags.sym_entry._)
        symmetry = kwargs.get(flags.symmetry._)
        if sym_entry_number is None and symmetry is None:
            self.sym_entry: SymEntry.SymEntry | str | None = None
        else:
            if symmetry and utils.symmetry.CRYST in symmetry.upper():
                # Later, symmetry information will be retrieved from the file header
                self.sym_entry = SymEntry.CrystRecord  # Input was provided as 'cryst'
            else:
                self.sym_entry = SymEntry.parse_symmetry_to_sym_entry(
                    sym_entry_number=sym_entry_number, symmetry=symmetry)

        # Selection flags
        self.save_total = kwargs.get(flags.save_total._)
        # self.total = kwargs.get('total')
        self.protocol = kwargs.get(flags.protocol._)
        _filter = kwargs.get(flags.filter_._)
        _filter_file = kwargs.get(flags.filter_file._)
        if _filter == list():
            # --filter was provided, but as a boolean-esq. Query the user once there is a df
            self.filter = True
        elif _filter or _filter_file is not None:
            self.filter = flags.parse_filters(_filter, file=_filter_file)
        else:
            self.filter = None
        _weight = kwargs.get(flags.weight._)
        _weight_file = kwargs.get(flags.weight_file._)
        if _weight == list():
            # --weight was provided, but as a boolean-esq. Query the user once there is a df
            self.weight = True
        elif _weight or _weight_file is not None:
            self.weight = flags.parse_weights(_weight, file=_weight_file)
        else:
            self.weight = None
        self.weight_function = kwargs.get(flags.weight_function._)
        self.select_number = kwargs.get(flags.select_number._)
        self.designs_per_pose = kwargs.get(flags.designs_per_pose._)
        # self.allow_multiple_poses = kwargs.get('allow_multiple_poses')
        self.tag_entities = kwargs.get(flags.tag_entities._)
        self.specification_file = kwargs.get(flags.specification_file._)
        # Don't need this at the moment...
        # self.poses = kwargs.get(flags.poses)
        """Used to specify whether specific designs should be fetched for select_* modules"""
        self.dataframe = kwargs.get(flags.dataframe._)

        # Sequence flags
        self.avoid_tagging_helices = kwargs.get(flags.avoid_tagging_helices._)
        self.csv = kwargs.get(flags.csv._)
        self.nucleotide = kwargs.get(flags.nucleotide)
        self.optimize_species = kwargs.get(flags.optimize_species._)
        self.preferred_tag = kwargs.get(flags.preferred_tag._)
        self.tag_linker = kwargs.get(flags.tag_linker._)
        self.multicistronic = kwargs.get(flags.multicistronic._)
        self.multicistronic_intergenic_sequence = kwargs.get(flags.multicistronic_intergenic_sequence._)

        # Output flags
        self.overwrite: bool = kwargs.get(flags.overwrite._)
        self.pose_format = kwargs.get(flags.pose_format._)
        prefix = kwargs.get(flags.prefix._)
        if prefix:
            self.prefix = f'{prefix}_'
        else:
            self.prefix = ''

        suffix = kwargs.get(flags.suffix._)
        if suffix:
            self.suffix = f'_{suffix}'
        else:
            self.suffix = ''

        # Check if output already exists or --overwrite is provided
        if self.module in flags.select_modules:
            if self.prefix == '':
                # self.location must not be None
                self.prefix = f'{utils.starttime}_{os.path.basename(os.path.splitext(self.input_source)[0])}_'
            output_directory = kwargs.get(flags.output_directory._)
            # if not self.output_to_directory:
            if not output_directory:
                output_directory = os.path.join(os.path.dirname(self.program_root), f'SelectedDesigns')
                #     os.path.join(os.path.dirname(self.program_root), f'{self.prefix}SelectedDesigns{self.suffix}')
        else:  # if output_directory:
            output_directory = kwargs.get(flags.output_directory._)

        if output_directory:
            self.output_directory = output_directory
            if os.path.exists(self.output_directory) and not self.overwrite:
                print(f"The specified output directory '{self.output_directory}' already exists. Proceeding may "
                      f'overwrite your old data. Either specify a new one or use the flags {flags.prefix.long} or '
                      f'{flags.suffix.long} to modify the name. To proceed, append {flags.overwrite.long} to your '
                      f'command')
                sys.exit(1)
            putils.make_path(self.output_directory)

        output_file = kwargs.get(flags.output_file._)
        if output_file:
            self.output_file = output_file
            if os.path.exists(self.output_file) and not self.overwrite:
                # if self.module in flags.analysis:  # Todo this was allowed, but it's outdated...
                print(f"The specified output file '{self.output_file}' already exists. Proceeding may "
                      f'overwrite your old data. Either specify a new one or use the flags {flags.prefix.long} or '
                      f'{flags.suffix.long} to modify the name. To proceed, append {flags.overwrite.long} to your '
                      f'command')
                sys.exit(1)

        # When we are performing expand-asu, make sure we set output_assembly to True
        if self.module == flags.expand_asu:
            self.output_assembly = True
        else:
            self.output_assembly: bool = kwargs.get(flags.output_assembly._)
        self.output_surrounding_uc: bool = kwargs.get(flags.output_surrounding_uc._)
        self.output_fragments: bool = kwargs.get(flags.output_fragments._)
        self.output_interface: bool = kwargs.get(flags.output_interface._)
        self.output_oligomers: bool = kwargs.get(flags.output_oligomers._)
        self.output_entities: bool = kwargs.get(flags.output_entities._)
        self.output_structures: bool = kwargs.get(flags.output_structures._)
        self.output_trajectory: bool = kwargs.get(flags.output_trajectory._)

        self.skip_logging: bool = kwargs.get(flags.skip_logging._)
        # self.merge: bool = kwargs.get(flags.merge._)
        # self.save: bool = kwargs.get(flags.save._)
        # self.figures: bool = kwargs.get(flags.figures._)

        if self.output_structures or self.output_assembly or self.output_surrounding_uc or self.output_fragments \
                or self.output_oligomers or self.output_entities or self.output_trajectory:
            self.output: bool = True
        else:
            self.output: bool = False

        # self.nanohedra_output: bool = kwargs.get(flags.nanohedra_output)
        # self.nanohedra_root: str | None = None
        # if self.nanohedra_output:
        #     self.construct_pose: bool = kwargs.get('construct_pose', True)
        # else:
        # self.construct_pose = True

        # Align helix flags
        self.aligned_start = kwargs.get(flags.aligned_start._)
        self.aligned_end = kwargs.get(flags.aligned_end._)
        self.aligned_chain = kwargs.get(flags.aligned_chain._)
        self.alignment_length = kwargs.get(flags.alignment_length._)
        self.bend = kwargs.get(flags.bend._)
        self.extension_length = kwargs.get(flags.extend._)
        self.target_start = kwargs.get(flags.target_start._)
        self.target_end = kwargs.get(flags.target_end._)
        self.target_chain = kwargs.get(flags.target_chain._)
        self.target_termini = kwargs.get(flags.target_termini._)
        self.trim_termini = kwargs.get(flags.trim_termini._)
        # Helix Bending flags
        self.direction = kwargs.get(flags.direction._)
        self.joint_residue = kwargs.get(flags.joint_residue._)
        self.joint_chain = kwargs.get(flags.joint_chain._)
        self.sample_number = kwargs.get(flags.sample_number._)
        # Prediction flags
        self.predict = Predict.from_flags(**kwargs)
        # self.num_predictions_per_model = kwargs.get('num_predictions_per_model')
        # if self.predict.num_predictions_per_model is None:
        #     if 'monomer' in self.predict.mode:
        #         self.num_predictions_per_model = 1
        #     else:  # 'multimer
        #         self.num_predictions_per_model = 5
        if self.predict.models_to_relax == 'none':
            self.predict.models_to_relax = None

        # Clustering flags
        # Todo
        #  This is pretty sloppy. Modify this DataClass mechanism...
        self.cluster_selection = kwargs.get(flags.cluster_selection._)
        # self.cluster_map = kwargs.get('cluster_map')
        # self.as_objects: bool = kwargs.get('as_objects')
        # self.mode: bool = kwargs.get('mode')
        if flags.cluster_poses in self.modules or flags.cluster_map._ in kwargs:
            self.cluster = Cluster.from_flags(**kwargs)
            # self.cluster.map: AnyStr
            # """The path to a file containing the currently loaded mapping from cluster representatives to members"""
        else:
            self.cluster = False

        # Finally perform checks on desired work to see if viable
        # if self.protocol_module:
        self.check_protocol_module_arguments()
        # Start with None and set this once a session is opened
        self.job_protocol = None
        # self.job_protocol = self.load_job_protocol()
        self.parsed_arguments = None

    # @staticmethod
    def get_parsed_arguments(self) -> list[str]:
        """Return the arguments submitted during application initialization

        Returns:
            Each of the submitted flags, removed of input arguments, and formatted such as were parsed at runtime,
                i.e. --file, --poses, or -d are removed, and the remainder are left in the same order so as coule be
                formatted by subprocess.list2cmdline()
        """
        if self.parsed_arguments:
            return self.parsed_arguments

        # Remove the program
        parsed_arguments = sys.argv[1:]
        logger.debug(f'Starting with arguments {parsed_arguments}')
        # Todo
        #  Should the module be removed?
        #  sys.argv.remove(self.module)
        # Remove the input
        possible_input_args = [arg for args in flags.input_mutual_arguments.keys() for arg in args] \
            + [arg for args in flags.pose_inputs.keys() for arg in args] \
            + [arg for args in flags.component_mutual1_arguments.keys() for arg in args] \
            + [arg for args in flags.component_mutual2_arguments.keys() for arg in args]
        for input_arg in possible_input_args:
            try:
                pop_index = parsed_arguments.index(input_arg)
            except ValueError:  # Not in list
                continue
            else:
                removed_flag = parsed_arguments.pop(pop_index)
                while parsed_arguments[pop_index][0] != flags.flag_delimiter:
                    removed_arg = parsed_arguments.pop(pop_index)
                    logger.debug(f'From {removed_flag}, removed argument {removed_arg}')
                # # If the flag requires an argument, pop the index a second time
                # if input_arg not in single_input_flags:

        # Remove distribution flags
        for arg in flags.distribute_args:
            try:
                pop_index = parsed_arguments.index(arg)
            except ValueError:  # Not in list
                continue
            else:
                parsed_arguments.pop(pop_index)

        # Set for the next time
        self.parsed_arguments = parsed_arguments

        return parsed_arguments

    @property
    def id(self) -> int:
        """Get the JobProtocol.id for reference to the work performed"""
        return self.job_protocol.id

    def load_job_protocol(self):
        """Acquire the JobProtocol for the current set of input instructions

        Sets:
            self.job_protocol (sql.JobProtocol)
        """
        # Tabulate the protocol arguments that should be provided to the JobProtocol search/creation
        if self.module == flags.design:
            protocol_kwargs = dict(
                ca_only=self.design.ca_only,
                evolution_constraint=self.design.evolution_constraint,
                interface=self.design.interface,
                term_constraint=self.design.term_constraint,
                neighbors=self.design.neighbors,
                proteinmpnn_model_name=self.design.proteinmpnn_model,
            )
        elif self.module == flags.nanohedra:
            protocol_kwargs = dict(
                ca_only=self.design.ca_only,
                contiguous_ghosts=self.dock.contiguous_ghosts,
                initial_z_value=self.dock.initial_z_value,
                match_value=self.dock.match_value,
                minimum_matched=self.dock.minimum_matched,
                proteinmpnn_model_name=self.design.proteinmpnn_model,
            )
        elif self.module == flags.predict_structure:
            protocol_kwargs = dict(
                number_predictions=self.predict.num_predictions_per_model,
                prediction_model=self.predict.models_to_relax,
                use_gpu_relax=self.predict.use_gpu_relax,
            )
        elif self.module == flags.analysis:
            protocol_kwargs = dict(
                ca_only=self.design.ca_only,
                proteinmpnn_model_name=self.design.proteinmpnn_model,
            )
        # Todo
        #  raise NotImplementedError()
        # elif self.module == flags.interface_metrics:
        # elif self.module == flags.generate_fragments:
        else:
            protocol_kwargs = {}

        protocol_kwargs.update(dict(
            module=self.module,
            commit=putils.commit,
        ))

        job_protocol_stmt = select(sql.JobProtocol)\
            .where(*[getattr(sql.JobProtocol, table_column) == job_resources_attr
                     for table_column, job_resources_attr in protocol_kwargs.items()])
        # logger.debug(job_protocol_stmt.compile(compile_kwargs={"literal_binds": True}))
        with self.db.session(expire_on_commit=False) as session:
            job_protocol_result = session.scalars(job_protocol_stmt).all()
            if not job_protocol_result:  # Create a new one
                job_protocol = sql.JobProtocol(**protocol_kwargs)
                session.add(job_protocol)
                session.commit()
            elif len(job_protocol_result) > 1:
                for result in job_protocol_result:
                    print(result)
                raise utils.InputError(
                    f"sqlalchemy.IntegrityError should've been raised. "
                    f"Can't have more than one matching {sql.JobProtocol.__name__}")
            else:
                job_protocol = job_protocol_result[0]

        self.job_protocol = job_protocol

    @property
    def modules(self) -> list[str]:
        """Return the modules slated to run during the job"""
        try:
            return self._modules
        except AttributeError:
            return [self.module]

    @modules.setter
    def modules(self, modules: Iterable[str]):
        self._modules = list(modules)

    # @property
    # def current_session(self) -> Session:
    #     """Contains the sqlalchemy.orm.Session that is currently in use for access to database attributes"""
    #     try:
    #         return self._session
    #     except AttributeError:  # No session connected
    #         raise AttributeError("Couldn't return current_session as there is not an active Session. Ensure you "
    #                              "initialize a job context manager, i.e.\n"
    #                              "with job.db.session() as session:\n"
    #                              "    job.current_session = session\n"
    #                              "Before you attempt to use the current_session")
    #
    # @current_session.setter
    # def current_session(self, session: Session):
    #     """Set the sqlalchemy.orm.Session that is currently in use to access database attributes later during the job"""
    #     if isinstance(session, Session):
    #         self._session = session

    @property
    def output_to_directory(self) -> bool:
        """If True, broadcasts that output is not typical putils.program_output directory structure"""
        # self.output_to_directory: bool = True if self.output_directory else False
        if self.module in flags.select_modules:
            # Make this explicitly False so that selection doesn't copy extra files
            return False
        return True if self.output_directory else False

    @property
    def output_directory(self) -> AnyStr | None:
        """Where to output the Job"""
        try:
            return self._output_directory
        except AttributeError:
            self._output_directory = None
            return self._output_directory

    @output_directory.setter
    def output_directory(self, output_directory: str):
        """Where to output the Job"""
        # Format the output_directory so that the basename, i.e. the part that is desired
        # can be formatted with prefix/suffix
        dirname, output_directory = os.path.split(output_directory.rstrip(os.sep))
        self._output_directory = os.path.join(dirname, f'{self.prefix}{output_directory}{self.suffix}')

    @property
    def output_file(self) -> AnyStr | None:
        """Where to output info related to successful Job operation"""
        try:
            return self._output_file
        except AttributeError:
            self._output_file = None
            return self._output_file

    @output_file.setter
    def output_file(self, output_file: str):
        """Where to output info related to successful Job operation"""
        # Format the output_file so that the basename, i.e. the part that is desired
        # can be formatted with prefix/suffix
        dirname, basename = os.path.split(output_file.rstrip(os.sep))
        self._output_file = os.path.join(dirname, f'{self.prefix}{basename}{self.suffix}')

    @property
    def location(self) -> str | None:
        """The location where PoseJob instances are located"""
        try:
            return self._location
        except AttributeError:
            self._location = self._input_source = None
            return self._location

    @location.setter
    def location(self, location: str | list[str]):
        if location is None:
            self._input_source = self.program_root
        elif isinstance(location, str):
            self._location = location
            self._input_source, extension = os.path.splitext(os.path.basename(location))
        elif isinstance(location, list):
            self._location = ', '.join(location)
            self._input_source = '+'.join(map(os.path.basename, [os.path.splitext(_location)[0]
                                                                 for _location in location]))
            if len(self._input_source) > 100:
                # Reduce the size
                self._input_source = self._input_source[:100]
        else:
            raise ValueError(
                f"Couldn't handle the provided location type {type(location).__name__}")

    def get_range_slice(self, jobs: Sequence) -> Sequence[Any]:
        """Slice the input work by a set increment. This is parsed from the flags.range_args

        Args:
            jobs: The work that should be sliced by the specified range
        Returns:
            The work, limited to the range provided by -r/--range input flag
        """
        if self.range:
            path_number = len(jobs)
            # Adding 0.5 to ensure rounding occurs
            low_range = int((self.low/100) * path_number + 0.5)
            high_range = int((self.high/100) * path_number + 0.5)
            if low_range < 0 or high_range > path_number:
                raise ValueError(
                    f'The {flags.format_args(flags.range_args)} flag is outside of the acceptable bounds [0-100]')
            logger.debug(f'Selecting input work ({path_number}) with range: {low_range}-{high_range}')
            range_slice = slice(low_range, high_range)
        else:
            range_slice = slice(None)

        return jobs[range_slice]

    @property
    def input_source(self) -> str:
        """Provide the name of the specified PoseJob instances to perform work on"""
        try:
            return self._input_source
        except AttributeError:
            self._input_source = self.program_root
            return self._input_source

    @property
    def default_output_tuple(self) -> tuple[str, str, str]:
        """Format fields for the output file depending on time, specified name and module type"""
        if self.low and self.high:
            design_source = f'{self.input_source}-{self.low:.2f}-{self.high:.2f}'
        else:
            design_source = self.input_source

        return utils.starttime, self.module, design_source

    @property
    def construct_pose(self):
        """Whether to construct the PoseJob"""
        return True  # self._construct_pose

    @construct_pose.setter
    def construct_pose(self, value: bool):
        self._construct_pose = value
        if self._construct_pose:
            pass
        else:  # No construction specific flags
            self.output_fragments = self.output_oligomers = self.output_entities = False

    # Todo make part of modules.setter routine
    def check_protocol_module_arguments(self):
        """Given provided modules for the 'protocol' module, check to ensure the work is adequate

        Raises:
            InputError if the inputs are found to be incompatible
        """
        protocol_module_allowed_modules = [
            flags.align_helices,
            flags.bend,
            flags.expand_asu,
            flags.rename_chains,
            flags.check_clashes,
            flags.generate_fragments,
            flags.nanohedra,
            flags.predict_structure,
            flags.interface_metrics,
            flags.optimize_designs,
            flags.refine,
            flags.design,
            flags.interface_design,
            flags.analysis,
            flags.cluster_poses,
            flags.select_poses,
            flags.select_designs
        ]
        disallowed_modules = [
            # 'custom_script',
            # flags.select_sequences,
            flags.initialize_building_blocks
        ]

        def check_gpu() -> str | bool:
            available_devices = jax.local_devices()
            for idx, device in enumerate(available_devices):
                if device.platform == 'gpu':
                    self.gpu_available = True
                    return device.device_kind
                    # device_id = idx
                    # return True
            return False

        problematic_modules = []
        not_recognized_modules = []
        nanohedra_prior = False
        gpu_device_kind = None
        for idx, module in enumerate(self.modules, 1):
            if module == flags.nanohedra:
                if idx > 1:
                    raise utils.InputError(
                        f"For {flags.protocol} module, {module} can only be run in --modules position #1")
                nanohedra_prior = True
                continue
            elif module in flags.select_modules and self.protocol_module:
                if idx != self.number_of_modules:
                    raise utils.InputError(
                        f"For {flags.protocol} module, {module} can only be run in --modules position N i.e. #1,2,...N")

            elif module == flags.predict_structure:
                if gpu_device_kind is None:
                    # Check for GPU access
                    gpu_device_kind = check_gpu()

                if gpu_device_kind:
                    logger.info(f'Running {module} on {gpu_device_kind} GPU')
                    # Disable GPU on tensorflow. I think that this is so tensorflow doesn't leak any calculations
                    tf.config.set_visible_devices([], 'GPU')
                else:  # device.platform == 'cpu':
                    logger.warning(f'No GPU detected, will {module} using CPU')
            elif module == flags.design:
                if self.design.method == flags.proteinmpnn:
                    if gpu_device_kind is None:
                        # Check for GPU access
                        gpu_device_kind = check_gpu()

                    if gpu_device_kind:
                        logger.info(f'Running {module} on {gpu_device_kind} GPU')
                    else:  # device.platform == 'cpu':
                        logger.warning(f'No GPU detected, will {module} using CPU')

            if nanohedra_prior:
                if module in flags.select_modules:
                    # We only should allow select-poses after nanohedra
                    if module == flags.select_poses:
                        logger.critical(f"Running {module} after {flags.nanohedra} won't produce any Designs to "
                                        f"operate on. In order to {module}, ensure you run a design protocol first")
                    else:  # flags.select_designs, flags.select_sequences
                        if not self.weight:  # not self.filter or
                            logger.critical(f'Using {module} after {flags.nanohedra} without specifying the flag '
                                            # f'{flags.format_args(flags.filter_args)} or '
                                            f'{flags.format_args(flags.weight_args)} defaults to selection '
                                            f'parameters {config.default_weight_parameter[flags.nanohedra]}')
            nanohedra_prior = False
            if self.protocol_module:
                if module in protocol_module_allowed_modules:
                    continue
                elif module in disallowed_modules:
                    problematic_modules.append(module)
                else:
                    not_recognized_modules.append(module)

        if not_recognized_modules:
            raise utils.InputError(
                f"For {flags.protocol} module, the --{flags.modules} {', '.join(not_recognized_modules)} aren't "
                f'recognized modules. See"{putils.program_help}" for available module names')

        if problematic_modules:
            raise utils.InputError(
                f"For {flags.protocol} module, the --{flags.modules} {', '.join(problematic_modules)} aren't possible "
                f'modules\n\nAllowed modules are {", ".join(protocol_module_allowed_modules)}')

    def report_specified_arguments(self, arguments: argparse.Namespace) -> dict[str, Any]:
        """Filter all flags for only those that were specified as different on the command line

        Args:
            arguments: The arguments as parsed from the command-line argparse namespace
        Returns:
            Arguments specified during program execution
        """
        arguments = vars(arguments).copy()

        reported_args = {}
        # Start with JobResources flags that should be reported, or if the argument is not important, format it
        if self.module:
            reported_args['module'] = self.module
        if self.sym_entry:
            reported_args[flags.sym_entry._] = self.sym_entry.number
        # if self.design_selector:
        #     reported_args.pop('design_selector', None)

        # # Custom removal/formatting for all remaining
        # for custom_arg in list(arguments.keys()):
        #     value = arguments.pop(custom_arg, None)
        #     if value is not None:
        #         reported_args[custom_arg] = value

        if self.debug:
            def report_arg(_dest, _default):
                try:
                    value = arguments.pop(_dest)
                    if value is not None:
                        reported_args[arg.dest] = value
                except KeyError:
                    return
        else:
            def report_arg(_dest, _default):
                try:
                    value = arguments.pop(_dest)
                    if value is not None and value != _default:
                        reported_args[arg.dest] = value
                except KeyError:
                    return

        # Get all the default program args and compare them to the provided values
        for group in flags.entire_parser._action_groups:
            for arg in group._group_actions:
                if isinstance(arg, argparse._SubParsersAction):  # This is a subparser, recurse
                    for name, sub_parser in arg.choices.items():
                        for sub_group in sub_parser._action_groups:
                            for arg in sub_group._group_actions:
                                report_arg(arg.dest, arg.default)
                else:
                    report_arg(arg.dest, arg.default)

        return dict(sorted(reported_args.items()))  # , key=lambda arg: arg[0]

    def calculate_memory_requirements(self, number_jobs: int):
        """Format memory requirements with module dependencies and set self.reduce_memory"""
        if self.module == flags.nanohedra:  # Todo
            required_memory = putils.baseline_program_memory + putils.nanohedra_memory  # 30 GB ?
        elif self.module == flags.analysis:
            required_memory = (putils.baseline_program_memory +
                               number_jobs * putils.approx_ave_design_directory_memory_w_assembly) * 1.2
        else:
            required_memory = (putils.baseline_program_memory +
                               number_jobs * putils.approx_ave_design_directory_memory_w_pose) * 1.2

        available_memory = psutil.virtual_memory().available
        logger.debug(f'Available memory: {available_memory / gb_divisior:.2f} GB')
        logger.debug(f'Required memory: {required_memory / gb_divisior:.2f} GB')
        # If we are running a protocol, check for reducing memory requirements
        if self.protocol_module and self.number_of_modules > 2:
            self.reduce_memory = True
        elif available_memory < required_memory:
            self.reduce_memory = True
        else:
            # Todo when requirements are more accurate with database
            #  self.reduce_memory = False
            self.reduce_memory = True
        logger.debug(f'Reduce job memory?: {self.reduce_memory}')

    @property
    def number_of_modules(self) -> int:
        """The number of modules for the specified Job"""
        return len(self.modules)

    @staticmethod
    def can_process_evolutionary_profiles() -> bool:
        """Return True if the current computer has the computational requirements to collect evolutionary profiles"""
        # Run specific checks
        if psutil.virtual_memory().available <= distribute.hhblits_memory_threshold:
            print('\n')
            logger.critical(f'The available RAM is probably insufficient to run {putils.hhblits}. '
                            f'Required/Available memory: {distribute.hhblits_memory_threshold / gb_divisior:.2f} GB/'
                            f'{psutil.virtual_memory().available / gb_divisior:.2f} GB')
            logger.critical(f'Creating scripts that can be distributed to a capable computer instead')
            return False
        return True

    @staticmethod
    def evolutionary_profile_processes() -> int:
        """Return the number of evolutionary profile processes that can be run given the available memory"""
        return int(psutil.virtual_memory().available <= distribute.hhblits_memory_threshold)

    def process_evolutionary_info(self, uniprot_entities: Iterable[wrapapi.UniProtEntity] = None,
                                  entities: Iterable[structure.sequence.GeneEntity] = None,
                                  batch_commands: bool = False) -> list[str]:
        """Format the job with evolutionary constraint options

        Args:
            uniprot_entities: A list of the UniProtIDs for the Job
            entities: A list of the Entity instances initialized for the Job
            batch_commands: Whether commands should be made for batch submission
        Returns:
            A list evolutionary setup instructions
        """
        info_messages = []
        hhblits_cmds, bmdca_cmds, msa_cmds = [], [], []
        # Set up sequence data using hhblits and profile bmDCA for each input entity
        # all_entity_ids = []
        putils.make_path(self.sequences)
        if not putils.uniclust_db:
            # No database has been set up
            logger.warning(f"Couldn't locate a compatible database to run hhblits. Skipping evolutionary acquisition")
        elif uniprot_entities is not None:
            for uniprot_entity in uniprot_entities:
                evolutionary_profile_file = self.api_db.hhblits_profiles.retrieve_file(name=uniprot_entity.id)
                if not evolutionary_profile_file:
                    hhblits_cmds.append(hhblits(uniprot_entity.id,
                                                sequence=uniprot_entity.reference_sequence,
                                                out_dir=self.profiles, threads=self.threads,
                                                return_command=True))
                    msa_file = None
                else:
                    msa_file = self.api_db.alignments.retrieve_file(name=uniprot_entity.id)

                if not msa_file:
                    if not os.access(putils.reformat_msa_exe_path, os.X_OK):
                        logger.error(f"Couldn't execute multiple sequence alignment reformatting script")
                        # raise RuntimeError(
                        #     f"Couldn't locate {putils.reformat_msa_exe_path}. Ensure the file exists then try your job "
                        #     f"again. Otherwise, use the argument --no-{flags.use_evolution} OR set up hhblits to run."
                        #     f"{utils.guide.hhblits_setup_instructions}")
                    else:
                        sto_cmd = [
                            putils.reformat_msa_exe_path, 'a3m', 'sto',
                            f"{os.path.join(self.profiles, f'{uniprot_entity.id}.a3m')}", '.sto', '-num', '-uc']
                        fasta_cmd = [
                            putils.reformat_msa_exe_path, 'a3m', 'fas',
                            f"{os.path.join(self.profiles, f'{uniprot_entity.id}.a3m')}", '.fasta', '-M', 'first', '-r']
                        msa_cmds.extend([sto_cmd, fasta_cmd])

        elif entities is not None:
            raise NotImplementedError(
                f'Currently must use {wrapapi.UniProtEntity.__class__.__name__} in '
                f'{self.process_evolutionary_info.__name__}'
            )
            for entity in entities:
                evolutionary_profile_file = self.api_db.hhblits_profiles.retrieve_file(name=entity.name)
                if not evolutionary_profile_file:
                    sequence_file = self.api_db.sequences.retrieve_file(name=entity.name)
                    if not sequence_file:
                        sequence_file = entity.write_sequence_to_fasta(out_dir=self.sequences)

                    hhblits_cmds.append(entity.hhblits(sequence_file=sequence_file, out_dir=self.profiles,
                                                       return_command=True))
                    msa_file = None
                else:
                    msa_file = self.api_db.alignments.retrieve_file(name=entity.name)

                if not msa_file:
                    if not os.access(putils.reformat_msa_exe_path, os.X_OK):
                        logger.error(f"Couldn't execute multiple sequence alignment reformatting script")
                        # raise RuntimeError(
                        #     f"Couldn't locate {putils.reformat_msa_exe_path}. Ensure the file exists then try your job "
                        #     f"again. Otherwise, use the argument --no-{flags.use_evolution} OR set up hhblits to run."
                        #     f"{utils.guide.hhblits_setup_instructions}")
                    else:
                        sto_cmd = [
                            putils.reformat_msa_exe_path, 'a3m', 'sto',
                            f"{os.path.join(self.profiles, f'{entity.name}.a3m')}", '.sto', '-num', '-uc']
                        fasta_cmd = [
                            putils.reformat_msa_exe_path, 'a3m', 'fas',
                            f"{os.path.join(self.profiles, f'{entity.name}.a3m')}", '.fasta', '-M', 'first', '-r']
                        msa_cmds.extend([sto_cmd, fasta_cmd])

        if hhblits_cmds:
            protocol = putils.hhblits
            logger.info(f"Starting Profile(dtype='{protocol}') generation")

            if protocol == putils.hhblits:
                if putils.hhblits_exe is None or not os.access(putils.hhblits_exe, os.X_OK):
                    raise RuntimeError(
                        f"Couldn't locate the {protocol} executable. Ensure the executable file referenced by "
                        f"'{putils.hhblits_exe}' exists then try your job again. Otherwise, use the argument "
                        f'--no-{flags.use_evolution} OR set up hhblits to run.{utils.guide.hhblits_setup_instructions}')
            else:
                assert_never(protocol)

            putils.make_path(self.profiles)
            putils.make_path(self.sbatch_scripts)
            protocol_log_file = os.path.join(self.profiles, 'generate_profiles.log')

            # Run hhblits commands
            if not batch_commands and self.can_process_evolutionary_profiles():
                logger.info(f'Writing {protocol} results to file: {protocol_log_file}')
                # Run commands in this process
                if self.multi_processing:
                    zipped_args = zip(hhblits_cmds, repeat(hhblits_log_file))
                    utils.mp_starmap(distribute.run, zipped_args, processes=self.cores)
                else:
                    for cmd in tqdm(hhblits_cmds):
                        logger.debug(f'Starting command: {subprocess.list2cmdline(cmd)}')
                        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        stdout, stderr = p.communicate()
                        if stderr:
                            logger.warning(stderr.decode('utf-8'))

                # Format .a3m multiple sequence alignments to .sto/.fasta
                for cmd in msa_cmds:
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = p.communicate()
                    if stderr:
                        logger.warning(stderr.decode('utf-8'))
            else:  # Convert each command to a string and write to distribute
                hhblits_cmds = [subprocess.list2cmdline(cmd) for cmd in hhblits_cmds]
                msa_cmds = [subprocess.list2cmdline(cmd) for cmd in msa_cmds]
                all_evolutionary_commands = hhblits_cmds + msa_cmds
                evolutionary_cmds_file = distribute.write_commands(
                    all_evolutionary_commands, name=f'{utils.starttime}-{protocol}', out_path=self.profiles)

                if distribute.is_sbatch_available():
                    shell = distribute.sbatch
                    max_jobs = len(hhblits_cmds)  # number_of_hhblits_cmds
                else:
                    shell = distribute.default_shell
                    max_jobs = self.evolutionary_profile_processes()

                # Todo
                #  distribution.Attrs()
                protocol_kwargs = dict(out_path=self.sbatch_scripts, scale=protocol,
                                       max_jobs=max_jobs, number_of_commands=len(all_evolutionary_commands),
                                       log_file=protocol_log_file)
                # reformat_msa_cmds_script = distribute.distribute(file=reformat_msa_cmd_file, **hhblits_kwargs)
                protocol_script = distribute.distribute(file=evolutionary_cmds_file, **protocol_kwargs)
                # Format messages
                info_messages.append(
                    'Please follow the instructions below to generate sequence profiles for input proteins')
                protocol_job_info_message = f'Enter the following to distribute {protocol} jobs:\n\t'
                protocol_job_info_message += f'{shell} {protocol_script}'
                info_messages.append(protocol_job_info_message)
        elif msa_cmds:  # These may still be missing
            putils.make_path(self.profiles)

            if not os.access(putils.reformat_msa_exe_path, os.X_OK):
                logger.error(f"Couldn't execute multiple sequence alignment reformatting script")

            # Format .a3m multiple sequence alignments to .sto/.fasta
            for cmd in msa_cmds:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = p.communicate()
                if stderr:
                    logger.warning(stderr.decode('utf-8'))

        if bmdca_cmds:
            putils.make_path(self.profiles)
            putils.make_path(self.sbatch_scripts)
            # bmdca_cmds = \
            #     [list2cmdline([putils.bmdca_exe_path, '-i', os.path.join(self.profiles, '%s.fasta' % entity.name),
            #                   '-d', os.path.join(self.profiles, '%s_bmDCA' % entity.name)])
            #      for entity in entities.values()]
            bmdca_cmd_file = \
                distribute.write_commands(bmdca_cmds, name=f'{utils.starttime}-bmDCA', out_path=self.profiles)
            bmdca_script = distribute.distribute(file=bmdca_cmd_file, out_path=self.sbatch_scripts,
                                                 scale='bmdca', max_jobs=len(bmdca_cmds),
                                                 number_of_commands=len(bmdca_cmds),
                                                 log_file=os.path.join(self.profiles, 'generate_couplings.log'))
            # reformat_msa_cmd_file = \
            #     SDUtils.write_commands(reformat_msa_cmds, name='%s-reformat_msa' % SDUtils.starttime,
            #                            out_path=self.profiles)
            # reformat_sbatch = distribute(file=reformat_msa_cmd_file, out_path=self.program_root,
            #                              scale='script', max_jobs=len(reformat_msa_cmds),
            #                              log_file=os.path.join(self.profiles, 'generate_profiles.log'),
            #                              number_of_commands=len(reformat_msa_cmds))
            if distribute.is_sbatch_available():
                shell = distribute.sbatch
            else:
                shell = distribute.default_shell

            bmdca_script_message = \
                f'Once you are satisfied, enter the following to distribute jobs:\n\t{shell} %s' \
                % bmdca_script if not info_messages else 'ONCE this job is finished, to calculate evolutionary ' \
                                                         'couplings i,j for each amino acid in the multiple ' \
                                                         f'sequence alignment, enter:\n\t{shell} {bmdca_script}'
            info_messages.append(bmdca_script_message)

        return info_messages


class JobResourcesFactory:
    """Return a JobResource instance by calling the Factory instance

    Handles creation and allotment to other processes by making a shared pointer to the JobResource for the current Job
    """
    def __init__(self, **kwargs):
        self._resources = {}
        self._warn = True

    def destruct(self, **kwargs):
        self._resources = {}
        self._warn = True

    def __call__(self, **kwargs) -> JobResources:
        """Return the specified JobResources object singleton

        Returns:
            The instance of the specified JobResources
        """
        #         Args:
        #             source: The JobResources source name
        source = 'single'
        job = self._resources.get(source)
        if job:
            if kwargs and self._warn:
                # try:
                #     fragment_db.update(kwargs)
                # except RuntimeError:
                self._warn = False
                logger.warning(f"Can't pass the new arguments {', '.join(kwargs.keys())} to JobResources "
                               f'since it was already initialized and is a singleton')
            return job
        else:
            logger.info(f'Initializing {JobResources.__name__}({kwargs.get("program_root", os.getcwd())})')
            self._resources[source] = JobResources(**kwargs)

        return self._resources[source]

    def get(self, **kwargs) -> JobResources:
        """Return the specified JobResources object singleton

        Returns:
            The instance of the specified JobResources
        """
        # Keyword Args:
        #     source: The JobResource source name
        return self.__call__(**kwargs)


job_resources_factory: Annotated[JobResourcesFactory,
                                 'Calling this factory method returns the single instance of the JobResources class'] \
    = JobResourcesFactory()
"""Calling this factory method returns the single instance of the JobResources class"""
