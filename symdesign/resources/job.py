from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import subprocess
from copy import deepcopy
from itertools import repeat
from subprocess import list2cmdline
from typing import Annotated, AnyStr, Any, Iterable

import jax
import psutil
import tensorflow as tf
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from . import config, sql, structure_db, wrapapi
from symdesign import flags, sequence, structure, utils
from symdesign.sequence import hhblits
from symdesign.structure.fragment import db
from symdesign.utils import distribute, guide, SymEntry, InputError, path as putils

logger = logging.getLogger(__name__)
gb_divisior = 1e9  # 1000000000


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
        raise ValueError('The sequence and design_selector are different lengths. Please correct the alignment and '
                         'lengths before proceeding.')

    return [idx for idx, aa in enumerate(mask, 1) if aa != '-']


def generate_chain_mask(chains: str) -> set[str]:
    """From a string with a design_selection, format the chains provided

    Args:
        chains: The specified chains separated by commas to split
    Returns:
        The provided chain ids in pose format
    """
    return set(utils.clean_comma_separated_string(chains))


def process_design_selector_flags(
        select_designable_residues_by_sequence: str = None,
        select_designable_residues_by_pdb_number: str = None,
        select_designable_residues_by_pose_number: str = None,
        select_designable_chains: str = None,
        mask_designable_residues_by_sequence: str = None,
        mask_designable_residues_by_pdb_number: str = None,
        mask_designable_residues_by_pose_number: str = None,
        mask_designable_chains: str = None,
        require_design_by_pdb_number: str = None,
        require_design_by_pose_number: str = None,
        require_design_by_chain: str = None,
        **kwargs: dict[str]) -> dict[str, dict[str, set | set[int] | set[str]]]:
    # Todo move to a verify design_selectors function inside of Pose? Own flags module?
    #  Pull mask_design_using_sequence out of flags
    # -------------------
    entity_select, chain_select, residue_select, residue_pdb_select = set(), set(), set(), set()
    if select_designable_residues_by_sequence is not None:
        residue_select = residue_select.union(generate_sequence_mask(select_designable_residues_by_sequence))

    if select_designable_residues_by_pdb_number is not None:
        residue_pdb_select = \
            residue_pdb_select.union(utils.format_index_string(select_designable_residues_by_pdb_number))

    if select_designable_residues_by_pose_number is not None:
        residue_select = residue_select.union(utils.format_index_string(select_designable_residues_by_pose_number))

    if select_designable_chains is not None:
        chain_select = chain_select.union(generate_chain_mask(select_designable_chains))
    # -------------------
    entity_mask, chain_mask, residue_mask, residue_pdb_mask = set(), set(), set(), set()
    if mask_designable_residues_by_sequence is not None:
        residue_mask = residue_mask.union(generate_sequence_mask(mask_designable_residues_by_sequence))

    if mask_designable_residues_by_pdb_number is not None:
        residue_pdb_mask = residue_pdb_mask.union(utils.format_index_string(mask_designable_residues_by_pdb_number))

    if mask_designable_residues_by_pose_number is not None:
        residue_mask = residue_mask.union(utils.format_index_string(mask_designable_residues_by_pose_number))

    if mask_designable_chains is not None:
        chain_mask = chain_mask.union(generate_chain_mask(mask_designable_chains))
    # -------------------
    entity_req, chain_req, residues_req, residues_pdb_req = set(), set(), set(), set()
    if require_design_by_pdb_number is not None:
        residues_pdb_req = residues_pdb_req.union(utils.format_index_string(require_design_by_pdb_number))

    if require_design_by_pose_number is not None:
        residues_req = residues_req.union(utils.format_index_string(require_design_by_pose_number))

    if require_design_by_chain is not None:
        chain_req = chain_req.union(generate_chain_mask(require_design_by_chain))

    return dict(selection=dict(entities=entity_select, chains=chain_select, residues=residue_select,
                               pdb_residues=residue_pdb_select),
                mask=dict(entities=entity_mask, chains=chain_mask, residues=residue_mask,
                          pdb_residues=residue_pdb_mask),
                required=dict(entities=entity_req, chains=chain_req, residues=residues_req,
                              pdb_residues=residues_pdb_req))


def format_args_for_namespace(args_: dict[str, Any], namespace: str, flags_: Iterable[str]) -> dict[str, Any]:
    namespace_args = {}
    for flag in flags_:
        try:
            arg_ = args_[flag]  # .pop(flag)  # , None)  # get(flag)
        except KeyError:  # No flag is here
            continue
        else:# logger.debug('flag', flag, 'arg', arg_)
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
# The commented out versions below had poor implementations
# DesignFlags = collections.namedtuple('DesignFlags', design_args.keys(), defaults=design_args.values())
# DesignFlags = types.SimpleNamespace(**design_args)
# Due to an error evaluating the singleton eval(type(None).__name__), need to pass a globals argument
nonetype_map = {'NoneType': None}
Design = dataclasses.make_dataclass(
    'Design',
    [(flag, eval(type(default).__name__, nonetype_map), dataclasses.field(default=default))
     for flag, default in flags.design_defaults.items()],
    bases=(FlagsBase,))
#     namespace={'from_flags': classmethod(from_flags)})
#     frozen=True)
#  self.design = DesignFlags(*[kwargs.get(argument_name) for argument_name in design_args.keys()])
#  self.design = types.SimpleNamespace(**{flag: kwargs.get(flag, default) for flag, default in flags.design})
#  self.design = types.SimpleNamespace(**{flag: kwargs.get(flag, default) for flag, default in flags.design})
Dock = dataclasses.make_dataclass(
    'Dock',
    [(flag, eval(type(default).__name__, nonetype_map), dataclasses.field(default=default))
     for flag, default in flags.dock_defaults.items()],
    bases=(FlagsBase,))
#     namespace={'from_flags': classmethod(from_flags)})
#     frozen=True)

Predict = dataclasses.make_dataclass(
    'Predict',
    [(flag, eval(type(default).__name__, nonetype_map), dataclasses.field(default=default))
     for flag, default in flags.predict_defaults.items()],
    bases=(FlagsBase,))
#     namespace={'from_flags': classmethod(from_flags)})
#     frozen=True)

Cluster = dataclasses.make_dataclass(
    'Cluster',
    [(flag, eval(type(default).__name__, nonetype_map), dataclasses.field(default=default))
     for flag, default in flags.cluster_defaults.items()],
    bases=(FlagsBase,))
#     namespace={'from_flags': classmethod(from_flags)})
#     frozen=True)


class DBInfo:
    def __init__(self, location: AnyStr, echo: bool = False):
        self.location = location
        self.engine: Engine = create_engine(f'sqlite:///{self.location}', echo=echo, future=True)
        self.session: sessionmaker = sessionmaker(self.engine, future=True)

        # The below functions are recommended to help overcome issues with SQLite transaction scope
        # See: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#pysqlite-serializable
        @event.listens_for(self.engine, "connect")
        def do_connect(dbapi_connection, connection_record):
            # Disable pysqlite's emitting of the BEGIN statement entirely
            # Also stops it from emitting COMMIT before any DDL
            dbapi_connection.isolation_level = None

        @event.listens_for(self.engine, "begin")
        def do_begin(conn):
            # Emit our own BEGIN
            conn.exec_driver_sql("BEGIN")


class JobResources:
    """The intention of JobResources is to serve as a singular source of design info which is common across all
    jobs. This includes common paths, databases, and design flags which should only be set once in program operation,
    then shared across all member designs"""
    _input_source: str | list[str] | None
    _location: str | None
    _modules: list[str]
    _output_directory: AnyStr | None
    _session: Session | None
    db: DBInfo | None
    reduce_memory: bool = False

    def __init__(self, program_root: AnyStr = None, arguments: argparse.Namespace = None, **kwargs):
        """Parse the program operation location, ensure paths to these resources are available, and parse arguments

        Args:
            program_root:
            arguments:
        """
        try:
            if os.path.exists(program_root):
                self.program_root = program_root
            else:
                raise FileNotFoundError(f"Path doesn't exist\n\t{program_root}")
        except TypeError:
            raise TypeError(f"Can't initialize {JobResources.__name__} without parameter 'program_root'")

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
        """Whether to reissue commands, only if distribute_work=False"""
        self.log_level: bool = kwargs.get('log_level')
        self.debug: bool = True if self.log_level == 1 else False
        self.force: bool = kwargs.get(putils.force)
        self.development: bool = kwargs.get(putils.development)
        self.profile_memory: bool = kwargs.get(putils.profile)
        if self.profile_memory and not self.development:
            logger.warning(f"--{flags.profile_memory} was set but --development wasn't")

        self.mpi: int = kwargs.get('mpi')
        if self.mpi is None:
            self.mpi = 0
            self.distribute_work: bool = kwargs.get(putils.distribute_work)
            # # Todo implement, see symdesign.utils and CommandDistributor
            # # extras = ' mpi {CommmandDistributer.mpi}'
            # number_mpi_processes = CommmandDistributer.mpi - 1
            # logger.info('Setting job up for submission to MPI capable computer. Pose trajectories run in parallel, '
            #             f'{number_mpi_processes} at a time. This will speed up processing ~
            #             f'{job.design.number / number_mpi_processes:2f}-fold.')
        else:  # self.mpi > 0
            self.distribute_work = True
            raise NotImplementedError(f"Can't compute the number of resources to allocate using --mpi just yet")

        self.multi_processing: int = kwargs.get(putils.multi_processing)
        if self.multi_processing:
            # Calculate the number of cores to use depending on computer resources
            self.cores = utils.calculate_mp_cores(cores=kwargs.get('cores'))  # Todo mpi=self.mpi
        else:
            self.cores: int = 1
        self.threads = self.cores * 2
        # self.reduce_memory = False

        # Input parameters
        self.project_name = kwargs.get('project_name')
        # program_root subdirectories
        self.data = os.path.join(self.program_root, putils.data.title())
        # if self.output_to_directory:
        #     self.projects = ''
        self.projects = os.path.join(self.program_root, putils.projects)
        self.job_paths = os.path.join(self.program_root, putils.job_paths)
        self.sbatch_scripts = os.path.join(self.program_root, putils.scripts.title())
        self.all_scores = os.path.join(self.program_root, putils.all_scores)

        # data subdirectories
        self.clustered_poses = os.path.join(self.data, 'ClusteredPoses')
        self.structure_info = os.path.join(self.data, putils.structure_info)
        self.pdbs = os.path.join(self.structure_info, 'PDBs')  # Used to store downloaded PDB's
        self.sequence_info = os.path.join(self.data, putils.sequence_info)
        self.external_db = os.path.join(self.data, 'ExternalDatabases')
        self.internal_db = os.path.join(self.data, f'{putils.program_name}.db')
        # pdbs subdirectories
        self.orient_dir = os.path.join(self.pdbs, 'oriented')
        self.orient_asu_dir = os.path.join(self.pdbs, 'oriented_asu')
        self.refine_dir = os.path.join(self.pdbs, 'refined')
        self.full_model_dir = os.path.join(self.pdbs, 'full_models')
        self.stride_dir = os.path.join(self.structure_info, 'stride')
        # sequence_info subdirectories
        self.sequences = os.path.join(self.sequence_info, 'sequences')
        self.profiles = os.path.join(self.sequence_info, 'profiles')
        # external database subdirectories
        self.pdb_api = os.path.join(self.external_db, 'pdb')
        # self.pdb_entity_api = os.path.join(self.external_db, 'pdb_entity')
        # self.pdb_assembly_api = os.path.join(self.external_db, 'pdb_assembly')
        self.uniprot_api = os.path.join(self.external_db, 'uniprot')

        # self.design_db = None
        # self.score_db = None
        putils.make_path(self.pdb_api)
        # putils.make_path(self.pdb_entity_api)
        # putils.make_path(self.pdb_assembly_api)
        putils.make_path(self.uniprot_api)

        self.api_db = wrapapi.api_database_factory.get(source=self.data)
        self.structure_db = structure_db.structure_database_factory.get(source=self.structure_info)
        # Set the job instance on these db objects
        self.api_db.job = self
        self.structure_db.job = self
        self.fragment_db: structure.fragment.db.FragmentDatabase | None = None
        if kwargs.get('database'):
            if self.development or self.debug:
                echo_db = True
            else:
                echo_db = False
            self.db: DBInfo = DBInfo(self.internal_db, echo=echo_db)
            # self.db: Engine = create_engine(f'sqlite:///{self.internal_db}', echo=True, future=True)
        else:  # When --no-database is provided as a flag
            self.db = None
        self.load_to_db = kwargs.get('load_to_db')
        self.reset_db = kwargs.get('reset_db')
        if self.reset_db:
            # All tables are deleted
            sql.Base.metadata.drop_all(self.db.engine)
            # Emit CREATE TABLE DDL
            sql.Base.metadata.create_all(self.db.engine)
        if not os.path.exists(self.internal_db):
            # Emit CREATE TABLE DDL
            sql.Base.metadata.create_all(self.db.engine)

        # PoseJob initialize Flags
        self.preprocessed = kwargs.get(flags.preprocessed)
        # self.pre_refined = kwargs.get('pre_refined')  # Todo
        # self.pre_loop_modeled = kwargs.get('pre_loop_modeled')  # Todo

        # Program flags
        # self.consensus: bool = kwargs.get(consensus, False)  # Whether to run consensus
        self.background_profile: str = kwargs.get('background_profile', putils.design_profile)
        """The type of position specific profile (per-residue amino acid frequencies) to utilize as the design 
        background profile. 
        Choices include putils.design_profile, putils.evolutionary_profile, and putils.fragment_profile
        """
        # Process design_selector
        self.design_selector: dict[str, dict[str, dict[str, set[int] | set[str]]]] | dict = \
            process_design_selector_flags(**kwargs)
        # self.design_selector = kwargs.get('design_selector', {})

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
                logger.warning(f"Couldn't use the flag --{flags.perturb_dof_steps} as --{flags.perturb_dof}"
                               f" wasn't set")
            if self.dock.perturb_dof_steps_rot is not None:
                logger.warning(f"Couldn't use the flag --{flags.perturb_dof_steps_rot} as --{flags.perturb_dof_rot}"
                               f" wasn't set")
            if self.dock.perturb_dof_steps_tx is not None:
                logger.warning(f"Couldn't use the flag --{flags.perturb_dof_steps_tx} as --{flags.perturb_dof_tx}"
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

        self.fuse_chains: list[tuple[str]] = [tuple(pair.split(':')) for pair in kwargs.get('fuse_chains', [])]

        self.interface = kwargs.get('interface')
        # Design flags
        self.design = Design.from_flags(**kwargs)
        # self.ignore_clashes: bool = kwargs.get(ignore_clashes, False)
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

        # Explicitly set to false if not designing or predicting
        if self.design.evolution_constraint and \
                (flags.design not in self.modules and flags.nanohedra not in self.modules and
                 flags.predict_structure not in self.modules):
            self.design.evolution_constraint = False

        # self.dock_only: bool = kwargs.get('dock_only')
        # if self.dock_only:
        #     self.design.sequences = self.design.structures = False
        self.only_write_frag_info: bool = kwargs.get('only_write_frag_info')
        # else:
        #     self.ignore_pose_clashes: bool = kwargs.get(ignore_pose_clashes, False)
        #     self.ignore_symmetric_clashes: bool = kwargs.get(ignore_symmetric_clashes, False)
        self.increment_chains: bool = kwargs.get('increment_chains')
        # self.evolution_constraint: bool = kwargs.get(evolution_constraint, False)
        # self.hbnet: bool = kwargs.get(hbnet, False)
        # self.term_constraint: bool = kwargs.get(term_constraint, False)
        # self.pre_refined: bool = kwargs.get('pre_refined', True)
        # self.pre_loop_modeled: bool = kwargs.get('pre_loop_modeled', True)
        self.interface_to_alanine: bool = kwargs.get('interface_to_alanine')
        self.metrics: bool = kwargs.get(flags._metrics)
        # self.scout: bool = kwargs.get(scout, False)
        self.specific_protocol: str = kwargs.get('specific_protocol')
        # self.structure_background: bool = kwargs.get(structure_background, False)
        # Process symmetry
        sym_entry = kwargs.get(putils.sym_entry)
        symmetry = kwargs.get('symmetry')
        if sym_entry is None and symmetry is None:
            self.sym_entry: SymEntry.SymEntry | str | None = None
        else:
            if symmetry and 'cryst' in symmetry.lower():
                # Later, symmetry information will be retrieved from the file header
                self.sym_entry = SymEntry.CRYST  # 'cryst'
            else:
                self.sym_entry = SymEntry.parse_symmetry_to_sym_entry(sym_entry=sym_entry, symmetry=symmetry)

        # Selection flags
        self.save_total = kwargs.get('save_total')
        # self.total = kwargs.get('total')
        self.protocol = kwargs.get(putils.protocol)
        _filter = kwargs.get('filter')
        _filter_file = kwargs.get('filter_file')
        if _filter or _filter_file is not None:
            self.filter = flags.parse_filters(_filter, file=_filter_file)
        elif _filter is not None:  # --filter was provided, but as a boolean-esq. Query the user for them once have a df
            self.filter = []
        else:
            self.filter = None
        _weight = kwargs.get('weight')
        _weight_file = kwargs.get('weight_file')
        if _weight or _weight_file is not None:
            self.weight = flags.parse_weights(_weight, file=_weight_file)
        elif _weight is not None:  # --weight was provided, but as a boolean-esq. Query the user once there is a df
            self.weight = []
        else:
            self.weight = None
        self.weight_function = kwargs.get('weight_function')
        self.select_number = kwargs.get('select_number')
        self.designs_per_pose = kwargs.get('designs_per_pose')
        # self.allow_multiple_poses = kwargs.get('allow_multiple_poses')
        self.tag_entities = kwargs.get(putils.tag_entities)
        # self.metric = kwargs.get('metric')
        self.specification_file = kwargs.get(putils.specification_file)
        # Don't need this at the moment...
        # self.poses = kwargs.get(flags.poses)
        """Used to specify whether specific designs should be fetched for select_* modules"""
        self.dataframe = kwargs.get('dataframe')
        self.metric = kwargs.get('metric')

        # Sequence flags
        self.avoid_tagging_helices = kwargs.get(putils.avoid_tagging_helices)
        self.csv = kwargs.get('csv')
        self.nucleotide = kwargs.get(flags.nucleotide)
        self.optimize_species = kwargs.get(putils.optimize_species)
        self.preferred_tag = kwargs.get(putils.preferred_tag)
        self.multicistronic = kwargs.get(putils.multicistronic)
        self.multicistronic_intergenic_sequence = kwargs.get(putils.multicistronic_intergenic_sequence)

        # Output flags
        self.overwrite: bool = kwargs.get('overwrite')
        self.pose_format = kwargs.get('pose_format')
        prefix = kwargs.get('prefix')
        if prefix:
            self.prefix = f'{prefix}_'
        else:
            self.prefix = ''

        suffix = kwargs.get('suffix')
        if suffix:
            self.suffix = f'_{suffix}'
        else:
            self.suffix = ''

        # Check if output already exists or --overwrite is provided
        if self.module in [flags.select_designs, flags.select_sequences]:
            if self.prefix == '':
                # self.location must not be None
                self.prefix = f'{utils.starttime}_{os.path.basename(os.path.splitext(self.input_source)[0])}_'
            output_directory = kwargs.get(putils.output_directory)
            # if not self.output_to_directory:
            if not output_directory:
                output_directory = os.path.join(os.path.dirname(self.program_root), f'SelectedDesigns')
                #     os.path.join(os.path.dirname(self.program_root), f'{self.prefix}SelectedDesigns{self.suffix}')
        else:  # if output_directory:
            output_directory = kwargs.get(putils.output_directory)

        if output_directory:
            if os.path.exists(output_directory):
                if not self.overwrite:
                    exit(f'The specified output directory "{output_directory}" already exists, this will overwrite '
                         f'your old data! Please specify a new name with with -Od/--{flags.output_directory}, '
                         '--prefix or --suffix, or append --overwrite to your command')
            else:
                putils.make_path(output_directory)
            self.output_directory = output_directory

        self.output_file = kwargs.get(putils.output_file)
        if self.output_file and os.path.exists(self.output_file) and self.module not in flags.analysis \
                and not self.overwrite:
            exit(f'The specified output file "{self.output_file}" already exists, this will overwrite your old '
                 f'data! Please specify a new name with with {flags.format_args(flags.output_file_args)}, '
                 'or append --overwrite to your command')

        # When we are performing expand-asu, make sure we set output_assembly to True
        if self.module == flags.expand_asu:
            self.output_assembly = True
        else:
            self.output_assembly: bool = kwargs.get(putils.output_assembly)
        self.output_surrounding_uc: bool = kwargs.get(putils.output_surrounding_uc)
        self.write_fragments: bool = kwargs.get(putils.output_fragments)
        self.write_oligomers: bool = kwargs.get(putils.output_oligomers)
        self.write_structures: bool = kwargs.get(putils.output_structures)
        self.write_trajectory: bool = kwargs.get(putils.output_trajectory)

        self.skip_logging: bool = kwargs.get(putils.skip_logging)
        self.merge: bool = kwargs.get('merge')
        self.save: bool = kwargs.get('save')
        self.figures: bool = kwargs.get('figures')

        if self.write_structures or self.output_assembly or self.output_surrounding_uc or self.write_fragments \
                or self.write_oligomers or self.write_trajectory:
            self.output: bool = True
        else:
            self.output: bool = False

        # self.nanohedra_output: bool = kwargs.get(flags.nanohedra_output)
        # self.nanohedra_root: str | None = None
        # if self.nanohedra_output:
        #     self.construct_pose: bool = kwargs.get('construct_pose', True)
        # else:
        self.construct_pose = True

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
        # Todo this is pretty sloppy. I should modify this DataClass mechanism...
        # self.cluster_map = kwargs.get('cluster_map')
        # self.as_objects: bool = kwargs.get('as_objects')
        # self.mode: bool = kwargs.get('mode')
        if flags.cluster_poses in self.modules or 'cluster_map' in kwargs:
            self.cluster = Cluster.from_flags(**kwargs)
            # self.cluster.map: AnyStr
            # """The path to a file containing the currently loaded mapping from cluster representatives to members"""
        else:
            self.cluster = False

        # Finally perform checks on desired work to see if viable
        if self.protocol_module:
            self.check_protocol_module_arguments()

    @property
    def modules(self) -> list[str]:
        """Return the modules slated to run during the job"""
        try:
            return self._modules
        except AttributeError:
            return [self.module]

    @modules.setter
    def modules(self, modules: Iterable[str]) -> list[str]:
        self._modules = list(modules)

    @property
    def current_session(self) -> Session:
        """Contains the sqlalchemy.orm.Session that is currently in use for access to database attributes"""
        try:
            return self._session
        except AttributeError:  # No session connected
            raise AttributeError("Couldn't return current_session as there is not an active Session. Ensure you "
                                 "initialize a job context manager, i.e.\n"
                                 "with job.db.session() as session:\n"
                                 "    job.current_session = session\n"
                                 "Before you attempt to use the current_session")

    @current_session.setter
    def current_session(self, session: Session):
        """Set the sqlalchemy.orm.Session that is currently in use to access database attributes later during the job"""
        if isinstance(session, Session):
            self._session = session

    @property
    def output_to_directory(self) -> bool:
        """Set so it is known that output is not typical putils.program_output directory structure"""
        # self.output_to_directory: bool = True if self.output_directory else False
        if self.module in [flags.select_designs, flags.select_sequences]:
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
            raise ValueError(f"Couldn't handle the provided location type {type(location).__name__}")

    @property
    def input_source(self) -> str:
        """Provide the name of the specified PoseJob instances to perform work on"""
        try:
            return self._input_source
        except AttributeError:
            self._input_source = self.program_root
            return self._input_source

    @property
    def construct_pose(self):
        """Whether to construct the PoseJob"""
        return self._construct_pose

    @construct_pose.setter
    def construct_pose(self, value: bool):
        self._construct_pose = value
        if self._construct_pose:
            pass
        else:  # No construction specific flags
            self.write_fragments = self.write_oligomers = False

    # Todo make part of modules.setter routine
    def check_protocol_module_arguments(self):
        """Given provided modules for the 'protocol' module, check to ensure the work is adequate

        Raises:
            InputError if the inputs are found to be incompatible
        """
        protocol_module_allowed_modules = [
            # 'find_asu',
            flags.orient,
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
        # disallowed_modules = [
        #     # 'custom_script',
        #     # flags.select_sequences,
        # ]

        def check_gpu() -> str | bool:
            available_devices = jax.local_devices()
            for idx, device in enumerate(available_devices):
                if device.platform == 'gpu':
                    # self.gpu_available = True  # Todo could be useful
                    return device.device_kind
                    # device_id = idx
                    # return True
            return False

        problematic_modules = []
        not_recognized_modules = []
        nanohedra_prior = False
        gpu_device_kind = None
        for idx, module in enumerate(self.modules):
            if module in protocol_module_allowed_modules:
                if module == flags.nanohedra:
                    if idx > 0:
                        raise InputError(f"For {flags.protocol} module, {flags.nanohedra} can currently only be run as "
                                         f"module position #1")
                    nanohedra_prior = True
                    continue
                elif module == flags.predict_structure:
                    if gpu_device_kind is None:
                        # Check for GPU access
                        gpu_device_kind = check_gpu()

                    if gpu_device_kind:
                        logger.info(f'Running {flags.predict_structure} on {gpu_device_kind} GPU')
                        # disable GPU on tensorflow
                        tf.config.set_visible_devices([], 'GPU')
                    else:  # device.platform == 'cpu':
                        logger.warning(f'No GPU detected, will {flags.predict_structure} using CPU')
                elif module == flags.design:
                    if self.design.method == putils.proteinmpnn:
                        if gpu_device_kind is None:
                            # Check for GPU access
                            gpu_device_kind = check_gpu()

                        if gpu_device_kind:
                            logger.info(f'Running {flags.design} on {gpu_device_kind} GPU')
                        else:  # device.platform == 'cpu':
                            logger.warning(f'No GPU detected, will {flags.design} using CPU')

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
            # elif module in disallowed_modules:
            #     problematic_modules.append(module)
            else:
                not_recognized_modules.append(module)

        if not_recognized_modules:
            raise InputError(f'For {flags.protocol} module, the --{flags.modules} '
                             f'{", ".join(not_recognized_modules)} are not recognized modules. See'
                             f'"{putils.program_help}" for the available module names')

        if problematic_modules:
            raise InputError(f'For {flags.protocol} module, the --{flags.modules} '
                             f'{", ".join(problematic_modules)} are not possible modules\n'
                             f'Allowed modules are {", ".join(protocol_module_allowed_modules)}')

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
            reported_args[putils.sym_entry] = self.sym_entry.number
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
        entire_parser = flags.argparsers[flags.parser_entire]
        for group in entire_parser._action_groups:
            for arg in group._group_actions:
                if isinstance(arg, argparse._SubParsersAction):  # We have a subparser, recurse
                    for name, sub_parser in arg.choices.items():
                        for sub_group in sub_parser._action_groups:
                            for arg in sub_group._group_actions:
                                report_arg(arg.dest, arg.default)
                else:
                    report_arg(arg.dest, arg.default)

        return reported_args

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
        if self.protocol_module and len(self.modules) > 2:
            self.reduce_memory = True
        elif available_memory < required_memory:
            self.reduce_memory = True
        else:
            # Todo when requirements are more accurate with database
            #  self.reduce_memory = False
            self.reduce_memory = True
        logger.debug(f'Reduce job memory?: {self.reduce_memory}')

    @staticmethod
    def can_process_evolutionary_profiles() -> bool:
        """Return True if the current computer has the computational requirements to collect evolutionary profiles"""
        # Run specific checks
        if psutil.virtual_memory().available <= distribute.hhblits_memory_threshold:
            logger.critical(f'The available RAM is insufficient to run {putils.hhblits}. Required memory: '
                            f'{distribute.hhblits_memory_threshold / gb_divisior:.2f} GB\n')
            #                 '\tPlease allocate the job to a computer with more memory or the process will fail, '
            #                 f'otherwise, submit the job with --no-{flags.evolution_constraint}')
            # exit(1)
            logger.critical(f'Creating scripts that can be distributed to a capable computer instead')
            return False
        return True

    def process_evolutionary_info(self, uniprot_entities: Iterable[wrapapi.UniProtEntity] = None,
                                  entities: Iterable[structure.sequence.SequenceProfile] = None) -> list[str]:
        """Format the job with evolutionary constraint options

        Args:
            uniprot_entities: A list of the UniProtIDs for the Job
            entities: A list of the Entity instances initialized for the Job
        Returns:
            A list evolutionary setup instructions
        """
        info_messages = []
        hhblits_cmds, bmdca_cmds = [], []
        # Set up sequence data using hhblits and profile bmDCA for each input entity
        putils.make_path(self.sequences)
        if uniprot_entities:
            for uniprot_entity in uniprot_entities:
                evolutionary_profile = self.api_db.hhblits_profiles.retrieve_data(name=uniprot_entity.id)
                if not evolutionary_profile:
                    hhblits_cmds.append(hhblits(uniprot_entity.id,
                                                sequence=uniprot_entity.reference_sequence,
                                                out_dir=self.profiles, threads=self.threads,
                                                return_command=True))
                # TODO reinstate
                #  Solve .h_fields/.j_couplings
                # # Before this is run, hhblits must be run and the file located at profiles/entity-name.fasta contains
                # # the multiple sequence alignment in .fasta format
                # bmdca_cmds.append([putils.bmdca_exe_path,
                #                    '-i', os.path.join(self.profiles, f'{uniprot_entity.id}.fasta'),
                #                    '-d', os.path.join(self.profiles, f'{uniprot_entity.id}_bmDCA')])
        else:
            for entity in entities:
                entity.sequence_file = self.api_db.sequences.retrieve_file(name=entity.name)
                if not entity.sequence_file:
                    entity.write_sequence_to_fasta('reference', out_dir=self.sequences)
                else:
                    entity.evolutionary_profile = self.api_db.hhblits_profiles.retrieve_data(name=entity.name)
                    # TODO reinstate
                    #  entity.h_fields = self.api_db.bmdca_fields.retrieve_data(name=entity.name)
                    #  entity.j_couplings = self.api_db.bmdca_couplings.retrieve_data(name=entity.name)
                if not entity.evolutionary_profile:
                    # To generate in current runtime
                    # entity.add_evolutionary_profile(out_dir=self.api_db.hhblits_profiles.location)
                    # To generate in a sbatch script
                    hhblits_cmds.append(entity.hhblits(out_dir=self.profiles, return_command=True))
                # TODO reinstate
                # # Before this is run, hhblits must be run and the file located at profiles/entity-name.fasta contains
                # # the multiple sequence alignment in .fasta format
                #  if not entity.j_couplings:
                #    bmdca_cmds.append([putils.bmdca_exe_path, '-i', os.path.join(self.profiles, f'{entity.name}.fasta'),
                #                       '-d', os.path.join(self.profiles, f'{entity.name}_bmDCA')])

        if hhblits_cmds:
            if not os.access(putils.hhblits_exe, os.X_OK):
                raise RuntimeError(f"Couldn't locate the {putils.hhblits} executable. Ensure the executable file "
                                   f"referenced by '{putils.hhblits_exe} exists then try your job again. Otherwise, use"
                                   f" the argument --no-{flags.evolution_constraint} OR set up hhblits to run."
                                   f'{guide.hhblits_setup_instructions}')
            putils.make_path(self.profiles)
            putils.make_path(self.sbatch_scripts)
            # hhblits_cmds, reformat_msa_cmds = zip(*profile_cmds)
            # hhblits_cmds, _ = zip(*hhblits_cmds)
            if not os.access(putils.reformat_msa_exe_path, os.X_OK):
                logger.error(f"Couldn't execute multiple sequence alignment reformatting script")
                reformat_msa_cmd1 = reformat_msa_cmd2 = []
            else:
                reformat_msa_cmd1 = [putils.reformat_msa_exe_path, 'a3m', 'sto',
                                     f"'{os.path.join(self.profiles, '*.a3m')}'", '.sto', '-num', '-uc']
                reformat_msa_cmd2 = [putils.reformat_msa_exe_path, 'a3m', 'fas',
                                     f"'{os.path.join(self.profiles, '*.a3m')}'", '.fasta', '-M', 'first', '-r']
            hhblits_log_file = os.path.join(self.profiles, 'generate_profiles.log')
            # Run hhblits commands
            if self.can_process_evolutionary_profiles():
                logger.info(f'Writing {putils.hhblits} results to file: {hhblits_log_file}')
                # Run commands in this process
                if self.multi_processing:
                    zipped_args = zip(hhblits_cmds, repeat(hhblits_log_file))
                    # utils.distribute.run(cmd, hhblits_log_file)
                    # Todo calculate how many cores are available to use given memory limit
                    utils.mp_starmap(utils.distribute.run, zipped_args, processes=self.cores)
                else:
                    with open(hhblits_log_file, 'w') as f:
                        for cmd in hhblits_cmds:
                            p = subprocess.Popen(cmd, stdout=f, stderr=f)
                            p.communicate()

                # Format .a3m multiple sequence alignments to .sto/.fasta
                with open(hhblits_log_file, 'w') as f:
                    p = subprocess.Popen(reformat_msa_cmd1, stdout=f, stderr=f)
                    p.communicate()
                    p = subprocess.Popen(reformat_msa_cmd2, stdout=f, stderr=f)
                    p.communicate()
                # Todo this would be more preferable
                # for cmd in hhblits_cmds:
                #     p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                #     stdout, stderr = p.communicate()
                #     if stdout or stderr:
                #         logger.info()
            else:  # Convert each command to a string and write to distribute
                instructions = 'Please follow the instructions below to generate sequence profiles for input proteins'
                info_messages.append(instructions)
                hhblits_cmds = [subprocess.list2cmdline(cmd) for cmd in hhblits_cmds]
                hhblits_cmd_file = utils.write_commands(hhblits_cmds, name=f'{utils.starttime}-{putils.hhblits}',
                                                        out_path=self.profiles)
                hhblits_script = \
                    distribute.distribute(file=hhblits_cmd_file, out_path=self.sbatch_scripts,
                                          scale=putils.hhblits, max_jobs=len(hhblits_cmds),
                                          number_of_commands=len(hhblits_cmds), log_file=hhblits_log_file,
                                          finishing_commands=[list2cmdline(reformat_msa_cmd1),
                                                              list2cmdline(reformat_msa_cmd2)])
                hhblits_job_info_message = \
                    f'Enter the following to distribute {putils.hhblits} jobs:\n\t'
                if distribute.is_sbatch_available():
                    hhblits_job_info_message += f'{distribute.sbatch} {hhblits_script}'
                else:
                    hhblits_job_info_message += f'{distribute.default_shell} {hhblits_script}'
                info_messages.append(hhblits_job_info_message)

        if bmdca_cmds:
            putils.make_path(self.profiles)
            putils.make_path(self.sbatch_scripts)
            # bmdca_cmds = \
            #     [list2cmdline([putils.bmdca_exe_path, '-i', os.path.join(self.profiles, '%s.fasta' % entity.name),
            #                   '-d', os.path.join(self.profiles, '%s_bmDCA' % entity.name)])
            #      for entity in entities.values()]
            bmdca_cmd_file = \
                utils.write_commands(bmdca_cmds, name=f'{utils.starttime}-bmDCA', out_path=self.profiles)
            bmdca_script = utils.distribute.distribute(file=bmdca_cmd_file, out_path=self.sbatch_scripts,
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

            print('\n' * 2)
            # Todo add bmdca_sbatch to hhblits_cmds finishing_commands kwarg
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
        #         Keyword Args:
        #             source: The JobResource source name
        return self.__call__(**kwargs)


job_resources_factory: Annotated[JobResourcesFactory,
                                 'Calling this factory method returns the single instance of the JobResources class'] \
    = JobResourcesFactory()
"""Calling this factory method returns the single instance of the JobResources class"""
