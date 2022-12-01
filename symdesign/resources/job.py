from __future__ import annotations

import argparse
import logging
import inspect
import os
from dataclasses import make_dataclass, field
from subprocess import list2cmdline
from typing import Annotated, AnyStr, Any, Iterable

import psutil

from symdesign import flags, utils
from symdesign.resources import structure_db, wrapapi
from symdesign.structure.fragment import db
from symdesign.structure.model import Entity
from symdesign.structure.sequence import read_fasta_file  # Todo refactor to structure.utils?
from symdesign.utils import CommandDistributer, SymEntry, InputError, path as putils

logger = logging.getLogger(__name__)


def generate_sequence_mask(fasta_file: AnyStr) -> list[int]:
    """From a sequence with a design_selector, grab the residue indices that should be designed in the target
    structural calculation

    Args:
        fasta_file: The path to a file with fasta information
    Returns:
        The residue numbers (in pose format) that should be ignored in design
    """
    sequence_and_mask = list(read_fasta_file(fasta_file))
    # sequences = sequence_and_mask
    sequence, mask, *_ = sequence_and_mask
    if not len(sequence) == len(mask):
        raise ValueError('The sequence and design_selector are different lengths! Please correct the alignment and '
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


def from_flags(cls, **kwargs):
    return cls(**{key: value for key, value in kwargs.items()
                  if key in inspect.signature(cls).parameters})


# These dataclasses help simplify the use of flags into namespaces
# The commented out versions below had poor implementations
# DesignFlags = collections.namedtuple('DesignFlags', design_args.keys(), defaults=design_args.values())
# DesignFlags = types.SimpleNamespace(**design_args)
Design = make_dataclass('Design',
                        [(flag, eval(type(default).__name__), field(default=default))
                         for flag, default in flags.design.items()],
                        namespace={'from_flags': classmethod(from_flags)})
#                         frozen=True)
#  self.design = DesignFlags(*[kwargs.get(argument_name) for argument_name in design_args.keys()])
#  self.design = types.SimpleNamespace(**{flag: kwargs.get(flag, default) for flag, default in flags.design})
#  self.design = types.SimpleNamespace(**{flag: kwargs.get(flag, default) for flag, default in flags.design})
Dock = make_dataclass('Dock',
                      [(flag, eval(type(default).__name__), field(default=default))
                       for flag, default in flags.dock.items()],
                      namespace={'from_flags': classmethod(from_flags)})
#                       frozen=True)

Predict = make_dataclass('Predict',
                         [(flag, eval(type(default).__name__), field(default=default))
                          for flag, default in flags.predict.items()],
                         namespace={'from_flags': classmethod(from_flags)})
#                          frozen=True)


class JobResources:
    """The intention of JobResources is to serve as a singular source of design info which is common across all
    jobs. This includes common paths, databases, and design flags which should only be set once in program operation,
    then shared across all member designs"""
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
                raise FileNotFoundError(f"Path doesn't exist!\n\t{program_root}")
        except TypeError:
            raise TypeError(f"Can't initialize {JobResources.__name__} without parameter 'program_root'")

        # Format argparse.Namespace arguments
        if arguments is not None:
            kwargs.update(vars(arguments))

        # program_root subdirectories
        self.data = os.path.join(self.program_root, putils.data.title())
        self.projects = os.path.join(self.program_root, putils.projects)
        self.job_paths = os.path.join(self.program_root, 'JobPaths')
        self.sbatch_scripts = os.path.join(self.program_root, 'Scripts')
        # TODO ScoreDatabase integration
        self.all_scores = os.path.join(self.program_root, putils.all_scores)

        # data subdirectories
        self.clustered_poses = os.path.join(self.data, 'ClusteredPoses')
        self.structure_info = os.path.join(self.data, putils.structure_info)
        self.pdbs = os.path.join(self.structure_info, 'PDBs')  # Used to store downloaded PDB's
        self.sequence_info = os.path.join(self.data, putils.sequence_info)
        self.external_db = os.path.join(self.data, 'ExternalDatabases')
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
        # try:
        # if not self.projects:  # used for subclasses
        # if not getattr(self, 'projects', None):  # used for subclasses
        #     self.projects = os.path.join(self.program_root, projects)
        # except AttributeError:
        #     self.projects = os.path.join(self.program_root, projects)
        # self.design_db = None
        # self.score_db = None
        putils.make_path(self.pdb_api)
        # putils.make_path(self.pdb_entity_api)
        # putils.make_path(self.pdb_assembly_api)
        putils.make_path(self.uniprot_api)
        self.module: str = kwargs.get(flags.module)
        self.modules: str = kwargs.get(flags.modules)
        self.check_module_arguments()
        # self.reduce_memory = False
        self.api_db = wrapapi.api_database_factory.get(source=self.data)
        self.structure_db = structure_db.structure_database_factory.get(source=self.data)
        self.fragment_db: 'db.FragmentDatabase' | None = None

        # Setup Flags
        self.initial_refinement = self.initial_loop_model = None
        # Computing environment and development Flags
        self.cores: int = kwargs.get('cores', 0)
        self.distribute_work: bool = kwargs.get(putils.distribute_work)
        self.mpi: int = kwargs.get('mpi', 0)
        self.multi_processing: int = kwargs.get(putils.multi_processing)
        if self.multi_processing:
            # Calculate the number of cores to use depending on computer resources
            self.cores = utils.calculate_mp_cores(cores=self.cores)  # mpi=self.mpi, Todo
        else:
            self.cores = 1

        if self.mpi > 0:
            self.distribute_work = True
        self.development: bool = kwargs.get(putils.development)
        # self.command_only: bool = kwargs.get('command_only', False)
        """Whether to reissue commands, only if distribute_work=False"""

        # Program flags
        # self.consensus: bool = kwargs.get(consensus, False)  # Whether to run consensus
        self.as_objects: bool = kwargs.get('as_objects')
        self.mode: bool = kwargs.get('mode')
        self.background_profile: str = kwargs.get('background_profile', putils.design_profile)
        """The type of position specific profile (per-residue amino acid frequencies) to utilize as the design 
        background profile. 
        Choices include putils.design_profile, putils.evolutionary_profile, and putils.fragment_profile
        """
        # Process design_selector
        self.design_selector: dict[str, dict[str, dict[str, set[int] | set[str]]]] | dict = \
            process_design_selector_flags(**kwargs)
        # self.design_selector = kwargs.get('design_selector', {})
        self.dock = Dock.from_flags(**kwargs)
        if self.dock.perturb_dof:
            if not self.dock.perturb_dof_rot and not self.dock.perturb_dof_tx:
                # Set all perturb_dof on and set to the provided default
                self.dock.perturb_dof_rot = self.dock.perturb_dof_tx = True
                self.dock.perturb_dof_steps_rot = self.dock.perturb_dof_steps_tx = self.dock.perturb_dof_steps
            else:  # Use provided values
                pass
        elif self.dock.perturb_dof_rot or self.dock.perturb_dof_tx:
            self.dock.perturb_dof = True
        else:  # None provided, set the unavailable dof to 1 step
            if not self.dock.perturb_dof_rot:
                self.dock.perturb_dof_steps_rot = 1
            elif not self.dock.perturb_dof_tx:
                self.dock.perturb_dof_steps_tx = 1
            else:
                self.dock.perturb_dof_steps_rot = self.dock.perturb_dof_steps_tx = 1
        # self.proteinmpnn_score: bool = kwargs.get('proteinmpnn_score', False)
        # self.contiguous_ghosts: bool = kwargs.get('contiguous_ghosts', False)

        # self.rotation_step1: bool = kwargs.get('rotation_step1', False)
        # self.rotation_step2: bool = kwargs.get('rotation_step2', False)
        # self.min_matched: bool = kwargs.get('min_matched', False)
        # self.match_value: bool = kwargs.get('match_value', False)
        # self.initial_z_value: bool = kwargs.get('initial_z_value', False)
        self.log_level: bool = kwargs.get('log_level')
        self.debug: bool = True if self.log_level == 1 else False
        self.force: bool = kwargs.get(putils.force)
        self.fuse_chains: list[tuple[str]] = [tuple(pair.split(':')) for pair in kwargs.get('fuse_chains', [])]
        self.design = Design.from_flags(**kwargs)
        # self.ignore_clashes: bool = kwargs.get(ignore_clashes, False)
        if self.design.ignore_clashes:
            self.design.ignore_pose_clashes = self.design.ignore_symmetric_clashes = True
        # Handle protocol specific flags
        if not self.design.term_constraint:
            self.generate_fragments: bool = False
        else:
            self.generate_fragments = True

        if self.design.structure_background:
            self.design.evolution_constraint = False
            self.design.hbnet = False
            self.design.scout = False
            self.design.term_constraint = False

        self.dock_only: bool = kwargs.get('dock_only')
        if self.dock_only:
            self.design.sequences = self.design.structures = False
        self.only_write_frag_info: bool = kwargs.get('only_write_frag_info')
        # else:
        #     self.ignore_pose_clashes: bool = kwargs.get(ignore_pose_clashes, False)
        #     self.ignore_symmetric_clashes: bool = kwargs.get(ignore_symmetric_clashes, False)
        self.increment_chains: bool = kwargs.get('increment_chains')
        # self.evolution_constraint: bool = kwargs.get(evolution_constraint, False)
        # self.hbnet: bool = kwargs.get(hbnet, False)
        # self.term_constraint: bool = kwargs.get(term_constraint, False)
        # self.number_of_trajectories: int = kwargs.get(number_of_trajectories, flags.nstruct)
        # self.pre_refine: bool = kwargs.get('pre_refine', True)
        # self.pre_loop_model: bool = kwargs.get('pre_loop_model', True)
        self.interface_to_alanine: bool = kwargs.get('interface_to_alanine')
        self.metrics: bool = kwargs.get(flags.metrics)
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
        self.total = kwargs.get('total')
        self.protocol = kwargs.get(putils.protocol)
        self.filter = kwargs.get('filter')
        self.weight = kwargs.get('weight')
        self.weight_function = kwargs.get('weight_function')
        self.select_number = kwargs.get('select_number')
        self.designs_per_pose = kwargs.get('designs_per_pose')
        self.allow_multiple_poses = kwargs.get('allow_multiple_poses')
        # self.metric = kwargs.get('metric')
        self.specification_file = kwargs.get('specification_file')
        self.dataframe = kwargs.get('dataframe')
        self.cluster_map = kwargs.get('cluster_map')

        # Output flags
        prefix = kwargs.get('prefix')
        if prefix:
            self.prefix = f'{prefix}_'
        else:
            self.prefix = ''

        suffix = kwargs.get('suffix')
        if suffix:
            self.suffix = f'{suffix}_'
        else:
            self.suffix = ''

        self.overwrite: bool = kwargs.get('overwrite')
        self.output_directory: AnyStr | None = kwargs.get(putils.output_directory)
        self.output_to_directory: bool = True if self.output_directory else False
        """Set so it is known that output is not typical SymDesignOutput directory structure"""
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
        self.skip_sequence_generation: bool = kwargs.get('skip_sequence_generation')

        if self.write_structures or self.output_assembly or self.output_surrounding_uc or self.write_fragments \
                or self.write_oligomers or self.write_trajectory:
            self.output: bool = True
        else:
            self.output: bool = False

        self.nanohedra_output: bool = kwargs.get(flags.nanohedra_output)
        self.nanohedra_root: str | None = None
        if self.nanohedra_output:
            self.construct_pose: bool = kwargs.get('construct_pose', True)
        else:
            self.construct_pose = True

        self.predict = Predict.from_flags(**kwargs)

    @property
    def construct_pose(self):
        """Whether to construct the PoseDirectory"""
        return self._construct_pose

    @construct_pose.setter
    def construct_pose(self, value: bool):
        self._construct_pose = value
        if self._construct_pose:
            pass
        else:  # No construction specific flags
            self.write_fragments = self.write_oligomers = False

    def check_module_arguments(self):
        """Given provided modules for the 'protocol' module, check to ensure the work is adequate

        Raises:
            InputError if the inputs are found to be incompatible
        """
        if self.module == flags.protocol:
            allowed_modules = [
                # 'find_asu',
                flags.orient,
                flags.expand_asu,
                flags.rename_chains,
                flags.check_clashes,
                flags.generate_fragments,
                flags.nanohedra,
                flags.interface_metrics,
                flags.optimize_designs,
                flags.refine,
                flags.interface_design,
                flags.analysis,

            ]
            disallowed_modules = [
                # 'custom_script',
                flags.cluster_poses,
                flags.select_poses,
                flags.select_sequences,
                flags.select_designs,  # As alias for select_sequences with --skip-sequence-generation
            ]
            problematic_modules = []
            not_recognized_modules = []
            for idx, module in enumerate(self.modules):
                if module in disallowed_modules:
                    problematic_modules.append(module)
                elif module in allowed_modules:
                    if idx > 0 and module == flags.nanohedra:
                        raise InputError(f"For {flags.protocol} module, {flags.nanohedra} can only be run as module 1")
                else:
                    not_recognized_modules.append(module)

            if not_recognized_modules:
                raise InputError(f'For {flags.protocol} module, the --{flags.modules} '
                                 f'{", ".join(not_recognized_modules)} are not recognized modules. See'
                                 f'\n{putils.symdesign_help}\nfor available module names')

            if problematic_modules:
                raise InputError(f'For {flags.protocol} module, the --{flags.modules} '
                                 f'{", ".join(problematic_modules)} are not possible modules\n'
                                 f'Allowed modules are {", ".join(allowed_modules)}')

    def report_specified_arguments(self, arguments: argparse.Namespace) -> dict[str, Any]:
        """Filter all flags for only those that were specified as different on the command line

        Args:
            arguments: The arguments as parsed from the command-line argparse namespace
        Returns:
            Arguments specified during program execution
        """
        arguments = vars(arguments).copy()

        # Get all the default program args and compare them to the provided values
        reported_args = {}
        entire_parser = flags.argparsers[flags.parser_entire]
        for group in entire_parser._action_groups:
            for arg in group._group_actions:
                if isinstance(arg, argparse._SubParsersAction):  # We have a subparser, recurse
                    for name, sub_parser in arg.choices.items():
                        for sub_group in sub_parser._action_groups:
                            for arg in sub_group._group_actions:
                                value = arguments.pop(arg.dest, None)  # Get the parsed flag value
                                if value is not None and value != arg.default:  # Compare it to the default
                                    reported_args[arg.dest] = value  # Add it to reported args if not the default
                else:
                    value = arguments.pop(arg.dest, None)  # Get the parsed flag value
                    if value is not None and value != arg.default:  # Compare it to the default
                        reported_args[arg.dest] = value  # Add it to reported args if not the default

        # # Custom removal/formatting for all remaining
        # for custom_arg in list(arguments.keys()):
        #     value = arguments.pop(custom_arg, None)
        #     if value is not None:
        #         reported_args[custom_arg] = value

        # Where input values should be reported instead of processed version, or the argument is not important, format
        if self.sym_entry:
            reported_args[putils.sym_entry] = self.sym_entry.entry_number
        # if self.design_selector:
        #     reported_args.pop('design_selector', None)

        return reported_args

    def report_unspecified_arguments(self, arguments: argparse.Namespace) -> dict[str, Any]:
        """Filter all flags for only those that were specified as different on the command line

        Args:
            arguments: The arguments as parsed from the command-line argparse namespace
        Returns:
            Arguments specified during program execution
        """
        arguments = vars(arguments).copy()

        # Get all the default program args and compare them to the provided values
        reported_args = {}
        entire_parser = flags.argparsers[flags.parser_entire]
        for group in entire_parser._action_groups:
            for arg in group._group_actions:
                if isinstance(arg, argparse._SubParsersAction):  # We have a subparser, recurse
                    for name, sub_parser in arg.choices.items():
                        for sub_group in sub_parser._action_groups:
                            for arg in sub_group._group_actions:
                                value = arguments.pop(arg.dest, None)  # Get the parsed flag value
                                if value is not None:  # Compare it to the default
                                    reported_args[arg.dest] = value  # Add it to reported args if not the default
                else:
                    value = arguments.pop(arg.dest, None)  # Get the parsed flag value
                    if value is not None:  # Compare it to the default
                        reported_args[arg.dest] = value  # Add it to reported args if not the default

        # # Custom removal/formatting for all remaining
        # for custom_arg in list(arguments.keys()):
        #     value = arguments.pop(custom_arg, None)
        #     if value is not None:
        #         reported_args[custom_arg] = value

        # Where input values should be reported instead of processed version, or the argument is not important, format
        if self.sym_entry:
            reported_args[putils.sym_entry] = self.sym_entry.entry_number
        # if self.design_selector:
        #     reported_args.pop('design_selector', None)

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
        logger.debug(f'Available memory: {available_memory:f}')
        logger.debug(f'Required memory: {required_memory:f}')
        if available_memory < required_memory:
            self.reduce_memory = True
        logger.debug(f'Reduce job memory?: {self.reduce_memory}')

        # Run specific checks
        if self.module == flags.interface_design and self.design.evolution_constraint:  # hhblits to run
            if psutil.virtual_memory().available <= required_memory + CommandDistributer.hhblits_memory_threshold:
                logger.critical(f'The amount of memory for the computer is insufficient to run {putils.hhblits} '
                                '(required for designing with evolution)! Please allocate the job to a computer with '
                                f'more memory or the process will fail. '
                                f'Otherwise, submit job with --no-{flags.evolution_constraint}')
                exit(1)
            putils.make_path(self.sequences)
            putils.make_path(self.profiles)

    def setup_evolution_constraint(self, all_entities: Iterable[Entity]) -> list[str]:
        """Format the job with evolutionary constraint options

        Args:
            all_entities: A list of the Entity instances initialized for the Job
        Returns:
            A list evolutionary setup instructions
        """
        info_messages = []
        hhblits_cmds, bmdca_cmds = [], []
        # Set up sequence data using hhblits and profile bmDCA for each input entity
        putils.make_path(self.sequences)
        for entity in all_entities:
            entity.sequence_file = self.api_db.sequences.retrieve_file(name=entity.name)
            if not entity.sequence_file:
                entity.write_sequence_to_fasta('reference', out_dir=self.sequences)
                # entity.add_evolutionary_profile(out_dir=self.api_db.hhblits_profiles.location)
            else:
                entity.evolutionary_profile = self.api_db.hhblits_profiles.retrieve_data(name=entity.name)
                # entity.h_fields = self.api_db.bmdca_fields.retrieve_data(name=entity.name)
                # TODO reinstate entity.j_couplings = self.api_db.bmdca_couplings.retrieve_data(name=entity.name)
            if not entity.evolutionary_profile:
                # To generate in current runtime
                # entity.add_evolutionary_profile(out_dir=self.api_db.hhblits_profiles.location)
                # To generate in a sbatch script
                # profile_cmds.append(entity.hhblits(out_dir=self.profiles, return_command=True))
                hhblits_cmds.append(entity.hhblits(out_dir=self.profiles, return_command=True))
            # TODO reinstate
            # before this is run, hhblits must be run and the file located at profiles/entity-name.fasta contains
            # the multiple sequence alignment in .fasta format
            #  if not entity.j_couplings:
            #    bmdca_cmds.append([putils.bmdca_exe_path, '-i', os.path.join(self.profiles, f'{entity.name}.fasta'),
            #                       '-d', os.path.join(self.profiles, f'{entity.name}_bmDCA')])

        if hhblits_cmds:
            if not os.access(putils.hhblits_exe, os.X_OK):
                print(f"Couldn't locate the {putils.hhblits} executable. Ensure the executable file referenced by "
                      f'{putils.hhblits_exe} exists then try your job again. Otherwise, use the argument'
                      f'--no-{flags.evolution_constraint}')
                exit()
            putils.make_path(self.profiles)
            putils.make_path(self.sbatch_scripts)
            # Prepare files for running hhblits commands
            instructions = 'Please follow the instructions below to generate sequence profiles for input proteins'
            info_messages.append(instructions)
            # hhblits_cmds, reformat_msa_cmds = zip(*profile_cmds)
            # hhblits_cmds, _ = zip(*hhblits_cmds)
            reformat_msa_cmd1 = [putils.reformat_msa_exe_path, 'a3m', 'sto',
                                 f"'{os.path.join(self.profiles, '*.a3m')}'", '.sto', '-num', '-uc']
            reformat_msa_cmd2 = [putils.reformat_msa_exe_path, 'a3m', 'fas',
                                 f"'{os.path.join(self.profiles, '*.a3m')}'", '.fasta', '-M', 'first', '-r']
            hhblits_cmd_file = utils.write_commands(hhblits_cmds, name=f'{utils.starttime}-{putils.hhblits}',
                                                    out_path=self.profiles)
            hhblits_sbatch = \
                CommandDistributer.distribute(file=hhblits_cmd_file, out_path=self.sbatch_scripts,
                                              scale=putils.hhblits, max_jobs=len(hhblits_cmds),
                                              number_of_commands=len(hhblits_cmds),
                                              log_file=os.path.join(self.profiles, 'generate_profiles.log'),
                                              finishing_commands=[list2cmdline(reformat_msa_cmd1),
                                                                  list2cmdline(reformat_msa_cmd2)])
            hhblits_sbatch_message = \
                f'Enter the following to distribute {putils.hhblits} jobs:\n\tsbatch {hhblits_sbatch}'
            info_messages.append(hhblits_sbatch_message)
        # else:
        #     hhblits_sbatch = None

        if bmdca_cmds:
            putils.make_path(self.profiles)
            putils.make_path(self.sbatch_scripts)
            # bmdca_cmds = \
            #     [list2cmdline([putils.bmdca_exe_path, '-i', os.path.join(self.profiles, '%s.fasta' % entity.name),
            #                   '-d', os.path.join(self.profiles, '%s_bmDCA' % entity.name)])
            #      for entity in all_entities.values()]
            bmdca_cmd_file = \
                utils.write_commands(bmdca_cmds, name=f'{utils.starttime}-bmDCA', out_path=self.profiles)
            bmdca_sbatch = utils.CommandDistributer.distribute(file=bmdca_cmd_file, out_path=self.sbatch_scripts,
                                                               scale='bmdca', max_jobs=len(bmdca_cmds),
                                                               number_of_commands=len(bmdca_cmds),
                                                               log_file=os.path.join(self.profiles,
                                                                                     'generate_couplings.log'))
            # reformat_msa_cmd_file = \
            #     SDUtils.write_commands(reformat_msa_cmds, name='%s-reformat_msa' % SDUtils.starttime,
            #                            out_path=self.profiles)
            # reformat_sbatch = distribute(file=reformat_msa_cmd_file, out_path=self.program_root,
            #                              scale='script', max_jobs=len(reformat_msa_cmds),
            #                              log_file=os.path.join(self.profiles, 'generate_profiles.log'),
            #                              number_of_commands=len(reformat_msa_cmds))
            print('\n' * 2)
            # Todo add bmdca_sbatch to hhblits_cmds finishing_commands kwarg
            bmdca_sbatch_message = \
                'Once you are satisfied, enter the following to distribute jobs:\n\tsbatch %s' \
                % bmdca_sbatch if not info_messages else 'ONCE this job is finished, to calculate evolutionary ' \
                                                         'couplings i,j for each amino acid in the multiple ' \
                                                         f'sequence alignment, enter:\n\tsbatch {bmdca_sbatch}'
            info_messages.append(bmdca_sbatch_message)
        # else:
        #     bmdca_sbatch = reformat_sbatch = None

        return info_messages


class JobResourcesFactory:
    """Return a JobResource instance by calling the Factory instance

    Handles creation and allotment to other processes by making a shared pointer to the JobResource for the current Job
    """
    def __init__(self, **kwargs):
        self._resources = {}
        self._warn = False

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
            if kwargs and not self._warn:
                # try:
                #     fragment_db.update(kwargs)
                # except RuntimeError:
                self._warn = True
                logger.warning(f"Can't pass the new arguments {', '.join(kwargs.keys())} to JobResources "
                               f'since it was already initialized and is a singleton')
                # raise RuntimeError(f'Can\'t pass the new arguments {", ".join(kwargs.keys())} to JobResources '
                #                    f'since it was already initialized')
            return job
        else:
            logger.info(f'Initializing {JobResources.__name__}')
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
