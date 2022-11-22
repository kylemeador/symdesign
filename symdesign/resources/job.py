from __future__ import annotations

import logging
import inspect
import os
from dataclasses import make_dataclass, field
from typing import Annotated, AnyStr

import psutil

from symdesign.resources import structure_db, wrapapi
from symdesign.structure.fragment import db
from symdesign import flags
from symdesign.utils import calculate_mp_cores, CommandDistributer, SymEntry, path as putils

logger = logging.getLogger(__name__)
# DesignFlags = collections.namedtuple('DesignFlags', design_args.keys(), defaults=design_args.values())
# DesignFlags = types.SimpleNamespace(**design_args)


def from_flags(cls, **kwargs):
    return cls(**{key: value for key, value in kwargs.items()
                  if key in inspect.signature(cls).parameters})


# These dataclasses help simplify the use of flags into namespaces
# The commented out versions below had poor implementations
#  self.design = DesignFlags(*[kwargs.get(argument_name) for argument_name in design_args.keys()])
#  self.design = types.SimpleNamespace(**{flag: kwargs.get(flag, default) for flag, default in flags.design})
#  self.design = types.SimpleNamespace(**{flag: kwargs.get(flag, default) for flag, default in flags.design})
Design = make_dataclass('Design',
                        [(flag, eval(type(default).__name__), field(default=default))
                         for flag, default in flags.design.items()],
                        namespace={'from_flags': classmethod(from_flags)})
#                         frozen=True)
Dock = make_dataclass('Dock',
                      [(flag, eval(type(default).__name__), field(default=default))
                       for flag, default in flags.dock.items()],
                      namespace={'from_flags': classmethod(from_flags)})
#                       frozen=True)


class JobResources:
    """The intention of JobResources is to serve as a singular source of design info which is common accross all
    designs. This includes common paths, databases, and design flags which should only be set once in program operation,
    then shared across all member designs"""
    reduce_memory: bool = False

    def __init__(self, program_root: AnyStr = None, **kwargs):
        """For common resources for all SymDesign outputs, ensure paths to these resources are available attributes"""
        try:
            if os.path.exists(program_root):
                self.program_root = program_root
            else:
                raise FileNotFoundError(f"Path doesn't exist!\n\t{program_root}")
        except TypeError:
            raise TypeError(f"Can't initialize {JobResources.__name__} without parameter 'program_root'")

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
        # self.reduce_memory = False
        self.api_db = wrapapi.api_database_factory.get(source=self.data)
        self.structure_db = structure_db.structure_database_factory.get(source=self.data)
        self.fragment_db: 'db.FragmentDatabase' | None = None

        # Computing environment and development Flags
        self.cores: int = kwargs.get('cores', 0)
        self.distribute_work: bool = kwargs.get(putils.distribute_work)
        self.mpi: int = kwargs.get('mpi', 0)
        self.multi_processing: int = kwargs.get(putils.multi_processing)
        if self.multi_processing:
            # Calculate the number of cores to use depending on computer resources
            self.cores = calculate_mp_cores(cores=self.cores)  # mpi=self.mpi, Todo
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
        self.design_selector: dict[str, dict[str, dict[str, set[int] | set[str]]]] | dict = \
            kwargs.get('design_selector', {})
        self.debug: bool = kwargs.get('debug', False)
        self.dock = Dock.from_flags(**kwargs)
        if self.dock.perturb_dof:
            self.dock.perturb_dof_rot = self.dock.perturb_dof_tx = True
            self.dock.perturb_dof_steps_rot = self.dock.perturb_dof_steps_tx = self.dock.perturb_dof_steps
        else:  # Set the unavailable dof to 1 step
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
        self.force: bool = kwargs.get(putils.force, False)
        self.fuse_chains: list[tuple[str]] = [tuple(pair.split(':')) for pair in kwargs.get('fuse_chains', [])]
        self.design = Design.from_flags(**kwargs)
        # self.ignore_clashes: bool = kwargs.get(ignore_clashes, False)
        if self.design.ignore_clashes:
            self.design.ignore_pose_clashes = self.design.ignore_symmetric_clashes = True
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
        self.generate_fragments: bool = kwargs.get(putils.generate_fragments, True)
        self.interface_to_alanine: bool = kwargs.get('interface_to_alanine')
        self.gather_metrics: bool = kwargs.get('gather_metrics')
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

        self.overwrite: bool = kwargs.get('overwrite')
        self.output_directory: AnyStr | None = kwargs.get(putils.output_directory)
        self.output_to_directory: bool = True if self.output_directory else False
        self.output_assembly: bool = kwargs.get(putils.output_assembly)
        self.output_surrounding_uc: bool = kwargs.get(putils.output_surrounding_uc)
        self.write_fragments: bool = kwargs.get(putils.output_fragments)
        self.write_oligomers: bool = kwargs.get(putils.output_oligomers)
        self.write_structures: bool = kwargs.get(putils.output_structures)
        self.write_trajectory: bool = kwargs.get(putils.output_trajectory)
        self.skip_logging: bool = kwargs.get(putils.skip_logging)
        self.merge: bool = kwargs.get('merge', False)
        self.save: bool = kwargs.get('save', False)
        self.figures: bool = kwargs.get('figures', False)
        self.skip_sequence_generation: bool = kwargs.get('skip_sequence_generation', False)

        if self.write_structures or self.output_assembly or self.output_surrounding_uc or self.write_fragments \
                or self.write_oligomers or self.write_trajectory:
            self.output: bool = True
        else:
            self.output: bool = False

        self.nanohedra_output: bool = kwargs.get('nanohedra_output', False)
        self.nanohedra_root: str | None = None
        if self.nanohedra_output:
            self.construct_pose: bool = kwargs.get('construct_pose', True)  # Whether to construct the PoseDirectory
        else:
            self.construct_pose = True

        # Handle protocol specific flags
        if not self.design.term_constraint:
            self.generate_fragments = False

        if self.design.structure_background:
            self.design.evolution_constraint = False
            self.design.hbnet = False
            self.design.scout = False
            self.design.term_constraint = False

    @property
    def construct_pose(self):
        return self._construct_pose

    @construct_pose.setter
    def construct_pose(self, value: bool):
        self._construct_pose = value
        if self._construct_pose:
            pass
        else:  # No construction specific flags
            self.write_fragments = False
            self.write_oligomers = False

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
