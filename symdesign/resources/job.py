from __future__ import annotations

# import collections
import inspect
import os
# import types
from dataclasses import make_dataclass, field
from typing import Annotated, AnyStr

from symdesign import resources
from structure.fragment import db
from symdesign import flags
from symdesign import utils

logger = utils.start_log(name=__name__)
# DesignFlags = collections.namedtuple('DesignFlags', design_args.keys(), defaults=design_args.values())
# DesignFlags = types.SimpleNamespace(**design_args)


def from_flags(cls, **kwargs):
    return cls(**{key: value for key, value in kwargs.items()
                  if key in inspect.signature(cls).parameters})


Design = make_dataclass('Design',
                        [(flag, eval(type(default).__name__), field(default=default))
                         for flag, default in flags.design.items()],
                        namespace={'from_flags': classmethod(from_flags)})
#                         frozen=True)


class JobResources:
    """The intention of JobResources is to serve as a singular source of design info which is common accross all
    designs. This includes common paths, databases, and design flags which should only be set once in program operation,
    then shared across all member designs"""
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
        self.data = os.path.join(self.program_root, utils.path.data.title())
        self.projects = os.path.join(self.program_root, utils.path.projects)
        self.job_paths = os.path.join(self.program_root, 'JobPaths')
        self.sbatch_scripts = os.path.join(self.program_root, 'Scripts')
        # TODO ScoreDatabase integration
        self.all_scores = os.path.join(self.program_root, utils.path.all_scores)

        # data subdirectories
        self.clustered_poses = os.path.join(self.data, 'ClusteredPoses')
        self.structure_info = os.path.join(self.data, utils.path.structure_info)
        self.pdbs = os.path.join(self.structure_info, 'PDBs')  # Used to store downloaded PDB's
        self.sequence_info = os.path.join(self.data, utils.path.sequence_info)
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
        utils.make_path(self.pdb_api)
        # utils.make_path(self.pdb_entity_api)
        # utils.make_path(self.pdb_assembly_api)
        utils.make_path(self.uniprot_api)
        # sequence database specific
        # self.make_path(self.sequence_info)
        # self.make_path(self.sequences)
        # self.make_path(self.profiles)
        # structure database specific
        # self.make_path(self.pdbs)
        # self.make_path(self.orient_dir)
        # self.make_path(self.orient_asu_dir)
        # self.make_path(self.refine_dir)
        # self.make_path(self.full_model_dir)
        # self.make_path(self.stride_dir)
        self.reduce_memory = False
        self.api_db = resources.wrapapi.api_database_factory.get(source=self.data)
        self.structure_db = resources.structure_db.structure_database_factory.get(source=self.data)
        # self.symmetry_factory = symmetry_factory
        self.fragment_db: 'db.FragmentDatabase' | None = None

        # Program flags
        # self.consensus: bool = kwargs.get(consensus, False)  # Whether to run consensus
        self.design_selector: dict[str, dict[str, dict[str, set[int] | set[str]]]] | dict = \
            kwargs.get('design_selector', {})
        self.debug: bool = kwargs.get('debug', False)
        self.dock_only: bool = kwargs.get('dock_only', False)
        self.log_level: bool = kwargs.get('log_level', flags.default_logging_level)
        self.force_flags: bool = kwargs.get(utils.path.force_flags, False)
        self.fuse_chains: list[tuple[str]] = [tuple(pair.split(':')) for pair in kwargs.get('fuse_chains', [])]
        # self.design = DesignFlags(*[kwargs.get(argument_name) for argument_name in design_args.keys()])
        # self.design = types.SimpleNamespace(**{flag: kwargs.get(flag, default) for flag, default in flags.design})
        self.design = Design.from_flags(**kwargs)
        # self.design = types.SimpleNamespace(**{flag: kwargs.get(flag, default) for flag, default in flags.design})
        # self.design.ignore_clashes: bool = kwargs.get(ignore_clashes, False)
        # self.ignore_clashes: bool = kwargs.get(ignore_clashes, False)
        if self.design.ignore_clashes:
            self.design.ignore_pose_clashes = self.design.ignore_symmetric_clashes = True
        if self.dock_only:
            self.design.sequences = self.design.structures = False
        # else:
        #     self.ignore_pose_clashes: bool = kwargs.get(ignore_pose_clashes, False)
        #     self.ignore_symmetric_clashes: bool = kwargs.get(ignore_symmetric_clashes, False)
        self.increment_chains: bool = kwargs.get('increment_chains', False)
        self.mpi: int = kwargs.get('mpi', 0)
        # self.evolution_constraint: bool = kwargs.get(evolution_constraint, False)
        # self.hbnet: bool = kwargs.get(hbnet, False)
        # self.term_constraint: bool = kwargs.get(term_constraint, False)
        # self.number_of_trajectories: int = kwargs.get(number_of_trajectories, flags.nstruct)
        self.overwrite: bool = kwargs.get('overwrite', False)
        self.output_directory: AnyStr | None = kwargs.get(utils.path.output_directory, None)
        self.output_to_directory: bool = True if self.output_directory else False
        self.output_assembly: bool = kwargs.get(utils.path.output_assembly, False)
        self.output_surrounding_uc: bool = kwargs.get(utils.path.output_surrounding_uc, False)
        self.distribute_work: bool = kwargs.get(utils.path.distribute_work, False)
        if self.mpi > 0:
            self.distribute_work = True
        # self.pre_refine: bool = kwargs.get('pre_refine', True)
        # self.pre_loop_model: bool = kwargs.get('pre_loop_model', True)
        self.generate_fragments: bool = kwargs.get(utils.path.generate_fragments, True)
        # self.scout: bool = kwargs.get(scout, False)
        self.specific_protocol: str = kwargs.get('specific_protocol', False)
        # self.structure_background: bool = kwargs.get(structure_background, False)
        self.sym_entry: utils.SymEntry.SymEntry | None = kwargs.get(utils.path.sym_entry, None)
        self.write_fragments: bool = kwargs.get(utils.path.output_fragments, False)
        self.write_oligomers: bool = kwargs.get(utils.path.output_oligomers, False)
        self.write_structures: bool = kwargs.get(utils.path.output_structures, True)
        self.write_trajectory: bool = kwargs.get(utils.path.output_trajectory, False)
        self.skip_logging: bool = kwargs.get(utils.path.skip_logging, False)
        self.nanohedra_output: bool = kwargs.get('nanohedra_output', False)
        self.nanohedra_root: str | None = None
        # Development Flags
        # Whether to reissue commands, only if distribute_work=False
        self.command_only: bool = kwargs.get('command_only', False)
        self.development: bool = kwargs.get(utils.path.development, False)

        if self.nanohedra_output:
            self.construct_pose: bool = kwargs.get('construct_pose', True)  # Whether to construct the PoseDirectory
            if not self.construct_pose:  # no construction specific flags
                self.write_fragments = False
                self.write_oligomers = False
        else:
            self.construct_pose = True

        # Handle protocol specific flags
        if not self.design.term_constraint:
            self.generate_fragments = False

        if self.design.structure_background:
            self.design.evolution_constraint = False
            self.design.hbnet = False
            self.design.term_constraint = False

    # @staticmethod
    # def make_path(path: AnyStr, condition: bool = True):
    #     """Make all required directories in specified path if it doesn't exist, and optional condition is True
    #
    #     Args:
    #         path: The path to create
    #         condition: A condition to check before the path production is executed
    #     """
    #     if condition:
    #         os.makedirs(path, exist_ok=True)


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
