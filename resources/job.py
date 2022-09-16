from __future__ import annotations

import os
from typing import Annotated, AnyStr

from resources.structure_db import structure_database_factory
from utils.path import sym_entry, all_scores, projects, sequence_info, data, output_oligomers, output_fragments, \
    structure_background, scout, generate_fragments, number_of_trajectories, no_hbnet, \
    ignore_symmetric_clashes, ignore_pose_clashes, ignore_clashes, force_flags, no_evolution_constraint, \
    no_term_constraint, consensus
from flags import nstruct
from utils import start_log, make_path
from resources.EulerLookup import EulerLookup
from utils.SymEntry import SymEntry
from resources import fragment
from resources.wrapapi import api_database_factory

logger = start_log(name=__name__)


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
        self.data = os.path.join(self.program_root, data.title())
        self.projects = os.path.join(self.program_root, projects)
        self.job_paths = os.path.join(self.program_root, 'JobPaths')
        self.sbatch_scripts = os.path.join(self.program_root, 'Scripts')
        # TODO ScoreDatabase integration
        self.all_scores = os.path.join(self.program_root, all_scores)

        # data subdirectories
        self.clustered_poses = os.path.join(self.data, 'ClusteredPoses')
        self.pdbs = os.path.join(self.data, 'PDBs')  # Used to store downloaded PDB's
        self.sequence_info = os.path.join(self.data, sequence_info)
        self.external_db = os.path.join(self.data, 'ExternalDatabases')
        # pdbs subdirectories
        self.orient_dir = os.path.join(self.pdbs, 'oriented')
        self.orient_asu_dir = os.path.join(self.pdbs, 'oriented_asu')
        self.refine_dir = os.path.join(self.pdbs, 'refined')
        self.full_model_dir = os.path.join(self.pdbs, 'full_models')
        self.stride_dir = os.path.join(self.pdbs, 'stride')
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
        make_path(self.pdb_api)
        # make_path(self.pdb_entity_api)
        # make_path(self.pdb_assembly_api)
        make_path(self.uniprot_api)
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
        self.api_db = api_database_factory.get(source=self.data)
        self.structure_db = structure_database_factory.get(source=self.data)
        # self.symmetry_factory = symmetry_factory
        self.fragment_db: 'fragment.FragmentDatabase' | None = None
        self.euler_lookup: EulerLookup | None = None

        # Program flags
        self.consensus: bool = kwargs.get(consensus, False)  # Whether to run consensus
        self.design_selector: dict[str, dict[str, dict[str, set[int] | set[str]]]] | dict = \
            kwargs.get('design_selector', {})
        self.debug: bool = kwargs.get('debug', False)
        self.force_flags: bool = kwargs.get(force_flags, False)
        self.fuse_chains: list[tuple[str]] = [tuple(pair.split(':')) for pair in kwargs.get('fuse_chains', [])]
        self.ignore_clashes: bool = kwargs.get(ignore_clashes, False)
        if self.ignore_clashes:
            self.ignore_pose_clashes = self.ignore_symmetric_clashes = True
        else:
            self.ignore_pose_clashes: bool = kwargs.get(ignore_pose_clashes, False)
            self.ignore_symmetric_clashes: bool = kwargs.get(ignore_symmetric_clashes, False)
        self.increment_chains: bool = kwargs.get('increment_chains', False)
        self.mpi: int = kwargs.get('mpi', 0)
        self.no_evolution_constraint: bool = kwargs.get(no_evolution_constraint, False)
        self.no_hbnet: bool = kwargs.get(no_hbnet, False)
        self.no_term_constraint: bool = kwargs.get(no_term_constraint, False)
        self.number_of_trajectories: int = kwargs.get(number_of_trajectories, nstruct)
        self.overwrite: bool = kwargs.get('overwrite', False)
        self.output_directory: AnyStr | None = kwargs.get('output_directory', None)
        self.output_to_directory: bool = True if self.output_directory else False
        self.output_assembly: bool = kwargs.get('output_assembly', False)
        self.output_surrounding_uc: bool = kwargs.get('output_surrounding_uc', False)
        self.run_in_shell: bool = kwargs.get('run_in_shell', False)
        # self.pre_refine: bool = kwargs.get('pre_refine', True)
        # self.pre_loop_model: bool = kwargs.get('pre_loop_model', True)
        self.generate_fragments: bool = kwargs.get(generate_fragments, True)
        self.scout: bool = kwargs.get(scout, False)
        self.specific_protocol: str = kwargs.get('specific_protocol', False)
        self.structure_background: bool = kwargs.get(structure_background, False)
        self.sym_entry: SymEntry | None = kwargs.get(sym_entry, None)
        self.write_fragments: bool = kwargs.get(output_fragments, False)
        self.write_oligomers: bool = kwargs.get(output_oligomers, False)
        self.skip_logging: bool = kwargs.get('skip_logging', False)
        self.nanohedra_output: bool = kwargs.get('nanohedra_output', False)
        self.nanohedra_root: str | None = None
        # Development Flags
        self.command_only: bool = kwargs.get('command_only', False)  # Whether to reissue commands, only if run_in_shell=False
        self.development: bool = kwargs.get('development', False)

        if self.nanohedra_output:
            self.construct_pose: bool = kwargs.get('construct_pose', True)  # Whether to construct the PoseDirectory
            if not self.construct_pose:  # no construction specific flags
                self.write_fragments = False
                self.write_oligomers = False
        else:
            self.construct_pose = True

        if self.no_term_constraint:
            self.generate_fragments = False

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
