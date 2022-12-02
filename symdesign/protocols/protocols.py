from __future__ import annotations

import functools
import json
import logging
import os
import pickle
import re
import shutil
from collections.abc import Sequence
from copy import copy
from glob import glob
from itertools import combinations, repeat
from logging import Logger
from pathlib import Path
from subprocess import Popen, list2cmdline
from typing import Callable, Any, Iterable, AnyStr, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import sklearn as skl
from cycler import cycler
# from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
# from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist

from symdesign import flags
from symdesign.metrics import read_scores, interface_composition_similarity, unnecessary, necessary_metrics, \
    rosetta_terms, columns_to_new_column, division_pairs, delta_pairs, dirty_hbond_processing, significance_columns, \
    df_permutation_test, clean_up_intermediate_columns, protocol_specific_columns, rank_dataframe_by_metric_weights, \
    filter_df_for_index_by_value, multiple_sequence_alignment_dependent_metrics, profile_dependent_metrics, \
    process_residue_info, collapse_significance_threshold, calculate_collapse_metrics, errat_1_sigma, errat_2_sigma, \
    calculate_residue_surface_area, position_specific_divergence, calculate_sequence_observations_and_divergence, \
    incorporate_mutation_info, residue_classification, sum_per_residue_metrics
from symdesign.resources.job import job_resources_factory
from symdesign.structure.base import Structure
from symdesign.structure.fragment.db import FragmentDatabase, fragment_info_type
from symdesign.structure.model import Pose, MultiModel, Models, Model, Entity  # Todo no import transformation_mapping ?
from symdesign.structure.sequence import generate_mutations_from_reference, sequence_difference, \
    MultipleSequenceAlignment, pssm_as_array, concatenate_profile, write_pssm_file, read_fasta_file, write_sequences
from symdesign.structure.utils import protein_letters_3to1, protein_letters_1to3, DesignError, ClashError, SymmetryError
from symdesign.utils import large_color_array, starttime, start_log, unpickle, pickle_object, index_intersection, \
    write_shell_script, all_vs_all, sym, condensed_to_square, rosetta, path as putils
from symdesign.utils.SymEntry import SymEntry, symmetry_factory
from symdesign.utils.nanohedra.general import get_components_from_nanohedra_docking

# Globals
logger = logging.getLogger(__name__)
# pose_logger = start_log(name='pose', handler_level=3, propagate=True)
zero_offset = 1
idx_slice = pd.IndexSlice
# design_directory_modes = [putils.interface_design, 'dock', 'filter']
cst_value = round(0.2 * rosetta.reference_average_residue_weight, 2)
mean, std = 'mean', 'std'
stats_metrics = [mean, std]
variance = 0.8
symmetry_protocols = {0: 'make_point_group', 2: 'make_layer', 3: 'make_lattice'}  # -1: 'asymmetric',
null_cmd = ['echo']
warn_missing_symmetry = \
    f'Cannot %s without providing symmetry! Provide symmetry with "--symmetry" or "--{putils.sym_entry}"'


def handle_design_errors(errors: tuple[Type[Exception], ...] = (Exception,)) -> Callable:
    """Wrap a function/method with try: except errors: and log exceptions to the functions first argument .log attribute

    This argument is typically self and is in a class with .log attribute

    Args:
        errors: A tuple of exceptions to monitor. Must be a tuple even if single exception
    Returns:
        Function return upon proper execution, else is error if exception raised, else None
    """
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs) -> Any:
            try:
                return func(self, *args, **kwargs)
            except errors as error:
                self.log.error(error)  # Allows exception reporting using self.log
                # self.info['error'] = error  # Todo? include the error code in the design state
                return error
        return wrapped
    return wrapper


def close_logs(func: Callable):
    """Wrap a function/method to close the functions first arguments .log attribute FileHandlers after use"""
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        func_return = func(self, *args, **kwargs)
        # adapted from https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile
        for handler in self.log.handlers:
            handler.close()
        return func_return
    return wrapped


class PoseProtocol:
    pass  # Todo. Put in methods that are useful on a pose, not necessarily attributes, not path resources


class PoseDirectory:
    _design_selector: dict[str, dict[str, dict[str, set[int] | set[str]]]] | dict
    _designed_sequences: list[Sequence]
    _entity_names: list[str]
    _fragment_observations: list[fragment_info_type]
    _pose_transformation: list  # list[transformation_mapping]):  # Todo why won't this import
    _symmetry_definition_files: list[AnyStr]
    directives: list[dict[int, str]]
    entities: list[Entity]
    fragment_db: FragmentDatabase
    fragment_observations: list[dict] | None
    frag_file: str | Path
    initial_model: Model | None
    initialized: bool
    name: str
    pose: Pose | None
    # pose_id: str
    pose_file: str | Path
    pre_refine: bool
    pre_loop_model: bool
    source: str | None
    source_path: str
    specific_designs: list
    specific_designs_file_paths: list[AnyStr]

    @classmethod
    def from_file(cls, design_path: str, root: AnyStr = None, **kwargs):
        return cls(design_path, root=root, **kwargs)

    @classmethod
    def from_pose_id(cls, design_path: str, root: AnyStr = None, **kwargs):
        return cls(design_path, pose_id=True, root=root, **kwargs)

    def directory_string_to_path(self, root: AnyStr, pose_id: str):
        """Set self.path to the root/poseID where the poseID is converted from dash "-" separation to path separators"""
        if root is None:
            raise ValueError("No 'root' argument was passed. Can't use a pose_id without a root directory")

        if self.job.nanohedra_output:
            self.path = os.path.join(root, pose_id.replace('-', os.sep))
        else:
            # Dev only
            if '_Designs-' in pose_id:
                self.path = os.path.join(root, putils.projects, pose_id.replace('_Designs-', f'_Designs{os.sep}'))
            else:
                self.path = os.path.join(root, putils.projects, pose_id.replace(f'_{putils.pose_directory}-',
                                                                                f'_{putils.pose_directory}{os.sep}'))

    def __init__(self, design_path: AnyStr, pose_id: bool = False, root: AnyStr = None, **kwargs):
        # self.job = job if job else job_resources_factory.get(program_root=root, **kwargs)
        self.job = job_resources_factory.get()
        # PoseDirectory flags
        self.log: Logger | None = None
        if pose_id:
            # self.pose_id = pose_id
            self.directory_string_to_path(root, design_path)  # sets self.path
            self.source_path = self.path
        else:
            self.source_path = os.path.abspath(design_path)

        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f'The specified Pose source "{self.source_path}" was not found!')

        # Symmetry attributes
        # self.cryst_record = None
        # self.expand_matrices = None
        # self.pose_transformation = None
        # Todo monitor if Rosetta energy mechanisms are modified for crystal set ups and adjust parameter accordingly
        # If a new sym_entry is provided it wouldn't be saved to the state
        if self.job.sym_entry is not None:
            self.sym_entry = self.job.sym_entry
        self.sym_def_file: str | None = None  # The symmetry definition file for the entire Pose
        self.symmetry_protocol: str | None = None
        self.protocol: str | None = None
        """The name of the currently utilized protocol for file naming and metric results"""

        # Design attributes
        # self.background_profile: str = kwargs.get('background_profile', putils.design_profile)
        # """The type of position specific profile (per-residue amino acid frequencies) to utilize as the design
        # background profile.
        # Choices include putils.design_profile, putils.evolutionary_profile, and putils.fragment_profile
        # """
        self.directives = kwargs.get('directives', [])
        self.info: dict = {}
        """Internal state info"""
        self._info: dict = {}
        """Internal state info at load time"""
        # self.interface_design_residue_numbers: set[int] | bool = False  # The residue numbers in the pose interface
        # self.interface_residue_ids: dict[str, str] = {}
        # {'interface1': '23A,45A,46A,...' , 'interface2': '234B,236B,239B,...'}
        # self.interface_residue_numbers: set[int] | bool = False  # The interface residues which are surface accessable
        # self.oligomer_names: list[str] = self.info.get('oligomer_names', [])
        self.entities = []
        self.entity_names = kwargs.get('entity_names', [])
        self.pose = None
        """Contains the design's Pose object"""
        # self.pose_id = None
        # self.pre_refine = self.info.get('pre_refine', True)
        # self.pre_loop_model = self.info.get('pre_loop_model', True)
        self.specific_designs = kwargs.get('specific_designs', [])
        self.pose_transformation = kwargs.get('pose_transformation', [])
        self.specific_designs_file_paths = []

        # Todo if I use output_identifier for design, it opens up a can of worms.
        #  Maybe it is better to include only for specific modules
        # Set name initially to the basename. This may change later, but we need to check for serialized info
        self.name = os.path.splitext(os.path.basename(self.source_path))[0]
        output_identifier = f'{self.name}_' if self.job.output_to_directory else ''

        self.serialized_info = os.path.join(self.source_path, f'{output_identifier}{putils.data}', putils.state_file)
        self.initialized = True if os.path.exists(self.serialized_info) else False
        if self.initialized:
            self.source = None  # Will be set to self.asu_path later
            if self.job.output_to_directory:
                self.projects = ''
                self.project_designs = ''
                self.path = self.job.program_root  # /output_directory<- self.path /design.pdb
            else:
                self.path = self.source_path
                self.project_designs = os.path.dirname(self.path)
                self.projects = os.path.dirname(self.project_designs)
        else:
            # Save job variables to the state during initialization
            if self.sym_entry:
                self.info['sym_entry_specification'] = self.sym_entry.entry_number, self.sym_entry.sym_map
            if self.job.design_selector:
                self.design_selector = self.job.design_selector
            else:
                self.design_selector = {}

            path_components = os.path.splitext(self.source_path)[0].split(os.sep)
            if self.job.nanohedra_output:
                # path_components = self.source_path.split(os.sep)
                # design_symmetry (P432)
                # path_components[-4] are the oligomeric names
                self.name = '-'.join(path_components[-4:])
                # self.name = self.pose_id.replace('_DEGEN_', '-DEGEN_').replace('_ROT_', '-ROT_').replace('_TX_', '-tx_')
                # design_symmetry/building_blocks (P432/4ftd_5tch)
                # path/to/[design_symmetry]/building_blocks/degen/rot/tx
                root = path_components[-5] if root is None else root
                self.source = os.path.join(self.source_path, putils.asu)
            else:  # Set up PoseDirectory initially from input file
                # path_components = path.splitext(self.source_path)[0].split(os.sep)
                try:
                    index = path_components.index(os.environ['USER'])
                except (KeyError, ValueError):  # Missing USER enviromental variable, missing in path_components
                    index = None
                self.name = '-'.join(path_components[index:])
                root = path_components[-2] if root is None else root  # path/to/job/[project]/design.pdb
                self.source = self.source_path

            # Remove a leading '-' character from abspath type results
            self.name = self.name.lstrip('-')
            if self.job.output_to_directory:
                self.projects = ''
                self.project_designs = ''
                self.path = self.job.program_root  # /output_directory<- self.path /design.pdb
            else:
                self.projects = os.path.join(self.job.program_root, putils.projects)
                self.project_designs = os.path.join(self.projects, f'{root}_{putils.pose_directory}')
                self.path = os.path.join(self.project_designs, self.name)
                # ^ /program_root/projects/project/design<- self.path /design.pdb

                # # copy the source file to the PoseDirectory for record keeping...
                # # Not using now that pose_format can be disregarded...
                # shutil.copy(self.source_path, self.path)

            putils.make_path(self.path, condition=self.job.construct_pose)

        # PoseDirectory path attributes. Set after finding correct path
        self.log_path: str | Path = os.path.join(self.path, f'{self.name}.log')
        self.designs: str | Path = os.path.join(self.path, putils.designs)
        # /root/Projects/project_Poses/design/designs
        self.scripts: str | Path = os.path.join(self.path, f'{output_identifier}{putils.scripts}')
        # /root/Projects/project_Poses/design/scripts
        self.frags: str | Path = os.path.join(self.path, f'{output_identifier}{putils.frag_dir}')
        # /root/Projects/project_Poses/design/matching_fragments
        self.flags: str | Path = os.path.join(self.scripts, 'flags')
        # /root/Projects/project_Poses/design/scripts/flags
        self.data: str | Path = os.path.join(self.path, f'{output_identifier}{putils.data}')
        # /root/Projects/project_Poses/design/data
        self.scores_file: str | Path = os.path.join(self.data, f'{self.name}.sc')
        # /root/Projects/project_Poses/design/data/name.sc
        self.serialized_info: str | Path = os.path.join(self.data, putils.state_file)
        # /root/Projects/project_Poses/design/data/info.pkl
        self.asu_path: str | Path = os.path.join(self.path, f'{self.name}_{putils.asu}')
        # /root/Projects/project_Poses/design/design_name_clean_asu.pdb
        self.assembly_path: str | Path = os.path.join(self.path, f'{self.name}_{putils.assembly}')
        # /root/Projects/project_Poses/design/design_name_assembly.pdb
        self.refine_pdb: str | Path = os.path.join(self.data, os.path.basename(self.asu_path))
        # self.refine_pdb: str | Path = f'{os.path.splitext(self.asu_path)[0]}_refine.pdb'
        # /root/Projects/project_Poses/design/clean_asu_for_refine.pdb
        self.consensus_pdb: str | Path = f'{os.path.splitext(self.asu_path)[0]}_for_consensus.pdb'
        # /root/Projects/project_Poses/design/design_name_for_consensus.pdb
        self.consensus_design_pdb: str | Path = os.path.join(self.designs, os.path.basename(self.consensus_pdb))
        # /root/Projects/project_Poses/design/designs/design_name_for_consensus.pdb
        self.pdb_list: str | Path = os.path.join(self.scripts, 'design_files.txt')
        # /root/Projects/project_Poses/design/scripts/design_files.txt
        self.design_profile_file: str | Path = os.path.join(self.data, 'design.pssm')
        # /root/Projects/project_Poses/design/data/design.pssm
        self.evolutionary_profile_file: str | Path = os.path.join(self.data, 'evolutionary.pssm')
        # /root/Projects/project_Poses/design/data/evolutionary.pssm
        self.fragment_profile_file: str | Path = os.path.join(self.data, 'fragment.pssm')
        # /root/Projects/project_Poses/design/data/fragment.pssm
        self.refined_pdb: str | Path | None = None  # /root/Projects/project_Poses/design/design_name_refined.pdb
        self.scouted_pdb: str | Path | None = None  # /root/Projects/project_Poses/design/design_name_scouted.pdb
        # These files may be present from Nanohedra outputs
        self.pose_file = os.path.join(self.source_path, putils.pose_file)
        self.frag_file = os.path.join(self.source_path, putils.frag_dir, putils.frag_text_file)
        # These files are used as output from analysis protocols
        self.trajectories = os.path.join(self.job.all_scores, f'{self}_Trajectories.csv')
        self.residues = os.path.join(self.job.all_scores, f'{self}_Residues.csv')
        # self.designed_sequences_file = os.path.join(self.job.all_scores, f'{self}_Sequences.pkl')
        self.designed_sequences_file = os.path.join(self.designs, f'{self}_sequences.fasta')

        self.initial_model = None
        """Used if the pose structure has never been initialized previously"""
        if not self.initialized and not self.entity_names:
            # None were provided at start up, find them
            # Starts self.log if not self.job.nanohedra_output
            self.find_entity_names()  # Sets self.entity_names
        # else:
        #     # input(f'Stopped here with: self.initialized({self.initialized}) self.entity_names({self.entity_names})'
        #     #       f'bool? {not self.initialized and not self.entity_names}')
        #     self.entity_names = self.info.get('entity_names', [])  # Set so that DataBase set up works

        # Configure standard pose loading mechanism with self.source
        if self.specific_designs:
            # Introduce flag handling current inability of specific_designs to handle iteration
            self._lock_optimize_designs = True
            self.specific_designs_file_paths = []
            for design in self.specific_designs:
                matching_path = os.path.join(self.designs, f'*{design}.pdb')
                matching_designs = sorted(glob(matching_path))
                if matching_designs:
                    for matching_design in matching_designs:
                        if os.path.exists(matching_design):
                            # self.specific_design_path = matching_design
                            self.specific_designs_file_paths.append(matching_design)
                    if len(matching_designs) > 1:
                        self.log.warning(f'Found {len(matching_designs)} matching designs to your specified design '
                                         f'using {matching_path}. Choosing the first {matching_designs[0]}')
                else:
                    raise DesignError(f"Couldn't locate a specific_design matching the name '{matching_path}'")
                # format specific_designs to a pose ID compatible format
            self.specific_designs = [f'{self.name}_{design}' for design in self.specific_designs]
            # self.source = specific_designs_file_paths  # Todo?
            # self.source = self.specific_design_path
        elif self.source is None:
            if os.path.exists(self.asu_path):  # Standard mechanism of loading the pose
                self.source = self.asu_path
            else:
                try:
                    glob_target = os.path.join(self.path, f'{self.name}*.pdb')
                    source = sorted(glob(glob_target))
                    if len(source) == 1:
                        self.source = source[0]
                    else:
                        raise ValueError(f'Found {len(source)} files matching the path "{glob_target}". '
                                         f'Only 1 expected')
                except IndexError:  # glob found no files
                    self.source = None
        else:  # If the PoseDirectory was loaded as .pdb/mmCIF, the source should be loaded already
            # self.source = self.initial_model
            pass

    @property
    def designed_sequences(self) -> list[Sequence]:
        """Return the designed sequences for the entire Pose associated with the PoseDirectory"""
        try:
            return self._designed_sequences
        except AttributeError:
            # Todo
            #  self._designed_sequences = {seq.id: seq.seq for seq in read_fasta_file(self.designed_sequences_file)}
            self._designed_sequences = [seq_record.seq for seq_record in read_fasta_file(self.designed_sequences_file)]
            return self._designed_sequences

    # Decorator static methods: These must be declared above their usage, but made static after each declaration
    def remove_structure_memory(func):
        """Decorator to remove large memory attributes from the instance after processing is complete"""
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            func_return = func(self, *args, **kwargs)
            if self.job.reduce_memory:
                self.pose = None
                self.entities.clear()
            return func_return
        return wrapped
    #
    # def handle_design_errors(errors: tuple = (Exception,)) -> Callable:
    #     """Decorator to wrap a method with try: ... except errors: and log errors to the PoseDirectory
    #
    #     Args:
    #         errors: A tuple of exceptions to monitor. Must be a tuple even if single exception
    #     Returns:
    #         Function return upon proper execution, else is error if exception raised, else None
    #     """
    #     def wrapper(func: Callable) -> Any:
    #         @functools.wraps(func)
    #         def wrapped(self, *args, **kwargs):
    #             try:
    #                 return func(self, *args, **kwargs)
    #             except errors as error:
    #                 self.log.error(error)  # Allows exception reporting using self.log
    #                 # self.info['error'] = error
    #                 return error
    #         return wrapped
    #     return wrapper
    #
    # def close_logs(func):
    #     """Decorator to close the instance log file after use in an instance method (protocol)"""
    #     @functools.wraps(func)
    #     def wrapped(self, *args, **kwargs):
    #         func_return = func(self, *args, **kwargs)
    #         # adapted from https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile
    #         for handler in self.log.handlers:
    #             handler.close()
    #         return func_return
    #     return wrapped

    # SymEntry object attributes
    @property
    def sym_entry(self) -> SymEntry | None:
        """The SymEntry"""
        try:
            return self._sym_entry
        except AttributeError:
            self._sym_entry = symmetry_factory.get(*self.info['sym_entry_specification']) \
                if 'sym_entry_specification' in self.info else None
            # temp_sym_entry = SymEntry(self.info['sym_entry_specification'][0])
            # self._sym_entry = symmetry_factory(self.info['sym_entry_specification'][0],
            #                                    [temp_sym_entry.resulting_symmetry] +
            #                                    list(self.info['sym_entry_specification'][1].values())) \
            #     if 'sym_entry_specification' in self.info else None
            # self.info['sym_entry_specification'] = \
            #     (self.info['sym_entry_specification'][0], [temp_sym_entry.resulting_symmetry] +
            #      list(self.info['sym_entry_specification'][1].values()))
            return self._sym_entry

    @sym_entry.setter
    def sym_entry(self, sym_entry: SymEntry):
        self._sym_entry = sym_entry

    @property
    def symmetric(self) -> bool:
        """Is the PoseDirectory symmetric?"""
        return self.sym_entry is not None

    @property
    def design_symmetry(self) -> str | None:
        """The result of the SymEntry"""
        try:
            return self.sym_entry.resulting_symmetry
        except AttributeError:
            return None

    # @property
    # def sym_entry_number(self) -> int | None:
    #     """The entry number of the SymEntry"""
    #     try:
    #         return self.sym_entry.entry_number
    #     except AttributeError:
    #         return None

    # @property
    # def sym_entry_map(self) -> list[str] | None:
    #     """The symmetry map of the SymEntry"""
    #     try:
    #         # return [self.sym_entry.resulting_symmetry] + list(self.sym_entry.sym_map.values())
    #         return self.sym_entry.sym_map
    #     except AttributeError:
    #         return None

    # @property
    # def sym_entry_combination(self) -> str | None:
    #     """The combination string of the SymEntry"""
    #     try:
    #         return self.sym_entry.combination_string
    #     except AttributeError:
    #         return None

    @property
    def design_dimension(self) -> int | None:
        """The dimension of the SymEntry"""
        try:
            return self.sym_entry.dimension
        except AttributeError:
            return None

    # @property
    # def number_of_symmetry_mates(self) -> int | None:
    #     """The number of symmetric copies in the full symmetric system"""
    #     try:
    #         return self.sym_entry.number_of_operations
    #     except AttributeError:
    #         return None

    # @property
    # def trajectory_metrics_file(self) -> AnyStr:
    #     return os.path.join(self.job.all_scores, f'{self}_Trajectories.csv')
    #
    # @property
    # def residue_metrics_file(self) -> AnyStr:
    #     return os.path.join(self.job.all_scores, f'{self}_Residues.csv')
    #

    @property
    def number_of_fragments(self) -> int:
        return len(self.fragment_observations) if self.fragment_observations else 0

    @property
    def pose_kwargs(self) -> dict[str, Any]:
        """Returns the kwargs necessary to initialize the Pose"""
        return dict(sym_entry=self.sym_entry, log=self.log, design_selector=self.design_selector,
                    entity_names=self.entity_names, transformations=self.pose_transformation,
                    # pass names ^ if available
                    ignore_clashes=self.job.design.ignore_pose_clashes, fragment_db=self.job.fragment_db)
        #             api_db=self.job.api_db,

    # def clear_pose_transformation(self):
    #     """Remove any pose transformation data from the Pose"""
    #     try:
    #         del self._pose_transformation
    #         self.info.pop('pose_transformation')
    #     except AttributeError:
    #         pass

    @property
    def entity_names(self) -> list[str]:
        """Provide the names of all Entity instances in the PoseDirectory"""
        try:
            return self._entity_names
        except AttributeError:
            # Get the names from the pose state
            self._entity_names = self.info.get('entity_names', [])
            return self._entity_names

    @entity_names.setter
    def entity_names(self, names: list):
        if isinstance(names, list):
            self._entity_names = self.info['entity_names'] = names
        else:
            raise ValueError(f'The attribute entity_names must be a list, not {type(names)}')

    @property
    def pose_transformation(self) -> list[dict[str, np.ndarray]]:
        """Provide the transformation parameters for the design in question

        Returns:
            [{'rotation': np.ndarray, 'translation': np.ndarray, 'rotation2': np.ndarray,
              'translation2': np.ndarray}, ...]
            A list with the transformations of each Entity in the Pose according to the symmetry
        """
        # self.log.critical('PoseDirectory.pose_transformation was accessed!')
        try:
            return self._pose_transformation
        except AttributeError:
            # Get the transformation from the pose state
            self._pose_transformation = self.info.get('pose_transformation', [])
            return self._pose_transformation

    @pose_transformation.setter
    def pose_transformation(self, transform: list):  # list[transformation_mapping]):  # Todo why won't this import
        if isinstance(transform, list):
            self._pose_transformation = self.info['pose_transformation'] = transform
        else:
            raise ValueError(f'The attribute pose_transformation must be a list, not {type(transform)}')

    @property
    def design_selector(self) -> dict[str, dict[str, dict[str, set[int] | set[str]]]] | dict:
        """Provide the design_selector parameters for the design in question

        Returns:
            {}, A mapping of the selection criteria in the Pose according to set up
        """
        try:
            return self._design_selector
        except AttributeError:
            # Get the design_selector from the pose state
            self._design_selector = self.info.get('design_selector', {})
            return self._design_selector

    @design_selector.setter
    def design_selector(self, design_selector: dict):
        if isinstance(design_selector, dict):
            self._design_selector = self.info['design_selector'] = design_selector
        else:
            raise ValueError(f'The attribute design_selector must be a dict, not {type(design_selector)}')

    @property
    def fragment_observations(self) -> list | None:
        """Provide the observed fragments as measured from the Pose

        Returns:
            The fragment instances observed from the pose
                Ex: [{'cluster': (1, 2, 24), 'mapped': 78, 'paired': 287, 'match':0.46843}, ...]
        """
        try:
            return self._fragment_observations
        except AttributeError:
            # Get the fragment_observations from the pose state
            self._fragment_observations = self.info.get('fragments', None)  # None signifies query wasn't attempted
            return self._fragment_observations

    @fragment_observations.setter
    def fragment_observations(self, fragment_observations: list):
        if isinstance(fragment_observations, list):
            self._fragment_observations = self.info['fragments'] = fragment_observations
        else:
            raise ValueError(f'The attribute fragment_observations must be a list, not {type(fragment_observations)}')

    @close_logs
    def find_entity_names(self):
        """Load the Structure source_path and extract the entity_names from the Structure"""
        if self.job.nanohedra_output:
            entity_names = get_components_from_nanohedra_docking(self.pose_file)
        else:
            self.start_log()
            self.initial_model = Model.from_file(self.source_path, log=self.log)
            entity_names = [entity.name for entity in self.initial_model.entities]

        self.entity_names = entity_names

    def start_log(self, level: int = 2):
        """Initialize the logger for the Pose"""
        if self.log:
            return

        if self.job.debug:
            handler, level = 1, 1  # Defaults to stdout, debug is level 1
            no_log_name = False
        else:
            handler = 2  # To a file
            no_log_name = True

        if self.job.skip_logging or not self.job.construct_pose:  # Set up null_logger
            self.log = logging.getLogger('null')
        else:  # f'{__name__}.{self}'
            if self.job.force:
                os.system(f'rm {self.log_path}')
            self.log = start_log(name=f'pose.{self}', handler=handler, level=level, location=self.log_path,
                                 no_log_name=no_log_name)  # propagate=propagate,

    @handle_design_errors(errors=(FileNotFoundError, ValueError))
    @close_logs
    def setup(self, pre_refine: bool = None, pre_loop_model: bool = None):
        """Prepare output Directory and File locations. Each PoseDirectory always includes this format

        Args:
            pre_refine: Whether the Pose has been refined previously (before loading)
            pre_loop_model: Whether the Pose had loops modeled previously (before loading)
        """
        self.start_log()
        if self.initialized:  # os.path.exists(self.serialized_info):  # gather state data
            try:
                serial_info = unpickle(self.serialized_info)
                if not self.info:  # empty dict
                    self.info = serial_info
                else:
                    serial_info.update(self.info)
                    self.info = serial_info
            except pickle.UnpicklingError as error:
                print(f'ERROR {self.name}: There was an issue retrieving design state from binary file...')
                raise error
            # Make a copy for the checking of current state
            self._info = self.info.copy()
            # # Dev branch only
            # except ModuleNotFoundError as error:
            #     self.log.error('%s: There was an issue retrieving design state from binary file...' % self.name)
            #     self.log.critical('Removing %s' % self.serialized_info)
            #     # raise error
            #     remove(self.serialized_info)
            # if stat(self.serialized_info).st_size > 10000:
            #     print('Found pickled file with huge size %d. fragment_database being removed'
            #           % stat(self.serialized_info).st_size)
            #     self.info['fragment_source'] = \
            #         getattr(self.info.get('fragment_database'), 'source', putils.biological_interfaces)
            #     self.pickle_info()  # save immediately so we don't have this issue with reading again!
            # Todo Remove Above this line to Dev branch only
            # # These statements are a temporary patch Todo remove for SymDesign master branch
            # # if not self.sym_entry:  # none was provided at initiation or in state
            # if putils.sym_entry in self.info:
            #     self.sym_entry = self.info[putils.sym_entry]  # get instance
            #     self.info.pop(putils.sym_entry)  # remove this object
            #     self.info['sym_entry_specification'] = self.sym_entry.entry_number, self.sym_entry.sym_map
            if 'oligomer_names' in self.info:
                self.info['entity_names'] = [f'{name}_1' for name in self.info['oligomer_names']]
            # if 'design_residue_ids' in self.info:  # format is old, convert
            #     try:
            #         self.info['interface_design_residues'] = self.info.pop('design_residues')
            #     except KeyError:
            #         pass
            #     self.info['interface_residue_ids'] = self.info.pop('design_residue_ids')
            #     try:
            #         self.info['interface_residues'] = self.info.pop('interface_residues')
            #     except KeyError:
            #         pass
            # else:  # format is old old, remove all
            #     for old_element in ['design_residues', 'interface_residues']:
            #         try:
            #             self.info.pop(old_element)
            #         except KeyError:
            #             pass
            #
            # if 'fragment_database' in self.info:
            #     self.info['fragment_source'] = self.info.get('fragment_database')
            #     self.info.pop('fragment_database')
            # fragment_data = self.info.get('fragment_data')
            # if fragment_data and not isinstance(fragment_data, dict):  # this is a .pkl file
            #     try:
            #         self.info['fragment_data'] = unpickle(fragment_data)
            #         remove(fragment_data)
            #     except FileNotFoundError:
            #         self.info.pop('fragment_data')
            # if 'pose_transformation' in self.info:
            #     self._pose_transformation = self.info.get('pose_transformation')
            #     if isinstance(self._pose_transformation, dict):  # old format
            #         del self._pose_transformation
            # self.pickle_info()
        else:  # We haven't initialized this PoseDirectory before
            # __init__ assumes structures have been refined so these only act to set false
            if pre_refine is not None:  # either True or False
                self.info['pre_refine'] = pre_refine  # this may have just been set
            if pre_loop_model is not None:  # either True or False
                self.info['pre_loop_model'] = pre_loop_model

        # self.design_selector = self.info.get('design_selector', self.design_selector)
        # self.pose_transformation = self.info.get('pose_transformation', [])
        # self.fragment_observations = self.info.get('fragments', None)  # None signifies query wasn't attempted
        # self.interface_design_residue_numbers = self.info.get('interface_design_residues', False)  # (set[int])
        # self.interface_residue_ids = self.info.get('interface_residue_ids', {})
        # self.interface_residue_numbers = self.info.get('interface_residues', False)  # (set[int])
        # self.entity_names = self.info.get('entity_names', [])
        self.pre_refine = self.info.get('pre_refine', True)
        self.pre_loop_model = self.info.get('pre_loop_model', True)

        if self.job.nanohedra_output and self.job.construct_pose:
            if not os.path.exists(os.path.join(self.path, putils.pose_file)):
                shutil.copy(self.pose_file, self.path)
                shutil.copy(self.frag_file, self.path)
            # self.info['oligomer_names'] = self.oligomer_names
            # self.info['entity_names'] = self.entity_names
            self.pickle_info()  # Save this info on the first copy so that we don't have to construct again

        # Check if the source of the pdb files was refined upon loading
        if self.pre_refine:
            self.refined_pdb = self.asu_path
            self.scouted_pdb = os.path.join(self.designs,
                                            f'{os.path.basename(os.path.splitext(self.refined_pdb)[0])}_scout.pdb')
        else:
            self.refined_pdb = os.path.join(self.designs,
                                            f'{os.path.basename(os.path.splitext(self.asu_path)[0])}_refine.pdb')
            self.scouted_pdb = f'{os.path.splitext(self.refined_pdb)[0]}_scout.pdb'

        # # Check if the source of the pdb files was loop modelled upon loading
        # if self.pre_loop_model:

        # # Configure standard pose loading mechanism with self.source
        # if self.specific_designs:
        #     # Introduce flag handling current inability of specific_designs to handle iteration
        #     self._lock_optimize_designs = True
        #     self.specific_designs_file_paths = []
        #     for design in self.specific_designs:
        #         matching_path = os.path.join(self.designs, f'*{design}.pdb')
        #         matching_designs = sorted(glob(matching_path))
        #         if matching_designs:
        #             for matching_design in matching_designs:
        #                 if os.path.exists(matching_design):
        #                     # self.specific_design_path = matching_design
        #                     self.specific_designs_file_paths.append(matching_design)
        #             if len(matching_designs) > 1:
        #                 self.log.warning(f'Found {len(matching_designs)} matching designs to your specified design '
        #                                  f'using {matching_path}. Choosing the first {matching_designs[0]}')
        #         else:
        #             raise DesignError(f"Couldn't locate a specific_design matching the name '{matching_path}'")
        #         # format specific_designs to a pose ID compatible format
        #     self.specific_designs = [f'{self.name}_{design}' for design in self.specific_designs]
        #     # self.source = specific_designs_file_paths  # Todo?
        #     # self.source = self.specific_design_path
        # elif self.source is None:
        #     if os.path.exists(self.asu_path):  # Standard mechanism of loading the pose
        #         self.source = self.asu_path
        #     else:
        #         try:
        #             glob_target = os.path.join(self.path, f'{self.name}*.pdb')
        #             self.source = sorted(glob(glob_target))
        #             if len(self.source) == 1:
        #                 self.source = self.source[0]
        #             else:
        #                 raise ValueError(f'Found {len(self.source)} files matching the path "{glob_target}". '
        #                                  f'Only 1 expected')
        #         except IndexError:  # glob found no files
        #             self.source = None
        # else:  # If the PoseDirectory was loaded as .pdb/mmCIF, the source should be loaded already
        #     # self.source = self.initial_model
        #     pass

    @property
    def symmetry_definition_files(self) -> list[AnyStr]:
        """Retrieve the symmetry definition files name from PoseDirectory"""
        try:
            return self._symmetry_definition_files
        except AttributeError:
            self._symmetry_definition_files = sorted(glob(os.path.join(self.data, '*.sdf')))
            return self._symmetry_definition_files

    def get_wildtype_file(self) -> AnyStr:
        """Retrieve the wild-type file name from PoseDirectory"""
        wt_file = glob(self.asu_path)
        if len(wt_file) != 1:
            raise ValueError(f'More than one matching file found during search {self.asu_path}')

        return wt_file[0]

    def get_designs(self, design_type: str = None) -> list[AnyStr]:  # design_type: str = putils.interface_design
        """Return the paths of all design files in a PoseDirectory

        Args:
            design_type: Specify if a particular type of design should be selected by a "type" string
        Returns:
            The sorted design files found in the designs directory
        """
        if design_type is None:
            design_type = ''
        return sorted(glob(os.path.join(self.designs, f'*{design_type}*.pdb*')))

    def pickle_info(self):
        """Write any design attributes that should persist over program run time to serialized file"""
        if not self.job.construct_pose:  # This is only true when self.job.nanohedra_output is True
            # Don't write anything as we are just querying
            return
        putils.make_path(self.data)
        # try:
        # Todo make better patch for numpy.ndarray compare value of array is ambiguous
        if self.info.keys() != self._info.keys():  # if the state has changed from the original version
            pickle_object(self.info, self.serialized_info, out_path='')
        # except ValueError:
        #     print(self.info)

    def transform_entities_to_pose(self, **kwargs):  # Todo to PoseProtocols?
        """Take the set of entities involved in a pose composition and transform them from a standard reference frame to
        the Pose reference frame using the pose.entity_transformations parameters

        Keyword Args:
            refined: bool = True - Whether to use refined models from the StructureDatabase
            oriented: bool = False - Whether to use oriented models from the StructureDatabase
        """
        self.get_entities(**kwargs)
        if self.pose.entity_transformations:
            self.log.debug('Entities were transformed to the found docking parameters')
            self.entities = [entity.get_transformed_copy(**transformation)
                             for entity, transformation in zip(self.entities, self.pose.entity_transformations)]
        else:  # Todo change below to handle asymmetric cases...
            raise SymmetryError('The design could not be transformed as it is missing the required '
                                'entity_transformations parameter. Were they generated properly?')

    def transform_structures_to_pose(self, structures: Iterable[Structure], **kwargs) -> list[Structure]:  # Todo to PoseProtocols?
        """Take a set of Structure instances and transform them from a standard reference frame to the Pose reference
        frame using the pose.entity_transformations parameters

        Args:
            structures: The Structure objects you would like to transform
        Returns:
            The transformed Structure objects if a transformation was possible
        """
        if self.pose.entity_transformations:
            self.log.debug('Structures were transformed to the found docking parameters')
            # Todo assumes a 1:1 correspondence between structures and transforms (component group numbers) CHANGE
            return [structure.get_transformed_copy(**transformation)
                    for structure, transformation in zip(structures, self.pose.entity_transformations)]
        else:
            return list(structures)

    def get_entities(self, refined: bool = True, oriented: bool = False, **kwargs):  # Todo to PoseProtocols?
        """Retrieve Entity files from the design Database using either the oriented directory, or the refined directory.
        If these don't exist, use the Pose directory, and load them into job for further processing

        Args:
            refined: Whether to use the refined directory
            oriented: Whether to use the oriented directory
        Sets:
            self.entities (list[Entity])
        """
        source_preference = ['refined', 'oriented_asu', 'design']  # Todo once loop_model works 'full_models'
        if self.job.structure_db:
            if refined:
                source_idx = 0
            elif oriented:
                source_idx = 1
            else:
                source_idx = 2
                self.log.info(f'Falling back on entities present in the {type(self).__name__} source')

            self.entities.clear()
            for name in self.entity_names:
                source_preference_iter = iter(source_preference)
                # Discard however many sources are unwanted (source_idx number)
                for it in range(source_idx):
                    _ = next(source_preference_iter)
                model = None
                while not model:
                    try:
                        source = next(source_preference_iter)
                    except StopIteration:
                        raise DesignError(f'{self.get_entities.__name__}: Couldn\'t locate the required files')
                    source_datastore = getattr(self.job.structure_db, source, None)
                    if source_datastore is None:  # if source == 'design':
                        search_path = os.path.join(self.path, f'{name}*.pdb*')
                        file = sorted(glob(search_path))
                        if file:
                            if len(file) > 1:
                                self.log.warning(f'The specified entity has multiple files at "{search_path}". '
                                                 f'Using the first')
                            model = Model.from_file(file[0], log=self.log)
                        else:
                            raise FileNotFoundError(f'Couldn\'t located the specified entity at "{file}"')
                    else:
                        model = source_datastore.retrieve_data(name=name)
                        if isinstance(model, Model):
                            self.log.info(f'Found Model at {source} DataStore and loaded into job')
                        else:
                            self.log.error(f'Couldn\'t locate the Model {name} at the source '
                                           f'"{source_datastore.location}"')

                self.entities.extend([entity for entity in model.entities])
            if source_idx == 0:
                self.pre_refine = True
        # else:  # Todo I don't think this code is reachable. Consolidate this with above as far as iterative mechanism
        #     out_dir = ''
        #     if refined:  # prioritize the refined version
        #         out_dir = self.job.refine_dir
        #         for name in self.entity_names:
        #             if not os.path.exists(glob(os.path.join(self.job.refine_dir, f'{name}*.pdb*'))[0]):
        #                 oriented = True  # fall back to the oriented version
        #                 self.log.debug('Couldn\'t find entities in the refined directory')
        #                 break
        #         self.pre_refine = True if not oriented else False
        #     if oriented:
        #         out_dir = self.job.orient_dir
        #         for name in self.entity_names:
        #             if not os.path.exists(glob(os.path.join(self.job.refine_dir, f'{name}*.pdb*'))[0]):
        #                 out_dir = self.path
        #                 self.log.debug('Couldn\'t find entities in the oriented directory')
        #
        #     if not refined and not oriented:
        #         out_dir = self.path
        #
        #     idx = 2  # initialize as 2. it doesn't matter if no names are found, but nominally it should be 2 for now
        #     oligomer_files = []
        #     for idx, name in enumerate(self.entity_names, 1):
        #         oligomer_files.extend(sorted(glob(os.path.join(out_dir, f'{name}*.pdb*'))))  # first * is PoseDirectoy
        #     assert len(oligomer_files) == idx, \
        #         f'Incorrect number of entities! Expected {idx}, {len(oligomer_files)} found. Matched files from ' \
        #         f'"{os.path.join(out_dir, "*.pdb*")}":\n\t{oligomer_files}'
        #
        #     self.entities.clear()  # for every call we should reset the list
        #     for file in oligomer_files:
        #         self.entities.append(Model.from_file(file, name=os.path.splitext(os.path.basename(file))[0],
        #                                              log=self.log))
        self.log.debug(f'{len(self.entities)} matching entities found')
        if len(self.entities) != len(self.entity_names):  # Todo need to make len(self.symmetry_groups) from SymEntry
            raise RuntimeError(f'Expected {len(self.entities)} entities, but found {len(self.entity_names)}')

    def load_pose(self, source: str = None, entities: list[Structure] = None):  # Todo to PoseProtocols?
        """For the design info given by a PoseDirectory source, initialize the Pose with self.source file,
        self.symmetry, self.job, self.fragment_database, and self.log objects

        Handles Pose clash testing, writing the Pose, adding state variables to the pose

        Args:
            source: The file path to a source file
            entities: The Entities desired in the Pose
        """
        if self.pose and not source and not entities:
            # Pose is already loaded and nothing new provided
            return

        # rename_chains = True  # Because the result of entities, we should rename
        if not entities and not self.source or not os.path.exists(self.source):
            # In case we initialized design without a .pdb or clean_asu.pdb (Nanohedra)
            self.log.info(f'No source file found. Fetching source from {type(self.job.structure_db).__name__} and '
                          f'transforming to Pose')
            # Minimize I/O with transform...
            self.transform_entities_to_pose()
            entities = self.entities
            # entities = []
            # for entity in self.entities:
            #     entities.extend(entity.entities)
            # # Because the file wasn't specified on the way in, no chain names should be binding
            # # rename_chains = True

        # Initialize the Pose with the pdb in PDB numbering so that residue_selectors are respected
        name = f'{self}-asu' if self.sym_entry else str(self)
        if entities:
            self.pose = Pose.from_entities(entities, name=name, **self.pose_kwargs)
        elif self.initial_model:  # This is a fresh Model, and we already loaded so reuse
            # Careful, if processing has occurred then this may be wrong!
            self.pose = Pose.from_model(self.initial_model, name=name, **self.pose_kwargs)
        else:
            self.pose = Pose.from_file(source if source else self.source, name=name, **self.pose_kwargs)

        if self.pose.is_symmetric():
            if self.job.write_oligomers:  # Write out new oligomers to the PoseDirectory
                for idx, entity in enumerate(self.pose.entities):
                    entity.write(oligomer=True, out_path=os.path.join(self.path, f'{entity.name}_oligomer.pdb'))
            if not self.pose_transformation:  # If an empty list, save the value identified
                self.pose_transformation = self.pose.entity_transformations
        # Then modify numbering to ensure standard and accurate use during protocols
        # self.pose.pose_numbering()
        if not self.entity_names:  # Store the entity names if they were never generated
            self.entity_names = [entity.name for entity in self.pose.entities]
            self.log.info(f'Input Entities: {", ".join(self.entity_names)}')
            self.info['entity_names'] = self.entity_names

        # Save renumbered PDB to clean_asu.pdb
        if not self.asu_path or not os.path.exists(self.asu_path) or self.job.force:
            if not self.job.construct_pose:  # This is only true when self.job.nanohedra_output is True
                return
            # elif self.job.output_to_directory:
            #     return

            self.save_asu()

    def save_asu(self):  # Todo to PoseProtocols? # , rename_chains=False
        """Save a new Structure from multiple Chain or Entity objects including the Pose symmetry"""
        if self.job.fuse_chains:
            # try:
            for fusion_nterm, fusion_cterm in self.job.fuse_chains:
                # rename_success = False
                new_success, same_success = False, False
                for idx, entity in enumerate(self.pose.entities):
                    if entity.chain_id == fusion_cterm:
                        entity_new_chain_idx = idx
                        new_success = True
                    if entity.chain_id == fusion_nterm:
                        # entity_same_chain_idx = idx
                        same_success = True
                        # rename_success = True
                        # break
                # if not rename_success:
                if not new_success and not same_success:
                    raise DesignError('Your requested fusion of chain %s with chain %s didn\'t work!' %
                                      (fusion_nterm, fusion_cterm))
                    # self.log.critical('Your requested fusion of chain %s with chain %s didn\'t work!' %
                    #                   (fusion_nterm, fusion_cterm))
                else:  # won't be accessed unless entity_new_chain_idx is set
                    self.pose.entities[entity_new_chain_idx].chain_id = fusion_nterm
            # except AttributeError:
            #     raise ValueError('One or both of the chain IDs %s were not found in the input model. Possible chain'
            #                      ' ID\'s are %s' % ((fusion_nterm, fusion_cterm), ','.join(new_asu.chain_ids)))
        self.pose.write(out_path=self.asu_path)
        self.log.info(f'Cleaned PDB: "{self.asu_path}"')

    def identify_interface(self):  # Todo to PoseProtocols?
        """Find the interface(s) between each Entity in the Pose. Handles symmetric clash testing, writing the assembly
        """
        #         Sets:
        #             self.interface_residue_ids (dict[str, str]):
        #                 Map each interface to the corresponding residue/chain pairs
        #             self.interface_design_residue_numbers (set[int]):
        #                 The residues in proximity of the interface, including buried residues
        #             self.interface_residue_numbers (set[int]):
        #                 The residues in contact across the interface

        self.load_pose()
        if self.symmetric:
            self.symmetric_assembly_is_clash()
            if self.job.output_assembly:
                self.pose.write(assembly=True, out_path=self.assembly_path, increment_chains=self.job.increment_chains)
                self.log.info(f'Symmetric assembly written to: "{self.assembly_path}"')

        self.pose.find_and_split_interface()

        # self.interface_design_residue_numbers = set()  # Replace set(). Add new residues
        # for number, residues_entities in self.pose.split_interface_residues.items():
        #     self.interface_design_residue_numbers.update([residue.number for residue, _ in residues_entities])
        # self.log.debug(f'Found interface design residues: '
        #                f'{", ".join(map(str, sorted(self.interface_design_residue_numbers)))}')

        # self.interface_residue_numbers = set()  # Replace set(). Add new residues
        # for entity in self.pose.entities:
        #     # Todo v clean as it is redundant with analysis and falls out of scope
        #     entity_oligomer = Model.from_chains(entity.chains, log=entity.log, entities=False)
        #     # entity.oligomer.get_sasa()
        #     # Must get_residues by number as the Residue instance will be different in entity_oligomer
        #     for residue in entity_oligomer.get_residues(self.interface_design_residue_numbers):
        #         if residue.sasa > 0:
        #             # Using set ensures that if we have repeats they won't be unique if Entity is symmetric
        #             self.interface_residue_numbers.add(residue.number)
        # self.log.debug(f'Found interface residues: {", ".join(map(str, sorted(self.interface_residue_numbers)))}')

        # for number, residues_entities in self.pose.split_interface_residues.items():
        #     self.interface_residue_ids[f'interface{number}'] = \
        #         ','.join(f'{residue.number}{entity.chain_id}' for residue, entity in residues_entities)

        # self.info['interface_design_residues'] = self.interface_design_residue_numbers
        # self.info['interface_residues'] = self.interface_residue_numbers
        # self.info['interface_residue_ids'] = self.interface_residue_ids

    def symmetric_assembly_is_clash(self):  # Todo to PoseProtocols?
        """Wrapper around the Pose symmetric_assembly_is_clash() to check at the Pose level for clashes and raise
        ClashError if any are found, otherwise, continue with protocol
        """
        if self.pose.symmetric_assembly_is_clash():
            if self.job.design.ignore_symmetric_clashes:
                self.log.critical(f'The Symmetric Assembly contains clashes! {self.source} is not viable')
            else:
                raise ClashError("The Symmetric Assembly contains clashes! Design won't be considered. If you "
                                 'would like to generate the Assembly anyway, re-submit the command with '
                                 f'--{flags.ignore_symmetric_clashes}')

    def prepare_rosetta_flags(self, symmetry_protocol: str = None, sym_def_file: str = None, pdb_out_path: str = None,
                              out_dir: AnyStr = os.getcwd()) -> str:
        """Prepare a protocol specific Rosetta flags file with program specific variables

        Args:
            symmetry_protocol: The type of symmetric protocol (specifying design dimension) to use for Rosetta jobs
            sym_def_file: A Rosetta specific file specifying the symmetry system
            pdb_out_path: Disk location to write the resulting design files
            out_dir: Disk location to write the flags file
        Returns:
            Disk location of the flags file
        """
        # flag_variables (list(tuple)): The variable value pairs to be filed in the RosettaScripts XML
        number_of_residues = self.pose.number_of_residues
        self.log.info(f'Total number of residues in Pose: {number_of_residues}')

        # Get ASU distance parameters
        if self.design_dimension:  # Check for None and dimension 0 simultaneously
            # The furthest point from the ASU COM + the max individual Entity radius
            distance = self.pose.radius + max([entity.radius for entity in self.pose.entities])  # all the radii
            self.log.info(f'Expanding ASU into symmetry group by {distance:.2f} Angstroms')
        else:
            distance = 0

        if not self.job.design.evolution_constraint:
            constraint_percent, free_percent = 0, 1
        else:
            constraint_percent = 0.5
            free_percent = 1 - constraint_percent

        variables = rosetta.rosetta_variables \
            + [('dist', distance), ('repack', 'yes'),
               ('constrained_percent', constraint_percent), ('free_percent', free_percent)]
        variables.extend([(putils.design_profile, self.design_profile_file)]
                         if os.path.exists(self.design_profile_file) else [])
        variables.extend([(putils.fragment_profile, self.fragment_profile_file)]
                         if os.path.exists(self.fragment_profile_file) else [])

        if self.symmetric:
            def prepare_symmetry_for_rosetta():
                """For the specified design, locate/make the symmetry files necessary for Rosetta input

                Sets:
                    self.symmetry_protocol (str)
                    self.sym_def_file (AnyStr)
                """
                if self.sym_entry is None:  # asymmetric
                    self.symmetry_protocol = 'asymmetric'
                    # self.sym_def_file = sdf_lookup()
                    self.log.debug('No symmetry invoked during design')
                else:  # Symmetric
                    self.symmetry_protocol = symmetry_protocols[self.design_dimension]
                    self.sym_def_file = self.sym_entry.sdf_lookup()
                self.log.info(f'Symmetry Option: {self.symmetry_protocol}')

            prepare_symmetry_for_rosetta()
            if symmetry_protocol is None:
                symmetry_protocol = self.symmetry_protocol
            if not sym_def_file:
                sym_def_file = self.sym_def_file
            variables.extend([] if symmetry_protocol is None else [('symmetry', symmetry_protocol),
                                                                   ('sdf', sym_def_file)])
            out_of_bounds_residue = number_of_residues*self.pose.number_of_symmetry_mates + 1
        else:
            variables.append(('symmetry', symmetry_protocol))
            out_of_bounds_residue = number_of_residues + 1

        interface_residue_ids = {}
        for number, residues_entities in self.pose.split_interface_residues.items():
            interface_residue_ids[f'interface{number}'] = \
                ','.join(f'{residue.number}{entity.chain_id}' for residue, entity in residues_entities)
        # self.info['interface_residue_ids'] = self.interface_residue_ids
        variables.extend([(interface, residues) if residues else (interface, out_of_bounds_residue)
                          for interface, residues in interface_residue_ids.items()])

        # Assign any additional designable residues
        if self.pose.required_residues:
            variables.extend([('required_residues', ','.join(f'{res.number}{res.chain}'
                                                             for res in self.pose.required_residues))])
        else:  # Get an out-of-bounds index
            variables.extend([('required_residues', out_of_bounds_residue)])

        # Allocate any "core" residues based on central fragment information
        residues = self.pose.residues
        center_residues = [residues[index] for index in self.pose.center_residue_indices]
        if center_residues:
            variables.extend([('core_residues', ','.join([f'{res.number}{res.chain}' for res in center_residues]))])
        else:  # Get an out-of-bounds index
            variables.extend([('core_residues', out_of_bounds_residue)])

        rosetta_flags = copy(rosetta.rosetta_flags)
        if pdb_out_path:
            rosetta_flags.extend([f'-out:path:pdb {pdb_out_path}', f'-scorefile {self.scores_file}'])
        else:
            rosetta_flags.extend([f'-out:path:pdb {self.designs}', f'-scorefile {self.scores_file}'])
        rosetta_flags.append(f'-in:file:native {self.refined_pdb}')
        rosetta_flags.append(f'-parser:script_vars {" ".join(f"{var}={val}" for var, val in variables)}')

        putils.make_path(out_dir)
        out_file = os.path.join(out_dir, 'flags')
        with open(out_file, 'w') as f:
            f.write('%s\n' % '\n'.join(rosetta_flags))

        return out_file

    # Below methods run Rosetta script set up for various design applications
    def generate_entity_metrics(self, entity_command) -> list[list[str] | None]:
        """Use the Pose state to generate metrics commands for each Entity instance

        Args:
            entity_command: The base command to build Entity metric commands off of
        Returns:
            The formatted command for every Entity in the Pose
        """
        # self.entity_names not dependent on Pose load
        if len(self.entity_names) == 1:  # there is no unbound state to query as only one entity
            return []
        if len(self.symmetry_definition_files) != len(self.entity_names) or self.job.force:
            for entity in self.pose.entities:
                if entity.is_oligomeric():  # make symmetric energy in line with SymDesign energies v
                    entity.make_sdf(out_path=self.data,
                                    modify_sym_energy_for_cryst=True if self.sym_entry.dimension in [2, 3] else False)
                else:
                    shutil.copy(os.path.join(putils.symmetry_def_files, 'C1.sym'),
                                os.path.join(self.data, f'{entity.name}.sdf'))

        entity_metric_commands = []
        for idx, (entity, name) in enumerate(zip(self.pose.entities, self.entity_names), 1):
            if self.symmetric:
                entity_sdf = f'sdf={os.path.join(self.data, f"{name}.sdf")}'
                entity_sym = 'symmetry=make_point_group'
            else:
                entity_sdf, entity_sym = '', 'symmetry=asymmetric'
            metric_cmd = entity_command + ['-parser:script_vars', 'repack=yes', f'entity={idx}', entity_sym] + \
                ([entity_sdf] if entity_sdf != '' else [])
            self.log.info(f'Metrics Command for Entity {name}: {list2cmdline(metric_cmd)}')
            entity_metric_commands.append(metric_cmd)

        return entity_metric_commands

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def predict_structure(self):
        """From a sequence input, predict the structure using one of various structure prediction pipelines"""
        self._predict_structure()

    def _predict_structure(self):
        match self.job.predict.method:
            case 'thread':
                self.thread_sequences_to_backbone()
            case 'proteinmpnn':
                self.thread_sequences_to_backbone()
            # Todo
            #  case 'alphafold':
            #      self.run_alphafold()
            case other:
                raise NotImplementedError(f"For {self.predict_structure.__name__}, the method {self.job.predict.method}"
                                          " isn't implemented yet")

    def thread_sequences_to_backbone(self, sequences: dict[str, str] = None):
        """From the starting Pose, thread sequences onto the backbone, modifying relevant side chains i.e., mutate the
        Pose and build/pack using Rosetta FastRelax. If no sequences are provided, will use self.designed_sequences

        Args:
            sequences: The sequences to thread
        """
        if sequences is None:  # Gather all already designed sequences
            # refine_sequences = unpickle(self.designed_sequences_file)
            sequences = {seq.id: seq.seq for seq in read_fasta_file(self.designed_sequences_file)}

        # if self.protocol is not None:  # This hasn't been set yet
        self.protocol = 'thread'

        # Write each "threaded" structure out for further processing
        number_of_residues = self.pose.number_of_residues
        design_files = []
        for sequence_id, sequence in sequences.items():
            if len(sequence) != number_of_residues:
                raise DesignError(f'The length of the sequence {len(sequence)} != {number_of_residues}, '
                                  f'the number of residues in the pose')
            for res_idx, residue_type in enumerate(sequence):
                self.pose.mutate_residue(index=res_idx, to=residue_type)
            # pre_threaded_file = os.path.join(self.data, f'{self.name}_{self.protocol}{seq_idx:04d}.pdb')
            pre_threaded_file = os.path.join(self.data, f'{sequence_id}.pdb')
            design_files.append(self.pose.write(out_path=pre_threaded_file))

        design_files_file = os.path.join(self.scripts, f'files_{self.protocol}.txt')
        putils.make_path(self.scripts)

        # Modify each sequence score to reflect the new "decoy" name
        sequence_ids = sequences.keys()
        design_scores = read_scores(self.scores_file)
        for design, scores in design_scores.items():
            if design in sequence_ids:
                # We previously saved data. Copy to the identifier that is present after threading
                scores['decoy'] = f'{design}_{self.protocol}'
                # write_json(_scores, self.scores_file)
                with open(self.scores_file, 'a') as f_save:
                    json.dump(scores, f_save)  # , **kwargs)
                    # Ensure JSON lines are separated by newline
                    f_save.write('\n')

        if design_files:
            with open(design_files_file, 'w') as f:
                f.write('%s\n' % '\n'.join(design_files))
        else:
            raise DesignError(f'{self.thread_sequences_to_backbone.__name__}: No designed sequences were located')

        self._refine(in_file_list=design_files_file)

    def _refine(self, to_pose_directory: bool = True, metrics: bool = True, in_file_list: AnyStr = None):
        """Refine the source Pose

        Will append the suffix "_refine" or that given by f'_{self.protocol}' if in_file_list is passed

        Args:
            to_pose_directory: Whether the refinement should be saved to the PoseDirectory
            metrics: Whether metrics should be calculated for the Pose
            in_file_list: A list of files to perform refinement on
        """
        main_cmd = copy(rosetta.script_cmd)

        infile = []
        if to_pose_directory:  # Original protocol to refine a pose as provided from Nanohedra
            flag_dir = self.scripts
            pdb_out_path = self.designs
            refine_pdb = self.refine_pdb
            refined_pdb = self.refined_pdb
            additional_flags = []
        else:  # Protocol to refine input structure, place in a common location, then transform for many jobs to source
            flag_dir = self.job.refine_dir
            pdb_out_path = self.job.refine_dir
            refine_pdb = self.source
            refined_pdb = os.path.join(pdb_out_path, refine_pdb)
            additional_flags = ['-no_scorefile', 'true']
            infile.extend(['-in:file:s', refine_pdb,
                           # -in:file:native is here to block flag file version, not actually useful for refine
                           '-in:file:native', refine_pdb])

        flags_file = os.path.join(flag_dir, 'flags')
        if not os.path.exists(flags_file) or self.job.force:
            self.prepare_rosetta_flags(pdb_out_path=pdb_out_path, out_dir=flag_dir)
            self.log.debug(f'Pose flags written to: {flags_file}')
            putils.make_path(pdb_out_path)

        # Assign designable residues to interface1/interface2 variables, not necessary for non-complexed PDB jobs
        if in_file_list is not None:  # Run a list of files produced elsewhere
            possible_refine_protocols = ['refine', 'thread']
            if self.protocol in possible_refine_protocols:
                switch = self.protocol
            elif self.protocol is None:
                switch = putils.refine
            else:
                switch = putils.refine
                self.log.warning(f'The requested protocol {self.protocol}, was not recognized by '
                                 f'{self._refine.__name__} and is being treated as the standard "{switch}" protocol')

            infile.extend(['-in:file:l', in_file_list,
                           # -in:file:native is here to block flag file version, not actually useful for refine
                           '-in:file:native', self.source])
            designed_files = os.path.join(self.scripts, f'design_files_{self.protocol}.txt')
            generate_files_cmd = \
                ['python', putils.list_pdb_files, '-d', self.designs, '-o', designed_files, '-s', '_' + self.protocol]
            metrics_pdb = ['-in:file:l', designed_files, '-in:file:native', self.source]
            # generate_files_cmdline = [list2cmdline(generate_files_cmd)]
        else:
            # if self.interface_residue_numbers is False or self.interface_design_residue_numbers is False:
            self.identify_interface()
            # else:  # We only need to load pose as we already calculated interface
            #     self.load_pose()

            self.protocol = switch = putils.refine
            if self.job.interface_to_alanine:  # Mutate all design positions to Ala before the Refinement
                for entity_pair, interface_residue_sets in self.pose.interface_residues_by_entity_pair.items():
                    if interface_residue_sets[0]:  # Check that there are residues present
                        for idx, interface_residue_set in enumerate(interface_residue_sets):
                            self.log.debug(f'Mutating residues from Entity {entity_pair[idx].name}')
                            for residue in interface_residue_set:
                                self.log.debug(f'Mutating {residue.number}{residue.type}')
                                if residue.type != 'GLY':  # No mutation from GLY to ALA as Rosetta would build a CB
                                    self.pose.mutate_residue(residue=residue, to='A')
                # Change the name to reflect mutation so we don't overwrite the self.source
                refine_pdb = f'{os.path.splitext(refine_pdb)[0]}_ala_mutant.pdb'
            # else:  # Do dothing and refine the source
            #     pass
            #     # raise ValueError(f"For {self.refine.__name__}, must pass interface_to_alanine")

            self.pose.write(out_path=refine_pdb)
            self.log.debug(f'Cleaned PDB for {self.protocol}: "{refine_pdb}"')
            infile.extend(['-in:file:s', refine_pdb,
                           # -in:file:native is here to block flag file version, not actually useful for refine
                           '-in:file:native', refine_pdb])
            generate_files_cmd = null_cmd
            # generate_files_cmdline = []
            metrics_pdb = ['-in:file:s', refined_pdb, '-in:file:native', refine_pdb]

        # RELAX: Prepare command
        # '-no_nstruct_label', 'true' comes from v
        relax_cmd = main_cmd + rosetta.relax_flags_cmdline + additional_flags + \
            (['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []) + infile + \
            [f'@{flags_file}', '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, f'refine.xml'),
             '-out:suffix', f'_{switch}', '-parser:script_vars', f'switch={switch}']
        self.log.info(f'{self.protocol.title()} Command: {list2cmdline(relax_cmd)}')

        if metrics or self.job.metrics:
            metrics = True
            main_cmd += metrics_pdb
            main_cmd += [f'@{flags_file}', '-out:file:score_only', self.scores_file,
                         '-no_nstruct_label', 'true', '-parser:protocol']
            if self.job.mpi > 0:
                main_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(self.job.mpi)] + main_cmd

            metric_cmd_bound = main_cmd + (['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []) + \
                [os.path.join(putils.rosetta_scripts_dir, f'{putils.interface_metrics}'
                              f'{"_DEV" if self.job.development else ""}.xml')]
            entity_cmd = main_cmd + [os.path.join(putils.rosetta_scripts_dir,
                                                  f'metrics_entity{"_DEV" if self.job.development else ""}.xml')]
            self.log.info(f'Metrics Command: {list2cmdline(metric_cmd_bound)}')
            metric_cmds = [metric_cmd_bound]
            metric_cmds.extend(self.generate_entity_metrics(entity_cmd))
        else:
            metric_cmds = []

        # Create executable/Run FastRelax on Clean ASU with RosettaScripts
        if self.job.distribute_work:
            analysis_cmd = ['python', putils.program_exe, putils.analysis, '--single', self.path, '--no-output',
                            f'--{flags.output_file}', os.path.join(self.job.all_scores, putils.default_analysis_file
                                                                   .format(starttime, self.protocol))]
            write_shell_script(list2cmdline(relax_cmd), name=self.protocol, out_path=flag_dir,
                               additional=[list2cmdline(generate_files_cmd)] +
                                          [list2cmdline(command) for command in metric_cmds] +
                                          [list2cmdline(analysis_cmd)])
            #                  status_wrap=self.serialized_info)
        else:
            relax_process = Popen(relax_cmd)
            relax_process.communicate()  # wait for command to complete
            list_all_files_process = Popen(generate_files_cmd)
            list_all_files_process.communicate()
            if metrics:
                for metric_cmd in metric_cmds:
                    metrics_process = Popen(metric_cmd)
                    metrics_process.communicate()

        # ANALYSIS: each output from the Design process based on score, Analyze Sequence Variation
        # return
        # # Todo this isn't working right now with mutations to structure
        if not self.job.distribute_work:
            pose_s = self._interface_design_analysis()
            out_path = os.path.join(self.job.all_scores, putils.default_analysis_file.format(starttime, 'All'))
            if os.path.exists(out_path):
                header = False
            else:
                header = True
            pose_s.to_csv(out_path, mode='a', header=header)

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def custom_rosetta_script(self, script, file_list=None, native=None, suffix=None,
                              score_only=None, variables=None, **kwargs):
        """Generate a custom script to dispatch to the design using a variety of parameters"""
        raise DesignError('This module is outdated, please update it to use')  # Todo reflect modern metrics collection
        cmd = copy(rosetta.script_cmd)
        script_name = os.path.splitext(os.path.basename(script))[0]
        # if self.interface_residue_numbers is False or self.interface_design_residue_numbers is False:
        self.identify_interface()
        # else:  # We only need to load pose as we already calculated interface
        #     self.load_pose()

        if not os.path.exists(self.flags) or self.job.force:
            self.prepare_rosetta_flags(out_dir=self.scripts)
            self.log.debug(f'Pose flags written to: {self.flags}')

        cmd += ['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []

        if file_list:
            pdb_input = os.path.join(self.scripts, 'design_files.txt')
            generate_files_cmd = ['python', putils.list_pdb_files, '-d', self.designs, '-o', pdb_input]
        else:
            pdb_input = self.refined_pdb
            generate_files_cmd = []  # empty command

        if native:
            native = getattr(self, native, 'refined_pdb')
        else:
            native = self.refined_pdb

        # if isinstance(suffix, str):
        #     suffix = ['-out:suffix', '_%s' % suffix]
        # if isinstance(suffix, bool):
        if suffix:
            suffix = ['-out:suffix', f'_{script_name}']
        else:
            suffix = []

        if score_only:
            score = ['-out:file:score_only', self.scores_file]
        else:
            score = []

        if self.job.design.number_of_trajectories:
            trajectories = ['-nstruct', str(self.job.design.number_of_trajectories)]
        else:
            trajectories = ['-no_nstruct_label true']

        if variables:
            for idx, var_val in enumerate(variables):
                variable, value = var_val.split('=')
                variables[idx] = '%s=%s' % (variable, getattr(self.pose, value, ''))
            variables = ['-parser:script_vars'] + variables
        else:
            variables = []

        cmd += [f'@{flags_file}', f'-in:file:{"l" if file_list else "s"}', pdb_input, '-in:file:native', native] \
            + score + suffix + trajectories + ['-parser:protocol', script] + variables
        if self.job.mpi > 0:
            cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(self.job.mpi)] + cmd

        if self.job.distribute_work:
            write_shell_script(list2cmdline(generate_files_cmd), name=script_name, out_path=self.scripts,
                               additional=[list2cmdline(cmd)])
        else:
            raise NotImplementedError('Need to implement this feature')

        # Todo  + [list2cmdline(analysis_cmd)])
        #  analysis_cmd = ['python', putils.program_exe, putils.analysis, '--single', self.path, '--no-output',
        #                 f'--{flags.output_file}', os.path.join(self.job.all_scores, putils.default_analysis_file
        #                                                        .format(starttime, self.protocol))]

        # ANALYSIS: each output from the Design process based on score, Analyze Sequence Variation
        if not self.job.distribute_work:
            pose_s = self._interface_design_analysis()
            out_path = os.path.join(self.job.all_scores, putils.default_analysis_file.format(starttime, 'All'))
            if os.path.exists(out_path):
                header = False
            else:
                header = True
            pose_s.to_csv(out_path, mode='a', header=header)

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def interface_metrics(self):
        """Generate a script capable of running Rosetta interface metrics analysis on the bound and unbound states"""
        # metrics_flags = 'repack=yes'
        self.protocol = putils.interface_metrics
        main_cmd = copy(rosetta.script_cmd)
        # if self.interface_residue_numbers is False or self.interface_design_residue_numbers is False:
        self.identify_interface()
        # else:  # We only need to load pose as we already calculated interface
        #     self.load_pose()

        # interface_secondary_structure
        if not os.path.exists(self.flags) or self.job.force:
            self.prepare_rosetta_flags(out_dir=self.scripts)
            self.log.debug(f'Pose flags written to: {self.flags}')

        design_files = \
            os.path.join(self.scripts, f'design_files'
                                       f'{f"_{self.job.specific_protocol}" if self.job.specific_protocol else ""}.txt')
        generate_files_cmd = ['python', putils.list_pdb_files, '-d', self.designs, '-o', design_files] + \
            (['-s', self.job.specific_protocol] if self.job.specific_protocol else [])
        main_cmd += [f'@{self.flags}', '-in:file:l', design_files,
                     # TODO out:file:score_only file is not respected if out:path:score_file given
                     #  -run:score_only true?
                     '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true', '-parser:protocol']
        #              '-in:file:native', self.refined_pdb,
        if self.job.mpi > 0:
            main_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(self.job.mpi)] + main_cmd

        metric_cmd_bound = main_cmd + (['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []) + \
            [os.path.join(putils.rosetta_scripts_dir, f'{self.protocol}{"_DEV" if self.job.development else ""}.xml')]
        entity_cmd = main_cmd + [os.path.join(putils.rosetta_scripts_dir,
                                              f'metrics_entity{"_DEV" if self.job.development else ""}.xml')]
        metric_cmds = [metric_cmd_bound]
        metric_cmds.extend(self.generate_entity_metrics(entity_cmd))

        # Create executable to gather interface Metrics on all Designs
        if self.job.distribute_work:
            analysis_cmd = ['python', putils.program_exe, putils.analysis, '--single', self.path, '--no-output',
                            f'--{flags.output_file}', os.path.join(self.job.all_scores, putils.default_analysis_file
                                                                   .format(starttime, self.protocol))]
            write_shell_script(list2cmdline(generate_files_cmd), name=putils.interface_metrics, out_path=self.scripts,
                               additional=[list2cmdline(command) for command in metric_cmds] +
                                          [list2cmdline(analysis_cmd)])
        else:
            list_all_files_process = Popen(generate_files_cmd)
            list_all_files_process.communicate()
            for metric_cmd in metric_cmds:
                metrics_process = Popen(metric_cmd)
                metrics_process.communicate()  # wait for command to complete

        # ANALYSIS: each output from the Design process based on score, Analyze Sequence Variation
        if not self.job.distribute_work:
            pose_s = self._interface_design_analysis()
            out_path = os.path.join(self.job.all_scores, putils.default_analysis_file.format(starttime, 'All'))
            if os.path.exists(out_path):
                header = False
            else:
                header = True
            pose_s.to_csv(out_path, mode='a', header=header)

    # Below are protocols for various design applications
    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def check_unmodelled_clashes(self, clashing_threshold: float = 0.75):
        """Given a multimodel file, measure the number of clashes is less than a percentage threshold"""
        raise DesignError('This module is not working correctly at the moment')
        models = [Models.from_PDB(self.job.structure_db.full_models.retrieve_data(name=entity), log=self.log)
                  for entity in self.entity_names]
        # models = [Models.from_file(self.job.structure_db.full_models.retrieve_data(name=entity))
        #           for entity in self.entity_names]

        # for each model, transform to the correct space
        models = self.transform_structures_to_pose(models)
        multimodel = MultiModel.from_models(models, independent=True, log=self.log)

        clashes = 0
        prior_clashes = 0
        for idx, state in enumerate(multimodel, 1):
            clashes += (1 if state.is_clash() else 0)
            state.write(out_path=os.path.join(self.path, f'state_{idx}.pdb'))
            print(f'State {idx} - Clashes: {"YES" if clashes > prior_clashes else "NO"}')
            prior_clashes = clashes

        if clashes/float(len(multimodel)) > clashing_threshold:
            raise DesignError(f'The frequency of clashes ({clashes/float(len(multimodel))}) exceeds the clashing '
                              f'threshold ({clashing_threshold})')

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def check_clashes(self):
        """Check for clashes in the input and in the symmetric assembly if symmetric"""
        self.load_pose()

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def rename_chains(self):
        """Standardize the chain names in incremental order found in the design source file"""
        model = Model.from_file(self.source, log=self.log)
        model.rename_chains()
        model.write(out_path=self.asu_path)

    @handle_design_errors(errors=(DesignError, ValueError, RuntimeError))
    @close_logs
    @remove_structure_memory
    def orient(self, to_pose_directory: bool = True):
        """Orient the Pose with the prescribed symmetry at the origin and symmetry axes in canonical orientations
        self.symmetry is used to specify the orientation
        """
        if self.initial_model:
            model = self.initial_model
        else:
            model = Model.from_file(self.source, log=self.log)

        if self.design_symmetry:
            if to_pose_directory:
                out_path = self.assembly_path
            else:
                out_path = os.path.join(self.job.orient_dir, f'{model.name}.pdb')

            model.orient(symmetry=self.design_symmetry)

            orient_file = model.write(out_path=out_path)
            self.log.info(f'The oriented file was saved to {orient_file}')
            for entity in model.entities:
                entity.remove_mate_chains()
                self.entity_names.append(entity.name)

            # Load the pose and save the asu
            self.initial_model = model
            self.load_pose()  # entities=model.entities)
        else:
            raise SymmetryError(warn_missing_symmetry % self.orient.__name__)

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def refine(self):
        """Refine the source Pose"""
        self._refine()

    @handle_design_errors(errors=(DesignError, FileNotFoundError))
    @close_logs
    @remove_structure_memory
    def find_asu(self):
        """From a PDB with multiple Chains from multiple Entities, return the minimal configuration of Entities.
        ASU will only be a true ASU if the starting PDB contains a symmetric system, otherwise all manipulations find
        the minimal unit of Entities that are in contact
        """
        if self.symmetric:  # if the symmetry isn't known then this wouldn't be a great option
            if os.path.exists(self.assembly_path):
                self.load_pose(source=self.assembly_path)
            else:
                self.load_pose()
        else:
            raise NotImplementedError('Not sure if asu format matches pose.get_contacting_asu standard with no symmetry'
                                      '. This might cause issues')
            # Todo ensure asu format matches pose.get_contacting_asu standard
            # pdb = Model.from_file(self.source, log=self.log)
            # asu = pdb.return_asu()
            self.load_pose()
            # asu.update_attributes_from_pdb(pdb)

        # Save the Pose.asu
        self.save_asu()

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def expand_asu(self):
        """For the design info given by a PoseDirectory source, initialize the Pose with self.source file,
        self.symmetry, and self.log objects then expand the design given the provided symmetry operators and write to a
        file

        Reports on clash testing
        """
        self.load_pose()
        if self.symmetric:
            self.symmetric_assembly_is_clash()
            self.pose.write(assembly=True, out_path=self.assembly_path, increment_chains=self.job.increment_chains)
            self.log.info(f'Symmetric assembly written to: "{self.assembly_path}"')
        else:
            raise SymmetryError(warn_missing_symmetry % self.expand_asu.__name__)
        self.pickle_info()  # Todo remove once PoseDirectory state can be returned to the SymDesign dispatch w/ MP

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def generate_interface_fragments(self):
        """For the design info given by a PoseDirectory source, initialize the Pose then generate interfacial fragment
        information between Entities. Aware of symmetry and design_selectors in fragment generation file
        """
        # if self.interface_residue_numbers is False or self.interface_design_residue_numbers is False:
        self.identify_interface()
        # else:  # We only need to load pose as we already calculated interface
        #     self.load_pose()

        putils.make_path(self.frags, condition=self.job.write_fragments)
        self.pose.generate_interface_fragments()
        if self.job.write_fragments:
            self.pose.write_fragment_pairs(out_path=self.frags)
        self.fragment_observations = self.pose.get_fragment_observations()
        self.info['fragment_source'] = self.job.fragment_db.source
        self.pickle_info()  # Todo remove once PoseDirectory state can be returned to the SymDesign dispatch w/ MP

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def interface_design(self):
        """For the design info given by a PoseDirectory source, initialize the Pose then prepare all parameters for
        interfacial redesign between Pose Entities. Aware of symmetry, design_selectors, fragments, and
        evolutionary information in interface design
        """
        # if self.job.command_only and not self.job.distribute_work:  # Just reissue the commands
        #     pass
        # else:
        # if self.interface_residue_numbers is False or self.interface_design_residue_numbers is False:
        self.identify_interface()
        # else:  # We only need to load pose as we already calculated interface
        #     self.load_pose()

        putils.make_path(self.data)  # Todo consolidate this check with pickle_info()
        # Create all files which store the evolutionary_profile and/or fragment_profile -> design_profile
        if self.job.design.method == putils.rosetta_str:
            favor_fragments = evo_fill = True
        else:
            favor_fragments = evo_fill = False

        if self.job.generate_fragments:
            self.pose.generate_interface_fragments()
            self.fragment_observations = self.pose.get_fragment_observations()
            self.info['fragment_source'] = self.job.fragment_db.source
            if self.job.write_fragments:
                self.pose.write_fragment_pairs(out_path=self.frags)
            self.pose.calculate_fragment_profile(evo_fill=evo_fill)
        elif isinstance(self.fragment_observations, list):
            raise NotImplementedError(f"Can't put fragment observations taken away from the pose onto the pose due to "
                                      f"entities")
            self.pose.fragment_pairs = self.fragment_observations
            self.pose.calculate_fragment_profile(evo_fill=evo_fill)

        # elif os.path.exists(self.frag_file):
        #     self.retrieve_fragment_info_from_file()

        if self.job.design.evolution_constraint:
            for entity in self.pose.entities:
                if entity not in self.pose.active_entities:  # We shouldn't design, add a null profile instead
                    entity.add_profile(null=True)
                else:  # Add a real profile
                    # entity.add_profile(evolution=self.job.design.evolution_constraint,
                    #                    fragments=self.job.generate_fragments,
                    #                    out_dir=self.job.api_db.hhblits_profiles.location)
                    entity.sequence_file = self.job.api_db.sequences.retrieve_file(name=entity.name)
                    entity.evolutionary_profile = self.job.api_db.hhblits_profiles.retrieve_data(name=entity.name)
                    if not entity.evolutionary_profile:
                        entity.add_evolutionary_profile(out_dir=self.job.api_db.hhblits_profiles.location)
                    else:  # ensure the file is attached as well
                        entity.pssm_file = self.job.api_db.hhblits_profiles.retrieve_file(name=entity.name)

                    if not entity.pssm_file:  # still no file found. this is likely broken
                        raise DesignError(f'{entity.name} has no profile generated. To proceed with this design/'
                                          f'protocol you must generate the profile!')

                    if not entity.verify_evolutionary_profile():
                        entity.fit_evolutionary_profile_to_structure()

                    if not entity.sequence_file:
                        entity.write_sequence_to_fasta('reference', out_dir=self.job.api_db.sequences.location)

            self.pose.evolutionary_profile = \
                concatenate_profile([entity.evolutionary_profile for entity in self.pose.entities])

        # self.pose.combine_sequence_profiles()
        # I could also add the combined profile here instead of at each Entity
        self.pose.add_profile(evolution=self.job.design.evolution_constraint,
                              fragments=self.job.generate_fragments, favor_fragments=favor_fragments,
                              out_dir=self.job.api_db.hhblits_profiles.location)

        # -------------------------------------------------------------------------
        # Todo self.solve_consensus()
        # -------------------------------------------------------------------------

        if not self.pre_refine and not os.path.exists(self.refined_pdb):
            self._refine(metrics=False)

        putils.make_path(self.designs)
        # putils.make_path(self.data)
        match self.job.design.method:
            case putils.rosetta_str:
                # Write generated files
                self.pose.pssm_file = \
                    write_pssm_file(self.pose.evolutionary_profile, file_name=self.evolutionary_profile_file)
                write_pssm_file(self.pose.profile, file_name=self.design_profile_file)
                self.pose.fragment_profile.write(file_name=self.fragment_profile_file)
                self.rosetta_interface_design()  # Sets self.protocol
            case putils.proteinmpnn:
                self.proteinmpnn_interface_design()  # Sets self.protocol
            case other:
                raise ValueError(f"The method '{self.job.design.method}' isn't available")
        self.pickle_info()  # Todo remove once PoseDirectory state can be returned to the SymDesign dispatch w/ MP

    def rosetta_interface_design(self):
        """For the basic process of sequence design between two halves of an interface, write the necessary files for
        refinement (FastRelax), redesign (FastDesign), and metrics collection (Filters & SimpleMetrics)

        Stores job variables in a [stage]_flags file and the command in a [stage].sh file. Sets up dependencies based
        on the PoseDirectory
        """
        # Set up the command base (rosetta bin and database paths)
        main_cmd = copy(rosetta.script_cmd)
        main_cmd += ['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []
        # Todo must set up a blank -in:file:pssm in case the evolutionary matrix is not used. Design will fail!!
        profile_cmd = ['-in:file:pssm', self.evolutionary_profile_file] \
            if os.path.exists(self.evolutionary_profile_file) else []

        additional_cmds = out_file = []
        if self.job.design.scout:
            self.protocol = protocol_xml1 = putils.scout
            generate_files_cmd = null_cmd
            metrics_pdb = ['-in:file:s', self.scouted_pdb]
            # metrics_flags = 'repack=no'
            nstruct_instruct = ['-no_nstruct_label', 'true']
        else:
            design_files = os.path.join(self.scripts, f'design_files_{self.protocol}.txt')
            generate_files_cmd = \
                ['python', putils.list_pdb_files, '-d', self.designs, '-o', design_files, '-s', '_' + self.protocol]
            metrics_pdb = ['-in:file:l', design_files]
            # metrics_flags = 'repack=yes'
            if self.job.design.structure_background:
                self.protocol = protocol_xml1 = putils.structure_background
                nstruct_instruct = ['-nstruct', str(self.job.design.number_of_trajectories)]
            elif self.job.design.hbnet:  # Run hbnet_design_profile protocol
                self.protocol, protocol_xml1 = putils.hbnet_design_profile, 'hbnet_scout'
                nstruct_instruct = ['-no_nstruct_label', 'true']
                # Set up an additional command to perform interface design on hydrogen bond network from hbnet_scout
                additional_cmds = \
                    [[putils.hbnet_sort, os.path.join(self.data, 'hbnet_silent.o'),
                      str(self.job.design.number_of_trajectories)]] + main_cmd + profile_cmd \
                    + ['-in:file:silent', os.path.join(self.data, 'hbnet_selected.o'), f'@{self.flags}',
                       '-in:file:silent_struct_type', 'binary',
                       # '-out:suffix', f'_{self.protocol}',  adding no_nstruct_label true as only hbnet uses this mechanism
                       # hbnet_design_profile.xml could be just design_profile.xml
                       '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, f'{self.protocol}.xml')] \
                    + nstruct_instruct
                # Set up additional out_file
                out_file = ['-out:file:silent', os.path.join(self.data, 'hbnet_silent.o'),
                            '-out:file:silent_struct_type', 'binary']
                # silent_file = os.path.join(self.data, 'hbnet_silent.o')
                # additional_commands = \
                #     [
                #      # ['grep', '^SCORE', silent_file, '>', os.path.join(self.data, 'hbnet_scores.sc')],
                #      main_cmd + [os.path.join(self.data, 'hbnet_selected.o')]
                #      [os.path.join(self.data, 'hbnet_selected.tags')]
                #     ]
            else:  # Run the legacy protocol
                self.protocol = protocol_xml1 = putils.interface_design
                nstruct_instruct = ['-nstruct', str(self.job.design.number_of_trajectories)]

        # DESIGN: Prepare command and flags file
        if not os.path.exists(self.flags) or self.job.force:
            self.prepare_rosetta_flags(out_dir=self.scripts)
            self.log.debug(f'Pose flags written to: {self.flags}')

        if self.job.design.consensus:  # Todo add consensus sbatch generator to SymDesign main
            if self.job.generate_fragments:  # design_with_fragments
                consensus_cmd = main_cmd + rosetta.relax_flags_cmdline + \
                                [f'@{self.flags}', '-in:file:s', self.consensus_pdb,
                                 # '-in:file:native', self.refined_pdb,
                                 '-parser:protocol', os.path.join(putils.rosetta_scripts_dir,
                                                                  f'{putils.consensus}.xml'),
                                 '-parser:script_vars', f'switch={putils.consensus}']
                self.log.info(f'Consensus command: {list2cmdline(consensus_cmd)}')
                if self.job.distribute_work:
                    write_shell_script(list2cmdline(consensus_cmd), name=putils.consensus, out_path=self.scripts)
                else:
                    consensus_process = Popen(consensus_cmd)
                    consensus_process.communicate()
            else:
                self.log.critical(f'Cannot run consensus design without fragment info and none was found.'
                                  f' Did you mean to include --no-{flags.term_constraint}')
                #                   f'as {self.job.design.term_constraint}?')
        design_cmd = main_cmd + profile_cmd + \
            [f'@{self.flags}', '-in:file:s', self.scouted_pdb if os.path.exists(self.scouted_pdb) else self.refined_pdb,
             '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, f'{protocol_xml1}.xml'),
             '-out:suffix', f'_{self.protocol}'] + (['-overwrite'] if self.job.overwrite else []) \
            + out_file + nstruct_instruct

        # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
        # Assumes all entity chains are renamed from A to Z for entities (1 to n)
        entity_cmd = rosetta.script_cmd + metrics_pdb + \
            [f'@{self.flags}', '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true',
             '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, 'metrics_entity.xml')]

        if self.job.mpi > 0 and not self.job.design.scout:
            design_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(self.job.mpi)] + design_cmd
            entity_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(self.job.mpi)] + entity_cmd

        self.log.info(f'{self.rosetta_interface_design.__name__} command: {list2cmdline(design_cmd)}')
        metric_cmds = []
        metric_cmds.extend(self.generate_entity_metrics(entity_cmd))

        # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics
        if self.job.distribute_work:
            analysis_cmd = ['python', putils.program_exe, putils.analysis, '--single', self.path, '--no-output',
                            f'--{flags.output_file}', os.path.join(self.job.all_scores, putils.default_analysis_file
                                                                   .format(starttime, self.protocol))]
            write_shell_script(list2cmdline(design_cmd), name=self.protocol, out_path=self.scripts,
                               additional=[list2cmdline(command) for command in additional_cmds] +
                                          [list2cmdline(generate_files_cmd)] +
                                          [list2cmdline(command) for command in metric_cmds] +
                                          [list2cmdline(analysis_cmd)])
            #                  status_wrap=self.serialized_info,
        else:
            design_process = Popen(design_cmd)
            design_process.communicate()  # wait for command to complete
            list_all_files_process = Popen(generate_files_cmd)
            list_all_files_process.communicate()
            for metric_cmd in metric_cmds:
                metrics_process = Popen(metric_cmd)
                metrics_process.communicate()

        # ANALYSIS: each output from the Design process based on score, Analyze Sequence Variation
        if not self.job.distribute_work:
            pose_s = self._interface_design_analysis()
            out_path = os.path.join(self.job.all_scores, putils.default_analysis_file.format(starttime, 'All'))
            if os.path.exists(out_path):
                header = False
            else:
                header = True
            pose_s.to_csv(out_path, mode='a', header=header)

    def proteinmpnn_interface_design(self):
        self.protocol = 'proteinmpnn'
        sequences_and_scores: dict[str, np.ndarray | list] = \
            self.pose.design_sequences(number=self.job.design.number_of_trajectories,
                                       ca_only=self.job.design.ca_only,
                                       temperatures=self.job.design.temperatures,
                                       )
        design_names = [f'{self.name}_{self.protocol}{seq_idx:04d}'
                        for seq_idx in range(len(sequences_and_scores['sequences']))]
        putils.make_path(self.designs)
        self.output_proteinmpnn_scores(design_names, sequences_and_scores)
        putils.make_path(self.data)
        write_sequences(sequences_and_scores['sequences'], names=design_names, file_name=self.designed_sequences_file)
        if self.job.design.structures:
            self._predict_structure()

    def output_proteinmpnn_scores(self, design_ids: Sequence[str], sequences_and_scores: dict[str, np.ndarray | list]):
        """Given the results of a ProteinMPNN design trajectory, format the sequences and scores for the PoseDirectory

        Args:
            design_ids: The associated design identifier for each corresponding entry in sequences_and_scores
            sequences_and_scores: The mapping of ProteinMPNN score type to it's corresponding data
        """
        # Convert each numpy array into a list for output
        for score_type, data in sequences_and_scores.items():
            # if isinstance(data, np.ndarray):
            sequences_and_scores[score_type] = data.tolist()

        # trajectories_temperatures_ids = [f'temp{temperature}' for idx in self.job.design.number_of_trajectories
        #                                  for temperature in self.job.design.temperatures]
        # trajectories_temperatures_ids = [{'temperature': temperature} for idx in self.job.design.number_of_trajectories
        #                                  for temperature in self.job.design.temperatures]
        protocol = 'proteinmpnn'
        sequences_and_scores[putils.protocol] = \
            repeat(protocol, len(self.job.design.number_of_trajectories * self.job.design.temperatures))
        sequences_and_scores['temperature'] = [temperature for temperature in self.job.design.temperatures
                                               for _ in range(self.job.design.number_of_trajectories)]

        def write_per_residue_scores(_design_ids: Sequence[str], scores: dict[str, list]) -> AnyStr:
            """"""
            # Create an initial score dictionary
            design_scores = {design_id: {'decoy': design_id} for design_id in _design_ids}
            # For each score type unpack the data
            for score_type, score in scores.items():
                # For each score's data, update the dictionary of the corresponding design_id
                for design_id, design_score in zip(_design_ids, score):
                    design_scores[design_id].update({score_type: design_score})

            for design_id, scores in design_scores.items():
                # write_json(_scores, self.scores_file)
                with open(self.scores_file, 'a') as f_save:
                    json.dump(scores, f_save)  # , **kwargs)
                    # Ensure JSON lines are separated by newline
                    f_save.write('\n')

            # return write_json(design_scores, self.scores_file)
            return self.scores_file

        write_per_residue_scores(design_ids, sequences_and_scores)

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def optimize_designs(self, threshold: float = 0.):
        """To touch up and optimize a design, provide a list of optional directives to view mutational landscape around
        certain residues in the design as well as perform wild-type amino acid reversion to mutated residues

        Args:
            # residue_directives=None (dict[Residue | int, str]):
            #     {Residue object: 'mutational_directive', ...}
            # design_file=None (str): The name of a particular design file present in the designs output
            threshold: The threshold above which background amino acid frequencies are allowed for mutation
        """
        # Todo Notes for PROSS implementation
        #  I need to use a mover like FilterScan to measure all the energies for a particular residue and it's possible
        #  mutational space. Using these measurements, I then need to choose only those ones which make a particular
        #  energetic contribution to the structure and test these out using a FastDesign protocol where each is tried.
        #  This will likely utilize a resfile as in PROSS implementation and here as creating a PSSM could work but is a
        #  bit convoluted. I think finding the energy threshold to use as a filter cut off is going to be a bit
        #  heuristic as the REF2015 scorefunction wasn't used in PROSS publication.
        if self._lock_optimize_designs:
            self.log.critical('Need to resolve the differences between multiple specified_designs and a single '
                              'specified_design. Only using the first design')
            specific_design = self.specific_designs_file_paths[0]
            # raise NotImplemented('Need to resolve the differences between multiple specified_designs and a single '
            #                      'specified_design')
        else:
            raise RuntimeError('IMPOSSIBLE')
            specific_design = self.specific_design_path

        self.identify_interface()  # self.load_pose()
        # for design_path in self.specific_designs_file_paths
        #     self.load_pose(source=design_path)
        #     self.identify_interface()

        # format all amino acids in self.interface_design_residue_numbers with frequencies above the threshold to a set
        # Todo, make threshold and return set of strings a property of a profile object
        # Locate the desired background profile from the pose
        background_profile = getattr(self.pose, self.job.background_profile)
        raise NotImplementedError("background_profile doesn't account for residue.index versus residue.number")
        background = {residue: {protein_letters_1to3.get(aa) for aa in protein_letters_1to3
                                if background_profile[residue.number].get(aa, -1) > threshold}
                      for residue in self.pose.interface_residues}
        # include the wild-type residue from PoseDirectory Pose source and the residue identity of the selected design
        wt = {residue: {background_profile[residue.number].get('type'), protein_letters_3to1[residue.type]}
              for residue in background}
        directives = dict(zip(background.keys(), repeat(None)))
        # directives.update({self.pose.residue(residue_number): directive
        #                    for residue_number, directive in self.directives.items()})
        directives.update({residue: self.directives[residue.number]
                           for residue in self.pose.get_residues(self.directives.keys())})

        res_file = self.pose.make_resfile(directives, out_path=self.data, include=wt, background=background)

        self.protocol = protocol_xml1 = putils.optimize_designs
        # nstruct_instruct = ['-no_nstruct_label', 'true']
        nstruct_instruct = ['-nstruct', str(self.job.design.number_of_trajectories)]
        design_list_file = os.path.join(self.scripts, f'design_files_{self.protocol}.txt')
        generate_files_cmd = \
            ['python', putils.list_pdb_files, '-d', self.designs, '-o', design_list_file, '-s', '_' + self.protocol]

        main_cmd = copy(rosetta.script_cmd)
        main_cmd += ['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []
        if not os.path.exists(self.flags) or self.job.force:
            self.prepare_rosetta_flags(out_dir=self.scripts)
            self.log.debug(f'Pose flags written to: {self.flags}')

        # DESIGN: Prepare command and flags file
        # Todo must set up a blank -in:file:pssm in case the evolutionary matrix is not used. Design will fail!!
        profile_cmd = ['-in:file:pssm', self.evolutionary_profile_file] \
            if os.path.exists(self.evolutionary_profile_file) else []
        design_cmd = main_cmd + profile_cmd \
            + ['-in:file:s', specific_design if specific_design else self.refined_pdb,
               f'@{self.flags}', '-out:suffix', f'_{self.protocol}', '-packing:resfile', res_file,
               '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, f'{protocol_xml1}.xml')] + nstruct_instruct

        # metrics_pdb = ['-in:file:l', design_list_file]  # self.pdb_list]
        # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
        # Assumes all entity chains are renamed from A to Z for entities (1 to n)
        # metric_cmd = main_cmd + ['-in:file:s', self.specific_design if self.specific_design else self.refined_pdb] + \
        entity_cmd = main_cmd + ['-in:file:l', design_list_file] + \
            [f'@{self.flags}', '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true',
             '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, 'metrics_entity.xml')]

        if self.job.mpi > 0:
            design_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(self.job.mpi)] + design_cmd
            entity_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(self.job.mpi)] + entity_cmd

        self.log.info(f'{self.optimize_designs.__name__} command: {list2cmdline(design_cmd)}')
        metric_cmds = []
        metric_cmds.extend(self.generate_entity_metrics(entity_cmd))

        # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics
        if self.job.distribute_work:
            analysis_cmd = ['python', putils.program_exe, putils.analysis, '--single', self.path, '--no-output',
                            f'--{flags.output_file}', os.path.join(self.job.all_scores, putils.default_analysis_file
                                                                   .format(starttime, self.protocol))]
            write_shell_script(list2cmdline(design_cmd), name=self.protocol, out_path=self.scripts,
                               additional=[list2cmdline(generate_files_cmd)] +
                                          [list2cmdline(command) for command in metric_cmds] +
                                          [list2cmdline(analysis_cmd)])
        else:
            design_process = Popen(design_cmd)
            design_process.communicate()  # wait for command to complete
            list_all_files_process = Popen(generate_files_cmd)
            list_all_files_process.communicate()
            for metric_cmd in metric_cmds:
                metrics_process = Popen(metric_cmd)
                metrics_process.communicate()

        # ANALYSIS: each output from the Design process based on score, Analyze Sequence Variation
        if not self.job.distribute_work:
            pose_s = self._interface_design_analysis()
            out_path = os.path.join(self.job.all_scores, putils.default_analysis_file.format(starttime, 'All'))
            if os.path.exists(out_path):
                header = False
            else:
                header = True
            pose_s.to_csv(out_path, mode='a', header=header)

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def interface_design_analysis(self, design_poses: Iterable[Pose] = None) -> pd.Series:
        """Retrieve all score information from a PoseDirectory and write results to .csv file

        Args:
            design_poses: The subsequent designs to perform analysis on
        Returns:
            Series containing summary metrics for all designs in the design directory
        """
        return self._interface_design_analysis(design_poses=design_poses)

    def _interface_design_analysis(self, design_poses: Iterable[Pose] = None) -> pd.Series:
        """Retrieve all score information from a PoseDirectory and write results to .csv file

        Args:
            design_poses: The subsequent designs to perform analysis on
        Returns:
            Series containing summary metrics for all designs in the design directory
        """
        # if self.interface_residue_numbers is False or self.interface_design_residue_numbers is False:
        self.identify_interface()
        # else:  # We only need to load pose as we already calculated interface
        #     self.load_pose()

        if self.job.generate_fragments and not self.pose.fragment_queries:
            self.pose.generate_interface_fragments()
            if self.job.write_fragments:
                self.pose.write_fragment_pairs(out_path=self.frags)  # Todo PoseDirectory(.path)

        # Gather miscellaneous pose specific metrics
        other_pose_metrics = self.pose.interface_metrics()
        # CAUTION: Assumes each structure is the same length
        pose_length = self.pose.number_of_residues
        residue_numbers = list(range(1, pose_length + 1))

        # Find all designs files
        # Todo fold these into Model(s) and attack metrics from Pose objects?
        if design_poses is None:
            design_poses = [Pose.from_file(file, **self.pose_kwargs) for file in self.get_designs()]  # Todo PoseDirectory(.path)

        # Todo handle design sequences from a read_fasta_file?
        pose_sequences = {putils.pose_source: self.pose.sequence}
        # Todo implement reference sequence from included file(s) or as with self.pose.sequence below
        pose_sequences.update({putils.reference_name: self.pose.sequence})
        pose_sequences.update({pose.name: pose.sequence for pose in design_poses})
        all_mutations = generate_mutations_from_reference(self.pose.sequence, pose_sequences, return_to=True)
        # generate_mutations_from_reference(''.join(self.pose.atom_sequences.values()), pose_sequences)

        entity_energies = [0. for _ in self.pose.entities]
        pose_source_residue_info = \
            {residue.number: {'complex': 0., 'bound': copy(entity_energies), 'unbound': copy(entity_energies),
                              'solv_complex': 0., 'solv_bound': copy(entity_energies),
                              'solv_unbound': copy(entity_energies), 'fsp': 0., 'cst': 0.,
                              'type': protein_letters_3to1.get(residue.type), 'hbond': 0}
             for entity in self.pose.entities for residue in entity.residues}
        residue_info = {putils.pose_source: pose_source_residue_info}
        job_key = 'no_energy'
        stat_s, sim_series = pd.Series(dtype=float), []
        if os.path.exists(self.scores_file):  # Rosetta scores file is present  # Todo PoseDirectory(.path)
            self.log.debug(f'Found design scores in file: {self.scores_file}')  # Todo PoseDirectory(.path)
            design_was_performed = True
            # Get the scores from the score file on design trajectory metrics
            source_df = pd.DataFrame({putils.pose_source: {putils.protocol: job_key}}).T
            for idx, entity in enumerate(self.pose.entities, 1):
                source_df[f'buns_{idx}_unbound'] = 0
                source_df[f'interface_energy_{idx}_bound'] = 0
                source_df[f'interface_energy_{idx}_unbound'] = 0
                source_df[f'solvation_energy_{idx}_bound'] = 0
                source_df[f'solvation_energy_{idx}_unbound'] = 0
                source_df[f'interface_connectivity_{idx}'] = 0
            source_df['buns_complex'] = 0
            # source_df['buns_unbound'] = 0
            source_df['contact_count'] = 0
            source_df['favor_residue_energy'] = 0
            # Used in sum_per_residue_df
            # source_df['interface_energy_complex'] = 0
            source_df['interaction_energy_complex'] = 0
            source_df['interaction_energy_per_residue'] = \
                source_df['interaction_energy_complex'] / len(self.pose.interface_residues)
            source_df['interface_separation'] = 0
            source_df['number_hbonds'] = 0
            source_df['rmsd_complex'] = 0  # Todo calculate this here instead of Rosetta using superposition3d
            source_df['rosetta_reference_energy'] = 0
            source_df['shape_complementarity'] = 0
            source_df['solvation_energy'] = 0
            source_df['solvation_energy_complex'] = 0
            design_scores = read_scores(self.scores_file)  # Todo PoseDirectory(.path)
            self.log.debug(f'All designs with scores: {", ".join(design_scores.keys())}')
            # Remove designs with scores but no structures
            all_viable_design_scores = {}
            for pose in design_poses:
                try:
                    all_viable_design_scores[pose.name] = design_scores.pop(pose.name)
                except KeyError:  # structure wasn't scored, we will remove this later
                    pass

            # Todo these need to be reconciled with taking the rosetta complex and unbound energies
            proteinmpnn_scores = ['sequences', 'complex_sequence_loss', 'unbound_sequence_loss']
            # Create protocol dataframe
            scores_df = pd.DataFrame(all_viable_design_scores).T
            scores_df = pd.concat([source_df, scores_df])
            # Gather all columns into specific types for processing and formatting
            per_res_columns, hbonds_columns = [], []
            proteinmpnn_columns = []
            for column in scores_df.columns.to_list():
                if column.startswith('per_res_'):
                    per_res_columns.append(column)
                elif column.startswith('hbonds_res_selection'):
                    hbonds_columns.append(column)
                elif column in proteinmpnn_scores:
                    proteinmpnn_columns.append(column)

            if proteinmpnn_columns:
                proteinmpnn_df = scores_df.loc[:, proteinmpnn_columns]

            # Check proper input
            metric_set = necessary_metrics.difference(set(scores_df.columns))
            # self.log.debug('Score columns present before required metric check: %s' % scores_df.columns.to_list())
            if metric_set:
                raise DesignError(f'Missing required metrics: "{", ".join(metric_set)}"')

            # Remove unnecessary (old scores) as well as Rosetta pose score terms besides ref (has been renamed above)
            # TODO learn know how to produce score terms in output score file. Not in FastRelax...
            remove_columns = per_res_columns + hbonds_columns + rosetta_terms + unnecessary + proteinmpnn_columns
            # TODO remove dirty when columns are correct (after P432)
            #  and column tabulation precedes residue/hbond_processing
            interface_hbonds = dirty_hbond_processing(all_viable_design_scores)
            # can't use hbond_processing (clean) in the case there is a design without metrics... columns not found!
            # interface_hbonds = hbond_processing(all_viable_design_scores, hbonds_columns)
            number_hbonds_s = \
                pd.Series({design: len(hbonds) for design, hbonds in interface_hbonds.items()}, name='number_hbonds')
            # number_hbonds_s = pd.Series({design: len(hbonds) for design, hbonds in interface_hbonds.items()})  #, name='number_hbonds')
            # scores_df = pd.merge(scores_df, number_hbonds_s, left_index=True, right_index=True)
            scores_df.loc[number_hbonds_s.index, 'number_hbonds'] = number_hbonds_s
            # scores_df = scores_df.assign(number_hbonds=number_hbonds_s)
            # residue_info = {'energy': {'complex': 0., 'unbound': 0.}, 'type': None, 'hbond': 0}
            residue_info.update(self.pose.rosetta_residue_processing(all_viable_design_scores))
            residue_info = process_residue_info(residue_info, hbonds=interface_hbonds)
            residue_info = incorporate_mutation_info(residue_info, all_mutations)
            # can't use residue_processing (clean) in the case there is a design without metrics... columns not found!
            # residue_info.update(residue_processing(all_viable_design_scores, simplify_mutation_dict(all_mutations),
            #                                        per_res_columns, hbonds=interface_hbonds))

            # Drop designs where required data isn't present
            # Format protocol columns
            missing_group_indices = scores_df[putils.protocol].isna()
            # Todo remove not DEV
            scout_indices = [idx for idx in scores_df[missing_group_indices].index if 'scout' in idx]
            scores_df.loc[scout_indices, putils.protocol] = putils.scout
            structure_bkgnd_indices = [idx for idx in scores_df[missing_group_indices].index if 'no_constraint' in idx]
            scores_df.loc[structure_bkgnd_indices, putils.protocol] = putils.structure_background
            # Todo Done remove
            # protocol_s.replace({'combo_profile': putils.design_profile}, inplace=True)  # ensure proper profile name

            scores_df.drop(missing_group_indices, axis=0, inplace=True, errors='ignore')
            # protocol_s.drop(missing_group_indices, inplace=True, errors='ignore')

            viable_designs = scores_df.index.to_list()
            if not viable_designs:
                raise DesignError('No viable designs remain after processing!')

            self.log.debug(f'Viable designs remaining after cleaning:\n\t{", ".join(viable_designs)}')
            pose_sequences = {design: sequence for design, sequence in pose_sequences.items() if
                              design in viable_designs}

            # Todo implement this protocol if sequence data is taken at multiple points along a trajectory and the
            #  sequence data along trajectory is a metric on it's own
            # # Gather mutations for residue specific processing and design sequences
            # for design, data in list(all_viable_design_scores.items()):  # make a copy as can be removed
            #     sequence = data.get('final_sequence')
            #     if sequence:
            #         if len(sequence) >= pose_length:
            #             pose_sequences[design] = sequence[:pose_length]  # Todo won't work if design had insertions
            #         else:
            #             pose_sequences[design] = sequence
            #     else:
            #         self.log.warning('Design %s is missing sequence data, removing from design pool' % design)
            #         all_viable_design_scores.pop(design)
            # # format {entity: {design_name: sequence, ...}, ...}
            # entity_sequences = \
            #     {entity: {design: sequence[entity.n_terminal_residue.number - 1:entity.c_terminal_residue.number]
            #               for design, sequence in pose_sequences.items()} for entity in self.pose.entities}
        else:
            self.log.debug(f'Missing design scores file at {self.scores_file}')  # Todo PoseDirectory(.path)
            design_was_performed = False
            # Todo add relevant missing scores such as those specified as 0 below
            # Todo may need to put source_df in scores file alternative
            source_df = pd.DataFrame({putils.pose_source: {putils.protocol: job_key}}).T
            scores_df = pd.DataFrame({pose.name: {putils.protocol: job_key} for pose in design_poses}).T
            scores_df = pd.concat([source_df, scores_df])
            for idx, entity in enumerate(self.pose.entities, 1):
                source_df[f'buns_{idx}_unbound'] = 0
                source_df[f'interface_energy_{idx}_bound'] = 0
                source_df[f'interface_energy_{idx}_unbound'] = 0
                source_df[f'solvation_energy_{idx}_bound'] = 0
                source_df[f'solvation_energy_{idx}_unbound'] = 0
                source_df[f'interface_connectivity_{idx}'] = 0
                # residue_info = {'energy': {'complex': 0., 'unbound': 0.}, 'type': None, 'hbond': 0}
                # design_info.update({residue.number: {'energy_delta': 0.,
                #                                      'type': protein_letters_3to1.get(residue.type),
                #                                      'hbond': 0} for residue in entity.residues})
            source_df['buns_complex'] = 0
            # source_df['buns_unbound'] = 0
            scores_df['contact_count'] = 0
            scores_df['favor_residue_energy'] = 0
            scores_df['interface_energy_complex'] = 0
            scores_df['interaction_energy_complex'] = 0
            scores_df['interaction_energy_per_residue'] = \
                scores_df['interaction_energy_complex'] / len(self.pose.interface_residues)
            scores_df['interface_separation'] = 0
            scores_df['number_hbonds'] = 0
            scores_df['rmsd_complex'] = 0  # Todo calculate this here instead of Rosetta using superposition3d
            scores_df['rosetta_reference_energy'] = 0
            scores_df['shape_complementarity'] = 0
            scores_df['solvation_energy'] = 0
            scores_df['solvation_energy_complex'] = 0
            remove_columns = rosetta_terms + unnecessary
            residue_info.update({struct_name: pose_source_residue_info for struct_name in scores_df.index.to_list()})
            # Todo generate energy scores internally which matches output from residue_processing
            # interface_hbonds = dirty_hbond_processing(design_scores)
            # residue_info = self.pose.rosetta_residue_processing(design_scores)
            # residue_info = process_residue_info(residue_info, simplify_mutation_dict(all_mutations),
            #                                     hbonds=interface_hbonds)
            viable_designs = [pose.name for pose in design_poses]

        scores_df.drop(remove_columns, axis=1, inplace=True, errors='ignore')
        other_pose_metrics['observations'] = len(viable_designs)

        entity_sequences = []
        for entity in self.pose.entities:
            entity_slice = slice(entity.n_terminal_residue.index, 1+entity.c_terminal_residue.index)
            entity_sequences.append({design: sequence[entity_slice] for design, sequence in pose_sequences.items()})
        # Todo generate_multiple_mutations accounts for offsets from the reference sequence. Not necessary YET
        # sequence_mutations = \
        #     generate_multiple_mutations(self.pose.atom_sequences, pose_sequences, pose_num=False)
        # sequence_mutations.pop('reference')
        # entity_sequences = generate_sequences(self.pose.atom_sequences, sequence_mutations)
        # entity_sequences = {chain: keys_from_trajectory_number(named_sequences)
        #                         for chain, named_sequences in entity_sequences.items()}

        # Find protocols for protocol specific data processing removing from scores_df
        protocol_s = scores_df.pop(putils.protocol).copy()
        designs_by_protocol = protocol_s.groupby(protocol_s).groups
        # remove refine and consensus if present as there was no design done over multiple protocols
        unique_protocols = list(designs_by_protocol.keys())
        # Todo change if we did multiple rounds of these protocols
        designs_by_protocol.pop(putils.refine, None)
        designs_by_protocol.pop(putils.consensus, None)
        # Get unique protocols
        unique_design_protocols = set(designs_by_protocol.keys())
        self.log.info(f'Unique Design Protocols: {", ".join(unique_design_protocols)}')

        # Replace empty strings with np.nan and convert remaining to float
        scores_df.replace('', np.nan, inplace=True)
        scores_df.fillna(dict(zip(protocol_specific_columns, repeat(0))), inplace=True)
        scores_df = scores_df.astype(float)  # , copy=False, errors='ignore')

        # per residue data includes every residue in the pose
        # per_residue_data = {'errat_deviation': {}, 'hydrophobic_collapse': {}, 'contact_order': {},
        #                     'sasa_hydrophobic_complex': {}, 'sasa_polar_complex': {}, 'sasa_relative_complex': {},
        #                     'sasa_hydrophobic_bound': {}, 'sasa_polar_bound': {}, 'sasa_relative_bound': {}}
        interface_local_density = {putils.pose_source: self.pose.local_density_interface()}
        # atomic_deviation = {}
        # pose_assembly_minimally_contacting = self.pose.assembly_minimally_contacting
        # perform SASA measurements
        # pose_assembly_minimally_contacting.get_sasa()
        # assembly_asu_residues = pose_assembly_minimally_contacting.residues[:pose_length]
        # per_residue_data['sasa_hydrophobic_complex'][putils.pose_source] = \
        #     [residue.sasa_apolar for residue in assembly_asu_residues]
        # per_residue_data['sasa_polar_complex'][putils.pose_source] = [residue.sasa_polar for residue in assembly_asu_residues]
        # per_residue_data['sasa_relative_complex'][putils.pose_source] = \
        #     [residue.relative_sasa for residue in assembly_asu_residues]

        # Grab metrics for the pose source. Checks if self.pose was designed
        # Favor pose source errat/collapse on a per-entity basis if design occurred
        # As the pose source assumes no legit interface present while designs have an interface
        # per_residue_sasa_unbound_apolar, per_residue_sasa_unbound_polar, per_residue_sasa_unbound_relative = [], [], []
        # source_errat_accuracy, source_errat, source_contact_order, inverse_residue_contact_order_z = [], [], [], []
        source_contact_order = []
        for idx, entity in enumerate(self.pose.entities):
            # Contact order is the same for every design in the Pose and not dependent on pose
            source_contact_order.append(entity.contact_order)

        per_residue_data: dict[str, dict[str, Any]] = \
            {putils.pose_source: self.pose.get_per_residue_interface_metrics()}
        pose_source_contact_order_s = \
            pd.Series(np.concatenate(source_contact_order), index=residue_numbers, name='contact_order')
        per_residue_data[putils.pose_source]['contact_order'] = pose_source_contact_order_s

        number_of_entities = self.pose.number_of_entities
        if design_was_performed:  # The input structure was not meant to be together, treat as such
            source_errat = []
            for idx, entity in enumerate(self.pose.entities):
                # Replace 'errat_deviation' measurement with uncomplexed entities
                # oligomer_errat_accuracy, oligomeric_errat = entity_oligomer.errat(out_path=os.path.devnull)
                # source_errat_accuracy.append(oligomer_errat_accuracy)
                # Todo when Entity.oligomer works
                #  _, oligomeric_errat = entity.oligomer.errat(out_path=os.path.devnull)
                entity_oligomer = Model.from_chains(entity.chains, log=self.pose.log, entities=False)
                _, oligomeric_errat = entity_oligomer.errat(out_path=os.path.devnull)
                source_errat.append(oligomeric_errat[:entity.number_of_residues])
            # atomic_deviation[putils.pose_source] = sum(source_errat_accuracy) / float(number_of_entities)
            pose_source_errat_s = pd.Series(np.concatenate(source_errat), index=residue_numbers)
        else:
            pose_assembly_minimally_contacting = self.pose.assembly_minimally_contacting
            # atomic_deviation[putils.pose_source], pose_per_residue_errat = \
            _, pose_per_residue_errat = \
                pose_assembly_minimally_contacting.errat(out_path=os.path.devnull)
            pose_source_errat_s = pose_per_residue_errat[:pose_length]

        per_residue_data[putils.pose_source]['errat_deviation'] = pose_source_errat_s

        # Compute structural measurements for all designs
        for pose in design_poses:  # Takes 1-2 seconds for Structure -> assembly -> errat
            # Must find interface residues before measure local_density
            pose.find_and_split_interface()
            per_residue_data[pose.name] = pose.get_per_residue_interface_metrics()
            # Todo remove Rosetta
            #  This is a measurement of interface_connectivity like from Rosetta
            interface_local_density[pose.name] = pose.local_density_interface()

        if proteinmpnn_columns:
            for pose in design_poses:
                per_residue_data[pose.name].update({
                    'complex_sequence_loss': proteinmpnn_df.loc[pose.name, 'complex_sequence_loss'],
                    'unbound_sequence_loss': proteinmpnn_df.loc[pose.name, 'unbound_sequence_loss']
                })

        # Calculate hydrophobic collapse for each design
        # warn = True
        # # Add Entity information to the Pose
        # for idx, entity in enumerate(self.pose.entities):
        #     try:  # To fetch the multiple sequence alignment for further processing
        #         entity.msa = self.job.api_db.alignments.retrieve_data(name=entity.name)
        #     except ValueError:  # When the Entity reference sequence and alignment are different lengths
        #         if not warn:
        #             self.log.info(f'Metrics relying on a multiple sequence alignment are not being collected as '
        #                           f'there is no MSA found. These include: '
        #                           f'{", ".join(multiple_sequence_alignment_dependent_metrics)}')
        #             warn = False
        measure_evolution = measure_alignment = True
        warn = False
        # Add Entity information to the Pose
        for idx, entity in enumerate(self.pose.entities):
            # entity.sequence_file = job.api_db.sequences.retrieve_file(name=entity.name)
            # if not entity.sequence_file:
            #     entity.write_sequence_to_fasta('reference', out_dir=job.sequences)
            #     # entity.add_evolutionary_profile(out_dir=job.api_db.hhblits_profiles.location)
            # else:
            profile = self.job.api_db.hhblits_profiles.retrieve_data(name=entity.name)
            if not profile:
                measure_evolution = False
                warn = True
            else:
                entity.evolutionary_profile = profile

            if not entity.verify_evolutionary_profile():
                entity.fit_evolutionary_profile_to_structure()

            try:  # To fetch the multiple sequence alignment for further processing
                msa = self.job.api_db.alignments.retrieve_data(name=entity.name)
                if not msa:
                    measure_alignment = False
                    warn = True
                else:
                    entity.msa = msa
            except ValueError as error:  # When the Entity reference sequence and alignment are different lengths
                self.pose.log.info(f'Entity reference sequence and provided alignment are different lengths: {error}')
                warn = True

        if warn:
            if not measure_evolution and not measure_alignment:
                self.pose.log.info(f'Metrics relying on multiple sequence alignment data are not being collected as'
                                   f' there were none found. These include: '
                                   f'{", ".join(multiple_sequence_alignment_dependent_metrics)}')
            elif not measure_alignment:
                self.pose.log.info(f'Metrics relying on a multiple sequence alignment are not being collected as '
                                   f'there was no MSA found. These include: '
                                   f'{", ".join(multiple_sequence_alignment_dependent_metrics)}')
            else:
                self.pose.log.info(f'Metrics relying on an evolutionary profile are not being collected as '
                                   f'there was no profile found. These include: '
                                   f'{", ".join(profile_dependent_metrics)}')
        # Include the pose_source in the measured designs
        contact_order_per_res_z, reference_collapse, collapse_profile = self.pose.get_folding_metrics()
        folding_and_collapse = calculate_collapse_metrics(list(zip(*[list(designed_sequences.values())
                                                                     for designed_sequences in entity_sequences])),
                                                          contact_order_per_res_z, reference_collapse, collapse_profile)
        # pose_collapse_df = pd.DataFrame(folding_and_collapse).T
        per_residue_collapse_df = pd.concat({design_id: pd.DataFrame(data, index=residue_numbers)
                                             for design_id, data in zip(viable_designs, folding_and_collapse)},
                                            ).unstack().swaplevel(0, 1, axis=1)

        # Convert per_residue_data into a dataframe matching residue_df orientation
        per_residue_df = pd.concat({name: pd.DataFrame(data, index=residue_numbers)
                                    for name, data in per_residue_data.items()}).unstack().swaplevel(0, 1, axis=1)
        # Fill in contact order for each design
        per_residue_df.fillna(per_residue_df.loc[putils.pose_source, idx_slice[:, 'contact_order']], inplace=True)
        per_residue_df = per_residue_df.join(per_residue_collapse_df)

        # Process mutational frequencies, H-bond, and Residue energy metrics to dataframe
        residue_df = pd.concat({design: pd.DataFrame(info) for design, info in residue_info.items()}).unstack()
        # returns multi-index column with residue number as first (top) column index, metric as second index
        # during residue_df unstack, all residues with missing dicts are copied as nan
        # Merge interface design specific residue metrics with total per residue metrics
        index_residues = [residue.number for residue in self.pose.interface_residues]
        per_residue_df = pd.merge(residue_df.loc[:, idx_slice[index_residues, :]],
                                  per_residue_df.loc[:, idx_slice[index_residues, :]],
                                  left_index=True, right_index=True)

        # Load profiles of interest into the analysis
        if measure_evolution:
            pose.evolutionary_profile = concatenate_profile([entity.evolutionary_profile for entity in pose.entities])

        # pose.generate_interface_fragments() was already called
        pose.calculate_fragment_profile()
        pose.add_profile(evolution=self.job.design.evolution_constraint,
                         fragments=self.job.generate_fragments)

        profile_background = {'design': pssm_as_array(self.pose.profile),
                              'evolution': pssm_as_array(self.pose.evolutionary_profile),
                              'fragment': self.pose.fragment_profile.as_array()}
        # if self.pose.profile:
        # else:
        #     self.log.info('Pose has no profile information')
        # if self.pose.evolutionary_profile:
        # else:
        #     self.log.info('No evolution information')
        # if self.pose.fragment_profile:
        # else:
        #     self.log.info('No fragment information')
        if self.job.fragment_db is not None:
            interface_bkgd = np.array(list(self.job.fragment_db.aa_frequencies.values()))
            profile_background['interface'] = np.tile(interface_bkgd, (self.pose.number_of_residues, 1))

        if not profile_background:
            divergence_s = pd.Series(dtype=float)
        else:  # Calculate sequence statistics
            # First, for entire pose
            interface_indexer = [residue.index for residue in self.pose.interface_residues]
            pose_alignment = MultipleSequenceAlignment.from_dictionary(pose_sequences)
            observed, divergence = \
                calculate_sequence_observations_and_divergence(pose_alignment, profile_background, interface_indexer)

            observed_dfs = []
            for profile, observed_values in observed.items():
                # scores_df[f'observed_{profile}'] = observed_values.mean(axis=1)
                observed_dfs.append(pd.DataFrame(data=observed_values, index=pose_sequences,  # design_obs_freqs.keys()
                                                 columns=pd.MultiIndex.from_product([residue_numbers,
                                                                                     [f'observed_{profile}']]))
                                    )
            # Add observation information into the per_residue_df
            per_residue_df = pd.concat([per_residue_df] + observed_dfs, axis=1)
            # Get pose sequence divergence
            pose_divergence_s = pd.concat([pd.Series({f'{divergence_type}_per_residue': _divergence.mean()
                                                      for divergence_type, _divergence in divergence.items()})],
                                          keys=[('sequence_design', 'pose')])
            # pose_divergence_s = pd.Series({f'{divergence_type}_per_residue': per_res_metric(stat)
            #                                for divergence_type, stat in divergence.items()},
            #                               name=('sequence_design', 'pose'))
            if designs_by_protocol:  # were multiple designs generated with each protocol?
                # find the divergence within each protocol
                divergence_by_protocol = {protocol: {} for protocol in designs_by_protocol}
                for protocol, designs in designs_by_protocol.items():
                    # Todo select from pose_alignment the indices of each design then pass to MultipleSequenceAlignment?
                    # protocol_alignment = \
                    #     MultipleSequenceAlignment.from_dictionary({design: pose_sequences[design]
                    #                                                for design in designs})
                    protocol_alignment = MultipleSequenceAlignment.from_dictionary({design: pose_sequences[design]
                                                                                    for design in designs})
                    # protocol_mutation_freq = filter_dictionary_keys(protocol_alignment.frequencies,
                    #                                                 self.pose.interface_residues)
                    # protocol_mutation_freq = protocol_alignment.frequencies
                    protocol_divergence = {f'divergence_{profile}':
                                           position_specific_divergence(protocol_alignment.frequencies,
                                                                        bgd)[interface_indexer]
                                           for profile, bgd in profile_background.items()}
                    # if interface_bkgd is not None:
                    #     protocol_divergence['divergence_interface'] = \
                    #         position_specific_divergence(protocol_alignment.frequencies[interface_indexer],
                    #                                      tiled_int_background)
                    # Get per residue divergence metric by protocol
                    divergence_by_protocol[protocol] = {f'{divergence_type}_per_residue': divergence.mean()
                                                        for divergence_type, divergence in protocol_divergence.items()}
                # new = dfd.columns.to_frame()
                # new.insert(0, 'new2_level_name', new_level_values)
                # dfd.columns = pd.MultiIndex.from_frame(new)
                protocol_divergence_s = \
                    pd.concat([pd.DataFrame(divergence_by_protocol).unstack()], keys=['sequence_design'])
            else:
                protocol_divergence_s = pd.Series(dtype=float)
            divergence_s = pd.concat([protocol_divergence_s, pose_divergence_s])

        scores_df['number_of_mutations'] = \
            pd.Series({design: len(mutations) for design, mutations in all_mutations.items()})
        scores_df['percent_mutations'] = \
            scores_df['number_of_mutations'] / other_pose_metrics['pose_length']
        # residue_indices_per_entity = self.pose.residue_indices_per_entity
        is_thermophilic = []
        idx = 1
        for idx, entity in enumerate(self.pose.entities, idx):
            pose_c_terminal_residue_number = entity.c_terminal_residue.index + 1
            scores_df[f'entity_{idx}_number_of_mutations'] = \
                pd.Series(
                    {design: len([1 for mutation_idx in mutations if mutation_idx < pose_c_terminal_residue_number])
                     for design, mutations in all_mutations.items()})
            scores_df[f'entity_{idx}_percent_mutations'] = \
                scores_df[f'entity_{idx}_number_of_mutations'] / other_pose_metrics[f'entity_{idx}_number_of_residues']
            is_thermophilic.append(getattr(other_pose_metrics, f'entity_{idx}_thermophile', 0))

        # Get the average thermophilicity for all entities
        other_pose_metrics['pose_thermophilicity'] = sum(is_thermophilic) / idx

        # entity_alignment = multi_chain_alignment(entity_sequences)
        # INSTEAD OF USING BELOW, split Pose.MultipleSequenceAlignment at entity.chain_break...
        # entity_alignments = \
        #     {idx: MultipleSequenceAlignment.from_dictionary(designed_sequences)
        #      for idx, designed_sequences in entity_sequences.items()}
        # entity_alignments = \
        #     {idx: msa_from_dictionary(designed_sequences) for idx, designed_sequences in entity_sequences.items()}
        # pose_collapse_ = pd.concat(pd.DataFrame(folding_and_collapse), axis=1, keys=[('sequence_design', 'pose')])
        dca_design_residues_concat = []
        dca_succeed = True
        # dca_background_energies, dca_design_energies = [], []
        dca_background_energies, dca_design_energies = {}, {}
        for idx, entity in enumerate(self.pose.entities):
            try:  # TODO add these to the analysis
                entity.h_fields = self.job.api_db.bmdca_fields.retrieve_data(name=entity.name)
                entity.j_couplings = self.job.api_db.bmdca_couplings.retrieve_data(name=entity.name)
                dca_background_residue_energies = entity.direct_coupling_analysis()
                # Todo INSTEAD OF USING BELOW, split Pose.MultipleSequenceAlignment at entity.chain_break...
                entity_alignment = MultipleSequenceAlignment.from_dictionary(entity_sequences[idx])
                # entity_alignment = msa_from_dictionary(entity_sequences[idx])
                dca_design_residue_energies = entity.direct_coupling_analysis(msa=entity_alignment)
                dca_design_residues_concat.append(dca_design_residue_energies)
                # dca_background_energies.append(dca_background_energies.sum(axis=1))
                # dca_design_energies.append(dca_design_energies.sum(axis=1))
                dca_background_energies[entity] = dca_background_residue_energies.sum(axis=1)  # turns data to 1D
                dca_design_energies[entity] = dca_design_residue_energies.sum(axis=1)
            except AttributeError:
                self.log.warning(f"For {entity.name}, DCA analysis couldn't be performed. "
                                 f"Missing required parameter files")
                dca_succeed = False

        if dca_succeed:
            # concatenate along columns, adding residue index to column, design name to row
            dca_concatenated_df = pd.DataFrame(np.concatenate(dca_design_residues_concat, axis=1),
                                               index=list(pose_sequences.keys()), columns=residue_numbers)
            # get all design names                                    ^
            # dca_concatenated_df.columns = pd.MultiIndex.from_product([dca_concatenated_df.columns, ['dca_energy']])
            dca_concatenated_df = pd.concat([dca_concatenated_df], keys=['dca_energy'], axis=1).swaplevel(0, 1, axis=1)
            # merge with per_residue_df
            per_residue_df = pd.merge(per_residue_df, dca_concatenated_df, left_index=True, right_index=True)

        # per_residue_df = pd.merge(residue_df, per_residue_df.loc[:, idx_slice[residue_df.columns.levels[0], :]],
        #                       left_index=True, right_index=True)
        # Add local_density information to scores_df
        # scores_df['interface_local_density'] = \
        #     per_residue_df.loc[:, idx_slice[self.interface_residue_numbers, 'local_density']].mean(axis=1)

        if self.job.design.structures:
            scores_df['interface_local_density'] = pd.Series(interface_local_density)
            # Make buried surface area (bsa) columns, and residue classification
            per_residue_df = calculate_residue_surface_area(per_residue_df)  # .loc[:, idx_slice[index_residues, :]])
            # # Make buried surface area (bsa) columns
            # per_residue_df = calculate_residue_surface_area(per_residue_df.loc[:, idx_slice[index_residues, :]])

        # Calculate new metrics from combinations of other metrics
        # Add design residue information to scores_df such as how many core, rim, and support residues were measured
        summed_scores_df = sum_per_residue_metrics(per_residue_df)  # .loc[:, idx_slice[index_residues, :]])
        scores_df = scores_df.join(summed_scores_df)

        if self.job.design.structures:
            scores_df['interface_area_total'] = bsa_assembly_df = \
                scores_df['interface_area_polar'] + scores_df['interface_area_hydrophobic']
            # Find the proportion of the residue surface area that is solvent accessible versus buried in the interface
            scores_df['interface_area_to_residue_surface_ratio'] = \
                (bsa_assembly_df / (bsa_assembly_df+scores_df['sasa_total_complex']))  # / scores_df['total_interface_residues']

            # scores_df['interface_area_polar'] = per_residue_df.loc[:, idx_slice[index_residues, 'bsa_polar']].sum(axis=1)
            # scores_df['interface_area_hydrophobic'] = \
            #     per_residue_df.loc[:, idx_slice[index_residues, 'bsa_hydrophobic']].sum(axis=1)
            # # scores_df['interface_area_total'] = \
            # #     per_residue_df.loc[not_pose_source_indices, idx_slice[index_residues, 'bsa_total']].sum(axis=1)
            # scores_df['interface_area_total'] = scores_df['interface_area_polar'] + scores_df['interface_area_hydrophobic']
            #
            # Make scores_df errat_deviation that takes into account the pose_source sequence errat_deviation
            # This overwrites the sum_per_residue_metrics() value
            # Include in errat_deviation if errat score is < 2 std devs and isn't 0 to begin with
            source_errat_inclusion_boolean = np.logical_and(pose_source_errat_s < errat_2_sigma, pose_source_errat_s != 0.)
            errat_df = per_residue_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)
            # find where designs deviate above wild-type errat scores
            errat_sig_df = errat_df.sub(pose_source_errat_s, axis=1) > errat_1_sigma  # axis=1 Series is column oriented
            # then select only those residues which are expressly important by the inclusion boolean
            scores_df['errat_deviation'] = (errat_sig_df.loc[:, source_errat_inclusion_boolean] * 1).sum(axis=1)

        scores_drop_columns = ['hydrophobic_collapse', 'sasa_relative_bound', 'sasa_relative_complex']
        scores_df = scores_df.drop(scores_drop_columns, errors='ignore', axis=1)
        scores_df = scores_df.rename(columns={'type': 'sequence'})

        # Find the proportion of the residue surface area that is solvent accessible versus buried in the interface
        sasa_assembly_df = per_residue_df.loc[:, idx_slice[index_residues, 'sasa_total_complex']].droplevel(-1, axis=1)
        bsa_assembly_df = per_residue_df.loc[:, idx_slice[index_residues, 'bsa_total']].droplevel(-1, axis=1)
        total_surface_area_df = sasa_assembly_df + bsa_assembly_df
        # ratio_df = bsa_assembly_df / total_surface_area_df
        scores_df['interface_area_to_residue_surface_ratio'] = (bsa_assembly_df / total_surface_area_df).mean(axis=1)

        # Check if any columns are > 50% interior (value can be 0 or 1). If so, return True for that column
        # interior_residue_df = per_residue_df.loc[:, idx_slice[:, 'interior']]
        # interior_residue_numbers = \
        #     interior_residues.loc[:, interior_residues.mean(axis=0) > 0.5].columns.remove_unused_levels().levels[0].
        #     to_list()
        # if interior_residue_numbers:
        #     self.log.info(f'Design Residues {",".join(map(str, sorted(interior_residue_numbers)))}
        #                   'are located in the interior')

        # This shouldn't be much different from the state variable self.interface_residue_numbers
        # perhaps the use of residue neighbor energy metrics adds residues which contribute, but not directly
        # interface_residues = set(per_residue_df.columns.levels[0].unique()).difference(interior_residue_numbers)

        # Add design residue information to scores_df such as how many core, rim, and support residues were measured
        # for residue_class in residue_classification:
        #     scores_df[residue_class] = per_residue_df.loc[:, idx_slice[:, residue_class]].sum(axis=1)

        # Calculate new metrics from combinations of other metrics
        scores_columns = scores_df.columns.to_list()
        self.log.debug(f'Metrics present: {scores_columns}')
        # sum columns using list[0] + list[1] + list[n]
        # complex_df = per_residue_df.loc[:, idx_slice[:, 'complex']]
        # bound_df = per_residue_df.loc[:, idx_slice[:, 'bound']]
        # unbound_df = per_residue_df.loc[:, idx_slice[:, 'unbound']]
        # solvation_complex_df = per_residue_df.loc[:, idx_slice[:, 'solv_complex']]
        # solvation_bound_df = per_residue_df.loc[:, idx_slice[:, 'solv_bound']]
        # solvation_unbound_df = per_residue_df.loc[:, idx_slice[:, 'solv_unbound']]
        # # scores_df['interface_energy_complex'] = complex_df.sum(axis=1)
        # scores_df['interface_energy_bound'] = bound_df.sum(axis=1)
        # scores_df['interface_energy_unbound'] = unbound_df.sum(axis=1)
        # scores_df['interface_solvation_energy_complex'] = solvation_complex_df.sum(axis=1)
        # scores_df['interface_solvation_energy_bound'] = solvation_bound_df.sum(axis=1)
        # scores_df['interface_solvation_energy_unbound'] = solvation_unbound_df.sum(axis=1)
        # per_residue_df = per_residue_df.drop([column
        #                                       for columns in [complex_df.columns, bound_df.columns, unbound_df.columns,
        #                                                       solvation_complex_df.columns, solvation_bound_df.columns,
        #                                                       solvation_unbound_df.columns]
        #                                       for column in columns], axis=1)
        summation_pairs = \
            {'buns_unbound': list(filter(re.compile('buns_[0-9]+_unbound$').match, scores_columns)),  # Rosetta
             # 'interface_energy_bound':
             #     list(filter(re.compile('interface_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
             # 'interface_energy_unbound':
             #     list(filter(re.compile('interface_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
             # 'interface_solvation_energy_bound':
             #     list(filter(re.compile('solvation_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
             # 'interface_solvation_energy_unbound':
             #     list(filter(re.compile('solvation_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
             'interface_connectivity':
                 list(filter(re.compile('interface_connectivity_[0-9]+').match, scores_columns)),  # Rosetta
             }
        # 'sasa_hydrophobic_bound':
        #     list(filter(re.compile('sasa_hydrophobic_[0-9]+_bound').match, scores_columns)),
        # 'sasa_polar_bound': list(filter(re.compile('sasa_polar_[0-9]+_bound').match, scores_columns)),
        # 'sasa_total_bound': list(filter(re.compile('sasa_total_[0-9]+_bound').match, scores_columns))}
        scores_df = columns_to_new_column(scores_df, summation_pairs)
        scores_df = columns_to_new_column(scores_df, delta_pairs, mode='sub')
        # add total_interface_residues for div_pairs and int_comp_similarity
        scores_df['total_interface_residues'] = other_pose_metrics.pop('total_interface_residues')
        scores_df = columns_to_new_column(scores_df, division_pairs, mode='truediv')
        scores_df['interface_composition_similarity'] = scores_df.apply(interface_composition_similarity, axis=1)
        scores_df.drop(clean_up_intermediate_columns, axis=1, inplace=True, errors='ignore')
        repacking = scores_df.get('repacking')
        if repacking is not None:
            # set interface_bound_activation_energy = NaN where repacking is 0
            # Currently is -1 for True (Rosetta Filter quirk...)
            scores_df.loc[scores_df[repacking == 0].index, 'interface_bound_activation_energy'] = np.nan
            scores_df.drop('repacking', axis=1, inplace=True)

        # Process dataframes for missing values and drop refine trajectory if present
        # refine_index = scores_df[scores_df[putils.protocol] == putils.refine].index
        # scores_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
        # per_residue_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
        # residue_info.pop(putils.refine, None)  # Remove refine from analysis
        # residues_no_frags = per_residue_df.columns[per_residue_df.isna().all(axis=0)].remove_unused_levels().levels[0]
        per_residue_df.dropna(how='all', inplace=True, axis=1)  # remove completely empty columns such as obs_interface
        per_residue_df = per_residue_df.fillna(0.).copy()
        # residue_indices_no_frags = per_residue_df.columns[per_residue_df.isna().all(axis=0)]

        # POSE ANALYSIS
        # scores_df = pd.concat([scores_df, proteinmpnn_df], axis=1)
        scores_df.dropna(how='all', inplace=True, axis=1)  # remove completely empty columns
        # refine is not considered sequence design and destroys mean. remove v
        # trajectory_df = scores_df.sort_index().drop(putils.refine, axis=0, errors='ignore')
        # consensus cst_weights are very large and destroy the mean.
        # remove this drop for consensus or refine if they are run multiple times
        trajectory_df = \
            scores_df.drop([putils.pose_source, putils.refine, putils.consensus], axis=0, errors='ignore').sort_index()

        # Get total design statistics for every sequence in the pose and every protocol specifically
        scores_df[putils.protocol] = protocol_s
        protocol_groups = scores_df.groupby(putils.protocol)
        # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        # print(trajectory_df.select_dtypes(exclude=numerics))

        pose_stats, protocol_stats = [], []
        for idx, stat in enumerate(stats_metrics):
            # Todo both groupby calls have this warning
            #  FutureWarning: The default value of numeric_only in DataFrame.mean is deprecated. In a future version,
            #  it will default to False. In addition, specifying 'numeric_only=None' is deprecated. Select only valid
            #  columns or specify the value of numeric_only to silence this warning.
            pose_stats.append(getattr(trajectory_df, stat)().rename(stat))
            protocol_stats.append(getattr(protocol_groups, stat)())

        # format stats_s for final pose_s Series
        protocol_stats[stats_metrics.index(mean)]['observations'] = protocol_groups.size()
        protocol_stats_s = pd.concat([stat_df.T.unstack() for stat_df in protocol_stats], keys=stats_metrics)
        pose_stats_s = pd.concat(pose_stats, keys=list(zip(stats_metrics, repeat('pose'))))
        stat_s = pd.concat([protocol_stats_s.dropna(), pose_stats_s.dropna()])  # dropna removes NaN metrics

        # change statistic names for all df that are not groupby means for the final trajectory dataframe
        for idx, stat in enumerate(stats_metrics):
            if stat != mean:
                protocol_stats[idx] = protocol_stats[idx].rename(index={protocol: f'{protocol}_{stat}'
                                                                        for protocol in unique_design_protocols})
        # trajectory_df = pd.concat([trajectory_df, pd.concat(pose_stats, axis=1).T] + protocol_stats)
        # remove std rows if there is no stdev
        number_of_trajectories = len(trajectory_df) + len(protocol_groups) + 1  # 1 for the mean
        final_trajectory_indices = trajectory_df.index.to_list() + unique_protocols + [mean]
        trajectory_df = pd.concat([trajectory_df]
                                  + [df.dropna(how='all', axis=0) for df in protocol_stats]  # v don't add if nothing
                                  + [pd.to_numeric(s).to_frame().T for s in pose_stats if not all(s.isna())])
        # this concat ^ puts back putils.pose_source, refine, consensus designs since protocol_stats is calculated on scores_df
        # add all docking and pose information to each trajectory, dropping the pose observations
        interface_metrics_s = pd.Series(other_pose_metrics)
        pose_metrics_df = pd.concat([interface_metrics_s] * number_of_trajectories, axis=1).T
        trajectory_df = pd.concat([trajectory_df,
                                   pose_metrics_df.rename(index=dict(zip(range(number_of_trajectories),
                                                                         final_trajectory_indices)))
                                  .drop(['observations'], axis=1)], axis=1)
        trajectory_df = trajectory_df.fillna({'observations': 1})

        # Calculate protocol significance
        pvalue_df = pd.DataFrame()
        scout_protocols = list(filter(re.compile(f'.*{putils.scout}').match,
                                      protocol_s[~protocol_s.isna()].unique().tolist()))
        similarity_protocols = unique_design_protocols.difference([putils.refine, job_key] + scout_protocols)
        if putils.structure_background not in unique_design_protocols:
            self.log.info(f'Missing background protocol "{putils.structure_background}". No protocol significance '
                          f'measurements available for this pose')
        elif len(similarity_protocols) == 1:  # measure significance
            self.log.info("Can't measure protocol significance, only one protocol of interest")
        else:  # Test significance between all combinations of protocols by grabbing mean entries per protocol
            for prot1, prot2 in combinations(sorted(similarity_protocols), 2):
                select_df = \
                    trajectory_df.loc[[design for designs in [designs_by_protocol[prot1], designs_by_protocol[prot2]]
                                       for design in designs], significance_columns]
                # prot1/2 pull out means from trajectory_df by using the protocol name
                difference_s = \
                    trajectory_df.loc[prot1, significance_columns].sub(trajectory_df.loc[prot2, significance_columns])
                pvalue_df[(prot1, prot2)] = df_permutation_test(select_df, difference_s, compare='mean',
                                                                group1_size=len(designs_by_protocol[prot1]))
            pvalue_df = pvalue_df.T  # transpose significance pairs to indices and significance metrics to columns
            trajectory_df = pd.concat([trajectory_df, pd.concat([pvalue_df], keys=['similarity']).swaplevel(0, 1)])

            # Compute residue energy/sequence differences between each protocol
            residue_energy_df = per_residue_df.loc[:, idx_slice[:, 'energy_delta']]

            scaler = skl.preprocessing.StandardScaler()
            res_pca = skl.decomposition.PCA(variance)  # P432 designs used 0.8 percent of the variance
            residue_energy_np = scaler.fit_transform(residue_energy_df.values)
            residue_energy_pc = res_pca.fit_transform(residue_energy_np)

            seq_pca = skl.decomposition.PCA(variance)
            designed_sequence_modifications = [''.join(info['type'] for residue_number, info in residues_info.items()
                                                       if residue_number in index_residues)
                                               for design, residues_info in residue_info.items()]
            pairwise_sequence_diff_np = scaler.fit_transform(all_vs_all(designed_sequence_modifications,
                                                                        sequence_difference))
            seq_pc = seq_pca.fit_transform(pairwise_sequence_diff_np)
            # Make principal components (PC) DataFrame
            residue_energy_pc_df = pd.DataFrame(residue_energy_pc, index=residue_energy_df.index,
                                                columns=[f'pc{idx}' for idx in range(1, len(res_pca.components_) + 1)])
            seq_pc_df = pd.DataFrame(seq_pc, index=list(residue_info.keys()),
                                     columns=[f'pc{idx}' for idx in range(1, len(seq_pca.components_) + 1)])
            # Compute the euclidean distance
            # pairwise_pca_distance_np = pdist(seq_pc)
            # pairwise_pca_distance_np = SDUtils.all_vs_all(seq_pc, euclidean)

            # Merge PC DataFrames with labels
            # seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)
            seq_pc_df[putils.protocol] = protocol_s
            # residue_energy_pc_df = pd.merge(protocol_s, residue_energy_pc_df, left_index=True, right_index=True)
            residue_energy_pc_df[putils.protocol] = protocol_s
            # Next group the labels
            sequence_groups = seq_pc_df.groupby(putils.protocol)
            residue_energy_groups = residue_energy_pc_df.groupby(putils.protocol)
            # Measure statistics for each group
            # All protocol means have pairwise distance measured to access similarity
            # Gather protocol similarity/distance metrics
            sim_measures = {'sequence_distance': {}, 'energy_distance': {}}
            # sim_stdev = {}  # 'similarity': None, 'seq_distance': None, 'energy_distance': None}
            # grouped_pc_seq_df_dict, grouped_pc_energy_df_dict, similarity_stat_dict = {}, {}, {}
            for stat in stats_metrics:
                grouped_pc_seq_df = getattr(sequence_groups, stat)()
                grouped_pc_energy_df = getattr(residue_energy_groups, stat)()
                similarity_stat = getattr(pvalue_df, stat)(axis=1)  # protocol pair : stat Series
                if stat == mean:
                    # for each measurement in residue_energy_pc_df, need to take the distance between it and the
                    # structure background mean (if structure background, is the mean is useful too?)
                    background_distance = \
                        cdist(residue_energy_pc,
                              grouped_pc_energy_df.loc[putils.structure_background, :].values[np.newaxis, :])
                    trajectory_df = \
                        pd.concat([trajectory_df,
                                   pd.Series(background_distance.flatten(), index=residue_energy_pc_df.index,
                                             name=f'energy_distance_from_{putils.structure_background}_mean')], axis=1)

                    # if renaming is necessary
                    # protocol_stats_s[stat].index = protocol_stats_s[stat].index.to_series().map(
                    #     {protocol: protocol + '_' + stat for protocol in sorted(unique_design_protocols)})
                    # find the pairwise distance from every point to every other point
                    seq_pca_mean_distance_vector = pdist(grouped_pc_seq_df)
                    energy_pca_mean_distance_vector = pdist(grouped_pc_energy_df)
                    # protocol_indices_map = list(tuple(condensed_to_square(k, len(seq_pca_mean_distance_vector)))
                    #                             for k in seq_pca_mean_distance_vector)
                    # find similarity between each protocol by taking row average of all p-values for each metric
                    # mean_pvalue_s = pvalue_df.mean(axis=1)  # protocol pair : mean significance Series
                    # mean_pvalue_s.index = pd.MultiIndex.from_tuples(mean_pvalue_s.index)
                    # sim_measures['similarity'] = mean_pvalue_s
                    similarity_stat.index = pd.MultiIndex.from_tuples(similarity_stat.index)
                    sim_measures['similarity'] = similarity_stat

                    # for vector_idx, seq_dist in enumerate(seq_pca_mean_distance_vector):
                    #     i, j = condensed_to_square(vector_idx, len(grouped_pc_seq_df.index))
                    #     sim_measures['sequence_distance'][(grouped_pc_seq_df.index[i],
                    #                                        grouped_pc_seq_df.index[j])] = seq_dist

                    for vector_idx, (seq_dist, energy_dist) in enumerate(zip(seq_pca_mean_distance_vector,
                                                                             energy_pca_mean_distance_vector)):
                        i, j = condensed_to_square(vector_idx, len(grouped_pc_energy_df.index))
                        sim_measures['sequence_distance'][(grouped_pc_seq_df.index[i],
                                                           grouped_pc_seq_df.index[j])] = seq_dist
                        sim_measures['energy_distance'][(grouped_pc_energy_df.index[i],
                                                         grouped_pc_energy_df.index[j])] = energy_dist
                elif stat == std:
                    # sim_stdev['similarity'] = similarity_stat_dict[stat]
                    pass
                    # # Todo need to square each pc, add them up, divide by the group number, then take the sqrt
                    # sim_stdev['sequence_distance'] = grouped_pc_seq_df
                    # sim_stdev['energy_distance'] = grouped_pc_energy_df

            # Find the significance between each pair of protocols
            protocol_sig_s = pd.concat([pvalue_df.loc[[pair], :].squeeze() for pair in pvalue_df.index.to_list()],
                                       keys=[tuple(pair) for pair in pvalue_df.index.to_list()])
            # squeeze turns the column headers into series indices. Keys appends to make a multi-index

            # Find total protocol similarity for different metrics
            # for measure, values in sim_measures.items():
            #     # measure_s = pd.Series({pair: similarity for pair, similarity in values.items()})
            #     # measure_s = pd.Series(values)
            #     similarity_sum['protocol_%s_sum' % measure] = pd.Series(values).sum()
            similarity_sum = {f'protocol_{measure}_sum': pd.Series(values).sum()
                              for measure, values in sim_measures.items()}
            similarity_sum_s = pd.concat([pd.Series(similarity_sum)], keys=[('sequence_design', 'pose')])

            # Process similarity between protocols
            sim_measures_s = pd.concat([pd.Series(values) for values in sim_measures.values()],
                                       keys=list(sim_measures.keys()))
            # # Todo test
            # sim_stdev_s = pd.concat(list(sim_stdev.values()),
            #                         keys=list(zip(repeat('std'), sim_stdev.keys()))).swaplevel(1, 2)
            # sim_series = [protocol_sig_s, similarity_sum_s, sim_measures_s, sim_stdev_s]
            sim_series = [protocol_sig_s, similarity_sum_s, sim_measures_s]

            # if self.job.figures:  # Todo ensure output is as expected then move below
            #     protocols_by_design = {design: protocol for protocol, designs in designs_by_protocol.items()
            #                            for design in designs}
            #     _path = os.path.join(self.job.all_scores, str(self))
            #     # Set up Labels & Plot the PC data
            #     protocol_map = {protocol: i for i, protocol in enumerate(designs_by_protocol)}
            #     integer_map = {i: protocol for (protocol, i) in protocol_map.items()}
            #     pc_labels_group = [protocols_by_design[design] for design in residue_info]
            #     # pc_labels_group = np.array([protocols_by_design[design] for design in residue_info])
            #     pc_labels_int = [protocol_map[protocols_by_design[design]] for design in residue_info]
            #     fig = plt.figure()
            #     # ax = fig.add_subplot(111, projection='3d')
            #     ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
            #     # plt.cla()
            #
            #     # for color_int, label in integer_map.items():  # zip(pc_labels_group, pc_labels_int):
            #     #     ax.scatter(seq_pc[pc_labels_group == label, 0],
            #     #                seq_pc[pc_labels_group == label, 1],
            #     #                seq_pc[pc_labels_group == label, 2],
            #     #                c=color_int, cmap=plt.cm.nipy_spectral, edgecolor='k')
            #     scatter = ax.scatter(seq_pc[:, 0], seq_pc[:, 1], seq_pc[:, 2], c=pc_labels_int, cmap='Spectral',
            #                          edgecolor='k')
            #     # handles, labels = scatter.legend_elements()
            #     # # print(labels)  # ['$\\mathdefault{0}$', '$\\mathdefault{1}$', '$\\mathdefault{2}$']
            #     # ax.legend(handles, labels, loc='upper right', title=protocol)
            #     # # ax.legend(handles, [integer_map[label] for label in labels], loc="upper right", title=protocol)
            #     # # plt.axis('equal') # not possible with 3D graphs
            #     # plt.legend()  # No handles with labels found to put in legend.
            #     colors = [scatter.cmap(scatter.norm(i)) for i in integer_map.keys()]
            #     custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20)
            #                     for c in colors]
            #     ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left',
            #               bbox_to_anchor=(1.0, .5))
            #     # # Add group mean to the plot
            #     # for name, label in integer_map.items():
            #     #     ax.scatter(seq_pc[pc_labels_group == label, 0].mean(),
            #     #                seq_pc[pc_labels_group == label, 1].mean(),
            #     #                seq_pc[pc_labels_group == label, 2].mean(), marker='x')
            #     ax.set_xlabel('PC1')
            #     ax.set_ylabel('PC2')
            #     ax.set_zlabel('PC3')
            #     # plt.legend(pc_labels_group)
            #     plt.savefig('%s_seq_pca.png' % _path)
            #     plt.clf()
            #     # Residue PCA Figure to assay multiple interface states
            #     fig = plt.figure()
            #     # ax = fig.add_subplot(111, projection='3d')
            #     ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
            #     scatter = ax.scatter(residue_energy_pc[:, 0], residue_energy_pc[:, 1], residue_energy_pc[:, 2],
            #                          c=pc_labels_int,
            #                          cmap='Spectral', edgecolor='k')
            #     colors = [scatter.cmap(scatter.norm(i)) for i in integer_map.keys()]
            #     custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20) for c in
            #                     colors]
            #     ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left',
            #               bbox_to_anchor=(1.0, .5))
            #     ax.set_xlabel('PC1')
            #     ax.set_ylabel('PC2')
            #     ax.set_zlabel('PC3')
            #     plt.savefig('%s_res_energy_pca.png' % _path)

        # Format output and save Trajectory, Residue DataFrames, and PDB Sequences
        if self.job.save:
            trajectory_df.sort_index(inplace=True, axis=1)
            per_residue_df = per_residue_df.loc[:, idx_slice[index_residues, :]]
            per_residue_df.sort_index(inplace=True)
            per_residue_df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
            per_residue_df[(putils.protocol, putils.protocol)] = protocol_s
            # per_residue_df.sort_index(inplace=True, key=lambda x: x.str.isdigit())  # put wt entry first
            putils.make_path(self.job.all_scores)
            if self.job.merge:
                trajectory_df = pd.concat([trajectory_df], axis=1, keys=['metrics'])
                trajectory_df = pd.merge(trajectory_df, per_residue_df, left_index=True, right_index=True)
            else:
                per_residue_df.to_csv(self.residues)  # Todo PoseDirectory(.path)
            trajectory_df.to_csv(self.trajectories)  # Todo PoseDirectory(.path)
            # pickle_object(pose_sequences, self.designed_sequences_file, out_path='')  # Todo PoseDirectory(.path)
            write_sequences(pose_sequences, file_name=self.designed_sequences_file)

        # Create figures
        if self.job.figures:  # for plotting collapse profile, errat data, contact order
            # Plot: Format the collapse data with residues as index and each design as column
            # collapse_graph_df = pd.DataFrame(per_residue_data['hydrophobic_collapse'])
            collapse_graph_df = per_residue_df.loc[:, idx_slice[:, 'hydrophobic_collapse']].droplevel(-1, axis=1)
            reference_collapse = [entity.hydrophobic_collapse for entity in self.pose.entities]
            reference_collapse_concatenated_s = \
                pd.Series(np.concatenate(reference_collapse), name=putils.reference_name)
            collapse_graph_df[putils.reference_name] = reference_collapse_concatenated_s
            # collapse_graph_df.columns += 1  # offset index to residue numbering
            # collapse_graph_df.sort_index(axis=1, inplace=True)
            # graph_collapse = sns.lineplot(data=collapse_graph_df)
            # g = sns.FacetGrid(tip_sumstats, col="sex", row="smoker")
            # graph_collapse = sns.relplot(data=collapse_graph_df, kind='line')  # x='Residue Number'

            # Set the base figure aspect ratio for all sequence designs
            figure_aspect_ratio = (pose_length / 25., 20)  # 20 is arbitrary size to fit all information in figure
            color_cycler = cycler(color=large_color_array)
            plt.rc('axes', prop_cycle=color_cycler)
            fig = plt.figure(figsize=figure_aspect_ratio)
            # legend_fill_value = int(15 * pose_length / 100)

            # collapse_ax, contact_ax, errat_ax = fig.subplots(3, 1, sharex=True)
            collapse_ax, errat_ax = fig.subplots(2, 1, sharex=True)
            # add the contact order to a new plot
            contact_ax = collapse_ax.twinx()
            contact_ax.plot(pose_source_contact_order_s, label='Contact Order',
                            color='#fbc0cb', lw=1, linestyle='-')  # pink
            # contact_ax.scatter(residue_indices, pose_source_contact_order_s, color='#fbc0cb', marker='o')  # pink
            # wt_contact_order_concatenated_min_s = pose_source_contact_order_s.min()
            # wt_contact_order_concatenated_max_s = pose_source_contact_order_s.max()
            # wt_contact_order_range = wt_contact_order_concatenated_max_s - wt_contact_order_concatenated_min_s
            # scaled_contact_order = ((pose_source_contact_order_s - wt_contact_order_concatenated_min_s)
            #                         / wt_contact_order_range)  # / wt_contact_order_range)
            # graph_contact_order = sns.relplot(data=errat_graph_df, kind='line')  # x='Residue Number'
            # collapse_ax1.plot(scaled_contact_order)
            # contact_ax.vlines(self.pose.chain_breaks, 0, 1, transform=contact_ax.get_xaxis_transform(),
            #                   label='Entity Breaks', colors='#cccccc')  # , grey)
            # contact_ax.vlines(index_residues, 0, 0.05, transform=contact_ax.get_xaxis_transform(),
            #                   label='Design Residues', colors='#f89938', lw=2)  # , orange)
            contact_ax.set_ylabel('Contact Order')
            # contact_ax.set_xlim(0, pose_length)
            contact_ax.set_ylim(0, None)
            # contact_ax.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse+contact.png'))
            # collapse_ax1.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse+contact.png'))

            # Get the plot of each collapse profile into a matplotlib axes
            # collapse_ax = collapse_graph_df.plot.line(legend=False, ax=collapse_ax, figsize=figure_aspect_ratio)
            # collapse_ax = collapse_graph_df.plot.line(legend=False, ax=collapse_ax)
            collapse_ax.plot(collapse_graph_df.T.values, label=collapse_graph_df.index)
            # collapse_ax = collapse_graph_df.plot.line(ax=collapse_ax)
            collapse_ax.xaxis.set_major_locator(MultipleLocator(20))
            collapse_ax.xaxis.set_major_formatter('{x:.0f}')
            # For the minor ticks, use no labels; default NullFormatter.
            collapse_ax.xaxis.set_minor_locator(MultipleLocator(5))
            collapse_ax.set_xlim(0, pose_length)
            collapse_ax.set_ylim(0, 1)
            # # CAN'T SET FacetGrid object for most matplotlib elements...
            # ax = graph_collapse.axes
            # ax = plt.gca()  # gca <- get current axis
            # labels = [fill(index, legend_fill_value) for index in collapse_graph_df.index]
            # collapse_ax.legend(labels, loc='lower left', bbox_to_anchor=(0., 1))
            # collapse_ax.legend(loc='lower left', bbox_to_anchor=(0., 1))
            # Plot the chain break(s) and design residues
            # linestyles={'solid', 'dashed', 'dashdot', 'dotted'}
            collapse_ax.vlines(self.pose.chain_breaks, 0, 1, transform=collapse_ax.get_xaxis_transform(),
                               label='Entity Breaks', colors='#cccccc')  # , grey)
            collapse_ax.vlines(index_residues, 0, 0.05, transform=collapse_ax.get_xaxis_transform(),
                               label='Design Residues', colors='#f89938', lw=2)  # , orange)
            # Plot horizontal significance
            collapse_ax.hlines([collapse_significance_threshold], 0, 1, transform=collapse_ax.get_yaxis_transform(),
                               label='Collapse Threshold', colors='#fc554f', linestyle='dotted')  # tomato
            # collapse_ax.set_xlabel('Residue Number')
            collapse_ax.set_ylabel('Hydrophobic Collapse Index')
            # collapse_ax.set_prop_cycle(color_cycler)
            # ax.autoscale(True)
            # collapse_ax.figure.tight_layout()  # no standardization
            # collapse_ax.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse.png'))  # no standardization

            # Plot: Collapse description of total profile against each design
            entity_collapse_mean, entity_collapse_std = [], []
            for entity in self.pose.entities:
                if entity.msa:
                    collapse = entity.collapse_profile()
                    entity_collapse_mean.append(collapse.mean(axis=-2))
                    entity_collapse_std.append(collapse.std(axis=-2))
                else:
                    break
            else:  # Only execute if we successfully looped
                profile_mean_collapse_concatenated_s = \
                    pd.concat([entity_collapse_mean[idx] for idx in range(number_of_entities)], ignore_index=True)
                profile_std_collapse_concatenated_s = \
                    pd.concat([entity_collapse_std[idx] for idx in range(number_of_entities)], ignore_index=True)
                profile_mean_collapse_concatenated_s.index += 1  # offset index to residue numbering
                profile_std_collapse_concatenated_s.index += 1  # offset index to residue numbering
                collapse_graph_describe_df = pd.DataFrame({
                    'std_min': profile_mean_collapse_concatenated_s - profile_std_collapse_concatenated_s,
                    'std_max': profile_mean_collapse_concatenated_s + profile_std_collapse_concatenated_s,
                })
                collapse_graph_describe_df.index += 1  # offset index to residue numbering
                collapse_graph_describe_df['Residue Number'] = collapse_graph_describe_df.index
                collapse_ax.vlines('Residue Number', 'std_min', 'std_max', data=collapse_graph_describe_df,
                                   color='#e6e6fa', linestyle='-', lw=1, alpha=0.8)  # lavender
                # collapse_ax.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse_versus_profile.png'))

            # Plot: Errat Accuracy
            # errat_graph_df = pd.DataFrame(per_residue_data['errat_deviation'])
            # errat_graph_df = per_residue_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)
            # errat_graph_df = errat_df
            # wt_errat_concatenated_s = pd.Series(np.concatenate(list(source_errat.values())), name='clean_asu')
            # errat_graph_df[putils.pose_source] = pose_source_errat_s
            # errat_graph_df.columns += 1  # offset index to residue numbering
            errat_df.sort_index(axis=0, inplace=True)
            # errat_ax = errat_graph_df.plot.line(legend=False, ax=errat_ax, figsize=figure_aspect_ratio)
            # errat_ax = errat_graph_df.plot.line(legend=False, ax=errat_ax)
            # errat_ax = errat_graph_df.plot.line(ax=errat_ax)
            errat_ax.plot(errat_df.T.values, label=collapse_graph_df.index)
            errat_ax.xaxis.set_major_locator(MultipleLocator(20))
            errat_ax.xaxis.set_major_formatter('{x:.0f}')
            # For the minor ticks, use no labels; default NullFormatter.
            errat_ax.xaxis.set_minor_locator(MultipleLocator(5))
            # errat_ax.set_xlim(0, pose_length)
            errat_ax.set_ylim(0, None)
            # graph_errat = sns.relplot(data=errat_graph_df, kind='line')  # x='Residue Number'
            # Plot the chain break(s) and design residues
            # labels = [fill(column, legend_fill_value) for column in errat_graph_df.columns]
            # errat_ax.legend(labels, loc='lower left', bbox_to_anchor=(0., 1.))
            # errat_ax.legend(loc='lower center', bbox_to_anchor=(0., 1.))
            errat_ax.vlines(self.pose.chain_breaks, 0, 1, transform=errat_ax.get_xaxis_transform(),
                            label='Entity Breaks', colors='#cccccc')  # , grey)
            errat_ax.vlines(index_residues, 0, 0.05, transform=errat_ax.get_xaxis_transform(),
                            label='Design Residues', colors='#f89938', lw=2)  # , orange)
            # Plot horizontal significance
            errat_ax.hlines([errat_2_sigma], 0, 1, transform=errat_ax.get_yaxis_transform(),
                            label='Significant Error', colors='#fc554f', linestyle='dotted')  # tomato
            errat_ax.set_xlabel('Residue Number')
            errat_ax.set_ylabel('Errat Score')
            # errat_ax.autoscale(True)
            # errat_ax.figure.tight_layout()
            # errat_ax.figure.savefig(os.path.join(self.data, 'errat.png'))
            collapse_handles, collapse_labels = collapse_ax.get_legend_handles_labels()
            contact_handles, contact_labels = contact_ax.get_legend_handles_labels()
            # errat_handles, errat_labels = errat_ax.get_legend_handles_labels()
            # print(handles, labels)
            handles = collapse_handles + contact_handles
            labels = collapse_labels + contact_labels
            # handles = errat_handles + contact_handles
            # labels = errat_labels + contact_labels
            labels = [label.replace(f'{self.name}_', '') for label in labels]
            # plt.legend(loc='upper right', bbox_to_anchor=(1, 1))  #, ncol=3, mode='expand')
            # print(labels)
            # plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -1.), ncol=3)  # , mode='expand'
            # v Why the hell doesn't this work??
            # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.), ncol=3,  # , mode='expand')
            # fig.subplots_adjust(bottom=0.1)
            plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -1.), ncol=3)  # , mode='expand')
            #            bbox_transform=plt.gcf().transFigure)  # , bbox_transform=collapse_ax.transAxes)
            fig.tight_layout()
            fig.savefig(os.path.join(self.data, 'DesignMetricsPerResidues.png'))  # Todo PoseDirectory(.path)

        # After parsing data sources
        interface_metrics_s = pd.concat([interface_metrics_s], keys=[('dock', 'pose')])

        # CONSTRUCT: Create pose series and format index names
        pose_s = pd.concat([interface_metrics_s, stat_s, divergence_s] + sim_series).swaplevel(0, 1)
        # Remove pose specific metrics from pose_s, sort, and name protocol_mean_df
        pose_s.drop([putils.protocol], level=2, inplace=True, errors='ignore')
        pose_s.sort_index(level=2, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
        pose_s.sort_index(level=1, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
        pose_s.sort_index(level=0, inplace=True, sort_remaining=False)  # ascending=False
        pose_s.name = str(self)

        return pose_s

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    # @remove_structure_memory  # structures not used in this protocol
    def select_sequences(self, filters: dict = None, weights: dict = None, number: int = 1, protocols: list[str] = None,
                         **kwargs) -> list[str]:
        """Select sequences for further characterization. If weights, then user can prioritize by metrics, otherwise
        sequence with the most neighbors as calculated by sequence distance will be selected. If there is a tie, the
        sequence with the lowest weight will be selected

        Args:
            filters: The filters to use in sequence selection
            weights: The weights to use in sequence selection
            number: The number of sequences to consider for each design
            protocols: Whether particular design protocol(s) should be chosen
        Returns:
            The selected designs for the Pose trajectories
        """
        # Load relevant data from the design directory
        trajectory_df = pd.read_csv(self.trajectories, index_col=0, header=[0])
        trajectory_df.dropna(inplace=True)
        if protocols:
            designs = []
            for protocol in protocols:
                designs.extend(trajectory_df[trajectory_df['protocol'] == protocol].index.to_list())

            if not designs:
                raise DesignError(f'No designs found for protocols {protocols}!')
        else:
            designs = trajectory_df.index.to_list()

        self.log.info(f'Number of starting trajectories = {len(trajectory_df)}')
        df = trajectory_df.loc[designs, :]

        if filters:
            self.log.info(f'Using filter parameters: {filters}')
            # Filter the DataFrame to include only those values which are le/ge the specified filter
            filtered_designs = index_intersection(filter_df_for_index_by_value(df, filters).values())
            df = df.loc[filtered_designs, :]

        if weights:
            # No filtering of protocol/indices to use as poses should have similar protocol scores coming in
            self.log.info(f'Using weighting parameters: {weights}')
            designs = rank_dataframe_by_metric_weights(df, weights=weights, **kwargs).index.to_list()
        else:
            # sequences_pickle = glob(os.path.join(self.job.all_scores, '%s_Sequences.pkl' % str(self)))
            # assert len(sequences_pickle) == 1, 'Couldn\'t find files for %s' % \
            #                                     os.path.join(self.job.all_scores, '%s_Sequences.pkl' % str(self))
            #
            # chain_sequences = SDUtils.unpickle(sequences_pickle[0])
            # {chain: {name: sequence, ...}, ...}
            # designed_sequences_by_entity: list[dict[str, str]] = unpickle(self.designed_sequences)
            # designed_sequences_by_entity: list[dict[str, str]] = self.designed_sequences
            # entity_sequences = list(zip(*[list(designed_sequences.values())
            #                               for designed_sequences in designed_sequences_by_entity]))
            # concatenated_sequences = [''.join(entity_sequence) for entity_sequence in entity_sequences]
            pose_sequences = self.designed_sequences
            self.log.debug(f'The final concatenated sequences are:\n{pose_sequences}')

            # pairwise_sequence_diff_np = SDUtils.all_vs_all(concatenated_sequences, sequence_difference)
            # Using concatenated sequences makes the values very similar and inflated as most residues are the same
            # doing min/max normalization to see variation
            pairwise_sequence_diff_l = [sequence_difference(*seq_pair)
                                        for seq_pair in combinations(pose_sequences, 2)]
            pairwise_sequence_diff_np = np.array(pairwise_sequence_diff_l)
            _min = min(pairwise_sequence_diff_l)
            # _max = max(pairwise_sequence_diff_l)
            pairwise_sequence_diff_np = np.subtract(pairwise_sequence_diff_np, _min)
            # self.log.info(pairwise_sequence_diff_l)

            # PCA analysis of distances
            pairwise_sequence_diff_mat = np.zeros((len(designs), len(designs)))
            for k, dist in enumerate(pairwise_sequence_diff_np):
                i, j = condensed_to_square(k, len(designs))
                pairwise_sequence_diff_mat[i, j] = dist
            pairwise_sequence_diff_mat = sym(pairwise_sequence_diff_mat)

            pairwise_sequence_diff_mat = skl.preprocessing.StandardScaler().fit_transform(pairwise_sequence_diff_mat)
            seq_pca = skl.decomposition.PCA(variance)
            seq_pc_np = seq_pca.fit_transform(pairwise_sequence_diff_mat)
            seq_pca_distance_vector = pdist(seq_pc_np)
            # epsilon = math.sqrt(seq_pca_distance_vector.mean()) * 0.5
            epsilon = seq_pca_distance_vector.mean() * 0.5
            self.log.info(f'Finding maximum neighbors within distance of {epsilon}')

            # self.log.info(pairwise_sequence_diff_np)
            # epsilon = pairwise_sequence_diff_mat.mean() * 0.5
            # epsilon = math.sqrt(seq_pc_np.myean()) * 0.5
            # epsilon = math.sqrt(pairwise_sequence_diff_np.mean()) * 0.5

            # Find the nearest neighbors for the pairwise distance matrix using the X*X^T (PCA) matrix, linear transform
            seq_neighbors = skl.neighbors.BallTree(seq_pc_np)
            seq_neighbor_counts = seq_neighbors.query_radius(seq_pc_np, epsilon,
                                                             count_only=True)  # , sort_results=True)
            top_count, top_idx = 0, None
            for count in seq_neighbor_counts:  # idx, enumerate()
                if count > top_count:
                    top_count = count

            sorted_seqs = sorted(seq_neighbor_counts, reverse=True)
            top_neighbor_counts = sorted(set(sorted_seqs[:number]), reverse=True)

            # Find only the designs which match the top x (number) of neighbor counts
            final_designs = {designs[idx]: num_neighbors for num_neighbors in top_neighbor_counts
                             for idx, count in enumerate(seq_neighbor_counts) if count == num_neighbors}
            self.log.info('The final sequence(s) and file(s):\nNeighbors\tDesign\n%s'
                          # % '\n'.join('%d %s' % (top_neighbor_counts.index(neighbors) + SDUtils.zero_offset,
                          % '\n'.join(f'\t{neighbors}\t{os.path.join(self.designs, design)}'
                                      for design, neighbors in final_designs.items()))

            # self.log.info('Corresponding PDB file(s):\n%s' % '\n'.join('%d %s' % (i, os.path.join(self.designs, seq))
            #                                                         for i, seq in enumerate(final_designs, 1)))

            # Compute the highest density cluster using DBSCAN algorithm
            # seq_cluster = DBSCAN(eps=epsilon)
            # seq_cluster.fit(pairwise_sequence_diff_np)
            #
            # seq_pc_df = pd.DataFrame(seq_pc, index=designs, columns=['pc' + str(x + SDUtils.zero_offset)
            #                                                          for x in range(len(seq_pca.components_))])
            # seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)

            # If final designs contains more sequences than specified, find the one with the lowest energy
            if len(final_designs) > number:
                energy_s = df.loc[final_designs.keys(), 'interface_energy']
                energy_s.sort_values(inplace=True)
                designs = energy_s.index.to_list()
            else:
                designs = list(final_designs.keys())

        designs = designs[:number]
        self.log.info(f'Final ranking of trajectories:\n{", ".join(design for design in designs)}')
        return designs

    # handle_design_errors = staticmethod(handle_design_errors)
    # close_logs = staticmethod(close_logs)
    remove_structure_memory = staticmethod(remove_structure_memory)

    def __key(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        if isinstance(other, PoseDirectory):
            return self.__key() == other.__key()
        raise NotImplementedError(f"Can't compare {PoseDirectory.__name__} instance to {other.__name__} instance")

    def __hash__(self) -> int:
        return hash(self.__key())

    def __str__(self) -> str:
        if self.job.nanohedra_output:
            return self.source_path.replace(f'{self.job.nanohedra_root}{os.sep}', '').replace(os.sep, '-')
        elif self.job.output_to_directory:
            return self.name
        else:
            # TODO integrate with designDB?
            return self.path.replace(f'{self.projects}{os.sep}', '').replace(os.sep, '-')


# @handle_design_errors(errors=(DesignError,))
# @close_logs
# @remove_structure_memory
def interface_design_analysis(pose: Pose, design_poses: Iterable[Pose] = None, scores_file: AnyStr = None, **kwargs) \
        -> pd.Series:
    """Retrieve all score information from a PoseDirectory and write results to .csv file

    Args:
        pose: The Pose to perform the analysis on
        design_poses: The subsequent designs to perform analysis on
        scores_file: A file that contains a JSON formatting metrics file with key, value paired metrics
    Returns:
        Series containing summary metrics for all designs in the design directory
    """
    # Todo move PoseDirectory to pose.path attribute
    job = job_resources_factory.get(**kwargs)
    # if pose.interface_residue_numbers is False or pose.interface_design_residue_numbers is False:
    pose.find_and_split_interface()

    if job.generate_fragments and not pose.fragment_queries:
        pose.generate_interface_fragments()
        if job.write_fragments:
            pose.write_fragment_pairs(out_path=pose.path.frags)
    # Gather miscellaneous pose specific metrics
    other_pose_metrics = pose.interface_metrics()

    # Find all designs files
    if design_poses is None:
        raise NotImplementedError(f'Attach the job.design arguments here first!')
        design_poses = pose.design_sequences()

    # Assumes each structure is the same length
    pose_length = pose.number_of_residues
    residue_indices = list(range(1, pose_length + 1))
    pose_sequences = {putils.pose_source: pose.sequence}
    # Todo implement reference sequence from included file(s) or as with pose.sequence below
    pose_sequences.update({putils.reference_name: pose.sequence})
    pose_sequences.update({pose.name: pose.sequence for pose in design_poses})
    all_mutations = generate_mutations_from_reference(pose.sequence, pose_sequences, return_to=True)  # , zero_index=True)
    #    generate_mutations_from_reference(''.join(pose.atom_sequences.values()), pose_sequences)

    entity_energies = [0. for _ in pose.entities]
    pose_source_residue_info = \
        {residue.number: {'complex': 0., 'bound': copy(entity_energies), 'unbound': copy(entity_energies),
                          'solv_complex': 0., 'solv_bound': copy(entity_energies),
                          'solv_unbound': copy(entity_energies), 'fsp': 0., 'cst': 0.,
                          'type': protein_letters_3to1.get(residue.type), 'hbond': 0}
         for entity in pose.entities for residue in entity.residues}
    residue_info = {putils.pose_source: pose_source_residue_info}
    job_key = 'no_energy'
    stat_s, sim_series = pd.Series(dtype=float), []
    if scores_file is not None and os.path.exists(scores_file):  # Rosetta scores file is present
        pose.log.debug(f'Found design scores in file: {scores_file}')
        design_was_performed = True
        # Get the scores from the score file on design trajectory metrics
        source_df = pd.DataFrame({putils.pose_source: {putils.protocol: job_key}}).T
        for idx, entity in enumerate(pose.entities, 1):
            source_df[f'buns_{idx}_unbound'] = 0
            source_df[f'interface_energy_{idx}_bound'] = 0
            source_df[f'interface_energy_{idx}_unbound'] = 0
            source_df[f'solvation_energy_{idx}_bound'] = 0
            source_df[f'solvation_energy_{idx}_unbound'] = 0
            source_df[f'interface_connectivity_{idx}'] = 0
        source_df['buns_complex'] = 0
        # source_df['buns_unbound'] = 0
        source_df['contact_count'] = 0
        source_df['favor_residue_energy'] = 0
        source_df['interface_energy_complex'] = 0
        source_df['interaction_energy_complex'] = 0
        source_df['interaction_energy_per_residue'] = \
            source_df['interaction_energy_complex'] / len(pose.interface_residues)
        source_df['interface_separation'] = 0
        source_df['number_hbonds'] = 0
        source_df['rmsd_complex'] = 0  # Todo calculate this here instead of Rosetta using superposition3d
        source_df['rosetta_reference_energy'] = 0
        source_df['shape_complementarity'] = 0
        source_df['solvation_energy'] = 0
        source_df['solvation_energy_complex'] = 0
        design_scores = read_scores(scores_file)
        pose.log.debug(f'All designs with scores: {", ".join(design_scores.keys())}')
        # Remove designs with scores but no structures
        all_viable_design_scores = {}
        for pose in design_poses:
            try:
                all_viable_design_scores[pose.name] = design_scores.pop(pose.name)
            except KeyError:  # structure wasn't scored, we will remove this later
                pass
        # Create protocol dataframe
        scores_df = pd.DataFrame(all_viable_design_scores).T
        scores_df = pd.concat([source_df, scores_df])
        # Gather all columns into specific types for processing and formatting
        per_res_columns, hbonds_columns = [], []
        for column in scores_df.columns.to_list():
            if column.startswith('per_res_'):
                per_res_columns.append(column)
            elif column.startswith('hbonds_res_selection'):
                hbonds_columns.append(column)

        # Check proper input
        metric_set = necessary_metrics.difference(set(scores_df.columns))
        # pose.log.debug('Score columns present before required metric check: %s' % scores_df.columns.to_list())
        if not metric_set:
            raise DesignError(f'Missing required metrics: "{", ".join(metric_set)}"')

        # Remove unnecessary (old scores) as well as Rosetta pose score terms besides ref (has been renamed above)
        # TODO learn know how to produce score terms in output score file. Not in FastRelax...
        remove_columns = per_res_columns + hbonds_columns + rosetta_terms + unnecessary
        # TODO remove dirty when columns are correct (after P432)
        #  and column tabulation precedes residue/hbond_processing
        interface_hbonds = dirty_hbond_processing(all_viable_design_scores)
        # can't use hbond_processing (clean) in the case there is a design without metrics... columns not found!
        # interface_hbonds = hbond_processing(all_viable_design_scores, hbonds_columns)
        number_hbonds_s = \
            pd.Series({design: len(hbonds) for design, hbonds in interface_hbonds.items()}, name='number_hbonds')
        # number_hbonds_s = pd.Series({design: len(hbonds) for design, hbonds in interface_hbonds.items()})  #, name='number_hbonds')
        # scores_df = pd.merge(scores_df, number_hbonds_s, left_index=True, right_index=True)
        scores_df.loc[number_hbonds_s.index, 'number_hbonds'] = number_hbonds_s
        # scores_df = scores_df.assign(number_hbonds=number_hbonds_s)
        # residue_info = {'energy': {'complex': 0., 'unbound': 0.}, 'type': None, 'hbond': 0}
        residue_info.update(pose.rosetta_residue_processing(all_viable_design_scores))
        residue_info = process_residue_info(residue_info, hbonds=interface_hbonds)
        residue_info = incorporate_mutation_info(residue_info, all_mutations)
        # can't use residue_processing (clean) in the case there is a design without metrics... columns not found!
        # residue_info.update(residue_processing(all_viable_design_scores, simplify_mutation_dict(all_mutations),
        #                                        per_res_columns, hbonds=interface_hbonds))

        # Drop designs where required data isn't present
        # Format protocol columns
        missing_group_indices = scores_df[putils.protocol].isna()
        # Todo remove not DEV
        scout_indices = [idx for idx in scores_df[missing_group_indices].index if 'scout' in idx]
        scores_df.loc[scout_indices, putils.protocol] = putils.scout
        structure_bkgnd_indices = [idx for idx in scores_df[missing_group_indices].index if 'no_constraint' in idx]
        scores_df.loc[structure_bkgnd_indices, putils.protocol] = putils.structure_background
        # Todo Done remove
        # protocol_s.replace({'combo_profile': putils.design_profile}, inplace=True)  # ensure proper profile name

        scores_df.drop(missing_group_indices, axis=0, inplace=True, errors='ignore')
        # protocol_s.drop(missing_group_indices, inplace=True, errors='ignore')

        viable_designs = scores_df.index.to_list()
        if not viable_designs:
            raise DesignError('No viable designs remain after processing!')

        pose.log.debug(f'Viable designs remaining after cleaning:\n\t{", ".join(viable_designs)}')
        pose_sequences = {design: sequence for design, sequence in pose_sequences.items() if design in viable_designs}

        # Todo implement this protocol if sequence data is taken at multiple points along a trajectory and the
        #  sequence data along trajectory is a metric on it's own
        # # Gather mutations for residue specific processing and design sequences
        # for design, data in list(all_viable_design_scores.items()):  # make a copy as can be removed
        #     sequence = data.get('final_sequence')
        #     if sequence:
        #         if len(sequence) >= pose_length:
        #             pose_sequences[design] = sequence[:pose_length]  # Todo won't work if design had insertions
        #         else:
        #             pose_sequences[design] = sequence
        #     else:
        #         pose.log.warning('Design %s is missing sequence data, removing from design pool' % design)
        #         all_viable_design_scores.pop(design)
        # # format {entity: {design_name: sequence, ...}, ...}
        # entity_sequences = \
        #     {entity: {design: sequence[entity.n_terminal_residue.number - 1:entity.c_terminal_residue.number]
        #               for design, sequence in pose_sequences.items()} for entity in pose.entities}
    else:
        pose.log.debug(f'Missing scores file')
        design_was_performed = False
        # Todo add relevant missing scores such as those specified as 0 below
        # Todo may need to put source_df in scores file alternative
        source_df = pd.DataFrame({putils.pose_source: {putils.protocol: job_key}}).T
        scores_df = pd.DataFrame({pose.name: {putils.protocol: job_key} for pose in design_poses}).T
        scores_df = pd.concat([source_df, scores_df])
        for idx, entity in enumerate(pose.entities, 1):
            source_df[f'buns_{idx}_unbound'] = 0
            source_df[f'interface_energy_{idx}_bound'] = 0
            source_df[f'interface_energy_{idx}_unbound'] = 0
            source_df[f'solvation_energy_{idx}_bound'] = 0
            source_df[f'solvation_energy_{idx}_unbound'] = 0
            source_df[f'interface_connectivity_{idx}'] = 0
            # residue_info = {'energy': {'complex': 0., 'unbound': 0.}, 'type': None, 'hbond': 0}
            # design_info.update({residue.number: {'energy_delta': 0.,
            #                                      'type': protein_letters_3to1.get(residue.type),
            #                                      'hbond': 0} for residue in entity.residues})
        source_df['buns_complex'] = 0
        # source_df['buns_unbound'] = 0
        scores_df['contact_count'] = 0
        scores_df['favor_residue_energy'] = 0
        scores_df['interface_energy_complex'] = 0
        scores_df['interaction_energy_complex'] = 0
        scores_df['interaction_energy_per_residue'] = \
            scores_df['interaction_energy_complex'] / len(pose.interface_design_residue_numbers)
        scores_df['interface_separation'] = 0
        scores_df['number_hbonds'] = 0
        scores_df['rmsd_complex'] = 0  # Todo calculate this here instead of Rosetta using superposition3d
        scores_df['rosetta_reference_energy'] = 0
        scores_df['shape_complementarity'] = 0
        scores_df['solvation_energy'] = 0
        scores_df['solvation_energy_complex'] = 0
        remove_columns = rosetta_terms + unnecessary
        residue_info.update({struct_name: pose_source_residue_info for struct_name in scores_df.index.to_list()})
        # Todo generate energy scores internally which matches output from residue_processing
        # interface_hbonds = dirty_hbond_processing(design_scores)
        # residue_info = pose.rosetta_residue_processing(design_scores)
        # residue_info = process_residue_info(residue_info, simplify_mutation_dict(all_mutations),
        #                                     hbonds=interface_hbonds)
        viable_designs = [pose.name for pose in design_poses]

    scores_df.drop(remove_columns, axis=1, inplace=True, errors='ignore')
    other_pose_metrics['observations'] = len(viable_designs)

    entity_sequences = []
    for entity in pose.entities:
        entity_slice = slice(entity.n_terminal_residue.index, 1+entity.c_terminal_residue.index)
        entity_sequences.append({design: sequence[entity_slice] for design, sequence in pose_sequences.items()})
    # Todo generate_multiple_mutations accounts for offsets from the reference sequence. Not necessary YET
    # sequence_mutations = \
    #     generate_multiple_mutations(pose.atom_sequences, pose_sequences, pose_num=False)
    # sequence_mutations.pop('reference')
    # entity_sequences = generate_sequences(pose.atom_sequences, sequence_mutations)
    # entity_sequences = {chain: keys_from_trajectory_number(named_sequences)
    #                         for chain, named_sequences in entity_sequences.items()}

    # Find protocols for protocol specific data processing removing from scores_df
    protocol_s = scores_df.pop(putils.protocol).copy()
    designs_by_protocol = protocol_s.groupby(protocol_s).groups
    # remove refine and consensus if present as there was no design done over multiple protocols
    unique_protocols = list(designs_by_protocol.keys())
    # Todo change if we did multiple rounds of these protocols
    designs_by_protocol.pop(putils.refine, None)
    designs_by_protocol.pop(putils.consensus, None)
    # Get unique protocols
    unique_design_protocols = set(designs_by_protocol.keys())
    pose.log.info(f'Unique Design Protocols: {", ".join(unique_design_protocols)}')

    # Replace empty strings with np.nan and convert remaining to float
    scores_df.replace('', np.nan, inplace=True)
    scores_df.fillna(dict(zip(protocol_specific_columns, repeat(0))), inplace=True)
    scores_df = scores_df.astype(float)  # , copy=False, errors='ignore')

    # per residue data includes every residue in the pose
    # per_residue_data = {'errat_deviation': {}, 'hydrophobic_collapse': {}, 'contact_order': {},
    #                     'sasa_hydrophobic_complex': {}, 'sasa_polar_complex': {}, 'sasa_relative_complex': {},
    #                     'sasa_hydrophobic_bound': {}, 'sasa_polar_bound': {}, 'sasa_relative_bound': {}}
    interface_local_density = {putils.pose_source: pose.local_density_interface()}
    # atomic_deviation = {}
    # pose_assembly_minimally_contacting = pose.assembly_minimally_contacting
    # perform SASA measurements
    # pose_assembly_minimally_contacting.get_sasa()
    # assembly_asu_residues = pose_assembly_minimally_contacting.residues[:pose_length]
    # per_residue_data['sasa_hydrophobic_complex'][putils.pose_source] = \
    #     [residue.sasa_apolar for residue in assembly_asu_residues]
    # per_residue_data['sasa_polar_complex'][putils.pose_source] = [residue.sasa_polar for residue in assembly_asu_residues]
    # per_residue_data['sasa_relative_complex'][putils.pose_source] = \
    #     [residue.relative_sasa for residue in assembly_asu_residues]

    # Grab metrics for the pose source. Checks if pose was designed
    # Favor pose source errat/collapse on a per-entity basis if design occurred
    # As the pose source assumes no legit interface present while designs have an interface
    # per_residue_sasa_unbound_apolar, per_residue_sasa_unbound_polar, per_residue_sasa_unbound_relative = [], [], []
    # source_errat_accuracy, source_errat, source_contact_order, inverse_residue_contact_order_z = [], [], [], []
    source_errat, source_contact_order = [], []
    for idx, entity in enumerate(pose.entities):
        # Contact order is the same for every design in the Pose and not dependent on pose
        source_contact_order.append(entity.contact_order)
        if design_was_performed:  # we should respect input structure was not meant to be together
            # oligomer_errat_accuracy, oligomeric_errat = entity_oligomer.errat(out_path=os.path.devnull)
            # source_errat_accuracy.append(oligomer_errat_accuracy)
            # Todo when Entity.oligomer works
            #  _, oligomeric_errat = entity.oligomer.errat(out_path=os.path.devnull)
            entity_oligomer = Model.from_chains(entity.chains, log=pose.log, entities=False)
            _, oligomeric_errat = entity_oligomer.errat(out_path=os.path.devnull)
            source_errat.append(oligomeric_errat[:entity.number_of_residues])

    per_residue_data = {putils.pose_source: pose.get_per_residue_interface_metrics()}
    pose_source_contact_order_s = \
        pd.Series(np.concatenate(source_contact_order), index=residue_indices, name='contact_order')
    per_residue_data[putils.pose_source]['contact_order'] = pose_source_contact_order_s

    number_of_entities = pose.number_of_entities
    if design_was_performed:  # The input structure was not meant to be together, treat as such
        source_errat = []
        for idx, entity in enumerate(pose.entities):
            # Replace 'errat_deviation' measurement with uncomplexed entities
            # oligomer_errat_accuracy, oligomeric_errat = entity_oligomer.errat(out_path=os.path.devnull)
            # source_errat_accuracy.append(oligomer_errat_accuracy)
            # Todo when Entity.oligomer works
            #  _, oligomeric_errat = entity.oligomer.errat(out_path=os.path.devnull)
            entity_oligomer = Model.from_chains(entity.chains, log=pose.log, entities=False)
            _, oligomeric_errat = entity_oligomer.errat(out_path=os.path.devnull)
            source_errat.append(oligomeric_errat[:entity.number_of_residues])
        # atomic_deviation[putils.pose_source] = sum(source_errat_accuracy) / float(number_of_entities)
        pose_source_errat_s = pd.Series(np.concatenate(source_errat), index=residue_indices)
    else:
        pose_assembly_minimally_contacting = pose.assembly_minimally_contacting
        # atomic_deviation[putils.pose_source], pose_per_residue_errat = \
        _, pose_per_residue_errat = \
            pose_assembly_minimally_contacting.errat(out_path=os.path.devnull)
        pose_source_errat_s = pose_per_residue_errat[:pose_length]

    per_residue_data[putils.pose_source]['errat_deviation'] = pose_source_errat_s

    # Compute structural measurements for all designs
    for pose in design_poses:  # Takes 1-2 seconds for Structure -> assembly -> errat
        # Must find interface residues before measure local_density
        pose.find_and_split_interface()
        per_residue_data[pose.name] = pose.get_per_residue_interface_metrics()
        # Todo remove Rosetta
        #  This is a measurement of interface_connectivity like from Rosetta
        interface_local_density[pose.name] = pose.local_density_interface()

    # scores_df['errat_accuracy'] = pd.Series(atomic_deviation)
    scores_df['interface_local_density'] = pd.Series(interface_local_density)
    # Include in errat_deviation if errat score is < 2 std devs and isn't 0 to begin with
    source_errat_inclusion_boolean = np.logical_and(pose_source_errat_s < errat_2_sigma, pose_source_errat_s != 0.)
    errat_df = per_residue_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)
    # find where designs deviate above wild-type errat scores
    errat_sig_df = (errat_df.sub(pose_source_errat_s, axis=1)) > errat_1_sigma  # axis=1 Series is column oriented
    # then select only those residues which are expressly important by the inclusion boolean
    scores_df['errat_deviation'] = (errat_sig_df.loc[:, source_errat_inclusion_boolean] * 1).sum(axis=1)

    # Calculate hydrophobic collapse for each design
    # Todo reconcile below with fragdock.py
    measure_evolution = measure_alignment = True
    warn = False
    # Add Entity information to the Pose
    for idx, entity in enumerate(pose.entities):
        # entity.sequence_file = job.api_db.sequences.retrieve_file(name=entity.name)
        # if not entity.sequence_file:
        #     entity.write_sequence_to_fasta('reference', out_dir=job.sequences)
        #     # entity.add_evolutionary_profile(out_dir=job.api_db.hhblits_profiles.location)
        # else:
        profile = job.api_db.hhblits_profiles.retrieve_data(name=entity.name)
        if not profile:
            measure_evolution = False
            warn = True
        else:
            entity.evolutionary_profile = profile

        if not entity.verify_evolutionary_profile():
            entity.fit_evolutionary_profile_to_structure()

        try:  # To fetch the multiple sequence alignment for further processing
            msa = job.api_db.alignments.retrieve_data(name=entity.name)
            if not msa:
                measure_alignment = False
                warn = True
            else:
                entity.msa = msa
        except ValueError as error:  # When the Entity reference sequence and alignment are different lengths
            pose.log.info(f'Entity reference sequence and provided alignment are different lengths: {error}')
            warn = True

    if warn:
        if not measure_evolution and not measure_alignment:
            pose.log.info(f'Metrics relying on multiple sequence alignment data are not being collected as '
                          f'there were none found. These include: '
                          f'{", ".join(multiple_sequence_alignment_dependent_metrics)}')
        elif not measure_alignment:
            pose.log.info(f'Metrics relying on a multiple sequence alignment are not being collected as '
                          f'there was no MSA found. These include: '
                          f'{", ".join(multiple_sequence_alignment_dependent_metrics)}')
        else:
            pose.log.info(f'Metrics relying on an evolutionary profile are not being collected as '
                          f'there was no profile found. These include: '
                          f'{", ".join(profile_dependent_metrics)}')

    # Include the putils.pose_source in the measured designs
    contact_order_per_res_z, reference_collapse, collapse_profile = pose.get_folding_metrics()
    folding_and_collapse = calculate_collapse_metrics(list(zip(*[list(designed_sequences.values())
                                                                 for designed_sequences in entity_sequences])),
                                                      contact_order_per_res_z, reference_collapse, collapse_profile)
    # pose_collapse_df = pd.DataFrame(folding_and_collapse).T
    per_residue_collapse_df = pd.concat({design_id: pd.DataFrame(data, index=residue_numbers)
                                         for design_id, data in zip(viable_designs, folding_and_collapse)},
                                        ).unstack().swaplevel(0, 1, axis=1)
    # Convert per_residue_data into a dataframe matching residue_df orientation
    per_residue_df = pd.concat({name: pd.DataFrame(data, index=residue_indices)
                                for name, data in per_residue_data.items()}).unstack().swaplevel(0, 1, axis=1)
    # Fill in contact order for each design
    per_residue_df.fillna(per_residue_df.loc[putils.pose_source, idx_slice[:, 'contact_order']], inplace=True)
    per_residue_df = per_residue_df.join(per_residue_collapse_df)
    # Process mutational frequencies, H-bond, and Residue energy metrics to dataframe
    residue_df = pd.concat({design: pd.DataFrame(info) for design, info in residue_info.items()}).unstack()
    # returns multi-index column with residue number as first (top) column index, metric as second index
    # during residue_df unstack, all residues with missing dicts are copied as nan
    # Merge interface design specific residue metrics with total per residue metrics
    index_residues = list(pose.interface_design_residue_numbers)
    residue_df = pd.merge(residue_df.loc[:, idx_slice[index_residues, :]],
                          per_residue_df.loc[:, idx_slice[index_residues, :]],
                          left_index=True, right_index=True)
    if measure_evolution:
        pose.evolutionary_profile = concatenate_profile([entity.evolutionary_profile for entity in pose.entities])

    # pose.generate_interface_fragments() was already called
    pose.calculate_fragment_profile()
    pose.add_profile(evolution=job.design.evolution_constraint,
                     fragments=job.generate_fragments)

    # Load profiles of interest into the analysis
    profile_background = {'design': pssm_as_array(pose.profile),
                          'evolution': pssm_as_array(pose.evolutionary_profile),
                          'fragment': pose.fragment_profile.as_array()}
    # if pose.profile:
    # else:
    #     pose.log.info('Design has no fragment information')
    # if pose.evolutionary_profile:
    # else:
    #     pose.log.info('No evolution information')
    # if pose.fragment_profile:
    # else:
    #     pose.log.info('No fragment information')
    if job.fragment_db is not None:
        interface_bkgd = np.array(list(job.fragment_db.aa_frequencies.values()))
        profile_background['interface'] = np.tile(interface_bkgd, (pose.number_of_residues, 1))

    if not profile_background:
        divergence_s = pd.Series(dtype=float)
    else:  # Calculate sequence statistics
        # First, for entire pose
        interface_indexer = [residue.index for residue in pose.interface_residues]
        pose_alignment = MultipleSequenceAlignment.from_dictionary(pose_sequences)
        observed, divergence = \
            calculate_sequence_observations_and_divergence(pose_alignment, profile_background, interface_indexer)

        observed_dfs = []
        for profile, observed_values in observed.items():
            scores_df[f'observed_{profile}'] = observed_values.mean(axis=1)
            observed_dfs.append(pd.DataFrame(data=observed_values, index=pose_sequences,  # design_obs_freqs.keys()
                                             columns=pd.MultiIndex.from_product([residue_indices,
                                                                                 [f'observed_{profile}']]))
                                )
        # Add observation information into the residue_df
        residue_df = pd.concat([residue_df] + observed_dfs, axis=1)
        # Get pose sequence divergence
        pose_divergence_s = pd.concat([pd.Series({f'{divergence_type}_per_residue': _divergence.mean()
                                                  for divergence_type, _divergence in divergence.items()})],
                                      keys=[('sequence_design', 'pose')])
        # pose_divergence_s = pd.Series({f'{divergence_type}_per_residue': per_res_metric(stat)
        #                                for divergence_type, stat in divergence.items()},
        #                               name=('sequence_design', 'pose'))
        if designs_by_protocol:  # Were multiple designs generated with each protocol?
            # Find the divergence within each protocol
            divergence_by_protocol = {protocol: {} for protocol in designs_by_protocol}
            for protocol, designs in designs_by_protocol.items():
                # Todo select from pose_alignment the indices of each design then pass to MultipleSequenceAlignment?
                # protocol_alignment = \
                #     MultipleSequenceAlignment.from_dictionary({design: pose_sequences[design]
                #                                                for design in designs})
                protocol_alignment = MultipleSequenceAlignment.from_dictionary({design: pose_sequences[design]
                                                                                for design in designs})
                # protocol_mutation_freq = filter_dictionary_keys(protocol_alignment.frequencies,
                #                                                 pose.interface_design_residue_numbers)
                # protocol_mutation_freq = protocol_alignment.frequencies
                protocol_divergence = {f'divergence_{profile}':
                                       position_specific_divergence(protocol_alignment.frequencies,
                                                                    bgd)[interface_indexer]
                                       for profile, bgd in profile_background.items()}
                # if interface_bkgd is not None:
                #     protocol_divergence['divergence_interface'] = \
                #         position_specific_divergence(protocol_alignment.frequencies[interface_indexer],
                #                                      tiled_int_background)
                # Get per residue divergence metric by protocol
                divergence_by_protocol[protocol] = {f'{divergence_type}_per_residue': divergence.mean()
                                                    for divergence_type, divergence in protocol_divergence.items()}
            # new = dfd.columns.to_frame()
            # new.insert(0, 'new2_level_name', new_level_values)
            # dfd.columns = pd.MultiIndex.from_frame(new)
            protocol_divergence_s = \
                pd.concat([pd.DataFrame(divergence_by_protocol).unstack()], keys=['sequence_design'])
        else:
            protocol_divergence_s = pd.Series(dtype=float)
        divergence_s = pd.concat([protocol_divergence_s, pose_divergence_s])

    scores_df['number_of_mutations'] = \
        pd.Series({design: len(mutations) for design, mutations in all_mutations.items()})
    scores_df['percent_mutations'] = \
        scores_df['number_of_mutations'] / other_pose_metrics['pose_length']
    # residue_indices_per_entity = pose.residue_indices_per_entity
    is_thermophilic = []
    idx = 1
    for idx, entity in enumerate(pose.entities, idx):
        pose_c_terminal_residue_number = entity.c_terminal_residue.index + 1
        scores_df[f'entity_{idx}_number_of_mutations'] = \
            pd.Series({design: len([1 for mutation_idx in mutations if mutation_idx < pose_c_terminal_residue_number])
                       for design, mutations in all_mutations.items()})
        scores_df[f'entity_{idx}_percent_mutations'] = \
            scores_df[f'entity_{idx}_number_of_mutations'] / other_pose_metrics[f'entity_{idx}_number_of_residues']
        is_thermophilic.append(getattr(other_pose_metrics, f'entity_{idx}_thermophile', 0))

    # Get the average thermophilicity for all entities
    other_pose_metrics['pose_thermophilicity'] = sum(is_thermophilic) / idx

    # entity_alignment = multi_chain_alignment(entity_sequences)
    # INSTEAD OF USING BELOW, split Pose.MultipleSequenceAlignment at entity.chain_break...
    # entity_alignments = \
    #     {idx: MultipleSequenceAlignment.from_dictionary(designed_sequences)
    #      for idx, designed_sequences in entity_sequences.items()}
    # entity_alignments = \
    #     {idx: msa_from_dictionary(designed_sequences) for idx, designed_sequences in entity_sequences.items()}
    # pose_collapse_ = pd.concat(pd.DataFrame(folding_and_collapse), axis=1, keys=[('sequence_design', 'pose')])
    dca_design_residues_concat = []
    dca_succeed = True
    # dca_background_energies, dca_design_energies = [], []
    dca_background_energies, dca_design_energies = {}, {}
    for idx, entity in enumerate(pose.entities):
        try:  # TODO add these to the analysis
            entity.h_fields = job.api_db.bmdca_fields.retrieve_data(name=entity.name)
            entity.j_couplings = job.api_db.bmdca_couplings.retrieve_data(name=entity.name)
            dca_background_residue_energies = entity.direct_coupling_analysis()
            # Todo INSTEAD OF USING BELOW, split Pose.MultipleSequenceAlignment at entity.chain_break...
            entity_alignment = MultipleSequenceAlignment.from_dictionary(entity_sequences[idx])
            # entity_alignment = msa_from_dictionary(entity_sequences[idx])
            dca_design_residue_energies = entity.direct_coupling_analysis(msa=entity_alignment)
            dca_design_residues_concat.append(dca_design_residue_energies)
            # dca_background_energies.append(dca_background_energies.sum(axis=1))
            # dca_design_energies.append(dca_design_energies.sum(axis=1))
            dca_background_energies[entity] = dca_background_residue_energies.sum(axis=1)  # turns data to 1D
            dca_design_energies[entity] = dca_design_residue_energies.sum(axis=1)
        except AttributeError:
            pose.log.warning(f"For {entity.name}, DCA analysis couldn't be performed. "
                             f"Missing required parameter files")
            dca_succeed = False

    if dca_succeed:
        # concatenate along columns, adding residue index to column, design name to row
        dca_concatenated_df = pd.DataFrame(np.concatenate(dca_design_residues_concat, axis=1),
                                           index=list(pose_sequences.keys()), columns=residue_indices)
        # get all design names                                    ^
        # dca_concatenated_df.columns = pd.MultiIndex.from_product([dca_concatenated_df.columns, ['dca_energy']])
        dca_concatenated_df = pd.concat([dca_concatenated_df], keys=['dca_energy'], axis=1).swaplevel(0, 1, axis=1)
        # merge with per_residue_df
        residue_df = pd.merge(residue_df, dca_concatenated_df, left_index=True, right_index=True)

    # residue_df = pd.merge(residue_df, per_residue_df.loc[:, idx_slice[residue_df.columns.levels[0], :]],
    #                       left_index=True, right_index=True)
    # Add local_density information to scores_df
    # scores_df['interface_local_density'] = \
    #     residue_df.loc[:, idx_slice[pose.interface_residue_numbers, 'local_density']].mean(axis=1)

    # Make buried surface area (bsa) columns
    residue_df = calculate_residue_surface_area(residue_df.loc[:, idx_slice[index_residues, :]])

    scores_df['interface_area_polar'] = residue_df.loc[:, idx_slice[index_residues, 'bsa_polar']].sum(axis=1)
    scores_df['interface_area_hydrophobic'] = \
        residue_df.loc[:, idx_slice[index_residues, 'bsa_hydrophobic']].sum(axis=1)
    # scores_df['interface_area_total'] = \
    #     residue_df.loc[not_pose_source_indices, idx_slice[index_residues, 'bsa_total']].sum(axis=1)
    scores_df['interface_area_total'] = scores_df['interface_area_polar'] + scores_df['interface_area_hydrophobic']

    # Find the proportion of the residue surface area that is solvent accessible versus buried in the interface
    sasa_assembly_df = residue_df.loc[:, idx_slice[index_residues, 'sasa_total_complex']].droplevel(-1, axis=1)
    bsa_assembly_df = residue_df.loc[:, idx_slice[index_residues, 'bsa_total']].droplevel(-1, axis=1)
    total_surface_area_df = sasa_assembly_df + bsa_assembly_df
    # ratio_df = bsa_assembly_df / total_surface_area_df
    scores_df['interface_area_to_residue_surface_ratio'] = (bsa_assembly_df / total_surface_area_df).mean(axis=1)

    # Check if any columns are > 50% interior (value can be 0 or 1). If so, return True for that column
    # interior_residue_df = residue_df.loc[:, idx_slice[:, 'interior']]
    # interior_residue_numbers = \
    #     interior_residues.loc[:, interior_residues.mean(axis=0) > 0.5].columns.remove_unused_levels().levels[0].
    #     to_list()
    # if interior_residue_numbers:
    #     pose.log.info(f'Design Residues {",".join(map(str, sorted(interior_residue_numbers)))}
    #                   'are located in the interior')

    # This shouldn't be much different from the state variable pose.interface_residue_numbers
    # perhaps the use of residue neighbor energy metrics adds residues which contribute, but not directly
    # interface_residue_numbers = set(residue_df.columns.levels[0].unique()).difference(interior_residue_numbers)

    # Add design residue information to scores_df such as how many core, rim, and support residues were measured
    for residue_class in residue_classification:
        scores_df[residue_class] = residue_df.loc[:, idx_slice[:, residue_class]].sum(axis=1)

    # Calculate new metrics from combinations of other metrics
    scores_columns = scores_df.columns.to_list()
    pose.log.debug(f'Metrics present: {scores_columns}')
    # sum columns using list[0] + list[1] + list[n]
    complex_df = residue_df.loc[:, idx_slice[:, 'complex']]
    bound_df = residue_df.loc[:, idx_slice[:, 'bound']]
    unbound_df = residue_df.loc[:, idx_slice[:, 'unbound']]
    solvation_complex_df = residue_df.loc[:, idx_slice[:, 'solv_complex']]
    solvation_bound_df = residue_df.loc[:, idx_slice[:, 'solv_bound']]
    solvation_unbound_df = residue_df.loc[:, idx_slice[:, 'solv_unbound']]
    scores_df['interface_energy_complex'] = complex_df.sum(axis=1)
    scores_df['interface_energy_bound'] = bound_df.sum(axis=1)
    scores_df['interface_energy_unbound'] = unbound_df.sum(axis=1)
    scores_df['interface_solvation_energy_complex'] = solvation_complex_df.sum(axis=1)
    scores_df['interface_solvation_energy_bound'] = solvation_bound_df.sum(axis=1)
    scores_df['interface_solvation_energy_unbound'] = solvation_unbound_df.sum(axis=1)
    residue_df = residue_df.drop([column for columns in [complex_df.columns, bound_df.columns, unbound_df.columns,
                                                         solvation_complex_df.columns, solvation_bound_df.columns,
                                                         solvation_unbound_df.columns]
                                  for column in columns], axis=1)
    summation_pairs = \
        {'buns_unbound': list(filter(re.compile('buns_[0-9]+_unbound$').match, scores_columns)),  # Rosetta
         # 'interface_energy_bound':
         #     list(filter(re.compile('interface_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
         # 'interface_energy_unbound':
         #     list(filter(re.compile('interface_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
         # 'interface_solvation_energy_bound':
         #     list(filter(re.compile('solvation_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
         # 'interface_solvation_energy_unbound':
         #     list(filter(re.compile('solvation_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
         'interface_connectivity':
             list(filter(re.compile('interface_connectivity_[0-9]+').match, scores_columns)),  # Rosetta
         }
    # 'sasa_hydrophobic_bound':
    #     list(filter(re.compile('sasa_hydrophobic_[0-9]+_bound').match, scores_columns)),
    # 'sasa_polar_bound': list(filter(re.compile('sasa_polar_[0-9]+_bound').match, scores_columns)),
    # 'sasa_total_bound': list(filter(re.compile('sasa_total_[0-9]+_bound').match, scores_columns))}
    scores_df = columns_to_new_column(scores_df, summation_pairs)
    scores_df = columns_to_new_column(scores_df, delta_pairs, mode='sub')
    # add total_interface_residues for div_pairs and int_comp_similarity
    scores_df['total_interface_residues'] = other_pose_metrics.pop('total_interface_residues')
    scores_df = columns_to_new_column(scores_df, division_pairs, mode='truediv')
    scores_df['interface_composition_similarity'] = scores_df.apply(interface_composition_similarity, axis=1)
    scores_df.drop(clean_up_intermediate_columns, axis=1, inplace=True, errors='ignore')
    if scores_df.get('repacking') is not None:
        # set interface_bound_activation_energy = NaN where repacking is 0
        # Currently is -1 for True (Rosetta Filter quirk...)
        scores_df.loc[scores_df[scores_df['repacking'] == 0].index, 'interface_bound_activation_energy'] = np.nan
        scores_df.drop('repacking', axis=1, inplace=True)

    # Process dataframes for missing values and drop refine trajectory if present
    # refine_index = scores_df[scores_df[putils.protocol] == putils.refine].index
    # scores_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
    # residue_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
    # residue_info.pop(putils.refine, None)  # Remove refine from analysis
    # residues_no_frags = residue_df.columns[residue_df.isna().all(axis=0)].remove_unused_levels().levels[0]
    residue_df.dropna(how='all', inplace=True, axis=1)  # remove completely empty columns such as obs_interface
    # fill in contact order for each design
    residue_df.fillna(residue_df.loc[putils.pose_source, idx_slice[:, 'contact_order']], inplace=True)  # method='pad',
    residue_df = residue_df.fillna(0.).copy()
    # residue_indices_no_frags = residue_df.columns[residue_df.isna().all(axis=0)]

    # POSE ANALYSIS
    scores_df = pd.concat([scores_df, pose_collapse_df], axis=1)
    scores_df.dropna(how='all', inplace=True, axis=1)  # remove completely empty columns
    # refine is not considered sequence design and destroys mean. remove v
    # trajectory_df = scores_df.sort_index().drop(putils.refine, axis=0, errors='ignore')
    # consensus cst_weights are very large and destroy the mean.
    # remove this drop for consensus or refine if they are run multiple times
    trajectory_df = \
        scores_df.drop([putils.pose_source, putils.refine, putils.consensus], axis=0, errors='ignore').sort_index()

    # Get total design statistics for every sequence in the pose and every protocol specifically
    scores_df[putils.protocol] = protocol_s
    protocol_groups = scores_df.groupby(putils.protocol)
    # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # print(trajectory_df.select_dtypes(exclude=numerics))

    pose_stats, protocol_stats = [], []
    for idx, stat in enumerate(stats_metrics):
        pose_stats.append(getattr(trajectory_df, stat)().rename(stat))
        protocol_stats.append(getattr(protocol_groups, stat)())

    # format stats_s for final pose_s Series
    protocol_stats[stats_metrics.index(mean)]['observations'] = protocol_groups.size()
    protocol_stats_s = pd.concat([stat_df.T.unstack() for stat_df in protocol_stats], keys=stats_metrics)
    pose_stats_s = pd.concat(pose_stats, keys=list(zip(stats_metrics, repeat('pose'))))
    stat_s = pd.concat([protocol_stats_s.dropna(), pose_stats_s.dropna()])  # dropna removes NaN metrics

    # change statistic names for all df that are not groupby means for the final trajectory dataframe
    for idx, stat in enumerate(stats_metrics):
        if stat != mean:
            protocol_stats[idx] = protocol_stats[idx].rename(index={protocol: f'{protocol}_{stat}'
                                                                    for protocol in unique_design_protocols})
    # trajectory_df = pd.concat([trajectory_df, pd.concat(pose_stats, axis=1).T] + protocol_stats)
    # remove std rows if there is no stdev
    number_of_trajectories = len(trajectory_df) + len(protocol_groups) + 1  # 1 for the mean
    final_trajectory_indices = trajectory_df.index.to_list() + unique_protocols + [mean]
    trajectory_df = pd.concat([trajectory_df] +
                              [df.dropna(how='all', axis=0) for df in protocol_stats] +  # v don't add if nothing
                              [pd.to_numeric(s).to_frame().T for s in pose_stats if not all(s.isna())])
    # this concat ^ puts back putils.pose_source, refine, consensus designs since protocol_stats is calculated on scores_df
    # add all docking and pose information to each trajectory, dropping the pose observations
    interface_metrics_s = pd.Series(other_pose_metrics)
    pose_metrics_df = pd.concat([interface_metrics_s] * number_of_trajectories, axis=1).T
    trajectory_df = pd.concat([pose_metrics_df.rename(index=dict(zip(range(number_of_trajectories),
                                                                     final_trajectory_indices)))
                              .drop(['observations'], axis=1), trajectory_df], axis=1)
    trajectory_df = trajectory_df.fillna({'observations': 1})

    # Calculate protocol significance
    pvalue_df = pd.DataFrame()
    scout_protocols = list(filter(re.compile(f'.*{putils.scout}').match, protocol_s.unique().tolist()))
    similarity_protocols = unique_design_protocols.difference([putils.refine, job_key] + scout_protocols)
    if putils.structure_background not in unique_design_protocols:
        pose.log.info(f'Missing background protocol "{putils.structure_background}". No protocol significance '
                      f'measurements available for this pose')
    elif len(similarity_protocols) == 1:  # measure significance
        pose.log.info("Can't measure protocol significance, only one protocol of interest")
    else:  # Test significance between all combinations of protocols by grabbing mean entries per protocol
        for prot1, prot2 in combinations(sorted(similarity_protocols), 2):
            select_df = \
                trajectory_df.loc[[design for designs in [designs_by_protocol[prot1], designs_by_protocol[prot2]]
                                   for design in designs], significance_columns]
            # prot1/2 pull out means from trajectory_df by using the protocol name
            difference_s = \
                trajectory_df.loc[prot1, significance_columns].sub(trajectory_df.loc[prot2, significance_columns])
            pvalue_df[(prot1, prot2)] = df_permutation_test(select_df, difference_s, compare='mean',
                                                            group1_size=len(designs_by_protocol[prot1]))
        pvalue_df = pvalue_df.T  # transpose significance pairs to indices and significance metrics to columns
        trajectory_df = pd.concat([trajectory_df, pd.concat([pvalue_df], keys=['similarity']).swaplevel(0, 1)])

        # Compute residue energy/sequence differences between each protocol
        residue_energy_df = residue_df.loc[:, idx_slice[:, 'energy_delta']]

        scaler = skl.preprocessing.StandardScaler()
        res_pca = skl.decomposition.PCA(variance)  # P432 designs used 0.8 percent of the variance
        residue_energy_np = scaler.fit_transform(residue_energy_df.values)
        residue_energy_pc = res_pca.fit_transform(residue_energy_np)

        seq_pca = skl.decomposition.PCA(variance)
        designed_sequence_modifications = [''.join(info['type'] for residue, info in residues_info.items()
                                                   if residue in pose.interface_design_residue_numbers)
                                           for design, residues_info in residue_info.items()]
        pairwise_sequence_diff_np = scaler.fit_transform(all_vs_all(designed_sequence_modifications, sequence_difference))
        seq_pc = seq_pca.fit_transform(pairwise_sequence_diff_np)
        # Make principal components (PC) DataFrame
        residue_energy_pc_df = pd.DataFrame(residue_energy_pc, index=residue_energy_df.index,
                                            columns=[f'pc{idx}' for idx in range(1, len(res_pca.components_) + 1)])
        seq_pc_df = pd.DataFrame(seq_pc, index=list(residue_info.keys()),
                                 columns=[f'pc{idx}' for idx in range(1, len(seq_pca.components_) + 1)])
        # Compute the euclidean distance
        # pairwise_pca_distance_np = pdist(seq_pc)
        # pairwise_pca_distance_np = SDUtils.all_vs_all(seq_pc, euclidean)

        # Merge PC DataFrames with labels
        # seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)
        seq_pc_df[putils.protocol] = protocol_s
        # residue_energy_pc_df = pd.merge(protocol_s, residue_energy_pc_df, left_index=True, right_index=True)
        residue_energy_pc_df[putils.protocol] = protocol_s
        # Next group the labels
        sequence_groups = seq_pc_df.groupby(putils.protocol)
        residue_energy_groups = residue_energy_pc_df.groupby(putils.protocol)
        # Measure statistics for each group
        # All protocol means have pairwise distance measured to access similarity
        # Gather protocol similarity/distance metrics
        sim_measures = {'sequence_distance': {}, 'energy_distance': {}}
        # sim_stdev = {}  # 'similarity': None, 'seq_distance': None, 'energy_distance': None}
        # grouped_pc_seq_df_dict, grouped_pc_energy_df_dict, similarity_stat_dict = {}, {}, {}
        for stat in stats_metrics:
            grouped_pc_seq_df = getattr(sequence_groups, stat)()
            grouped_pc_energy_df = getattr(residue_energy_groups, stat)()
            similarity_stat = getattr(pvalue_df, stat)(axis=1)  # protocol pair : stat Series
            if stat == mean:
                # for each measurement in residue_energy_pc_df, need to take the distance between it and the
                # structure background mean (if structure background, is the mean is useful too?)
                background_distance = \
                    cdist(residue_energy_pc,
                          grouped_pc_energy_df.loc[putils.structure_background, :].values[np.newaxis, :])
                trajectory_df = \
                    pd.concat([trajectory_df,
                               pd.Series(background_distance.flatten(), index=residue_energy_pc_df.index,
                                         name=f'energy_distance_from_{putils.structure_background}_mean')], axis=1)

                # if renaming is necessary
                # protocol_stats_s[stat].index = protocol_stats_s[stat].index.to_series().map(
                #     {protocol: protocol + '_' + stat for protocol in sorted(unique_design_protocols)})
                # find the pairwise distance from every point to every other point
                seq_pca_mean_distance_vector = pdist(grouped_pc_seq_df)
                energy_pca_mean_distance_vector = pdist(grouped_pc_energy_df)
                # protocol_indices_map = list(tuple(condensed_to_square(k, len(seq_pca_mean_distance_vector)))
                #                             for k in seq_pca_mean_distance_vector)
                # find similarity between each protocol by taking row average of all p-values for each metric
                # mean_pvalue_s = pvalue_df.mean(axis=1)  # protocol pair : mean significance Series
                # mean_pvalue_s.index = pd.MultiIndex.from_tuples(mean_pvalue_s.index)
                # sim_measures['similarity'] = mean_pvalue_s
                similarity_stat.index = pd.MultiIndex.from_tuples(similarity_stat.index)
                sim_measures['similarity'] = similarity_stat

                # for vector_idx, seq_dist in enumerate(seq_pca_mean_distance_vector):
                #     i, j = condensed_to_square(vector_idx, len(grouped_pc_seq_df.index))
                #     sim_measures['sequence_distance'][(grouped_pc_seq_df.index[i],
                #                                        grouped_pc_seq_df.index[j])] = seq_dist

                for vector_idx, (seq_dist, energy_dist) in enumerate(zip(seq_pca_mean_distance_vector,
                                                                         energy_pca_mean_distance_vector)):
                    i, j = condensed_to_square(vector_idx, len(grouped_pc_energy_df.index))
                    sim_measures['sequence_distance'][(grouped_pc_seq_df.index[i],
                                                       grouped_pc_seq_df.index[j])] = seq_dist
                    sim_measures['energy_distance'][(grouped_pc_energy_df.index[i],
                                                     grouped_pc_energy_df.index[j])] = energy_dist
            elif stat == std:
                # sim_stdev['similarity'] = similarity_stat_dict[stat]
                pass
                # # Todo need to square each pc, add them up, divide by the group number, then take the sqrt
                # sim_stdev['sequence_distance'] = grouped_pc_seq_df
                # sim_stdev['energy_distance'] = grouped_pc_energy_df

        # Find the significance between each pair of protocols
        protocol_sig_s = pd.concat([pvalue_df.loc[[pair], :].squeeze() for pair in pvalue_df.index.to_list()],
                                   keys=[tuple(pair) for pair in pvalue_df.index.to_list()])
        # squeeze turns the column headers into series indices. Keys appends to make a multi-index

        # Find total protocol similarity for different metrics
        # for measure, values in sim_measures.items():
        #     # measure_s = pd.Series({pair: similarity for pair, similarity in values.items()})
        #     # measure_s = pd.Series(values)
        #     similarity_sum['protocol_%s_sum' % measure] = pd.Series(values).sum()
        similarity_sum = {f'protocol_{measure}_sum': pd.Series(values).sum()
                          for measure, values in sim_measures.items()}
        similarity_sum_s = pd.concat([pd.Series(similarity_sum)], keys=[('sequence_design', 'pose')])

        # Process similarity between protocols
        sim_measures_s = pd.concat([pd.Series(values) for values in sim_measures.values()],
                                   keys=list(sim_measures.keys()))
        # # Todo test
        # sim_stdev_s = pd.concat(list(sim_stdev.values()),
        #                         keys=list(zip(repeat('std'), sim_stdev.keys()))).swaplevel(1, 2)
        # sim_series = [protocol_sig_s, similarity_sum_s, sim_measures_s, sim_stdev_s]
        sim_series = [protocol_sig_s, similarity_sum_s, sim_measures_s]

        # if self.job.figures:  # Todo ensure output is as expected then move below
        #     protocols_by_design = {design: protocol for protocol, designs in designs_by_protocol.items()
        #                            for design in designs}
        #     _path = os.path.join(job.all_scores, str(pose))
        #     # Set up Labels & Plot the PC data
        #     protocol_map = {protocol: i for i, protocol in enumerate(designs_by_protocol)}
        #     integer_map = {i: protocol for (protocol, i) in protocol_map.items()}
        #     pc_labels_group = [protocols_by_design[design] for design in residue_info]
        #     # pc_labels_group = np.array([protocols_by_design[design] for design in residue_info])
        #     pc_labels_int = [protocol_map[protocols_by_design[design]] for design in residue_info]
        #     fig = plt.figure()
        #     # ax = fig.add_subplot(111, projection='3d')
        #     ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
        #     # plt.cla()
        #
        #     # for color_int, label in integer_map.items():  # zip(pc_labels_group, pc_labels_int):
        #     #     ax.scatter(seq_pc[pc_labels_group == label, 0],
        #     #                seq_pc[pc_labels_group == label, 1],
        #     #                seq_pc[pc_labels_group == label, 2],
        #     #                c=color_int, cmap=plt.cm.nipy_spectral, edgecolor='k')
        #     scatter = ax.scatter(seq_pc[:, 0], seq_pc[:, 1], seq_pc[:, 2], c=pc_labels_int, cmap='Spectral',
        #                          edgecolor='k')
        #     # handles, labels = scatter.legend_elements()
        #     # # print(labels)  # ['$\\mathdefault{0}$', '$\\mathdefault{1}$', '$\\mathdefault{2}$']
        #     # ax.legend(handles, labels, loc='upper right', title=protocol)
        #     # # ax.legend(handles, [integer_map[label] for label in labels], loc="upper right", title=protocol)
        #     # # plt.axis('equal') # not possible with 3D graphs
        #     # plt.legend()  # No handles with labels found to put in legend.
        #     colors = [scatter.cmap(scatter.norm(i)) for i in integer_map.keys()]
        #     custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20)
        #                     for c in colors]
        #     ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left',
        #               bbox_to_anchor=(1.0, .5))
        #     # # Add group mean to the plot
        #     # for name, label in integer_map.items():
        #     #     ax.scatter(seq_pc[pc_labels_group == label, 0].mean(),
        #     #                seq_pc[pc_labels_group == label, 1].mean(),
        #     #                seq_pc[pc_labels_group == label, 2].mean(), marker='x')
        #     ax.set_xlabel('PC1')
        #     ax.set_ylabel('PC2')
        #     ax.set_zlabel('PC3')
        #     # plt.legend(pc_labels_group)
        #     plt.savefig('%s_seq_pca.png' % _path)
        #     plt.clf()
        #     # Residue PCA Figure to assay multiple interface states
        #     fig = plt.figure()
        #     # ax = fig.add_subplot(111, projection='3d')
        #     ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
        #     scatter = ax.scatter(residue_energy_pc[:, 0], residue_energy_pc[:, 1], residue_energy_pc[:, 2],
        #                          c=pc_labels_int,
        #                          cmap='Spectral', edgecolor='k')
        #     colors = [scatter.cmap(scatter.norm(i)) for i in integer_map.keys()]
        #     custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20) for c in
        #                     colors]
        #     ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left',
        #               bbox_to_anchor=(1.0, .5))
        #     ax.set_xlabel('PC1')
        #     ax.set_ylabel('PC2')
        #     ax.set_zlabel('PC3')
        #     plt.savefig('%s_res_energy_pca.png' % _path)

    # Format output and save Trajectory, Residue DataFrames, and PDB Sequences
    if job.save:
        trajectory_df.sort_index(inplace=True, axis=1)
        residue_df.sort_index(inplace=True)
        residue_df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
        residue_df[(putils.protocol, putils.protocol)] = protocol_s
        # residue_df.sort_index(inplace=True, key=lambda x: x.str.isdigit())  # put wt entry first
        if job.merge:
            trajectory_df = pd.concat([trajectory_df], axis=1, keys=['metrics'])
            trajectory_df = pd.merge(trajectory_df, residue_df, left_index=True, right_index=True)
        else:
            residue_df.to_csv(pose.path.residues)
        trajectory_df.to_csv(pose.path.trajectories)
        # pickle_object(pose_sequences, pose.path.designed_sequence_file, out_path='')
        write_sequences(pose_sequences, file_name=pose.path.designed_sequences_file)
    # Create figures
    if job.figures:  # for plotting collapse profile, errat data, contact order
        # Plot: Format the collapse data with residues as index and each design as column
        # collapse_graph_df = pd.DataFrame(per_residue_data['hydrophobic_collapse'])
        collapse_graph_df = per_residue_df.loc[:, idx_slice[:, 'hydrophobic_collapse']].droplevel(-1, axis=1)
        reference_collapse = [entity.hydrophobic_collapse for entity in pose.entities]
        reference_collapse_concatenated_s = \
            pd.Series(np.concatenate(reference_collapse), name=putils.reference_name)
        collapse_graph_df[putils.reference_name] = reference_collapse_concatenated_s
        # collapse_graph_df.columns += 1  # offset index to residue numbering
        # collapse_graph_df.sort_index(axis=1, inplace=True)
        # graph_collapse = sns.lineplot(data=collapse_graph_df)
        # g = sns.FacetGrid(tip_sumstats, col="sex", row="smoker")
        # graph_collapse = sns.relplot(data=collapse_graph_df, kind='line')  # x='Residue Number'

        # Set the base figure aspect ratio for all sequence designs
        figure_aspect_ratio = (pose_length / 25., 20)  # 20 is arbitrary size to fit all information in figure
        color_cycler = cycler(color=large_color_array)
        plt.rc('axes', prop_cycle=color_cycler)
        fig = plt.figure(figsize=figure_aspect_ratio)
        # legend_fill_value = int(15 * pose_length / 100)

        # collapse_ax, contact_ax, errat_ax = fig.subplots(3, 1, sharex=True)
        collapse_ax, errat_ax = fig.subplots(2, 1, sharex=True)
        # add the contact order to a new plot
        contact_ax = collapse_ax.twinx()
        contact_ax.plot(pose_source_contact_order_s, label='Contact Order',
                        color='#fbc0cb', lw=1, linestyle='-')  # pink
        # contact_ax.scatter(residue_indices, pose_source_contact_order_s, color='#fbc0cb', marker='o')  # pink
        # wt_contact_order_concatenated_min_s = pose_source_contact_order_s.min()
        # wt_contact_order_concatenated_max_s = pose_source_contact_order_s.max()
        # wt_contact_order_range = wt_contact_order_concatenated_max_s - wt_contact_order_concatenated_min_s
        # scaled_contact_order = ((pose_source_contact_order_s - wt_contact_order_concatenated_min_s)
        #                         / wt_contact_order_range)  # / wt_contact_order_range)
        # graph_contact_order = sns.relplot(data=errat_graph_df, kind='line')  # x='Residue Number'
        # collapse_ax1.plot(scaled_contact_order)
        # contact_ax.vlines(pose.chain_breaks, 0, 1, transform=contact_ax.get_xaxis_transform(),
        #                   label='Entity Breaks', colors='#cccccc')  # , grey)
        # contact_ax.vlines(index_residues, 0, 0.05, transform=contact_ax.get_xaxis_transform(),
        #                   label='Design Residues', colors='#f89938', lw=2)  # , orange)
        contact_ax.set_ylabel('Contact Order')
        # contact_ax.set_xlim(0, pose_length)
        contact_ax.set_ylim(0, None)
        # contact_ax.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse+contact.png'))
        # collapse_ax1.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse+contact.png'))

        # Get the plot of each collapse profile into a matplotlib axes
        # collapse_ax = collapse_graph_df.plot.line(legend=False, ax=collapse_ax, figsize=figure_aspect_ratio)
        # collapse_ax = collapse_graph_df.plot.line(legend=False, ax=collapse_ax)
        collapse_ax.plot(collapse_graph_df.T.values, label=collapse_graph_df.index)
        # collapse_ax = collapse_graph_df.plot.line(ax=collapse_ax)
        collapse_ax.xaxis.set_major_locator(MultipleLocator(20))
        collapse_ax.xaxis.set_major_formatter('{x:.0f}')
        # For the minor ticks, use no labels; default NullFormatter.
        collapse_ax.xaxis.set_minor_locator(MultipleLocator(5))
        collapse_ax.set_xlim(0, pose_length)
        collapse_ax.set_ylim(0, 1)
        # # CAN'T SET FacetGrid object for most matplotlib elements...
        # ax = graph_collapse.axes
        # ax = plt.gca()  # gca <- get current axis
        # labels = [fill(index, legend_fill_value) for index in collapse_graph_df.index]
        # collapse_ax.legend(labels, loc='lower left', bbox_to_anchor=(0., 1))
        # collapse_ax.legend(loc='lower left', bbox_to_anchor=(0., 1))
        # Plot the chain break(s) and design residues
        # linestyles={'solid', 'dashed', 'dashdot', 'dotted'}
        collapse_ax.vlines(pose.chain_breaks, 0, 1, transform=collapse_ax.get_xaxis_transform(),
                           label='Entity Breaks', colors='#cccccc')  # , grey)
        collapse_ax.vlines(index_residues, 0, 0.05, transform=collapse_ax.get_xaxis_transform(),
                           label='Design Residues', colors='#f89938', lw=2)  # , orange)
        # Plot horizontal significance
        collapse_ax.hlines([collapse_significance_threshold], 0, 1, transform=collapse_ax.get_yaxis_transform(),
                           label='Collapse Threshold', colors='#fc554f', linestyle='dotted')  # tomato
        # collapse_ax.set_xlabel('Residue Number')
        collapse_ax.set_ylabel('Hydrophobic Collapse Index')
        # collapse_ax.set_prop_cycle(color_cycler)
        # ax.autoscale(True)
        # collapse_ax.figure.tight_layout()  # no standardization
        # collapse_ax.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse.png'))  # no standardization

        # Plot: Collapse description of total profile against each design
        entity_collapse_mean, entity_collapse_std = [], []
        for entity in pose.entities:
            if entity.msa:
                collapse = entity.collapse_profile()
                entity_collapse_mean.append(collapse.mean(axis=-2))
                entity_collapse_std.append(collapse.std(axis=-2))
            else:
                break
        else:  # Only execute if we successfully looped
            profile_mean_collapse_concatenated_s = \
                pd.concat([entity_collapse_mean[idx] for idx in range(number_of_entities)], ignore_index=True)
            profile_std_collapse_concatenated_s = \
                pd.concat([entity_collapse_std[idx] for idx in range(number_of_entities)], ignore_index=True)
            profile_mean_collapse_concatenated_s.index += 1  # offset index to residue numbering
            profile_std_collapse_concatenated_s.index += 1  # offset index to residue numbering
            collapse_graph_describe_df = pd.DataFrame({
                'std_min': profile_mean_collapse_concatenated_s - profile_std_collapse_concatenated_s,
                'std_max': profile_mean_collapse_concatenated_s + profile_std_collapse_concatenated_s,
            })
            collapse_graph_describe_df.index += 1  # offset index to residue numbering
            collapse_graph_describe_df['Residue Number'] = collapse_graph_describe_df.index
            collapse_ax.vlines('Residue Number', 'std_min', 'std_max', data=collapse_graph_describe_df,
                               color='#e6e6fa', linestyle='-', lw=1, alpha=0.8)  # lavender
            # collapse_ax.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse_versus_profile.png'))

        # Plot: Errat Accuracy
        # errat_graph_df = pd.DataFrame(per_residue_data['errat_deviation'])
        # errat_graph_df = per_residue_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)
        # errat_graph_df = errat_df
        # wt_errat_concatenated_s = pd.Series(np.concatenate(list(source_errat.values())), name='clean_asu')
        # errat_graph_df[putils.pose_source] = pose_source_errat_s
        # errat_graph_df.columns += 1  # offset index to residue numbering
        errat_df.sort_index(axis=0, inplace=True)
        # errat_ax = errat_graph_df.plot.line(legend=False, ax=errat_ax, figsize=figure_aspect_ratio)
        # errat_ax = errat_graph_df.plot.line(legend=False, ax=errat_ax)
        # errat_ax = errat_graph_df.plot.line(ax=errat_ax)
        errat_ax.plot(errat_df.T.values, label=collapse_graph_df.index)
        errat_ax.xaxis.set_major_locator(MultipleLocator(20))
        errat_ax.xaxis.set_major_formatter('{x:.0f}')
        # For the minor ticks, use no labels; default NullFormatter.
        errat_ax.xaxis.set_minor_locator(MultipleLocator(5))
        # errat_ax.set_xlim(0, pose_length)
        errat_ax.set_ylim(0, None)
        # graph_errat = sns.relplot(data=errat_graph_df, kind='line')  # x='Residue Number'
        # Plot the chain break(s) and design residues
        # labels = [fill(column, legend_fill_value) for column in errat_graph_df.columns]
        # errat_ax.legend(labels, loc='lower left', bbox_to_anchor=(0., 1.))
        # errat_ax.legend(loc='lower center', bbox_to_anchor=(0., 1.))
        errat_ax.vlines(pose.chain_breaks, 0, 1, transform=errat_ax.get_xaxis_transform(),
                        label='Entity Breaks', colors='#cccccc')  # , grey)
        errat_ax.vlines(index_residues, 0, 0.05, transform=errat_ax.get_xaxis_transform(),
                        label='Design Residues', colors='#f89938', lw=2)  # , orange)
        # Plot horizontal significance
        errat_ax.hlines([errat_2_sigma], 0, 1, transform=errat_ax.get_yaxis_transform(),
                        label='Significant Error', colors='#fc554f', linestyle='dotted')  # tomato
        errat_ax.set_xlabel('Residue Number')
        errat_ax.set_ylabel('Errat Score')
        # errat_ax.autoscale(True)
        # errat_ax.figure.tight_layout()
        # errat_ax.figure.savefig(os.path.join(self.data, 'errat.png'))
        collapse_handles, collapse_labels = collapse_ax.get_legend_handles_labels()
        contact_handles, contact_labels = contact_ax.get_legend_handles_labels()
        # errat_handles, errat_labels = errat_ax.get_legend_handles_labels()
        # print(handles, labels)
        handles = collapse_handles + contact_handles
        labels = collapse_labels + contact_labels
        # handles = errat_handles + contact_handles
        # labels = errat_labels + contact_labels
        labels = [label.replace(f'{pose.name}_', '') for label in labels]
        # plt.legend(loc='upper right', bbox_to_anchor=(1, 1))  #, ncol=3, mode='expand')
        # print(labels)
        # plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -1.), ncol=3)  # , mode='expand'
        # v Why the hell doesn't this work??
        # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.), ncol=3,  # , mode='expand')
        # fig.subplots_adjust(bottom=0.1)
        plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -1.), ncol=3)  # , mode='expand')
        #            bbox_transform=plt.gcf().transFigure)  # , bbox_transform=collapse_ax.transAxes)
        fig.tight_layout()
        fig.savefig(os.path.join(pose.path.data, 'DesignMetricsPerResidues.png'))

    # After parsing data sources
    interface_metrics_s = pd.concat([interface_metrics_s], keys=[('dock', 'pose')])

    # CONSTRUCT: Create pose series and format index names
    pose_s = pd.concat([interface_metrics_s, stat_s, divergence_s] + sim_series).swaplevel(0, 1)
    # Remove pose specific metrics from pose_s, sort, and name protocol_mean_df
    pose_s.drop([putils.protocol], level=2, inplace=True, errors='ignore')
    pose_s.sort_index(level=2, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
    pose_s.sort_index(level=1, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
    pose_s.sort_index(level=0, inplace=True, sort_remaining=False)  # ascending=False
    pose_s.name = str(pose)

    return pose_s
