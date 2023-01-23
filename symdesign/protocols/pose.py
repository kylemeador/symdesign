from __future__ import annotations

import json
import logging
import os
import pickle
import re
import shutil
import warnings
from abc import ABC
from glob import glob
from itertools import combinations, repeat
from pathlib import Path
from subprocess import Popen, list2cmdline
from typing import Any, Iterable, AnyStr, Sequence

# from matplotlib.axes import Axes
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
# import seaborn as sns
import sklearn as skl
from cycler import cycler
from scipy.spatial.distance import pdist, cdist
# from sqlalchemy import select
# from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import reconstructor

from symdesign import flags, metrics, resources
from symdesign.structure import fragment
from symdesign.structure.base import Structure
from symdesign.structure.model import Pose, Models, Model, Entity
from symdesign.structure.sequence import sequence_difference, pssm_as_array, concatenate_profile, sequences_to_numeric
from symdesign.sequence import MultipleSequenceAlignment, protein_letters_3to1, read_fasta_file, write_sequences
from symdesign.structure.utils import DesignError, ClashError
from symdesign.utils import large_color_array, starttime, start_log, pickle_object, write_shell_script, \
    all_vs_all, condensed_to_square, rosetta, InputError, sql, path as putils, timestamp
from symdesign.utils.SymEntry import SymEntry, symmetry_factory, parse_symmetry_specification
# from symdesign.utils.nanohedra.general import get_components_from_nanohedra_docking

# Globals
transformation_mapping: dict[str, list[float] | list[list[float]] | np.ndarray]
logger = logging.getLogger(__name__)
pose_logger = start_log(name='pose', handler_level=3, propagate=True)
zero_offset = 1
idx_slice = pd.IndexSlice
cst_value = round(0.2 * rosetta.reference_average_residue_weight, 2)
mean, std = 'mean', 'std'
stats_metrics = [mean, std]
null_cmd = ['echo']
observed_types = ('evolution', 'fragment', 'design', 'interface')
missing_pose_transformation = "The design couldn't be transformed as it is missing the required " \
                              '"pose_transformation" attribute. Was this generated properly?'


# class PoseDirectory(ABC):  # Raises error about the subclass types not being the same...
# metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its
# bases
class PoseDirectory:
    _designed_sequences: list[Sequence]
    _id: int
    # _sym_entry: SymEntry
    _refined_pdb: str | Path
    _scouted_pdb: str | Path
    _symmetry_definition_files: list[AnyStr]
    # frag_file: str | Path
    name: str
    # pose_file: str | Path

    def __init__(self, directory: AnyStr = None, output_modifier: AnyStr = '', **kwargs):
        """

        Args:
            directory:
            output_modifier:
            initial:
        """
        if output_modifier is None:
            output_modifier = ''
        self.info: dict = {}
        """Internal state info"""
        self._info: dict = {}
        """Internal state info at load time"""

        if directory is not None:
            self.out_directory = directory
            # PoseDirectory attributes. Set after finding correct path
            self.log_path: str | Path = os.path.join(self.out_directory, f'{self.name}.log')
            self.designs_path: str | Path = os.path.join(self.out_directory, putils.designs)
            # /root/Projects/project_Poses/design/designs
            self.scripts_path: str | Path = os.path.join(self.out_directory, f'{output_modifier}{putils.scripts}')
            # /root/Projects/project_Poses/design/scripts
            self.frags_path: str | Path = os.path.join(self.out_directory, f'{output_modifier}{putils.frag_dir}')
            # /root/Projects/project_Poses/design/matching_fragments
            self.flags: str | Path = os.path.join(self.scripts_path, 'flags')
            # /root/Projects/project_Poses/design/scripts/flags
            self.data_path: str | Path = os.path.join(self.out_directory, f'{output_modifier}{putils.data}')
            # /root/Projects/project_Poses/design/data
            self.scores_file: str | Path = os.path.join(self.data_path, f'{self.name}.sc')
            # /root/Projects/project_Poses/design/data/name.sc
            self.serialized_info: str | Path = os.path.join(self.data_path, putils.state_file)
            # /root/Projects/project_Poses/design/data/info.pkl
            self.asu_path: str | Path = os.path.join(self.out_directory, putils.asu)
            # /root/Projects/project_Poses/design/asu.pdb
            # self.asu_path: str | Path = os.path.join(self.out_directory, f'{self.name}_{putils.asu}')
            # # /root/Projects/project_Poses/design/design_name_asu.pdb
            self.assembly_path: str | Path = os.path.join(self.out_directory, f'{self.name}_{putils.assembly}')
            # /root/Projects/project_Poses/design/design_name_assembly.pdb
            self.refine_pdb: str | Path = os.path.join(self.data_path, os.path.basename(self.asu_path))
            # self.refine_pdb: str | Path = f'{os.path.splitext(self.asu_path)[0]}_refine.pdb'
            # /root/Projects/project_Poses/design/clean_asu_for_refine.pdb
            self.consensus_pdb: str | Path = f'{os.path.splitext(self.asu_path)[0]}_for_consensus.pdb'
            # /root/Projects/project_Poses/design/design_name_for_consensus.pdb
            self.consensus_design_pdb: str | Path = os.path.join(self.designs_path, os.path.basename(self.consensus_pdb))
            # /root/Projects/project_Poses/design/designs/design_name_for_consensus.pdb
            self.pdb_list: str | Path = os.path.join(self.scripts_path, 'design_files.txt')
            # /root/Projects/project_Poses/design/scripts/design_files.txt
            self.design_profile_file: str | Path = os.path.join(self.data_path, 'design.pssm')
            # /root/Projects/project_Poses/design/data/design.pssm
            self.evolutionary_profile_file: str | Path = os.path.join(self.data_path, 'evolutionary.pssm')
            # /root/Projects/project_Poses/design/data/evolutionary.pssm
            self.fragment_profile_file: str | Path = os.path.join(self.data_path, 'fragment.pssm')
            # /root/Projects/project_Poses/design/data/fragment.pssm
            # These next two files may be present from NanohedraV1 outputs
            # self.pose_file = os.path.join(self.out_directory, putils.pose_file)
            # self.frag_file = os.path.join(self.frags_path, putils.frag_text_file)
            # These files are used as output from analysis protocols
            # self.designs_metrics_csv = os.path.join(self.job.all_scores, f'{self}_Trajectories.csv')
            self.designs_metrics_csv = os.path.join(self.data_path, f'designs.csv')
            self.residues_metrics_csv = os.path.join(self.data_path, f'residues.csv')
            # self.designed_sequences_file = os.path.join(self.job.all_scores, f'{self}_Sequences.pkl')
            self.designed_sequences_file = os.path.join(self.designs_path, f'sequences.fasta')

            # try:
            #     if self.initial:  # This is the first creation
            #         if os.path.exists(self.serialized_info):
            #             # This has been initialized without a database, gather existing state data
            #             try:
            #                 serial_info = unpickle(self.serialized_info)
            #                 if not self.info:  # Empty dict
            #                     self.info = serial_info
            #                 else:
            #                     serial_info.update(self.info)
            #                     self.info = serial_info
            #             except pickle.UnpicklingError as error:
            #                 logger.error(f'{self.name}: There was an issue retrieving design state from binary file...')
            #                 raise error
            #
            #             # # Make a copy to check for changes to the current state
            #             # self._info = self.info.copy()
            #             raise NotImplementedError("Still working this out")
            #             self.load_initial_model()
            #             for entity in self.initial_model:
            #                 self.entity_data.append(sql.EntityData(
            #                     pose=self,
            #                     meta=entity.metadata,
            #                     metrics=entity.metrics,
            #                     transform=EntityTransform(**transform)
            #                 ))
            #             # Todo
            #             #  if self.job.db:
            #             #      self.put_info_in_db()
            # except AttributeError:  # Missing self.initial as this was loaded from SQL database
            #     pass
        else:
            self.out_directory = os.path.join(os.getcwd(), 'temp')
            raise NotImplementedError(f"{putils.program_name} hasn't been set up to run without directories yet... "
                                      f"Please solve the {type(self).__name__}.__init__() method")

        super().__init__(**kwargs)

    # These next two file locations are used to dynamically update whether preprocessing should occur for designs
    @property
    def refined_pdb(self) -> str:
        try:
            return self._refined_pdb
        except AttributeError:
            if self.pre_refine:
                self._refined_pdb = self.asu_path
            else:
                # self.refined_pdb = None  # /root/Projects/project_Poses/design/design_name_refined.pdb
                self._refined_pdb = \
                    os.path.join(self.designs_path,
                                 f'{os.path.basename(os.path.splitext(self.asu_path)[0])}_refine.pdb')
            return self._refined_pdb

    @property
    def scouted_pdb(self) -> str:
        try:
            return self._scouted_pdb
        except AttributeError:
            if self.pre_refine:
                self._scouted_pdb = os.path.join(self.designs_path,
                                                 f'{os.path.basename(os.path.splitext(self.refined_pdb)[0])}_scout.pdb')
            else:
                # self.scouted_pdb = None  # /root/Projects/project_Poses/design/design_name_scouted.pdb
                self._scouted_pdb = f'{os.path.splitext(self.refined_pdb)[0]}_scout.pdb'
            return self._scouted_pdb

    # @property
    # def id(self) -> int:
    #     """Return the database id for the PoseJob"""
    #     try:
    #         return self._id
    #     except AttributeError:
    #         self._id = self.info.get('id')
    #         return self._id
    #
    # @id.setter
    # def id(self, _id: int):
    #     self._id = self.info['id'] = _id
    #
    # # SymEntry object attributes
    # @property
    # def sym_entry(self) -> SymEntry | None:
    #     """The SymEntry"""
    #     try:
    #         return self._sym_entry
    #     except AttributeError:
    #         self._sym_entry = symmetry_factory.get(*self.info['sym_entry_specification']) \
    #             if 'sym_entry_specification' in self.info else None
    #         # temp_sym_entry = SymEntry(self.info['sym_entry_specification'][0])
    #         # self._sym_entry = symmetry_factory(self.info['sym_entry_specification'][0],
    #         #                                    [temp_sym_entry.resulting_symmetry] +
    #         #                                    list(self.info['sym_entry_specification'][1].values())) \
    #         #     if 'sym_entry_specification' in self.info else None
    #         # self.info['sym_entry_specification'] = \
    #         #     (self.info['sym_entry_specification'][0], [temp_sym_entry.resulting_symmetry] +
    #         #      list(self.info['sym_entry_specification'][1].values()))
    #         return self._sym_entry
    #
    # @sym_entry.setter
    # def sym_entry(self, sym_entry: SymEntry):
    #     self.info['sym_entry_specification'] = self.sym_entry.number, self.sym_entry.sym_map
    #     self._sym_entry = sym_entry
    #
    # @property
    # def symmetry(self) -> str | None:
    #     """The result of the SymEntry"""
    #     try:
    #         return self.sym_entry.resulting_symmetry
    #     except AttributeError:
    #         return None
    #
    # @property
    # def sym_entry_number(self) -> int | None:
    #     """The entry number of the SymEntry"""
    #     try:
    #         return self.sym_entry.number
    #     except AttributeError:
    #         return None
    #
    # @property
    # def sym_entry_map(self) -> list[str] | None:
    #     """The symmetry map of the SymEntry"""
    #     try:
    #         # return [self.sym_entry.resulting_symmetry] + list(self.sym_entry.sym_map.values())
    #         return self.sym_entry.sym_map
    #     except AttributeError:
    #         return None
    #
    # @property
    # def sym_entry_combination(self) -> str | None:
    #     """The combination string of the SymEntry"""
    #     try:
    #         return self.sym_entry.specification
    #     except AttributeError:
    #         return None
    #
    # @property
    # def symmetry_dimension(self) -> int | None:
    #     """The dimension of the SymEntry"""
    #     try:
    #         return self.sym_entry.dimension
    #     except AttributeError:
    #         return None

    @property
    def symmetry_definition_files(self) -> list[AnyStr]:
        """Retrieve the symmetry definition files name from PoseJob"""
        try:
            return self._symmetry_definition_files
        except AttributeError:
            self._symmetry_definition_files = sorted(glob(os.path.join(self.data_path, '*.sdf')))
            return self._symmetry_definition_files

    # @property
    # def entity_names(self) -> list[str]:
    #     """Provide the names of all Entity instances in the PoseJob"""
    #     try:
    #         return self._entity_names
    #     except AttributeError:  # Get from the pose state
    #         self._entity_names = self.info.get('entity_names', [])
    #         return self._entity_names
    #
    # @entity_names.setter
    # def entity_names(self, names: list):
    #     if isinstance(names, Sequence):
    #         self._entity_names = self.info['entity_names'] = list(names)
    #     else:
    #         raise ValueError(f'The attribute entity_names must be a Sequence of str, not {type(names).__name__}')

    @property
    def designed_sequences(self) -> list[Sequence]:
        """Return the designed sequences for the entire Pose associated with the PoseJob"""
        # Currently only accessed in protocols.select_sequences
        try:
            return self._designed_sequences
        except AttributeError:
            # Todo
            #  self._designed_sequences = {seq.id: seq.seq for seq in read_fasta_file(self.designed_sequences_file)}
            self._designed_sequences = [seq_record.seq for seq_record in read_fasta_file(self.designed_sequences_file)]
            return self._designed_sequences

    def get_wildtype_file(self) -> AnyStr:
        """Retrieve the wild-type file name from PoseJob"""
        wt_file = glob(self.asu_path)
        if len(wt_file) != 1:
            raise ValueError(f'More than one matching file found during search {self.asu_path}')

        return wt_file[0]

    def get_design_files(self, design_type: str = '') -> list[AnyStr]:
        """Return the paths of all design files in a PoseJob

        Args:
            design_type: Specify if a particular type of design should be selected by a "type" string
        Returns:
            The sorted design files found in the designs directory with an absolute path
        """
        # if design_type is None:
        #     design_type = ''
        return sorted(glob(os.path.join(self.designs_path, f'*{design_type}*.pdb*')))

    def pickle_info(self):
        """Write any design attributes that should persist over program run time to serialized file"""
        # if not self.job.construct_pose:  # This is only true when self.job.nanohedra_output is True
        #     # Don't write anything as we are just querying
        #     return
        # try:
        # Todo make better patch for numpy.ndarray compare value of array is ambiguous
        if self.info.keys() != self._info.keys():  # if the state has changed from the original version
            putils.make_path(self.data_path)
            pickle_object(self.info, self.serialized_info, out_path='')
        # except ValueError:
        #     print(self.info)


# This MRO requires __init__ in PoseMetadata to pass PoseDirectory kwargs
# class PoseData(sql.PoseMetadata, PoseDirectory):
class PoseData(PoseDirectory, sql.PoseMetadata):
    # _design_indices: list[int]
    # _fragment_observations: list[fragment.db.fragment_info_type]
    _design_selector: dict[str, dict[str, dict[str, set[int] | set[str]]]] | dict
    _sym_entry: SymEntry
    # _entity_names: list[str]
    # _pose_transformation: list[transformation_mapping]
    # entity_data: list[EntityData]  # DB
    # name: str  # DB
    # project: str  # DB
    # pose_identifier: str  # DB
    # """The identifier which is created by a concatenation of the project, os.sep, and name"""
    # id: int  # DB
    # """The database row id for the 'pose_data' table"""
    _source: AnyStr
    _directives: list[dict[int, str]]
    _specific_designs: Sequence[str]
    initial_model: Model = None
    """Used if the pose structure has never been initialized previously"""
    measure_evolution: bool = False
    measure_alignment: bool = False
    pose: Pose = None
    """Contains the Pose object"""
    protocol: str = None
    """The name of the currently utilized protocol for file naming and metric results"""
    source_path: str | None
    specific_designs_file_paths: list[AnyStr] = []
    """Contains the various file paths for each design of interest according to self.specific_designs"""

    # START classmethod where the PoseData isn't initialized
    @classmethod
    def from_path(cls, path: str, project: str = None, **kwargs):
        # path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified {cls.__name__} path '{path}' wasn't found")
        filename, extension = os.path.splitext(path)
        # if 'pdb' in extension or 'cif' in extension:  # Initialize from input file
        if extension != '':  # Initialize from input file
            project_path, name = os.path.split(filename)
            if project is None:
                # path_components = filename.split(os.sep)
                remainder, project = os.path.split(project_path)
                # try:
                #     project = path_components[-2]
                # except IndexError:  # We only have a 2 index list
                if project == '':
                    raise InputError(f"Couldn't get the project from the path '{path}'. Please provide "
                                     f"project name with --{flags.project_name}")

            return cls(name=name, project=project, source_path=path, initial=True, **kwargs)
        elif os.path.isdir(path):
            # Same as from_directory. This is an existing pose_identifier that hasn't been initialized
            try:
                name, project, *_ = reversed(path.split(os.sep))
            except ValueError:  # Only got 1 value during unpacking... This isn't a "pose_directory" identifier
                raise InputError(f"Couldn't coerce {path} to a {cls.__name__}. The directory must contain the "
                                 f'"project{os.sep}pose_name" string')
            return cls(name=name, project=project, initial=True, **kwargs)  # source_path=None,
        else:
            raise InputError(f"{cls.__name__} couldn't load the specified source path '{path}'")

    @classmethod
    def from_file(cls, file: str, project: str = None, **kwargs):
        """Load the PoseJob from a Structure file including .pdb/.cif file types

        Args:
            file: The file where the PoseJob instance should load Structure instances
            project: The project where the file should be included
        Returns:
            The PoseJob instance
        """
        # file = os.path.abspath(file)
        if not os.path.exists(file):
            raise FileNotFoundError(f"The specified {cls.__name__} structure source file '{file}' wasn't found!")
        filename, extension = os.path.splitext(file)
        # if 'pdb' in extension or 'cif' in extension:  # Initialize from input file
        if extension != '':  # Initialize from input file
            project_path, name = os.path.split(filename)
        # elif os.path.isdir(design_path):  # This is a pose_name that hasn't been initialized
        #     file = None
        #     # self.project_path = os.path.dirname(design_path)
        #     # self.project = os.path.basename(self.project_path)
        #     # self.pose_identifier = f'{self.project}{os.sep}{self.name}'
        else:
            raise InputError(f"{cls.__name__} couldn't load the specified source file '{file}'")

        # if self.job.nanohedra_outputV1:
        #     # path/to/design_symmetry/building_blocks/degen/rot/tx
        #     # path_components[-4] are the oligomeric (building_blocks) names
        #     name = '-'.join(path_components[-4:])
        #     project = path_components[-5]  # if root is None else root
        #     file = os.path.join(file, putils.asu)

        if project is None:
            # path_components = filename.split(os.sep)
            remainder, project = os.path.split(project_path)
            # try:
            #     project = path_components[-2]
            # except IndexError:  # We only have a 2 index list
            if project == '':
                raise InputError(f"Couldn't get the project from the path '{file}'. Please provide "
                                 f"project name with --{flags.project_name}")

        return cls(name=name, project=project, source_path=file, initial=True, **kwargs)

    @classmethod
    def from_directory(cls, source_path: str, **kwargs):  # root: AnyStr,
        """Assumes the PoseJob is constructed from the pose_name (project/pose_name) and job.projects

        Args:
            source_path: The path to the directory where PoseJob information is stored
        Returns:
            The PoseJob instance
        """
        #     root: The base directory which contains the provided source_path
        try:
            name, project, *_ = reversed(source_path.split(os.sep))
        except ValueError:  # Only got 1 value during unpacking... This isn't a "pose_directory" identifier
            raise InputError(f"Couldn't coerce {source_path} to a {cls.__name__}. The directory must contain the "
                             f'"project{os.sep}pose_name" string')

        return cls(name=name, project=project, initial=True, **kwargs)  # source_path=os.path.join(root, source_path),

    @classmethod
    def from_name(cls, name: str = None, project: str = None, **kwargs):
        """Load the PoseJob from the name and project
        Args:
            name: The name to identify this PoseJob
            project: The project where the file should be included
        Returns:
            The PoseJob instance
        """
        return cls(name=name, project=project, initial=True, **kwargs)

    @classmethod
    def from_pose_identifier(cls, pose_identifier: str, **kwargs):
        """Load the PoseJob from the name and project
        Args:
            pose_identifier: The project and the name concatenated that identify this PoseJob
        Returns:
            The PoseJob instance
        """
        try:
            project, name = pose_identifier.split(os.sep)
        except ValueError:  # We don't have a pose_identifier
            raise InputError(f"Couldn't coerce {pose_identifier} to 'project' {os.sep} 'name'. Please ensure the "
                             f'pose_identifier is passed with the "project{os.sep}name" string')
        return cls(name=name, project=project, initial=True, **kwargs)
    # END classmethod where the PoseData hasn't been initialized before

    # def pose_string_to_path(self, root: AnyStr, pose_id: str):
    #     """Set self.out_directory to the root/poseID where the poseID is converted from dash "-" separation to path separators"""
    #     # if root is None:
    #     #     raise ValueError("No 'root' argument was passed. Can't use a pose_id without a root directory")
    #
    #     if self.job.nanohedra_output:
    #         self.out_directory = os.path.join(root, pose_id.replace('-', os.sep))
    #     else:
    #         self.out_directory = os.path.join(root, putils.projects, pose_id)  # .replace(f'_{putils.pose_directory}-')
    #         # # Dev only
    #         # if '_Designs-' in pose_id:
    #         #     self.out_directory = os.path.join(root, putils.projects, pose_id.replace('_Designs-', f'_Designs{os.sep}'))
    #         # else:
    #         #     self.out_directory = os.path.join(root, putils.projects, pose_id.replace(f'_{putils.pose_directory}-',
    #         #                                                                     f'_{putils.pose_directory}{os.sep}'))

    @reconstructor
    def __init_from_db__(self):
        """Initialize PoseData after the instance is "initialized", i.e. loaded from the database"""
        self.current_designs = []
        """Hold DesignData that has been generated in the scope of this job"""
        # Get the main program options
        self.job = resources.job.job_resources_factory.get()
        # Symmetry attributes
        # If a new sym_entry is provided it wouldn't be saved to the state but could be attempted to be used
        if self.job.sym_entry is not None:
            self.sym_entry = self.job.sym_entry

        # Design attributes
        # self.protocol = None
        # """The name of the currently utilized protocol for file naming and metric results"""
        # self.measure_evolution = self.measure_alignment = False
        # # self.entities = []
        # self.initial_model = None
        # """Used if the pose structure has never been initialized previously"""
        # self.pose = None
        # """Contains the Pose object"""
        # self.specific_designs_file_paths = []
        # """Contains the various file paths for each design of interest according to self.specific_designs"""

        if self.job.output_to_directory:
            # Todo if I use output_modifier for design, it opens up a can of worms.
            #  Maybe it is better to include only for specific modules
            output_modifier = f'{self.name}_'
            out_directory = self.job.program_root  # /output_directory <- self.out_directory/design.pdb
        else:
            output_modifier = ''  # None
            out_directory = os.path.join(self.job.projects, self.project, self.name)

        # These arguments are for PoseDirectory. initial signifies that this is the first load of this PoseJob
        # which can help in gathering the self.serialized_info file and converting this to the proper utils.sql
        # table data
        # self.initial = False
        super().__init__(directory=out_directory, output_modifier=output_modifier)  # **kwargs)

        putils.make_path(self.out_directory, condition=self.job.construct_pose)

        # Initialize the logger for the Pose
        log_path = self.log_path
        if self.job.debug:
            handler = level = 1  # Defaults to stdout, debug is level 1
            no_log_name = False
        elif self.log_path:
            if self.job.force:
                os.system(f'rm {self.log_path}')
            handler = level = 2  # To a file
            no_log_name = True
        else:  # Log to the __main__ file logger
            log_path = None  # Todo figure this out...
            handler = level = 2  # To a file
            no_log_name = False

        if self.job.skip_logging or not self.job.construct_pose:  # Set up null_logger
            self.log = logging.getLogger('null')
        else:  # f'{__name__}.{self}'
            self.log = start_log(name=f'pose.{self.project}.{self.name}', handler=handler, level=level,
                                 location=log_path, no_log_name=no_log_name, propagate=True)
            # propagate=True allows self.log to pass messages to 'pose' and 'project' logger

        # Configure standard pose loading mechanism with self.source_path
        if self.source_path is None:
            if os.path.exists(self.asu_path):  # Standard mechanism of loading the pose
                self.source_path = self.asu_path
            else:
                glob_target = os.path.join(self.out_directory, f'{self.name}*.pdb')
                source = sorted(glob(glob_target))
                if len(source) > 1:
                    raise ValueError(f'Found {len(source)} files matching the path "{glob_target}". '
                                     'No more than one expected')
                else:
                    try:
                        self.source_path = source[0]
                    except IndexError:  # glob found no files
                        self.log.debug(f"Couldn't find any Structure files matching the path '{glob_target}'")
                        self.source_path = None

    def __init__(self, name: str = None, project: str = None, source_path: AnyStr = None, initial: bool = False,
                 **kwargs):
                 # pose_transformation: Sequence[transformation_mapping] = None,
                 # entity_metadata: list[sql.EntityData] = None,
                 # entity_names: Sequence[str] = None,
                 # specific_designs: Sequence[str] = None, directives: list[dict[int, str]] = None, **kwargs):
        # pose_identifier: bool = False, initialized: bool = True,

        # PoseJob attributes
        self.name = name
        self.project = project
        self.source_path = source_path
        self.initial = initial
        # self.pose_identifier = f'{self.project}{os.sep}{self.name}'
        # self.designs = [sql.DesignData(name=name)]
        # self.designs.append(sql.DesignData(name=name))
        # self.designs.append(sql.DesignData(name=name, design_parent=None))
        # Set up original DesignData entry for the pose baseline
        pose_source = sql.DesignData(name=name, pose=self, design_parent=None)
        self.__init_from_db__()
        # Most __init__ code is called in __init_from_db__() according to sqlalchemy needs and DRY principles

        # Save job variables to the state during initialization
        if self.sym_entry:
            self.symmetry_dimension = self.sym_entry.dimension
            """The dimension of the SymEntry"""
            self.symmetry = self.sym_entry.resulting_symmetry
            """The resulting symmetry of the SymEntry"""
            self.sym_entry_specification = self.sym_entry.specification
            """The specification string of the SymEntry"""
            self.sym_entry_number = self.sym_entry.number
            """The SymEntry entry number"""
        if self.job.design_selector:
            self.design_selector = self.job.design_selector
        # # if not entity_names:  # None were provided at start up, find them
        # if not entity_metadata:  # None provided at start up
        #     # Load the Structure source_path and extract the entity_names from the Structure
        #     # if self.job.nanohedra_output:
        #     #     entity_names = get_components_from_nanohedra_docking(self.pose_file)
        #     # else:
        #     self.load_initial_model()
        #     entity_names = [entity.name for entity in self.initial_model.entities]
        #     # Todo is this efficient? NO! needs to be ProteinMetadata?!
        #     entity_stmt = select(sql.EntityData).where(sql.EntityData.name.in_(entity_names))
        #     rows = self.job.db.current_session.scalars(entity_stmt)
        #     self.entity_data.extend(rows)
        # else:
        #     # self.entity_names = entity_names
        #     self.entity_data = entity_metadata
        #
        # if pose_transformation:
        #     self.pose_transformation = pose_transformation

        # else:  # This has been initialized, gather state data
        #     try:
        #         serial_info = unpickle(self.serialized_info)
        #         if not self.info:  # Empty dict
        #             self.info = serial_info
        #         else:
        #             serial_info.update(self.info)
        #             self.info = serial_info
        #     except pickle.UnpicklingError as error:
        #         logger.error(f'{self.name}: There was an issue retrieving design state from binary file...')
        #         raise error
        #
        #     # Make a copy to check for changes to the current state
        #     self._info = self.info.copy()
        #     # self.source_path = None  # Will be set to self.asu_path later
        #     # if self.job.output_to_directory:
        #     #     # self.job.projects = ''
        #     #     # self.project_path = ''
        #     #     self.out_directory = self.job.program_root  # /output_directory<- self.out_directory /design.pdb
        #     # else:
        #     #     # self.project_path = os.path.dirname(self.out_directory)
        #     # self.out_directory = self.source_path
        #     # self.out_directory = os.path.join(self.job.projects, self.project, self.name)
        #     #     # self.job.projects = os.path.dirname(self.project_path)
        #     # self.project = os.path.basename(self.project_path)

        # # if self.specific_designs:
        # #     # Introduce flag handling current inability of specific_designs to handle iteration
        # #     self._lock_optimize_designs = True
        # #     # self.specific_designs_file_paths = []
        # #     for design in self.specific_designs:
        # #         matching_path = os.path.join(self.designs_path, f'*{design}.pdb')
        # #         matching_designs = sorted(glob(matching_path))
        # #         if matching_designs:
        # #             if len(matching_designs) > 1:
        # #                 self.log.warning(f'Found {len(matching_designs)} matching designs to your specified design '
        # #                                  f'using {matching_path}. Choosing the first {matching_designs[0]}')
        # #             for matching_design in matching_designs:
        # #                 if os.path.exists(matching_design):  # Shouldn't be necessary from glob
        # #                     self.specific_designs_file_paths.append(matching_design)
        # #                     break
        # #         else:
        # #             raise DesignError(f"Couldn't locate a specific_design matching the name '{matching_path}'")
        # #     # Format specific_designs to a pose ID compatible format
        # #     self.specific_designs = [f'{self.name}_{design}' for design in self.specific_designs]
        # #     # self.source_path = specific_designs_file_paths  # Todo?
        # #     # self.source_path = self.specific_design_path
        # # elif self.source_path is None:
        # # Configure standard pose loading mechanism with self.source_path
        # if self.source_path is None:
        #     if os.path.exists(self.asu_path):  # Standard mechanism of loading the pose
        #         self.source_path = self.asu_path
        #     else:
        #         glob_target = os.path.join(self.out_directory, f'{self.name}*.pdb')
        #         source = sorted(glob(glob_target))
        #         if len(source) > 1:
        #             raise ValueError(f'Found {len(source)} files matching the path "{glob_target}". '
        #                              'No more than one expected')
        #         else:
        #             try:
        #                 self.source_path = source[0]
        #             except IndexError:  # glob found no files
        #                 self.source_path = None
        # # else:  # If the PoseJob was loaded as .pdb/mmCIF, the structure_source should be the self.initial_model
        # #     pass

        # # Set up specific_design mechanisms
        # if specific_designs:
        #     self.specific_designs = specific_designs
        # # else:
        # #     self.specific_designs = []
        # if directives:
        #     self.directives = directives
        # # else:
        # #     self.directives = []

        # # Mark that this has been initialized if it has
        # self.pickle_info()

    def use_specific_designs(self, specific_designs: Sequence[str] = None, directives: list[dict[int, str]] = None,
                             **kwargs):
        """Set up the instance with the names and instructions to perform further sequence design

        Args:
            specific_designs: The names of designs which to include in modules that support PoseJob designs
            directives: Instructions to guide further sampling of specific_designs
        """
        if specific_designs:
            self.specific_designs = specific_designs
        # else:
        #     self.specific_designs = []
        if directives:
            self.directives = directives
        # else:
        #     self.directives = []

    def load_initial_model(self):
        """Parse the Structure at the source_path attribute"""
        if self.initial_model is None:
            if self.source_path is None:
                raise InputError(f"Couldn't {self.load_initial_model.__name__} for {self.name} as there isn't a "
                                 "specified file")
            self.initial_model = Model.from_file(self.source_path, log=self.log)

    @property
    def directives(self) -> list[dict[int, str]]:
        """The design directives given to each design in specific_designs to guide further sampling"""
        try:
            return self._directives
        except AttributeError:
            self._directives = []
            return self._directives

    @directives.setter
    def directives(self, directives: AnyStr):
        """Configure the directives"""
        if directives:
            self._directives = directives

    @property
    def specific_designs(self) -> Sequence[str]:
        """The names of designs which to include in modules that support PoseJob designs"""
        try:
            return self._specific_designs
        except AttributeError:
            self._specific_designs = []
            return self._specific_designs

    @specific_designs.setter
    def specific_designs(self, specific_designs: Iterable[str]):
        """Configure the specific_designs"""
        if specific_designs:
            # Introduce flag handling current inability of specific_designs to handle iteration
            self._lock_optimize_designs = True
            # self.specific_designs_file_paths = []
            for design in specific_designs:
                # Todo update this to use DesignData, maybe DesignProtocol.file ...
                matching_path = os.path.join(self.designs_path, f'*{design}.pdb')
                matching_designs = sorted(glob(matching_path))
                if matching_designs:
                    if len(matching_designs) > 1:
                        self.log.warning(f'Found {len(matching_designs)} matching designs to your specified design '
                                         f'using {matching_path}. Choosing the first {matching_designs[0]}')
                    for matching_design in matching_designs:
                        if os.path.exists(matching_design):  # Shouldn't be necessary from glob
                            self.specific_designs_file_paths.append(matching_design)
                            break
                else:
                    raise DesignError(f"Couldn't locate a specific_design matching the name '{matching_path}'")
            # Format specific_designs to a pose ID compatible format
            self.specific_designs = [f'{self.name}_{design}' for design in self.specific_designs]
            # self.source_path = specific_designs_file_paths  # Todo?
            # self.source_path = self.specific_design_path

    @property
    def structure_source(self) -> AnyStr:
        """Return the source of the Pose structural information for the PoseJob"""
        try:
            return self._source
        except AttributeError:
            self._source = 'Database'
            return self._source

    @structure_source.setter
    def structure_source(self, source: AnyStr):
        self._source = source

    # SymEntry object attributes
    @property  # @hybrid_property
    def sym_entry(self) -> SymEntry | None:
        """The SymEntry"""
        try:
            return self._sym_entry
        except AttributeError:
            try:
                self._sym_entry = symmetry_factory.get(self.sym_entry_number,
                                                       parse_symmetry_specification(self.sym_entry_specification))
            except AttributeError:  # self.sym_entry_specification is None?
                return None  # No symmetry specified
            return self._sym_entry

    @sym_entry.setter
    def sym_entry(self, sym_entry: SymEntry):
        if isinstance(sym_entry, SymEntry):
            self._sym_entry = sym_entry
        else:
            raise InputError(f"Couldn't set the 'sym_entry' attribute with a type {type(sym_entry).__name__}. "
                             f"Expected a SymEntry instance")

    def is_symmetric(self) -> bool:
        """Is the PoseJob symmetric?"""
        return self.sym_entry is not None

    @property
    def pose_kwargs(self) -> dict[str, Any]:
        """Returns the kwargs necessary to initialize the Pose"""
        return dict(sym_entry=self.sym_entry, log=self.log, design_selector=self.design_selector,
                    # entity_metadata=self.entity_data,
                    entity_names=[data.meta.entity_id for data in self.entity_data],  # self.entity_names,
                    transformations=[data.transformation for data in self.entity_data],  # self.pose_transformation,
                    ignore_clashes=self.job.design.ignore_pose_clashes, fragment_db=self.job.fragment_db)
        #             api_db=self.job.api_db,

    # @property
    # def pose_transformation(self) -> list[transformation_mapping]:
    #     """Provide the transformation parameters for each Entity in the PoseData Pose
    #
    #     Returns:
    #         [{'rotation': np.ndarray, 'translation': np.ndarray, 'rotation2': np.ndarray,
    #           'translation2': np.ndarray}, ...]
    #         A list with the transformations of each Entity in the Pose according to the symmetry
    #     """
    #     try:
    #         return self._pose_transformation
    #     except AttributeError:  # Get from the pose state
    #         self._pose_transformation = self.info.get('pose_transformation', [])
    #         return self._pose_transformation
    #
    # @pose_transformation.setter
    # def pose_transformation(self, transform: Sequence[transformation_mapping]):
    #     if all(isinstance(operation_set, dict) for operation_set in transform):
    #         self._pose_transformation = self.info['pose_transformation'] = list(transform)
    #     else:
    #         try:
    #             raise ValueError(f'The attribute pose_transformation must be a Sequence of '
    #                              f'{transformation_mapping.__name__}, not {type(transform[0]).__name__}')
    #         except TypeError:  # Not a Sequence
    #             raise TypeError(f'The attribute pose_transformation must be a Sequence of '
    #                             f'{transformation_mapping.__name__}, not {type(transform).__name__}')

    @property
    def design_selector(self) -> dict[str, dict[str, dict[str, set[int] | set[str]]]] | dict:
        """Provide the design_selector parameters for the design in question

        Returns:
            {}, A mapping of the selection criteria in the Pose according to set up
        """
        try:
            return self._design_selector
        except AttributeError:  # Get from the pose state
            self._design_selector = self.info.get('design_selector', {})
            return self._design_selector

    @design_selector.setter
    def design_selector(self, design_selector: dict):
        if isinstance(design_selector, dict):
            self._design_selector = self.info['design_selector'] = design_selector
        else:
            raise ValueError(f'The attribute design_selector must be a dict, not {type(design_selector).__name__}')

    # @property
    # def design_indices(self) -> list[int]:
    #     """Provide the design_indices for the design in question
    #
    #     Returns:
    #         All the indices in the design which are considered designable
    #     """
    #     try:
    #         return self._design_indices
    #     except AttributeError:  # Get from the pose state
    #         self._design_indices = self.info.get('design_indices', [])
    #         return self._design_indices
    #
    # @design_indices.setter
    # def design_indices(self, design_indices: Sequence[int]):
    #     if isinstance(design_indices, Sequence):
    #         self._design_indices = self.info['design_indices'] = list(design_indices)
    #     else:
    #         raise ValueError(f'The attribute design_indices must be a Sequence type, not '
    #                          f'{type(design_indices).__name__}')

    # @property
    # def fragment_observations(self) -> list | None:
    #     """Provide the observed fragments as measured from the Pose
    #
    #     Returns:
    #         The fragment instances observed from the pose
    #             Ex: [{'cluster': (1, 2, 24), 'mapped': 78, 'paired': 287, 'match':0.46843}, ...]
    #     """
    #     try:
    #         return self._fragment_observations
    #     except AttributeError:  # Get from the pose state
    #         self._fragment_observations = self.info.get('fragments', None)  # None signifies query wasn't attempted
    #         return self._fragment_observations
    #
    # @fragment_observations.setter
    # def fragment_observations(self, fragment_observations: list):
    #     if isinstance(fragment_observations, list):
    #         self._fragment_observations = self.info['fragments'] = fragment_observations
    #     else:
    #         raise ValueError(f'The attribute fragment_observations must be a list, not '
    #                          f'{type(fragment_observations).__name__}')
    #
    # @property
    # def number_of_fragments(self) -> int:
    #     return len(self.fragment_observations) if self.fragment_observations else 0

    # Both pre_* properties are really implemented to take advantage of .setter
    @property
    def pre_refine(self) -> bool:
        """Provide the state attribute regarding the source files status as "previously refined"

        Returns:
            Whether refinement has occurred
        """
        # return self._pre_refine
        return all(data.meta.pre_refine for data in self.entity_data)
        # try:
        #     return self._pre_refine
        # except AttributeError:  # Get from the pose state
        #     self._pre_refine = self.info.get('pre_refine', True)
        #     return self._pre_refine

    # @pre_refine.setter
    # def pre_refine(self, pre_refine: bool):
    #     if isinstance(pre_refine, bool):
    #         self._pre_refine = self.info['pre_refine'] = pre_refine
    #         if pre_refine:
    #             self.refined_pdb = self.asu_path
    #             self.scouted_pdb = os.path.join(self.designs_path,
    #                                             f'{os.path.basename(os.path.splitext(self.refined_pdb)[0])}_scout.pdb')
    #     elif pre_refine is None:
    #         pass
    #     else:
    #         raise ValueError(f'The attribute pre_refine must be a boolean or NoneType, not {type(pre_refine).__name__}')

    @property
    def pre_loop_model(self) -> bool:
        """Provide the state attribute regarding the source files status as "previously loop modeled"

        Returns:
            Whether loop modeling has occurred
        """
        # return self._pre_loop_model
        return all(data.meta.pre_loop_model for data in self.entity_data)
        # try:
        #     return self._pre_loop_model
        # except AttributeError:  # Get from the pose state
        #     self._pre_loop_model = self.info.get('pre_loop_model', True)
        #     return self._pre_loop_model
    #
    # @pre_loop_model.setter
    # def pre_loop_model(self, pre_loop_model: bool):
    #     if isinstance(pre_loop_model, bool):
    #         self._pre_loop_model = self.info['pre_loop_model'] = pre_loop_model
    #         # if pre_loop_model:
    #         #     do_something
    #     elif pre_loop_model is None:
    #         pass
    #     else:
    #         raise ValueError(f'The attribute pre_loop_model must be a boolean or NoneType, not '
    #                          f'{type(pre_loop_model).__name__}')

    def get_designs_without_structure(self) -> list[sql.DesignData]:
        """For each design, access whether there is a structure that exists for it. If not, return the design

        Returns:
            Each instance of the DesignData that is missing a structure
        """
        missing_sequences = []
        for design in self.designs:
            if design.structure_path and os.path.exists(design.structure_path):
                continue
            else:
                missing_sequences.append(design)

        return missing_sequences

    def transform_entities_to_pose(self, **kwargs) -> list[Entity]:
        """Take the set of entities involved in a pose composition and transform them from a standard reference frame to
        the Pose reference frame using the pose_transformation attribute

        Keyword Args:
            refined: bool = True - Whether to use refined models from the StructureDatabase
            oriented: bool = False - Whether to use oriented models from the StructureDatabase
        """
        entities = self.get_entities(**kwargs)
        if self.transformations:  # pose_transformation:
            self.log.debug('Entities were transformed to the found docking parameters')
            entities = [entity.get_transformed_copy(**transformation)
                        for entity, transformation in zip(entities, self.transformations)]
        else:  # Todo change below to handle asymmetric cases...
            # raise SymmetryError("The design couldn't be transformed as it is missing the required "
            self.log.error(missing_pose_transformation)
        return entities

    def transform_structures_to_pose(self, structures: Iterable[Structure], **kwargs) -> list[Structure]:
        """Take a set of Structure instances and transform them from a standard reference frame to the Pose reference
        frame using the pose_transformation attribute

        Args:
            structures: The Structure objects you would like to transform
        Returns:
            The transformed Structure objects if a transformation was possible
        """
        if self.transformations:  # pose_transformation:
            self.log.debug('Structures were transformed to the found docking parameters')
            # Todo assumes a 1:1 correspondence between structures and transforms (component group numbers) CHANGE
            return [structure.get_transformed_copy(**transformation)
                    for structure, transformation in zip(structures, self.transformations)]
        else:
            # raise SymmetryError("The design couldn't be transformed as it is missing the required "
            self.log.error(missing_pose_transformation)
            return list(structures)

    def get_entities(self, refined: bool = True, oriented: bool = False, **kwargs) -> list[Entity]:
        """Retrieve Entity files from the design Database using either the oriented directory, or the refined directory.
        If these don't exist, use the Pose directory, and load them into job for further processing

        Args:
            refined: Whether to use the refined directory
            oriented: Whether to use the oriented directory
        Returns:
            The list of Entity instances that belong to this PoseData
        """
        # Todo change to rely on EntityData
        source_preference = ['refined', 'oriented_asu', 'design']
        # Todo once loop_model works 'full_models'
        # if self.job.structure_db:
        if refined:
            source_idx = 0
        elif oriented:
            source_idx = 1
        else:
            source_idx = 2
            self.log.info(f'Falling back on Entity instances present in the {type(self).__name__} structure_source')

        # self.entities.clear()
        entities = []
        # for name in self.entity_names:
        for data in self.entity_data:
            name = data.name
            source_preference_iter = iter(source_preference)
            # Discard however many sources are unwanted (source_idx number)
            for it in range(source_idx):
                _ = next(source_preference_iter)

            model = None
            while not model:
                try:
                    source = next(source_preference_iter)
                except StopIteration:
                    raise DesignError(f"{self.get_entities.__name__}: Couldn't locate the required files")
                source_datastore = getattr(self.job.structure_db, source, None)
                # Todo this course of action isn't set up anymore. It should be depreciated...
                if source_datastore is None:  # Try to get file from the PoseDirectory
                    search_path = os.path.join(self.out_directory, f'{name}*.pdb*')
                    file = sorted(glob(search_path))
                    if file:
                        if len(file) > 1:
                            self.log.warning(f'The specified entity has multiple files at "{search_path}". '
                                             f'Using the first')
                        model = Model.from_file(file[0], log=self.log)
                    else:
                        raise FileNotFoundError(f"Couldn't locate the specified entity at '{search_path}'")
                else:
                    model = source_datastore.retrieve_data(name=name)
                    # Todo I ran into an error where the EntityID loaded from 2gtr_1.pdb was 2gtr_1_1
                    #  This might help resolve this issue
                    # model_file = source_datastore.retrieve_file(name=name)
                    # if model_file:
                    #     model = Entity.from_file(model_file)
                    # else:
                    #     model = None
                    if isinstance(model, Structure):  # Model):  # Entity):
                        self.log.info(f'Found Model at {source} DataStore and loaded into job')
                    else:
                        self.log.error(f"Couldn't locate the Model {name} at the Database source "
                                       f'"{source_datastore.location}"')

            entities.extend([entity for entity in model.entities])
        # if source_idx == 0:
        #     self.pre_refine = True
        # if source_idx == 0:  # Todo
        #     self.pre_loop_model = True

        # self.log.debug(f'{len(entities)} matching entities found')
        if len(entities) != len(self.entity_data):
            raise RuntimeError(f'Expected {len(entities)} entities, but found {len(self.entity_data)}')

        return entities
        # else:  # Todo not reachable. Consolidate this with above as far as iterative mechanism
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
        #                 out_dir = self.out_directory
        #                 self.log.debug('Couldn\'t find entities in the oriented directory')
        #
        #     if not refined and not oriented:
        #         out_dir = self.out_directory
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

    def load_pose(self, file: str = None, entities: list[Structure] = None):
        """For the design info given by a PoseJob source, initialize the Pose with self.source_path file,
        self.symmetry, self.job, self.fragment_database, and self.log objects

        Handles Pose clash testing, writing the Pose, adding state variables to the pose

        Args:
            file: The file path to a structure_source file
            entities: The Entities desired in the Pose
        """
        # Check to see if Pose is already loaded and nothing new provided
        if self.pose and file is None and entities is None:
            return

        # rename_chains = True  # Because the result of entities, we should rename
        if entities is not None:
            pass  # Use the entities as provided
        elif self.source_path is None or not os.path.exists(self.source_path):
            # In case we initialized design without a .pdb or clean_asu.pdb (Nanohedra)
            if not self.job.structure_db:
                raise RuntimeError(f"Couldn't {self.get_entities.__name__} as there was no "
                                   f"{resources.structure_db.StructureDatabase.__name__}"
                                   f" attached to the {type(self).__name__}")
            self.log.info(f'No structure source_path file found. Fetching structure_source from '
                          f'{type(self.job.structure_db).__name__} and transforming to Pose')
            # Minimize I/O with transform...
            entities = self.transform_entities_to_pose()
            # entities = self.entities
            # entities = []
            # for entity in self.entities:
            #     entities.extend(entity.entities)
            # # Because the file wasn't specified on the way in, no chain names should be binding
            # # rename_chains = True

        # Initialize the Pose with the pdb in PDB numbering so that residue_selectors are respected
        # name = f'{self.pose_identifier}-asu' if self.sym_entry else self.pose_identifier
        # name = self.pose_identifier
        # name = self.name  # Ensure this name is the tracked across Pose init from fragdock() to _design() methods
        if entities:
            self.structure_source = 'Database'
            self.pose = Pose.from_entities(entities, name=self.name, **self.pose_kwargs)
        elif self.initial_model:  # This is a fresh Model, and we already loaded so reuse
            self.structure_source = self.source_path
            # Careful, if processing has occurred to the initial_model, then this may be wrong!
            self.pose = Pose.from_model(self.initial_model, entity_info=self.initial_model.entity_info,
                                        name=self.name, **self.pose_kwargs)
        else:
            self.structure_source = file if file else self.source_path
            self.pose = Pose.from_file(self.structure_source, name=self.name, **self.pose_kwargs)

        if self.pose.is_symmetric():
            self._symmetric_assembly_is_clash()
            if self.job.output_assembly:
                self.pose.write(assembly=True, out_path=self.assembly_path,
                                increment_chains=self.job.increment_chains)
                self.log.info(f'Symmetric assembly written to: "{self.assembly_path}"')
            if self.job.write_oligomers:  # Write out new oligomers to the PoseJob
                for idx, entity in enumerate(self.pose.entities):
                    oligomer_path = os.path.join(self.out_directory, f'{entity.name}_oligomer.pdb')
                    entity.write(oligomer=True, out_path=oligomer_path)
                    self.log.info(f'Entity {entity.name} oligomer written to: "{oligomer_path}"')

            # If we have an empty list for the pose_transformation, save the identified transformations from the Pose
            if not any(self.transformations):
                for data, transformation in zip(self.entity_data, self.pose.entity_transformations):
                    # Make an empty EntityTransform
                    data.transform = sql.EntityTransform()
                    data.transform.transformation = transformation
        # # Then modify numbering to ensure standard and accurate use during protocols
        # # self.pose.pose_numbering()
        # if not self.entity_names:  # Store the entity names if they were never generated
        #     self.entity_names = [entity.name for entity in self.pose.entities]
        #     self.log.info(f'Input Entities: {", ".join(self.entity_names)}')

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
                    raise DesignError(f"Your requested fusion of chain {fusion_nterm} with chain {fusion_cterm} didn't "
                                      f"work!")
                    # self.log.critical('Your requested fusion of chain %s with chain %s didn\'t work!' %
                    #                   (fusion_nterm, fusion_cterm))
                else:  # won't be accessed unless entity_new_chain_idx is set
                    self.pose.entities[entity_new_chain_idx].chain_id = fusion_nterm
            # except AttributeError:
            #     raise ValueError('One or both of the chain IDs %s were not found in the input model. Possible chain'
            #                      ' ID\'s are %s' % ((fusion_nterm, fusion_cterm), ','.join(new_asu.chain_ids)))
        self.pose.write(out_path=self.asu_path)
        self.log.info(f'Cleaned PDB: "{self.asu_path}"')

    def _symmetric_assembly_is_clash(self):
        """Wrapper around the Pose symmetric_assembly_is_clash() to check at the Pose level for clashes and raise
        ClashError if any are found, otherwise, continue with protocol
        """
        if self.pose.symmetric_assembly_is_clash():
            if self.job.design.ignore_symmetric_clashes:
                self.log.critical('The Symmetric Assembly contains clashes. The Pose from '
                                  f"'{self.structure_source}' isn't viable")
            else:
                raise ClashError("The Symmetric Assembly contains clashes! Design won't be considered. If you "
                                 'would like to generate the Assembly anyway, re-submit the command with '
                                 f'--{flags.ignore_symmetric_clashes}')

    def generate_evolutionary_profile(self, warn_metrics: bool = False):
        """Add evolutionary profile information for each Entity to the Pose

        Args:
            warn_metrics: Whether to warn the user about missing files for metric collection
        """
        if self.measure_evolution and self.measure_alignment:
            # We have already set and succeeded
            return

        # Assume True given this function call and set False if not possible for one of the Entities
        self.measure_evolution = self.measure_alignment = True
        warn = False
        for entity in self.pose.entities:
            if entity not in self.pose.active_entities:  # We shouldn't design, add a null profile instead
                entity.add_profile(null=True)
                continue

            if entity.evolutionary_profile:
                continue

            profile = self.job.api_db.hhblits_profiles.retrieve_data(name=entity.name)
            if not profile:
                # # We can try and add... This would be better at the program level due to memory issues
                # entity.add_evolutionary_profile(out_dir=self.job.api_db.hhblits_profiles.location)
                # if not entity.pssm_file:
                #     # Still no file found. this is likely broken
                #     # raise DesignError(f'{entity.name} has no profile generated. To proceed with this design/'
                #     #                   f'protocol you must generate the profile!')
                #     pass
                self.measure_evolution = False
                warn = True
                entity.evolutionary_profile = entity.create_null_profile()
            else:
                entity.evolutionary_profile = profile
                # Ensure the file is attached as well
                entity.pssm_file = self.job.api_db.hhblits_profiles.retrieve_file(name=entity.name)

            if not entity.verify_evolutionary_profile():
                entity.fit_evolutionary_profile_to_structure()
            if not entity.sequence_file:
                entity.write_sequence_to_fasta('reference', out_dir=self.job.api_db.sequences.location)

            if entity.msa:
                continue

            try:  # To fetch the multiple sequence alignment for further processing
                msa = self.job.api_db.alignments.retrieve_data(name=entity.name)
                if not msa:
                    self.measure_alignment = False
                    warn = True
                else:
                    entity.msa = msa
            except ValueError as error:  # When the Entity reference sequence and alignment are different lengths
                # raise error
                self.log.info(f'Entity reference sequence and provided alignment are different lengths: {error}')
                warn = True

        if warn_metrics and warn:
            if not self.measure_evolution and not self.measure_alignment:
                self.log.info("Metrics relying on evolution aren't being collected as the required files weren't "
                              f'found. These include: {", ".join(metrics.all_evolutionary_metrics)}')
            elif not self.measure_alignment:
                self.log.info('Metrics relying on a multiple sequence alignment including: '
                              f'{", ".join(metrics.multiple_sequence_alignment_dependent_metrics)}'
                              "are being calculated with the reference sequence as there was no MSA found")
            else:
                self.log.info("Metrics relying on an evolutionary profile aren't being collected as "
                              'there was no profile found. These include: '
                              f'{", ".join(metrics.profile_dependent_metrics)}')

        # if self.measure_evolution:
        self.pose.evolutionary_profile = \
            concatenate_profile([entity.evolutionary_profile for entity in self.pose.entities])
        # else:
        #     self.pose.evolutionary_profile = self.pose.create_null_profile()

    def generate_fragments(self, interface: bool = False):
        """For the design info given by a PoseJob source, initialize the Pose then generate interfacial fragment
        information between Entities. Aware of symmetry and design_selectors in fragment generation file

        Args:
            interface: Whether to perform fragment generation on the interface
        """
        if interface:
            self.pose.generate_interface_fragments()
        else:
            self.pose.generate_fragments()

        if self.job.write_fragments:
            putils.make_path(self.frags_path)
            # Write trajectory if specified
            if self.job.write_trajectory:
                # Create a Models instance to collect each model
                trajectory_models = Models()

                if self.sym_entry.unit_cell:
                    self.log.warning('No unit cell dimensions applicable to the trajectory file.')

                trajectory_models.write(out_path=os.path.join(self.frags_path, 'all_frags.pdb'),
                                        oligomer=True)

                ghost_frags = [ghost_frag for ghost_frag, _, _ in self.pose.fragment_pairs]
                fragment.visuals.write_fragments_as_multimodel(ghost_frags, os.path.join(self.frags_path, 'all_frags.pdb'))
                # for frag_idx, (ghost_frag, frag, match) in enumerate(self.pose.fragment_pairs):
                #     continue
            else:
                self.pose.write_fragment_pairs(out_path=self.frags_path)

        # self.fragment_observations = self.pose.get_fragment_observations()
        # # Todo move to ProtocolMetaData?
        # self.info['fragment_source'] = self.job.fragment_db.source
        # self.pickle_info()  # Todo remove once PoseJob state can be returned to the SymDesign dispatch w/ MP

    def __key(self) -> str:
        return self.pose_identifier

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        raise NotImplementedError(f"Can't compare {self.__class__} instance to {other.__class__} instance")

    def __hash__(self) -> int:
        return hash(self.__key())

    def __str__(self) -> str:
        # if self.job.nanohedra_output:
        #     return self.source_path.replace(f'{self.job.nanohedra_root}{os.sep}', '').replace(os.sep, '-')
        # elif self.job.output_to_directory:
        #     return self.name
        # else:
        #     return self.out_directory.replace(f'{self.job.projects}{os.sep}', '').replace(os.sep, '-')
        return self.pose_identifier


class PoseProtocol(PoseData):

    def identify_interface(self):
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
        self.pose.find_and_split_interface()

        # self.interface_design_residue_numbers = set()  # Replace set(). Add new residues
        # for number, residues_entities in self.pose.split_interface_residues.items():
        #     self.interface_design_residue_numbers.update([residue.number for residue, _ in residues_entities])
        # self.log.debug(f'Found interface design residues: '
        #                f'{", ".join(map(str, sorted(self.interface_design_residue_numbers)))}')

        # self.interface_residue_numbers = set()  # Replace set(). Add new residues
        # for entity in self.pose.entities:
        #     # Tod0 v clean as it is redundant with analysis and falls out of scope
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
        # Relies on PoseDirecotry attributes
        # .designs_path
        # .refined_pdb
        # .scores_file
        # .design_profile_file
        # .fragment_profile_file
        number_of_residues = self.pose.number_of_residues
        self.log.info(f'Total number of residues in Pose: {number_of_residues}')

        # Get ASU distance parameters
        if self.symmetry_dimension:  # Check for None and dimension 0 simultaneously
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

        if self.pose.is_symmetric():
            if symmetry_protocol is None:
                symmetry_protocols = {0: 'make_point_group', 2: 'make_layer', 3: 'make_lattice',
                                      None: 'asymmetric'}  # -1: 'asymmetric'
                symmetry_protocol = symmetry_protocols[self.symmetry_dimension]
            variables.append(('symmetry', symmetry_protocol))
            # The current self.sym_entry can still be None if requested for this particular job
            if sym_def_file is None and self.sym_entry is not None:
                sym_def_file = self.sym_entry.sdf_lookup()
                variables.append(('sdf', sym_def_file))

            self.log.info(f'Symmetry Option: {symmetry_protocol}')
            out_of_bounds_residue = number_of_residues*self.pose.number_of_symmetry_mates + 1
        else:
            if symmetry_protocol is not None:
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

        rosetta_flags = rosetta.rosetta_flags.copy()
        if pdb_out_path:
            rosetta_flags.extend([f'-out:path:pdb {pdb_out_path}', f'-scorefile {self.scores_file}'])
        else:
            rosetta_flags.extend([f'-out:path:pdb {self.designs_path}', f'-scorefile {self.scores_file}'])
        rosetta_flags.append(f'-in:file:native {self.refined_pdb}')
        rosetta_flags.append(f'-parser:script_vars {" ".join(f"{var}={val}" for var, val in variables)}')

        putils.make_path(out_dir)
        out_file = os.path.join(out_dir, 'flags')
        with open(out_file, 'w') as f:
            f.write('%s\n' % '\n'.join(rosetta_flags))

        return out_file

    def generate_entity_metrics_commands(self, base_command) -> list[list[str] | None]:
        """Use the Pose state to generate metrics commands for each Entity instance

        Args:
            base_command: The base command to build Entity metric commands off of
        Returns:
            The formatted command for every Entity in the Pose
        """
        # self.entity_names not dependent on Pose load
        if len(self.entity_data) == 1:  # There is no unbound state to query as only one entity
            return []
        if len(self.symmetry_definition_files) != len(self.entity_data) or self.job.force:
            for entity in self.pose.entities:
                if entity.is_oligomeric():  # make symmetric energy in line with SymDesign energies v
                    entity.make_sdf(out_path=self.data_path,
                                    modify_sym_energy_for_cryst=True if self.sym_entry.dimension in [2, 3] else False)
                # Todo monitor if Rosetta energy modifier changed from 2x for crystal set up and adjust accordingly
                else:
                    shutil.copy(os.path.join(putils.symmetry_def_files, 'C1.sym'),
                                os.path.join(self.data_path, f'{entity.name}.sdf'))

        entity_metric_commands = []
        for idx, data in enumerate(self.entity_data, 1):
            name = data.name
            if self.is_symmetric():
                entity_sdf = f'sdf={os.path.join(self.data_path, f"{name}.sdf")}'
                entity_sym = 'symmetry=make_point_group'
            else:
                entity_sdf, entity_sym = '', 'symmetry=asymmetric'
            metric_cmd = base_command \
                + ['-parser:script_vars', 'repack=yes', f'entity={idx}', entity_sym] \
                + ([entity_sdf] if entity_sdf != '' else [])
            self.log.info(f'Metrics Command for Entity {name}: {list2cmdline(metric_cmd)}')
            entity_metric_commands.append(metric_cmd)

        return entity_metric_commands

    def make_analysis_cmd(self) -> list[str]:
        """Generate a list compatible with subprocess.Popen()/subprocess.list2cmdline()"""
        return ['python', putils.program_exe, putils.process_rosetta_metrics, '--single', self.out_directory]

    def thread_sequences_to_backbone(self, sequences: dict[str, str] = None):
        """From the starting Pose, thread sequences onto the backbone, modifying relevant side chains i.e., mutate the
        Pose and build/pack using Rosetta FastRelax. If no sequences are provided, will use self.designed_sequences

        Args:
            sequences: A mapping of sequence alias to it's sequence. These will be used for producing outputs and as
                the input sequence
        """
        if sequences is None:  # Gather all already designed sequences
            # refine_sequences = unpickle(self.designed_sequences_file)
            sequences = {seq.id: seq.seq for seq in read_fasta_file(self.designed_sequences_file)}

        # Write each "threaded" structure out for further processing
        number_of_residues = self.pose.number_of_residues
        design_files = []
        for sequence_id, sequence in sequences.items():
            if len(sequence) != number_of_residues:
                raise DesignError(f'The length of the sequence {len(sequence)} != {number_of_residues}, '
                                  f'the number of residues in the pose')
            for res_idx, residue_type in enumerate(sequence):
                self.pose.mutate_residue(index=res_idx, to=residue_type)
            # pre_threaded_file = os.path.join(self.data_path, f'{self.name}_{self.protocol}{seq_idx:04d}.pdb')
            pre_threaded_file = os.path.join(self.data_path, f'{sequence_id}.pdb')
            design_files.append(self.pose.write(out_path=pre_threaded_file))

        # Ensure that mutations to the Pose are wiped. We can reload if continuing to use
        self.pose = None

        # if self.protocol is not None:  # This hasn't been set yet
        self.protocol = 'thread'
        design_files_file = os.path.join(self.scripts_path, f'{timestamp()}_{self.protocol}_files.txt')
        putils.make_path(self.scripts_path)

        # # Modify each sequence score to reflect the new "decoy" name
        # # Todo update as a consequence of new SQL
        # sequence_ids = sequences.keys()
        # design_scores = metrics.read_scores(self.scores_file)
        # for design, scores in design_scores.items():
        #     if design in sequence_ids:
        #         # We previously saved data. Copy to the identifier that is present after threading
        #         scores['decoy'] = f'{design}_{self.protocol}'
        #         # write_json(_scores, self.scores_file)
        #         with open(self.scores_file, 'a') as f_save:
        #             json.dump(scores, f_save)  # , **kwargs)
        #             # Ensure JSON lines are separated by newline
        #             f_save.write('\n')

        if design_files:
            with open(design_files_file, 'w') as f:
                f.write('%s\n' % '\n'.join(design_files))
        else:
            raise DesignError(f'{self.thread_sequences_to_backbone.__name__}: No designed sequences were located')

        # self.refine(in_file_list=design_files_file)
        self.refine(design_files=design_files)

    def predict_structure(self):
        """"""
        if self.current_designs:
            sequences = [design.sequence for design in self.current_designs]
        else:
            sequences = self.get_designs_without_structure()

        match self.job.predict.method:
            case ['thread', 'proteinmpnn']:
                self.thread_sequences_to_backbone(sequences)
            # Todo
            #  case 'alphafold':
            #  Sequences use within alphafold requires .fasta...
            #      self.run_alphafold()
            case _:
                raise NotImplementedError(f"For {self.predict_structure.__name__}, the method {self.job.predict.method}"
                                          " isn't implemented yet")

    def interface_design_analysis(self, designs: Iterable[Pose] | Iterable[AnyStr] = None) -> pd.Series:
        """Retrieve all score information from a PoseJob and write results to .csv file

        Args:
            designs: The designs to perform analysis on. By default, fetches all available structures
        Returns:
            Series containing summary metrics for all designs
        """
        self.load_pose()
        # self.identify_interface()

        # Load fragment_profile into the analysis
        if self.job.design.term_constraint and not self.pose.fragment_queries:
            self.generate_fragments(interface=True)
            self.pose.calculate_fragment_profile()

        # CAUTION: Assumes each structure is the same length
        # Todo get residues_df['design_indices']
        pose_length = self.pose.number_of_residues
        residue_indices = list(range(pose_length))
        # residue_numbers = [residue.number for residue in self.pose.residues]
        # interface_residue_indices = [residue.index for residue in self.pose.interface_residues]

        # Find all designs files
        # Todo fold these into Model(s) and attack metrics from Pose objects?
        if designs is None:
            designs = [Pose.from_file(file, **self.pose_kwargs) for file in self.get_design_files()]  # Todo PoseJob(.path)

        pose_sequences = {pose.name: pose.sequence for pose in designs}
        sequences_df = self.analyze_sequence_metrics_per_design(pose_sequences)
        # all_mutations = generate_mutations_from_reference(self.pose.sequence, pose_sequences,
        #                                                   zero_index=True, return_to=True)

        entity_energies = [0. for _ in self.pose.entities]
        pose_source_residue_info = \
            {residue.index: {'complex': 0., 'bound': entity_energies.copy(), 'unbound': entity_energies.copy(),
                             'solv_complex': 0., 'solv_bound': entity_energies.copy(),
                             'solv_unbound': entity_energies.copy(), 'fsp': 0., 'cst': 0., 'hbond': 0}
             for entity in self.pose.entities for residue in entity.residues}
        pose_name = self.pose.name
        residue_info = {pose_name: pose_source_residue_info}

        # Gather miscellaneous pose specific metrics
        # other_pose_metrics = self.pose.calculate_metrics()
        # Create metrics for the pose_source
        empty_source = dict(
            # **other_pose_metrics,
            buns_complex=0,
            # buns_unbound=0,
            contact_count=0,
            favor_residue_energy=0,
            interaction_energy_complex=0,
            interaction_energy_per_residue=0,
            interface_separation=0,
            number_of_hbonds=0,
            rmsd_complex=0,  # Todo calculate this here instead of Rosetta using superposition3d
            rosetta_reference_energy=0,
            shape_complementarity=0,
        )
        job_key = 'no_energy'
        empty_source[putils.protocol] = job_key
        for idx, entity in enumerate(self.pose.entities, 1):
            empty_source[f'buns{idx}_unbound'] = 0
            empty_source[f'entity{idx}_interface_connectivity'] = 0

        source_df = pd.DataFrame(empty_source, index=[pose_name])

        # Get the metrics from the score file for each design
        if os.path.exists(self.scores_file):  # Rosetta scores file is present  # Todo PoseJob(.path)
            self.log.debug(f'Found design scores in file: {self.scores_file}')  # Todo PoseJob(.path)
            design_was_performed = True
            # # Get the scores from the score file on design trajectory metrics
            # source_df = pd.DataFrame.from_dict({pose_name: {putils.protocol: job_key}}, orient='index')
            # for idx, entity in enumerate(self.pose.entities, 1):
            #     source_df[f'buns{idx}_unbound'] = 0
            #     source_df[f'entity{idx}_interface_connectivity'] = 0
            # source_df['buns_complex'] = 0
            # # source_df['buns_unbound'] = 0
            # source_df['contact_count'] = 0
            # source_df['favor_residue_energy'] = 0
            # # Used in sum_per_residue_df
            # # source_df['interface_energy_complex'] = 0
            # source_df['interaction_energy_complex'] = 0
            # source_df['interaction_energy_per_residue'] = \
            #     source_df['interaction_energy_complex'] / len(self.pose.interface_residues)
            # source_df['interface_separation'] = 0
            # source_df['number_of_hbonds'] = 0
            # source_df['rmsd_complex'] = 0
            # source_df['rosetta_reference_energy'] = 0
            # source_df['shape_complementarity'] = 0
            design_scores = metrics.read_scores(self.scores_file)  # Todo PoseJob(.path)
            self.log.debug(f'All designs with scores: {", ".join(design_scores.keys())}')
            # Find designs with scores and structures
            structure_design_scores = {}
            for pose in designs:
                try:
                    structure_design_scores[pose.name] = design_scores.pop(pose.name)
                except KeyError:  # Structure wasn't scored, we will remove this later
                    pass

            # Create protocol dataframe
            scores_df = pd.DataFrame.from_dict(structure_design_scores, orient='index')
            # # Fill in all the missing values with that of the default pose_source
            # scores_df = pd.concat([source_df, scores_df]).fillna(method='ffill')
            # Gather all columns into specific types for processing and formatting
            per_res_columns = []
            # hbonds_columns = []
            for column in scores_df.columns.to_list():
                if 'res_' in column:  # if column.startswith('per_res_'):
                    per_res_columns.append(column)
                # elif column.startswith('hbonds_res_selection'):
                #     hbonds_columns.append(column)

            # Check proper input
            metric_set = metrics.necessary_metrics.difference(set(scores_df.columns))
            # self.log.debug('Score columns present before required metric check: %s' % scores_df.columns.to_list())
            if metric_set:
                raise DesignError(f'Missing required metrics: "{", ".join(metric_set)}"')

            # Remove unnecessary (old scores) as well as Rosetta pose score terms besides ref (has been renamed above)
            # Todo learn know how to produce Rosetta score terms in output score file. Not in FastRelax...
            remove_columns = metrics.rosetta_terms + metrics.unnecessary + per_res_columns
            # Todo remove dirty when columns are correct (after P432)
            #  and column tabulation precedes residue/hbond_processing
            # residue_info = {'energy': {'complex': 0., 'unbound': 0.}, 'type': None, 'hbond': 0}
            residue_info.update(self.pose.rosetta_residue_processing(structure_design_scores))
            # Can't use residue_processing (clean) ^ in the case there is a design without metrics... columns not found!
            interface_hbonds = metrics.dirty_hbond_processing(structure_design_scores)
            # Can't use hbond_processing (clean) in the case there is a design without metrics... columns not found!
            # interface_hbonds = hbond_processing(structure_design_scores, hbonds_columns)
            # Convert interface_hbonds to indices
            interface_hbonds = {design: [residue.index for residue in self.pose.get_residues(hbond_residues)]
                                for design, hbond_residues in interface_hbonds.items()}
            residue_info = metrics.process_residue_info(residue_info, hbonds=interface_hbonds)
            # residue_info = metrics.incorporate_sequence_info(residue_info, pose_sequences)

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

            self.log.debug(f'Viable designs with structures remaining after cleaning:\n\t{", ".join(viable_designs)}')
            pose_sequences = {design: sequence for design, sequence in pose_sequences.items() if
                              design in viable_designs}

            # Todo implement this protocol if sequence data is taken at multiple points along a trajectory and the
            #  sequence data along trajectory is a metric on it's own
            # # Gather mutations for residue specific processing and design sequences
            # for design, data in list(structure_design_scores.items()):  # make a copy as can be removed
            #     sequence = data.get('final_sequence')
            #     if sequence:
            #         if len(sequence) >= pose_length:
            #             pose_sequences[design] = sequence[:pose_length]  # Todo won't work if design had insertions
            #         else:
            #             pose_sequences[design] = sequence
            #     else:
            #         self.log.warning('Design %s is missing sequence data, removing from design pool' % design)
            #         structure_design_scores.pop(design)
            # # format {entity: {design_name: sequence, ...}, ...}
            # entity_sequences = \
            #     {entity: {design: sequence[entity.n_terminal_residue.number - 1:entity.c_terminal_residue.number]
            #               for design, sequence in pose_sequences.items()} for entity in self.pose.entities}
        else:
            self.log.debug(f'Missing design scores file at {self.scores_file}')  # Todo PoseJob(.path)
            design_was_performed = False
            # Todo add relevant missing scores such as those specified as 0 below
            # Todo may need to put source_df in scores file alternative
            # source_df = pd.DataFrame.from_dict({pose_name: {putils.protocol: job_key}}, orient='index')
            # for idx, entity in enumerate(self.pose.entities, 1):
            #     source_df[f'buns{idx}_unbound'] = 0
            #     source_df[f'entity{idx}_interface_connectivity'] = 0
            #     # residue_info = {'energy': {'complex': 0., 'unbound': 0.}, 'type': None, 'hbond': 0}
            #     # design_info.update({residue.number: {'energy_delta': 0.,
            #     #                                      'type': protein_letters_3to1.get(residue.type),
            #     #                                      'hbond': 0} for residue in entity.residues})
            # source_df['buns_complex'] = 0
            # # source_df['buns_unbound'] = 0
            # source_df['contact_count'] = 0
            # source_df['favor_residue_energy'] = 0
            # # source_df['interface_energy_complex'] = 0
            # source_df['interaction_energy_complex'] = 0
            # source_df['interaction_energy_per_residue'] = \
            #     source_df['interaction_energy_complex'] / len(self.pose.interface_residues)
            # source_df['interface_separation'] = 0
            # source_df['number_of_hbonds'] = 0
            # source_df['rmsd_complex'] = 0
            # source_df['rosetta_reference_energy'] = 0
            # source_df['shape_complementarity'] = 0
            scores_df = pd.DataFrame.from_dict({pose.name: {putils.protocol: job_key} for pose in designs},
                                               orient='index')
            # # Fill in all the missing values with that of the default pose_source
            # scores_df = pd.concat([source_df, scores_df]).fillna(method='ffill')

            remove_columns = metrics.rosetta_terms + metrics.unnecessary
            residue_info.update({struct_name: pose_source_residue_info for struct_name in scores_df.index.to_list()})
            # Todo generate energy scores internally which matches output from residue_processing
            viable_designs = [pose.name for pose in designs]

        scores_df.drop(remove_columns, axis=1, inplace=True, errors='ignore')

        # Find protocols for protocol specific data processing removing from scores_df
        protocol_s = scores_df.pop(putils.protocol).copy()
        designs_by_protocol = protocol_s.groupby(protocol_s).groups
        # unique_protocols = list(designs_by_protocol.keys())
        # Remove refine and consensus if present as there was no design done over multiple protocols
        # Todo change if we did multiple rounds of these protocols
        designs_by_protocol.pop(putils.refine, None)
        designs_by_protocol.pop(putils.consensus, None)
        # Get unique protocols
        unique_design_protocols = set(designs_by_protocol.keys())
        self.log.info(f'Unique Design Protocols: {", ".join(unique_design_protocols)}')

        # Replace empty strings with np.nan and convert remaining to float
        scores_df.replace('', np.nan, inplace=True)
        scores_df.fillna(dict(zip(metrics.protocol_specific_columns, repeat(0))), inplace=True)
        # scores_df = scores_df.astype(float)  # , copy=False, errors='ignore')
        # Fill in all the missing values with that of the default pose_source
        scores_df = pd.concat([source_df, scores_df]).fillna(method='ffill')

        # atomic_deviation = {}
        # pose_assembly_minimally_contacting = self.pose.assembly_minimally_contacting
        # perform SASA measurements
        # pose_assembly_minimally_contacting.get_sasa()
        # assembly_asu_residues = pose_assembly_minimally_contacting.residues[:pose_length]
        # per_residue_data['sasa_hydrophobic_complex'][pose_name] = \
        #     [residue.sasa_apolar for residue in assembly_asu_residues]
        # per_residue_data['sasa_polar_complex'][pose_name] = [residue.sasa_polar for residue in assembly_asu_residues]
        # per_residue_data['sasa_relative_complex'][pose_name] = \
        #     [residue.relative_sasa for residue in assembly_asu_residues]

        # Grab metrics for the pose source. Checks if self.pose was designed
        # Favor pose source errat/collapse on a per-entity basis if design occurred
        # As the pose source assumes no legit interface present while designs have an interface
        # per_residue_sasa_unbound_apolar, per_residue_sasa_unbound_polar, per_residue_sasa_unbound_relative = [], [], []
        # source_errat_accuracy, inverse_residue_contact_order_z = [], []

        per_residue_data: dict[str, dict[str, Any]] = \
            {pose_name: {**self.pose.per_residue_interface_surface_area(),
                         **self.pose.per_residue_contact_order()}
             }

        number_of_entities = self.pose.number_of_entities
        if design_was_performed:  # The input structure was not meant to be together, treat as such
            source_errat = []
            for idx, entity in enumerate(self.pose.entities):
                # Replace 'errat_deviation' measurement with uncomplexed entities
                # oligomer_errat_accuracy, oligomeric_errat = entity_oligomer.errat(out_path=os.path.devnull)
                # source_errat_accuracy.append(oligomer_errat_accuracy)
                # Todo when Entity.oligomer works
                #  _, oligomeric_errat = entity.oligomer.errat(out_path=os.path.devnull)
                entity_oligomer = Model.from_chains(entity.chains, entities=False, log=self.pose.log)
                _, oligomeric_errat = entity_oligomer.errat(out_path=os.path.devnull)
                source_errat.append(oligomeric_errat[:entity.number_of_residues])
            # atomic_deviation[pose_name] = sum(source_errat_accuracy) / float(number_of_entities)
            pose_source_errat = np.concatenate(source_errat)
        else:
            # pose_assembly_minimally_contacting = self.pose.assembly_minimally_contacting
            # # atomic_deviation[pose_name], pose_per_residue_errat = \
            # _, pose_per_residue_errat = \
            #     pose_assembly_minimally_contacting.errat(out_path=os.path.devnull)
            # pose_source_errat = pose_per_residue_errat[:pose_length]
            # Get errat measurement
            # per_residue_data[pose_name].update(self.pose.per_residue_interface_errat())
            pose_source_errat = self.pose.per_residue_interface_errat()['errat_deviation']

        per_residue_data[pose_name]['errat_deviation'] = pose_source_errat

        # Compute structural measurements for all designs
        interface_local_density = {pose_name: self.pose.local_density_interface()}
        for pose in designs:  # Takes 1-2 seconds for Structure -> assembly -> errat
            # Must find interface residues before measure local_density
            pose.find_and_split_interface()
            per_residue_data[pose.name] = pose.per_residue_interface_surface_area()
            # Get errat measurement
            per_residue_data[pose.name].update(pose.per_residue_interface_errat())
            # Todo remove Rosetta
            #  This is a measurement of interface_connectivity like from Rosetta
            interface_local_density[pose.name] = pose.local_density_interface()

        scores_df['interface_local_density'] = pd.Series(interface_local_density)

        # Load profiles of interest into the analysis
        if self.job.design.evolution_constraint:
            self.generate_evolutionary_profile(warn_metrics=True)

        # self.generate_fragments() was already called
        self.pose.calculate_profile()

        profile_background = {'design': pssm_as_array(self.pose.profile),
                              'evolution': pssm_as_array(self.pose.evolutionary_profile),
                              'fragment': self.pose.fragment_profile.as_array()}
        if self.job.fragment_db is not None:
            interface_bkgd = np.array(list(self.job.fragment_db.aa_frequencies.values()))
            profile_background['interface'] = np.tile(interface_bkgd, (self.pose.number_of_residues, 1))

        # Calculate hydrophobic collapse for each design
        # Include the pose_source in the measured designs
        if self.measure_evolution:
            # This is set for figures as well
            hydrophobicity = 'expanded'
        else:
            hydrophobicity = 'standard'
        contact_order_per_res_z, reference_collapse, collapse_profile = \
            self.pose.get_folding_metrics(hydrophobicity=hydrophobicity)
        if self.measure_evolution:  # collapse_profile.size:  # Not equal to zero, use the profile instead
            reference_collapse = collapse_profile
        #     reference_mean = np.nanmean(collapse_profile, axis=-2)
        #     reference_std = np.nanstd(collapse_profile, axis=-2)
        # else:
        #     reference_mean = reference_std = None
        entity_sequences = []
        for entity in self.pose.entities:
            entity_slice = slice(entity.n_terminal_residue.index, 1 + entity.c_terminal_residue.index)
            entity_sequences.append({design: sequence[entity_slice] for design, sequence in pose_sequences.items()})

        all_sequences_split = [list(designed_sequences.values()) for designed_sequences in entity_sequences]
        all_sequences_by_entity = list(zip(*all_sequences_split))

        folding_and_collapse = \
            metrics.collapse_per_residue(all_sequences_by_entity, contact_order_per_res_z, reference_collapse)
        per_residue_collapse_df = pd.concat({design_id: pd.DataFrame(data, index=residue_indices)
                                             for design_id, data in zip(viable_designs, folding_and_collapse)},
                                            ).unstack().swaplevel(0, 1, axis=1)

        # Convert per_residue_data into a dataframe matching residues_df orientation
        residues_df = pd.concat({name: pd.DataFrame(data, index=residue_indices)
                                for name, data in per_residue_data.items()}).unstack().swaplevel(0, 1, axis=1)
        # Fill in missing pose_source metrics for each design not calculated in rosetta
        # residues_df.fillna(residues_df.loc[pose_name, idx_slice[:, 'contact_order']], inplace=True)
        residues_df.fillna(residues_df.loc[pose_name, :], inplace=True)

        residues_df = residues_df.join(per_residue_collapse_df)

        # Process mutational frequencies, H-bond, and Residue energy metrics to dataframe
        rosetta_info_df = pd.concat({design: pd.DataFrame(info) for design, info in residue_info.items()}).unstack()
        # returns multi-index column with residue number as first (top) column index, metric as second index
        # during residues_df unstack, all residues with missing dicts are copied as nan
        # Merge interface design specific residue metrics with total per residue metrics
        # residues_df = pd.merge(residues_df, rosetta_info_df, left_index=True, right_index=True)

        # Join each residues_df like dataframe
        # Each of these can have difference index, so we use concat to perform an outer merge
        residues_df = pd.concat([residues_df, sequences_df, rosetta_info_df], axis=1)
        # # Join rosetta_info_df and sequence metrics
        # residues_df = residues_df.join([rosetta_info_df, sequences_df])

        if not profile_background:
            divergence_s = pd.Series(dtype=float)
        else:  # Calculate sequence statistics
            # First, for entire pose
            pose_alignment = MultipleSequenceAlignment.from_dictionary(pose_sequences)
            observed, divergence = \
                metrics.calculate_sequence_observations_and_divergence(pose_alignment, profile_background)
            observed_dfs = []
            for profile, observed_values in observed.items():
                # scores_df[f'observed_{profile}'] = observed_values.mean(axis=1)
                observed_dfs.append(pd.DataFrame(observed_values, index=viable_designs,
                                                 columns=pd.MultiIndex.from_product([residue_indices,
                                                                                     [f'observed_{profile}']]))
                                    )
            # Add observation information into the residues_df
            residues_df = residues_df.join(observed_dfs)
            # Get pose sequence divergence
            design_residue_indices = [residue.index for residue in self.pose.design_residues]
            pose_divergence_s = pd.concat([pd.Series({f'{divergence_type}_per_residue':
                                                      _divergence[design_residue_indices].mean()
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
                                           metrics.position_specific_divergence(protocol_alignment.frequencies, bgd)
                                           for profile, bgd in profile_background.items()}
                    # if interface_bkgd is not None:
                    #     protocol_divergence['divergence_interface'] = \
                    #         metrics.position_specific_divergence(protocol_alignment.frequencies, tiled_int_background)
                    # Get per residue divergence metric by protocol
                    divergence_by_protocol[protocol] = \
                        {f'{divergence_type}_per_residue': divergence[design_residue_indices].mean()
                         for divergence_type, divergence in protocol_divergence.items()}
                # new = dfd.columns.to_frame()
                # new.insert(0, 'new2_level_name', new_level_values)
                # dfd.columns = pd.MultiIndex.from_frame(new)
                protocol_divergence_s = \
                    pd.concat([pd.DataFrame(divergence_by_protocol).unstack()], keys=['sequence_design'])
            else:
                protocol_divergence_s = pd.Series(dtype=float)

            # Get profile mean observed
            # Perform a frequency extraction for each background profile
            per_residue_background_frequency_df = \
                pd.concat([pd.DataFrame(pose_alignment.get_probabilities_from_profile(background), index=viable_designs,
                                        columns=pd.MultiIndex.from_product([residue_indices, [f'observed_{profile}']]))
                           for profile, background in profile_background.items()], axis=1)
            # Todo
            #  Ensure that only interface residues are selected, not only by those that are 0 as interface can be 0!
            observed_frequencies_from_fragment_profile = profile_background.pop('fragment', None)
            if observed_frequencies_from_fragment_profile:
                observed_frequencies_from_fragment_profile[observed_frequencies_from_fragment_profile == 0] = np.nan
                # Todo RuntimeWarning: Mean of empty slice
                scores_df['observed_fragment_mean_probability'] = \
                    np.nanmean(observed_frequencies_from_fragment_profile, axis=1)
            for profile, background in profile_background.items():
                scores_df[f'observed_{profile}_mean_probability'] = scores_df[f'observed_{profile}'] / pose_length
                # scores_df['observed_evolution_mean_probability'] = scores_df['observed_evolution'] / pose_length

            divergence_s = pd.concat([protocol_divergence_s, pose_divergence_s])

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
            try:  # Todo add these to the analysis
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
                                               index=viable_designs, columns=residue_indices)
            # dca_concatenated_df.columns = pd.MultiIndex.from_product([dca_concatenated_df.columns, ['dca_energy']])
            dca_concatenated_df = pd.concat([dca_concatenated_df], keys=['dca_energy'], axis=1).swaplevel(0, 1, axis=1)
            # Merge with residues_df
            residues_df = pd.merge(residues_df, dca_concatenated_df, left_index=True, right_index=True)

        # Make buried surface area (bsa) columns, and residue classification
        residues_df = metrics.calculate_residue_surface_area(residues_df)

        # Calculate new metrics from combinations of other metrics
        # Add design residue information to scores_df such as how many core, rim, and support residues were measured
        mean_columns = ['hydrophobicity']  # , 'sasa_relative_bound', 'sasa_relative_complex']
        scores_df = scores_df.join(metrics.sum_per_residue_metrics(residues_df, mean_metrics=mean_columns))

        # scores_df['collapse_new_positions'] /= pose_length
        # scores_df['collapse_new_position_significance'] /= pose_length
        scores_df['collapse_significance_by_contact_order_z_mean'] = \
            scores_df['collapse_significance_by_contact_order_z'] / \
            (residues_df.loc[:, idx_slice[:, 'collapse_significance_by_contact_order_z']] != 0).sum(axis=1)
        # if self.measure_alignment:
        # Todo THESE ARE NOW DIFFERENT SOURCE if not self.measure_alignment
        collapse_increased_df = residues_df.loc[:, idx_slice[:, 'collapse_increased_z']]
        total_increased_collapse = (collapse_increased_df != 0).sum(axis=1)
        # scores_df['collapse_increase_significance_by_contact_order_z_mean'] = \
        #     scores_df['collapse_increase_significance_by_contact_order_z'] / total_increased_collapse
        # scores_df['collapse_increased_z'] /= pose_length
        scores_df['collapse_increased_z_mean'] = \
            collapse_increased_df.sum(axis=1) / total_increased_collapse
        scores_df['collapse_variance'] = scores_df['collapse_deviation_magnitude'] / pose_length
        scores_df['collapse_sequential_peaks_z_mean'] = \
            scores_df['collapse_sequential_peaks_z'] / total_increased_collapse
        scores_df['collapse_sequential_z_mean'] = \
            scores_df['collapse_sequential_z'] / total_increased_collapse

        # scores_df['interface_area_total'] = bsa_assembly_df = \
        #     scores_df['interface_area_polar'] + scores_df['interface_area_hydrophobic']
        # Find the proportion of the residue surface area that is solvent accessible versus buried in the interface
        bsa_assembly_df = scores_df['interface_area_total']
        scores_df['interface_area_to_residue_surface_ratio'] = \
            (bsa_assembly_df / (bsa_assembly_df + scores_df['area_total_complex']))

        # Make scores_df errat_deviation that takes into account the pose_source sequence errat_deviation
        # Include in errat_deviation if errat score is < 2 std devs and isn't 0 to begin with
        # Get per-residue errat scores from the residues_df
        errat_df = residues_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)

        pose_source_errat = errat_df.loc[pose_name, :]
        source_errat_inclusion_boolean = \
            np.logical_and(pose_source_errat < metrics.errat_2_sigma, pose_source_errat != 0.)
        # Find residues where designs deviate above wild-type errat scores
        errat_sig_df = errat_df.sub(pose_source_errat, axis=1) > metrics.errat_1_sigma  # axis=1 is per-residue subtract
        # Then select only those residues which are expressly important by the inclusion boolean
        # This overwrites the metrics.sum_per_residue_metrics() value
        scores_df['errat_deviation'] = (errat_sig_df.loc[:, source_errat_inclusion_boolean] * 1).sum(axis=1)

        # Calculate mutational content
        mutation_df = residues_df.loc[:, idx_slice[:, 'mutation']]
        # scores_df['number_of_mutations'] = mutation_df.sum(axis=1)
        scores_df['percent_mutations'] = scores_df['number_of_mutations'] / pose_length

        idx = 1
        # prior_slice = 0
        for idx, entity in enumerate(self.pose.entities, idx):
            # entity_n_terminal_residue_index = entity.n_terminal_residue.index
            # entity_c_terminal_residue_index = entity.c_terminal_residue.index
            scores_df[f'entity{idx}_number_of_mutations'] = \
                mutation_df.loc[:, idx_slice[residue_indices[entity.n_terminal_residue.index:  # prior_slice
                                                             1 + entity.c_terminal_residue.index], :]].sum(axis=1)
            # prior_slice = entity_c_terminal_residue_index
            scores_df[f'entity{idx}_percent_mutations'] = \
                scores_df[f'entity{idx}_number_of_mutations'] / entity.number_of_residues

        # scores_df['number_of_mutations'] = \
        #     pd.Series({design: len(mutations) for design, mutations in all_mutations.items()})
        # scores_df['percent_mutations'] = scores_df['number_of_mutations'] / pose_length
        #
        # idx = 1
        # for idx, entity in enumerate(self.pose.entities, idx):
        #     entity_c_terminal_residue_index = entity.c_terminal_residue.index
        #     scores_df[f'entity{idx}_number_of_mutations'] = \
        #         pd.Series(
        #             {design: len([1 for mutation_idx in mutations if mutation_idx <= entity_c_terminal_residue_index])
        #              for design, mutations in all_mutations.items()})
        #     scores_df[f'entity{idx}_percent_mutations'] = \
        #         scores_df[f'entity{idx}_number_of_mutations'] / scores_df[f'entity{idx}_number_of_residues']
        #     #     scores_df[f'entity{idx}_number_of_mutations'] / other_pose_metrics[f'entity{idx}_number_of_residues']

        # Check if any columns are > 50% interior (value can be 0 or 1). If so, return True for that column
        # interior_residue_df = residues_df.loc[:, idx_slice[:, 'interior']]
        # interior_residue_numbers = \
        #     interior_residues.loc[:, interior_residues.mean(axis=0) > 0.5].columns.remove_unused_levels().levels[0].
        #     to_list()
        # if interior_residue_numbers:
        #     self.log.info(f'Design Residues {",".join(map(str, sorted(interior_residue_numbers)))}
        #                   'are located in the interior')

        # This shouldn't be much different from the state variable self.interface_residue_numbers
        # perhaps the use of residue neighbor energy metrics adds residues which contribute, but not directly
        # interface_residues = set(residues_df.columns.levels[0].unique()).difference(interior_residue_numbers)

        # Add design residue information to scores_df such as how many core, rim, and support residues were measured
        # for residue_class in metrics.residue_classification:
        #     scores_df[residue_class] = residues_df.loc[:, idx_slice[:, residue_class]].sum(axis=1)

        # Calculate metrics from combinations of metrics with variable integer number metric names
        scores_columns = scores_df.columns.to_list()
        self.log.debug(f'Metrics present: {scores_columns}')
        # sum columns using list[0] + list[1] + list[n]
        # residues_df = residues_df.drop([column
        #                                 for columns in [complex_df.columns, bound_df.columns, unbound_df.columns,
        #                                                 solvation_complex_df.columns, solvation_bound_df.columns,
        #                                                 solvation_unbound_df.columns]
        #                                 for column in columns], axis=1)
        summation_pairs = \
            {'buns_unbound': list(filter(re.compile('buns[0-9]+_unbound$').match, scores_columns)),  # Rosetta
             # 'interface_energy_bound':
             #     list(filter(re.compile('interface_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
             # 'interface_energy_unbound':
             #     list(filter(re.compile('interface_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
             # 'interface_solvation_energy_bound':
             #     list(filter(re.compile('solvation_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
             # 'interface_solvation_energy_unbound':
             #     list(filter(re.compile('solvation_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
             'interface_connectivity':
                 list(filter(re.compile('entity[0-9]+_interface_connectivity').match, scores_columns)),  # Rosetta
             }

        scores_df = metrics.columns_to_new_column(scores_df, summation_pairs)
        scores_df = metrics.columns_to_new_column(scores_df, metrics.rosetta_delta_pairs, mode='sub')
        # Add number_interface_residues for div_pairs and int_comp_similarity
        # scores_df['number_interface_residues'] = other_pose_metrics.pop('number_interface_residues')
        scores_df = metrics.columns_to_new_column(scores_df, metrics.rosetta_division_pairs, mode='truediv')
        scores_df['interface_composition_similarity'] = \
            scores_df.apply(metrics.interface_composition_similarity, axis=1)
        scores_df.drop(metrics.clean_up_intermediate_columns, axis=1, inplace=True, errors='ignore')
        repacking = scores_df.get('repacking')
        if repacking is not None:
            # Set interface_bound_activation_energy = np.nan where repacking is 0
            # Currently is -1 for True (Rosetta Filter quirk...)
            scores_df.loc[scores_df[repacking == 0].index, 'interface_bound_activation_energy'] = np.nan
            scores_df.drop('repacking', axis=1, inplace=True)

        # Process dataframes for missing values and drop refine trajectory if present
        # refine_index = scores_df[scores_df[putils.protocol] == putils.refine].index
        # scores_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
        # residues_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
        # residue_info.pop(putils.refine, None)  # Remove refine from analysis
        # residues_no_frags = residues_df.columns[residues_df.isna().all(axis=0)].remove_unused_levels().levels[0]
        # Remove completely empty columns such as obs_interface
        residues_df.dropna(how='all', inplace=True, axis=1)
        residues_df = residues_df.fillna(0.).copy()
        # residue_indices_no_frags = residues_df.columns[residues_df.isna().all(axis=0)]

        # POSE ANALYSIS
        # scores_df = pd.concat([scores_df, proteinmpnn_df], axis=1)
        scores_df.dropna(how='all', inplace=True, axis=1)  # Remove completely empty columns
        # Refine is not considered sequence design and destroys mean. remove v
        # designs_df = scores_df.sort_index().drop(putils.refine, axis=0, errors='ignore')
        # Consensus cst_weights are very large and destroy the mean.
        # Remove this drop for consensus or refine if they are run multiple times
        designs_df = \
            scores_df.drop([pose_name, putils.refine, putils.consensus], axis=0, errors='ignore').sort_index()

        # Get total design statistics for every sequence in the pose and every protocol specifically
        scores_df[putils.protocol] = protocol_s
        protocol_groups = scores_df.groupby(putils.protocol)
        # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        # print(designs_df.select_dtypes(exclude=numerics))

        pose_stats, protocol_stats = [], []
        for idx, stat in enumerate(stats_metrics):
            # Todo both groupby calls have this warning
            #  FutureWarning: The default value of numeric_only in DataFrame.mean is deprecated. In a future version,
            #  it will default to False. In addition, specifying 'numeric_only=None' is deprecated. Select only valid
            #  columns or specify the value of numeric_only to silence this warning.
            pose_stats.append(getattr(designs_df, stat)().rename(stat))
            protocol_stats.append(getattr(protocol_groups, stat)())

        # Add the number of observations of each protocol
        protocol_stats[stats_metrics.index(mean)]['observations'] = protocol_groups.size()
        # Format stats_s for final pose_s Series
        protocol_stats_s = pd.concat([stat_df.T.unstack() for stat_df in protocol_stats], keys=stats_metrics)
        pose_stats_s = pd.concat(pose_stats, keys=list(zip(stats_metrics, repeat('pose'))))
        pose_stats_s[('mean', 'pose', 'observations')] = len(viable_designs)
        stat_s = pd.concat([protocol_stats_s.dropna(), pose_stats_s.dropna()])  # dropna removes NaN metrics

        # Change statistic names for all df that are not groupby means for the final trajectory dataframe
        for idx, stat in enumerate(stats_metrics):
            if stat != mean:
                protocol_stats[idx] = protocol_stats[idx].rename(index={protocol: f'{protocol}_{stat}'
                                                                        for protocol in unique_design_protocols})
        # Remove std rows if there is no stdev
        # number_of_trajectories = len(designs_df) + len(protocol_groups) + 1  # 1 for the mean
        # final_trajectory_indices = designs_df.index.to_list() + unique_protocols + [mean]
        designs_df = pd.concat([designs_df]
                               + [df.dropna(how='all', axis=0) for df in protocol_stats]  # v don't add if nothing
                               + [pd.to_numeric(s).to_frame().T for s in pose_stats if not all(s.isna())])
        # This concat puts back pose_name, refine, consensus index as protocol_stats is calculated on scores_df
        # # Add all pose information to each trajectory
        # pose_metrics_df = pd.DataFrame.from_dict({idx: other_pose_metrics for idx in final_trajectory_indices},
        #                                          orient='index')
        # designs_df = pd.concat([designs_df, pose_metrics_df], axis=1)
        designs_df = designs_df.fillna({'observations': 1})

        # Calculate protocol significance
        pvalue_df = pd.DataFrame()
        scout_protocols = list(filter(re.compile(f'.*{putils.scout}').match,
                                      protocol_s[~protocol_s.isna()].unique().tolist()))
        similarity_protocols = unique_design_protocols.difference([putils.refine, job_key] + scout_protocols)
        sim_series = []
        if putils.structure_background not in unique_design_protocols:
            self.log.info(f'Missing background protocol "{putils.structure_background}". No protocol significance '
                          f'measurements available for this pose')
        elif len(similarity_protocols) == 1:  # measure significance
            self.log.info("Can't measure protocol significance, only one protocol of interest")
        else:  # Test significance between all combinations of protocols by grabbing mean entries per protocol
            for prot1, prot2 in combinations(sorted(similarity_protocols), 2):
                select_df = \
                    designs_df.loc[[design for designs in [designs_by_protocol[prot1], designs_by_protocol[prot2]]
                                    for design in designs], metrics.significance_columns]
                # prot1/2 pull out means from designs_df by using the protocol name
                difference_s = \
                    designs_df.loc[prot1, metrics.significance_columns].sub(
                        designs_df.loc[prot2, metrics.significance_columns])
                pvalue_df[(prot1, prot2)] = metrics.df_permutation_test(select_df, difference_s, compare='mean',
                                                                        group1_size=len(designs_by_protocol[prot1]))
            pvalue_df = pvalue_df.T  # transpose significance pairs to indices and significance metrics to columns
            designs_df = pd.concat([designs_df, pd.concat([pvalue_df], keys=['similarity']).swaplevel(0, 1)])

            # Compute residue energy/sequence differences between each protocol
            residue_energy_df = residues_df.loc[:, idx_slice[:, 'energy_delta']]

            scaler = skl.preprocessing.StandardScaler()
            res_pca = skl.decomposition.PCA(resources.config.default_pca_variance)
            residue_energy_np = scaler.fit_transform(residue_energy_df.values)
            residue_energy_pc = res_pca.fit_transform(residue_energy_np)

            seq_pca = skl.decomposition.PCA(resources.config.default_pca_variance)
            designed_sequence_modifications = residues_df.loc[:, idx_slice[:, 'type']].sum(axis=1).to_list()
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
                    designs_df = \
                        pd.concat([designs_df,
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
        # if self.job.save:
        # residues_df[(putils.protocol, putils.protocol)] = protocol_s
        # residues_df.sort_index(inplace=True, key=lambda x: x.str.isdigit())  # put wt entry first
        self.output_metrics(residues=residues_df, designs=designs_df)
        # Commit the newly acquired metrics
        self.job.current_session.commit()

        # pickle_object(pose_sequences, self.designed_sequences_file, out_path='')  # Todo PoseJob(.path)
        write_sequences(pose_sequences, file_name=self.designed_sequences_file)

        # Create figures
        if self.job.figures:  # For plotting collapse profile, errat data, contact order
            interface_residue_indices = [residue.index for residue in self.pose.interface_residues]
            # Plot: Format the collapse data with residues as index and each design as column
            # collapse_graph_df = pd.DataFrame(per_residue_data['hydrophobic_collapse'])
            collapse_graph_df = residues_df.loc[:, idx_slice[:, 'hydrophobic_collapse']].droplevel(-1, axis=1)
            reference_collapse = [entity.hydrophobic_collapse() for entity in self.pose.entities]
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
            contact_order_df = residues_df.loc[pose_name, idx_slice[:, 'contact_order']].droplevel(0, axis=1)
            # source_contact_order_s = pd.Series(source_contact_order, index=residue_indices, name='contact_order')
            contact_ax.plot(contact_order_df, label='Contact Order',
                            color='#fbc0cb', lw=1, linestyle='-')  # pink
            # contact_ax.scatter(residue_indices, source_contact_order_s, color='#fbc0cb', marker='o')  # pink
            # wt_contact_order_concatenated_min_s = source_contact_order_s.min()
            # wt_contact_order_concatenated_max_s = source_contact_order_s.max()
            # wt_contact_order_range = wt_contact_order_concatenated_max_s - wt_contact_order_concatenated_min_s
            # scaled_contact_order = ((source_contact_order_s - wt_contact_order_concatenated_min_s)
            #                         / wt_contact_order_range)  # / wt_contact_order_range)
            # graph_contact_order = sns.relplot(data=errat_graph_df, kind='line')  # x='Residue Number'
            # collapse_ax1.plot(scaled_contact_order)
            # contact_ax.vlines(self.pose.chain_breaks, 0, 1, transform=contact_ax.get_xaxis_transform(),
            #                   label='Entity Breaks', colors='#cccccc')  # , grey)
            # contact_ax.vlines(interface_residue_indices, 0, 0.05, transform=contact_ax.get_xaxis_transform(),
            #                   label='Design Residues', colors='#f89938', lw=2)  # , orange)
            contact_ax.set_ylabel('Contact Order')
            # contact_ax.set_xlim(0, pose_length)
            contact_ax.set_ylim(0, None)
            # contact_ax.figure.savefig(os.path.join(self.data_path, 'hydrophobic_collapse+contact.png'))
            # collapse_ax1.figure.savefig(os.path.join(self.data_path, 'hydrophobic_collapse+contact.png'))

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
            collapse_ax.vlines(interface_residue_indices, 0, 0.05, transform=collapse_ax.get_xaxis_transform(),
                               label='Design Residues', colors='#f89938', lw=2)  # , orange)
            # Plot horizontal significance
            collapse_significance_threshold = metrics.collapse_thresholds[hydrophobicity]
            collapse_ax.hlines([collapse_significance_threshold], 0, 1, transform=collapse_ax.get_yaxis_transform(),
                               label='Collapse Threshold', colors='#fc554f', linestyle='dotted')  # tomato
            # collapse_ax.set_xlabel('Residue Number')
            collapse_ax.set_ylabel('Hydrophobic Collapse Index')
            # collapse_ax.set_prop_cycle(color_cycler)
            # ax.autoscale(True)
            # collapse_ax.figure.tight_layout()  # no standardization
            # collapse_ax.figure.savefig(os.path.join(self.data_path, 'hydrophobic_collapse.png'))  # no standardization

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
                # collapse_ax.figure.savefig(os.path.join(self.data_path, 'hydrophobic_collapse_versus_profile.png'))

            # Plot: Errat Accuracy
            # errat_graph_df = pd.DataFrame(per_residue_data['errat_deviation'])
            # errat_graph_df = residues_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)
            # errat_graph_df = errat_df
            # wt_errat_concatenated_s = pd.Series(np.concatenate(list(source_errat.values())), name='clean_asu')
            # errat_graph_df[pose_name] = pose_source_errat
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
            errat_ax.vlines(interface_residue_indices, 0, 0.05, transform=errat_ax.get_xaxis_transform(),
                            label='Design Residues', colors='#f89938', lw=2)  # , orange)
            # Plot horizontal significance
            errat_ax.hlines([metrics.errat_2_sigma], 0, 1, transform=errat_ax.get_yaxis_transform(),
                            label='Significant Error', colors='#fc554f', linestyle='dotted')  # tomato
            errat_ax.set_xlabel('Residue Number')
            errat_ax.set_ylabel('Errat Score')
            # errat_ax.autoscale(True)
            # errat_ax.figure.tight_layout()
            # errat_ax.figure.savefig(os.path.join(self.data_path, 'errat.png'))
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
            fig.savefig(os.path.join(self.data_path, 'DesignMetricsPerResidues.png'))  # Todo PoseJob(.path)

        # After parsing data sources
        # other_pose_metrics['observations'] = len(viable_designs)
        # interface_metrics_s = pd.concat([pd.Series(other_pose_metrics)], keys=[('dock', 'pose')])

        # CONSTRUCT: Create pose series and format index names
        # pose_s = pd.concat([interface_metrics_s, stat_s, divergence_s] + sim_series).swaplevel(0, 1)
        pose_s = pd.concat([stat_s, divergence_s] + sim_series).swaplevel(0, 1)
        # Remove pose specific metrics from pose_s, sort, and name protocol_mean_df
        pose_s.drop([putils.protocol], level=2, inplace=True, errors='ignore')
        pose_s.sort_index(level=2, inplace=True, sort_remaining=False)
        pose_s.sort_index(level=1, inplace=True, sort_remaining=False)
        pose_s.sort_index(level=0, inplace=True, sort_remaining=False)
        pose_s.name = str(self)

        return pose_s

    def refine(self, to_pose_directory: bool = True, gather_metrics: bool = True,
               design_files: AnyStr = None, in_file_list: AnyStr = None):
        """Refine the PoseJob.pose instance or design Model instances associated with this instance

        ## Will append the suffix "_refine" or that given by f'_{self.protocol}' if in_file_list is passed

        Args:
            to_pose_directory: Whether the refinement should be saved to the PoseJob
            gather_metrics: Whether metrics should be calculated for the Pose
            design_files: A list of files to perform refinement on
            in_file_list: The path to a file containing a list of files to pass to Rosetta refinement
        """
        main_cmd = rosetta.script_cmd.copy()

        infile = []
        suffix = []
        generate_files_cmd = null_cmd
        if to_pose_directory:  # Original protocol to refine a Nanohedra pose
            flag_dir = self.scripts_path
            pdb_out_path = self.designs_path
            refine_pdb = refined_pdb = self.refined_pdb
            additional_flags = []
        else:  # Protocol to refine input structure, place in a common location, then transform for many jobs to source
            flag_dir = pdb_out_path = self.job.refine_dir
            refine_pdb = self.source_path
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
        if design_files is not None or in_file_list is not None:  # Run a list of files produced elsewhere
            possible_refine_protocols = ['refine', 'thread']
            if self.protocol in possible_refine_protocols:
                switch = self.protocol
            elif self.protocol is None:
                switch = putils.refine
            else:
                switch = putils.refine
                self.log.warning(f'The requested protocol "{self.protocol}", was not recognized by '
                                 f'{self.refine.__name__} and is being treated as the standard "{switch}" protocol')

            # Create file output
            designed_files_file = os.path.join(self.scripts_path, f'{timestamp()}_{switch}_files_output.txt')
            if in_file_list:
                generate_files_cmd = \
                    ['python', putils.list_pdb_files, '-d', self.designs_path, '-o', designed_files_file, '-e', '.pdb',
                     '-s', f'_{switch}']
                suffix = ['-out:suffix', f'_{switch}']
            elif design_files:
                design_files_file = os.path.join(self.scripts_path, f'{timestamp()}_{self.protocol}_files.txt')
                with open(design_files_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(design_files))
                # Write the designed_files_file with all "tentatively" designed file paths
                out_file_string = f'%s{os.sep}{pdb_out_path}{os.sep}%s'
                with open(design_files_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(out_file_string % os.path.split(file) for file in design_files))
            else:
                raise ValueError(f"Couldn't run {self.refine.__name__} without passing parameter 'design_files' as an "
                                 f"iterable of files")

            # -in:file:native is here to block flag file version, not actually useful for refine
            infile.extend(['-in:file:l', in_file_list, '-in:file:native', self.source_path])
            metrics_pdb = ['-in:file:l', designed_files_file, '-in:file:native', self.source_path]
            # generate_files_cmdline = [list2cmdline(generate_files_cmd)]
        else:
            # if self.interface_residue_numbers is False or self.interface_design_residue_numbers is False:
            self.identify_interface()
            # else:  # We only need to load pose as we already calculated interface
            #     self.load_pose()

            self.protocol = switch = putils.refine
            if self.job.interface_to_alanine:  # Mutate all design positions to Ala before the Refinement
                for entity_pair, interface_residues_pair in self.pose.interface_residues_by_entity_pair.items():
                    # if interface_residues_pair[0]:  # Check that there are residues present
                    for entity, interface_residues in zip(entity_pair, interface_residues_pair):
                        entity_name = entity.name
                        for residue in interface_residues:
                            self.log.debug(f'Mutating Entity {entity_name}, {residue.number}{residue.type}')
                            if residue.type != 'GLY':  # No mutation from GLY to ALA as Rosetta would build a CB
                                self.pose.mutate_residue(residue=residue, to='A')
                # Change the name to reflect mutation so we don't overwrite the self.source_path
                refine_pdb = f'{os.path.splitext(refine_pdb)[0]}_ala_mutant.pdb'
            # else:  # Do dothing and refine the source
            #     pass
            #     # raise ValueError(f"For {self.refine.__name__}, must pass interface_to_alanine")

            self.pose.write(out_path=refine_pdb)
            # Ensure the mutations to the pose are wiped
            self.pose = None
            self.log.debug(f'Cleaned PDB for {switch}: "{refine_pdb}"')
            # -in:file:native is here to block flag file version, not actually useful for refine
            infile.extend(['-in:file:s', refine_pdb, '-in:file:native', refine_pdb])
            metrics_pdb = ['-in:file:s', refined_pdb, '-in:file:native', refine_pdb]

        # RELAX: Prepare command
        if self.symmetry_dimension is not None and self.symmetry_dimension > 0:
            symmetry_definition = ['-symmetry_definition', 'CRYST1']
        else:
            symmetry_definition = []

        # '-no_nstruct_label', 'true' comes from v
        relax_cmd = main_cmd + rosetta.relax_flags_cmdline + additional_flags + symmetry_definition + infile + suffix \
            + [f'@{flags_file}', '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, f'refine.xml'),
               '-parser:script_vars', f'switch={switch}']
        self.log.info(f'{switch.title()} Command: {list2cmdline(relax_cmd)}')

        if gather_metrics or self.job.metrics:
            gather_metrics = True
            if self.job.mpi > 0:
                main_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(self.job.mpi)] + main_cmd
            main_cmd += metrics_pdb
            main_cmd += [f'@{flags_file}', '-out:file:score_only', self.scores_file,
                         '-no_nstruct_label', 'true', '-parser:protocol']
            metric_cmd_bound = main_cmd \
                + [os.path.join(putils.rosetta_scripts_dir, f'{putils.interface_metrics}'
                                f'{"_DEV" if self.job.development else ""}.xml')] \
                + symmetry_definition
            entity_cmd = main_cmd + [os.path.join(putils.rosetta_scripts_dir,
                                                  f'metrics_entity{"_DEV" if self.job.development else ""}.xml')]
            self.log.info(f'Metrics Command: {list2cmdline(metric_cmd_bound)}')
            metric_cmds = [metric_cmd_bound]
            metric_cmds.extend(self.generate_entity_metrics_commands(entity_cmd))
        else:
            metric_cmds = []

        # Create executable/Run FastRelax on Clean ASU with RosettaScripts
        if self.job.distribute_work:
            analysis_cmd = self.make_analysis_cmd()
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
            if gather_metrics:
                for metric_cmd in metric_cmds:
                    metrics_process = Popen(metric_cmd)
                    metrics_process.communicate()

            # Gather metrics for each design produced from this proceedure
            self.process_rosetta_metrics()

    def rosetta_interface_design(self):
        """For the basic process of sequence design between two halves of an interface, write the necessary files for
        refinement (FastRelax), redesign (FastDesign), and metrics collection (Filters & SimpleMetrics)

        Stores job variables in a [stage]_flags file and the command in a [stage].sh file. Sets up dependencies based
        on the PoseJob
        """
        # Set up the command base (rosetta bin and database paths)
        main_cmd = rosetta.script_cmd.copy()
        if self.symmetry_dimension is not None and self.symmetry_dimension > 0:
            main_cmd += ['-symmetry_definition', 'CRYST1']

        # Todo - Has this been solved?
        #  must set up a blank -in:file:pssm in case the evolutionary matrix is not used. Design will fail!!
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
            design_files = os.path.join(self.scripts_path, f'design_files_{self.protocol}.txt')
            generate_files_cmd = \
                ['python', putils.list_pdb_files, '-d', self.designs_path, '-o', design_files, '-e', '.pdb',
                 '-s', f'_{self.protocol}']
            metrics_pdb = ['-in:file:l', design_files]
            # metrics_flags = 'repack=yes'
            if self.job.design.structure_background:
                self.protocol = protocol_xml1 = putils.structure_background
                nstruct_instruct = ['-nstruct', str(self.job.design.number)]
            elif self.job.design.hbnet:  # Run hbnet_design_profile protocol
                self.protocol, protocol_xml1 = putils.hbnet_design_profile, 'hbnet_scout'
                nstruct_instruct = ['-no_nstruct_label', 'true']
                # Set up an additional command to perform interface design on hydrogen bond network from hbnet_scout
                additional_cmds = \
                    [[putils.hbnet_sort, os.path.join(self.data_path, 'hbnet_silent.o'),
                      str(self.job.design.number)]] \
                    + [main_cmd + profile_cmd
                       + ['-in:file:silent', os.path.join(self.data_path, 'hbnet_selected.o'), f'@{self.flags}',
                          '-in:file:silent_struct_type', 'binary',  # '-out:suffix', f'_{self.protocol}',
                          # adding no_nstruct_label true as only hbnet uses this mechanism
                          # hbnet_design_profile.xml could be just design_profile.xml
                          '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, f'{self.protocol}.xml')] \
                       + nstruct_instruct]
                # Set up additional out_file
                out_file = ['-out:file:silent', os.path.join(self.data_path, 'hbnet_silent.o'),
                            '-out:file:silent_struct_type', 'binary']
                # silent_file = os.path.join(self.data_path, 'hbnet_silent.o')
                # additional_commands = \
                #     [
                #      # ['grep', '^SCORE', silent_file, '>', os.path.join(self.data_path, 'hbnet_scores.sc')],
                #      main_cmd + [os.path.join(self.data_path, 'hbnet_selected.o')]
                #      [os.path.join(self.data_path, 'hbnet_selected.tags')]
                #     ]
            else:  # Run the legacy protocol
                self.protocol = protocol_xml1 = putils.interface_design
                nstruct_instruct = ['-nstruct', str(self.job.design.number)]

        # DESIGN: Prepare command and flags file
        if not os.path.exists(self.flags) or self.job.force:
            self.prepare_rosetta_flags(out_dir=self.scripts_path)
            self.log.debug(f'Pose flags written to: {self.flags}')

        if self.job.design.method == putils.consensus:
            self.protocol = putils.consensus
            consensus_cmd = main_cmd + rosetta.relax_flags_cmdline \
                + [f'@{self.flags}', '-in:file:s', self.consensus_pdb,
                   # '-in:file:native', self.refined_pdb,
                   '-parser:protocol', os.path.join(putils.rosetta_scripts_dir,
                                                    f'{putils.consensus}.xml'),
                   '-parser:script_vars', f'switch={putils.consensus}']
            self.log.info(f'Consensus command: {list2cmdline(consensus_cmd)}')
            if self.job.distribute_work:
                write_shell_script(list2cmdline(consensus_cmd), name=putils.consensus, out_path=self.scripts_path)
            else:
                consensus_process = Popen(consensus_cmd)
                consensus_process.communicate()

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
        metric_cmds.extend(self.generate_entity_metrics_commands(entity_cmd))

        # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics
        if self.job.distribute_work:
            analysis_cmd = self.make_analysis_cmd()
            write_shell_script(list2cmdline(design_cmd), name=self.protocol, out_path=self.scripts_path,
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

            # Gather metrics for each design produced from this proceedure
            self.process_rosetta_metrics()

    def analyze_proteinmpnn_metrics(self, design_ids: Sequence[str], sequences_and_scores: dict[str, np.array]):
        #                      designs: Iterable[Pose] | Iterable[AnyStr] = None
        """

        Args:
            design_ids: The associated design identifier for each corresponding entry in sequences_and_scores
            sequences_and_scores: The mapping of ProteinMPNN score type to it's corresponding data
        Returns:

        """
        #     designs: The designs to perform analysis on. By default, fetches all available structures
        # Calculate metrics on input Pose before any manipulation
        pose_length = self.pose.number_of_residues
        residue_indices = list(range(pose_length))  # [residue.index for residue in self.pose.residues]
        # residue_numbers = [residue.number for residue in self.pose.residues]

        # metadata_df = pd.DataFrame(sequences_and_scores['temperatures'], index=design_ids, columns=['temperature'])
        # metadata_df[putils.protocol] = sequences_and_scores[putils.protocol]
        # numeric_sequences = sequences_and_scores['numeric_sequences']
        # torch_numeric_sequences = torch.from_numpy(numeric_sequences)
        # nan_blank_data = list(repeat(np.nan, pose_length))
        sequences = sequences_and_scores['sequences']
        per_residue_design_indices = sequences_and_scores['design_indices']
        per_residue_complex_sequence_loss = sequences_and_scores['proteinmpnn_loss_complex']
        per_residue_unbound_sequence_loss = sequences_and_scores['proteinmpnn_loss_unbound']

        # # Make requisite profiles
        # profile_background = {}
        # # Load fragment_profile into the analysis
        # # This is currently called in design() and this function (analyze_proteinmpnn_metrics) is not used elsewhere
        # if self.job.design.term_constraint:
        #     if not self.pose.fragment_queries:
        #         self.generate_fragments(interface=True)
        #         self.pose.calculate_fragment_profile()
        #     profile_background['fragment'] = fragment_profile_array = self.pose.fragment_profile.as_array()

        # if self.job.design.evolution_constraint:
        #     self.generate_evolutionary_profile(warn_metrics=True)
        #     # if self.pose.evolutionary_profile:
        #     profile_background['evolution'] = evolutionary_profile_array = pssm_as_array(self.pose.evolutionary_profile)
        #     torch_log_evolutionary_profile = torch.from_numpy(np.log(evolutionary_profile_array))
        #     self.pose.calculate_profile()
        #     profile_background['design'] = design_profile_array = pssm_as_array(self.pose.profile)
        #     torch_log_design_profile = torch.from_numpy(np.log(design_profile_array))
        # else:
        #     torch_log_evolutionary_profile = torch_log_design_profile = torch.tensor(nan_blank_data)
        #     per_residue_evolutionary_profile_loss = per_residue_design_profile_loss = nan_blank_data

        # number_of_temperatures = len(self.job.design.temperatures)
        # per_residue_data = {}
        # fragment_profile_frequencies = []

        # Construct residues_df
        proteinmpnn_data = {
            'design_residue': per_residue_design_indices,
            'proteinmpnn_loss_complex': per_residue_complex_sequence_loss,
            'proteinmpnn_loss_unbound': per_residue_unbound_sequence_loss
        }
        proteinmpnn_residue_info_df = \
            pd.concat([pd.DataFrame(data, index=design_ids,
                                    columns=pd.MultiIndex.from_product([residue_indices, [metric]]))
                       for metric, data in proteinmpnn_data.items()], axis=1)
        # for idx, design_id in enumerate(design_ids):
        #     # Add pose metrics
        #     interface_metrics[design_id] = pose_interface_metrics
        #
        #     # For each Pose, save each sequence design data such as energy # probabilites
        #     # all_probabilities[design_id] = probabilities[idx]
        #     # Todo process the all_probabilities to a DataFrame?
        #     #  The probabilities are the actual probabilities at each residue for each AA
        #     #  These differ from the log_probabilities in that those are scaled by the log()
        #     #  and therefore are negative. The use of probabilities is how I have calculated divergence.
        #     #  Perhaps I should transition to take the log of probabilities and calculate the loss.
        #     # all_probabilities is
        #     # {'2gtr-3m6n-DEGEN_1_1-ROT_13_10-TX_1-PT_1':
        #     #  array([[1.55571969e-02, 6.64833433e-09, 3.03523801e-03, ...,
        #     #          2.94689467e-10, 8.92133514e-08, 6.75683381e-12],
        #     #         [9.43517406e-03, 2.54900701e-09, 4.43358254e-03, ...,
        #     #          2.19431431e-10, 8.18614296e-08, 4.94338381e-12],
        #     #         [1.50658926e-02, 1.43449803e-08, 3.27082584e-04, ...,
        #     #          1.70684064e-10, 8.77646258e-08, 6.67974660e-12],
        #     #         ...,
        #     #         [1.23516358e-07, 2.98688293e-13, 3.48888407e-09, ...,
        #     #          1.17041141e-14, 4.72279464e-12, 5.79130243e-16],
        #     #         [9.99999285e-01, 2.18584519e-19, 3.87702094e-16, ...,
        #     #          7.12933229e-07, 5.22657113e-13, 3.19411591e-17],
        #     #         [2.11755684e-23, 2.32944583e-23, 3.86148234e-23, ...,
        #     #          1.16764793e-22, 1.62743156e-23, 7.65081924e-23]]),
        #     #  '2gtr-3m6n-DEGEN_1_1-ROT_13_10-TX_1-PT_2':
        #     #  array([[1.72123183e-02, 7.31348226e-09, 3.28084361e-03, ...,
        #     #          3.16341731e-10, 9.09206364e-08, 7.41259137e-12],
        #     #         [6.17256807e-03, 1.86070248e-09, 2.70802877e-03, ...,
        #     #          1.61229460e-10, 5.94660143e-08, 3.73394328e-12],
        #     #         [1.28052337e-02, 1.10993081e-08, 3.89973022e-04, ...,
        #     #          2.21829027e-10, 1.03226760e-07, 8.43660298e-12],
        #     #         ...,
        #     #         [1.31807008e-06, 2.47859654e-12, 2.27575967e-08, ...,
        #     #          5.34223104e-14, 2.06900348e-11, 3.35126595e-15],
        #     #         [9.99999821e-01, 1.26853575e-19, 2.05691231e-16, ...,
        #     #          2.02439509e-07, 5.02121131e-13, 1.38719620e-17],
        #     #         [2.01858383e-23, 2.29340987e-23, 3.59583879e-23, ...,
        #     #          1.13548109e-22, 1.60868618e-23, 7.25537526e-23]])}
        #
        #     # # Calculate sequence statistics
        #     # # Before calculation, we must set this (v) to get the correct values from the profile
        #     # self.pose._sequence_numeric = numeric_sequences[idx]
        #     # try:
        #     #     fragment_profile_frequencies.append(
        #     #         self.pose.get_sequence_probabilities_from_profile(precomputed=fragment_profile_array))
        #     # except IndexError as error:  # We are missing fragments for this Pose
        #     #     self.log.warning(f"We didn't find any fragment information... due to: {error}")
        #     #     #                "\nSetting the self.pose.fragment_profile = None")
        #     #     # raise IndexError(f'With new updates to calculate_fragment_profile this code should be '
        #     #     #                  f'unreachable. Original error:\n{error}')
        #     #     fragment_profile_frequencies.append(nan_blank_data)
        #
        #     # observed, divergence = \
        #     #     calculate_sequence_observations_and_divergence(pose_alignment, profile_background)
        #     # # Get pose sequence divergence
        #     # divergence_s = pd.Series({f'{divergence_type}_per_residue': _divergence.mean()
        #     #                           for divergence_type, _divergence in divergence.items()},
        #     #                          name=design_id)
        #     # all_pose_divergence.append(divergence_s)
        #     # Todo extract the observed values out of the observed dictionary
        #     #  Each Pose only has one trajectory, so measurement of divergence is pointless (no distribution)
        #     # observed_dfs = []
        #     # # Todo must ensure the observed_values is the length of the design_ids
        #     # # for profile, observed_values in observed.items():
        #     # #     designs_df[f'observed_{profile}'] = observed_values.mean(axis=1)
        #     # #     observed_dfs.append(pd.DataFrame(data=observed_values, index=design_id,
        #     # #                                      columns=pd.MultiIndex.from_product([residue_indices,
        #     # #                                                                          [f'observed_{profile}']]))
        #     # #                         )
        #     # # # Add observation information into the residues_df
        #     # # residues_df = pd.concat([residues_df] + observed_dfs, axis=1)
        #     # # Todo get divergence?
        #     # # Get the negative log likelihood of the .evolutionary_ and .fragment_profile
        #     # # torch_numeric = torch.from_numpy(self.pose.sequence_numeric)
        #     # torch_numeric = torch_numeric_sequences[idx]
        #     # # Todo I had noted that these may not be Softmax probabilities. Are they?? Should it matter
        #     # if self.measure_evolution:
        #     #     per_residue_design_profile_loss = \
        #     #         resources.ml.sequence_nllloss(torch_numeric, torch_log_design_profile)
        #     #     per_residue_evolutionary_profile_loss = \
        #     #         resources.ml.sequence_nllloss(torch_numeric, torch_log_evolutionary_profile)
        #
        #     # if self.pose.fragment_profile:
        #     #     with warnings.catch_warnings():
        #     #         # np.log causes -inf at 0, thus we correct these to a very large number
        #     #         warnings.simplefilter('ignore', category=RuntimeWarning)
        #     #         corrected_frag_array = np.nan_to_num(np.log(fragment_profile_array), copy=False,
        #     #                                              nan=np.nan, neginf=metrics.zero_probability_frag_value)
        #     #     per_residue_fragment_profile_loss = \
        #     #         resources.ml.sequence_nllloss(torch_numeric, torch.from_numpy(corrected_frag_array))
        #     #     # Find the non-zero sites in the profile
        #     #     # observed_frequencies_from_fragment_profile = \
        #     #     #     fragment_profile_frequencies[idx][per_residue_design_indices[idx]]
        #     # else:
        #     #     per_residue_fragment_profile_loss = nan_blank_data
        #
        #     per_residue_data[design_id] = {
        #         'design_residue': per_residue_design_indices[idx],
        #         'proteinmpnn_loss_complex': per_residue_complex_sequence_loss[idx],
        #         'proteinmpnn_loss_unbound': per_residue_unbound_sequence_loss[idx],
        #         # 'type': sequences[idx],
        #         # 'sequence_loss_design': per_residue_design_profile_loss,
        #         # 'sequence_loss_evolution': per_residue_evolutionary_profile_loss,
        #         # 'sequence_loss_fragment': per_residue_fragment_profile_loss,
        #     }
        # # Construct residues_df
        # residues_df = pd.concat({design_id: pd.DataFrame(data, index=residue_indices)
        #                          for design_id, data in per_residue_data.items()}).unstack().swaplevel(0, 1, axis=1)

        # Todo get the keys right here
        # all_pose_divergence_df = pd.DataFrame()
        # all_pose_divergence_df = pd.concat(all_pose_divergence, keys=[('sequence', 'pose')], axis=1)

        # Initialize the main scoring DataFrame

        # Calculate pose metrics
        # pose_interface_metrics = self.pose.calculate_metrics()
        # interface_metrics = {}
        # for idx, design_id in enumerate(design_ids):
        #     # Add pose metrics
        #     interface_metrics[design_id] = pose_interface_metrics
        # designs_df = pd.DataFrame.from_dict(interface_metrics, orient='index')
        # designs_df = designs_df.join(metadata_df)

        # Incorporate residue, design, and sequence metrics on every designed Pose
        # per_residue_sequence_df.loc[pose_name, :] = list(self.pose.sequence)
        # per_residue_sequence_df.append(pd.DataFrame(list(self.pose.sequence), columns=[pose_name]).T)
        # Todo UPDATE These are now from a different collapse 'hydrophobicity' source, 'expanded'
        sequences_df = self.analyze_sequence_metrics_per_design(sequences=sequences, design_ids=design_ids)
        # Since no structure design completed, no residue_metrics is performed, but the pose source can be...
        # # The residues_df here has the wrong .index. It needs to become the design.id not design.name
        # residues_df = self.analyze_residue_metrics_per_design()  # designs=designs)
        # # Join each per-residue like dataframe
        # # Each of these can have difference index, so we use concat to perform an outer merge
        # residues_df = pd.concat([residues_df, sequences_df, proteinmpnn_residue_info_df], axis=1)
        residues_df = pd.concat([sequences_df, proteinmpnn_residue_info_df], axis=1)

        designs_df = self.analyze_design_metrics_per_residue(residues_df)  # designs_df=metadata_df, designs=designs)

        designed_df = residues_df.loc[:, idx_slice[:, 'design_residue']].droplevel(1, axis=1)

        # designs_df[putils.protocol] = 'proteinmpnn'
        designs_df['proteinmpnn_score_complex'] = designs_df['proteinmpnn_loss_complex'] / pose_length
        designs_df['proteinmpnn_score_unbound'] = designs_df['proteinmpnn_loss_unbound'] / pose_length
        designs_df['proteinmpnn_score_delta'] = \
            designs_df['proteinmpnn_score_complex'] - designs_df['proteinmpnn_score_unbound']
        designs_df['proteinmpnn_score_complex_per_designed_residue'] = \
            (residues_df.loc[:, idx_slice[:, 'proteinmpnn_loss_complex']].droplevel(1, axis=1)
             * designed_df).mean(axis=1)
        designs_df['proteinmpnn_score_unbound_per_designed_residue'] = \
            (residues_df.loc[:, idx_slice[:, 'proteinmpnn_loss_unbound']].droplevel(1, axis=1)
             * designed_df).mean(axis=1)
        designs_df['proteinmpnn_score_delta_per_designed_residue'] = \
            designs_df['proteinmpnn_score_complex_per_designed_residue'] / \
            designs_df['proteinmpnn_score_unbound_per_designed_residue']

        # # Drop unused particular residues_df columns that have been summed
        # per_residue_drop_columns = per_residue_energy_states + energy_metric_names + per_residue_sasa_states \
        #                            + collapse_metrics + residue_classification \
        #                            + ['errat_deviation', 'hydrophobic_collapse', 'contact_order'] \
        #                            + ['hbond', 'evolution', 'fragment', 'type'] + ['surface', 'interior']
        # # Slice each of these columns as the first level residue number needs to be accounted for in MultiIndex
        # residues_df = residues_df.drop(
        #     list(residues_df.loc[:, idx_slice[:, per_residue_drop_columns]].columns),
        #     errors='ignore', axis=1)

        self.output_metrics(residues=residues_df, designs=designs_df)
        # Commit the newly acquired metrics
        self.job.current_session.commit()

    def proteinmpnn_design(self, interface: bool = False, neighbors: bool = False):
        """Perform design based on the ProteinMPNN graph encoder/decoder network and output sequences and scores to the
        PoseJob scorefile

        Sets:
            self.protocol = 'proteinmpnn'
        Args:
            interface: Whether to only specify the interface as designable, otherwise, use all residues
            neighbors: Whether to design interface neighbors
        Returns:

        """
        self.protocol = 'proteinmpnn'
        self.log.info(f'Starting {self.protocol} design calculation with {self.job.design.number} '
                      f'designs over each of the temperatures: {self.job.design.temperatures}')
        # design_start = time.time()
        sequences_and_scores: dict[str, np.ndarray | list] = \
            self.pose.design_sequences(number=self.job.design.number,
                                       temperatures=self.job.design.temperatures,
                                       interface=interface, neighbors=neighbors,
                                       ca_only=self.job.design.ca_only
                                       )
        # self.log.debug(f"Took {time.time() - design_start:8f}s for design_sequences")

        # Update the Pose with the number of designs
        designs_data = self.update_design_data(design_parent=self.pose_source)

        # self.output_proteinmpnn_scores(design_names, sequences_and_scores)
        # # Write every designed sequence to the sequences file...
        # write_sequences(sequences_and_scores['sequences'], names=design_names, file_name=self.designed_sequences_file)
        # Convert sequences to a plain string sequence representation
        sequences_and_scores['sequences'] = \
            [''.join(sequence) for sequence in sequences_and_scores['sequences'].tolist()]

        # Add protocol (job info) and temperature to sequences_and_scores
        # number_of_new_designs = len(designs_metadata)
        # sequences_and_scores[putils.protocol] = list(repeat(self.protocol, len(designs_metadata)))
        # sequences_and_scores['temperatures'] = [temperature for temperature in self.job.design.temperatures
        # protocols = list(repeat(self.protocol, len(designs_metadata)))
        temperatures = [temperature for temperature in self.job.design.temperatures
                        for _ in range(self.job.design.number)]
        design_ids = [design_data.id for design_data in designs_data]

        # # Write every designed sequence to an individual file...
        # putils.make_path(self.designs_path)
        # design_names = [design_data.name for design_data in designs_data]
        # sequence_files = [
        #     write_sequences(sequence, names=name, file_name=os.path.join(self.designs_path, name))
        #     for name, sequence in zip(design_names, sequences_and_scores['sequences'])
        # ]
        # Update the Pose with the design protocols
        for idx, design_data in enumerate(designs_data):
            design_data.protocols.append(
                sql.DesignProtocol(design=design_data,
                                   # design_id=design_ids[idx],
                                   protocol=self.protocol,  # sql.Protocol(name=self.protocol),  # protocols[idx],
                                   temperature=temperatures[idx],
                                   ))
                                   # file=sequence_files[idx]))
        # design_protocol = self.update_design_protocols(protocols=protocols, temperatures=temperatures,
        #                                                files=sequence_files)

        # analysis_start = time.time()
        self.analyze_proteinmpnn_metrics(design_ids, sequences_and_scores)
        # self.log.debug(f"Took {time.time() - analysis_start:8f}s for analyze_proteinmpnn_metrics. "
        #                f"{time.time() - design_start:8f}s total")

    # def output_proteinmpnn_scores(self, design_ids: Sequence[str], sequences_and_scores: dict[str, np.ndarray | list]):
    #     """Given the results of a ProteinMPNN design trajectory, format the sequences and scores for the PoseJob
    #
    #     Args:
    #         design_ids: The associated design identifier for each corresponding entry in sequences_and_scores
    #         sequences_and_scores: The mapping of ProteinMPNN score type to it's corresponding data
    #     """
    #     # # Convert each numpy array into a list for output
    #     # for score_type, data in sequences_and_scores.items():
    #     #     # if isinstance(data, np.ndarray):
    #     #     sequences_and_scores[score_type] = data.tolist()
    #
    #     # # trajectories_temperatures_ids = [f'temp{temperature}' for idx in self.job.design.number
    #     # #                                  for temperature in self.job.design.temperatures]
    #     # # trajectories_temperatures_ids = [{'temperatures': temperature} for idx in self.job.design.number
    #     # #                                  for temperature in self.job.design.temperatures]
    #     # protocol = 'proteinmpnn'
    #     # sequences_and_scores[putils.protocol] = \
    #     #     repeat(protocol, len(self.job.design.number * self.job.design.temperatures))
    #     # sequences_and_scores['temperatures'] = [temperature for temperature in self.job.design.temperatures
    #     #                                         for _ in range(self.job.design.number)]
    #
    #     def write_per_residue_scores(_design_ids: Sequence[str], scores: dict[str, list]) -> AnyStr:
    #         """"""
    #         # Create an initial score dictionary
    #         design_scores = {design_id: {'decoy': design_id} for design_id in _design_ids}
    #         # For each score type unpack the data
    #         for score_type, score in scores.items():
    #             # For each score's data, update the dictionary of the corresponding design_id
    #             if isinstance(score, np.ndarray):
    #                 # Convert each numpy array into a list for output
    #                 score = score.tolist()
    #             for design_id, design_score in zip(_design_ids, score):
    #                 design_scores[design_id].update({score_type: design_score})
    #
    #         for design_id, scores in design_scores.items():
    #             # write_json(_scores, self.scores_file)
    #             with open(self.scores_file, 'a') as f_save:
    #                 json.dump(scores, f_save)  # , **kwargs)
    #                 # Ensure JSON lines are separated by newline
    #                 f_save.write('\n')
    #
    #         # return write_json(design_scores, self.scores_file)
    #         return self.scores_file
    #
    #     putils.make_path(self.data_path)
    #     write_per_residue_scores(design_ids, sequences_and_scores)

    def update_design_protocols(self, design_ids: Sequence[str], protocols: Sequence[str] = None,
                                temperatures: Sequence[float] = None, files: Sequence[AnyStr] = None) \
            -> list[sql.DesignProtocol]:
        """Associate newly created DesignData with DesignProtocol

        Args:
            design_ids: The identifiers for each DesignData
            protocols: The sequence of protocols to associate with DesignProtocol
            temperatures: The temperatures to associate with DesignProtocol
            files: The sequence of files to associate with DesignProtocol
        Returns:
            The new instances of the sql.DesignProtocol
        """
        metadata = [sql.DesignProtocol(design_id=design_id, protocol=protocol, temperature=temperature, file=file)
                    for design_id, protocol, temperature, file in zip(design_ids, protocols, temperatures, files)]
        return metadata

    def update_design_data(self, design_parent: sql.DesignData, number: int = None) -> list[sql.DesignData]:
        """Update the PoseData with the newly created design identifiers using DesignData and flush to the database

        Args:
            design_parent: The design whom all new designs are based
            number: The number of designs. If not provided, set according to job.design.number * job.design.temperature
        Returns:
            The new instances of the DesignData
        """
        if number is None:
            number = len(self.job.design.number * self.job.design.temperatures)

        # self.number_of_designs += number_of_new_designs  # Now done with self.number_of_designs in SQL
        # with self.job.db.session(expire_on_commit=False) as session:
        first_new_design_idx = self.number_of_designs + 1
        # design_names = [f'{self.protocol}{seq_idx:04d}'  # f'{self.name}_{self.protocol}{seq_idx:04d}'
        design_names = [f'{self.name}-{design_idx:04d}'  # f'{self.name}_{self.protocol}{seq_idx:04d}'
                        for design_idx in range(first_new_design_idx,
                                                first_new_design_idx + number)]
        # # designs = [sql.DesignData(name=name, pose_id=self.id) for name in design_names]
        # # session.add_all(designs)
        # # session.commit()
        # designs = [sql.DesignData(name=name, design_parent=design_parent) for name in design_names]
        # self.designs.extend(designs)
        designs = [sql.DesignData(name=name, pose=self, design_parent=design_parent)
                   for name in design_names]
        # Set the PoseJob.current_designs for access by subsequence protocols
        self.current_designs.extend(designs)
        # Get the DesignData.id for each design
        self.job.current_session.flush()

        return designs

    def output_metrics(self, designs: pd.DataFrame = None, residues: pd.DataFrame = None,
                       pose_metrics: bool = False, update: bool = False):
        """Format each possible DataFrame type for output via csv or SQL database

        Args:
            designs: The typical per-design metric DataFrame where each index is the design id and the columns are
                design metrics
            residues: The typical per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
            pose_metrics: Whether the metrics being included are based on Pose (self.pose) measurements
            update: Whether the output identifiers are already present in the metrics
        """
        # Remove completely empty columns
        if designs is not None:
            designs.dropna(how='all', axis=1, inplace=True)
        if residues is not None:
            residues.dropna(how='all', axis=1, inplace=True)

        if self.job.db:
            # Add the pose identifier to the dataframes
            # pose_identifier = self._pose_id  # reliant on foreign keys...
            # pose_identifier = self.pose_id  # reliant on SymDesign names...
            # pose_identifier = self.id
            if designs is not None:
                # design_index_names = ['pose', 'design']
                # These are reliant on foreign keys...
                # design_index_names = [sql.DesignMetrics.pose_id.name, sql.DesignMetrics.name.name]
                # designs = pd.concat([designs], keys=[pose_identifier], axis=0)
                # design_index_names = [sql.DesignMetrics.pose_id.name, sql.DesignMetrics.design_id.name]
                # designs = pd.concat([designs], keys=[pose_identifier], axis=0)
                # #                     names=design_index_names, axis=0)
                # designs.index.set_names(design_index_names, inplace=True)
                designs.index.set_names(sql.DesignMetrics.design_id.name, inplace=True)
                # _design_ids = metrics.sql.write_dataframe(designs=designs)
                metrics.sql.write_dataframe(designs=designs)  # , update=False)
            # else:
            #     _design_ids = []

            if residues is not None:
                # residue_index_names = ['pose', 'design']
                # These are reliant on foreign keys...
                # residue_index_names = [sql.ResidueMetrics.pose_id.name, sql.ResidueMetrics.design_id.name, sql.ResidueMetrics.design_name.name]
                # residues = pd.concat([residues], keys=list(zip(repeat(pose_identifier), _design_ids)), axis=0)
                # residue_index_names = [sql.ResidueMetrics.pose_id.name, sql.ResidueMetrics.design_id.name]
                # residues = pd.concat([residues], keys=[pose_identifier], axis=0)
                # #                      names=residue_index_names, axis=0)
                # residues.index.set_names(residue_index_names, inplace=True)
                if pose_metrics:
                    index_name = sql.PoseResidueMetrics.pose_id.name
                    dataframe_kwargs = dict(pose_residues=residues)
                else:
                    index_name = sql.ResidueMetrics.design_id.name
                    dataframe_kwargs = dict(residues=residues)

                residues.index.set_names(index_name, inplace=True)
                metrics.sql.write_dataframe(**dataframe_kwargs)  # , update=False)
        else:
            putils.make_path(self.data_path)
            if residues is not None:
                # Process dataframes for missing values NOT USED with SQL...
                residues = residues.fillna(0.)
                residues.sort_index(inplace=True)
                residues.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
                # residue_metric_columns = residues.columns.levels[-1].to_list()
                # self.log.debug(f'Residues metrics present: {residue_metric_columns}')

                residues.to_csv(self.residues_metrics_csv)
                self.log.info(f'Wrote Residues metrics to {self.residues_metrics_csv}')

            if designs is not None:
                designs.sort_index(inplace=True, axis=1)
                # designs_metric_columns = designs.columns.to_list()
                # self.log.debug(f'Designs metrics present: {designs_metric_columns}')

                # if self.job.merge:
                #     designs_df = pd.concat([designs_df], axis=1, keys=['metrics'])
                #     designs_df = pd.merge(designs_df, residues_df, left_index=True, right_index=True)
                # else:
                designs.to_csv(self.designs_metrics_csv)
                self.log.info(f'Wrote Designs metrics to {self.designs_metrics_csv}')

        # # Concatenate all design information after parsing data sources
        # designs = pd.concat([designs], keys=[('dock', 'pose')], axis=1)
        #
        # # CONSTRUCT: Create pose series and format index names
        # pose_df = designs.swaplevel(0, 1, axis=1)
        # # pose_df = pd.concat([designs_df, interface_metrics_df, all_pose_divergence_df]).swaplevel(0, 1)
        # # Remove pose specific metrics from pose_df and sort
        # pose_df.sort_index(level=2, axis=1, inplace=True, sort_remaining=False)
        # pose_df.sort_index(level=1, axis=1, inplace=True, sort_remaining=False)
        # pose_df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
        # pose_df.name = str(self)
        #
        # self.job.dataframe = self.designs_metrics_csv
        # pose_df.to_csv(self.designs_metrics_csv)

    def analyze_predict_structure_metrics(self, design_ids: Sequence[str], designs: Iterable[Pose] | Iterable[AnyStr]):
        """"""
        raise NotImplementedError('Please copy the method used for structural analysis from '
                                  f'{self.process_rosetta_metrics.__name__} except removing the .scores_file attr')

        residues_df = self.analyze_residue_metrics_per_design(designs)
        designs_df = self.analyze_design_metrics_per_design(residues_df, designs)
        self.output_metrics(residues=residues_df, designs=designs_df, update=True)
        # Commit the newly acquired metrics
        self.job.current_session.commit()

    def analyze_sequence_metrics_per_design(self, sequences: dict[str, Sequence[str]] | Sequence[Sequence[str]] = None,
                                            design_ids: Sequence[str] = None) -> pd.DataFrame:
        """Gather metrics based on provided sequences in comparison to the Pose sequence

        Args:
            sequences: The sequences to analyze compared to the "pose source" sequence
            design_ids: If sequences isn't a mapping from identifier to sequence, the identifiers for each sequence
        Returns:
            A per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
        """
        # Ensure the pose.sequence (or reference sequence) is used as the first sequence during analysis
        # if sequences is None:
        #     # Todo handle design sequences from a read_fasta_file?
        #     # Todo implement reference sequence from included file(s) or as with self.pose.sequence below
        if isinstance(sequences, dict):
            if sequences:
                design_ids = list(sequences.keys())  # [self.pose.name] +
                sequences = list(sequences.values())  # [self.pose.sequence] +
            else:  # Nothing passed, return an empty DataFrame
                return pd.DataFrame()
        else:
            if isinstance(design_ids, Sequence):
                design_ids = list(design_ids)  # [self.pose.name] +
            else:
                raise ValueError(f"Can't perform {self.analyze_sequence_metrics_per_design.__name__} without argument "
                                 "'design_ids' when 'sequences' isn't a dictionary")
            # All sequences must be string for Biopython
            if isinstance(sequences, np.ndarray):
                sequences = [''.join(sequence) for sequence in sequences.tolist()]  # [self.pose.sequence] +
            elif isinstance(sequences, Sequence):
                # design_sequences = {self.pose.name: self.pose.sequence}
                sequences = [''.join(sequence) for sequence in sequences]  # [self.pose.sequence] +
            else:
                design_sequences = design_ids = sequences = None
                raise ValueError(f"Can't perform {self.analyze_sequence_metrics_per_design.__name__} with argument "
                                 f"'sequences' as a {type(sequences).__name__}. Pass 'sequences' as a Sequence[str]")
        # print(f'Found sequences: {sequences}')
        if len(design_ids) != len(sequences):
            raise ValueError(f"The length of the design_ids ({len(design_ids)}) != sequences ({len(sequences)})")

        # Create numeric sequence types
        number_of_sequences = len(sequences)
        numeric_sequences = sequences_to_numeric(sequences)
        torch_numeric_sequences = torch.from_numpy(numeric_sequences)

        pose_length = self.pose.number_of_residues
        residue_indices = list(range(pose_length))
        nan_blank_data = np.tile(list(repeat(np.nan, pose_length)), (number_of_sequences, 1))

        # Make requisite profiles
        if self.job.design.evolution_constraint:
            self.generate_evolutionary_profile(warn_metrics=True)

        # Try to add each of the profile types in observed_types to profile_background
        profile_background = {}
        if self.measure_evolution:
            profile_background['evolution'] = evolutionary_profile_array = \
                pssm_as_array(self.pose.evolutionary_profile)
            batch_evolutionary_profile = np.tile(evolutionary_profile_array, (number_of_sequences, 1, 1))
            torch_log_evolutionary_profile = torch.from_numpy(np.log(batch_evolutionary_profile))
            per_residue_evolutionary_profile_loss = \
                resources.ml.sequence_nllloss(torch_numeric_sequences, torch_log_evolutionary_profile)
        else:
            # Because we use self.pose.calculate_profile() below, we need to ensure there is a null_profile attached
            if not self.pose.evolutionary_profile:
                self.pose.evolutionary_profile = self.pose.create_null_profile()
            # per_residue_evolutionary_profile_loss = per_residue_design_profile_loss = nan_blank_data
            per_residue_evolutionary_profile_loss = nan_blank_data

        # Load fragment_profile into the analysis
        # if self.job.design.term_constraint and not self.pose.fragment_queries:
        if self.job.design.term_constraint:
            if not self.pose.fragment_queries:
                self.generate_fragments(interface=True)
                self.pose.calculate_fragment_profile()
            profile_background['fragment'] = fragment_profile_array = self.pose.fragment_profile.as_array()
            batch_fragment_profile = np.tile(fragment_profile_array, (number_of_sequences, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                # np.log causes -inf at 0, thus we correct these to a 'large' number
                corrected_frag_array = np.nan_to_num(np.log(batch_fragment_profile), copy=False,
                                                     nan=np.nan, neginf=metrics.zero_probability_frag_value)
            per_residue_fragment_profile_loss = \
                resources.ml.sequence_nllloss(torch_numeric_sequences, torch.from_numpy(corrected_frag_array))
        else:
            per_residue_fragment_profile_loss = nan_blank_data

        # Set up "design" profile
        self.pose.calculate_profile()
        profile_background['design'] = design_profile_array = pssm_as_array(self.pose.profile)
        batch_design_profile = np.tile(design_profile_array, (number_of_sequences, 1, 1))
        if self.pose.fragment_queries:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                # np.log causes -inf at 0, thus we correct these to a 'large' number
                corrected_design_array = np.nan_to_num(np.log(batch_design_profile), copy=False,
                                                       nan=np.nan, neginf=metrics.zero_probability_frag_value)
                torch_log_design_profile = torch.from_numpy(corrected_design_array)
        else:
            torch_log_design_profile = torch.from_numpy(np.log(batch_design_profile))
        per_residue_design_profile_loss = \
            resources.ml.sequence_nllloss(torch_numeric_sequences, torch_log_design_profile)

        if self.job.fragment_db is not None:
            interface_bkgd = np.array(list(self.job.fragment_db.aa_frequencies.values()))
            profile_background['interface'] = np.tile(interface_bkgd, (pose_length, 1))

        # Format all sequence info to DataFrame
        sequence_data = {
            'type': [list(sequence) for sequence in sequences],  # Ensure 2D array like
            'sequence_loss_design': per_residue_design_profile_loss,
            'sequence_loss_evolution': per_residue_evolutionary_profile_loss,
            'sequence_loss_fragment': per_residue_fragment_profile_loss
        }

        sequence_df = pd.concat([pd.DataFrame(data, index=design_ids,
                                              columns=pd.MultiIndex.from_product([residue_indices, [metric]]))
                                 for metric, data in sequence_data.items()], axis=1)

        # if profile_background:
        # Todo This is pretty much already done!
        #  pose_alignment = MultipleSequenceAlignment.from_array(sequences)
        #  Make this capability
        #   pose_alignment.tolist()
        # Ensure we have strings as MultipleSequenceAlignment.from_dictionary() SeqRecord requires ids as strings
        design_names = [str(_id) for _id in design_ids]
        design_sequences = dict(zip(design_names, sequences))
        pose_alignment = MultipleSequenceAlignment.from_dictionary(design_sequences)
        # Todo this must be calculated on the entire Designs batch
        # # Calculate Jensen Shannon Divergence using design frequencies and different profile occurrence data
        # per_residue_divergence_df = \
        #     pd.concat([pd.DataFrame(metrics.position_specific_divergence(pose_alignment.frequencies, background),
        #                             index=design_ids,
        #                             columns=pd.MultiIndex.from_product([residue_indices, [f'divergence_{profile}']]))
        #                for profile, background in profile_background.items()])
        # Perform a frequency extraction for each background profile
        per_residue_background_frequency_df = \
            pd.concat([pd.DataFrame(pose_alignment.get_probabilities_from_profile(background), index=design_ids,
                                    columns=pd.MultiIndex.from_product([residue_indices, [f'observed_{profile}']]))
                       for profile, background in profile_background.items()], axis=1)

        sequences_by_entity: list[list[str]] = []
        for entity in self.pose.entities:
            entity_slice = slice(entity.n_terminal_residue.index, 1 + entity.c_terminal_residue.index)
            sequences_by_entity.append([sequence[entity_slice] for sequence in sequences])

        sequences_split_at_entity: list[tuple[str, ...]] = list(zip(*sequences_by_entity))

        # Calculate hydrophobic collapse for each design
        if self.measure_evolution:
            hydrophobicity = 'expanded'
        else:
            hydrophobicity = 'standard'
        contact_order_per_res_z, reference_collapse, collapse_profile = \
            self.pose.get_folding_metrics(hydrophobicity=hydrophobicity)
        # collapse_significance_threshold = metrics.collapse_thresholds[hydrophobicity]
        if self.measure_evolution:  # collapse_profile.size:  # Not equal to zero, use the profile instead
            reference_collapse = collapse_profile
        #     reference_mean = np.nanmean(collapse_profile, axis=-2)
        #     reference_std = np.nanstd(collapse_profile, axis=-2)
        # else:
        #     reference_mean = reference_std = None
        folding_and_collapse = \
            metrics.collapse_per_residue(sequences_split_at_entity, contact_order_per_res_z, reference_collapse)
        per_residue_collapse_df = pd.concat({design_id: pd.DataFrame(data, index=residue_indices)
                                             for design_id, data in zip(design_ids, folding_and_collapse)},
                                            ).unstack().swaplevel(0, 1, axis=1)
        # Calculate mutational content
        # Make a mutational array, i.e. find those sites that have been mutated from the reference
        # reference_numeric_sequence = numeric_sequences[0]
        reference_numeric_sequence = self.pose.sequence_numeric
        mutational_array = (numeric_sequences - reference_numeric_sequence != 0)
        mutation_df = pd.DataFrame(mutational_array, index=design_ids,
                                   columns=pd.MultiIndex.from_product([residue_indices, ['mutation']]))
        # Join all results
        residues_df = sequence_df.join([  # per_residue_divergence_df,
            # Make background_frequency dataframe according to whether indicated residue was allowed in profile
            per_residue_background_frequency_df > 0,
            per_residue_collapse_df,
            mutation_df])

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
        for entity, entity_sequences in zip(self.pose.entities, sequences_by_entity):
            try:  # Todo add these to the analysis
                entity.h_fields = self.job.api_db.bmdca_fields.retrieve_data(name=entity.name)
                entity.j_couplings = self.job.api_db.bmdca_couplings.retrieve_data(name=entity.name)
                dca_background_residue_energies = entity.direct_coupling_analysis()
                # Todo INSTEAD OF USING BELOW, split Pose.MultipleSequenceAlignment at entity.chain_break...
                entity_alignment = \
                    MultipleSequenceAlignment.from_dictionary(dict(zip(design_names, entity_sequences)))
                # entity_alignment = msa_from_dictionary(entity_sequences[idx])
                dca_design_residue_energies = entity.direct_coupling_analysis(msa=entity_alignment)
                dca_design_residues_concat.append(dca_design_residue_energies)
                # dca_background_energies.append(dca_background_energies.sum(axis=1))
                # dca_design_energies.append(dca_design_energies.sum(axis=1))
                dca_background_energies[entity] = dca_background_residue_energies.sum(axis=1)  # Turns data to 1D
                dca_design_energies[entity] = dca_design_residue_energies.sum(axis=1)
            except AttributeError:
                self.log.warning(f"For {entity.name}, DCA analysis couldn't be performed. "
                                 f"Missing required parameter files")
                dca_succeed = False

        if dca_succeed:
            # concatenate along columns, adding residue index to column, design name to row
            dca_concatenated_df = pd.DataFrame(np.concatenate(dca_design_residues_concat, axis=1),
                                               index=design_ids, columns=residue_indices)
            # dca_concatenated_df.columns = pd.MultiIndex.from_product([dca_concatenated_df.columns, ['dca_energy']])
            dca_concatenated_df = pd.concat([dca_concatenated_df], keys=['dca_energy'], axis=1).swaplevel(0, 1, axis=1)
            # Merge with residues_df
            residues_df = pd.merge(residues_df, dca_concatenated_df, left_index=True, right_index=True)

        return residues_df

    def process_rosetta_metrics(self):
        """From Rosetta based protocols, tally the resulting metrics and integrate with SymDesign metrics database"""
        self.log.debug(f'Found design scores in file: {self.scores_file}')  # Todo PoseJob(.path)
        design_scores = metrics.read_scores(self.scores_file)  # Todo PoseJob(.path)

        # Todo get residues_df['design_indices']
        # Find all designs files
        # if designs is None:
        design_ids = self.design_ids
        design_files = self.get_design_files()  # Todo PoseJob(.path)
        new_design_filenames = []
        rosetta_provided_new_design_names = []
        if self.job.force:
            # Collect metrics on all designs (possibly again)
            for idx, path in enumerate(reversed(design_files[:])):
                file_name, ext = os.path.splitext(os.path.basename(path))
                if file_name in design_ids:  # We already processed this file
                    continue
                else:
                    new_design_filenames.append(path)  # file_name)
                    rosetta_provided_new_design_names.append(file_name)
        else:  # Process to get rid of designs that were already calculated
            for idx, path in enumerate(reversed(design_files[:])):
                file_name, ext = os.path.splitext(os.path.basename(path))
                if file_name in design_ids:  # We already processed this file
                    design_files.pop(idx)
                else:
                    new_design_filenames.append(path)  # file_name)
                    rosetta_provided_new_design_names.append(file_name)

        # Process all desired files to Pose
        designs = [Pose.from_file(file, **self.pose_kwargs) for file in design_files]  # Todo PoseJob(.path)
        # Find designs with scores but no structures
        structure_design_scores = {}
        for pose in designs:
            try:
                structure_design_scores[pose.name] = design_scores.pop(pose.name)
            except KeyError:  # Structure wasn't scored, we will remove this later
                pass

        # Create protocol dataframe
        scores_df = pd.DataFrame.from_dict(structure_design_scores, orient='index')
        # # Fill in all the missing values with that of the default pose_source
        # scores_df = pd.concat([source_df, scores_df]).fillna(method='ffill')
        # Gather all columns into specific types for processing and formatting
        per_res_columns = []
        for column in scores_df.columns.to_list():
            if 'res_' in column:
                per_res_columns.append(column)

        # Check proper input
        metric_set = metrics.necessary_metrics.difference(set(scores_df.columns))
        # self.log.debug('Score columns present before required metric check: %s' % scores_df.columns.to_list())
        if metric_set:
            raise DesignError(f'Missing required metrics: "{", ".join(metric_set)}"')

        # Remove unnecessary (old scores) as well as Rosetta pose score terms besides ref (has been renamed above)
        # Todo learn know how to produce Rosetta score terms in output score file. Not in FastRelax...
        remove_columns = metrics.rosetta_terms + metrics.unnecessary + per_res_columns
        # Todo remove dirty when columns are correct (after P432)
        #  and column tabulation precedes residue/hbond_processing

        # Drop designs where required data isn't present
        # Format protocol columns
        # # Todo remove not DEV
        # missing_group_indices = scores_df[putils.protocol].isna()
        # scout_indices = [idx for idx in scores_df[missing_group_indices].index if 'scout' in idx]
        # scores_df.loc[scout_indices, putils.protocol] = putils.scout
        # structure_bkgnd_indices = [idx for idx in scores_df[missing_group_indices].index if 'no_constraint' in idx]
        # scores_df.loc[structure_bkgnd_indices, putils.protocol] = putils.structure_background
        # # Todo Done remove
        missing_group_indices = scores_df[putils.protocol].isna()
        # protocol_s.replace({'combo_profile': putils.design_profile}, inplace=True)  # ensure proper profile name

        scores_df.drop(missing_group_indices, axis=0, inplace=True, errors='ignore')
        # protocol_s.drop(missing_group_indices, inplace=True, errors='ignore')
        # Find protocol info and remove from scores_df
        if putils.design_parent in scores_df:
            # Set as None to start. We will update after the fact
            design_parent = None  # parent_s
            parents = scores_df.pop(putils.design_parent).tolist()
        else:
            # Assume this is a offspring of the pose
            design_parent = self.pose_source
            parents = None

        # Update the Pose.design by the number of new designs,
        # to generate DesignData with a prescribed name and design design_parent (which may be updated below)
        new_designs_data = self.update_design_data(design_parent=design_parent, number=len(new_design_filenames))

        if parents is not None:
            for design_data, parent, provided_name in zip(new_designs_data, parents, rosetta_provided_new_design_names):
                design_data.design_parent = parent
                design_data.provided_name = provided_name
        else:
            for design_data, provided_name in zip(new_designs_data, rosetta_provided_new_design_names):
                design_data.provided_name = provided_name

        # This is all done in update_design_data
        # self.designs.append(new_designs_data)
        # # Flush the newly acquired DesignData and DesignProtocol to generate .id primary keys
        # self.job.current_session.flush()
        new_design_ids = [design_data.id for design_data in new_designs_data]

        # Take metrics for the pose_source
        entity_energies = [0. for _ in self.pose.entities]
        pose_source_residue_info = \
            {residue.index: {'complex': 0., 'bound': entity_energies.copy(), 'unbound': entity_energies.copy(),
                             'solv_complex': 0., 'solv_bound': entity_energies.copy(),
                             'solv_unbound': entity_energies.copy(), 'fsp': 0., 'cst': 0., 'hbond': 0}
             for entity in self.pose.entities for residue in entity.residues}
        pose_name = self.pose.name
        residue_info = {pose_name: pose_source_residue_info}

        # residue_info = {'energy': {'complex': 0., 'unbound': 0.}, 'type': None, 'hbond': 0}
        residue_info.update(self.pose.rosetta_residue_processing(structure_design_scores))
        # Can't use residue_processing (clean) ^ in the case there is a design without metrics... columns not found!
        interface_hbonds = metrics.dirty_hbond_processing(structure_design_scores)
        # Can't use hbond_processing (clean) in the case there is a design without metrics... columns not found!
        # interface_hbonds = hbond_processing(structure_design_scores, hbonds_columns)
        # Convert interface_hbonds to indices
        interface_hbonds = {design: [residue.index for residue in self.pose.get_residues(hbond_residues)]
                            for design, hbond_residues in interface_hbonds.items()}
        residue_info = metrics.process_residue_info(residue_info, hbonds=interface_hbonds)

        viable_designs = scores_df.index.to_list()
        if not viable_designs:
            raise DesignError(f'No viable designs remain after {self.process_rosetta_metrics.__name__} data processing '
                              f'steps')

        self.log.debug(f'Viable designs with structures remaining after cleaning:\n\t{", ".join(viable_designs)}')

        # Process mutational frequencies, H-bond, and Residue energy metrics to dataframe
        rosetta_info_df = pd.concat({design: pd.DataFrame(info) for design, info in residue_info.items()})
        # Returns multi-index column with residue number as first (top) column index, metric as second index
        # Set each position that was parsed as "designable"
        # Todo does this include packable residues from neighborhoods?
        rosetta_info_df = rosetta_info_df.stack().unstack(1)
        # During rosetta_info_df unstack, all residues with missing dicts are copied as nan
        rosetta_info_df['design_residue'] = 1
        rosetta_info_df = rosetta_info_df.unstack().swaplevel(0, 1, axis=1)
        # Todo implement this protocol if sequence data is taken at multiple points along a trajectory and the
        #  sequence data along trajectory is a metric on it's own
        # # Gather mutations for residue specific processing and design sequences
        # for design, data in list(structure_design_scores.items()):  # make a copy as can be removed
        #     sequence = data.get('final_sequence')
        #     if sequence:
        #         if len(sequence) >= pose_length:
        #             pose_sequences[design] = sequence[:pose_length]  # Todo won't work if design had insertions
        #         else:
        #             pose_sequences[design] = sequence
        #     else:
        #         self.log.warning('Design %s is missing sequence data, removing from design pool' % design)
        #         structure_design_scores.pop(design)
        # # format {entity: {design_name: sequence, ...}, ...}
        # entity_sequences = \
        #     {entity: {design: sequence[entity.n_terminal_residue.number - 1:entity.c_terminal_residue.number]
        #               for design, sequence in pose_sequences.items()} for entity in self.pose.entities}
        # return rosetta_info_df

        scores_df.drop(remove_columns, axis=1, inplace=True, errors='ignore')

        # # Find protocols for protocol specific data processing removing from scores_df
        # protocol_s = scores_df.pop(putils.protocol).copy()
        # designs_by_protocol = protocol_s.groupby(protocol_s).groups
        # unique_protocols = list(designs_by_protocol.keys())
        # # Remove refine and consensus if present as there was no design done over multiple protocols
        # # Todo change if we did multiple rounds of these protocols
        # designs_by_protocol.pop(putils.refine, None)
        # designs_by_protocol.pop(putils.consensus, None)
        # # Get unique protocols
        # unique_design_protocols = set(designs_by_protocol.keys())
        # self.log.info(f'Unique Design Protocols: {", ".join(unique_design_protocols)}')

        # Replace empty strings with np.nan and convert remaining to float
        scores_df.replace('', np.nan, inplace=True)
        scores_df.fillna(dict(zip(metrics.protocol_specific_columns, repeat(0))), inplace=True)
        # scores_df = scores_df.astype(float)  # , copy=False, errors='ignore')

        # Calculate metrics from combinations of metrics with variable integer number metric names
        scores_columns = scores_df.columns.to_list()
        self.log.debug(f'Metrics present: {scores_columns}')
        summation_pairs = \
            {'buns_unbound': list(filter(re.compile('buns[0-9]+_unbound$').match, scores_columns)),  # Rosetta
             # 'interface_energy_bound':
             #     list(filter(re.compile('interface_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
             # 'interface_energy_unbound':
             #     list(filter(re.compile('interface_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
             # 'interface_solvation_energy_bound':
             #     list(filter(re.compile('solvation_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
             # 'interface_solvation_energy_unbound':
             #     list(filter(re.compile('solvation_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
             'interface_connectivity':
                 list(filter(re.compile('entity[0-9]+_interface_connectivity').match, scores_columns)),  # Rosetta
             }
        scores_df = metrics.columns_to_new_column(scores_df, summation_pairs)

        scores_df.drop(metrics.clean_up_intermediate_columns, axis=1, inplace=True, errors='ignore')
        repacking = scores_df.get('repacking')
        if repacking is not None:
            # Set interface_bound_activation_energy = np.nan where repacking is 0
            # Currently is -1 for True (Rosetta Filter quirk...)
            scores_df.loc[scores_df[repacking == 0].index, 'interface_bound_activation_energy'] = np.nan
            scores_df.drop('repacking', axis=1, inplace=True)

        # The DataFrame.index is wrong here. It needs to become the design.id not design.name. Modify after processing
        pose_sequences = {pose.name: pose.sequence for pose in designs}
        residues_df = self.analyze_residue_metrics_per_design(designs=designs)
        # Join Rosetta per-residue with Structure analysis per-residue like DataFrames
        residues_df = pd.concat([residues_df, rosetta_info_df], axis=1)
        designs_df = scores_df.join(self.analyze_design_metrics_per_design(residues_df, designs))

        sequences_df = self.analyze_sequence_metrics_per_design(sequences=pose_sequences)
        designs_df = designs_df.join(self.analyze_design_metrics_per_residue(sequences_df))

        # Join per-residue like DataFrames
        # Each of these could have different index/column, so we use concat to perform an outer merge
        residues_df = pd.concat([residues_df, sequences_df], axis=1)
        # Todo should this "different index" be allowed? be possible
        #  residues_df = residues_df.join(rosetta_info_df)

        # Rename all designs and clean up resulting metrics for storage
        # In keeping with "unit of work", only rename once all data is processed incase we run into any errors
        # Get the filenames mapped to the DesignData
        # filename_to_design_data_map = list(zip(new_design_filenames, new_designs_data))
        # filename_to_new_design_names = dict(zip(new_design_filenames, new_design_files))
        design_name_to_id_map = \
            dict(((design_data.provided_name, design_data.id) for design_data in new_designs_data))
        # design_name_to_id_map = {}
        # filename_to_new_design_names = {}
        # for filename, design_data in filename_to_design_data_map:
        #     filename_to_new_design_names[filename] = os.path.join(self.designs_path, f'{design_data.name}.pdb')
        #     design_name_to_id_map[filename] = design_data.id
        designs_df.index = designs_df.index.map(design_name_to_id_map)
        residues_df.index = residues_df.index.map(design_name_to_id_map)

        designs_path = self.designs_path
        new_design_new_filenames = [os.path.join(designs_path, f'{design_data.name}.pdb')
                                    for design_data in new_designs_data]
        # Commit the newly acquired metrics to the database
        # First check if the files are situated correctly
        files_to_move = {}
        for filename, new_filename in zip(new_design_filenames, new_design_new_filenames):
            if filename == new_filename:
                # These are the same file, proceed without processing
                continue
            elif os.path.exists(filename) and not os.path.exists(new_filename):
                # We have the target file and nothing exists where we are moving it
                files_to_move[filename] = new_filename
            else:
                raise DesignError('The specified file renaming scheme creates a conflict:\n'
                                  f'\t{filename} -> {new_filename}')
        # If so, proceed with insert, rename and commit
        self.output_metrics(residues=residues_df, designs=designs_df)
        # Rename the incoming files to their prescribed names
        for filename, new_filename in files_to_move.items():
            shutil.move(filename, new_filename)

        # Find protocol info and remove from scores_df
        protocol_s = scores_df.pop(putils.protocol).copy()
        # Update the Pose with the design protocols
        for idx, design_data in enumerate(new_designs_data):
            design_data.protocols.append(
                sql.DesignProtocol(design_id=new_design_ids[idx],
                                   protocol=protocol_s[idx],
                                   # temperature=temperatures[idx],
                                   file=new_design_new_filenames[idx]))  # design_files[idx]))
        self.job.current_session.commit()

    def calculate_pose_metrics(self):
        """Perform a metrics update only on the reference Pose"""
        self.load_pose()

        # Check if PoseMetrics have been captured
        if self.job.db:
            if self.metrics is None:
                self.metrics = self.pose.metrics  # Also calculates entity.metrics
                # pose_metrics = self.pose.metrics
                # pose_metrics.pose_id = self.id
                # Add metrics objects to the current session
                # self.job.current_session.add(pose_metrics)
                idx = 1
                is_thermophilic = []
                for idx, (entity, data) in enumerate(zip(self.pose.entities, self.entity_data), idx):
                    # Todo remove entity.thermophilic once sql load more streamlined
                    is_thermophilic.append(1 if entity.thermophilic else 0)
                    data.metrics = entity.metrics
                    # entity.metrics.pose_id = self.id
                    # self.job.current_session.add(entity.metrics)

                self.metrics.pose_thermophilicity = sum(is_thermophilic) / idx

            else:
                return
        else:
            raise NotImplementedError(f"This method, {self.calculate_pose_metrics.__name__} doesn't output anything yet"
                                      f" when {type(self.job).__name__}.db = {self.job.db}")
            raise NotImplementedError(f"The reference=SymEntry.resulting_symmetry center_of_mass is needed as well")
            pose_df = self.pose.df  # Also performs entity.calculate_metrics()

            entity_dfs = []
            for entity in self.pose.entities:
                # entity.calculate_metrics()  # Todo add reference=
                # entity_dfs.append(entity.df)
                entity_s = pd.Series(**entity.calculate_metrics())  # Todo add reference=
                entity_dfs.append(entity_s)

            # Stack the Series on the columns to turn into a dataframe where the metrics are rows and entity are columns
            entity_df = pd.concat(entity_dfs, keys=list(range(1, 1 + len(entity_dfs))), axis=1)

        # Output
        residues_df = self.analyze_pose_metrics_per_residue()
        self.output_metrics(residues=residues_df)
        # Commit the newly acquired metrics
        self.job.current_session.commit()

    def analyze_pose_metrics_per_residue(self, novel_interface: bool = True) -> pd.DataFrame:
        """Perform per-residue analysis on the PoseJob.pose

        Returns:
            A per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
        """
        self.load_pose()

        pose_length = self.pose.number_of_residues
        residue_indices = list(range(pose_length))

        if novel_interface:  # The input structure wasn't meant to be together, take the errat measurement as such
            source_errat = []
            for idx, entity in enumerate(self.pose.entities):
                # Todo when Entity.oligomer works
                #  _, oligomeric_errat = entity.oligomer.errat(out_path=os.path.devnull)
                entity_oligomer = Model.from_chains(entity.chains, entities=False, log=self.pose.log)
                _, oligomeric_errat = entity_oligomer.errat(out_path=os.path.devnull)
                source_errat.append(oligomeric_errat[:entity.number_of_residues])
            # atomic_deviation[pose_source_name] = sum(source_errat_accuracy) / float(self.pose.number_of_entities)
            pose_source_errat = np.concatenate(source_errat)
        else:
            # pose_assembly_minimally_contacting = self.pose.assembly_minimally_contacting
            # # atomic_deviation[pose_source_name], pose_per_residue_errat = \
            # _, pose_per_residue_errat = \
            #     pose_assembly_minimally_contacting.errat(out_path=os.path.devnull)
            # pose_source_errat = pose_per_residue_errat[:pose_length]
            # Get errat measurement
            # per_residue_data[pose_source_name].update(self.pose.per_residue_interface_errat())
            pose_source_errat = self.pose.per_residue_interface_errat()['errat_deviation']

        pose_name = self.pose.name
        # Collect reference Structure metrics
        per_residue_data = {pose_name:
                            {**self.pose.per_residue_interface_surface_area(),
                             **self.pose.per_residue_contact_order(),
                             'errat_deviation': pose_source_errat}
                            }
        # Convert per_residue_data into a dataframe matching residues_df orientation
        residues_df = pd.concat({name: pd.DataFrame(data, index=residue_indices)
                                for name, data in per_residue_data.items()}).unstack().swaplevel(0, 1, axis=1)

        sequences_df = self.analyze_sequence_metrics_per_design(sequences=[self.pose.sequence],
                                                                design_ids=[pose_name])
        return residues_df.join(sequences_df)

    def analyze_residue_metrics_per_design(self, designs: Iterable[Pose] | Iterable[AnyStr]) -> pd.DataFrame:
        """Perform per-residue analysis on design Model instances

        Args:
            designs: The designs to perform analysis on. By default, fetches all available structures
        Returns:
            A per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
        """
        self.load_pose()

        # CAUTION: Assumes each structure is the same length
        pose_length = self.pose.number_of_residues
        residue_indices = list(range(pose_length))
        # residue_numbers = [residue.number for residue in self.pose.residues]
        # design_residue_indices = [residue.index for residue in self.pose.design_residues]
        # interface_residue_indices = [residue.index for residue in self.pose.interface_residues]

        # Compute structural measurements for all designs
        per_residue_data: dict[str, dict[str, Any]] = {}
        # for pose in [self.pose] + designs:  # Takes 1-2 seconds for Structure -> assembly -> errat
        for pose in designs:  # Takes 1-2 seconds for Structure -> assembly -> errat
            try:
                pose_name = pose.name
            except AttributeError:  # This is likely a filepath
                pose = Pose.from_file(pose, **self.pose_kwargs)
                pose_name = pose.name
            # # Must find interface residues before measure local_density
            # pose.find_and_split_interface()
            # per_residue_data[pose_name] = pose.per_residue_interface_surface_area()
            # # Get errat measurement
            # per_residue_data[pose_name].update(pose.per_residue_interface_errat())
            per_residue_data[pose_name] = {
                **pose.per_residue_interface_surface_area(),
                **pose.per_residue_interface_errat()}

        # Convert per_residue_data into a dataframe matching residues_df orientation
        residues_df = pd.concat({name: pd.DataFrame(data, index=residue_indices)
                                for name, data in per_residue_data.items()}).unstack().swaplevel(0, 1, axis=1)

        # Make buried surface area (bsa) columns, and residue classification
        residues_df = metrics.calculate_residue_surface_area(residues_df)

        return residues_df

    def analyze_design_metrics_per_design(self, residues_df: pd.DataFrame,
                                          designs: Iterable[Pose] | Iterable[AnyStr]) -> pd.DataFrame:
        """Take every design Model and perform design level structural analysis

        Args:
            residues_df: The typical per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
            designs: The designs to perform analysis on. By default, fetches all available structures
        Returns:
            The designs_df DataFrame with updated columns containing per-design metrics
        """
        #     designs_df: The typical per-design metric DataFrame where each index is the design id and the columns are
        #         design metrics
        # Find all designs files
        #   Todo fold these into Model(s) and attack metrics from Pose objects?
        # if designs is None:
        #     designs = []

        # Compute structural measurements for all designs
        interface_local_density = {}  # self.pose.name: self.pose.local_density_interface()}
        # for pose in [self.pose] + designs:  # Takes 1-2 seconds for Structure -> assembly -> errat
        for pose in designs:  # Takes 1-2 seconds for Structure -> assembly -> errat
            try:
                pose_name = pose.name
            except AttributeError:  # This is likely a filepath
                pose = Pose.from_file(pose, **self.pose_kwargs)
                pose_name = pose.name
            # Must find interface residues before measure local_density
            pose.find_and_split_interface()
            interface_local_density[pose_name] = pose.local_density_interface()

        # designs_df = pd.Series(interface_local_density, index=residues_df.index,
        #                        name='interface_local_density').to_frame()
        designs_df = metrics.sum_per_residue_metrics(residues_df)
        designs_df['interface_local_density'] = pd.Series(interface_local_density)

        # Make designs_df errat_deviation that takes into account the pose_source sequence errat_deviation
        # Get per-residue errat scores from the residues_df
        errat_df = residues_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)

        # Todo improve efficiency by using precomputed. Something like:
        #  stmt = select(ResidueMetrics).where(ResidueMetrics.pose_id == self.id, ResidueMetrics.name == self.name)
        #  rows = current_session.scalars(stmt)
        #  row_dict = {row.index: row.errat_deviation for row in rows}
        #  pd.Series(row_dict, name='errat_Deviation')
        # pose_source_errat = errat_df.loc[self.pose.name, :]
        pose_source_errat = self.pose.per_residue_interface_errat()['errat_deviation']
        # Include in errat_deviation if errat score is < 2 std devs and isn't 0 to begin with
        source_errat_inclusion_boolean = \
            np.logical_and(pose_source_errat < metrics.errat_2_sigma, pose_source_errat != 0.)
        # Find residues where designs deviate above wild-type errat scores
        errat_sig_df = errat_df.sub(pose_source_errat, axis=1) > metrics.errat_1_sigma  # axis=1 is per-residue subtract
        # Then select only those residues which are expressly important by the inclusion boolean
        # This overwrites the metrics.sum_per_residue_metrics() value
        designs_df['errat_deviation'] = (errat_sig_df.loc[:, source_errat_inclusion_boolean] * 1).sum(axis=1)

        # # Calculate metrics from combinations of metrics with variable integer number metric names
        # scores_columns = designs_df.columns.to_list()
        # self.log.debug(f'Metrics present: {scores_columns}')
        #
        # summation_pairs = \
        #     {'buns_unbound': list(filter(re.compile('buns[0-9]+_unbound$').match, scores_columns)),  # Rosetta
        #      # 'interface_energy_bound':
        #      #     list(filter(re.compile('interface_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
        #      # 'interface_energy_unbound':
        #      #     list(filter(re.compile('interface_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
        #      # 'interface_solvation_energy_bound':
        #      #     list(filter(re.compile('solvation_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
        #      # 'interface_solvation_energy_unbound':
        #      #     list(filter(re.compile('solvation_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
        #      'interface_connectivity':
        #          list(filter(re.compile('entity[0-9]+_interface_connectivity').match, scores_columns)),  # Rosetta
        #      }
        #
        # designs_df = metrics.columns_to_new_column(designs_df, summation_pairs)
        # designs_df = metrics.columns_to_new_column(designs_df, metrics.rosetta_delta_pairs, mode='sub')
        # Add number_interface_residues for div_pairs and int_comp_similarity
        # designs_df['number_interface_residues'] = other_pose_metrics.pop('number_interface_residues')
        self.load_pose()
        pose_df = self.pose.df
        designs_df['number_interface_residues'] = pose_df['number_interface_residues']

        # Find the proportion of the residue surface area that is solvent accessible versus buried in the interface
        # if 'interface_area_total' in designs_df and 'area_total_complex' in designs_df:
        interface_bsa_df = designs_df['interface_area_total']
        designs_df['interface_area_to_residue_surface_ratio'] = \
            (interface_bsa_df / (interface_bsa_df + designs_df['area_total_complex']))

        # designs_df['interface_area_total'] = pose_df['interface_area_total']
        designs_df = metrics.columns_to_new_column(designs_df, metrics.division_pairs, mode='truediv')
        designs_df['interface_composition_similarity'] = \
            designs_df.apply(metrics.interface_composition_similarity, axis=1)
        designs_df = designs_df.drop(['number_interface_residues'], axis=1)
        #                               'interface_area_total']

        return designs_df

    def analyze_design_metrics_per_residue(self, residues_df: pd.DataFrame) -> pd.DataFrame:
        # designs_df: pd.DataFrame = None
        """Take every design in the residues_df and perform design level statistical analysis

        Args:
            residues_df: The typical per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
        Returns:
            The designs_df DataFrame with updated columns containing per-design metrics
        """
        #     designs_df: The typical per-design metric DataFrame where each index is the design id and the columns are
        #         design metrics
        self.load_pose()
        # self.identify_interface()

        # Load fragment_profile into the analysis
        if self.job.design.term_constraint and not self.pose.fragment_queries:
            self.generate_fragments(interface=True)
            self.pose.calculate_fragment_profile()

        # CAUTION: Assumes each structure is the same length
        pose_length = self.pose.number_of_residues
        residue_indices = list(range(pose_length))
        # residue_numbers = [residue.number for residue in self.pose.residues]
        # design_residue_indices = [residue.index for residue in self.pose.design_residues]
        # interface_residue_indices = [residue.index for residue in self.pose.interface_residues]

        # pose_sequences = {pose.name: pose.sequence for pose in designs}
        # sequences_df = self.analyze_sequence_metrics_per_design(pose_sequences)

        # # Gather miscellaneous pose specific metrics
        # # other_pose_metrics = self.pose.df  # calculate_metrics()
        # # Create metrics for the pose_source
        # empty_source = dict(
        #     # **other_pose_metrics,
        #     buns_complex=0,
        #     contact_count=0,
        #     favor_residue_energy=0,
        #     interaction_energy_complex=0,
        #     interaction_energy_per_residue=0,
        #     interface_separation=0,
        #     number_of_hbonds=0,
        #     rmsd_complex=0,  # Todo calculate this here instead of Rosetta using superposition3d
        #     rosetta_reference_energy=0,
        #     shape_complementarity=0,
        # )
        # job_key = 'no_energy'
        # empty_source[putils.protocol] = job_key
        # for idx, entity in enumerate(self.pose.entities, 1):
        #     empty_source[f'buns{idx}_unbound'] = 0
        #     empty_source[f'entity{idx}_interface_connectivity'] = 0
        #
        # source_df = pd.DataFrame(empty_source, index=[self.pose.name])
        # # Get the scores from the score file on design trajectory metrics
        # if os.path.exists(self.scores_file):  # Rosetta scores file is present  # Todo PoseJob(.path)
        #     design_was_performed = True
        #     self.log.debug(f'Found design scores in file: {self.scores_file}')  # Todo PoseJob(.path)
        #     design_scores = metrics.read_scores(self.scores_file)  # Todo PoseJob(.path)
        #     # Find designs with scores but no structures
        #     structure_design_scores = {}
        #     for pose in designs:
        #         try:
        #             structure_design_scores[pose.name] = design_scores.pop(pose.name)
        #         except KeyError:  # Structure wasn't scored, we will remove this later
        #             pass
        #
        #     # Create protocol dataframe
        #     designs_df = pd.DataFrame.from_dict(structure_design_scores, orient='index')
        #     # Gather all columns into specific types for processing and formatting
        #     per_res_columns = []
        #     for column in designs_df.columns.to_list():
        #         if 'res_' in column:
        #             per_res_columns.append(column)
        #
        #     # Check proper input
        #     metric_set = metrics.necessary_metrics.difference(set(designs_df.columns))
        #     # self.log.debug('Score columns present before required metric check: %s' % designs_df.columns.to_list())
        #     if metric_set:
        #         raise DesignError(f'Missing required metrics: "{", ".join(metric_set)}"')
        #
        #     # Remove unnecessary (old scores) as well as Rosetta pose score terms besides ref (has been renamed above)
        #     # Todo learn know how to produce Rosetta score terms in output score file. Not in FastRelax...
        #     remove_columns = metrics.rosetta_terms + metrics.unnecessary + per_res_columns
        #     # Todo remove dirty when columns are correct (after P432)
        #     #  and column tabulation precedes residue/hbond_processing
        #
        #     # Drop designs where required data isn't present
        #     # Format protocol columns
        #     # Todo remove not DEV
        #     missing_group_indices = designs_df[putils.protocol].isna()
        #     scout_indices = [idx for idx in designs_df[missing_group_indices].index if 'scout' in idx]
        #     designs_df.loc[scout_indices, putils.protocol] = putils.scout
        #     structure_bkgnd_indices = [idx for idx in designs_df[missing_group_indices].index if 'no_constraint' in idx]
        #     designs_df.loc[structure_bkgnd_indices, putils.protocol] = putils.structure_background
        #     # Todo Done remove
        #     missing_group_indices = designs_df[putils.protocol].isna()
        #     # protocol_s.replace({'combo_profile': putils.design_profile}, inplace=True)  # ensure proper profile name
        #
        #     designs_df.drop(missing_group_indices, axis=0, inplace=True, errors='ignore')
        #     # protocol_s.drop(missing_group_indices, inplace=True, errors='ignore')
        #
        #     viable_designs = designs_df.index.to_list()
        #     if not viable_designs:
        #         raise DesignError('No viable designs remain after processing!')
        #
        #     self.log.debug(f'Viable designs with structures remaining after cleaning:\n\t{", ".join(viable_designs)}')
        #
        #     # Todo implement this protocol if sequence data is taken at multiple points along a trajectory and the
        #     #  sequence data along trajectory is a metric on it's own
        #     # # Gather mutations for residue specific processing and design sequences
        #     # for design, data in list(structure_design_scores.items()):  # make a copy as can be removed
        #     #     sequence = data.get('final_sequence')
        #     #     if sequence:
        #     #         if len(sequence) >= pose_length:
        #     #             pose_sequences[design] = sequence[:pose_length]  # Todo won't work if design had insertions
        #     #         else:
        #     #             pose_sequences[design] = sequence
        #     #     else:
        #     #         self.log.warning('Design %s is missing sequence data, removing from design pool' % design)
        #     #         structure_design_scores.pop(design)
        #     # # format {entity: {design_name: sequence, ...}, ...}
        #     # entity_sequences = \
        #     #     {entity: {design: sequence[entity.n_terminal_residue.number - 1:entity.c_terminal_residue.number]
        #     #               for design, sequence in pose_sequences.items()} for entity in self.pose.entities}
        # else:
        #     design_was_performed = False
        #     self.log.debug(f'Missing design scores file at {self.scores_file}')  # Todo PoseJob(.path)
        #     designs_df = pd.DataFrame.from_dict({pose.name: {putils.protocol: job_key} for pose in designs},
        #                                        orient='index')
        #     remove_columns = metrics.rosetta_terms + metrics.unnecessary
        #     # residue_info.update({struct_name: pose_source_residue_info for struct_name in designs_df.index.to_list()})
        #     # # Todo add relevant missing scores such as those specified as 0 above
        #     # # Todo generate energy scores internally which matches output from residue_processing
        #     # viable_designs = [pose.name for pose in designs]
        #
        # designs_df.drop(remove_columns, axis=1, inplace=True, errors='ignore')

        # Find protocols for protocol specific data processing removing from designs_df
        # protocol_s = designs_df.pop(putils.protocol).copy()
        # designs_by_protocol = protocol_s.groupby(protocol_s).groups
        # # unique_protocols = list(designs_by_protocol.keys())
        # # Remove refine and consensus if present as there was no design done over multiple protocols
        # # Todo change if we did multiple rounds of these protocols
        # designs_by_protocol.pop(putils.refine, None)
        # designs_by_protocol.pop(putils.consensus, None)
        # # Get unique protocols
        # unique_design_protocols = set(designs_by_protocol.keys())
        # self.log.info(f'Unique Design Protocols: {", ".join(unique_design_protocols)}')

        # # Fill in all the missing values with that of the default pose_source
        # designs_df = pd.concat([source_df, designs_df]).fillna(method='ffill')

        # Calculate new metrics from combinations of other metrics
        # Add design residue information to designs_df such as how many core, rim, and support residues were measured
        # if designs_df is None:
        designs_df = metrics.sum_per_residue_metrics(residues_df)
        # else:
        #     # Must perform concat to get pose_source values...
        #     designs_df = pd.concat([designs_df, metrics.sum_per_residue_metrics(residues_df)], axis=1)
        #     # mean_columns = ['hydrophobicity', 'sasa_relative_bound', 'sasa_relative_complex']
        #     # designs_df = designs_df.join(metrics.sum_per_residue_metrics(residues_df))  # , mean_metrics=mean_columns))

        pose_df = self.pose.df
        # Calculate mutational content
        # designs_df['number_of_mutations'] = mutation_df.sum(axis=1)
        designs_df['percent_mutations'] = designs_df['number_of_mutations'] / pose_length

        mutation_df = residues_df.loc[:, idx_slice[:, 'mutation']]
        idx = 1
        # prior_slice = 0
        for idx, entity in enumerate(self.pose.entities, idx):
            # entity_n_terminal_residue_index = entity.n_terminal_residue.index
            # entity_c_terminal_residue_index = entity.c_terminal_residue.index
            designs_df[f'entity{idx}_number_of_mutations'] = \
                mutation_df.loc[:, idx_slice[residue_indices[entity.n_terminal_residue.index:  # prior_slice
                                                             1 + entity.c_terminal_residue.index], :]].sum(axis=1)
            # prior_slice = entity_c_terminal_residue_index
            designs_df[f'entity{idx}_percent_mutations'] = \
                designs_df[f'entity{idx}_number_of_mutations'] / entity.number_of_residues

        designs_df['sequence_loss_design_per_residue'] = designs_df['sequence_loss_design'] / pose_length
        # The per residue average loss compared to the design profile
        designs_df['sequence_loss_evolution_per_residue'] = designs_df['sequence_loss_evolution'] / pose_length
        # The per residue average loss compared to the evolution profile
        designs_df['sequence_loss_fragment_per_residue'] = \
            designs_df['sequence_loss_fragment'] / pose_df['number_fragment_residues_total']
        # The per residue average loss compared to the fragment profile

        # designs_df['collapse_new_positions'] /= pose_length
        # designs_df['collapse_new_position_significance'] /= pose_length
        designs_df['collapse_significance_by_contact_order_z_mean'] = \
            designs_df['collapse_significance_by_contact_order_z'] / \
            (residues_df.loc[:, idx_slice[:, 'collapse_significance_by_contact_order_z']] != 0).sum(axis=1)
        # if self.measure_alignment:
        # Todo THESE ARE NOW DIFFERENT SOURCE if not self.measure_alignment
        collapse_increased_df = residues_df.loc[:, idx_slice[:, 'collapse_increased_z']]
        total_increased_collapse = (collapse_increased_df != 0).sum(axis=1)
        # designs_df['collapse_increase_significance_by_contact_order_z_mean'] = \
        #     designs_df['collapse_increase_significance_by_contact_order_z'] / total_increased_collapse
        # designs_df['collapse_increased_z'] /= pose_length
        designs_df['collapse_increased_z_mean'] = collapse_increased_df.sum(axis=1) / total_increased_collapse
        designs_df['collapse_violation'] = designs_df['collapse_new_positions'] > 0
        designs_df['collapse_variance'] = designs_df['collapse_deviation_magnitude'] / pose_length
        designs_df['collapse_sequential_peaks_z_mean'] = \
            designs_df['collapse_sequential_peaks_z'] / total_increased_collapse
        designs_df['collapse_sequential_z_mean'] = designs_df['collapse_sequential_z'] / total_increased_collapse

        # Ensure summed observed_types are taken as the average over the pose_length
        for _type in observed_types:
            try:
                designs_df[f'observed_{_type}'] /= pose_length
            except KeyError:
                continue
        # POSE ANALYSIS
        # designs_df = pd.concat([designs_df, proteinmpnn_df], axis=1)
        # designs_df.dropna(how='all', inplace=True, axis=1)  # Remove completely empty columns
        # # # Refine is not considered sequence design and destroys mean. remove v
        # # # designs_df = designs_df.sort_index().drop(putils.refine, axis=0, errors='ignore')
        # # # Consensus cst_weights are very large and destroy the mean.
        # # # Remove this drop for consensus or refine if they are run multiple times
        # # designs_df = \
        # #     designs_df.drop([self.pose.name, putils.refine, putils.consensus], axis=0, errors='ignore').sort_index()
        # #
        # # # Get total design statistics for every sequence in the pose and every protocol specifically
        # # designs_df[putils.protocol] = protocol_s
        # # protocol_groups = designs_df.groupby(putils.protocol)
        # #
        # # pose_stats, protocol_stats = [], []
        # # for idx, stat in enumerate(stats_metrics):
        # #     # Todo both groupby calls have this warning
        # #     #  FutureWarning: The default value of numeric_only in DataFrame.mean is deprecated. In a future version,
        # #     #  it will default to False. In addition, specifying 'numeric_only=None' is deprecated. Select only valid
        # #     #  columns or specify the value of numeric_only to silence this warning.
        # #     pose_stats.append(getattr(designs_df, stat)().rename(stat))
        # #     protocol_stats.append(getattr(protocol_groups, stat)())
        # #
        # # # Add the number of observations of each protocol
        # # protocol_stats[stats_metrics.index(mean)]['observations'] = protocol_groups.size()
        # #
        # # # Change statistic names for all df that are not groupby means for the final trajectory dataframe
        # # for idx, stat in enumerate(stats_metrics):
        # #     if stat != mean:
        # #         protocol_stats[idx] = protocol_stats[idx].rename(index={protocol: f'{protocol}_{stat}'
        # #                                                                 for protocol in unique_design_protocols})
        # # # Remove std rows if there is no stdev
        # # # final_trajectory_indices = designs_df.index.to_list() + unique_protocols + [mean]
        # # designs_df = pd.concat([designs_df]
        # #                        + [df.dropna(how='all', axis=0) for df in protocol_stats]  # v don't add if nothing
        # #                        + [pd.to_numeric(s).to_frame().T for s in pose_stats if not all(s.isna())])
        # # # This concat puts back self.pose.name, refine, consensus index as protocol_stats is calculated on designs_df
        # # designs_df = designs_df.fillna({'observations': 1})
        #
        # # # Calculate protocol significance
        # # pvalue_df = pd.DataFrame()
        # # scout_protocols = list(filter(re.compile(f'.*{putils.scout}').match,
        # #                               protocol_s[~protocol_s.isna()].unique().tolist()))
        # # similarity_protocols = unique_design_protocols.difference([putils.refine, job_key] + scout_protocols)
        # # if putils.structure_background not in unique_design_protocols:
        # #     self.log.info(f'Missing background protocol "{putils.structure_background}". No protocol significance '
        # #                   f'measurements available for this pose')
        # # elif len(similarity_protocols) == 1:  # measure significance
        # #     self.log.info("Can't measure protocol significance, only one protocol of interest")
        # # else:  # Test significance between all combinations of protocols by grabbing mean entries per protocol
        # #     for prot1, prot2 in combinations(sorted(similarity_protocols), 2):
        # #         select_df = \
        # #             designs_df.loc[[design for designs in [designs_by_protocol[prot1], designs_by_protocol[prot2]]
        # #                             for design in designs], metrics.significance_columns]
        # #         # prot1/2 pull out means from designs_df by using the protocol name
        # #         difference_s = \
        # #             designs_df.loc[prot1, metrics.significance_columns].sub(
        # #                 designs_df.loc[prot2, metrics.significance_columns])
        # #         pvalue_df[(prot1, prot2)] = metrics.df_permutation_test(select_df, difference_s, compare='mean',
        # #                                                                 group1_size=len(designs_by_protocol[prot1]))
        # #     pvalue_df = pvalue_df.T  # transpose significance pairs to indices and significance metrics to columns
        # #     designs_df = pd.concat([designs_df, pd.concat([pvalue_df], keys=['similarity']).swaplevel(0, 1)])
        # #
        # #     # Compute residue energy/sequence differences between each protocol
        # #     residue_energy_df = residues_df.loc[:, idx_slice[:, 'energy_delta']]
        # #
        # #     scaler = skl.preprocessing.StandardScaler()
        # #     res_pca = skl.decomposition.PCA(resources.config.default_pca_variance)
        # #     residue_energy_np = scaler.fit_transform(residue_energy_df.values)
        # #     residue_energy_pc = res_pca.fit_transform(residue_energy_np)
        # #     # Make principal components (PC) DataFrame
        # #     residue_energy_pc_df = pd.DataFrame(residue_energy_pc, index=residue_energy_df.index,
        # #                                         columns=[f'pc{idx}' for idx in range(1, len(res_pca.components_) +1)])
        # #     residue_energy_pc_df[putils.protocol] = protocol_s
        # #     # Next group the labels
        # #     # sequence_groups = seq_pc_df.groupby(putils.protocol)
        # #     residue_energy_groups = residue_energy_pc_df.groupby(putils.protocol)
        # #     # Measure statistics for each group
        # #     # All protocol means have pairwise distance measured to access similarity
        # #     # Gather protocol similarity/distance metrics
        # #     for stat in stats_metrics:
        # #         # grouped_pc_seq_df = getattr(sequence_groups, stat)()
        # #         grouped_pc_energy_df = getattr(residue_energy_groups, stat)()
        # #         # similarity_stat = getattr(pvalue_df, stat)(axis=1)  # protocol pair : stat Series
        # #         if stat == mean:
        # #             # for each measurement in residue_energy_pc_df, need to take the distance between it and the
        # #             # structure background mean (if structure background, is the mean is useful too?)
        # #             background_distance = \
        # #                 cdist(residue_energy_pc,
        # #                       grouped_pc_energy_df.loc[putils.structure_background, :].values[np.newaxis, :])
        # #             designs_df = \
        # #                 pd.concat([designs_df,
        # #                            pd.Series(background_distance.flatten(), index=residue_energy_pc_df.index,
        # #                                      name=f'energy_distance_from_{putils.structure_background}_mean')],axis=1)
        #
        # designs_df.sort_index(inplace=True, axis=1)

        return designs_df

    def analyze_pose_metrics_per_design(self, residues_df: pd.DataFrame, designs_df: pd.DataFrame = None,
                                        designs: Iterable[Pose] | Iterable[AnyStr] = None) -> pd.Series:
        """Perform Pose level analysis on every design produced from this Pose

        Args:
            residues_df: The typical per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
            designs_df: The typical per-design metric DataFrame where each index is the design id and the columns are
                design metrics
            designs: The subsequent designs to perform analysis on
        Returns:
            Series containing summary metrics for all designs
        """
        self.load_pose()

        return pose_s


class PoseJob(PoseProtocol):
    pass
