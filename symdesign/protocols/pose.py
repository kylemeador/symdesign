from __future__ import annotations

import abc
import logging
import os
import re
import shutil
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence, MutableMapping
from glob import glob
from itertools import combinations, repeat, count
from math import sqrt
from pathlib import Path
from subprocess import Popen, list2cmdline
from typing import Any, AnyStr

from cycler import cycler
import jax.numpy as jnp
from matplotlib import pyplot as plt
# from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
# import seaborn as sns
import sklearn as skl
from sqlalchemy import select, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, reconstructor
from sqlalchemy.orm.exc import DetachedInstanceError
import torch

from .utils import warn_missing_symmetry
from symdesign import flags, metrics, resources
from symdesign.resources import distribute, sql
from symdesign.sequence import MultipleSequenceAlignment, read_fasta_file, write_sequences
from symdesign.structure.base import StructureBase
from symdesign.structure.coordinates import superposition3d
from symdesign.structure.model import ContainsEntities, Entity, Structure, Pose, PoseSpecification
from symdesign.structure.sequence import sequence_difference, pssm_as_array, concatenate_profile, sequences_to_numeric
from symdesign.structure.utils import ClashError, DesignError, SymmetryError
import alphafold.data.pipeline as af_pipeline
from alphafold.common import residue_constants
from symdesign.utils import all_vs_all, condensed_to_square, InputError, large_color_array, start_log, path as putils, \
    rosetta, starttime, SymDesignException
from symdesign.utils.SymEntry import SymEntry, symmetry_factory, parse_symmetry_specification

# Globals
FeatureDict = MutableMapping[str, np.ndarray]
logger = logging.getLogger(__name__)
pose_logger = start_log(name='pose', handler_level=3, propagate=True)
idx_slice = pd.IndexSlice
cst_value = round(0.2 * rosetta.reference_average_residue_weight, 2)
mean, std = 'mean', 'std'
stats_metrics = [mean, std]
null_cmd = ['echo']
observed_types = ('evolution', 'fragment', 'design', 'interface')
missing_pose_transformation = "The design couldn't be transformed as it is missing the required " \
                              '"pose_transformation" attribute. Was this generated properly?'


def load_evolutionary_profile(api_db: resources.wrapapi.APIDatabase, model: ContainsEntities,
                              warn_metrics: bool = False) -> tuple[bool, bool]:
    """Add evolutionary profile information to the provided Entity

    Args:
        api_db: The database which store all information pertaining to evolutionary information
        model: The ContainsEntities instance to check for Entity instances to load evolutionary information
        warn_metrics: Whether to warn the user about missing files for metric collection

    Returns:
        A tuple of boolean values of length two indicating if, 1-evolutionary and 2-alignment information was added to
        the Entity instances
    """
    # Assume True given this function call and set False if not possible for one of the Entities
    measure_evolution = measure_alignment = True
    warn = False
    for entity in model.entities:
        if entity.evolutionary_profile:
            continue

        # if len(entity.uniprot_ids) > 1:
        #     raise SymDesignException(
        #         f"Can't set the profile for an {entity.__class__.__name__} with number of UniProtIDs "
        #         f"({len(entity.uniprot_ids)}) > 1. Please remove this or update the code")
        # for idx, uniprot_id in enumerate(entity.uniprot_ids):
        evolutionary_profile = {}
        for uniprot_id in entity.uniprot_ids:
            profile = api_db.hhblits_profiles.retrieve_data(name=uniprot_id)
            if not profile:
                null_entries = entity.create_null_entries(range(entity.number_of_residues))
                for entry, residue in zip(null_entries.values(), entity.residues):
                    entry['type'] = residue.type1

                evolutionary_profile.update(null_entries)
                # # Try and add... This would be better at the program level due to memory issues
                # entity.add_evolutionary_profile(out_dir=self.job.api_db.hhblits_profiles.location)
            else:
                if evolutionary_profile:
                    # Renumber the profile based on the current length
                    profile = {entry_number: entry
                               for entry_number, entry in enumerate(profile.values(), len(evolutionary_profile))}
                evolutionary_profile.update(profile)

        if not evolutionary_profile:
            measure_evolution = False
            warn = True
        else:
            logger.debug(f'Adding {entity.name}.evolutionary_profile')
            entity.evolutionary_profile = evolutionary_profile

        if entity.msa:
            continue

        # Fetch the multiple sequence alignment for further processing
        msas = []
        for uniprot_id in entity.uniprot_ids:
            msa = api_db.alignments.retrieve_data(name=uniprot_id)
            if not msa:
                measure_alignment = False
                warn = True
            else:
                logger.debug(f'Adding {entity.name}.msa')
                msas.append(msa)
                # Todo
                #  The alignment of concatenated evolutionary profiles is sensitive to the length of the internal
                #  gaps when there is an extend penalty
                #  modify_alignment_algorithm = True
                #  query_internal_extend_gap_score=0

        if msas:
            # Combine all
            max_alignment_size = max([msa_.length for msa_ in msas])
            msa, *other_msas = msas
            combined_alignment = msa.alignment
            # modify_alignment_algorithm = False
            msa_: MultipleSequenceAlignment
            for msa_ in other_msas:
                length_difference = max_alignment_size - msa_.length
                if length_difference:  # Not 0
                    msa_.pad_alignment(length_difference)
                combined_alignment += msa_.alignment
            # To create the full MultipleSequenceAlignment
            entity.msa = MultipleSequenceAlignment(combined_alignment)

    if warn_metrics and warn:
        if not measure_evolution and not measure_alignment:
            logger.info("Metrics relying on evolution aren't being collected as the required files weren't "
                        f'found. These include: {", ".join(metrics.all_evolutionary_metrics)}')
        elif not measure_alignment:
            logger.info('Metrics relying on a multiple sequence alignment including: '
                        f'{", ".join(metrics.multiple_sequence_alignment_dependent_metrics)}'
                        "are being calculated with the reference sequence as there was no MSA found")
        else:
            logger.info("Metrics relying on an evolutionary profile aren't being collected as there was no profile "
                        f'found. These include: {", ".join(metrics.profile_dependent_metrics)}')

    # if measure_evolution:
    model.evolutionary_profile = \
        concatenate_profile([entity.evolutionary_profile for entity in model.entities])
    # else:
    #     self.pose.evolutionary_profile = self.pose.create_null_profile()
    return measure_evolution, measure_alignment


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
    # pose_file: str | Path

    def __init__(self, root_directory: AnyStr = None, project: str = None, name: str = None, **kwargs):
        """Construct the instance

        Args:
            root_directory: The base of the directory tree that houses all project directories
            project: The name of the project where the files should be stored in the root_directory
            name: The name of the pose, which becomes the final dirname where all files are stored
        """
        self.info: dict = {}
        """Internal state info"""
        self._info: dict = {}
        """Internal state info at load time"""

        # try:
        directory = os.path.join(root_directory, project, name)
        # except TypeError:  # Can't pass None
        #     missing_args = ", ".join((str_ for str_, arg in (("root_directory", root_directory),
        #                                                     ("project", project), ("name", name))
        #                               if arg is None))
        #     raise TypeError(f'{PoseDirectory.__name__} is missing the required arguments {missing_args}')
        # if directory is not None:
        self.pose_directory = directory
        # PoseDirectory attributes. Set after finding correct path
        self.log_path: str | Path = os.path.join(self.pose_directory, f'{name}.log')
        self.designs_path: str | Path = os.path.join(self.pose_directory, 'designs')
        # /root/Projects/project_Poses/design/designs
        self.scripts_path: str | Path = os.path.join(self.pose_directory, 'scripts')
        # /root/Projects/project_Poses/design/scripts
        self.frags_path: str | Path = os.path.join(self.pose_directory, putils.frag_dir)
        # /root/Projects/project_Poses/design/matching_fragments
        self.flags: str | Path = os.path.join(self.scripts_path, 'flags')
        # /root/Projects/project_Poses/design/scripts/flags
        self.data_path: str | Path = os.path.join(self.pose_directory, putils.data)
        # /root/Projects/project_Poses/design/data
        self.scores_file: str | Path = os.path.join(self.data_path, f'{name}.sc')
        # /root/Projects/project_Poses/design/data/name.sc
        self.serialized_info: str | Path = os.path.join(self.data_path, 'info.pkl')
        # /root/Projects/project_Poses/design/data/info.pkl
        self.pose_path: str | Path = os.path.join(self.pose_directory, f'{name}.pdb')
        self.asu_path: str | Path = os.path.join(self.pose_directory, 'asu.pdb')
        # /root/Projects/project_Poses/design/asu.pdb
        # self.asu_path: str | Path = os.path.join(self.pose_directory, f'{self.name}_{putils.asu}')
        # # /root/Projects/project_Poses/design/design_name_asu.pdb
        self.assembly_path: str | Path = os.path.join(self.pose_directory, f'{name}_assembly.pdb')
        # /root/Projects/project_Poses/design/design_name_assembly.pdb
        self.refine_pdb: str | Path = os.path.join(self.data_path, os.path.basename(self.pose_path))
        # /root/Projects/project_Poses/design/data/design_name.pdb
        self.consensus_pdb: str | Path = f'{os.path.splitext(self.pose_path)[0]}_for_consensus.pdb'
        # /root/Projects/project_Poses/design/design_name_for_consensus.pdb
        # self.consensus_design_pdb: str | Path = os.path.join(self.designs_path, os.path.basename(self.consensus_pdb))
        # # /root/Projects/project_Poses/design/designs/design_name_for_consensus.pdb
        self.design_profile_file: str | Path = os.path.join(self.data_path, 'design.pssm')
        # /root/Projects/project_Poses/design/data/design.pssm
        self.evolutionary_profile_file: str | Path = os.path.join(self.data_path, 'evolutionary.pssm')
        # /root/Projects/project_Poses/design/data/evolutionary.pssm
        self.fragment_profile_file: str | Path = os.path.join(self.data_path, 'fragment.pssm')
        # /root/Projects/project_Poses/design/data/fragment.pssm
        # These next two files may be present from NanohedraV1 outputs
        # self.pose_file = os.path.join(self.pose_directory, putils.pose_file)
        # self.frag_file = os.path.join(self.frags_path, putils.frag_text_file)
        # These files are used as output from analysis protocols
        # self.designs_metrics_csv = os.path.join(self.job.all_scores, f'{self}_Trajectories.csv')
        self.designs_metrics_csv = os.path.join(self.data_path, f'designs.csv')
        self.residues_metrics_csv = os.path.join(self.data_path, f'residues.csv')
        self.designed_sequences_file = os.path.join(self.designs_path, f'sequences.fasta')
        self.current_script = None

        if self.job.output_directory:
            self.output_path = self.job.output_directory
            self.output_modifier = f'{project}-{name}'
            """string with contents '{self.project}-{self.name}'"""
            self.output_pose_path = os.path.join(self.output_path, f'{self.output_modifier}.pdb')
            """/output_path/{self.output_modifier}.pdb"""
            # self.output_asu_path: str | Path = os.path.join(self.output_path, f'{self.output_modifier}_{putils.asu}')
            # """/output_path/{self.output_modifier}_asu.pdb"""
            self.output_assembly_path: str | Path = \
                os.path.join(self.output_path, f'{self.output_modifier}_assembly.pdb')
            """/output_path/{self.output_modifier}_{putils.assembly}.pdb """
            # pose_directory = self.job.output_directory  # /output_directory <- self.pose_directory/design.pdb

        super().__init__(**kwargs)

    @property
    @abc.abstractmethod
    def job(self):
        """"""

    @property
    @abc.abstractmethod
    def refined(self):
        """"""

    # These next two file locations are used to dynamically update whether preprocessing should occur for designs
    @property
    def refined_pdb(self) -> str:
        """Access the file which holds refined coordinates. These are not necessaryily refined, but if a file exists in
        the designs_path with the pose name, it was generated from either Rosetta energy function minimization or from
        structure prediction, which is assumed to be refined. If none, exists, use the self.pose_path
        """
        try:
            return self._refined_pdb
        except AttributeError:
            if self.refined:
                self._refined_pdb = self.pose_path
            else:
                # self.refined_pdb = None  # /root/Projects/project_Poses/design/design_name_refined.pdb
                designs_refine_file = \
                    os.path.join(self.designs_path,
                                 f'{os.path.basename(os.path.splitext(self.pose_path)[0])}.pdb')
                                 # f'{os.path.basename(os.path.splitext(self.pose_path)[0])}_refined.pdb')
                if os.path.exists(designs_refine_file):
                    self._refined_pdb = designs_refine_file
                else:
                    self._refined_pdb = self.pose_path
            return self._refined_pdb

    @property
    def scouted_pdb(self) -> str:
        try:
            return self._scouted_pdb
        except AttributeError:
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

    def get_pose_file(self) -> AnyStr:
        """Retrieve the pose file name from PoseJob"""
        # glob_target = os.path.join(self.pose_directory, f'{self.name}*.pdb')
        glob_target = self.pose_path
        source = sorted(glob(glob_target))
        if len(source) > 1:
            raise InputError(
                f'Found {len(source)} files matching the path "{glob_target}". No more than one expected')
        else:
            try:
                file = source[0]
            except IndexError:  # glob found no files
                self.log.debug(f"Couldn't find any structural files matching the path '{glob_target}'")
                file = None
                raise FileNotFoundError

        return file

    def get_design_files(self, design_type: str = '') -> list[AnyStr]:
        """Return the paths of all design files in a PoseJob

        Args:
            design_type: Specify if a particular type of design should be selected by a "type" string

        Returns:
            The sorted design files found in the designs directory with an absolute path
        """
        return sorted(glob(os.path.join(self.designs_path, f'*{design_type}*.pdb*')))


# This MRO requires __init__ in PoseMetadata to pass PoseDirectory kwargs
# class PoseData(sql.PoseMetadata, PoseDirectory):
class PoseData(PoseDirectory, sql.PoseMetadata):
    _current_designs: list | list[sql.DesignData]
    """Hold DesignData that were specified/generated in the scope of this job"""
    _design_selector: PoseSpecification
    _directives: list[dict[int, str]]
    _sym_entry: SymEntry
    # _entity_names: list[str]
    # _pose_transformation: list[types.TransformationMapping]
    _job_kwargs: dict[str, Any]
    _source: AnyStr
    # entity_data: list[EntityData]  # DB
    # name: str  # DB
    # project: str  # DB
    # pose_identifier: str  # DB
    # """The identifier which is created by a concatenation of the project, os.sep, and name"""
    # id: int  # DB
    # """The database row id for the 'pose_data' table"""
    initial_pose: Pose | None
    """Used if the structure has never been initialized previously"""
    measure_evolution: bool | None
    measure_alignment: bool | None
    pose: Pose | None
    """Contains the Pose object"""
    protocol: str | None
    """The name of the currently utilized protocol for file naming and metric results"""
    job: 'resources.job.JobResources' = None
    # specific_designs_file_paths: list[AnyStr] = []
    # """Contains the various file paths for each design of interest according to self.specific_designs"""

    # START classmethod where PoseData hasn't been initialized from sqlalchemy
    @classmethod
    def from_path(cls, path: str, project: str = None, **kwargs):
        """Load the PoseJob from a path with file types or a directory

        Args:
            path: The path where the PoseJob instance should load structural data
            project: The project where the file should be included

        Returns:
            The PoseJob instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"The specified {cls.__name__} path '{path}' wasn't found")
        filename, extension = os.path.splitext(path)
        # if 'pdb' in extension or 'cif' in extension:  # Initialize from input file
        if extension != '':  # Initialize from input file
            project_path, name = os.path.split(filename)
            if project is None:
                remainder, project = os.path.split(project_path)
                if project == '':
                    raise InputError(
                        f"Couldn't get the project from the path '{path}'. Please provide "
                        f"project name with {flags.format_args(flags.project_name_args)}")

            return cls(name=name, project=project, source_path=path, **kwargs)
        elif os.path.isdir(path):
            # Same as from_directory. This is an existing pose_identifier that hasn't been initialized
            try:
                name, project, *_ = reversed(path.split(os.sep))
            except ValueError:  # Only got 1 value during unpacking... This isn't a "pose_directory" identifier
                raise InputError(
                    f"Couldn't coerce {path} to a {cls.__name__}. The directory must contain the "
                    f'"project{os.sep}pose_name" string')
            return cls(name=name, project=project, **kwargs)
        else:
            raise InputError(
                f"{cls.__name__} couldn't load the specified source path '{path}'")

    @classmethod
    def from_file(cls, file: str, project: str = None, **kwargs):
        """Load the PoseJob from a structural file including .pdb/.cif file types

        Args:
            file: The file where the PoseJob instance should load structure files
            project: The project where the file should be included

        Returns:
            The PoseJob instance
        """
        # file = os.path.abspath(file)
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"The specified {cls.__name__} structure source file '{file}' wasn't found!")
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
            raise InputError(
                f"{cls.__name__} couldn't load the specified source file '{file}'")

        # if self.job.nanohedra_outputV1:
        #     # path/to/design_symmetry/building_blocks/degen/rot/tx
        #     # path_components[-4] are the oligomeric (building_blocks) names
        #     name = '-'.join(path_components[-4:])
        #     project = path_components[-5]  # if root is None else root
        #     file = os.path.join(file, putils.asu)

        if project is None:
            remainder, project = os.path.split(project_path)
            if project == '':
                raise InputError(
                    f"Couldn't get the project from the path '{file}'. Please provide "
                    f"project name with {flags.format_args(flags.project_name_args)}")

        return cls(name=name, project=project, source_path=file, **kwargs)

    @classmethod
    def from_directory(cls, source_path: str, **kwargs):
        """Assumes the PoseJob is constructed from the pose_name (project/pose_name) and job.projects

        Args:
            source_path: The path to the directory where PoseJob information is stored

        Returns:
            The PoseJob instance
        """
        try:
            name, project, *_ = reversed(source_path.split(os.sep))
        except ValueError:  # Only got 1 value during unpacking... This isn't a "pose_directory" identifier
            raise InputError(
                f"Couldn't coerce {source_path} to a {cls.__name__}. The directory must contain the "
                f'"project{os.sep}pose_name" string')

        return cls(name=name, project=project, **kwargs)

    @classmethod
    def from_name(cls, name: str = None, project: str = None, **kwargs):
        """Load the PoseJob from the name and project

        Args:
            name: The name to identify this PoseJob
            project: The project where the file should be included

        Returns:
            The PoseJob instance
        """
        return cls(name=name, project=project, **kwargs)

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
            raise InputError(
                f"Couldn't coerce {pose_identifier} to 'project' {os.sep} 'name'. Please ensure the "
                f'pose_identifier is passed with the "project{os.sep}name" string')
        return cls(name=name, project=project, **kwargs)

    @classmethod
    def from_pose(cls, pose: Pose, project: str = None, **kwargs):
        """Load the PoseJob from an existing Pose

        Args:
            pose: The Pose to initialize the PoseJob with
            project: The project where the file should be included

        Returns:
            The PoseJob instance
        """
        return cls(name=pose.name, project=project, pose=pose, **kwargs)
    # END classmethods where the PoseData hasn't been initialized from sqlalchemy

    @reconstructor
    def __init_from_db__(self):
        """Initialize PoseData after the instance is "initialized", i.e. loaded from the database"""
        # Design attributes
        # self.current_designs = []
        self.measure_evolution = self.measure_alignment = None
        self.pose = self.initial_pose = self.protocol = None
        # Get the main program options
        self.job = resources.job.job_resources_factory.get()
        # Symmetry attributes
        # If a new sym_entry is provided it wouldn't be saved to the state but could be attempted to be used
        if self.job.sym_entry is not None:
            self.sym_entry = self.job.sym_entry

        if self.job.design_selector:
            self.design_selector = self.job.design_selector

        # self.specific_designs_file_paths = []
        # """Contains the various file paths for each design of interest according to self.specific_designs"""

        # These arguments are for PoseDirectory
        # directory = os.path.join(self.job.projects, self.project, self.name)
        super().__init__(root_directory=self.job.projects, project=self.project, name=self.name)  # Todo **kwargs)

        putils.make_path(self.pose_directory)  # , condition=self.job.construct_pose)

        # Initialize the logger for the PoseJob
        if self.job.debug:
            log_path = None
            handler = level = 1  # Defaults to stdout, debug is level 1
            # Don't propagate, emit ourselves
            propagate = no_log_name = False
        elif self.log_path:
            log_path = self.log_path
            handler = level = 2  # To a file
            propagate = no_log_name = True
        else:  # Log to the __main__ file logger
            log_path = None
            handler = level = 2  # To a file
            propagate = no_log_name = False

        if self.job.skip_logging:  # Set up null_logger
            self.log = logging.getLogger('null')
        else:
            self.log = start_log(name=f'pose.{self.project}.{self.name}', handler=handler, level=level,
                                 location=log_path, no_log_name=no_log_name, propagate=propagate)
            # propagate=True allows self.log to pass messages to 'pose' and 'project' logger

        # Configure standard pose loading mechanism with self.source_path
        if self.source_path is None:
            try:
                self.source_path = self.get_pose_file()
            except FileNotFoundError:
                if os.path.exists(self.asu_path):  # Standard mechanism of loading the pose
                    self.source_path = self.asu_path
                else:
                    self.source_path = None

    def __init__(self, name: str = None, project: str = None, source_path: AnyStr = None, pose: Pose = None,
                 protocol: str = None, pose_source: sql.DesignData = None,
                 # pose_transformation: Sequence[types.TransformationMapping] = None,
                 # entity_metadata: list[sql.EntityData] = None,
                 # entity_names: Sequence[str] = None,
                 # specific_designs: Sequence[str] = None, directives: list[dict[int, str]] = None,
                 **kwargs):
        """Construct the instance

        Args:
            name: The identifier
            project: The project which this work belongs too
            source_path: If a path exists, where is structure files information stored?
            protocol: If a protocol was used to generate this Pose, which protocol?
            pose_source: If this is a descendant of another Design, which one?
        """
        # PoseJob attributes
        self.name = name
        self.project = project
        self.source_path = source_path

        # Most __init__ code is called in __init_from_db__() according to sqlalchemy needs and DRY principles
        self.__init_from_db__()
        if pose is not None:  # It should be saved
            self.initial_pose = pose
            self.initial_pose.write(out_path=self.pose_path)
            self.source_path = self.pose_path

        # Save job variables to the state during initialization
        sym_entry = kwargs.get('sym_entry')
        if sym_entry:
            self.sym_entry = sym_entry

        if self.sym_entry:
            self.symmetry_dimension = self.sym_entry.dimension
            """The dimension of the SymEntry"""
            self.symmetry = self.sym_entry.resulting_symmetry
            """The resulting symmetry of the SymEntry"""
            self.sym_entry_specification = self.sym_entry.specification
            """The specification string of the SymEntry"""
            self.sym_entry_number = self.sym_entry.number
            """The SymEntry entry number"""

        # Set up original DesignData entry for the pose baseline
        if pose_source is None:
            pose_source = sql.DesignData(name=name, pose=self, design_parent=None)  # , structure_path=source_path)
        else:
            pose_source = sql.DesignData(name=name, pose=self, design_parent=pose_source,
                                         structure_path=pose_source.structure_path)
        if protocol is not None:
            pose_source.protocols.append(sql.DesignProtocol(protocol=protocol, job_id=self.job.id))

    def clear_state(self):
        """Set the current instance structural and sequence attributes to None to remove excess memory"""
        self.measure_evolution = self.measure_alignment = self.pose = self.initial_pose = None

    def use_specific_designs(self, designs: Sequence[str] = None, directives: list[dict[int, str]] = None,
                             **kwargs):
        """Set up the instance with the names and instructions to perform further sequence design

        Args:
            designs: The names of designs which to include in modules that support PoseJob designs
            directives: Instructions to guide further sampling of designs
        """
        if designs:
            self.current_designs = designs

        if directives:
            self.directives = directives

    def load_initial_pose(self):
        """Loads the Pose at the source_path attribute"""
        if self.pose:
            self.initial_pose = self.pose
        elif self.initial_pose is None:
            if self.source_path is None:
                raise InputError(
                    f"Couldn't {self.load_initial_pose.__name__}() for {self.name} as there isn't a file specified")
            self.initial_pose = Pose.from_file(self.source_path, log=self.log)
            # # Todo ensure that the chain names are renamed if they are imported a certain way
            # if len(set([entity.chain_id for entity in self.initial_pose.entities])) \
            #         != self.initial_pose.number_of_entities:
            #     rename = True
            #     self.initial_pose.rename_chains()
            # # else:
            # #     rename = False

    @property
    def new_pose_identifier(self) -> str:
        """Return the pose_identifier when the PoseData isn't part of the database"""
        return f'{self.project}{os.sep}{self.name}'

    @property
    def directives(self) -> list[dict[int, str]]:
        """The design directives given to each design in current_designs to guide further sampling"""
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
    def current_designs(self) -> list[sql.DesignData] | list:
        """DesignData instances which were generated in the scope of this job and serve as a pool for additional work"""
        try:
            return self._current_designs
        except AttributeError:
            self._current_designs = []
            return self._current_designs

    @current_designs.setter
    def current_designs(self, designs: Iterable[sql.DesignData | str | int]):
        """Configure the current_designs"""
        if designs:
            _current_designs = []
            for potential_design in designs:
                if isinstance(potential_design, sql.DesignData):
                    _current_designs.append(potential_design)
                elif isinstance(potential_design, str):
                    for design in self.designs:
                        if design.name == potential_design:
                            _current_designs.append(design)
                            break
                    else:
                        raise DesignError(
                            f"Couldn't set {self}.current_designs as there was no {sql.DesignData.__name__} "
                            f"matching the name '{potential_design}'")
                elif isinstance(potential_design, int):
                    for design in self.designs:
                        if design.id == potential_design:
                            _current_designs.append(design)
                            break
                    else:
                        raise DesignError(
                            f"Couldn't set {self}.current_designs as there was no {sql.DesignData.__name__} "
                            f"matching the id '{potential_design}'")
                else:
                    raise ValueError(
                        f"Couldn't set {self}.current_designs with a 'design'={potential_design} of type "
                        f"{type(potential_design).__name__}")

            self._current_designs = _current_designs
            self.log.debug(f'Added {len(self._current_designs)} designs to self.current_designs')

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
                self._sym_entry = symmetry_factory.get(
                    self.sym_entry_number, parse_symmetry_specification(self.sym_entry_specification))
            except AttributeError:  # self.sym_entry_specification is None?
                return None  # No symmetry specified
            return self._sym_entry

    @sym_entry.setter
    def sym_entry(self, sym_entry: SymEntry):
        if isinstance(sym_entry, SymEntry):
            self._sym_entry = sym_entry
        else:
            raise InputError(
                f"Couldn't set the 'sym_entry' attribute with a type {type(sym_entry).__name__}. "
                f"Expected a {SymEntry.__name__} instance")

    def is_symmetric(self) -> bool:
        """Is the PoseJob symmetric?"""
        return self.sym_entry is not None

    @property
    def job_kwargs(self) -> dict[str, Any]:
        """Returns the keyword args that initialize the Pose given program input and PoseJob state"""
        try:
            return self._job_kwargs
        except AttributeError:
            self._job_kwargs = dict(sym_entry=self.sym_entry, log=self.log, fragment_db=self.job.fragment_db,
                                    pose_format=self.job.pose_format)
            return self._job_kwargs

    @property
    def pose_kwargs(self) -> dict[str, Any]:
        """Returns the keyword args that initialize the Pose given program input and PoseJob state"""
        entity_data = self.entity_data
        entity_info = {}
        for data in entity_data:
            entity_info.update(data.entity_info)

        # If this fails with a sqlalchemy.orm.exc.DetachedInstanceError. The PoseJob isn't completely set up
        transformations = [data.transformation for data in entity_data]
        return dict(entity_info=entity_info, transformations=transformations,
                    **self.job_kwargs)

    # @property
    # def pose_transformation(self) -> list[types.TransformationMapping]:
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
    # def pose_transformation(self, transform: Sequence[types.TransformationMapping]):
    #     if all(isinstance(operation_set, dict) for operation_set in transform):
    #         self._pose_transformation = self.info['pose_transformation'] = list(transform)
    #     else:
    #         try:
    #             raise ValueError(f'The attribute pose_transformation must be a Sequence'
    #                              f'[types.TransformationMapping], not {type(transform[0]).__name__}')
    #         except TypeError:  # Not a Sequence
    #             raise TypeError(f'The attribute pose_transformation must be a Sequence of '
    #                             f'[{types.TransformationMapping.__name__}], not {type(transform).__name__}')

    @property
    def design_selector(self) -> PoseSpecification:
        """Provide the design_selector parameters for the design in question

        Returns:
            A mapping of the selection criteria for StructureBase objects in the Pose
        """
        try:
            return self._design_selector
        except AttributeError:  # Get from the pose state
            self._design_selector = {}
            return self._design_selector

    @design_selector.setter
    def design_selector(self, design_selector: dict):
        if isinstance(design_selector, dict):
            self._design_selector = design_selector
        else:
            raise ValueError(
                f'The attribute design_selector must be a dict, not {type(design_selector).__name__}')

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
    # def number_fragments_interface(self) -> int:
    #     return len(self.fragment_observations) if self.fragment_observations else 0

    # Both pre_* properties are really implemented to take advantage of .setter
    @property
    def refined(self) -> bool:
        """Provide the state attribute regarding the source files status as "previously refined"

        Returns:
            Whether refinement has occurred
        """
        # return self._pre_refine
        return all(data.meta.refined for data in self.entity_data)
        # try:
        #     return self._pre_refine
        # except AttributeError:  # Get from the pose state
        #     self._pre_refine = self.info.get('refined', True)
        #     return self._pre_refine

    @property
    def loop_modeled(self) -> bool:
        """Provide the state attribute regarding the source files status as "previously loop modeled"

        Returns:
            Whether loop modeling has occurred
        """
        # return self._pre_loop_model
        return all(data.meta.loop_modeled for data in self.entity_data)
        # try:
        #     return self._pre_loop_model
        # except AttributeError:  # Get from the pose state
        #     self._pre_loop_model = self.info.get('loop_modeled', True)
        #     return self._pre_loop_model

    def get_designs_without_structure(self) -> list[sql.DesignData]:
        """For each design, access whether there is a structure that exists for it. If not, return the design

        Returns:
            Each instance of the DesignData that is missing a structure
        """
        missing = []
        for design in self.designs[1:]:  # Slice only real designs, not the pose_source
            if design.structure_path and os.path.exists(design.structure_path):
                continue
            else:
                missing.append(design)

        return missing

    def transform_entities_to_pose(self, **kwargs) -> list[Entity]:
        """Take the set of entities involved in a pose composition and transform them from a standard reference frame to
        the Pose reference frame using the pose_transformation attribute

        Keyword Args:
            refined: bool = True - Whether to use refined models from the StructureDatabase
            oriented: bool = False - Whether to use oriented models from the StructureDatabase
        """
        entities = self.get_entities(**kwargs)
        if self.transformations:
            entities = [entity.get_transformed_copy(**transformation)
                        for entity, transformation in zip(entities, self.transformations)]
            self.log.debug('Entities were transformed to the found docking parameters')
        else:
            # Todo change below to handle asymmetric cases...
            self.log.error(missing_pose_transformation)

        return entities

    def transform_structures_to_pose(
        self, structures: Iterable[StructureBase], **kwargs
    ) -> list[StructureBase]:
        """Take a set of ContainsResidues instances and transform them from a standard reference frame to the Pose reference
        frame using the pose_transformation attribute

        Args:
            structures: The ContainsResidues instances to transform
        Returns:
            The transformed ContainsResidues instances if a transformation was possible
        """
        if self.transformations:
            # Todo
            #  Assumes a 1:1 correspondence between structures and transforms (component group numbers) CHANGE
            structures = [structure.get_transformed_copy(**transformation)
                          for structure, transformation in zip(structures, self.transformations)]
            self.log.debug('Structures were transformed to the found docking parameters')
        else:
            self.log.error(missing_pose_transformation)
            structures = list(structures)

        return structures

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

        if refined:
            source_idx = 0
        elif oriented:
            source_idx = 1
        else:
            source_idx = 2
            self.log.info(f'Falling back on Entity instances present in the {self.__class__.__name__} structure_source')

        entities = []
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
                    raise SymDesignException(
                        f"{self.get_entities.__name__}: Couldn't locate the required files")
                source_datastore = getattr(self.job.structure_db, source, None)
                # Todo this course of action isn't set up anymore. It should be depreciated...
                if source_datastore is None:  # Try to get file from the PoseDirectory
                    raise SymDesignException(
                        f"Couldn't locate the specified Entity '{name}' from any Database sources")

                model = source_datastore.retrieve_data(name=name)
                #  Error where the EntityID loaded from 2gtr_1.pdb was 2gtr_1_1
                #  Below might help resolve this issue
                # model_file = source_datastore.retrieve_file(name=name)
                # if model_file:
                #     model = Entity.from_file(model_file)
                # else:
                #     model = None
                if isinstance(model, Structure):
                    self.log.info(f'Found Entity at {source} DataStore and loaded into job')
                else:
                    self.log.warning(f"Couldn't locate the Entity {name} at the Database source "
                                     f"'{source_datastore.location}'")

            entities.extend([entity for entity in model.entities])
        # Todo is this useful or is the ProteinMetadata already set elsewhere?
        # if source_idx == 0:
        #     self.refined = True
        # if source_idx == 0:
        #     self.loop_modeled = True

        if len(entities) != len(self.entity_data):
            raise SymDesignException(
                f'Expected {len(entities)} entities, but found {len(self.entity_data)}')

        return entities

    def report_exception(self, context: str = None) -> None:
        """Reports error traceback to the log file.

        Args:
            context: A string describing the function or situation where the error occurred.
        """
        if context is None:
            msg = ''
        else:
            msg = f'{context} resulted in the exception:\n\n'
        self.log.exception(msg)

    def format_see_log_msg(self) -> str:
        """Issue a standard informational message indicating the location of further error information"""
        return f"See the log '{self.log_path}' for further information"

    def load_pose(self, file: str = None, entities: list[Structure] = None) -> None:
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
        else:
            if entities is not None:
                self.structure_source = 'Protocol derived'
                # Use the entities as provided
            elif self.source_path is None or not os.path.exists(self.source_path):
                # In case a design was initialized without a file
                self.log.info(f"No '.source_path' found. Fetching structure_source from "
                              f'{type(self.job.structure_db).__name__} and transforming to Pose')
                # Minimize I/O with transform...
                # Must add the instance to a session to load any self.transformations present
                with self.job.db.session(expire_on_commit=False) as session:
                    session.add(self)
                    entities = self.transform_entities_to_pose()
                # Todo should use the ProteinMetadata version if the constituent Entity coordinates aren't modified by
                #  some small amount as this will ensure that files are refined and that loops are included...
                # Collect the EntityTransform for these regardless of symmetry since the file will be coming from the
                # oriented/origin position...
                self.structure_source = 'Database'

            # Initialize the Pose using provided PDB numbering so that design_selectors are respected
            if entities:
                # Chains should be renamed if the chain_id's are the same
                if len({entity.chain_id for entity in entities}) != len(entities):
                    rename = True
                else:
                    rename = False
                self.pose = Pose.from_entities(entities, name=self.name, rename_chains=rename, **self.job_kwargs)
            elif self.initial_pose:
                # This is a fresh Model that was already loaded so reuse
                # Careful, if processing has occurred to the initial_pose, then this may be wrong!
                self.pose = Pose.from_structure(
                    self.initial_pose, name=self.name, entity_info=self.initial_pose.entity_info, **self.job_kwargs)
                # Use entity_info from already parsed
                self.structure_source = self.source_path
            else:
                self.structure_source = file if file else self.source_path
                self.pose = Pose.from_file(self.structure_source, name=self.name, **self.pose_kwargs)

            if self.design_selector:
                self.pose.apply_design_selector(**self.design_selector)

            try:
                self.pose.is_clash(measure=self.job.design.clash_criteria,
                                   distance=self.job.design.clash_distance,
                                   warn=not self.job.design.ignore_clashes)
            except ClashError:  # as error:
                if self.job.design.ignore_pose_clashes:
                    self.report_exception(context='Clash checking')
                    self.log.warning(f"The Pose from '{self.structure_source}' contains clashes. "
                                     f"{self.format_see_log_msg()}")
                else:
                    # Todo get the message from error and raise a ClashError(
                    #      f'{message}. If you would like to proceed regardless, re-submit the job with '
                    #      f'{flags.format_args(flags.ignore_pose_clashes_args)}'
                    raise
            if self.pose.is_symmetric():
                if self.pose.symmetric_assembly_is_clash(measure=self.job.design.clash_criteria,
                                                         distance=self.job.design.clash_distance):
                    if self.job.design.ignore_symmetric_clashes:
                        # self.format_error_for_log()
                        self.log.warning(
                            f"The Pose symmetric assembly from '{self.structure_source}' contains clashes.")
                        #     f"{self.format_see_log_msg()}")
                    else:
                        raise ClashError(
                            "The symmetric assembly contains clashes and won't be considered. If you "
                            'would like to proceed regardless, re-submit the job with '
                            f'{flags.format_args(flags.ignore_symmetric_clashes_args)}')

                # If there is an empty list for the transformations, save the transformations identified from the Pose
                try:
                    any_database_transformations = any(self.transformations)
                except DetachedInstanceError:
                    any_database_transformations = False

                if not any_database_transformations:
                    # Add the transformation data to the database
                    for data, transformation in zip(self.entity_data, self.pose.entity_transformations):
                        # Make an empty EntityTransform
                        data.transform = sql.EntityTransform()
                        data.transform.transformation = transformation
                    # Ensure this information is persistent
                    self._update_db()

        # Save the Pose asu
        if not os.path.exists(self.pose_path) or self.job.overwrite or self.job.load_to_db:
            # Set the pose_path as the source_path
            self.source_path = out_path = self.pose_path
            # # Propagate to the PoseJob parent DesignData
            # self.source_path = out_path = self.pose_source.structure_path = self.pose_path
            # Ensure this information is persistent
            self._update_db()
        else:  # Explicitly set None and write anything else requested by input options
            out_path = None
        self.output_pose(out_path=out_path)

    def _update_db(self):
        """Ensure information added to the PoseJob is persistent in the database"""
        with self.job.db.session(expire_on_commit=False) as session:
            session.add(self)
            session.commit()

    def output_pose(self, out_path: AnyStr = 'POSE', force: bool = False):
        """Save a new Structure from multiple Chain or Entity objects including the Pose symmetry

        Args:
            out_path: The path to save the self.pose.
                All other outputs are done to the self.pose_directory or self.output_path
            force: Whether to force writing of the pose
        Returns:
            None
        """
        # if self.job.pose_format:
        #     self.pose.pose_numbering()

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
                    raise DesignError(
                        f"Your requested fusion of chain '{fusion_nterm}' with chain '{fusion_cterm}' didn't work")
                    # self.log.critical('Your requested fusion of chain %s with chain %s didn\'t work!' %
                    #                   (fusion_nterm, fusion_cterm))
                else:  # Won't be accessed unless entity_new_chain_idx is set
                    self.pose.entities[entity_new_chain_idx].chain_id = fusion_nterm
            # except AttributeError:
            #     raise ValueError('One or both of the chain IDs %s were not found in the input model. Possible chain'
            #                      ' ID\'s are %s' % ((fusion_nterm, fusion_cterm), ','.join(new_asu.chain_ids)))

        if self.pose.is_symmetric():
            if self.job.output_assembly:
                if self.job.output_to_directory:
                    assembly_path = self.output_assembly_path
                else:
                    assembly_path = self.assembly_path
                if not os.path.exists(assembly_path) or self.job.overwrite:
                    self.pose.write(assembly=True, out_path=assembly_path,
                                    increment_chains=self.job.increment_chains,
                                    surrounding_uc=self.job.output_surrounding_uc)
                    self.log.info(f"Symmetric assembly written to: '{assembly_path}'")
            if self.job.output_oligomers:  # Write out Entity.assembly instances to the PoseJob
                for idx, entity in enumerate(self.pose.entities):
                    if self.job.output_to_directory:
                        oligomer_path = os.path.join(self.output_path, f'{entity.name}_oligomer.pdb')
                    else:
                        oligomer_path = os.path.join(self.pose_directory, f'{entity.name}_oligomer.pdb')
                    if not os.path.exists(oligomer_path) or self.job.overwrite:
                        entity.write(assembly=True, out_path=oligomer_path)
                        self.log.info(f"Entity {entity.name} oligomer written to: '{oligomer_path}'")

        if self.job.output_entities:  # Write out Entity instances to the PoseJob
            for idx, entity in enumerate(self.pose.entities):
                if self.job.output_to_directory:
                    entity_path = os.path.join(self.output_path, f'{entity.name}_entity_.pdb')
                else:
                    entity_path = os.path.join(self.pose_directory, f'{entity.name}_entity.pdb')
                if not os.path.exists(entity_path) or self.job.overwrite:
                    entity.write(out_path=entity_path)
                    self.log.info(f"Entity {entity.name} written to: '{entity_path}'")

        if self.job.output_fragments:
            # if not self.pose.fragment_pairs:
            #     self.pose.generate_interface_fragments()
            putils.make_path(self.frags_path)
            if self.job.output_trajectory:
                # Write all fragments as one trajectory
                if self.sym_entry.unit_cell:
                    self.log.warning('No unit cell dimensions applicable to the Fragment trajectory file')

            self.pose.write_fragment_pairs(out_path=self.frags_path, multimodel=self.job.output_trajectory)

        if self.job.output_interface:
            interface_structure = self.pose.get_interface()
            if self.job.output_to_directory:
                interface_path = os.path.join(self.output_path, f'{self.name}_interface.pdb')
            else:
                interface_path = os.path.join(self.pose_directory, f'{self.name}_interface.pdb')
            interface_structure.write(out_path=interface_path)

        if out_path == 'POSE':
            out_path = self.pose_path

        if out_path:
            if not os.path.exists(out_path) or self.job.overwrite or force:
                self.pose.write(out_path=out_path)
                self.log.info(f"Wrote Pose file to: '{out_path}'")

        if self.job.output_to_directory:
            if not os.path.exists(self.output_pose_path) or self.job.overwrite or force:
                out_path = self.output_pose_path
                self.pose.write(out_path=out_path)
                self.log.info(f"Wrote Pose file to: '{out_path}'")

    def set_up_evolutionary_profile(self, **kwargs):
        """Add evolutionary profile information for each Entity to the Pose

        Keyword Args:
            warn_metrics: Whether to warn the user about missing files for metric collection
        """
        if self.job.use_evolution:
            if self.measure_evolution is None and self.measure_alignment is None:
                self.measure_evolution, self.measure_alignment = \
                    load_evolutionary_profile(self.job.api_db, self.pose, **kwargs)
            # else:  # Already set these
            #     return

    def generate_fragments(self, interface: bool = False, oligomeric_interfaces: bool = False, entities: bool = False):
        """For the design info given by a PoseJob source, initialize the Pose then generate interfacial fragment
        information between Entities. Aware of symmetry and design_selectors in fragment generation file

        Args:
            interface: Whether to perform fragment generation on the interface
            oligomeric_interfaces: Whether to perform fragment generation on the oligomeric interface
            entities: Whether to perform fragment generation on each Entity
        """
        if interface:
            self.pose.generate_interface_fragments(oligomeric_interfaces=oligomeric_interfaces,
                                                   distance=self.job.interface_distance)
        if entities:
            self.pose.generate_fragments(oligomeric_interfaces=oligomeric_interfaces,
                                         distance=self.job.interface_distance)
        if self.job.output_fragments:
            self.output_pose()

        # self.info['fragment_source'] = self.job.fragment_db.source

    @property
    def _key(self) -> str:
        return self.pose_identifier

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self._key == other._key
        raise NotImplementedError(f"Can't compare {self.__class__} instance to {other.__class__} instance")

    def __hash__(self) -> int:
        return hash(self._key)

    def __str__(self) -> str:
        return self.pose_identifier

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.pose_identifier})'


class PoseProtocol(PoseData):

    def identify_interface(self):
        """Find the interface(s) between each Entity in the Pose. Handles symmetric clash testing, writing the assembly
        """
        self.load_pose()
        # Measure the interface by_distance if the pose is the result of known docking inputs and
        # the associated sequence is non-sense
        self.pose.find_and_split_interface(by_distance=(self.protocol == putils.nanohedra
                                                        or self.protocol == putils.fragment_docking),
                                           distance=self.job.interface_distance)
        # Todo
        #                                    oligomeric_interfaces=self.job.oligomeric_interfaces)

    def orient(self, to_pose_directory: bool = True):
        """Orient the Pose with the prescribed symmetry at the origin and symmetry axes in canonical orientations
        job.symmetry is used to specify the orientation

        Args:
            to_pose_directory: Whether to write the file to the pose_directory or to another source
        """
        if not self.initial_pose:
            self.load_initial_pose()

        if self.symmetry:
            # This may raise an error if symmetry is malformed
            self.initial_pose.orient(symmetry=self.symmetry)

            if to_pose_directory:
                out_path = self.assembly_path
            else:
                putils.make_path(self.job.orient_dir)
                out_path = os.path.join(self.job.orient_dir, f'{self.initial_pose.name}.pdb')

            orient_file = self.initial_pose.write(out_path=out_path)
            self.log.info(f'The oriented file was saved to {orient_file}')

            # Now that symmetry is set, just set the pose attribute
            self.pose = self.initial_pose
            self.output_pose()
        else:
            raise SymmetryError(
                warn_missing_symmetry % self.orient.__name__)

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

        variables = rosetta.variables \
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
        for number, residues in self.pose.interface_residues_by_interface.items():
            interface_residue_ids[f'interface{number}'] = \
                ','.join(f'{residue.number}{residue.chain_id}' for residue in residues)
        # self.info['interface_residue_ids'] = self.interface_residue_ids
        variables.extend([(interface, residues) if residues else (interface, out_of_bounds_residue)
                          for interface, residues in interface_residue_ids.items()])

        # Assign any additional designable residues
        if self.pose.required_residues:
            variables.extend(
                [('required_residues', ','.join(f'{res.number}{res.chain_id}' for res in self.pose.required_residues))])
        else:  # Get an out-of-bounds index
            variables.extend([('required_residues', out_of_bounds_residue)])

        # Allocate any "core" residues based on central fragment information
        residues = self.pose.residues
        fragment_residues = [residues[index] for index in self.pose.interface_fragment_residue_indices]
        if fragment_residues:
            variables.extend([('fragment_residues',
                               ','.join([f'{res.number}{res.chain_id}' for res in fragment_residues]))])
        else:  # Get an out-of-bounds index
            variables.extend([('fragment_residues', out_of_bounds_residue)])
        core_residues = self.pose.core_residues
        if core_residues:
            variables.extend([('core_residues', ','.join([f'{res.number}{res.chain_id}' for res in core_residues]))])
        else:  # Get an out-of-bounds index
            variables.extend([('core_residues', out_of_bounds_residue)])

        rosetta_flags = rosetta.flags.copy()
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
        self.log.debug(f'Rosetta flags written to: {out_file}')

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
            putils.make_path(self.data_path)
            for entity in self.pose.entities:
                if entity.is_symmetric():  # Make symmetric energy in line with SymDesign energies v
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
            self.log.info(f'Metrics command for Entity {name}: {list2cmdline(metric_cmd)}')
            entity_metric_commands.append(metric_cmd)

        return entity_metric_commands

    def get_cmd_process_rosetta_metrics(self) -> list[list[str], ...]:
        """Generate a list of commands compatible with subprocess.Popen()/subprocess.list2cmdline() to run
        process-rosetta-metrics
        """
        return [putils.program_exe, flags.process_rosetta_metrics, '--single', self.pose_directory] \
            + self.job.parsed_arguments

    # def get_cmd_analysis(self) -> list[list[str], ...]:
    #     """Generate a list of commands compatible with subprocess.Popen()/subprocess.list2cmdline() to run analysis"""
    #     return [putils.program_exe, flags.analysis, '--single', self.pose_directory] + self.job.parsed_arguments

    def thread_sequences_to_backbone(self, sequences: dict[str, str]):
        """From the starting Pose, thread sequences onto the backbone, modifying relevant side chains i.e., mutate the
        Pose and build/pack using Rosetta FastRelax.

        Args:
            sequences: A mapping of sequence alias to its sequence. These will be used for producing outputs and as
                the input sequence
        """
        self.load_pose()
        # Write each "threaded" structure out for further processing
        number_of_residues = self.pose.number_of_residues
        # Ensure that mutations to the Pose aren't saved to state
        pose_copy = self.pose.copy()
        design_files = []
        for sequence_id, sequence in sequences.items():
            if len(sequence) != number_of_residues:
                raise DesignError(
                    f'The length of the sequence, {len(sequence)} != {number_of_residues}, '
                    f'the number of residues in the pose')
            for res_idx, residue_type in enumerate(sequence):
                pose_copy.mutate_residue(index=res_idx, to=residue_type)
            # pre_threaded_file = os.path.join(self.data_path, f'{self.name}_{self.protocol}{seq_idx:04d}.pdb')
            pre_threaded_file = os.path.join(self.data_path, f'{sequence_id}.pdb')
            design_files.append(pose_copy.write(out_path=pre_threaded_file))

        putils.make_path(self.scripts_path)
        design_files_file = os.path.join(self.scripts_path, f'{starttime}_{self.protocol}_files.txt')

        # # Modify each sequence score to reflect the new "decoy" name
        # # Todo update as a consequence of new SQL
        # sequence_ids = sequences.keys()
        # design_scores = metrics.parse_rosetta_scores(self.scores_file)
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
            raise DesignError(
                f'{self.thread_sequences_to_backbone.__name__}: No designed sequences were located')

        # self.refine(in_file_list=design_files_file)
        self.refine(design_files=design_files)
        # # Todo Ensure that the structure_path is updated, currently setting in self.process_rosetta_metrics()
        # design_data.structure_path = \
        #     pose.write(out_path=os.path.join(self.designs_path, f'{design_data.name}.pdb'))

    def predict_structure(self):
        """Perform structure prediction on the .current_designs. If there are no .current_designs, will use any
        design that is missing a structure file. Additionally, add the Pose sequence to the prediction by using the
        job.predict.pose flag
        """
        with self.job.db.session(expire_on_commit=False) as session:
            session.add(self)
            if self.current_designs:
                sequences = {design: design.sequence for design in self.current_designs}
            else:
                sequences = {design: design.sequence for design in self.get_designs_without_structure()}
                self.current_designs.extend(sequences.keys())

            if self.job.predict.pose:
                # self.pose_source is loaded in above session through get_designs_without_structure()
                pose_sequence = {self.pose_source: self.pose_source.sequence}
                if self.pose_source.structure_path is None:
                    sequences = {**pose_sequence, **sequences}
                elif self.job.overwrite:
                    sequences = {**pose_sequence, **sequences}
                else:
                    logger.warning(f"The flag --{flags.format_args(flags.predict_pose_args)} was specified, but the "
                                   "pose has already been predicted. If you meant to overwrite this pose, explicitly "
                                   "pass --overwrite")
        if not sequences:
            raise DesignError(
                f"Couldn't find any sequences to {self.predict_structure.__name__}")

        # match self.job.predict.method:  # Todo python 3.10
        #     case 'thread':
        #         self.thread_sequences_to_backbone(sequences)
        #     case 'alphafold':
        #         # Sequences use within alphafold requires .fasta...
        #         self.alphafold_predict_structure(sequences)
        #     case _:
        #         raise NotImplementedError(f"For {self.predict_structure.__name__}, the method '
        #                                   f"{self.job.predict.method} isn't implemented yet")

        self.protocol = self.job.predict.method
        if self.job.predict.method == 'thread':
            self.thread_sequences_to_backbone(sequences)
        elif self.job.predict.method == 'alphafold':
            self.alphafold_predict_structure(sequences)
        else:
            raise NotImplementedError(
                f"For {self.predict_structure.__name__}, the method {self.job.predict.method} isn't implemented yet")

    def alphafold_predict_structure(self, sequences: dict[sql.DesignData, str],
                                    model_type: resources.ml.af_model_literal = 'monomer', **kwargs):
        """Use Alphafold to predict structures for sequences of interest. The sequences will be fit to the Pose
        parameters, however will have sequence features unique to that sequence. By default, no multiple sequence
        alignment will be used and the AlphafoldInitialGuess model will be used wherein the starting coordinates for
        the Pose will be supplied as the initial guess

        According to Deepmind Alphafold.ipynb (2/3/23), the usage of multimer with > 3000 residues isn't validated.
        while a memory requirement of 4000 is the theoretical limit. I think it depends on the available memory
        Args:
            sequences: The mapping for DesignData to sequence
            model_type: The type of Alphafold model to use. Choose from 'monomer', 'monomer_casp14', 'monomer_ptm',
                or 'multimer'
        Returns:
            None
        """
        self.load_pose()
        number_of_residues = self.pose.number_of_residues
        logger.info(f'Performing structure prediction on {len(sequences)} sequences')
        for design, sequence in sequences.items():
            # Todo if differentially sized sequence inputs
            self.log.debug(f'Found sequence {sequence}')
            # max_sequence_length = max([len(sequence) for sequence in sequences.values()])
            if len(sequence) != number_of_residues:
                raise DesignError(
                    f'The sequence length {len(sequence)} != {number_of_residues}, the number of residues in the pose')
        # Get the DesignData.ids for metrics output
        design_ids = [design.id for design in sequences]

        # Hardcode parameters
        if model_type == 'monomer_casp14':
            num_ensemble = 8
        else:
            num_ensemble = 1

        number_of_entities = self.number_of_entities
        heteromer = number_of_entities > 1
        run_multimer_system = heteromer or self.pose.number_of_chains > 1
        if run_multimer_system:
            model_type = 'multimer'
            self.log.info(f'The AlphaFold model was automatically set to {model_type} due to detected multimeric pose')

        # # Todo enable compilation time savings by returning a precomputed model_factory. Padding the size of this may
        # #  help quite a bit
        # model_runners = resources.ml.alphafold_model_factory.get()
        if self.job.predict.num_predictions_per_model is None:
            if run_multimer_system:
                # Default is 5, with 5 models for 25 outputs. Could do 1 to increase speed...
                num_predictions_per_model = 5
            else:
                num_predictions_per_model = 1
        else:
            num_predictions_per_model = self.job.predict.num_predictions_per_model

        # Set up the various model_runners to supervise the prediction task for each sequence
        model_runners = resources.ml.set_up_model_runners(model_type=model_type,
                                                          num_predictions_per_model=num_predictions_per_model,
                                                          num_ensemble=num_ensemble,
                                                          development=self.job.development)

        def get_sequence_features_to_merge(seq_of_interest: str, multimer_length: int = None) -> FeatureDict:
            """Set up a sequence that has similar features to the Pose, but different sequence, say from design output

            Args:
                seq_of_interest: The sequence of interest
                multimer_length: The length of the multimer features, if the features are multimeric
            Returns:
                The Alphafold FeatureDict which is essentially a dictionary with dict[str, np.ndarray]
            """
            # Set up scores for each model
            sequence_length = len(seq_of_interest)
            # Set the sequence up in the FeatureDict
            # Todo the way that I am adding this here instead of during construction, seems that I should
            #  symmetrize the sequence before passing to af_predict(). This would occur by entity, where the first
            #  entity is combined, then the second entity is combined, etc. Any entity agnostic features such
            #  as all_atom_mask would be able to be made here
            _seq_features = af_pipeline.make_sequence_features(
                sequence=seq_of_interest, description='', num_res=sequence_length)
            # Always use the outer "domain_name" feature if there is one
            _seq_features.pop('domain_name')
            if multimer_length is not None:
                # Remove "object" dtype arrays. This may be required for "monomer" runs too
                # _seq_features.pop('domain_name')
                _seq_features.pop('between_segment_residues')
                _seq_features.pop('sequence')
                _seq_features.pop('seq_length')
                # The multimer model performs the one-hot operation itself. So processing gets the sequence as
                # the idx encoded by this v argmax on the one-hot
                _seq_features['aatype'] = np.argmax(_seq_features['aatype'], axis=-1).astype(np.int32)

                multimer_number, remainder = divmod(multimer_length, sequence_length)
                if remainder:
                    raise ValueError(
                        'The multimer_sequence_length and the sequence_length must differ by an integer number. Found '
                        f'multimer/monomer ({multimer_sequence_length})/({sequence_length}) with remainder {remainder}')
                for key in ['aatype', 'residue_index']:
                    _seq_features[key] = np.tile(_seq_features[key], multimer_number)
                # # For 'domain_name', 'sequence', and 'seq_length', transform the 1-D array to a scaler
                # # np.asarray(np.array(['pope'.encode('utf-8')], dtype=np.object_)[0], dtype=np.object_)
                # # Not sure why this transformation happens for multimer... as the multimer gets rid of them,
                # # but they are ready for the monomer pipeline
                # for key in ['domain_name', 'sequence']:
                #     _seq_features[key] = np.asarray(_seq_features[key][0], dtype=np.object_)
                # _seq_features['seq_length'] = np.asarray(sequence_length, dtype=np.int32)

            # Todo ensure that the sequence is merged such as in the merge_and_pair subroutine
            #  a good portion of which is below
            # if feature_name_split in SEQ_FEATURES:
            #     # Merges features along their length [MLKF...], [MLKF...] -> [MLKF...MLKF...]
            #     merged_example[feature_name] = np.concatenate(feats, axis=0)
            #     # If sequence is the same length as Entity/Pose, then 'residue_index' should be correct and
            #     # increasing for each sequence position, restarting at the beginning of a chain
            # Make the atom positions according to the sequence
            # Add all_atom_mask and dummy all_atom_positions based on aatype.
            all_atom_mask = residue_constants.STANDARD_ATOM_MASK[_seq_features['aatype']]
            _seq_features['all_atom_mask'] = all_atom_mask
            _seq_features['all_atom_positions'] = np.zeros(list(all_atom_mask.shape) + [3])

            # Todo check on 'seq_mask' introduction point for multimer...
            # elif feature_name_split in TEMPLATE_FEATURES:
            # DONT WORRY ABOUT THESE
            #     merged_example[feature_name] = np.concatenate(feats, axis=1)
            # elif feature_name_split in CHAIN_FEATURES:
            #     merged_example[feature_name] = np.sum(x for x in feats).astype(np.int32)
            # 'num_alignments' should be fine as msa incorporation has happened prior
            # else:
            #     # NO use at this time, these take the value of the first chain and eventually get thrown out
            #     # 'domain_name', 'sequence', 'between_segment_residues'
            #     merged_example[feature_name] = feats[0]
            return _seq_features

        def output_alphafold_structures(structure_types: dict[str, dict[str, str]], design_name: str = None):
            """From a PDB formatted string, output structures by design_name to self.designs_path/design_name"""
            if design_name is None:
                design_name = ''
            # for design, design_scores in structures.items():
            #     # # Output unrelaxed
            #     # structures = design_scores['structures_unrelaxed']
            for type_str in ['', 'un']:
                structures = structure_types.get(f'{type_str}relaxed', [])
                idx = count(1)
                for model_name, structure in structures.items():
                    out_dir = os.path.join(self.designs_path, design_name)
                    putils.make_path(out_dir)
                    path = os.path.join(out_dir,
                                        f'{design_name}-{model_name}_rank{next(idx)}-{type_str}relaxed.pdb')
                    with open(path, 'w') as f:
                        f.write(structure)
                # # Repeat for relaxed
                # structures = design_scores['structures']
                # idx = count(1)
                # for model_name, structure in structures.items():
                #     path = os.path.join(self.designs_path, f'{model_name}_rank{next(idx)}.pdb')
                #     with open(path, 'w') as f:
                #         f.write(structure)

        def _find_model_with_minimal_rmsd(models: dict[str, Structure], template_cb_coords) -> tuple[list[float], str]:
            """Use the CB coords to calculate the RMSD to template coordinates. Find the lowest RMSD Structure
            and transform to the template coordinates

            Returns:
                The calculated rmsds, and the name of the model with the minimal rmsd
            """
            min_rmsd = float('inf')
            minimum_model = None
            rmsds = []
            for af_model_name, model in models.items():
                rmsd, rot, tx = superposition3d(template_cb_coords, model.cb_coords)
                rmsds.append(rmsd)
                if rmsd < min_rmsd:
                    min_rmsd = rmsd
                    minimum_model = af_model_name
                    # Move the Alphafold model into the Pose reference frame
                    model.transform(rotation=rot, translation=tx)

            return rmsds, minimum_model

        def combine_model_scores(_scores: Sequence[dict[str, np.ndarray | float]]) -> dict[str, list[int | np.ndarray]]:
            """Add each of the different score types produced by each model to a dictionary with a list of model
            folding_scores in each category in the order the models appear

            Returns:
                A dictionary with format {'score_type': [score1, score2, ...], } where score1 can have a shape float,
                (n_residues,), or (n_residues, n_residues)
            """
            return {score_type: [scores[score_type] for scores in _scores] for score_type in _scores[0].keys()}

        def get_prev_pos_coords(sequence_: Iterable[str] = None, assembly: bool = False, entity: str = None) \
                -> jnp.ndarray:
            """Using the PoseJob.pose instance, get coordinates compatible with AlphafoldInitialGuess

            Args:
                sequence_: The sequence to use as a template to generate the coordinates
                assembly: Whether the coordinates should reflect the Pose Assembly
                entity: If coordinates should come from a Pose Entity, which one?
            Returns:
                The alphafold formatted sequence coords in a JAX array
            """
            pose_copy: Pose = self.pose.copy()
            # Choose which Structure to iterate over residues
            if entity is not None:
                structure = pose_copy.get_entity(entity)
            else:
                structure = pose_copy

            if sequence_ is None:
                # Make an all Alanine structure backbone as the prev_pos
                sequence_ = 'A' * structure.number_of_residues

            # Mutate to a 'lame' version of sequence/structure, removing any side-chain atoms not present
            for residue, residue_type in zip(structure.residues, sequence_):
                deleted_indices = pose_copy.mutate_residue(index=residue.index, to=residue_type)

            if assembly:
                af_coords = structure.assembly.alphafold_coords
            elif entity:
                af_coords = structure.assembly.alphafold_coords
            else:
                af_coords = structure.alphafold_coords

            return jnp.asarray(af_coords)

        if self.job.predict.models_to_relax is not None:
            relaxed = True
        else:
            relaxed = False

        # Hard code in the use of only design based single sequence models
        # if self.job.design:
        #     no_msa = True
        # else:
        #     no_msa = False
        no_msa = True
        debug_entities_and_oligomers = False
        # Ensure clashes aren't checked as these stop operation
        pose_kwargs = self.pose_kwargs.copy()

        # Get features for the Pose and predict
        if self.job.predict.designs:
            # For interface analysis the interface residues are needed
            self.identify_interface()
            if self.job.predict.assembly:
                if self.pose.number_of_symmetric_residues > resources.ml.MULTIMER_RESIDUE_LIMIT:
                    logger.critical(
                        f"Predicting on a symmetric input with {self.pose.number_of_symmetric_residues} isn't "
                        'recommended due to memory limitations')
                features = self.pose.get_alphafold_features(symmetric=True, multimer=run_multimer_system, no_msa=no_msa)
                # Todo may need to modify this if the pose isn't completely symmetric despite it being specified as such
                number_of_residues = self.pose.number_of_symmetric_residues
            else:
                features = self.pose.get_alphafold_features(symmetric=False, multimer=run_multimer_system,
                                                            no_msa=no_msa)

            if run_multimer_system:  # Get the length
                multimer_sequence_length = features['seq_length']
            else:
                multimer_sequence_length = None

            putils.make_path(self.designs_path)
            model_names = []
            design_poses_from_asu = []
            asu_design_scores = {}
            for design, sequence in sequences.items():
                this_seq_features = get_sequence_features_to_merge(sequence, multimer_length=multimer_sequence_length)
                logger.debug(f'Found this_seq_features:\n\t%s'
                                      % "\n\t".join((f"{k}={v}" for k, v in this_seq_features.items())))
                model_features = {'prev_pos': get_prev_pos_coords(sequence, assembly=self.job.predict.assembly)}
                logger.info(f'Predicting Design {design.name} structure')
                asu_structures, asu_scores = \
                    resources.ml.af_predict({**features, **this_seq_features, **model_features}, model_runners,
                                            gpu_relax=self.job.predict.use_gpu_relax,
                                            models_to_relax=self.job.predict.models_to_relax)
                if relaxed:
                    structures_to_load = asu_structures.get('relaxed', [])
                else:
                    structures_to_load = asu_structures.get('unrelaxed', [])
                # # Tod0 remove this after debug is done
                # output_alphafold_structures(asu_structures, design_name=f'{design}-asu')
                # asu_models = load_alphafold_structures(structures_to_load, name=str(design),  # Get '.name'
                #                                        entity_info=self.pose.entity_info)
                # Load the Model in while ignoring any potential clashes
                # Todo should I limit the .splitlines by the number_of_residues? Assembly v asu considerations
                asu_models = {model_name: Pose.from_pdb_lines(structure.splitlines(), name=design.name, **pose_kwargs)
                              for model_name, structure in structures_to_load.items()}
                # Because the pdb_lines aren't oriented, must handle orientation of incoming files to match sym_entry
                # This is handled in _find_model_with_minimal_rmsd(), however, the symmetry isn't set up correctly, i.e.
                # call pose.make_oligomers() in the correct orientation.
                # Do this all at once after every design
                if relaxed:  # Set b-factor data as relaxed get overwritten
                    for model_name, model in asu_models.items():
                        model.set_b_factor_data(asu_scores[model_name]['plddt'][:number_of_residues])

                # Check for the prediction rmsd between the cb coords of the Entity Model and Alphafold Model
                rmsds, minimum_model = _find_model_with_minimal_rmsd(asu_models, self.pose.cb_coords)
                if minimum_model is None:
                    raise DesignError(
                        f"Couldn't find the asu model with the minimal rmsd for Design {design}")
                # Append each ASU result to the full return
                design_poses_from_asu.append(asu_models[minimum_model])
                model_names.append(minimum_model)
                # structure_by_design[design].append(asu_models[minimum_model])
                # asu_design_scores.append({'rmsd_prediction_ensemble': rmsds, **asu_scores[minimum_model]})
                # Average all models scores to get the ensemble of the predictions
                combined_scores = combine_model_scores(list(asu_scores.values()))
                asu_design_scores[design.name] = {'rmsd_prediction_ensemble': rmsds, **combined_scores}
                # asu_design_scores[str(design)] = {'rmsd_prediction_ensemble': rmsds, **asu_scores[minimum_model]}
                """Each design in asu_design_scores contain the following features
                {'predicted_aligned_error': [(n_residues, n_residues), ...]  # multimer/monomer_ptm
                 'plddt': [(n_residues,), ...]
                 'predicted_interface_template_modeling_score': [float, ...]  # multimer
                 'predicted_template_modeling_score': [float, ...]  # multimer/monomer_ptm
                 'rmsd_prediction_ensemble: [(number_of_models), ...]
                 }
                """

            # Write the folded structure to designs_path and update DesignProtocols
            for idx, (design_data, pose) in enumerate(zip(sequences.keys(), design_poses_from_asu)):
                design_data.structure_path = \
                    pose.write(out_path=os.path.join(self.designs_path, f'{design_data.name}.pdb'))
                design_data.protocols.append(
                    sql.DesignProtocol(design_id=design_data.id, job_id=self.job.id, protocol=self.protocol,
                                       alphafold_model=model_names[idx], file=design_data.structure_path)
                )
                if debug_entities_and_oligomers:
                    # Explicitly pass the transformation parameters which are correct for the PoseJob
                    pose.make_oligomers(transformations=self.transformations)
                    pose.write(out_path=os.path.join(self.designs_path, f'{pose.name}-asu-check.pdb'))
                    pose.write(out_path=os.path.join(self.designs_path, f'{pose.name}-assembly-check.pdb'),
                               assembly=True)
                    for entity in pose.entities:
                        entity.write(out_path=os.path.join(self.designs_path,
                                                           f'{pose.name}{entity.name}-oligomer-asu-check.pdb'))
                        entity.write(out_path=os.path.join(self.designs_path,
                                                           f'{pose.name}{entity.name}-oligomer-check.pdb'),
                                     assembly=True)

            # Using the 2-fold aware pose.interface_residues_by_interface
            interface_indices = tuple([residue.index for residue in residues]
                                      for residues in self.pose.interface_residues_by_interface.values())
            # All index are based on design.name
            residues_df = self.analyze_residue_metrics_per_design(design_poses_from_asu)
            designs_df = self.analyze_design_metrics_per_design(residues_df, design_poses_from_asu)
            predict_designs_df, predict_residues_df = \
                self.analyze_alphafold_metrics(asu_design_scores, number_of_residues,
                                               model_type=model_type, interface_indices=interface_indices)
            residue_indices = list(range(number_of_residues))
            # Set the index to use the design.id for each design instance
            design_index = pd.Index(design_ids, name=sql.ResidueMetrics.design_id.name)
            residue_sequences_df = pd.DataFrame([list(seq) for seq in sequences.values()],
                                                index=predict_residues_df.index,
                                                columns=pd.MultiIndex.from_product([residue_indices,
                                                                                    [sql.ResidueMetrics.type.name]]))
            residues_df = residues_df.join([predict_residues_df, residue_sequences_df])
            designs_df = designs_df.join(predict_designs_df)
            designs_df.index = residues_df.index = design_index
            # Output collected metrics
            with self.job.db.session(expire_on_commit=False) as session:
                self.output_metrics(session, designs=designs_df)
                output_residues = False
                if output_residues:
                    self.output_metrics(session, residues=residues_df)
                # else:  # Only save the 'design_residue' columns
                #     residues_df = residues_df.loc[:, idx_slice[:, sql.DesignResidues.design_residue.name]]
                #     self.output_metrics(session, design_residues=residues_df)
                # Commit the newly acquired metrics
                session.commit()

        # Prepare the features to feed to the model
        if self.job.predict.entities:  # and self.number_of_entities > 1:
            # Get the features for each oligomeric Entity
            # The folding_scores will all be the length of the gene Entity, not oligomer
            # entity_scores_by_design = {design: [] for design in sequences}

            # Sort the entity instances by their length to improve compile time.
            entities = self.pose.entities
            # The only compile should be the first prediction
            entity_number_of_residues = [(entity.number_of_residues, idx) for idx, entity in enumerate(entities)]
            entity_idx_sorted_residue_number_highest_to_lowest = \
                [idx for _, idx in sorted(entity_number_of_residues, key=lambda pair: pair[0], reverse=True)]
            sorted_entities_and_data = [(entities[idx], self.entity_data[idx])
                                        for idx in entity_idx_sorted_residue_number_highest_to_lowest]
            entity_structure_by_design: dict[str, list[ContainsEntities]] = defaultdict(list)
            entity_design_dfs = []
            entity_residue_dfs = []
            for entity, entity_data in sorted_entities_and_data:
                # Fold with symmetry True. If it isn't symmetric, symmetry won't be used
                features = entity.get_alphafold_features(symmetric=True, no_msa=no_msa)
                if run_multimer_system:  # Get the length
                    multimer_sequence_length = features['seq_length']
                    entity_number_of_residues = entity.assembly.number_of_residues
                else:
                    multimer_sequence_length = None
                    entity_number_of_residues = entity.number_of_residues

                logger.debug(f'Found oligomer with length: {entity_number_of_residues}')

                # if run_multimer_system:
                entity_cb_coords = np.concatenate([mate.cb_coords for mate in entity.chains])
                # Todo
                #  entity_backbone_and_cb_coords = entity.assembly.cb_coords
                # else:
                #     entity_cb_coords = entity.cb_coords

                entity_interface_residues = \
                    self.pose.get_interface_residues(entity1=entity, entity2=entity,
                                                     distance=self.job.interface_distance, oligomeric_interfaces=True)
                offset_index = entity.offset_index
                entity_interface_indices = tuple([residue.index - offset_index for residue in residues]
                                                 for residues in entity_interface_residues)
                entity_name = entity.name
                this_entity_info = {entity_name: self.pose.entity_info[entity_name]}
                entity_model_kwargs = dict(name=entity_name, entity_info=this_entity_info)
                entity_slice = slice(entity.n_terminal_residue.index, 1 + entity.c_terminal_residue.index)
                entity_scores_by_design = {}
                # Iterate over provided sequences. Find the best structural model and it's folding_scores
                for design, sequence in sequences.items():
                    sequence = sequence[entity_slice]
                    this_seq_features = \
                        get_sequence_features_to_merge(sequence, multimer_length=multimer_sequence_length)
                    logger.debug(f'Found this_seq_features:\n\t'
                                 '%s' % "\n\t".join((f"{k}={v}" for k, v in this_seq_features.items())))
                    # If not an oligomer, then get_prev_pos_coords() will just use the entity
                    model_features = {'prev_pos': get_prev_pos_coords(sequence, entity=entity_name)}
                    logger.info(f'Predicting Design {design.name} Entity {entity_name} structure')
                    entity_structures, entity_scores = \
                        resources.ml.af_predict({**features, **this_seq_features, **model_features}, model_runners)
                    # NOT using relaxation as these won't be output for design so their coarse features are all that
                    # are desired
                    #     gpu_relax=self.job.predict.use_gpu_relax, models_to_relax=self.job.predict.models_to_relax)
                    # if relaxed:
                    #     structures_to_load = entity_structures.get('relaxed', [])
                    # else:
                    structures_to_load = entity_structures.get('unrelaxed', [])

                    design_models = {model_name: Pose.from_pdb_lines(structure.splitlines(), **entity_model_kwargs)
                                     for model_name, structure in structures_to_load.items()}
                    # if relaxed:  # Set b-factor data as relaxed get overwritten
                    #     type_str = ''
                    #     for model_name, model in design_models.items():
                    #         model.set_b_factor_data(entity_scores[model_name]['plddt'][:entity_number_of_residues])
                    #     entity_structures['relaxed'] = \
                    #         {model_name: model.get_atom_record() for model_name, model in design_models.items()}
                    # else:
                    type_str = 'un'

                    # output_alphafold_structures(entity_structures, design_name=f'{design}-{entity.name}')
                    # Check for the prediction rmsd between the backbone of the Entity Model and Alphafold Model
                    # Also, perform an alignment to the pose Entity
                    rmsds, minimum_model = _find_model_with_minimal_rmsd(design_models, entity_cb_coords)
                    if minimum_model is None:
                        raise DesignError(
                            f"Couldn't find the Entity {entity.name} model with the minimal rmsd for Design {design}")

                    # Put Entity Model into a directory in pose/designs/pose-design_id/entity.name.pdb
                    out_dir = os.path.join(self.designs_path, f'{design.name}')
                    putils.make_path(out_dir)
                    path = os.path.join(out_dir, f'{entity.name}-{minimum_model}-{type_str}relaxed.pdb')
                    minimum_entity = design_models[minimum_model]
                    minimum_entity.write(out_path=path)
                    # Append each Entity result to the full return
                    entity_structure_by_design[design].append(minimum_entity)
                    # Average all models scores to get the ensemble of the predictions
                    combined_scores = combine_model_scores(list(entity_scores.values()))
                    entity_scores_by_design[design.name] = {'rmsd_prediction_ensemble': rmsds, **combined_scores}
                    # entity_scores_by_design[str(design)] = \
                    #     {'rmsd_prediction_ensemble': rmsds, **entity_scores[minimum_model]}
                    """Each design in entity_scores_by_design contains the following features
                    {'predicted_aligned_error': [(n_residues, n_residues), ...]  # multimer/monomer_ptm
                     'plddt': [(n_residues,), ...]
                     'predicted_interface_template_modeling_score': [float, ...]  # multimer
                     'predicted_template_modeling_score': [float, ...]  # multimer/monomer_ptm
                     'rmsd_prediction_ensemble: [(number_of_models), ...]
                     }
                    """

                # Todo
                #  Ensure the sequence length is the size of the entity. If saving the entity_residues_df need to
                #  change the column index to reflect the number of residues
                entity_sequence_length = entity_slice.stop - entity_slice.start
                entity_designs_df, entity_residues_df = \
                    self.analyze_alphafold_metrics(entity_scores_by_design, entity_sequence_length,
                                                   model_type=model_type, interface_indices=entity_interface_indices)
                # Set the index to use the design.id for each design instance and EntityData.id as an additional column
                entity_designs_df.index = pd.MultiIndex.from_product(
                    [design_ids, [entity_data.id]], names=[sql.DesignEntityMetrics.design_id.name,
                                                           sql.DesignEntityMetrics.entity_id.name])
                entity_design_dfs.append(entity_designs_df)

                # These aren't currently written...
                entity_residue_dfs.append(entity_residues_df)

            # Save the entity_designs_df DataFrames
            with self.job.db.session(expire_on_commit=False) as session:
                for entity_designs_df in entity_design_dfs:
                    sql.write_dataframe(session, entity_designs=entity_designs_df)
                session.commit()

            # Try to perform an analysis of the separated versus the combined prediction
            if self.job.predict.designs:
                # Combine Entity structure to compare with the Pose prediction
                design_pose_from_entities = []
                for design, entity_structs in entity_structure_by_design.items():
                    # Reorder the entity structures as before AF compile
                    entity_models = [entity_structs[idx] for idx in entity_idx_sorted_residue_number_highest_to_lowest]
                    _pose = Pose.from_entities([entity for model in entity_models for entity in model.entities],
                                               name=design.name, **pose_kwargs)
                    design_pose_from_entities.append(_pose)
                # Combine Entity scores to compare with the Pose prediction
                for residue_df, entity in zip(entity_residue_dfs, entities):
                    # Rename the residue_indices along the top most column of DataFrame
                    residue_df.columns = residue_df.columns.set_levels(
                        list(range(entity.n_terminal_residue.index,
                                   1 + entity.c_terminal_residue.index)),
                        level=0
                    )
                # residue_df.rename(columns=dict(zip(range(entity.number_of_residues),
                #                                    range(entity.n_terminal_residue.index,
                #                                          entity.c_terminal_residue.index)
                #                                    )))
                entity_residues_df = pd.concat(entity_residue_dfs, axis=1)
                try:
                    self.log.info('Testing the addition of entity_designs_df. They were not adding correctly without '
                                  '*unpack')
                    entity_designs_df, *extra_entity_designs_df = entity_design_dfs
                    for df in extra_entity_designs_df:
                        entity_designs_df += df

                    entity_designs_df /= number_of_entities
                    # entity_designs_df = pd.concat(entity_design_dfs, axis=0)
                    # score_types_mean = ['rmsd_prediction_ensemble']
                    # if 'multimer' in model_type:
                    #     score_types_mean += ['predicted_interface_template_modeling_score',
                    #                          'predicted_template_modeling_score']
                    # elif 'ptm' in model_type:
                    #     score_types_mean += ['predicted_template_modeling_score']
                    #
                    # score_types_concat = ['predicted_aligned_error', 'plddt']
                    #
                    # entity_design_scores = []
                    # for design in sequences:
                    #     entity_scores = entity_scores_by_design[design]
                    #     logger.debug(f'Found entity_scores with contents:\n{entity_scores}')
                    #     scalar_scores = {score_type: sum([sum(scores[score_type]) for scores in entity_scores])
                    #                      / number_of_entities
                    #                      for score_type in score_types_mean}
                    #     # 'predicted_aligned_error' won't concat correctly, so we average over each residue first
                    #     for scores in entity_scores:
                    #         scores['predicted_aligned_error'] = scores['predicted_aligned_error'].mean(axis=-1)
                    #     array_scores = {score_type: np.concatenate([scores[score_type] for scores in entity_scores])
                    #                     for score_type in score_types_concat}
                    #     scalar_scores.update(array_scores)
                    #     logger.debug(f'Found scalar_scores with contents:\n{scalar_scores}')
                    #     entity_design_scores.append(scalar_scores)

                    # scores = {}
                    rmsds = []
                    for idx, design in enumerate(sequences):
                        entity_pose = design_pose_from_entities[idx]
                        asu_model = design_poses_from_asu[idx]
                        # Find the RMSD between each type
                        rmsd, rot, tx = superposition3d(asu_model.backbone_and_cb_coords,
                                                        entity_pose.backbone_and_cb_coords)
                        # score_deviation['rmsd'] = rmsd
                        # scores[design] = score_deviation
                        # self.log.critical(f'Found rmsd between separated entities and combined pose: {rmsd}')
                        rmsds.append(rmsd)

                    # Compare all folding_scores
                    design_deviation_df = (predict_designs_df - entity_designs_df).abs()
                    design_deviation_df['rmsd_prediction_deviation'] = rmsds
                    design_deviation_file = \
                        os.path.join(self.data_path, f'{starttime}-af_pose-entity-designs-deviation_scores.csv')
                    design_deviation_df.to_csv(design_deviation_file)
                    logger.info('Wrote the design deviation file (between separate Entity instances and Pose)'
                                f' to: {design_deviation_file}')
                    residue_deviation_df = (predict_residues_df - entity_residues_df).abs()
                    deviation_file = \
                        os.path.join(self.data_path, f'{starttime}-af_pose-entity-residues-deviation_scores.csv')
                    residue_deviation_df.to_csv(deviation_file)
                except Exception as error:
                    raise DesignError(error)

    # Todo move to_pose_directory version to own PoseProtocol method.
    #  Use refine as a means to refine Designs?
    def refine(self, to_pose_directory: bool = True, gather_metrics: bool = True,
               design_files: list[AnyStr] = None, in_file_list: AnyStr = None):
        """Refine the PoseJob.pose instance or design Model instances associated with this instance

        Args:
            to_pose_directory: Whether the refinement should be saved to the PoseJob
            gather_metrics: Whether metrics should be calculated for the Pose
            design_files: A list of files to perform refinement on
            in_file_list: The path to a file containing a list of files to pass to Rosetta refinement
        """
        main_cmd = rosetta.script_cmd.copy()

        suffix = []
        generate_files_cmd = null_cmd
        if to_pose_directory:  # Original protocol to refine a Nanohedra pose
            flag_dir = self.scripts_path
            pdb_out_path = self.designs_path
            refine_pdb = self.refine_pdb
            refined_pdb = self.refined_pdb
            additional_flags = []
        else:  # Protocol to refine input structure, place in a common location, then transform for many jobs to source
            flag_dir = pdb_out_path = self.job.refine_dir
            refine_pdb = self.source_path
            refined_pdb = os.path.join(pdb_out_path, refine_pdb)
            additional_flags = ['-no_scorefile', 'true']

        flags_file = os.path.join(flag_dir, 'flags')
        self.prepare_rosetta_flags(pdb_out_path=pdb_out_path, out_dir=flag_dir)
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
                self.log.warning(f"{self.refine.__name__}: The requested protocol '{self.protocol}', wasn't recognized "
                                 f"and is being treated as the standard '{switch}' protocol")

            # Create file output
            designed_files_file = os.path.join(self.scripts_path, f'{starttime}_{switch}_files_output.txt')
            if in_file_list:
                design_files_file = in_file_list
                generate_files_cmd = \
                    ['python', putils.list_pdb_files, '-d', self.designs_path, '-o', designed_files_file, '-e', '.pdb',
                     '-s', f'_{switch}']
            elif design_files:
                design_files_file = os.path.join(self.scripts_path, f'{starttime}_{self.protocol}_files.txt')
                with open(design_files_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(design_files))
                # Write the designed_files_file with all "tentatively" designed file paths
                out_file_string = f'%s{os.sep}{pdb_out_path}{os.sep}%s'
                with open(designed_files_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(os.path.join(pdb_out_path, os.path.basename(file))
                                               for file in design_files))
            else:
                raise ValueError(
                    f"Couldn't run {self.refine.__name__} without passing parameter 'design_files' or 'in_file_list'")

            # -in:file:native is here to block flag file version, not actually useful for refine
            infile = ['-in:file:l', design_files_file, '-in:file:native', self.source_path]
            metrics_pdb = ['-in:file:l', designed_files_file, '-in:file:native', self.source_path]
            # generate_files_cmdline = [list2cmdline(generate_files_cmd)]
        else:
            self.protocol = switch = putils.refine
            if self.job.interface_to_alanine:  # Mutate all design positions to Ala before the Refinement
                # Ensure the mutations to the pose are wiped
                pose_copy = self.pose.copy()
                self.log.critical(f'In {self.refine.__name__}, ensure that the pose was copied correctly before '
                                  'trusting output. The residue in mutate_residue() should be equal after the copy... '
                                  'Delete this log if everything checks out')
                for entity_pair, interface_residues_pair in self.pose.interface_residues_by_entity_pair.items():
                    # if interface_residues_pair[0]:  # Check that there are residues present
                    for entity, interface_residues in zip(entity_pair, interface_residues_pair):
                        for residue in interface_residues:
                            if residue.type != 'GLY':  # No mutation from GLY to ALA as Rosetta would build a CB
                                pose_copy.mutate_residue(residue=residue, to='A')
                # Change the name to reflect mutation so the self.pose_path isn't overwritten
                refine_pdb = f'{os.path.splitext(refine_pdb)[0]}_ala_mutant.pdb'
            # else:  # Do nothing and refine the source
            #     pass

            # # Set up self.refined_pdb by using a suffix
            # suffix = ['-out:suffix', f'_{switch}']

            self.pose.write(out_path=refine_pdb)
            self.log.debug(f'Cleaned PDB for {switch}: "{refine_pdb}"')
            # -in:file:native is here to block flag file version, not actually useful for refine
            infile = ['-in:file:s', refine_pdb, '-in:file:native', refine_pdb]
            metrics_pdb = ['-in:file:s', refined_pdb, '-in:file:native', refine_pdb]

        # RELAX: Prepare command
        if self.symmetry_dimension is not None and self.symmetry_dimension > 0:
            symmetry_definition = ['-symmetry_definition', 'CRYST1']
        else:
            symmetry_definition = []

        # '-no_nstruct_label', 'true' comes from v
        relax_cmd = main_cmd + rosetta.relax_flags_cmdline + additional_flags + symmetry_definition \
            + [f'@{flags_file}', '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, f'refine.xml'),
               '-parser:script_vars', f'switch={switch}'] + infile + suffix \
            + (['-overwrite'] if self.job.overwrite else [])
        self.log.info(f'{switch.title()} Command: {list2cmdline(relax_cmd)}')

        if gather_metrics or self.job.metrics:
            gather_metrics = True
            if self.job.mpi > 0:
                main_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(self.job.mpi)] + main_cmd
            # main_cmd += metrics_pdb
            main_cmd += [f'@{flags_file}', '-out:file:score_only', self.scores_file,
                         '-no_nstruct_label', 'true'] + metrics_pdb + ['-parser:protocol']
            dev_label = '_DEV' if self.job.development else ''
            metric_cmd_bound = main_cmd \
                + [os.path.join(putils.rosetta_scripts_dir, f'interface_metrics{dev_label}.xml')] \
                + symmetry_definition
            entity_cmd = main_cmd + [os.path.join(putils.rosetta_scripts_dir, f'metrics_entity{dev_label}.xml')]
            self.log.info(f'Metrics command for Pose: {list2cmdline(metric_cmd_bound)}')
            metric_cmds = [metric_cmd_bound] \
                + self.generate_entity_metrics_commands(entity_cmd)
        else:
            metric_cmds = []

        # Create executable/Run FastRelax on Clean ASU with RosettaScripts
        if self.job.distribute_work:
            analysis_cmd = self.get_cmd_process_rosetta_metrics()
            self.current_script = distribute.write_script(
                list2cmdline(relax_cmd), name=f'{starttime}_{self.protocol}.sh', out_path=flag_dir,
                additional=[list2cmdline(generate_files_cmd)]
                + [list2cmdline(command) for command in metric_cmds] + [list2cmdline(analysis_cmd)])
        else:
            relax_process = Popen(relax_cmd)
            relax_process.communicate()  # Wait for command to complete
            list_all_files_process = Popen(generate_files_cmd)
            list_all_files_process.communicate()
            if gather_metrics:
                for metric_cmd in metric_cmds:
                    metrics_process = Popen(metric_cmd)
                    metrics_process.communicate()

                # Gather metrics for each design produced from this procedure
                if os.path.exists(self.scores_file):
                    self.process_rosetta_metrics()

    def rosetta_interface_design(self):
        """For the basic process of sequence design between two halves of an interface, write the necessary files for
        refinement (FastRelax), redesign (FastDesign), and metrics collection (Filters & SimpleMetrics)

        Stores job variables in a [stage]_flags file and the command in a [stage].sh file. Sets up dependencies based
        on the PoseJob
        """
        raise NotImplementedError(
            f'There are multiple outdated dependencies that need to be updated to use Rosetta {flags.interface_design}'
            f'with modern {putils.program_name}')
        # Todo
        #  Modify the way that files are generated/named and later listed for metrics. Right now, reliance on the file
        #  suffix to get this right, but with the database, this is not needed and will result in inaccurate use
        # Set up the command base (rosetta bin and database paths)
        main_cmd = rosetta.script_cmd.copy()
        if self.symmetry_dimension is not None and self.symmetry_dimension > 0:
            main_cmd += ['-symmetry_definition', 'CRYST1']

        # Todo - Has this been solved?
        #  must set up a blank -in:file:pssm in case the evolutionary matrix is not used. Design will fail!!
        profile_cmd = ['-in:file:pssm', self.evolutionary_profile_file] \
            if os.path.exists(self.evolutionary_profile_file) else []

        additional_cmds = []
        out_file = []
        design_files = os.path.join(self.scripts_path, f'{starttime}_design-files_{self.protocol}.txt')
        if self.job.design.scout:
            self.protocol = protocol_xml1 = putils.scout
            # metrics_pdb = ['-in:file:s', self.scouted_pdb]
            generate_files_cmd = \
                ['python', putils.list_pdb_files, '-d', self.designs_path, '-o', design_files, '-e', '.pdb',
                 '-s', f'_{self.protocol}']
            metrics_pdb = ['-in:file:l', design_files]
            # metrics_flags = 'repack=no'
            nstruct_instruct = ['-no_nstruct_label', 'true']
        else:
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
                    [[str(putils.hbnet_sort), os.path.join(self.data_path, 'hbnet_silent.o'),
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
                self.protocol = protocol_xml1 = flags.interface_design
                nstruct_instruct = ['-nstruct', str(self.job.design.number)]

        # DESIGN: Prepare command and flags file
        self.prepare_rosetta_flags(out_dir=self.scripts_path)

        if self.job.design.method == putils.consensus:
            self.protocol = putils.consensus
            design_cmd = main_cmd + rosetta.relax_flags_cmdline \
                + [f'@{self.flags}', '-in:file:s', self.consensus_pdb,
                   # '-in:file:native', self.refined_pdb,
                   '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, f'consensus.xml'),
                   '-out:suffix', f'_{self.protocol}', '-parser:script_vars', f'switch={putils.consensus}']
        else:
            design_cmd = main_cmd + profile_cmd + \
                [f'@{self.flags}', '-in:file:s', self.refined_pdb,
                 # self.scouted_pdb if os.path.exists(self.scouted_pdb) else self.refined_pdb,
                 '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, f'{protocol_xml1}.xml'),
                 '-out:suffix', f'_{self.protocol}'] + out_file + nstruct_instruct
        if self.job.overwrite:
            design_cmd += ['-overwrite']

        # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
        # Assumes all entity chains are renamed from A to Z for entities (1 to n)
        entity_cmd = rosetta.script_cmd + metrics_pdb + \
            [f'@{self.flags}', '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true',
             '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, 'metrics_entity.xml')]

        if self.job.mpi > 0 and not self.job.design.scout:
            design_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(self.job.mpi)] + design_cmd
            entity_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(self.job.mpi)] + entity_cmd

        self.log.info(f'{self.rosetta_interface_design.__name__} command: {list2cmdline(design_cmd)}')
        metric_cmds = self.generate_entity_metrics_commands(entity_cmd)

        # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics
        if self.job.distribute_work:
            analysis_cmd = self.get_cmd_process_rosetta_metrics()
            self.current_script = distribute.write_script(
                list2cmdline(design_cmd), name=f'{starttime}_{self.protocol}', out_path=self.scripts_path,
                additional=[list2cmdline(command) for command in additional_cmds]
                + [list2cmdline(generate_files_cmd)]
                + [list2cmdline(command) for command in metric_cmds] + [list2cmdline(analysis_cmd)])
        else:
            design_process = Popen(design_cmd)
            design_process.communicate()  # Wait for command to complete
            for command in additional_cmds:
                process = Popen(command)
                process.communicate()
            list_all_files_process = Popen(generate_files_cmd)
            list_all_files_process.communicate()
            for metric_cmd in metric_cmds:
                metrics_process = Popen(metric_cmd)
                metrics_process.communicate()

            # Gather metrics for each design produced from this procedure
            if os.path.exists(self.scores_file):
                self.process_rosetta_metrics()

    def proteinmpnn_design(self, interface: bool = False, neighbors: bool = False) -> None:
        """Perform design based on the ProteinMPNN graph encoder/decoder network

        Sets:
            self.protocol = 'proteinmpnn'

        Args:
            interface: Whether to only specify the interface as designable, otherwise, use all residues
            neighbors: Whether to design interface neighbors

        Returns:
            None
        """
        self.protocol = flags.proteinmpnn
        self.log.info(f'Starting {self.protocol} design calculation with {self.job.design.number} '
                      f'designs over each of the temperatures: {self.job.design.temperatures}')
        # design_start = time.time()
        sequences_and_scores: dict[str, np.ndarray | list] = \
            self.pose.design_sequences(number=self.job.design.number,
                                       temperatures=self.job.design.temperatures,
                                       # interface=interface, neighbors=neighbors,
                                       interface=self.job.design.interface, neighbors=self.job.design.neighbors,
                                       ca_only=self.job.design.ca_only,
                                       model_name=self.job.design.proteinmpnn_model
                                       )
        # self.log.debug(f"Took {time.time() - design_start:8f}s for design_sequences")

        # self.output_proteinmpnn_scores(design_names, sequences_and_scores)
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

        # # Write every designed sequence to an individual file...
        # putils.make_path(self.designs_path)
        # design_names = [design_data.name for design_data in designs_data]
        # sequence_files = [
        #     write_sequences(sequence, names=name, file_name=os.path.join(self.designs_path, name))
        #     for name, sequence in zip(design_names, sequences_and_scores['sequences'])
        # ]

        # Update the Pose with the number of designs
        with self.job.db.session(expire_on_commit=False) as session:
            session.add(self)
            designs_data = self.update_design_data(design_parent=self.pose_source)
            session.add_all(designs_data)
            session.flush()
            design_ids = [design_data.id for design_data in designs_data]

            # Update the Pose with the design protocols
            for idx, design_data in enumerate(designs_data):
                design_data.protocols.append(
                    sql.DesignProtocol(design=design_data,
                                       job_id=self.job.id,
                                       protocol=self.protocol,
                                       temperature=temperatures[idx],
                                       ))

            # analysis_start = time.time()
            designs_df, residues_df = self.analyze_proteinmpnn_metrics(design_ids, sequences_and_scores)
            entity_designs_df = self.analyze_design_entities_per_residue(residues_df)
            sql.write_dataframe(session, entity_designs=entity_designs_df)

            # self.log.debug(f"Took {time.time() - analysis_start:8f}s for analyze_proteinmpnn_metrics. "
            #                f"{time.time() - design_start:8f}s total")
            self.output_metrics(session, designs=designs_df)
            output_residues = False
            if output_residues:
                self.output_metrics(session, residues=residues_df)
            else:  # Only save the 'design_residue' columns
                residues_df = residues_df.loc[:, idx_slice[:, sql.DesignResidues.design_residue.name]]
                self.output_metrics(session, design_residues=residues_df)
            # Commit the newly acquired metrics
            session.commit()

    # de output_proteinmpnn_scores(self, design_ids: Sequence[str], sequences_and_scores: dict[str, np.ndarray | list]):
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

    def update_design_protocols(
            self, design_ids: Sequence[str], protocols: Sequence[str] = None,
            temperatures: Sequence[float] = None, files: Sequence[AnyStr] = None
    ) -> list[sql.DesignProtocol]:  # Unused
        """Associate newly created DesignData with a DesignProtocol

        Args:
            design_ids: The identifiers for each DesignData
            protocols: The sequence of protocols to associate with DesignProtocol
            temperatures: The temperatures to associate with DesignProtocol
            files: The sequence of files to associate with DesignProtocol
        Returns:
            The new instances of the sql.DesignProtocol
        """
        metadata = [
            sql.DesignProtocol(
                design_id=design_id, job_id=self.job.id, protocol=protocol, temperature=temperature, file=file
            )
            for design_id, protocol, temperature, file in zip(design_ids, protocols, temperatures, files)
        ]
        return metadata

    def update_design_data(self, design_parent: sql.DesignData, number: int = None) -> list[sql.DesignData]:
        """Updates the PoseData with newly created design identifiers using DesignData

        Sets:
            self.current_designs (list[DesignData]): Extends with the newly created DesignData instances

        Args:
            design_parent: The design whom all new designs are based
            number: The number of designs. If not provided, set according to job.design.number * job.design.temperature

        Returns:
            The new instances of the DesignData
        """
        if number is None:
            number = len(self.job.design.number * self.job.design.temperatures)

        first_new_design_idx = self.number_of_designs  # + 1 <- don't add 1 since the first design is .pose_source
        # design_names = [f'{self.protocol}{seq_idx:04d}'  # f'{self.name}_{self.protocol}{seq_idx:04d}'
        design_names = [f'{self.name}-{design_idx:04d}'
                        for design_idx in range(first_new_design_idx, first_new_design_idx + number)]
        designs = [sql.DesignData(name=name, pose_id=self.id, design_parent=design_parent)
                   for name in design_names]
        # Set the PoseJob.current_designs for access by subsequent functions/protocols
        self.current_designs.extend(designs)

        return designs

    def output_metrics(self, session: Session = None, designs: pd.DataFrame = None,
                       design_residues: pd.DataFrame = None, residues: pd.DataFrame = None, pose_metrics: bool = False):
        """Format each possible DataFrame type for output via csv or SQL database

        Args:
            session: A currently open transaction within sqlalchemy
            designs: The typical per-design metric DataFrame where each index is the design id and the columns are
                design metrics
            design_residues: The typical per-residue metric DataFrame where each index is the design id and the columns
                are (residue index, Boolean for design utilization)
            residues: The typical per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
            pose_metrics: Whether the metrics being included are based on Pose (self.pose) measurements
        """
        # Remove completely empty columns
        if designs is not None:
            designs.dropna(how='all', axis=1, inplace=True)
        if residues is not None:
            residues.dropna(how='all', axis=1, inplace=True)

        if session is not None:
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
                # _design_ids = metrics.sql.write_dataframe(session, designs=designs)
                sql.write_dataframe(session, designs=designs)

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
                sql.write_dataframe(session, **dataframe_kwargs)

            if design_residues is not None:
                design_residues.index.set_names(sql.ResidueMetrics.design_id.name, inplace=True)
                sql.write_dataframe(session, design_residues=design_residues)
        else:
            putils.make_path(self.data_path)
            if residues is not None:
                # Process dataframes for missing values NOT USED with SQL...
                residues = residues.fillna(0.)
                residues.sort_index(inplace=True)
                residues.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
                # residue_metric_columns = residues.columns.levels[-1].tolist()
                # self.log.debug(f'Residues metrics present: {residue_metric_columns}')

                residues.to_csv(self.residues_metrics_csv)
                self.log.info(f'Wrote Residues metrics to {self.residues_metrics_csv}')

            if designs is not None:
                designs.sort_index(inplace=True, axis=1)
                # designs_metric_columns = designs.columns.tolist()
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

    def parse_rosetta_scores(self, scores: dict[str, dict[str, str | int | float]]) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process Rosetta scoring dictionary into suitable values

        Args:
            scores: The dictionary of scores to be parsed
        Returns:
            A tuple of DataFrame where each contains (
                A per-design metric DataFrame where each index is the design id and the columns are design metrics,
                A per-residue metric DataFrame where each index is the design id and the columns are
                    (residue index, residue metric)
            )
        """
        # Create protocol dataframe
        scores_df = pd.DataFrame.from_dict(scores, orient='index')
        # # Fill in all the missing values with that of the default pose_source
        # scores_df = pd.concat([source_df, scores_df]).fillna(method='ffill')
        # Gather all columns into specific types for processing and formatting
        per_res_columns = [column for column in scores_df.columns.tolist() if 'res_' in column]

        # Check proper input
        metric_set = metrics.rosetta_required.difference(set(scores_df.columns))
        if metric_set:
            self.log.debug(f'Score columns present before required metric check: {scores_df.columns.tolist()}')
            raise DesignError(
                f'Missing required metrics: "{", ".join(metric_set)}"')

        # Remove unnecessary (old scores) as well as Rosetta pose score terms besides ref (has been renamed above)
        # Todo learn know how to produce Rosetta score terms in output score file. Not in FastRelax...
        remove_columns = metrics.rosetta_terms + metrics.unnecessary + per_res_columns
        # Todo remove dirty when columns are correct (after P432)
        #  and column tabulation precedes residue/hbond_processing
        scores_df.drop(remove_columns, axis=1, inplace=True, errors='ignore')

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

        # Drop designs where required data isn't present
        if putils.protocol in scores_df.columns:
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

        # Replace empty strings with np.nan and convert remaining to float
        scores_df.replace('', np.nan, inplace=True)
        scores_df.fillna(dict(zip(metrics.protocol_specific_columns, repeat(0))), inplace=True)
        # scores_df = scores_df.astype(float)  # , copy=False, errors='ignore')

        # protocol_s.drop(missing_group_indices, inplace=True, errors='ignore')
        viable_designs = scores_df.index.tolist()
        if not viable_designs:
            raise DesignError(
                f'No viable designs remain after {self.process_rosetta_metrics.__name__} data processing steps')

        self.log.debug(f'Viable designs with structures remaining after cleaning:\n\t{", ".join(viable_designs)}')

        # Take metrics for the pose_source
        # entity_energies = [0. for _ in self.pose.entities]
        # pose_source_residue_info = \
        #     {residue.index: {'complex': 0., 'bound': entity_energies.copy(), 'unbound': entity_energies.copy(),
        #                      'solv_complex': 0., 'solv_bound': entity_energies.copy(),
        #                      'solv_unbound': entity_energies.copy(), 'fsp': 0., 'cst': 0., 'hbond': 0}
        #      for entity in self.pose.entities for residue in entity.residues}
        # pose_source_id = self.pose_source.id
        # residue_info = {pose_source_id: pose_source_residue_info}

        # residue_info = {'energy': {'complex': 0., 'unbound': 0.}, 'type': None, 'hbond': 0}
        # residue_info.update(self.pose.rosetta_residue_processing(structure_design_scores))
        residue_info = self.pose.process_rosetta_residue_scores(scores)
        # Can't use residue_processing (clean) ^ in the case there is a design without metrics... columns not found!
        residue_info = metrics.process_residue_info(residue_info, hbonds=self.pose.rosetta_hbond_processing(scores))

        # Process mutational frequencies, H-bond, and Residue energy metrics to dataframe
        # which ends up with multi-index column with residue index as first (top) column index, metric as second index
        rosetta_residues_df = pd.concat({design: pd.DataFrame(info) for design, info in residue_info.items()}) \
            .unstack()

        return scores_df, rosetta_residues_df

    def rosetta_column_combinations(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics from combinations of metrics with variable integer number metric names

        Args:
            scores_df: A DataFrame with Rosetta based metrics that should be combined with other metrics to produce new
                summary metrics
        Returns:
            A per-design metric DataFrame where each index is the design id and the columns are design metrics,
        """
        scores_columns = scores_df.columns.tolist()
        self.log.debug(f'Metrics present: {scores_columns}')
        summation_pairs = \
            {'buried_unsatisfied_hbonds_unbound':
                list(filter(re.compile('buns[0-9]+_unbound$').match, scores_columns)),  # Rosetta
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
        # This is a nightmare as the column.map() turns all non-existing to np.nan need to fix upstream instead of
        # changing here
        # # Rename buns columns
        # # This isn't stable long term given mapping of interface number (sql) and entity number (Rosetta)
        # buns_columns = summation_pairs['buried_unsatisfied_hbonds_unbound']
        # scores_df.columns = scores_df.columns.map(dict((f'buns{idx}_unbound', f'buried_unsatisfied_hbonds_unbound{idx}')
        #                                                for idx in range(1, 1 + len(buns_columns))))\
        #     .fillna(scores_df.columns)
        # Add number_residues_interface for div_pairs and int_comp_similarity
        # scores_df['number_residues_interface'] = other_pose_metrics.pop('number_residues_interface')
        scores_df = metrics.columns_to_new_column(scores_df, metrics.rosetta_delta_pairs, mode='sub')
        scores_df = metrics.columns_to_new_column(scores_df, metrics.rosetta_division_pairs, mode='truediv')

        scores_df.drop(metrics.clean_up_intermediate_columns, axis=1, inplace=True, errors='ignore')
        repacking = scores_df.get('repacking')
        if repacking is not None:
            # Set interface_bound_activation_energy = np.nan where repacking is 0
            # Currently is -1 for True (Rosetta Filter quirk...)
            scores_df.loc[scores_df[repacking == 0].index, 'interface_bound_activation_energy'] = np.nan
            scores_df.drop('repacking', axis=1, inplace=True)

        return scores_df

    def process_rosetta_metrics(self):
        """From Rosetta based protocols, tally the resulting metrics and integrate with SymDesign metrics database"""
        self.log.debug(f'Found design scores in file: {self.scores_file}')  # Todo PoseJob(.path)
        design_scores = metrics.parse_rosetta_scorefile(self.scores_file)

        pose_design_scores = design_scores.pop(self.name, None)
        if pose_design_scores:
            # The pose_source is included in the calculations
            self.calculate_pose_metrics(scores=pose_design_scores)

        if not design_scores:
            # No design scores were found
            return

        # Find all designs files
        with self.job.db.session(expire_on_commit=False) as session:
            session.add(self)
            design_names = self.design_names
        design_files = self.get_design_files()  # Todo PoseJob(.path)
        scored_design_names = design_scores.keys()
        new_design_paths_to_process = []
        rosetta_provided_new_design_names = []
        existing_design_paths_to_process = []
        existing_design_indices = []
        # Collect designs that we have metrics on (possibly again)
        for idx, path in enumerate(design_files):
            file_name, ext = os.path.splitext(os.path.basename(path))
            if file_name in scored_design_names:
                try:
                    design_data_index = design_names.index(file_name)
                except ValueError:  # file_name not in design_names
                    # New, this file hasn't been processed
                    new_design_paths_to_process.append(path)
                    rosetta_provided_new_design_names.append(file_name)
                else:
                    existing_design_paths_to_process.append(path)
                    existing_design_indices.append(design_data_index)

        # Ensure that design_paths_to_process and self.current_designs have the same order
        design_paths_to_process = existing_design_paths_to_process + new_design_paths_to_process
        if not design_paths_to_process:
            return

        # # Find designs with scores but no structures
        # structure_design_scores = {}
        # for pose in designs:
        #     try:
        #         structure_design_scores[pose.name] = design_scores.pop(pose.name)
        #     except KeyError:  # Structure wasn't scored, we will remove this later
        #         pass

        # Process the parsed scores to scores_df, rosetta_residues_df
        scores_df, rosetta_residues_df = self.parse_rosetta_scores(design_scores)

        # Format DesignData
        # Get all existing
        design_data = self.designs  # This is loaded above at 'design_names = self.design_names'
        # Set current_designs to a fresh list
        self._current_designs = [design_data[idx] for idx in existing_design_indices]
        # Find parent info and remove from scores_df
        if putils.design_parent in scores_df:
            # Replace missing values with the pose_source DesignData
            # This is loaded above at 'design_names = self.design_names'
            parents = scores_df.pop(putils.design_parent)  # .fillna(self.pose_source)
            logger.critical("Setting parents functionality hasn't been tested. Proceed with caution")
            for design, parent in parents.items():
                if parent is np.nan:
                    parents[design] = self.pose_source
                try:
                    design_name_index = design_names.index(parent)
                except ValueError:  # This name isn't right
                    raise DesignError(
                        f"Couldn't find the design_parent for the design with name '{design}' and parent value of '"
                        f"{parent}'. The available parents are:\n\t{', '.join(design_names)}")
                else:
                    parents[design] = design_data[design_name_index]
        else:  # Assume this is an offspring of the pose
            parents = {provided_name: self.pose_source for provided_name in rosetta_provided_new_design_names}

        # Find protocol info and remove from scores_df
        if putils.protocol in scores_df:
            # Replace missing values with the pose_source DesignData
            protocol_s = scores_df.pop(putils.protocol).fillna('metrics')
            self.log.debug(f'Found "protocol_s" variable with dtype: {protocol_s.dtype}')
        else:
            protocol_s = {provided_name: 'metrics' for provided_name in rosetta_provided_new_design_names}

        # Process all desired files to Pose
        pose_kwargs = self.pose_kwargs
        design_poses = [Pose.from_file(file, **pose_kwargs) for file in design_paths_to_process]
        design_sequences = {pose.name: pose.sequence for pose in design_poses}
        # sequences_df = self.analyze_sequence_metrics_per_design(sequences=design_sequences)

        # The DataFrame.index needs to become design.id not design.name as it is here. Modify after processing
        residues_df = self.analyze_residue_metrics_per_design(designs=design_poses)
        # Join Rosetta per-residue DataFrame taking Structure analysis per-residue DataFrame index order
        residues_df = residues_df.join(rosetta_residues_df)

        designs_df = self.analyze_design_metrics_per_design(residues_df, design_poses)
        # Join Rosetta per-design DataFrame taking Structure analysis per-design DataFrame index order
        designs_df = designs_df.join(scores_df)

        # Finish calculation of Rosetta scores with included metrics
        designs_df = self.rosetta_column_combinations(designs_df)

        # Score using proteinmpnn only if the design was created by Rosetta
        if rosetta_provided_new_design_names:
            score_sequences = [design_sequences.pop(new_file_name)
                               for new_file_name in rosetta_provided_new_design_names]
            sequences_and_scores = self.pose.score_sequences(
                score_sequences, model_name=self.job.design.proteinmpnn_model)
            # Set each position that was parsed as "designable"
            # This includes packable residues from neighborhoods. How can we get only designable?
            # Right now, it is only the interface residues that go into Rosetta
            # Use simple reporting here until that changes...
            # Todo get residues_df['design_indices'] worked out with set up using sql.DesignProtocol?
            #  See self.analyze_pose_designs()
            design_residues = residues_df.loc[rosetta_provided_new_design_names,
                                              idx_slice[:, 'interface_residue']].to_numpy()
            sequences_and_scores.update({'design_indices': design_residues})
            mpnn_designs_df, mpnn_residues_df = \
                self.analyze_proteinmpnn_metrics(rosetta_provided_new_design_names, sequences_and_scores)
            # Join DataFrames
            designs_df = designs_df.join(mpnn_designs_df)
            residues_df = residues_df.join(mpnn_residues_df)
        # else:
        #     mpnn_residues_df = pd.DataFrame()
        #
        # # Calculate sequence metrics for the remaining sequences
        # sequences_df = self.analyze_sequence_metrics_per_design(sequences=design_sequences)
        # sequences_df = sequences_df.join(mpnn_residues_df)

        # Update the Pose.designs with DesignData for each of the new designs
        with self.job.db.session(expire_on_commit=False) as session:
            session.add(self)
            new_designs_data = self.update_design_data(
                design_parent=self.pose_source, number=len(new_design_paths_to_process))
            session.add_all(new_designs_data)
            # Generate ids for new entries
            session.flush()
            session.add_all(self.current_designs)

            # Add attribute to DesignData to save the provided_name and design design_parent
            for design_data, provided_name in zip(new_designs_data, rosetta_provided_new_design_names):
                design_data.design_parent = parents[provided_name]
                design_data.provided_name = provided_name

            designs_path = self.designs_path
            new_design_new_filenames = {data.provided_name: os.path.join(designs_path, f'{data.name}.pdb')
                                        for data in new_designs_data}

            # Update the Pose with the design protocols
            for design in self.current_designs:
                name_or_provided_name = getattr(design, 'provided_name', getattr(design, 'name'))
                protocol_kwargs = dict(design_id=design.id,
                                       job_id=self.job.id,
                                       protocol=protocol_s[name_or_provided_name],
                                       # temperature=temperatures[idx],)  # Todo from Rosetta?
                                       )
                new_filename = new_design_new_filenames.get(name_or_provided_name)
                if new_filename:
                    protocol_kwargs['file'] = new_filename
                    # Set the structure_path for this DesignData
                    design.structure_path = new_filename
                else:
                    protocol_kwargs['file'] = os.path.join(designs_path, f'{name_or_provided_name}.pdb')
                    if not design.structure_path:
                        design.structure_path = protocol_kwargs['file']
                design.protocols.append(sql.DesignProtocol(**protocol_kwargs))
            # else:  # Assume that no design was done and only metrics were acquired
            #     pass

            # This is all done in update_design_data
            # self.designs.append(new_designs_data)
            # # Flush the newly acquired DesignData and DesignProtocol to generate .id primary keys
            # self.job.current_session.flush()
            # new_design_ids = [design_data.id for design_data in new_designs_data]

            # Get the name/provided_name to design_id mapping
            design_name_to_id_map = {
                getattr(design, 'provided_name', getattr(design, 'name')): design.id
                for design in self.current_designs}

            # This call is redundant with the analyze_proteinmpnn_metrics(design_names, sequences_and_scores) above
            # Todo remove from above the sequences portion..? Commenting out below for now
            # designs_df = designs_df.join(self.analyze_design_metrics_per_residue(sequences_df))

            # Rename all designs and clean up resulting metrics for storage
            # In keeping with "unit of work", only rename once all data is processed incase we run into any errors
            designs_df.index = designs_df.index.map(design_name_to_id_map)
            residues_df.index = residues_df.index.map(design_name_to_id_map)
            if rosetta_provided_new_design_names:
                # Must move the entity_id to the columns for index.map to work
                entity_designs_df = self.analyze_design_entities_per_residue(mpnn_residues_df)
                entity_designs_df.reset_index(level=0, inplace=True)
                entity_designs_df.index = entity_designs_df.index.map(design_name_to_id_map)

            # Commit the newly acquired metrics to the database
            # First check if the files are situated correctly
            temp_count = count()
            temp_files_to_move = {}
            files_to_move = {}
            for filename, new_filename in zip(new_design_paths_to_process, new_design_new_filenames.values()):
                if filename == new_filename:
                    # These are the same file, proceed without processing
                    continue
                elif os.path.exists(filename):
                    if not os.path.exists(new_filename):
                        # Target file exists and nothing exists where it will be moved
                        files_to_move[filename] = new_filename
                    else:
                        # The new_filename already exists. Redirect the filename to a temporary file, then complete move
                        dir_, base = os.path.split(new_filename)
                        temp_filename = os.path.join(dir_, f'TEMP{next(temp_count)}')
                        temp_files_to_move[filename] = temp_filename
                        files_to_move[temp_filename] = new_filename
                else:  # filename doesn't exist
                    raise DesignError(
                        f"The specified file {filename} doesn't exist")

            # If so, proceed with insert, file rename and commit
            self.output_metrics(session, designs=designs_df)
            if rosetta_provided_new_design_names:
                sql.write_dataframe(session, entity_designs=entity_designs_df)
            output_residues = False
            if output_residues:
                self.output_metrics(session, residues=residues_df)
            else:  # Only save the 'design_residue' columns
                if rosetta_provided_new_design_names:
                    residues_df = residues_df.loc[:, idx_slice[:, sql.DesignResidues.design_residue.name]]
                    self.output_metrics(session, design_residues=residues_df)
            # Rename the incoming files to their prescribed names
            for filename, temp_filename in temp_files_to_move.items():
                shutil.move(filename, temp_filename)
            for filename, new_filename in files_to_move.items():
                shutil.move(filename, new_filename)

            # Commit all new data
            session.commit()

    def calculate_pose_metrics(self, **kwargs):
        """Collect Pose metrics using the reference Pose

        Keyword Args:
            scores: dict[str, str | int | float] = None - Parsed Pose scores from Rosetta output
            novel_interface: bool = True - Whether the pose interface is novel (i.e. docked) or from a bona-fide input
        """
        self.load_pose()
        # self.identify_interface()
        if not self.pose.fragment_info_by_entity_pair:
            self.generate_fragments(interface=True)

        def get_metrics():
            _metrics = self.pose.metrics  # Also calculates entity.metrics
            # Todo
            # # Gather the docking metrics if not acquired from Nanohedra
            # pose_residues_df = self.analyze_docked_metrics()
            # self.output_metrics(residues=pose_residues_df, pose_metrics=True)
            # Todo move into this mechanism PoseMetrics level calculations of the following:
            #  dock_collapse_*,
            #  dock_hydrophobicity,
            #  proteinmpnn_v_evolution_probability_cross_entropy_loss
            #
            idx = 1
            is_thermophilic = []
            for idx, (entity, data) in enumerate(zip(self.pose.entities, self.entity_data), idx):
                # Todo remove entity.thermophilicity once sql load more streamlined
                # is_thermophilic.append(1 if entity.thermophilicity else 0)
                is_thermophilic.append(entity.thermophilicity)
                # Save entity.metrics to db
                data.metrics = entity.metrics
            _metrics.pose_thermophilicity = sum(is_thermophilic) / idx

            return _metrics

        # Check if PoseMetrics have been captured
        if self.job.db:
            if self.metrics is None or self.job.force:
                with self.job.db.session(expire_on_commit=False) as session:
                    session.add(self)
                    metrics_ = get_metrics()
                    if self.metrics is None:
                        self.metrics = metrics_
                    else:  # Update existing self.metrics
                        current_metrics = self.metrics
                        for attr, value in metrics_.__dict__.items():
                            if attr == '_sa_instance_state':
                                continue
                            setattr(current_metrics, attr, value)
                    # Update the design_metrics for this Pose
                    self.calculate_pose_design_metrics(session, **kwargs)
                    session.commit()
        else:
            raise NotImplementedError(
                f"{self.calculate_pose_metrics.__name__} doesn't output anything yet when {type(self.job).__name__}.db"
                f"={self.job.db}")
            raise NotImplementedError(f"The reference=SymEntry.resulting_symmetry center_of_mass is needed as well")
            pose_df = self.pose.df  # Also performs entity.calculate_spatial_orientation_metrics()

            entity_dfs = []
            for entity in self.pose.entities:
                entity_s = pd.Series(**entity.calculate_spatial_orientation_metrics())
                entity_dfs.append(entity_s)

            # Stack the Series on the columns to turn into a dataframe where the metrics are rows and entity are columns
            entity_df = pd.concat(entity_dfs, keys=list(range(1, 1 + len(entity_dfs))), axis=1)

    def calculate_pose_design_metrics(self, session: Session, scores: dict[str, str | int | float] = None, novel_interface: bool = True):
        """Collects 'design' and per-residue metrics on the reference Pose

        Args:
            session: An open transaction within sqlalchemy
            scores: Parsed Pose scores from Rosetta output
            novel_interface: Whether the pose interface is novel (i.e. docked) or from a bona-fide input
        """
        pose = self.pose
        pose_length = pose.number_of_residues
        residue_indices = list(range(pose_length))

        designs = [pose]
        # Todo
        #  This function call is sort of accomplished by the following code
        #  residues_df = self.analyze_residue_metrics_per_design(designs=designs)
        #  Only differences are inclusion of interface_residue (^) and the novel_interface flag

        # novel_interface = False if not self.pose_source.protocols else True
        # if novel_interface:  # The input structure wasn't meant to be together, take the errat measurement as such
        #     source_errat = []
        #     for idx, entity in enumerate(self.pose.entities):
        #         _, oligomeric_errat = entity.assembly.errat(out_path=os.path.devnull)
        #         source_errat.append(oligomeric_errat[:entity.number_of_residues])
        #     # atomic_deviation[pose_source_name] = sum(source_errat_accuracy) / float(self.pose.number_of_entities)
        #     pose_source_errat = np.concatenate(source_errat)
        # else:
        #     # pose_assembly_minimally_contacting = self.pose.assembly_minimally_contacting
        #     # # atomic_deviation[pose_source_name], pose_per_residue_errat = \
        #     # _, pose_per_residue_errat = \
        #     #     pose_assembly_minimally_contacting.errat(out_path=os.path.devnull)
        #     # pose_source_errat = pose_per_residue_errat[:pose_length]
        #     # Get errat measurement
        #     # per_residue_data[pose_source_name].update(self.pose.per_residue_errat())
        #     pose_source_errat = self.pose.per_residue_errat()['errat_deviation']

        interface_residue_indices = np.array([[residue.index for residue in pose.interface_residues]])
        pose_name = pose.name
        # pose_source_id = self.pose_source.id
        # Collect reference Structure metrics
        per_residue_data = {pose_name:
                            {**pose.per_residue_interface_surface_area(),
                             **pose.per_residue_contact_order(),
                             # 'errat_deviation': pose_source_errat
                             **pose.per_residue_spatial_aggregation_propensity()
                             }}
        # Convert per_residue_data into a dataframe matching residues_df orientation
        residues_df = pd.concat({name: pd.DataFrame(data, index=residue_indices)
                                 for name, data in per_residue_data.items()}).unstack().swaplevel(0, 1, axis=1)
        # Construct interface residue array
        interface_residue_bool = np.zeros((len(designs), pose_length), dtype=int)
        for idx, interface_indices in enumerate(list(interface_residue_indices)):
            interface_residue_bool[idx, interface_indices] = 1
        interface_residue_df = pd.DataFrame(data=interface_residue_bool, index=residues_df.index,
                                            columns=pd.MultiIndex.from_product(
                                                (residue_indices, ['interface_residue'])))
        residues_df = residues_df.join(interface_residue_df)
        # Make buried surface area (bsa) columns, and residue classification
        residues_df = metrics.calculate_residue_buried_surface_area(residues_df)
        residues_df = metrics.classify_interface_residues(residues_df)
        # Todo same to here

        designs_df = self.analyze_design_metrics_per_design(residues_df, designs)

        # Score using proteinmpnn
        if self.job.use_proteinmpnn:
            sequences = [pose.sequence]  # Expected ASU sequence
            sequences_and_scores = pose.score_sequences(
                sequences, model_name=self.job.design.proteinmpnn_model)
            # design_residues = np.zeros((1, pose_length), dtype=bool)
            # design_residues[interface_residue_indices] = 1
            sequences_and_scores['design_indices'] = np.zeros((1, pose_length), dtype=bool)
            mpnn_designs_df, mpnn_residues_df = self.analyze_proteinmpnn_metrics([pose_name], sequences_and_scores)
            entity_designs_df = self.analyze_design_entities_per_residue(mpnn_residues_df)
            designs_df = designs_df.join(mpnn_designs_df)
            # sequences_df = self.analyze_sequence_metrics_per_design(sequences=[self.pose.sequence],
            #                                                         design_ids=[pose_source_id])
            # designs_df = designs_df.join(self.analyze_design_metrics_per_residue(sequences_df))
            # # Join per-residue like DataFrames
            # # Each of these could have different index/column, so we use concat to perform an outer merge
            # residues_df = residues_df.join([mpnn_residues_df, sequences_df])
            residues_df = residues_df.join(mpnn_residues_df)
        else:
            entity_designs_df = pd.DataFrame()

        if scores:
            # pose_source_id = self.pose_source.id
            scores_with_identifier = {pose_name: scores}
            scores_df, rosetta_residues_df = self.parse_rosetta_scores(scores_with_identifier)
            # Currently the metrics putils.protocol and putils.design_parent are not handle as this is the pose_source
            # and no protocols should have been run on this, nor should it have a parent. They will be removed when
            # output to the database
            scores_df = scores_df.join(metrics.sum_per_residue_metrics(rosetta_residues_df))
            # Finish calculation of Rosetta scores with included metrics
            designs_df = self.rosetta_column_combinations(designs_df.join(scores_df))
            residues_df = residues_df.join(rosetta_residues_df)

        # with self.job.db.session(expire_on_commit=False) as session:
        # Currently, self isn't producing any new information from database, session only important for metrics
        # session.add(self)
        # Correct the index of the DataFrame by changing from "name" to database ID
        name_to_id_map = {pose_name: self.pose_source.id}
        designs_df.index = designs_df.index.map(name_to_id_map)
        # Must move the entity_id to the columns for index.map to work
        entity_designs_df.reset_index(level=0, inplace=True)
        entity_designs_df.index = entity_designs_df.index.map(name_to_id_map)
        residues_df.index = residues_df.index.map(name_to_id_map)
        self.output_metrics(session, designs=designs_df)
        output_residues = False
        if output_residues:
            self.output_metrics(session, residues=residues_df)
        else:  # Only save the 'design_residue' columns
            try:
                residues_df = residues_df.loc[:, idx_slice[:, sql.DesignResidues.design_residue.name]]
            except KeyError:  # When self.job.use_proteinmpnn is false
                pass
            else:
                self.output_metrics(session, design_residues=residues_df)
        sql.write_dataframe(session, entity_designs=entity_designs_df)
        #     # Commit the newly acquired metrics
        #     session.commit()

    def analyze_alphafold_metrics(self, folding_scores: dict[str, [dict[str, np.ndarray]]], pose_length: int,
                                  model_type: str = None,
                                  interface_indices: tuple[Iterable[int], Iterable[int]] = False) \
            -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
        """From a set of folding metrics output by Alphafold (or possible other)

        Args:
            folding_scores: Metrics which may contain the following features as single metric or list or metrics
                {'predicted_aligned_error': [(n_residues, n_residues), ...]  # multimer/monomer_ptm
                 'plddt': [(n_residues,), ...]
                 'predicted_interface_template_modeling_score': [float, ...]  # multimer
                 'predicted_template_modeling_score': [float, ...]  # multimer/monomer_ptm
                 'rmsd_prediction_ensemble: [(number_of_models), ...]
                 }
            pose_length: The length of the scores to return for metrics with an array
            model_type: The type of model used during prediction
            interface_indices: The Residue instance of two sides of a predicted interface
        Returns:
            A tuple of DataFrame where each contains (
                A per-design metric DataFrame where each index is the design id and the columns are design metrics,
                A per-residue metric DataFrame where each index is the design id and the columns are
                    (residue index, residue metric)
            )
        """
        if not folding_scores:
            return None, None

        if interface_indices:
            if len(interface_indices) != 2:
                raise ValueError(
                    f'The argument "interface_indices" must contain a pair of indices for each side of an interfaces. '
                    f'Found the number of interfaces, {len(interface_indices)} != 2, the number expected')
            interface_indices1, interface_indices2 = interface_indices
            # Check if this interface is 2-fold symetric. If so, the interface_indices2 will be empty
            if not interface_indices2:
                interface_indices2 = interface_indices1
        else:
            interface_indices1 = interface_indices2 = slice(None)

        # def describe_metrics(array_like: Iterable[int] | np.ndarray) -> dict[str, float]:
        #     length = len(array_like)
        #     middle, remain = divmod(length, 2)
        #     if remain:  # Odd
        #         median_div = 1
        #     else:
        #         middle = slice(middle, 1 + middle)
        #         median_div = 2
        #
        #     return dict(
        #         min=min(array_like),
        #         max=max(array_like),
        #         mean=sum(array_like) / length,
        #         median=sum(list(sorted(array_like))[middle]) / median_div
        #     )

        score_types_mean = ['rmsd_prediction_ensemble']
        if 'multimer' in model_type:
            measure_pae = True
            score_types_mean += ['predicted_interface_template_modeling_score',
                                 'predicted_template_modeling_score']
        elif 'ptm' in model_type:
            measure_pae = True
            score_types_mean += ['predicted_template_modeling_score']
        else:
            measure_pae = False

        # number_models, number_of_residues = len(representative_plddt_per_model[0])
        # number_models = 1
        per_residue_data = {}
        design_scores = {}
        for design_name, scores in folding_scores.items():
            logger.debug(f'Found metrics with contents:\n{scores}')
            # This shouldn't fail as plddt should always be present
            array_scores = {}
            scalar_scores = {}
            for score_type, score in scores.items():
                # rmsd_metrics = describe_metrics(rmsds)
                if score_type in score_types_mean:
                    if isinstance(score, list):
                        score_len = len(score)
                        scalar_scores[score_type] = mean_ = sum(score) / score_len
                        if score_len > 1:
                            # Using the standard deviation of a sample
                            deviation = sqrt(sum([(score_-mean_) ** 2 for score_ in score]) / (score_len-1))
                        else:
                            deviation = 0.
                        scalar_scores[f'{score_type}_deviation'] = deviation
                    else:
                        scalar_scores[score_type] = score
                # Process 'predicted_aligned_error' when multimer/monomer_ptm. shape is (n_residues, n_residues)
                elif measure_pae and score_type == 'predicted_aligned_error':
                    if isinstance(score, list):
                        # First average over each residue, storing in a container
                        number_models = len(score)
                        pae: np.ndarray
                        pae, *other_pae = score
                        for pae_ in other_pae:
                            pae += pae_
                        if number_models > 1:
                            pae /= number_models
                        # pae_container = np.zeros((number_models, pose_length), dtype=np.float32)
                        # for idx, pae_ in enumerate(score):
                        #     pae_container[idx, :] = pae_.mean(axis=0)[:pose_length]
                        # # Next, average over each model
                        # pae = pae_container.mean(axis=0)
                    else:
                        pae = score
                    array_scores['predicted_aligned_error'] = pae.mean(axis=0)[:pose_length]

                    if interface_indices1:
                        # Index the resulting pae to get the error at the interface residues in particular
                        # interface_pae_means = [model_pae[interface_indices1][:, interface_indices2].mean()
                        #                        for model_pae in scores['predicted_aligned_error']]
                        # scalar_scores['predicted_aligned_error_interface'] = sum(interface_pae_means) / number_models
                        self.log.critical(f'Found interface_indices1: {interface_indices1}')
                        self.log.critical(f'Found interface_indices2: {interface_indices2}')
                        interface_pae = pae[interface_indices1][:, interface_indices2]
                        scalar_scores['predicted_aligned_error_interface'] = interface_pae.mean()
                        scalar_scores['predicted_aligned_error_interface_deviation'] = interface_pae.std()
                elif score_type == 'plddt':  # Todo combine with above
                    if isinstance(score, list):
                        number_models = len(score)
                        plddt: np.ndarray
                        plddt, *other_plddt = score
                        for plddt_ in other_plddt:
                            plddt += plddt_
                        if number_models > 1:
                            plddt /= number_models
                    else:
                        plddt = score
                    array_scores['plddt'] = plddt[:pose_length]

            logger.debug(f'Found scalar_scores with contents:\n{scalar_scores}')
            logger.debug(f'Found array_scores with contents:\n{array_scores}')

            per_residue_data[design_name] = array_scores
            design_scores[design_name] = scalar_scores

        designs_df = pd.DataFrame.from_dict(design_scores, orient='index')
        # residues_df = pd.DataFrame.from_dict(per_residue_data, orient='index')
        residue_indices = range(pose_length)
        residues_df = pd.concat({name: pd.DataFrame(data, index=residue_indices)
                                 for name, data in per_residue_data.items()}).unstack().swaplevel(0, 1, axis=1)
        designs_df = designs_df.join(metrics.sum_per_residue_metrics(residues_df))
        designs_df['plddt'] /= pose_length
        designs_df['plddt_deviation'] = residues_df.loc[:, idx_slice[:, 'plddt']].std(axis=1)
        designs_df['predicted_aligned_error'] /= pose_length
        designs_df['predicted_aligned_error_deviation'] = \
            residues_df.loc[:, idx_slice[:, 'predicted_aligned_error']].std(axis=1)

        return designs_df, residues_df

    def analyze_design_entities_per_residue(self, residues_df: pd.DataFrame) -> pd.DataFrame:
        """Gather sequence metrics on a per-entity basis and write to the database

        Args:
            residues_df: A per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
        Returns:
            A per-entity metric DataFrame where each index is a combination of (design_id, entity_id) and the columns
                are design metrics
        """
        residue_indices = list(range(self.pose.number_of_residues))
        mutation_df = residues_df.loc[:, idx_slice[:, 'mutation']]

        entity_designs = {}
        for entity_data, entity in zip(self.entity_data, self.pose.entities):
            number_mutations = \
                mutation_df.loc[:, idx_slice[residue_indices[entity.n_terminal_residue.index:
                                                             1 + entity.c_terminal_residue.index], :]]\
                .sum(axis=1)
            entity_designs[entity_data.id] = dict(
                number_mutations=number_mutations,
                percent_mutations=number_mutations / entity.number_of_residues)

        # Set up the DesignEntityMetrics dataframe for writing
        entity_designs_df = pd.concat([pd.DataFrame(data) for data in entity_designs.values()],
                                      keys=entity_designs.keys())
        design_ids = residues_df.index.tolist()
        entity_designs_df.index = entity_designs_df.index.set_levels(design_ids, level=-1)
        # entity_designs_df = pd.concat([pd.DataFrame(data) for data in entity_designs.values()])
        # design_name_to_id_map = dict(zip(entity_designs_df.index.get_level_values(-1), design_ids))
        # # mapped_index = entity_designs_df.index.map(design_name_to_id_map)
        # entity_designs_df.index = \
        #     pd.MultiIndex.from_tuples(zip(entity_designs.keys(),
        #                                   entity_designs_df.index.map(design_name_to_id_map).tolist()))
        # input(entity_designs_df)
        entity_designs_df.index = entity_designs_df.index.rename(
            [sql.DesignEntityMetrics.entity_id.name, sql.DesignEntityMetrics.design_id.name])
        # entity_designs_df.reset_index(level=-1, inplace=True)
        return entity_designs_df

    def analyze_design_metrics_per_design(self, residues_df: pd.DataFrame,
                                          designs: Iterable[Pose] | Iterable[AnyStr]) -> pd.DataFrame:
        """Take every design Model and perform design level structural analysis. Sums per-residue metrics (residues_df)

        Args:
            residues_df: The typical per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
            designs: The designs to perform analysis on. By default, fetches all available structures
        Returns:
            A per-design metric DataFrame where each index is the design id and the columns are design metrics
            Including metrics 'interface_area_total' and 'number_residues_interface' which are used in other analysis
                functions
        """
        #     designs_df: The typical per-design metric DataFrame where each index is the design id and the columns are
        #         design metrics
        # Find all designs files
        #   Todo fold these into Model(s) and attack metrics from Pose objects?
        # if designs is None:
        #     designs = []

        # Compute structural measurements for all designs
        pose_kwargs = self.pose_kwargs
        interface_local_density = {}
        # number_residues_interface = {}
        for pose in designs:
            try:
                pose_name = pose.name
            except AttributeError:  # This is likely a filepath
                pose = Pose.from_file(pose, **pose_kwargs)
                pose_name = pose.name
            # Must find interface residues before measure local_density
            pose.find_and_split_interface()
            # number_residues_interface[pose.name] = len(pose.interface_residues)
            interface_local_density[pose_name] = pose.local_density_interface()

        # designs_df = pd.Series(interface_local_density, index=residues_df.index,
        #                        name='interface_local_density').to_frame()
        designs_df = metrics.sum_per_residue_metrics(residues_df)
        interface_df = residues_df.loc[:, idx_slice[:, 'interface_residue']].droplevel(-1, axis=1)
        designs_df['spatial_aggregation_propensity_interface'] = \
            ((residues_df.loc[:, idx_slice[:, 'spatial_aggregation_propensity_unbound']].droplevel(-1, axis=1)
              - residues_df.loc[:, idx_slice[:, 'spatial_aggregation_propensity']].droplevel(-1, axis=1))
             * interface_df).sum(axis=1)
        # Divide by the number of interface residues
        designs_df['spatial_aggregation_propensity_interface'] /= interface_df.sum(axis=1)

        # Find the average for these summed design metrics
        pose_length = self.pose.number_of_residues
        designs_df['spatial_aggregation_propensity_unbound'] /= pose_length
        designs_df['spatial_aggregation_propensity'] /= pose_length
        # designs_df['number_residues_interface'] = pd.Series(number_residues_interface)
        designs_df['interface_local_density'] = pd.Series(interface_local_density)

        # self.load_pose()
        # # Make designs_df errat_deviation that takes into account the pose_source sequence errat_deviation
        # # Get per-residue errat scores from the residues_df
        # errat_df = residues_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)
        #
        # # Todo improve efficiency by using precomputed. Something like:
        # #  stmt = select(ResidueMetrics).where(ResidueMetrics.pose_id == self.id, ResidueMetrics.name == self.name)
        # #  rows = session.scalars(stmt)
        # #  row_dict = {row.index: row.errat_deviation for row in rows}
        # #  pd.Series(row_dict, name='errat_Deviation')
        # # pose_source_errat = errat_df.loc[self.pose.name, :]
        # pose_source_errat = self.pose.per_residue_errat()['errat_deviation']
        # # Include in errat_deviation if errat score is < 2 std devs and isn't 0 to begin with
        # source_errat_inclusion_boolean = \
        #     np.logical_and(pose_source_errat < metrics.errat_2_sigma, pose_source_errat != 0.)
        # # Find residues where designs deviate above wild-type errat scores
        # errat_sig_df = errat_df.sub(pose_source_errat, axis=1) > metrics.errat_1_sigma  # axis=1 is per-residue subtract
        # # Then select only those residues which are expressly important by the inclusion boolean
        # # This overwrites the metrics.sum_per_residue_metrics() value
        # designs_df['errat_deviation'] = (errat_sig_df.loc[:, source_errat_inclusion_boolean] * 1).sum(axis=1)

        # pose_df = self.pose.df
        # # Todo should each have the same number_residues_interface? need each design specifics
        # designs_df['number_residues_interface'] = pose_df['number_residues_interface']

        # Find the proportion of the residue surface area that is solvent accessible versus buried in the interface
        # if 'interface_area_total' in designs_df and 'area_total_complex' in designs_df:
        interface_bsa_df = designs_df['interface_area_total']
        designs_df['interface_area_to_residue_surface_ratio'] = \
            (interface_bsa_df / (interface_bsa_df + designs_df['area_total_complex']))

        # designs_df['interface_area_total'] = pose_df['interface_area_total']
        designs_df['number_of_interface_class'] = designs_df.loc[:, metrics.residue_classification].sum(axis=1)
        designs_df = metrics.columns_to_new_column(designs_df, metrics.division_pairs, mode='truediv')
        designs_df['interface_composition_similarity'] = \
            designs_df.apply(metrics.interface_composition_similarity, axis=1)
        designs_df.drop('number_of_interface_class', axis=1, inplace=True)

        return designs_df

    def analyze_design_metrics_per_residue(self, residues_df: pd.DataFrame) -> pd.DataFrame:
        # designs_df: pd.DataFrame = None
        """Take every design in the residues_df and perform design level statistical analysis

        Args:
            residues_df: The typical per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
        Returns:
            A per-design metric DataFrame where each index is the design id and the columns are design metrics
        """
        #     designs_df: The typical per-design metric DataFrame where each index is the design id and the columns are
        #         design metrics
        self.load_pose()
        # self.identify_interface()

        # Load fragment_profile into the analysis
        if not self.pose.fragment_info_by_entity_pair:
            self.generate_fragments(interface=True)
            self.pose.calculate_fragment_profile()

        # CAUTION: Assumes each structure is the same length
        pose_length = self.pose.number_of_residues
        # residue_indices = list(range(pose_length))

        designs_df = metrics.sum_per_residue_metrics(residues_df)

        pose_df = self.pose.df
        # Calculate mutational content
        # designs_df['number_mutations'] = mutation_df.sum(axis=1)
        designs_df['percent_mutations'] = designs_df['number_mutations'] / pose_length

        designs_df['sequence_loss_design_per_residue'] = designs_df['sequence_loss_design'] / pose_length
        # The per residue average loss compared to the design profile
        designs_df['sequence_loss_evolution_per_residue'] = designs_df['sequence_loss_evolution'] / pose_length
        # The per residue average loss compared to the evolution profile
        # Todo modify this when fragments come from elsewhere, not just interface
        designs_df['sequence_loss_fragment_per_residue'] = \
            designs_df['sequence_loss_fragment'] / pose_df['number_residues_interface_fragment_total']
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

        return designs_df

    def analyze_pose_designs(self, designs: Iterable[sql.DesignData] = None):
        """Retrieve all score information from a PoseJob and write results to .csv file

        Args:
            designs: The DesignData instances to perform analysis on. By default, fetches all PoseJob.designs
        Returns:
            Series containing summary metrics for all designs
        """
        with self.job.db.session(expire_on_commit=False) as session:
            if designs is None:
                session.add(self)
                if self.current_designs:
                    designs = self.current_designs
                else:
                    # Slice off the self.pose_source from these
                    self.current_designs = designs = self.designs[1:]

            if not designs:
                return  # There is nothing to analyze

            # Fetch data from the database
            # Get the name/provided_name to design_id mapping
            design_names = [design.name for design in designs]
            design_ids = [design.id for design in designs]
            design_name_to_id_map = dict(zip(design_names, design_ids))
            design_sequences = [design.sequence for design in designs]
            design_paths_to_process = [design.structure_path for design in designs]
            select_stmt = select((sql.DesignResidues.design_id, sql.DesignResidues.index, sql.DesignResidues.design_residue))\
                .where(sql.DesignResidues.design_id.in_(design_ids))
            index_col = sql.ResidueMetrics.index.name
            design_residue_df = \
                pd.DataFrame.from_records(session.execute(select_stmt).all(),
                                          columns=['id', index_col, sql.DesignResidues.design_residue.name])
        design_residue_df = design_residue_df.set_index(['id', index_col]).unstack()
        # Use simple reporting here until that changes...
        # interface_residue_indices = [residue.index for residue in self.pose.interface_residues]
        # pose_length = self.pose.number_of_residues
        # design_residues = np.zeros((len(designs), pose_length), dtype=bool)
        # design_residues[:, interface_residue_indices] = 1

        # Score using proteinmpnn
        # sequences_df = self.analyze_sequence_metrics_per_design(sequences=design_sequences)
        # sequences_and_scores = self.pose.score(
        sequences_and_scores = self.pose.score_sequences(
            design_sequences, model_name=self.job.design.proteinmpnn_model)
        sequences_and_scores['design_indices'] = design_residue_df.values

        mpnn_designs_df, mpnn_residues_df = self.analyze_proteinmpnn_metrics(design_names, sequences_and_scores)
        # The DataFrame.index needs to become the design.id not design.name. Modify after all processing
        entity_designs_df = self.analyze_design_entities_per_residue(mpnn_residues_df)

        # Process all desired files to Pose
        pose_kwargs = self.pose_kwargs
        designs_poses = \
            [Pose.from_file(file, **pose_kwargs) for file in design_paths_to_process if file is not None]

        if designs_poses:
            residues_df = self.analyze_residue_metrics_per_design(designs=designs_poses)
            designs_df = self.analyze_design_metrics_per_design(residues_df, designs_poses)
            # Join DataFrames
            designs_df = designs_df.join(mpnn_designs_df)
            residues_df = residues_df.join(mpnn_residues_df)
        else:
            designs_df = mpnn_designs_df
            residues_df = mpnn_residues_df

        # Each of these could have different index/column, so we use concat to perform an outer merge
        # residues_df = pd.concat([residues_df, mpnn_residues_df, sequences_df], axis=1) WORKED!!
        # residues_df = residues_df.join([mpnn_residues_df, sequences_df])
        # Todo should this "different index" be allowed? be possible
        #  residues_df = residues_df.join(rosetta_residues_df)

        # Rename all designs and clean up resulting metrics for storage
        # In keeping with "unit of work", only rename once all data is processed incase we run into any errors
        designs_df.index = designs_df.index.map(design_name_to_id_map)
        # Must move the entity_id to the columns for index.map to work
        entity_designs_df.reset_index(level=0, inplace=True)
        entity_designs_df.index = entity_designs_df.index.map(design_name_to_id_map)
        residues_df.index = residues_df.index.map(design_name_to_id_map)

        # Commit the newly acquired metrics to the database
        with self.job.db.session(expire_on_commit=False) as session:
            sql.write_dataframe(session, entity_designs=entity_designs_df)
            self.output_metrics(session, designs=designs_df)
            output_residues = False
            if output_residues:
                self.output_metrics(session, residues=residues_df)
            # This function doesn't generate any 'design_residue'
            # else:  # Only save the 'design_residue' columns
            #     residues_df = residues_df.loc[:, idx_slice[:, sql.DesignResidues.design_residue.name]]
            #     self.output_metrics(session, design_residues=residues_df)
            # Commit the newly acquired metrics
            session.commit()

    # def analyze_pose_metrics_per_design(self, residues_df: pd.DataFrame, designs_df: pd.DataFrame = None,
    #                                     designs: Iterable[Pose] | Iterable[AnyStr] = None) -> pd.Series:
    #     """Perform Pose level analysis on every design produced from this Pose
    #
    #     Args:
    #         residues_df: The typical per-residue metric DataFrame where each index is the design id and the columns are
    #             (residue index, residue metric)
    #         designs_df: The typical per-design metric DataFrame where each index is the design id and the columns are
    #             design metrics
    #         designs: The subsequent designs to perform analysis on
    #     Returns:
    #         Series containing summary metrics for all designs
    #     """
    #     self.load_pose()
    #
    #     return pose_s

    def analyze_proteinmpnn_metrics(self, design_ids: Sequence[str], sequences_and_scores: dict[str, np.array])\
            -> tuple[pd.DataFrame, pd.DataFrame]:
        #                      designs: Iterable[Pose] | Iterable[AnyStr] = None
        """Takes the sequences/scores ProteinMPNN features including 'design_indices', 'proteinmpnn_loss_complex',
        and 'proteinmpnn_loss_unbound' to format summary metrics. Performs sequence analysis with 'sequences' feature

        Args:
            design_ids: The associated design identifier for each corresponding entry in sequences_and_scores
            sequences_and_scores: The mapping of ProteinMPNN score type to it's corresponding data
        Returns:
            A tuple of DataFrame where each contains (
                A per-design metric DataFrame where each index is the design id and the columns are design metrics,
                A per-residue metric DataFrame where each index is the design id and the columns are
                    (residue index, residue metric)
            )
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

        # Construct residues_df
        sequences = sequences_and_scores.pop('sequences')
        sequences_and_scores['design_residue'] = sequences_and_scores.pop('design_indices')
        # If sequences_and_scores gets any other keys in it this isn't explicitly enough
        # proteinmpnn_data = {
        #     'design_residue': sequences_and_scores['design_indices'],
        #     'proteinmpnn_loss_complex': sequences_and_scores['proteinmpnn_loss_complex'],
        #     'proteinmpnn_loss_unbound': sequences_and_scores['proteinmpnn_loss_unbound']
        # }
        proteinmpnn_residue_info_df = \
            pd.concat([pd.DataFrame(data, index=design_ids,
                                    columns=pd.MultiIndex.from_product([residue_indices, [metric]]))
                       for metric, data in sequences_and_scores.items()], axis=1)

        # Incorporate residue, design, and sequence metrics on every designed Pose
        # Todo UPDATE These are now from a different collapse 'hydrophobicity' source, 'expanded'
        sequences_df = self.analyze_sequence_metrics_per_design(sequences=sequences, design_ids=design_ids)
        # Since no structure design completed, no residue_metrics is performed, but the pose source can be...
        # # The residues_df here has the wrong .index. It needs to become the design.id not design.name
        # residues_df = self.analyze_residue_metrics_per_design()  # designs=designs)
        # # Join each per-residue like dataframe
        # # Each of these can have difference index, so we use concat to perform an outer merge
        # residues_df = pd.concat([residues_df, sequences_df, proteinmpnn_residue_info_df], axis=1)
        # residues_df = pd.concat([sequences_df, proteinmpnn_residue_info_df], axis=1)
        residues_df = sequences_df.join(proteinmpnn_residue_info_df)

        designs_df = self.analyze_design_metrics_per_residue(residues_df)

        designed_df = residues_df.loc[:, idx_slice[:, 'design_residue']].droplevel(-1, axis=1)
        number_designed_residues_s = designed_df.sum(axis=1)

        # designs_df[putils.protocol] = 'proteinmpnn'
        designs_df['proteinmpnn_score_complex'] = designs_df['proteinmpnn_loss_complex'] / pose_length
        designs_df['proteinmpnn_score_unbound'] = designs_df['proteinmpnn_loss_unbound'] / pose_length
        designs_df['proteinmpnn_score_delta'] = \
            designs_df['proteinmpnn_score_complex'] - designs_df['proteinmpnn_score_unbound']
        # Find the mean of each of these using the boolean like array and the sum
        designs_df['proteinmpnn_score_complex_per_designed_residue'] = \
            (residues_df.loc[:, idx_slice[:, 'proteinmpnn_loss_complex']].droplevel(-1, axis=1)
             * designed_df).sum(axis=1)
        designs_df['proteinmpnn_score_complex_per_designed_residue'] /= number_designed_residues_s
        designs_df['proteinmpnn_score_unbound_per_designed_residue'] = \
            (residues_df.loc[:, idx_slice[:, 'proteinmpnn_loss_unbound']].droplevel(-1, axis=1)
             * designed_df).sum(axis=1)
        designs_df['proteinmpnn_score_unbound_per_designed_residue'] /= number_designed_residues_s
        designs_df['proteinmpnn_score_delta_per_designed_residue'] = \
            designs_df['proteinmpnn_score_complex_per_designed_residue'] \
            - designs_df['proteinmpnn_score_unbound_per_designed_residue']

        # Make an array the length of the designs and size of the pose to calculate the interface residues
        interface_residues = np.zeros((len(designed_df), pose_length), dtype=bool)
        interface_residues[:, [residue.index for residue in self.pose.interface_residues]] = 1
        number_interface_residues = interface_residues.sum(axis=1)
        designs_df['proteinmpnn_score_complex_per_interface_residue'] = \
            (residues_df.loc[:, idx_slice[:, 'proteinmpnn_loss_complex']].droplevel(-1, axis=1)
             * interface_residues).sum(axis=1)
        designs_df['proteinmpnn_score_complex_per_interface_residue'] /= number_interface_residues
        designs_df['proteinmpnn_score_unbound_per_interface_residue'] = \
            (residues_df.loc[:, idx_slice[:, 'proteinmpnn_loss_unbound']].droplevel(-1, axis=1)
             * interface_residues).sum(axis=1)
        designs_df['proteinmpnn_score_unbound_per_interface_residue'] /= number_interface_residues
        designs_df['proteinmpnn_score_delta_per_interface_residue'] = \
            designs_df['proteinmpnn_score_complex_per_interface_residue'] \
            - designs_df['proteinmpnn_score_unbound_per_interface_residue']

        # # Drop unused particular residues_df columns that have been summed
        # per_residue_drop_columns = per_residue_energy_states + energy_metric_names + per_residue_sasa_states \
        #                            + collapse_metrics + residue_classification \
        #                            + ['errat_deviation', 'hydrophobic_collapse', 'contact_order'] \
        #                            + ['hbond', 'evolution', 'fragment', 'type'] + ['surface', 'interior']
        # # Slice each of these columns as the first level residue number needs to be accounted for in MultiIndex
        # residues_df = residues_df.drop(
        #     list(residues_df.loc[:, idx_slice[:, per_residue_drop_columns]].columns),
        #     errors='ignore', axis=1)

        return designs_df, residues_df

    def analyze_residue_metrics_per_design(self, designs: Iterable[Pose] | Iterable[AnyStr]) -> pd.DataFrame:
        """Perform per-residue analysis on design Model instances

        Args:
            designs: The designs to analyze. The StructureBase.name attribute is used for naming DataFrame indices
        Returns:
            A per-residue metric DataFrame where each index is the design id and the columns are
                (residue index, residue metric)
        """
        # Compute structural measurements for all designs
        per_residue_data: dict[str, dict[str, Any]] = {}
        interface_residues = []
        pose_kwargs = self.pose_kwargs
        for pose in designs:
            try:
                name = pose.name
            except AttributeError:  # This is likely a filepath
                pose = Pose.from_file(pose, **pose_kwargs)
                name = pose.name
            # Get interface residues
            pose.find_and_split_interface()
            interface_residues.append([residue.index for residue in pose.interface_residues])
            per_residue_data[name] = {
                **pose.per_residue_interface_surface_area(),
                **pose.per_residue_contact_order(),
                # **pose.per_residue_errat()
                **pose.per_residue_spatial_aggregation_propensity()
            }

        self.load_pose()

        # CAUTION: Assumes each structure is the same length
        pose_length = self.pose.number_of_residues
        residue_indices = list(range(pose_length))
        # residue_numbers = [residue.number for residue in self.pose.residues]
        # design_residue_indices = [residue.index for residue in self.pose.design_residues]
        # interface_residue_indices = [residue.index for residue in self.pose.interface_residues]

        # Convert per_residue_data into a dataframe matching residues_df orientation
        residues_df = pd.concat({name: pd.DataFrame(data, index=residue_indices)
                                for name, data in per_residue_data.items()}).unstack().swaplevel(0, 1, axis=1)
        # Construct interface residue array
        interface_residue_bool = np.zeros((len(designs), pose_length), dtype=int)
        for idx, interface_indices in enumerate(interface_residues):
            interface_residue_bool[idx, interface_indices] = 1
        interface_residue_df = pd.DataFrame(data=interface_residue_bool, index=residues_df.index,
                                            columns=pd.MultiIndex.from_product((residue_indices,
                                                                                ['interface_residue'])))
        residues_df = residues_df.join(interface_residue_df)
        # Make buried surface area (bsa) columns, and classify residue types
        residues_df = metrics.calculate_residue_buried_surface_area(residues_df)
        return metrics.classify_interface_residues(residues_df)

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
                raise ValueError(
                    f"Can't perform {self.analyze_sequence_metrics_per_design.__name__} without argument "
                    "'design_ids' when 'sequences' isn't a dictionary")
            # All sequences must be string for Biopython
            if isinstance(sequences, np.ndarray):
                sequences = [''.join(sequence) for sequence in sequences.tolist()]  # [self.pose.sequence] +
            elif isinstance(sequences, Sequence):
                sequences = [''.join(sequence) for sequence in sequences]  # [self.pose.sequence] +
            else:
                design_sequences = design_ids = sequences = None
                raise ValueError(
                    f"Can't perform {self.analyze_sequence_metrics_per_design.__name__} with argument "
                    f"'sequences' as a {type(sequences).__name__}. Pass 'sequences' as a Sequence[str]")
        if len(design_ids) != len(sequences):
            raise ValueError(
                f"The length of the design_ids ({len(design_ids)}) != sequences ({len(sequences)})")

        # Create numeric sequence types
        number_of_sequences = len(sequences)
        numeric_sequences = sequences_to_numeric(sequences)
        torch_numeric_sequences = torch.from_numpy(numeric_sequences)

        pose_length = self.pose.number_of_residues
        residue_indices = list(range(pose_length))
        nan_blank_data = np.tile(list(repeat(np.nan, pose_length)), (number_of_sequences, 1))

        # Make requisite profiles
        self.set_up_evolutionary_profile(warn_metrics=True)

        # Try to add each of the profile types in observed_types to profile_background
        profile_background = {}
        if self.measure_evolution:
            profile_background['evolution'] = evolutionary_profile_array = \
                pssm_as_array(self.pose.evolutionary_profile)
            batch_evolutionary_profile = np.tile(evolutionary_profile_array, (number_of_sequences, 1, 1))
            # torch_log_evolutionary_profile = torch.from_numpy(np.log(batch_evolutionary_profile))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                # np.log causes -inf at 0, so they are corrected to a 'large' number
                corrected_evol_array = np.nan_to_num(np.log(batch_evolutionary_profile), copy=False,
                                                     nan=np.nan, neginf=metrics.zero_probability_evol_value)
            torch_log_evolutionary_profile = torch.from_numpy(corrected_evol_array)
            per_residue_evolutionary_profile_loss = \
                resources.ml.sequence_nllloss(torch_numeric_sequences, torch_log_evolutionary_profile)
        else:
            # Because self.pose.calculate_profile() is used below, need to ensure there is a null_profile attached
            if not self.pose.evolutionary_profile:
                self.pose.evolutionary_profile = self.pose.create_null_profile()
            # per_residue_evolutionary_profile_loss = per_residue_design_profile_loss = nan_blank_data
            per_residue_evolutionary_profile_loss = nan_blank_data

        # Load fragment_profile into the analysis
        # if self.job.design.term_constraint and not self.pose.fragment_queries:
        # if self.job.design.term_constraint:
        if not self.pose.fragment_info_by_entity_pair:
            self.generate_fragments(interface=True)
        if not self.pose.fragment_profile:
            self.pose.calculate_fragment_profile()
        profile_background['fragment'] = fragment_profile_array = self.pose.fragment_profile.as_array()
        batch_fragment_profile = np.tile(fragment_profile_array, (number_of_sequences, 1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # np.log causes -inf at 0, so they are corrected to a 'large' number
            corrected_frag_array = np.nan_to_num(np.log(batch_fragment_profile), copy=False,
                                                 nan=np.nan, neginf=metrics.zero_probability_frag_value)
        per_residue_fragment_profile_loss = \
            resources.ml.sequence_nllloss(torch_numeric_sequences, torch.from_numpy(corrected_frag_array))
        # else:
        #     per_residue_fragment_profile_loss = nan_blank_data

        # Set up "design" profile
        self.pose.calculate_profile()
        profile_background['design'] = design_profile_array = pssm_as_array(self.pose.profile)
        batch_design_profile = np.tile(design_profile_array, (number_of_sequences, 1, 1))
        with warnings.catch_warnings():
            # np.log causes -inf at 0, so they are corrected to a 'large' number
            warnings.simplefilter('ignore', category=RuntimeWarning)
            log_batch_profile = np.log(batch_design_profile)
        if self.pose.fragment_info_by_entity_pair:
            corrected_design_array = np.nan_to_num(log_batch_profile, copy=False,
                                                   nan=np.nan, neginf=metrics.zero_probability_evol_value)
            torch_log_design_profile = torch.from_numpy(corrected_design_array)
        else:
            torch_log_design_profile = torch.from_numpy(log_batch_profile)
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
        # Ensure sequences are strings as MultipleSequenceAlignment.from_dictionary() SeqRecord requires ids as strings
        design_names = [str(id_) for id_ in design_ids]
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

        # INSTEAD OF USING BELOW, split Pose.MultipleSequenceAlignment at entity.chain_break...
        # entity_alignments = \
        #     {idx: MultipleSequenceAlignment.from_dictionary(designed_sequences)
        #      for idx, designed_sequences in entity_sequences.items()}
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
                entity.msa = entity_alignment
                dca_design_residue_energies = entity.direct_coupling_analysis()
                dca_design_residues_concat.append(dca_design_residue_energies)
                # dca_background_energies.append(dca_background_energies.sum(axis=1))
                # dca_design_energies.append(dca_design_energies.sum(axis=1))
                dca_background_energies[entity] = dca_background_residue_energies.sum(axis=1)  # Turns data to 1D
                dca_design_energies[entity] = dca_design_residue_energies.sum(axis=1)
            except AttributeError:
                self.log.debug(f"For {entity.name}, DCA analysis couldn't be performed. "
                               f"Missing required parameter files")
                dca_succeed = False
                break

        if dca_succeed:
            # concatenate along columns, adding residue index to column, design name to row
            dca_concatenated_df = pd.DataFrame(np.concatenate(dca_design_residues_concat, axis=1),
                                               index=design_ids, columns=residue_indices)
            # dca_concatenated_df.columns = pd.MultiIndex.from_product([dca_concatenated_df.columns, ['dca_energy']])
            dca_concatenated_df = pd.concat([dca_concatenated_df], keys=['dca_energy'], axis=1).swaplevel(0, 1, axis=1)
            # Merge with residues_df
            residues_df = pd.merge(residues_df, dca_concatenated_df, left_index=True, right_index=True)

        return residues_df

    def interface_design_analysis(self, designs: Iterable[Pose] | Iterable[AnyStr] = None) -> pd.Series:
        """Retrieve all score information from a PoseJob and write results to .csv file

        Args:
            designs: The designs to perform analysis on. By default, fetches all available structures
        Returns:
            Series containing summary metrics for all designs
        """
        raise NotImplementedError(
            'This is in place for backward compatibility but is currently not debugged. Please'
            f' consider using the module "{flags.process_rosetta_metrics}" instead, or debug '
            f'{self.interface_design_analysis.__name__} from the module "{flags.analysis}"')
        self.load_pose()
        # self.identify_interface()

        # Load fragment_profile into the analysis
        if not self.pose.fragment_info_by_entity_pair:
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
            pose_kwargs = self.pose_kwargs
            designs = [Pose.from_file(file, **pose_kwargs) for file in self.get_design_files()]  # Todo PoseJob(.path)

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
        pose_name = self.name
        residue_info = {pose_name: pose_source_residue_info}

        # Gather miscellaneous pose specific metrics
        # other_pose_metrics = self.pose.calculate_metrics()
        # Create metrics for the pose_source
        empty_source = dict(
            # **other_pose_metrics,
            buried_unsatisfied_hbonds_complex=0,
            # buried_unsatisfied_hbonds_unbound=0,
            contact_count=0,
            favor_residue_energy=0,
            interaction_energy_complex=0,
            interaction_energy_per_residue=0,
            interface_separation=0,
            number_hbonds=0,
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
            # source_df['buried_unsatisfied_hbonds_complex'] = 0
            # # source_df['buried_unsatisfied_hbonds_unbound'] = 0
            # source_df['contact_count'] = 0
            # source_df['favor_residue_energy'] = 0
            # # Used in sum_per_residue_df
            # # source_df['interface_energy_complex'] = 0
            # source_df['interaction_energy_complex'] = 0
            # source_df['interaction_energy_per_residue'] = \
            #     source_df['interaction_energy_complex'] / len(self.pose.interface_residues)
            # source_df['interface_separation'] = 0
            # source_df['number_hbonds'] = 0
            # source_df['rmsd_complex'] = 0
            # source_df['rosetta_reference_energy'] = 0
            # source_df['shape_complementarity'] = 0
            design_scores = metrics.parse_rosetta_scorefile(self.scores_file)
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
            for column in scores_df.columns.tolist():
                if 'res_' in column:  # if column.startswith('per_res_'):
                    per_res_columns.append(column)
                # elif column.startswith('hbonds_res_selection'):
                #     hbonds_columns.append(column)

            # Check proper input
            metric_set = metrics.rosetta_required.difference(set(scores_df.columns))
            if metric_set:
                raise DesignError(
                    f'Missing required metrics: "{", ".join(metric_set)}"')

            # Remove unnecessary (old scores) as well as Rosetta pose score terms besides ref (has been renamed above)
            # Todo learn know how to produce Rosetta score terms in output score file. Not in FastRelax...
            remove_columns = metrics.rosetta_terms + metrics.unnecessary + per_res_columns
            # Todo remove dirty when columns are correct (after P432)
            #  and column tabulation precedes residue/hbond_processing
            # residue_info = {'energy': {'complex': 0., 'unbound': 0.}, 'type': None, 'hbond': 0}
            residue_info.update(self.pose.process_rosetta_residue_scores(structure_design_scores))
            # Can't use residue_processing (clean) ^ in the case there is a design without metrics... columns not found!
            residue_info = metrics.process_residue_info(
                residue_info, hbonds=self.pose.rosetta_hbond_processing(structure_design_scores))
            # residue_info = metrics.incorporate_sequence_info(residue_info, pose_sequences)

            # Drop designs where required data isn't present
            # Format protocol columns
            if putils.protocol in scores_df.columns:
                missing_group_indices = scores_df[putils.protocol].isna()
                # Todo remove not DEV
                scout_indices = [idx for idx in scores_df[missing_group_indices].index if 'scout' in idx]
                scores_df.loc[scout_indices, putils.protocol] = putils.scout
                structure_bkgnd_indices = [idx for idx in scores_df[missing_group_indices].index
                                           if 'no_constraint' in idx]
                scores_df.loc[structure_bkgnd_indices, putils.protocol] = putils.structure_background
                # Todo Done remove
                # protocol_s.replace({'combo_profile': putils.design_profile}, inplace=True)  # Ensure proper name

                scores_df.drop(missing_group_indices, axis=0, inplace=True, errors='ignore')
                # protocol_s.drop(missing_group_indices, inplace=True, errors='ignore')

            viable_designs = scores_df.index.tolist()
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
            # source_df['buried_unsatisfied_hbonds_complex'] = 0
            # # source_df['buried_unsatisfied_hbonds_unbound'] = 0
            # source_df['contact_count'] = 0
            # source_df['favor_residue_energy'] = 0
            # # source_df['interface_energy_complex'] = 0
            # source_df['interaction_energy_complex'] = 0
            # source_df['interaction_energy_per_residue'] = \
            #     source_df['interaction_energy_complex'] / len(self.pose.interface_residues)
            # source_df['interface_separation'] = 0
            # source_df['number_hbonds'] = 0
            # source_df['rmsd_complex'] = 0
            # source_df['rosetta_reference_energy'] = 0
            # source_df['shape_complementarity'] = 0
            scores_df = pd.DataFrame.from_dict({pose.name: {putils.protocol: job_key} for pose in designs},
                                               orient='index')
            # # Fill in all the missing values with that of the default pose_source
            # scores_df = pd.concat([source_df, scores_df]).fillna(method='ffill')

            remove_columns = metrics.rosetta_terms + metrics.unnecessary
            residue_info.update({struct_name: pose_source_residue_info for struct_name in scores_df.index.tolist()})
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
        # if design_was_performed:  # The input structure was not meant to be together, treat as such
        #     source_errat = []
        #     for idx, entity in enumerate(self.pose.entities):
        #         # Replace 'errat_deviation' measurement with uncomplexed entities
        #         # oligomer_errat_accuracy, oligomeric_errat = entity_oligomer.errat(out_path=os.path.devnull)
        #         # source_errat_accuracy.append(oligomer_errat_accuracy)
        #         _, oligomeric_errat = entity.assembly.errat(out_path=os.path.devnull)
        #         source_errat.append(oligomeric_errat[:entity.number_of_residues])
        #     # atomic_deviation[pose_name] = sum(source_errat_accuracy) / float(number_of_entities)
        #     pose_source_errat = np.concatenate(source_errat)
        # else:
        #     # Get errat measurement
        #     # per_residue_data[pose_name].update(self.pose.per_residue_errat())
        #     pose_source_errat = self.pose.per_residue_errat()['errat_deviation']
        #
        # per_residue_data[pose_name]['errat_deviation'] = pose_source_errat

        # Compute structural measurements for all designs
        interface_local_density = {pose_name: self.pose.local_density_interface()}
        for pose in designs:  # Takes 1-2 seconds for Structure -> assembly -> errat
            # Must find interface residues before measure local_density
            pose.find_and_split_interface()
            per_residue_data[pose.name] = pose.per_residue_interface_surface_area()
            # Get errat measurement
            # per_residue_data[pose.name].update(pose.per_residue_errat())
            # Todo remove Rosetta
            #  This is a measurement of interface_connectivity like from Rosetta
            interface_local_density[pose.name] = pose.local_density_interface()

        scores_df['interface_local_density'] = pd.Series(interface_local_density)

        # Load profiles of interest into the analysis
        self.set_up_evolutionary_profile(warn_metrics=True)

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
        rosetta_residues_df = pd.concat({design: pd.DataFrame(info) for design, info in residue_info.items()}).unstack()
        # returns multi-index column with residue number as first (top) column index, metric as second index
        # during residues_df unstack, all residues with missing dicts are copied as nan
        # Merge interface design specific residue metrics with total per residue metrics
        # residues_df = pd.merge(residues_df, rosetta_residues_df, left_index=True, right_index=True)

        # Join each residues_df like dataframe
        # Each of these can have difference index, so we use concat to perform an outer merge
        residues_df = pd.concat([residues_df, sequences_df, rosetta_residues_df], axis=1)
        # # Join rosetta_residues_df and sequence metrics
        # residues_df = residues_df.join([rosetta_residues_df, sequences_df])

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

        # INSTEAD OF USING BELOW, split Pose.MultipleSequenceAlignment at entity.chain_break...
        # entity_alignments = \
        #     {idx: MultipleSequenceAlignment.from_dictionary(designed_sequences)
        #      for idx, designed_sequences in entity_sequences.items()}
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
                # Todo
                #  Instead of below, split Pose.MultipleSequenceAlignment at entity.chain_break...
                entity_alignment = MultipleSequenceAlignment.from_dictionary(entity_sequences[idx])
                entity.msa = entity_alignment
                dca_design_residue_energies = entity.direct_coupling_analysis()
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
        residues_df = metrics.calculate_residue_buried_surface_area(residues_df)
        residues_df = metrics.classify_interface_residues(residues_df)

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

        # # Make scores_df errat_deviation that takes into account the pose_source sequence errat_deviation
        # # Include in errat_deviation if errat score is < 2 std devs and isn't 0 to begin with
        # # Get per-residue errat scores from the residues_df
        # errat_df = residues_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)
        #
        # pose_source_errat = errat_df.loc[pose_name, :]
        # source_errat_inclusion_boolean = \
        #     np.logical_and(pose_source_errat < metrics.errat_2_sigma, pose_source_errat != 0.)
        # # Find residues where designs deviate above wild-type errat scores
        # errat_sig_df = errat_df.sub(pose_source_errat, axis=1) > metrics.errat_1_sigma  # axis=1 is per-residue subtract
        # # Then select only those residues which are expressly important by the inclusion boolean
        # # This overwrites the metrics.sum_per_residue_metrics() value
        # scores_df['errat_deviation'] = (errat_sig_df.loc[:, source_errat_inclusion_boolean] * 1).sum(axis=1)

        # Calculate mutational content
        mutation_df = residues_df.loc[:, idx_slice[:, 'mutation']]
        # scores_df['number_mutations'] = mutation_df.sum(axis=1)
        scores_df['percent_mutations'] = scores_df['number_mutations'] / pose_length

        idx = 1
        # prior_slice = 0
        for idx, entity in enumerate(self.pose.entities, idx):
            # entity_n_terminal_residue_index = entity.n_terminal_residue.index
            # entity_c_terminal_residue_index = entity.c_terminal_residue.index
            scores_df[f'entity{idx}_number_mutations'] = \
                mutation_df.loc[:, idx_slice[residue_indices[entity.n_terminal_residue.index:  # prior_slice
                                                             1 + entity.c_terminal_residue.index], :]].sum(axis=1)
            # prior_slice = entity_c_terminal_residue_index
            scores_df[f'entity{idx}_percent_mutations'] = \
                scores_df[f'entity{idx}_number_mutations'] / entity.number_of_residues

        # scores_df['number_mutations'] = \
        #     pd.Series({design: len(mutations) for design, mutations in all_mutations.items()})
        # scores_df['percent_mutations'] = scores_df['number_mutations'] / pose_length

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
        scores_columns = scores_df.columns.tolist()
        self.log.debug(f'Metrics present: {scores_columns}')
        # sum columns using list[0] + list[1] + list[n]
        # residues_df = residues_df.drop([column
        #                                 for columns in [complex_df.columns, bound_df.columns, unbound_df.columns,
        #                                                 solvation_complex_df.columns, solvation_bound_df.columns,
        #                                                 solvation_unbound_df.columns]
        #                                 for column in columns], axis=1)
        summation_pairs = \
            {'buried_unsatisfied_hbonds_unbound': list(filter(re.compile('buns[0-9]+_unbound$').match, scores_columns)),  # Rosetta
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
        # Add number_residues_interface for div_pairs and int_comp_similarity
        # scores_df['number_residues_interface'] = other_pose_metrics.pop('number_residues_interface')
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
        # final_trajectory_indices = designs_df.index.tolist() + unique_protocols + [mean]
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
            self.log.info(f"Missing background protocol '{putils.structure_background}'. No protocol significance "
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
            designed_sequence_modifications = residues_df.loc[:, idx_slice[:, 'type']].sum(axis=1).tolist()
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
            protocol_sig_s = pd.concat([pvalue_df.loc[[pair], :].squeeze() for pair in pvalue_df.index.tolist()],
                                       keys=[tuple(pair) for pair in pvalue_df.index.tolist()])
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
        with self.job.db.session(expire_on_commit=False) as session:
            self.output_metrics(session, designs=designs_df)
            output_residues = False
            if output_residues:
                self.output_metrics(session, residues=residues_df)
            else:  # Only save the 'design_residue' columns
                residues_df = residues_df.loc[:, idx_slice[:, sql.DesignResidues.design_residue.name]]
                self.output_metrics(session, design_residues=residues_df)
            # Commit the newly acquired metrics
            session.commit()

        write_sequences(pose_sequences, file_name=self.designed_sequences_file)

        # Create figures
        if self.job.figures:  # For plotting collapse profile, errat data, contact order
            interface_residue_indices = [residue.index for residue in self.pose.interface_residues]
            # Plot: Format the collapse data with residues as index and each design as column
            # collapse_graph_df = pd.DataFrame(per_residue_data['hydrophobic_collapse'])
            collapse_graph_df = residues_df.loc[:, idx_slice[:, 'hydrophobic_collapse']].droplevel(-1, axis=1)
            reference_collapse = [entity.hydrophobic_collapse() for entity in self.pose.entities]
            reference_collapse_concatenated_s = \
                pd.Series(np.concatenate(reference_collapse), name=putils.reference)
            collapse_graph_df[putils.reference] = reference_collapse_concatenated_s
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

            # sharex=True allows the axis to be shared across plots
            # collapse_ax, contact_ax, errat_ax = fig.subplots(3, 1, sharex=True)
            # collapse_ax, errat_ax = fig.subplots(2, 1, sharex=True)
            collapse_ax = fig.subplots(1, 1, sharex=True)
            # Add the contact order to the same plot with a separate axis
            contact_ax = collapse_ax.twinx()
            contact_order_df = residues_df.loc[pose_name, idx_slice[:, 'contact_order']].droplevel(-1, axis=1)
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

            # # Plot: Errat Accuracy
            # # errat_graph_df = pd.DataFrame(per_residue_data['errat_deviation'])
            # # errat_graph_df = residues_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)
            # # errat_graph_df = errat_df
            # # wt_errat_concatenated_s = pd.Series(np.concatenate(list(source_errat.values())), name='clean_asu')
            # # errat_graph_df[pose_name] = pose_source_errat
            # # errat_graph_df.columns += 1  # offset index to residue numbering
            # errat_df.sort_index(axis=0, inplace=True)
            # # errat_ax = errat_graph_df.plot.line(legend=False, ax=errat_ax, figsize=figure_aspect_ratio)
            # # errat_ax = errat_graph_df.plot.line(legend=False, ax=errat_ax)
            # # errat_ax = errat_graph_df.plot.line(ax=errat_ax)
            # errat_ax.plot(errat_df.T.values, label=collapse_graph_df.index)
            # errat_ax.xaxis.set_major_locator(MultipleLocator(20))
            # errat_ax.xaxis.set_major_formatter('{x:.0f}')
            # # For the minor ticks, use no labels; default NullFormatter.
            # errat_ax.xaxis.set_minor_locator(MultipleLocator(5))
            # # errat_ax.set_xlim(0, pose_length)
            # errat_ax.set_ylim(0, None)
            # # graph_errat = sns.relplot(data=errat_graph_df, kind='line')  # x='Residue Number'
            # # Plot the chain break(s) and design residues
            # # labels = [fill(column, legend_fill_value) for column in errat_graph_df.columns]
            # # errat_ax.legend(labels, loc='lower left', bbox_to_anchor=(0., 1.))
            # # errat_ax.legend(loc='lower center', bbox_to_anchor=(0., 1.))
            # errat_ax.vlines(self.pose.chain_breaks, 0, 1, transform=errat_ax.get_xaxis_transform(),
            #                 label='Entity Breaks', colors='#cccccc')  # , grey)
            # errat_ax.vlines(interface_residue_indices, 0, 0.05, transform=errat_ax.get_xaxis_transform(),
            #                 label='Design Residues', colors='#f89938', lw=2)  # , orange)
            # # Plot horizontal significance
            # errat_ax.hlines([metrics.errat_2_sigma], 0, 1, transform=errat_ax.get_yaxis_transform(),
            #                 label='Significant Error', colors='#fc554f', linestyle='dotted')  # tomato
            # errat_ax.set_xlabel('Residue Number')
            # errat_ax.set_ylabel('Errat Score')
            # # errat_ax.autoscale(True)
            # # errat_ax.figure.tight_layout()
            # # errat_ax.figure.savefig(os.path.join(self.data_path, 'errat.png'))
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


class PoseJob(PoseProtocol):
    pass


def insert_pose_jobs(session: Session, pose_jobs: Iterable[PoseJob], project: str) -> list[PoseJob]:
    """Add PoseJobs to the database accounting for existing entries

    Args:
        session: An open sqlalchemy session
        pose_jobs: The PoseJob instances which should be inserted
        project: The name of the project which the `pose_jobs` belong
    Raises:
        sqlalchemy.exc.IntegrityError
    Returns:
        The PoseJob instances that are already present
    """
    error_count = count(1)
    # logger.debug(f"Start: {getattr(pose_jobs[0], 'id', 'No id')}")
    while True:
        pose_name_to_pose_jobs = {pose_job.name: pose_job for pose_job in pose_jobs}
        session.add_all(pose_jobs)
        # logger.debug(f"ADD ALL: {getattr(pose_jobs[0], 'id', 'No id')}")
        try:  # Flush PoseJobs to the current session to generate ids
            session.flush()
        except IntegrityError:  # PoseJob.project/.name already inserted
            # logger.debug(f"FLUSH: {getattr(pose_jobs[0], 'id', 'No id')}")
            session.rollback()
            # logger.debug(f"ROLLBACK: {getattr(pose_jobs[0], 'id', 'No id')}")
            number_flush_attempts = next(error_count)
            logger.debug(f'rollback() #{number_flush_attempts}')

            # Find the actual pose_jobs_to_commit and place in session
            pose_names = list(pose_name_to_pose_jobs.keys())
            fetch_jobs_stmt = select(PoseJob).where(PoseJob.project.is_(project)) \
                .where(PoseJob.name.in_(pose_names))
            existing_pose_jobs = session.scalars(fetch_jobs_stmt).all()
            # Note: Values are sorted by alphanumerical, not numerical
            # ex, design 11 is processed before design 2
            existing_pose_names = {pose_job_.name for pose_job_ in existing_pose_jobs}
            new_pose_names = set(pose_names).difference(existing_pose_names)
            if not new_pose_names:  # No new PoseJobs
                return existing_pose_jobs
            else:
                pose_jobs = [pose_name_to_pose_jobs[pose_name] for pose_name in new_pose_names]
                # Set each of the primary id to None so they are updated again during the .flush()
                for pose_job in pose_jobs:
                    pose_job.id = None
                # logger.debug(f"RESET: {getattr(pose_jobs[0], 'id', 'No id')}")

                if number_flush_attempts == 1:
                    logger.debug(
                        f'From {len(pose_names)} pose_jobs:\n{sorted(pose_names)}\n'
                        f'Found {len(new_pose_names)} new_pose_jobs:\n{sorted(new_pose_names)}\n'
                        f'Removing existing docked poses from output: {", ".join(existing_pose_names)}')
                elif number_flush_attempts == 2:
                    # Try to attach existing protein_metadata.entity_id
                    # possibly_new_uniprot_to_prot_metadata = {}
                    possibly_new_uniprot_to_prot_metadata = defaultdict(list)
                    # pose_name_to_prot_metadata = defaultdict(list)
                    for pose_job in pose_jobs:
                        for entity_data in pose_job.entity_data:
                            possibly_new_uniprot_to_prot_metadata[
                                entity_data.meta.uniprot_ids].append(entity_data.meta)

                    all_uniprot_id_to_prot_data = sql.initialize_metadata(
                        session, possibly_new_uniprot_to_prot_metadata)

                    # logger.debug([[data.meta.entity_id for data in pose_job.entity_data] for pose_job in pose_jobs])
                    # Get all uniprot_entities, and fix ProteinMetadata that is already loaded
                    for pose_name, pose_job in pose_name_to_pose_jobs.items():
                        for entity_data in pose_job.entity_data:
                            entity_id = entity_data.meta.entity_id
                            # Search the updated ProteinMetadata
                            for protein_metadata in all_uniprot_id_to_prot_data.values():
                                for data in protein_metadata:
                                    if entity_id == data.entity_id:
                                        # Set with the valid ProteinMetadata
                                        entity_data.meta = data
                                        break
                                else:  # No break occurred, continue with outer loop
                                    continue
                                break  # outer loop too
                            else:
                                insp = inspect(entity_data)
                                logger.warning(
                                    f'Missing the {sql.ProteinMetadata.__name__} instance for {entity_data} with '
                                    f'entity_id {entity_id}')
                                logger.debug(f'\tThis instance is transient? {insp.transient}, pending?'
                                             f' {insp.pending}, persistent? {insp.persistent}')
                    logger.debug(f'Found the newly added Session instances:\n{session.new}')
                elif number_flush_attempts == 3:
                    attrs_of_interest = \
                        ['id', 'entity_id', 'reference_sequence', 'thermophilicity', 'symmetry_group', 'model_source']
                    properties = []
                    for pose_job in pose_jobs:
                        for entity_data in pose_job.entity_data:
                            properties.append('\t'.join([f'{attr}={getattr(entity_data.meta, attr)}'
                                                         for attr in attrs_of_interest]))
                    pose_job_properties = '\n\t'.join(properties)
                    logger.warning(f"The remaining PoseJob instances have the following "
                                   f"{sql.ProteinMetadata.__name__} properties:\n\t{pose_job_properties}")
                    # This is another error
                    raise
        else:
            break

    return pose_jobs
