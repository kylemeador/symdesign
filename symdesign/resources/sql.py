from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from time import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean, select, UniqueConstraint, inspect
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import declarative_base, relationship, Session  # column_property
# from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, declarative_base  # Todo sqlalchemy 2.0
# from sqlalchemy import create_engine
# from sqlalchemy.dialects.sqlite import insert

from . import config
from .query.pdb import UKB
from symdesign.utils import path as putils, SymDesignException, symmetry, types

logger = logging.getLogger(__name__)


# class Base(DeclarativeBase):  # Todo sqlalchemy 2.0
class _Base:
    __allow_unmapped__ = True

    # def next_primary_key(self, session: Session):
    #     stmt = select(self).order_by(tuple(key.desc() for key in self.__table__.primary_key)).limit(1)
    def next_key(self, session: Session) -> int:
        stmt = select(self).order_by(self.id.desc()).limit(1)
        return session.scalars(stmt).first() + 1

    # @property
    @classmethod
    def numeric_columns(cls) -> list[Column]:
        """Return all the columns that contain numeric data"""
        # for column in cls.__table__.columns:
        #     # print(type(column.type))
        #     if isinstance(column.type, (Integer, Float)):
        #         print(True)

        return [column for column in cls.__table__.columns if isinstance(column.type, (Integer, Float))]

    @property
    def _key(self) -> str:
        return self.id

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self._key == other._key
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._key)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.id})'


Base = declarative_base(cls=_Base)


# class SymmetryGroup(Base):
#     __tablename__ = 'symmetry_groups'
#     id = Column(Integer, primary_key=True)
#     name = Column(String, nullable=False)
#     rotation = Column(Float, nullable=False)


# Has 8 columns
class PoseMetadata(Base):
    __tablename__ = 'pose_data'
    id = Column(Integer, primary_key=True)

    __table_args__ = (UniqueConstraint('project', 'name', name='_project_name_uc'),
                      )
    project = Column(String(100), nullable=False)  # Todo? , index=True)
    name = Column(String(100), nullable=False, index=True)
    # # This isn't a column in the __table__, but is an attribute of Class and derived instances
    # pose_identifier = column_property(project + os.sep + name)
    # # pose_identifier = column_property(f'{project}{os.sep}{name}')

    @hybrid_property
    def pose_identifier(self) -> str:
        return self.project + os.sep + self.name

    @staticmethod
    def convert_pose_identifier(project, name) -> str:
        return f'{project}{os.sep}{name}'

    # # Relationships concerning construction
    # # Set up many-to-one relationship with trajectory_metadata table
    # trajectory_id = Column(ForeignKey('trajectory_metadata.id'))
    # protocol = relationship('TrajectoryMetadata', back_populates='poses')
    # # # Set up one-to-one relationship with design_data table
    # # parent_design_id = Column(ForeignKey('design_data.id'))
    # # parent_design = relationship('DesignData', back_populates='child_pose')
    # parent_design = association_proxy('protocol', 'parent_design')

    # Relationships concerning associated entities and their metrics
    # # Set up many-to-one relationship with entity_data table
    # entity_data = Column(ForeignKey('entity_data.id'))
    # # Set up one-to-many relationship with entity_data table
    # Set up many-to-many relationship with entity_data table
    # entity_data = relationship('EntityData', secondary='pose_entity_association',
    #                                back_populates='poses')
    # # Set up one-to-many relationship with entity_metrics table
    # entity_metrics = relationship('EntityMetrics', back_populates='pose')
    # Set up one-to-many relationship with entity_data table
    entity_data = relationship('EntityData', back_populates='pose',
                               order_by='EntityData.id', lazy='selectin')
    # Set up one-to-one relationship with pose_metrics table
    metrics = relationship('PoseMetrics', back_populates='pose', uselist=False, lazy='selectin')
    # Set up one-to-many relationship with design_data table
    designs = relationship('DesignData', back_populates='pose',
                           # collection_class=ordering_list('id'),
                           order_by='DesignData.id')  # lazy='selectin'

    @property
    def number_of_designs(self) -> int:
        return len(self.designs)

    @property
    def number_of_entities(self) -> int:
        """Return the number of distinct entities (Gene/Protein products) found in the PoseMetadata"""
        return len(self.entity_data)

    @property
    def entity_names(self) -> list[str]:
        """Return the names of each entity (Gene/Protein products) found in the PoseMetadata"""
        return [entity.name for entity in self.entity_data]

    # @property
    # def entity_transformations(self) -> list[dict]:
    #     """Return the names of each entity (Gene/Protein products) found in the PoseMetadata"""
    #     return [entity.transform for entity in self.entity_data]

    @property
    def design_ids(self) -> list[str]:
        """Get the names of each DesignData in the PoseJob"""
        return [design.id for design in self.designs]

    @property
    def design_names(self) -> list[str]:
        """Get the names of each DesignData in the PoseJob"""
        return [design.name for design in self.designs]

    # # Set up one-to-one relationship with design_data table
    # pose_source_id = Column(ForeignKey('design_data.id'))
    # pose_source = relationship('DesignData', lazy='selectin')

    @property
    def pose_source(self):
        """Provide the DesignData for the Pose itself"""
        return self.designs[0]

    # # Set up one-to-many relationship with residue_metrics table
    # residues = relationship('ResidueMetrics', back_populates='pose')
    # Set up one-to-many relationship with pose_residue_metrics table
    residues = relationship('PoseResidueMetrics', back_populates='pose')

    transformations = association_proxy('entity_data', 'transformation')

    # State
    source_path = Column(String(500))
    # Symmetry
    sym_entry_number = Column(Integer)
    symmetry = Column(String(8))  # Result
    """The result of the SymEntry"""
    symmetry_dimension = Column(Integer)
    # symmetry_groups = relationship('SymmetryGroup')
    sym_entry_specification = Column(String(100))  # RESULT:{SUBSYMMETRY1}{SUBSYMMETRY2}...

    # @property
    # def symmetry_groups(self) -> list[str]:
    #     return [entity.symmetry for entity in self.entity_data]
    #
    # @property
    # def entity_names(self) -> list[str]:
    #     """Provide the names of all Entity instances mapped to the Pose"""
    #     return [entity.entity_name for entity in self.entity_data]


# Has 52 measures
class PoseMetrics(Base):
    __tablename__ = 'pose_metrics'

    id = Column(Integer, primary_key=True)
    # name = Column(String, nullable=False, index=True)  # String(60)
    # project = Column(String)  # , nullable=False)  # String(60)
    # Set up one-to-one relationship with pose_data table
    pose_id = Column(ForeignKey('pose_data.id'), nullable=False, unique=True)
    pose = relationship('PoseJob', back_populates='metrics')

    # design_ids = relationship('DesignMetrics', back_populates='pose')
    number_designs = Column(Integer)  # Todo integrate with full pose metric acquisition
    # Dock features
    proteinmpnn_dock_cross_entropy_loss = Column(Float)
    proteinmpnn_dock_cross_entropy_per_residue = Column(Float)
    proteinmpnn_v_design_probability_cross_entropy_loss = Column(Float)
    proteinmpnn_v_design_probability_cross_entropy_per_residue = Column(Float)
    proteinmpnn_v_evolution_probability_cross_entropy_loss = Column(Float)
    proteinmpnn_v_evolution_probability_cross_entropy_per_residue = Column(Float)
    proteinmpnn_v_fragment_probability_cross_entropy_loss = Column(Float)
    proteinmpnn_v_fragment_probability_cross_entropy_per_residue = Column(Float)
    # dock_collapse_deviation_magnitude = Column(Float)
    dock_collapse_significance_by_contact_order_z_mean = Column(Float)
    dock_collapse_increased_z_mean = Column(Float)
    dock_collapse_sequential_peaks_z_mean = Column(Float)
    dock_collapse_sequential_z_mean = Column(Float)
    dock_collapse_increase_significance_by_contact_order_z = Column(Float)
    dock_collapse_increased_z = Column(Float)
    dock_collapse_new_positions = Column(Integer)
    dock_collapse_new_position_significance = Column(Float)
    dock_collapse_sequential_peaks_z = Column(Float)
    dock_collapse_sequential_z = Column(Float)
    dock_collapse_significance_by_contact_order_z = Column(Float)
    dock_collapse_variance = Column(Float)
    dock_collapse_violation = Column(Boolean)
    dock_hydrophobicity = Column(Float)
    # Pose features
    #  Fragment features
    interface_secondary_structure_fragment_topology = Column(String(120))
    interface_secondary_structure_fragment_count = Column(Integer)
    nanohedra_score_normalized = Column(Float)
    nanohedra_score_center_normalized = Column(Float)
    nanohedra_score = Column(Float)
    nanohedra_score_center = Column(Float)
    multiple_fragment_ratio = Column(Float)
    number_residues_interface_fragment_total = Column(Integer)
    number_residues_interface_fragment_center = Column(Integer)
    number_fragments_interface = Column(Integer)
    number_residues_interface_non_fragment = Column(Float)
    percent_fragment_helix = Column(Float)
    percent_fragment_strand = Column(Float)
    percent_fragment_coil = Column(Float)
    percent_interface_coil = Column(Float)
    percent_interface_helix = Column(Float)
    percent_interface_strand = Column(Float)
    percent_residues_fragment_interface_total = Column(Float)
    percent_residues_fragment_interface_center = Column(Float)
    percent_residues_non_fragment_interface = Column(Float)
    #  Fragment features end
    entity_max_radius_average_deviation = Column(Float)
    entity_min_radius_average_deviation = Column(Float)
    # entity_number_of_residues_average_deviation = Column(Float)
    entity_radius_average_deviation = Column(Float)
    interface_b_factor = Column(Float)
    interface_secondary_structure_topology = Column(String(120))
    interface_secondary_structure_count = Column(Integer)
    maximum_radius = Column(Float)
    minimum_radius = Column(Float)
    number_residues_interface = Column(Integer)
    pose_length = Column(Integer)
    # Pose features end
    pose_thermophilicity = Column(Float)
    symmetric_interface = Column(Boolean)
    """Thermophilicity implies this is a spectrum, while thermophilic implies binary"""


interface_pose_metrics = dict(
    interface_secondary_structure_fragment_count=Integer,
    interface_secondary_structure_fragment_topology=String(60),
    interface_secondary_structure_count=Integer,
    interface_secondary_structure_topology=String(60)
)
for idx in range(1, 1 + config.MAXIMUM_INTERFACES):
    for metric, value in interface_pose_metrics.items():
        setattr(PoseMetrics, metric.replace('interface_', f'interface{idx}_'), Column(value))

# ratio_design_metrics = dict(
#     entity_radius_ratio_v=Float,
#     entity_min_radius_ratio_v=Float,
#     entity_max_radius_ratio_v=Float,
#     entity_number_of_residues_ratio_v=Float,
# )
# for idx1, idx2 in combinations(range(1, 1 + config.MAXIMUM_ENTITIES), 2):
#     for metric, value in ratio_design_metrics.items():
#         setattr(PoseMetrics, metric.replace('_v', f'_{idx1}v{idx2}'), Column(value))

# # class PoseEntityAssociation(Base):
# pose_entity_association = Table(
#     'pose_entity_association',
#     Base.metadata,
#     # Todo upon sqlalchemy 2.0, use the sqlalchemy.Column construct not mapped_column()
#     Column('pose_id', ForeignKey('pose_data.id'), primary_key=True),
#     Column('entity_id', ForeignKey('entity_data.id'), primary_key=True)
# )


# uniprot_protein_association = Table(
class UniProtProteinAssociation(Base):
    __tablename__ = 'uniprot_protein_association'
    # Todo upon sqlalchemy 2.0, use the sqlalchemy.Column construct not mapped_column()
    uniprot_id = Column(ForeignKey('uniprot_entity.id'), primary_key=True)
    entity_id = Column(ForeignKey('protein_metadata.id'), primary_key=True)
    uniprot = relationship('UniProtEntity', back_populates='_protein_metadata',
                           lazy='selectin')
    protein = relationship('ProteinMetadata', back_populates='_uniprot_entities',
                           lazy='selectin')
    position = Column(Integer)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.uniprot_id},{self.entity_id},{self.position})'


# Has 10 columns
class ProteinMetadata(Base):
    """Used for hold fixed metadata of protein structures, typically pulled from PDB API"""
    __tablename__ = 'protein_metadata'
    id = Column(Integer, primary_key=True)

    entity_id = Column(String(20), nullable=False, index=True, unique=True)  # entity_name is used in config.metrics
    # Todo String(20) may need to be increased if de-novo protein names (no identifier) are included
    """This could be described as the PDB API EntityID"""
    # # Set up many-to-one relationship with uniprot_entity table. Using "many" because there can be chimera's
    # uniprot_id = Column(ForeignKey('uniprot_entity.id'))
    # Set up many-to-many relationship with uniprot_entity table. Using "many" on both sides because there can be
    # ProteinMetadata chimera's and different domains of a UniProtID could be individual ProteinMetadata
    # uniprot_entities = relationship('UniProtEntity', secondary='uniprot_protein_association',
    #                                 back_populates='protein_metadata')
    # # Create a Collection[str] association between this ProteinMetadata and linked uniprot_ids
    # uniprot_ids = association_proxy('uniprot_entities', 'id')
    _uniprot_entities = relationship('UniProtProteinAssociation',
                                     back_populates='protein',
                                     collection_class=ordering_list('position'),
                                     order_by=UniProtProteinAssociation.position,
                                     lazy='selectin'
                                     )
    uniprot_entities = association_proxy('_uniprot_entities', 'uniprot',
                                         creator=lambda _unp_ent: UniProtProteinAssociation(uniprot=_unp_ent))
    # Set up one-to-many relationship with entity_data table
    entity_data = relationship('EntityData', back_populates='meta')

    model_source = Column(String(500))
    refined = Column(Boolean, default=False)
    loop_modeled = Column(Boolean, default=False)
    reference_sequence = Column(String(config.MAXIMUM_SEQUENCE))
    # number_of_residues = Column(Integer)  # entity_ is used in config.metrics
    n_terminal_helix = Column(Boolean)  # entity_ is used in config.metrics
    c_terminal_helix = Column(Boolean)  # entity_ is used in config.metrics
    thermophilicity = Column(Float)  # entity_ is used in config.metrics
    """Thermophilicity implies this is a spectrum, while thermophilic implies binary"""
    # Symmetry parameters
    symmetry_group = Column(String(50))  # entity_ is used in config.metrics
    # symmetry = Column(ForeignKey('symmetry_groups.id'))

    @property
    def uniprot_ids(self) -> tuple[str, ...]:
        """Access the UniProtID's associated with this instance"""
        return tuple(uni_entity.id for uni_entity in self.uniprot_entities)

    @property
    def entity_info(self) -> dict[str, dict[str, Any]]:
        """Format the instance for population of metadata via the entity_info kwargs"""
        return {self.entity_id:
                dict(chains=[],
                     dbref=dict(accession=self.uniprot_ids, db=UKB),
                     reference_sequence=self.reference_sequence,
                     thermophilicity=self.thermophilicity)
                }

    def __repr__(self):
        return f'{self.__class__.__name__}({self.entity_id})'


# Has 3 columns
class EntityData(Base):
    """Used for unique Pose instances to connect multiple sources of information"""
    __tablename__ = 'entity_data'
    id = Column(Integer, primary_key=True)

    __table_args__ = (UniqueConstraint('pose_id', 'properties_id', name='_pose_properties_id_uc'),
                      )
    # Set up many-to-one relationship with pose_data table
    pose_id = Column(ForeignKey('pose_data.id'))
    pose = relationship('PoseJob', back_populates='entity_data')
    # Todo
    # Shouldn't use EntityData unless a pose is provided... perhaps I should make an init
    # pose = ..., nullable=False)
    # # Set up many-to-one relationship with uniprot_entity table
    # uniprot_id = Column(ForeignKey('uniprot_entity.id'))
    # uniprot_entity = relationship('UniProtEntity', back_populates='entities', nullable=False)
    # # OR
    # # Todo
    # #  this class may be a case where an Association Proxy should be used given only use of relationships
    # #  As this references an Association Object (the ProteinMetadata class) this would return the uniprot_entity
    # #  If I want the uniprot_id, I need to make a plain association table it seems...
    # #  uniprot_entity = association_proxy('meta', 'uniprot_entity')
    # Set up many-to-one relationship with protein_metadata table
    properties_id = Column(ForeignKey('protein_metadata.id'))
    meta = relationship('ProteinMetadata', back_populates='entity_data',
                        lazy='selectin', innerjoin=True)  # Trying this to get around detached state...
                        # lazy='joined', innerjoin=True)
    # Set up one-to-one relationship with entity_metrics table
    metrics = relationship('EntityMetrics', back_populates='entity', uselist=False)
    # Todo setup 'selectin' load for select-* modules
    # Set up one-to-many relationship with design_entity_data table
    design_metrics = relationship('DesignEntityMetrics', back_populates='entity')
    # Set up one-to-one relationship with entity_transform table
    transform = relationship('EntityTransform', back_populates='entity', uselist=False,
                             lazy='selectin')
    # transformation = association_proxy('_transform', 'transformation')
    # # Set up many-to-many relationship with pose_data table
    # poses = relationship('PoseMetadata', secondary='pose_entity_association',
    #                      back_populates='entity_data')

    # Use these accessors to ensure that passing EntityData to Entity.from_chains() can access these variables
    @property
    def name(self):
        return self.meta.entity_id

    @property
    def reference_sequence(self):
        return self.meta.reference_sequence

    @property
    def thermophilicity(self):
        return self.meta.thermophilicity

    @property
    def uniprot_ids(self):
        # return [entity.uniprot_id for entity in self.meta.uniprot_entities]
        return self.meta.uniprot_ids

    @property
    def entity_info(self) -> dict[str, dict[str, Any]]:
        """Format the instance for population of metadata via the entity_info kwargs"""
        return self.meta.entity_info

    @property
    def transformation(self) -> types.TransformationMapping | dict:
        try:
            return self.transform.transformation
        except AttributeError:  # self.transform is probably None
            return {}

    # @property
    # def uniprot_id(self):
    #     return self.meta.uniprot_id


# Has 9 measurements
class EntityMetrics(Base):
    __tablename__ = 'entity_metrics'
    id = Column(Integer, primary_key=True)

    # # Set up many-to-one relationship with pose_data table
    # pose_id = Column(ForeignKey('pose_data.id'))
    # pose = relationship('PoseMetadata', back_populates='entity_metrics')
    # Set up one-to-one relationship with entity_data table
    entity_id = Column(ForeignKey('entity_data.id'))
    entity = relationship('EntityData', back_populates='metrics')

    # number_of_residues = Column(Integer)  # entity_ is used in config.metrics
    max_radius = Column(Float)  # entity_ is used in config.metrics
    min_radius = Column(Float)  # entity_ is used in config.metrics
    radius = Column(Float)  # entity_ is used in config.metrics
    n_terminal_orientation = Column(Integer)  # entity_ is used in config.metrics
    c_terminal_orientation = Column(Integer)  # entity_ is used in config.metrics
    interface_secondary_structure_fragment_topology = Column(String(60))  # entity_ is used in config.metrics
    interface_secondary_structure_topology = Column(String(60))  # entity_ is used in config.metrics


# Has 14 columns
class EntityTransform(Base):
    __tablename__ = 'entity_transform'
    id = Column(Integer, primary_key=True)

    # Set up one-to-one relationship with entity_data table
    entity_id = Column(ForeignKey('entity_data.id'))
    entity = relationship('EntityData', back_populates='transform')
    rotation_x = Column(Float, default=0.)
    rotation_y = Column(Float, default=0.)
    rotation_z = Column(Float, default=0.)
    setting_matrix = Column(Integer)
    external_rotation_x = Column(Float)
    external_rotation_y = Column(Float)
    external_rotation_z = Column(Float)
    internal_translation_x = Column(Float, default=0.)
    internal_translation_y = Column(Float, default=0.)
    internal_translation_z = Column(Float, default=0.)
    external_translation_x = Column(Float, default=0.)
    external_translation_y = Column(Float, default=0.)
    external_translation_z = Column(Float, default=0.)

    @property
    def transformation(self) -> types.TransformationMapping | dict:
        """Provide the names of all Entity instances mapped to the Pose"""
        # Todo hook in self.rotation_x, self.rotation_y
        if self.setting_matrix is None:
            rotation2 = None
            # rotation2 = Rotation.from_rotvec([self.external_rotation_x,
            #                                   self.external_rotation_y,
            #                                   self.external_rotation_z], degrees=True).as_matrix()
        else:
            rotation2 = symmetry.setting_matrices[self.setting_matrix]
        return dict(
            rotation=Rotation.from_rotvec([self.rotation_x, self.rotation_y, self.rotation_z], degrees=True).as_matrix(),
            translation=np.array([self.internal_translation_x,
                                  self.internal_translation_y,
                                  self.internal_translation_z]),
            # translation=np.array([0., 0., entity.internal_translation]),
            rotation2=rotation2,
            translation2=np.array([self.external_translation_x,
                                   self.external_translation_y,
                                   self.external_translation_z]),
        )

    @transformation.setter
    def transformation(self, transform: types.TransformationMapping):
        if any((self.rotation_x, self.rotation_y, self.rotation_z,
                self.internal_translation_x, self.internal_translation_y, self.internal_translation_z,
                self.setting_matrix,
                self.external_translation_x, self.external_translation_y, self.external_translation_z)):
            raise RuntimeError(
                "Can't set the transformation as this would disrupt the persistence of the table "
                f"{EntityTransform.__tablename__}")

        if not isinstance(transform, dict):
            raise ValueError(
                f"The attribute 'transformation' must be a {types.TransformationMapping.__name__}, not "
                f'{type(transform).__name__}')

        for operation_type, operation in transform.items():
            if operation is None:
                continue
            elif operation_type == 'rotation':
                self.rotation_x, self.rotation_y, self.rotation_z = \
                    Rotation.from_matrix(operation).as_rotvec(degrees=True)
            elif operation_type == 'translation':
                self.internal_translation_x, \
                    self.internal_translation_y, \
                    self.internal_translation_z = operation
            elif operation_type == 'rotation2':
                # Convert to setting number
                for number, matrix in symmetry.setting_matrices.items():
                    # input(f'setting matrix number: {number}')
                    if np.allclose(operation, matrix):
                        # input(f'found all_close: {operation}\n{matrix}')
                        self.setting_matrix = number
                        break
                else:
                    self.external_rotation_x, self.external_rotation_y, self.external_rotation_z = \
                        Rotation.from_matrix(operation).as_rotvec(degrees=True)
            elif operation_type == 'translation2':
                self.external_translation_x, \
                    self.external_translation_y, \
                    self.external_translation_z = operation

            # self._pose_transformation = self.info['pose_transformation'] = list(transform)

# class Protocol(Base):
#     __tablename__ = 'protocols'
#     id = Column(Integer, primary_key=True)
#     name = Column(String, nullable=False, unique=True)
#
#     # # Set up one-to-many relationship with trajectory_metadata table
#     # trajectories = relationship('TrajectoryMetadata', back_populates='protocol')
#     # Set up one-to-many relationship with design_protocol table
#     designs = relationship('DesignProtocol', back_populates='protocol')


# class TrajectoryMetadata(Base):
#     __tablename__ = 'trajectory_metadata'
#     id = Column(Integer, primary_key=True)
#
#     # protocol = Column(String, nullable=False)
#     # Set up many-to-one relationship with protocols table
#     protocol_id = Column(ForeignKey('protocols.id'), nullable=False)
#     protocol = relationship('Protocol', back_populates='trajectories')
#     # # Set up one-to-many relationship with pose_data table
#     # poses = relationship('PoseJob', back_populates='protocol')
#     # Set up many-to-one relationship with design_data table
#     parent_design_id = Column(ForeignKey('design_data.id'))
#     parent_design = relationship('DesignData', back_populates='trajectories')


class JobProtocol(Base):
    __tablename__ = 'job_protocols'
    id = Column(Integer, primary_key=True)
    __table_args__ = (
        UniqueConstraint(
            'module', 'commit', 'proteinmpnn_model_name',
            'ca_only', 'contiguous_ghosts', 'evolution_constraint',
            'initial_z_value', 'interface', 'match_value',
            'minimum_matched', 'neighbors', 'number_predictions',
            'prediction_model', 'term_constraint', 'use_gpu_relax',
            name='_job_protocol_uc'
        ),
    )
    module = Column(String(30), nullable=False)
    commit = Column(String(40), nullable=False)
    # Set up one-to-many relationship with design_protocol table
    design_protocols = relationship('DesignProtocol', back_populates='job')

    ca_only = Column(Boolean)  # design/score
    proteinmpnn_model_name = Column(String(10))  # design/score
    contiguous_ghosts = Column(Boolean)  # dock
    evolution_constraint = Column(Boolean)  # design
    initial_z_value = Column(Float)  # dock
    interface = Column(Boolean)  # design
    match_value = Column(Float)  # dock
    minimum_matched = Column(Integer)  # dock
    neighbors = Column(Boolean)  # design
    number_predictions = Column(Integer)  # structure-predict
    prediction_model = Column(String(5))  # structure-predict # ENUM - all, best, none
    term_constraint = Column(Boolean)  # design
    use_gpu_relax = Column(Boolean)  # structure-predict

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'id={self.id}, ' \
               f'module={self.module}, ' \
               f'commit={self.commit}, ' \
               f'ca_only={self.ca_only}, ' \
               f'proteinmpnn_model_name={self.proteinmpnn_model_name}, ' \
               f'contiguous_ghosts={self.contiguous_ghosts}, ' \
               f'evolution_constraint={self.evolution_constraint}, ' \
               f'initial_z_value={self.initial_z_value}, ' \
               f'interface={self.interface}, ' \
               f'match_value={self.match_value}, ' \
               f'minimum_matched={self.minimum_matched}, ' \
               f'neighbors={self.neighbors}, ' \
               f'number_predictions={self.number_predictions}, ' \
               f'prediction_model={self.prediction_model}, ' \
               f'term_constraint={self.term_constraint}, ' \
               f'use_gpu_relax={self.use_gpu_relax}, ' \
               ')'


class DesignProtocol(Base):
    __tablename__ = 'design_protocol'
    id = Column(Integer, primary_key=True)

    protocol = Column(String(50), nullable=False)
    # Set up many-to-one relationship with job_protocols table
    job_id = Column(ForeignKey('job_protocols.id'), nullable=False)
    job = relationship('JobProtocol', back_populates='design_protocols')
    # Set up many-to-one relationship with design_data table
    design_id = Column(ForeignKey('design_data.id'))
    design = relationship('DesignData', back_populates='protocols')
    file = Column(String(500))
    temperature = Column(Float)
    alphafold_model = Column(String(20))  # Column(Float)


class DesignData(Base):
    """Account for design metadata created from pose metadata"""
    __tablename__ = 'design_data'
    id = Column(Integer, primary_key=True)
    __table_args__ = (UniqueConstraint('pose_id', 'name', name='_pose_name_uc'),
                      )
    name = Column(String(106), nullable=False)  # , unique=True)
    # Set up many-to-one relationship with pose_data table
    pose_id = Column(ForeignKey('pose_data.id'), nullable=False)
    pose = relationship('PoseJob', back_populates='designs')
    # # Set up one-to-many relationship with trajectory_metadata table
    # trajectories = relationship('TrajectoryMetadata', back_populates='parent_design')
    # design_parent = relationship('DesignData', back_populates='parent_design')
    # design_children = relationship('DesignData', back_populates='design_parent')
    # Set up one-to-many relationship with design_data (self) table
    design_parent_id = Column(ForeignKey('design_data.id'))
    design_parent = relationship('DesignData', remote_side=[id],
                                 back_populates='design_children', uselist=False)
    design_children = relationship('DesignData', back_populates='design_parent',
                                   lazy='joined', join_depth=1)  # Only get the immediate children
    # Set up one-to-many relationship with design_protocol table
    protocols = relationship('DesignProtocol', back_populates='design')
    # Set up one-to-one relationship with design_metrics table
    metrics = relationship('DesignMetrics', back_populates='design', uselist=False)
    # Set up one-to-many relationship with design_entity_data table
    entity_metrics = relationship('DesignEntityMetrics', back_populates='design')
    # Set up one-to-many relationship with residue_metrics table
    residues = relationship('DesignResidues', back_populates='design')
    # Set up one-to-many relationship with residue_metrics table
    residue_metrics = relationship('ResidueMetrics', back_populates='design')

    structure_path = Column(String(500))
    sequence = association_proxy('metrics', 'sequence')

    def __str__(self):
        return self.name


# This has 105 columns. Most of these don't get filled unless there is a Pose structure associated
#  3 always
#  38 for Sequence
#  59 for Structure
#  12 for Alphafold
# Todo
#  class StructureDesignMetrics(Base):
#  class SequenceDesignMetrics(Base):
class DesignMetrics(Base):
    __tablename__ = 'design_metrics'
    id = Column(Integer, primary_key=True)

    # name = Column(String, nullable=False)  # String(60)
    # pose = Column(String, nullable=False)  # String(60)
    # pose_name = Column(String, nullable=False)  # String(60)
    # Set up one-to-one relationship with design_data table
    design_id = Column(ForeignKey('design_data.id'), nullable=False, unique=True)
    design = relationship('DesignData', back_populates='metrics')

    # SEQUENCE BASED
    number_residues_design = Column(Integer)  # ResidueMetrics sum 'design_residue', nullable=False)
    # Sequence metrics
    number_mutations = Column(Integer)  # ResidueMetrics sum 'mutation'
    percent_mutations = Column(Float)
    # SymDesign metrics
    # Collapse measurements
    # NOT taken hydrophobic_collapse = Column(Float)
    hydrophobicity = Column(Float)  # Renamed from ResidueMetrics
    # collapse_deviation_magnitude = Column(Float)
    collapse_increase_significance_by_contact_order_z = Column(Float)
    collapse_increased_z = Column(Float)
    collapse_new_positions = Column(Integer)
    collapse_new_position_significance = Column(Float)
    collapse_sequential_peaks_z = Column(Float)
    collapse_sequential_z = Column(Float)
    collapse_significance_by_contact_order_z = Column(Float)
    collapse_variance = Column(Float)
    collapse_violation = Column(Float)  # Todo Potential column_property
    collapse_significance_by_contact_order_z_mean = Column(Float)
    collapse_increased_z_mean = Column(Float)
    collapse_sequential_peaks_z_mean = Column(Float)
    collapse_sequential_z_mean = Column(Float)
    interface_local_density = Column(Float)
    interface_composition_similarity = Column(Float)
    interface_area_to_residue_surface_ratio = Column(Float)
    sequence = Column(String(config.MAXIMUM_SEQUENCE))
    # ProteinMPNN score terms
    proteinmpnn_loss_complex = Column(Float)
    proteinmpnn_loss_unbound = Column(Float)
    proteinmpnn_score_complex = Column(Float)
    proteinmpnn_score_complex_per_designed_residue = Column(Float)
    proteinmpnn_score_complex_per_interface_residue = Column(Float)
    proteinmpnn_score_delta = Column(Float)
    proteinmpnn_score_delta_per_designed_residue = Column(Float)
    proteinmpnn_score_delta_per_interface_residue = Column(Float)
    proteinmpnn_score_unbound = Column(Float)
    proteinmpnn_score_unbound_per_designed_residue = Column(Float)
    proteinmpnn_score_unbound_per_interface_residue = Column(Float)
    # Sequence loss terms
    sequence_loss_design = Column(Float)
    sequence_loss_design_per_residue = Column(Float)
    sequence_loss_evolution = Column(Float)
    sequence_loss_evolution_per_residue = Column(Float)
    sequence_loss_fragment = Column(Float)
    sequence_loss_fragment_per_residue = Column(Float)
    # Observed in profile measurements
    observed_design = Column(Float)
    observed_evolution = Column(Float)
    observed_fragment = Column(Float)
    observed_interface = Column(Float)
    # # Direct coupling analysis energy
    # dca_energy = Column(Float)
    # STRUCTURE BASED
    # Pose features
    number_residues_interface = Column(Integer)
    contact_order = Column(Float)  # Todo Duplicated of Pose, unless number of residues/coordinates change
    # Rosetta metrics
    buried_unsatisfied_hbond_density = Column(Float)
    buried_unsatisfied_hbonds = Column(Integer)
    buried_unsatisfied_hbonds_complex = Column(Integer)
    # buried_unsatisfied_hbonds_unbound = Column(Integer)  # Has a # after buns LIST
    contact_count = Column(Integer)
    # entity_interface_connectivity = Column(Float)  # Has a # after entity LIST
    favor_residue_energy = Column(Float)
    interaction_energy_complex = Column(Float)
    interface_energy_density = Column(Float)
    interface_bound_activation_energy = Column(Float)
    interface_solvation_energy = Column(Float)
    interface_solvation_energy_activation = Column(Float)
    interface_solvation_energy_density = Column(Float)
    interaction_energy_per_residue = Column(Float)
    interface_separation = Column(Float)
    interface_separation_core = Column(Float)
    interface_separation_fragment = Column(Float)
    rmsd_complex = Column(Float)
    rosetta_reference_energy = Column(Float)
    shape_complementarity = Column(Float)
    shape_complementarity_core = Column(Float)
    shape_complementarity_fragment = Column(Float)
    # solvation_energy = Column(Float)
    # solvation_energy_complex = Column(Float)
    # Summed ResidueMetrics metrics
    # -----------------------
    # column name is changed from Rosetta energy values
    interface_energy_complex = Column(Float)
    interface_energy_bound = Column(Float)
    interface_energy_unbound = Column(Float)
    interface_energy = Column(Float)
    interface_solvation_energy_complex = Column(Float)
    interface_solvation_energy_bound = Column(Float)
    interface_solvation_energy_unbound = Column(Float)
    number_hbonds = Column(Integer)
    coordinate_constraint = Column(Float)
    residue_favored = Column(Float)
    # FreeSASA - column name is changed from ResidueMetrics
    area_hydrophobic_complex = Column(Float)
    area_hydrophobic_unbound = Column(Float)
    area_polar_complex = Column(Float)
    area_polar_unbound = Column(Float)
    area_total_complex = Column(Float)  # Todo Potential column_property
    area_total_unbound = Column(Float)  # Todo Potential column_property
    interface_area_polar = Column(Float)
    interface_area_hydrophobic = Column(Float)
    interface_area_total = Column(Float)  # Todo Potential column_property
    # sasa_relative_complex = Column(Float)  # NOT useful
    # sasa_relative_bound = Column(Float)  # NOT useful
    # SymDesign measurements
    # errat_deviation = Column(Float)  # Todo include when measured
    interior = Column(Float)
    surface = Column(Float)
    support = Column(Float)
    rim = Column(Float)
    core = Column(Float)
    percent_interface_area_hydrophobic = Column(Float)  # Todo Potential column_property
    percent_interface_area_polar = Column(Float)  # Todo Potential column_property
    percent_core = Column(Float)  # Todo Potential column_property
    percent_rim = Column(Float)  # Todo Potential column_property
    percent_support = Column(Float)  # Todo Potential column_property
    spatial_aggregation_propensity_interface = Column(Float)
    spatial_aggregation_propensity = Column(Float)
    spatial_aggregation_propensity_unbound = Column(Float)
    # Alphafold metrics
    plddt = Column(Float)
    plddt_deviation = Column(Float)
    predicted_aligned_error = Column(Float)
    predicted_aligned_error_deviation = Column(Float)
    predicted_aligned_error_interface = Column(Float)
    predicted_aligned_error_interface_deviation = Column(Float)
    predicted_interface_template_modeling_score = Column(Float)
    predicted_interface_template_modeling_score_deviation = Column(Float)
    predicted_template_modeling_score = Column(Float)
    predicted_template_modeling_score_deviation = Column(Float)
    rmsd_prediction_ensemble = Column(Float)
    rmsd_prediction_ensemble_deviation = Column(Float)


interface_design_metrics = dict(
    buried_unsatisfied_hbonds_unbound=Integer
)
for idx in range(1, 1 + config.MAXIMUM_INTERFACES):
    for metric, value in interface_design_metrics.items():
        setattr(DesignMetrics, f'{metric}{idx}', Column(value))


# This has 18 columns:
#  3 always
#  3 for Design
#  12 for AlphaFold
class DesignEntityMetrics(Base):
    __tablename__ = 'design_entity_metrics'
    id = Column(Integer, primary_key=True)

    __table_args__ = (
        UniqueConstraint('design_id', 'entity_id', name='_design_id_entity_id_uc'),
    )
    # Set up many-to-one relationship with design_data table
    design_id = Column(ForeignKey('design_data.id'), nullable=False)  # , unique=True)
    design = relationship('DesignData', back_populates='entity_metrics')
    # Set up many-to-one relationship with entity_data table
    entity_id = Column(ForeignKey('entity_data.id'), nullable=False)  # , unique=True)
    entity = relationship('EntityData', back_populates='design_metrics')

    # Design descriptors
    interface_connectivity = Column(Float)  # entity_ is in config.metrics
    percent_mutations = Column(Float)  # entity_ is in config.metrics
    number_mutations = Column(Integer)  # entity_ is in config.metrics. ResidueMetrics sum 'mutation'
    # Alphafold metrics
    plddt = Column(Float)  # entity_ is in config.metrics
    plddt_deviation = Column(Float)  # entity_ is in config.metrics
    predicted_aligned_error = Column(Float)  # entity_ is in config.metrics
    predicted_aligned_error_deviation = Column(Float)  # entity_ is in config.metrics
    predicted_aligned_error_interface = Column(Float)  # entity_ is in config.metrics
    predicted_aligned_error_interface_deviation = Column(Float)  # entity_ is in config.metrics
    predicted_interface_template_modeling_score = Column(Float)  # entity_ is in config.metrics
    predicted_interface_template_modeling_score_deviation = Column(Float)  # entity_ is in config.metrics
    predicted_template_modeling_score = Column(Float)  # entity_ is in config.metrics
    predicted_template_modeling_score_deviation = Column(Float)  # entity_ is in config.metrics
    # rmsd_oligomer = Column(Float)  # entity_ is in config.metrics
    rmsd_prediction_ensemble = Column(Float)  # entity_ is in config.metrics
    rmsd_prediction_ensemble_deviation = Column(Float)  # entity_ is in config.metrics


# Has 4 columns
class DesignResidues(Base):
    __tablename__ = 'design_residues'
    id = Column(Integer, primary_key=True)

    __table_args__ = (
        # UniqueConstraint('pose_id', 'design_id', name='_pose_design_uc'),
        UniqueConstraint('design_id', 'index', name='_design_index_uc'),
    )
    # Set up many-to-one relationship with design_data table
    design_id = Column(ForeignKey('design_data.id'))
    design = relationship('DesignData', back_populates='residues')

    # Residue index (surrogate for residue number)
    index = Column(Integer, nullable=False)
    design_residue = Column(Boolean, nullable=False)


# Has 15 columns
class PoseResidueMetrics(Base):
    __tablename__ = 'pose_residue_metrics'
    id = Column(Integer, primary_key=True)

    __table_args__ = (
        UniqueConstraint('pose_id', 'index', name='_pose_index_uc'),
    )
    # Residue index (surrogate for residue number) and type information
    index = Column(Integer, nullable=False)
    interface_residue = Column(Boolean)

    # Set up many-to-one relationship with pose_data table
    pose_id = Column(ForeignKey('pose_data.id'))
    pose = relationship('PoseJob', back_populates='residues')
    # ProteinMPNN score terms
    proteinmpnn_dock_cross_entropy_loss = Column(Float)
    proteinmpnn_v_design_probability_cross_entropy_loss = Column(Float)
    proteinmpnn_v_evolution_probability_cross_entropy_loss = Column(Float)
    proteinmpnn_v_fragment_probability_cross_entropy_loss = Column(Float)
    dock_collapse_deviation_magnitude = Column(Float)
    dock_collapse_increase_significance_by_contact_order_z = Column(Float)
    dock_collapse_increased_z = Column(Float)
    dock_collapse_new_positions = Column(Boolean)
    dock_collapse_new_position_significance = Column(Float)
    dock_collapse_sequential_peaks_z = Column(Float)
    dock_collapse_sequential_z = Column(Float)
    dock_collapse_significance_by_contact_order_z = Column(Float)
    dock_hydrophobic_collapse = Column(Float)


# This has 54 columns, most of which rely on structures
#  None (besides design_residue) has a use in the processing of modules... just used to produce DesignMetrics
class ResidueMetrics(Base):
    __tablename__ = 'residue_metrics'
    id = Column(Integer, primary_key=True)

    __table_args__ = (
        # UniqueConstraint('pose_id', 'design_id', name='_pose_design_uc'),
        UniqueConstraint('design_id', 'index', name='_design_index_uc'),
    )
    # # Set up many-to-one relationship with pose_data table
    # pose_id = Column(ForeignKey('pose_data.id'))
    # pose = relationship('PoseJob', back_populates='residues')
    # Set up many-to-one relationship with design_data table
    design_id = Column(ForeignKey('design_data.id'))
    design = relationship('DesignData', back_populates='residue_metrics')

    # Residue index (surrogate for residue number) and type information
    index = Column(Integer, nullable=False)
    type = Column(String(1))  # , nullable=False)
    design_residue = Column(Boolean)
    interface_residue = Column(Boolean)
    mutation = Column(Boolean)
    # Rosetta energy values
    complex = Column(Float)
    bound = Column(Float)
    unbound = Column(Float)
    energy_delta = Column(Float)
    solv_complex = Column(Float)
    solv_bound = Column(Float)
    solv_unbound = Column(Float)
    hbond = Column(Boolean)
    coordinate_constraint = Column(Float)
    residue_favored = Column(Float)
    # SymDesign measurements
    contact_order = Column(Float)
    # errat_deviation = Column(Float)  # Todo include when measured
    sasa_hydrophobic_complex = Column(Float)
    sasa_polar_complex = Column(Float)
    sasa_relative_complex = Column(Float)
    sasa_hydrophobic_bound = Column(Float)
    sasa_polar_bound = Column(Float)
    sasa_relative_bound = Column(Float)
    bsa_hydrophobic = Column(Float)
    bsa_polar = Column(Float)
    bsa_total = Column(Float)
    sasa_total_bound = Column(Float)
    sasa_total_complex = Column(Float)
    interior = Column(Float)
    surface = Column(Float)
    support = Column(Float)
    rim = Column(Float)
    core = Column(Float)
    # Collapse measurements
    collapse_deviation_magnitude = Column(Float)
    collapse_increase_significance_by_contact_order_z = Column(Float)
    collapse_increased_z = Column(Float)
    collapse_new_positions = Column(Boolean)
    collapse_new_position_significance = Column(Float)
    collapse_sequential_peaks_z = Column(Float)
    collapse_sequential_z = Column(Float)
    collapse_significance_by_contact_order_z = Column(Float)
    hydrophobic_collapse = Column(Float)
    spatial_aggregation_propensity = Column(Float)
    spatial_aggregation_propensity_unbound = Column(Float)
    # ProteinMPNN score terms
    proteinmpnn_loss_complex = Column(Float)
    proteinmpnn_loss_unbound = Column(Float)
    sequence_loss_design = Column(Float)
    sequence_loss_evolution = Column(Float)
    sequence_loss_fragment = Column(Float)
    # Observed in profile measurements
    observed_design = Column(Boolean)
    observed_evolution = Column(Boolean)
    observed_fragment = Column(Boolean)
    observed_interface = Column(Boolean)
    # # Direct coupling analysis energy
    # dca_energy = Column(Float)
    # Folding metrics
    plddt = Column(Float)
    predicted_aligned_error = Column(Float)


from . import wrapapi


def initialize_metadata(session: Session,
                        possibly_new_uniprot_to_prot_data: dict[tuple[str, ...], Iterable[ProteinMetadata]] = None,
                        existing_uniprot_entities: Iterable[wrapapi.UniProtEntity] = None,
                        existing_protein_metadata: Iterable[ProteinMetadata] = None) -> \
        dict[tuple[str, ...], list[ProteinMetadata]] | dict:
    """Compare newly described work to the existing database and set up metadata for all described entities

    Doesn't commit new instances to the database in case they are attached to existing objects

    Args:
        session: A currently open transaction within sqlalchemy
        possibly_new_uniprot_to_prot_data: A mapping of the possibly required UniProtID entries and their associated
            ProteinMetadata. These could already exist in database, but were indicated they are needed
        existing_uniprot_entities: If any UniProtEntity instances are already loaded, pass them to expedite setup
        existing_protein_metadata: If any ProteinMetadata instances are already loaded, pass them to expedite setup
    """
    if not possibly_new_uniprot_to_prot_data:
        if existing_protein_metadata:
            pass
            # uniprot_id_to_metadata = {protein_data.uniprot_ids:
            #                           protein_data for protein_data in existing_protein_metadata}
        elif existing_uniprot_entities:
            existing_protein_metadata = {unp_entity.protein_metadata for unp_entity in existing_uniprot_entities}
        else:
            existing_protein_metadata = {}

        return {protein_data.uniprot_ids: [protein_data] for protein_data in existing_protein_metadata}

    # Todo
    #  If I ever adopt the UniqueObjectValidatedOnPending recipe, that could perform the work of getting the
    #  correct objects attached to the database

    # Get the set of all UniProtIDs
    possibly_new_uniprot_ids = set()
    for uniprot_ids in possibly_new_uniprot_to_prot_data.keys():
        possibly_new_uniprot_ids.update(uniprot_ids)
    # Find existing UniProtEntity instances from database
    if existing_uniprot_entities is None:
        existing_uniprot_entities = []
    else:
        existing_uniprot_entities = list(existing_uniprot_entities)

    existing_uniprot_ids = {unp_ent.id for unp_ent in existing_uniprot_entities}
    # Remove the certainly existing from possibly new and query for any new that already exist
    query_additional_existing_uniprot_entities_stmt = \
        select(wrapapi.UniProtEntity).where(wrapapi.UniProtEntity.id.in_(
            possibly_new_uniprot_ids.difference(existing_uniprot_ids)))
    # Add all requested to those known about
    existing_uniprot_entities += session.scalars(query_additional_existing_uniprot_entities_stmt).all()

    # Todo Maybe needed?
    #  Emit this select when there is a stronger association between the multiple
    #  UniProtEntity.uniprot_ids and referencing a unique ProteinMetadata
    #  The below were never tested
    # existing_uniprot_entities_stmt = \
    #     select(UniProtProteinAssociation.protein)\
    #     .where(UniProtProteinAssociation.uniprot_id.in_(possibly_new_uniprot_ids))
    #     # NEED TO GROUP THESE BY ProteinMetadata.uniprot_entities
    # OR
    # existing_uniprot_entities_stmt = \
    #     select(wrapapi.UniProtEntity).join(ProteinMetadata)\
    #     .where(wrapapi.UniProtEntity.uniprot_id.in_(possibly_new_uniprot_ids))
    #     # NEED TO GROUP THESE BY ProteinMetadata.uniprot_entities

    # Map the existing uniprot_id to UniProtEntity
    uniprot_id_to_unp_entity = {unp_entity.id: unp_entity for unp_entity in existing_uniprot_entities}
    insert_uniprot_ids = possibly_new_uniprot_ids.difference(uniprot_id_to_unp_entity.keys())

    # Get the remaining UniProtIDs as UniProtEntity entries
    new_uniprot_id_to_unp_entity = {uniprot_id: wrapapi.UniProtEntity(id=uniprot_id)
                                    for uniprot_id in insert_uniprot_ids}
    # Update entire dictionary for ProteinMetadata ops below
    uniprot_id_to_unp_entity.update(new_uniprot_id_to_unp_entity)
    # Insert new
    new_uniprot_entities = list(new_uniprot_id_to_unp_entity.values())
    session.add_all(new_uniprot_entities)

    # Repeat the process for ProteinMetadata
    # Map entity_id to uniprot_id for later cleaning of UniProtEntity
    possibly_new_entity_id_to_uniprot_ids = \
        {protein_data.entity_id: uniprot_ids
         for uniprot_ids, protein_datas in possibly_new_uniprot_to_prot_data.items()
         for protein_data in protein_datas}
    # Map entity_id to ProteinMetadata
    possibly_new_entity_id_to_protein_data = \
        {protein_data.entity_id: protein_data
         for protein_datas in possibly_new_uniprot_to_prot_data.values()
         for protein_data in protein_datas}
    possibly_new_entity_ids = set(possibly_new_entity_id_to_protein_data.keys())

    if existing_protein_metadata is None:
        existing_protein_metadata = []
    else:
        existing_protein_metadata = list(existing_protein_metadata)

    existing_entity_ids = {protein_data.entity_id for protein_data in existing_protein_metadata}
    # Remove the certainly existing from possibly new and query the new
    existing_protein_metadata_stmt = \
        select(ProteinMetadata) \
        .where(ProteinMetadata.entity_id.in_(possibly_new_entity_ids.difference(existing_entity_ids)))
    # Add all requested to those known about
    existing_protein_metadata += session.scalars(existing_protein_metadata_stmt).all()

    # Get all the existing ProteinMetadata.entity_ids to handle the certainly new ones
    existing_entity_ids = {protein_data.entity_id for protein_data in existing_protein_metadata}
    # Any remaining entity_ids are new and must be added
    new_entity_ids = possibly_new_entity_ids.difference(existing_entity_ids)
    # uniprot_ids_to_new_metadata = {
    #     possibly_new_entity_id_to_uniprot_ids[entity_id]: possibly_new_entity_id_to_protein_data[entity_id]
    #     for entity_id in new_entity_ids}
    uniprot_ids_to_new_metadata = defaultdict(list)
    for entity_id in new_entity_ids:
        uniprot_ids_to_new_metadata[possibly_new_entity_id_to_uniprot_ids[entity_id]].append(
            possibly_new_entity_id_to_protein_data[entity_id])

    # Add all existing to UniProtIDs to ProteinMetadata mapping
    all_uniprot_id_to_prot_data = defaultdict(list)
    for protein_data in existing_protein_metadata:
        all_uniprot_id_to_prot_data[protein_data.uniprot_ids].append(protein_data)

    # Collect all new ProteinMetadata which remain
    all_protein_metadata = []
    for uniprot_ids, metadatas in uniprot_ids_to_new_metadata.items():
        all_protein_metadata.extend(metadatas)
        # Add to UniProtIDs to ProteinMetadata map
        all_uniprot_id_to_prot_data[uniprot_ids].extend(metadatas)
        # Attach UniProtEntity to new ProteinMetadata by UniProtID
        for protein_metadata in metadatas:
            # Create the ordered_list of UniProtIDs (UniProtEntity) on ProteinMetadata entry
            try:
                # protein_metadata.uniprot_entities.extend(
                #     uniprot_id_to_unp_entity[uniprot_id] for uniprot_id in uniprot_ids)
                protein_metadata.uniprot_entities = \
                    [uniprot_id_to_unp_entity[uniprot_id] for uniprot_id in uniprot_ids]
            except KeyError:
                # uniprot_id_to_unp_entity is missing a key. Not sure why it wouldn't be here...
                raise SymDesignException(putils.report_issue)

    # Insert remaining ProteinMetadata
    session.add_all(all_protein_metadata)
    # # Finalize additions to the database
    # session.commit()

    return all_uniprot_id_to_prot_data


# def choose_insert_dialect(session):
#     # Choose insert dialect
#     if session.bind.dialect.name == 'sqlite':
#         insert = sqlite_insert
#     elif session.bind.dialect.name == 'mysql':
#         insert = mysql_insert
#     else:  # if session.bind.dialect.name == 'postgresql':
#         # insert = *_insert
#         raise ValueError(
#             f"{insert_dataframe.__name__} isn't configured for the dialect={session.bind.dialect.name} yet")
#     return insert
# - DRAFTING SPACE -
# For updating foreign keys, I can use the ORM method, which seems a bit simpler if I refactor properly
# table_instance = table(**record)
# foreign_instance = table1(**other_record)
# table_instance.append(foreign_instance)
# Or using UPDATE table foreign_key_id WHERE table.non_nullable_column == foreign_key_table.non_nullable_column
#  table = table.__table__
#  primary_keys = [key.name for key in table.primary_key]
#  non_null_keys = [col.name for col in table.columns if not col.nullable]
#  index_keys = [key for key in non_null_keys if key not in primary_keys]
# for key in table.foreign_keys:
#  foreign_key_name = key.column.name
#  table2 = key.column.table
#  primary_keys2 = [key.name for key in table2.primary_key]
#  non_null_keys2 = [col.name for col in table2.columns if not col.nullable]
#  index_keys2 = [key for key in non_null_keys2 if key not in primary_keys2]
# table.update()
#      .values({key.parent.name: key.column})
#      .where(*tuple(key1 == key2 for key1, key2 in zip(index_keys1, index_keys2)))


# def which_dialect(session) -> dict[str, bool]:
#     """Provide the database dialect as a dict with the dialect as the key and the value as True"""
#     return {session.bind.dialect.name: True}
def insert_dataframe(session: Session, table: Base, df: pd.DataFrame, mysql: bool = False, **kwargs):
    """Take a formatted pandas DataFrame and insert values into a sqlalchemy session, then commit the transaction

    Args:
        session: A currently open transaction within sqlalchemy
        table: A Class mapped to SQL table with sqlalchemy
        df: The DataFrame with records to insert
        mysql: Whether the database is a MySQL dialect
    """
    if mysql:
        insert = mysql_insert
    else:
        insert = sqlite_insert

    insert_stmt = insert(table)
    # # Get the columns that should be updated
    # new_columns = df.columns.tolist()
    # # logger.debug(f'Provided columns: {new_columns}')
    # excluded_columns = insert_stmt.excluded
    # update_columns = [c for c in excluded_columns if c.name in new_columns]
    # update_dict = {getattr(c, 'name'): c for c in update_columns if not c.primary_key}
    # table_ = table.__table__
    # # Find relevant column indicators to parse the non-primary key non-nullable columns
    # primary_keys = [key for key in table_.primary_key]
    # non_null_keys = [col for col in table_.columns if not col.nullable]
    # index_keys = [key for key in non_null_keys if key not in primary_keys]

    # do_update_stmt = insert_stmt.on_conflict_do_update(
    #     index_elements=index_keys,  # primary_keys,
    #     set_=update_dict
    # )
    # # Can't insert with .returning() until version 2.0...
    # # try:
    # #     result = session.execute(do_update_stmt.returning(table_.id), df.reset_index().to_dict('records'))
    # # except exc.CompileError as error:
    # #     logger.error(error)
    # #     try:
    # #         result = session.execute(insert_stmt.returning(table_.id), df.reset_index().to_dict('records'))
    # #     except exc.CompileError as _error:
    # #         logger.error(_error)
    # # try:
    # This works using insert with conflict, however, doesn't return the auto-incremented ids
    # result = session.execute(do_update_stmt, df.to_dict('records'))
    # result = session.execute(insert_stmt, df.to_dict('records'))
    start_time = time()
    session.execute(insert_stmt, df.to_dict('records'))
    logger.debug(f'Transaction with table "{table.__tablename__}" took {time() - start_time:8f}s')

    # session.commit()
    # # foreign_key = [key.column.name for key in table_.foreign_keys]
    # for key in table_.foreign_keys:
    #     foreign_key_name = key.column.name
    #     table2_ = key.column.table
    #     # Repeat the Find procedure for table2_
    #     # Find relevant column indicators to parse the non-primary key non-nullable columns
    #     primary_keys2 = [key for key in table2_.primary_key]
    #     non_null_keys2 = [col for col in table2_.columns if not col.nullable]
    #     index_keys2 = [key for key in non_null_keys2 if key not in primary_keys2]
    #     # Todo this statement fails due to the error:
    #     #  This backend (sqlite) does not support multiple-table criteria within UPDATE
    #     #  This doesn't appear to be a multiple-table update, but a multiple-table criteria,
    #     #  which is supported by sqlite...
    #     # foreign_key_update_stmt = table.update()\
    #     #     .values({key.parent.name: key.column})\
    #     #     .where(*tuple(key1 == key2 for key1, key2 in zip(index_keys, index_keys2)))
    #     # logger.info(foreign_key_update_stmt)
    #
    #     select_stmt = select(key.column).where(key.parent.is_(None))\
    #         .where(*tuple(key1 == key2 for key1, key2 in zip(index_keys, index_keys2))).scalar_subquery()
    #     foreign_key_update_stmt2 = table.update()\
    #         .values({key.parent.name: select_stmt})
    #     logger.info(foreign_key_update_stmt2)
    #     # session.execute(foreign_key_update_stmt)
    #     start_time = time()
    #     session.execute(foreign_key_update_stmt2)
    #     logger.info(f'Transaction took {time() - start_time:8f}s')

    # session.commit()
    # return result

    # # ORM based method which updates objects with each row .id (Takes much longer time
    # # - https://benchling.engineering/sqlalchemy-batch-inserts-a-module-for-when-youre-inserting-thousands-of-rows-and-it-s-slow-16ece0ef5bf7
    # new_objects = [table(**record) for record in df.reset_index().to_dict('records')]
    # # select(table)
    # session.add_all(new_objects)
    # session.commit()
    # # result = [_object.id for _object in new_objects]
    # return [_object.id for _object in new_objects]


def upsert_dataframe(session: Session, table: Base, df: pd.DataFrame, mysql: bool = False, **kwargs):
    """Take a formatted pandas DataFrame and insert/update values into a sqlalchemy session, then commit the transaction

    Args:
        session: A currently open transaction within sqlalchemy
        table: A Class mapped to SQL table with sqlalchemy
        df: The DataFrame with records to insert
        mysql: Whether the database is a MySQL dialect
    """
    if mysql:
        insert_stmt = mysql_insert(table)
        excluded_columns = insert_stmt.inserted
    else:
        insert_stmt = sqlite_insert(table)
        excluded_columns = insert_stmt.excluded

    # Get the columns that should be updated
    new_columns = df.columns.tolist()
    # logger.debug(f'Provided columns: {new_columns}')
    update_columns = [c for c in excluded_columns if c.name in new_columns]
    update_dict = {c.name: c for c in update_columns if not c.primary_key}
    tablename = table.__tablename__
    if mysql:
        do_update_stmt = insert_stmt.on_duplicate_key_update(
            update_dict
        )
    else:  # SQLite and postgresql are the same
        # Find relevant column indicators to parse the non-primary key non-nullable columns
        unique_constraints = inspect(session.connection()).get_unique_constraints(tablename)
        # Returns
        #  [{'name': '_pose_design_uc', 'column_names': ['pose_id', 'design_id']}]
        table_unique_constraint_keys = set()
        for constraint in unique_constraints:
            table_unique_constraint_keys.update(constraint['column_names'])

        table_ = table.__table__
        unique_constraint_keys = {col.name for col in table_.columns if col.unique}
        index_keys = unique_constraint_keys.union(table_unique_constraint_keys)
        # primary_keys = [key for key in table_.primary_key]
        # non_null_keys = [col for col in table_.columns if not col.nullable]
        # index_keys = [key for key in non_null_keys if key not in primary_keys] \
        #     + unique_constraint_keys
        do_update_stmt = insert_stmt.on_conflict_do_update(
            index_elements=index_keys,  # primary_keys,
            set_=update_dict
        )
    # Todo Error
    #  sqlalchemy.exc.OperationalError:
    #  MySQLdb._exceptions.OperationalError:
    #    (1213, 'Deadlock found when trying to get lock; try restarting transaction')
    start_time = time()
    session.execute(do_update_stmt, df.to_dict('records'))
    logger.debug(f'Transaction with table "{tablename}" took {time() - start_time:8f}s')
    # session.commit()

    # return result.scalars().all()


def format_residues_df_for_write(df: pd.DataFrame) -> pd.DataFrame:
    """Take a typical per-residue DataFrame and orient the top column level (level=0) containing the residue numbers on
    the index innermost level

    Args:
        df: A per-residue DataFrame to transform
    Returns:
        The transformed DataFrame
    """
    # df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
    # # residue_metric_columns = residues.columns.levels[-1].tolist()
    # # self.log.debug(f'Residues metrics present: {residue_metric_columns}')

    # Place the residue indices from the column names into the index at position -1
    df = df.stack(0)
    df.index.set_names('index', level=-1, inplace=True)

    return df


pd.set_option('future.no_silent_downcasting', True)


def write_dataframe(session: Session, designs: pd.DataFrame = None,
                    design_residues: pd.DataFrame = None, entity_designs: pd.DataFrame = None,
                    poses: pd.DataFrame = None, pose_residues: pd.DataFrame = None, residues: pd.DataFrame = None,
                    update: bool = True, transaction_kwargs: dict = dict()):
    """Format each possible DataFrame type for output via csv or SQL database

    Args:
        session: A currently open transaction within sqlalchemy
        designs: The typical per-design metric DataFrame where each index is the design id and the columns are
            design metrics
        design_residues: The typical per-residue metric DataFrame where each index is the design id and the columns
            are (residue index, Boolean for design utilization)
        entity_designs: The typical per-design metric DataFrame for Entity instances where each index is the design id
            and the columns are design metrics
        poses: The typical per-pose metric DataFrame where each index is the pose id and the columns are
            pose metrics
        pose_residues: The typical per-residue metric DataFrame where each index is the design id and the columns are
            (residue index, residue metric)
        residues: The typical per-residue metric DataFrame where each index is the design id and the columns are
            (residue index, residue metric)
        update: Whether the output identifiers are already present in the metrics
        transaction_kwargs: Any keyword arguments that should be passed for the transaction. Automatically populated
            with the database backend as located from the session
    """
    #     job: The resources for the current job
    if update:
        dataframe_function = upsert_dataframe
    else:
        dataframe_function = insert_dataframe

    # If this is the first call, update the dictionary to specify the database dialect
    if transaction_kwargs == dict():
        transaction_kwargs.update({session.bind.dialect.name: True})
        # transaction_kwargs.update(which_dialect(session))
    # else:
    #     input(transaction_kwargs)
    # warn = warned = False
    #
    # def warn_multiple_update_results():
    #     nonlocal warned
    #     if warn and not warned:
    #         logger.warning(
    #             "Performing multiple metrics SQL transactions will only return results for the last transaction")
    #         warned = True
    replace_values = {np.nan: None, float('inf'): 1e6, float('-inf'): -1e6}

    if poses is not None and not poses.empty:
        # warn = True
        df = poses.replace(replace_values).reset_index()
        table = PoseMetrics
        dataframe_function(session, table=table, df=df, **transaction_kwargs)
        logger.info(f'Wrote {table.__tablename__} to Database')

    if designs is not None and not designs.empty:
        # warn_multiple_update_results()
        # warn = True
        df = designs.replace(replace_values).reset_index()
        table = DesignMetrics
        dataframe_function(session, table=table, df=df, **transaction_kwargs)
        logger.info(f'Wrote {table.__tablename__} to Database')

    if entity_designs is not None and not entity_designs.empty:
        # warn_multiple_update_results()
        # warn = True
        df = entity_designs.replace(replace_values).reset_index()
        table = DesignEntityMetrics
        dataframe_function(session, table=table, df=df, **transaction_kwargs)
        logger.info(f'Wrote {table.__tablename__} to Database')

    if design_residues is not None and not design_residues.empty:
        # warn_multiple_update_results()
        # warn = True
        df = format_residues_df_for_write(design_residues).replace(replace_values).reset_index()
        table = DesignResidues
        dataframe_function(session, table=table, df=df, **transaction_kwargs)
        logger.info(f'Wrote {table.__tablename__} to Database')

    if residues is not None and not residues.empty:
        # warn_multiple_update_results()
        # warn = True
        df = format_residues_df_for_write(residues).replace(replace_values).reset_index()
        table = ResidueMetrics
        dataframe_function(session, table=table, df=df, **transaction_kwargs)
        logger.info(f'Wrote {table.__tablename__} to Database')

    if pose_residues is not None and not pose_residues.empty:
        # warn_multiple_update_results()
        # warn = True
        df = format_residues_df_for_write(pose_residues).replace(replace_values).reset_index()
        table = PoseResidueMetrics
        dataframe_function(session, table=table, df=df, **transaction_kwargs)
        logger.info(f'Wrote {table.__tablename__} to Database')
