from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean, select, UniqueConstraint
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import declarative_base, relationship, Session  # column_property
# from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, declarative_base  # Todo sqlalchemy 2.0
# from sqlalchemy import create_engine
# from sqlalchemy.dialects.sqlite import insert

from . import config
from .query.pdb import UKB
from symdesign.utils.symmetry import setting_matrices
from symdesign.utils.types import TransformationMapping

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


class PoseMetadata(Base):
    """Has 8 columns"""
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
    #                            back_populates='poses')
    # # Set up one-to-many relationship with entity_metrics table
    # entity_metrics = relationship('EntityMetrics', back_populates='pose')
    # Set up one-to-many relationship with entity_data table
    entity_data = relationship('EntityData', back_populates='pose',
                               order_by='EntityData.id', lazy='selectin')
    # Set up one-to-one relationship with pose_metrics table
    metrics = relationship('PoseMetrics', back_populates='pose', uselist=False, lazy='selectin')
    # Set up one-to-many relationship with design_data table
    _designs = relationship('DesignData', back_populates='pose',
                            # collection_class=ordering_list('id'),
                            order_by='DesignData.id')  # lazy='selectin'

    @property
    def designs(self) -> list[DesignData]:
        return self._designs[1:]

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
    def pose_source(self) -> DesignData:
        """Provide the DesignData for the Pose itself"""
        return self._designs[0]

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


class ProteinMetadata(Base):
    """Used for hold fixed metadata of protein structures, typically pulled from PDB API. Has 10 columns"""
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


class EntityData(Base):
    """Used for unique Pose instances to connect multiple sources of information. Has 3 columns"""
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
    def transformation(self) -> TransformationMapping | dict:
        try:
            return self.transform.transformation
        except AttributeError:  # self.transform is probably None
            return {}

    # @property
    # def uniprot_id(self):
    #     return self.meta.uniprot_id


class EntityTransform(Base):
    """Has 14 columns"""
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
    def transformation(self) -> TransformationMapping | dict:
        """Provide the names of all Entity instances mapped to the Pose"""
        # Todo hook in self.rotation_x, self.rotation_y
        if self.setting_matrix is None:
            rotation2 = None
            # rotation2 = Rotation.from_rotvec([self.external_rotation_x,
            #                                   self.external_rotation_y,
            #                                   self.external_rotation_z], degrees=True).as_matrix()
        else:
            rotation2 = setting_matrices[self.setting_matrix]
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
    def transformation(self, transform: TransformationMapping):
        if any((self.rotation_x, self.rotation_y, self.rotation_z,
                self.internal_translation_x, self.internal_translation_y, self.internal_translation_z,
                self.setting_matrix,
                self.external_translation_x, self.external_translation_y, self.external_translation_z)):
            raise RuntimeError(
                "Can't set the transformation as this would disrupt the persistence of the table "
                f"{EntityTransform.__tablename__}")

        if not isinstance(transform, dict):
            raise ValueError(
                f"The attribute 'transformation' must be a {TransformationMapping.__name__}, not "
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
                for number, matrix in setting_matrices.items():
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
               f'{self.id=}' \
               f', {self.module=}' \
               f', {self.commit=}' \
               f', {self.ca_only=}' \
               f', {self.proteinmpnn_model_name=}' \
               f', {self.contiguous_ghosts=}' \
               f', {self.evolution_constraint=}' \
               f', {self.initial_z_value=}' \
               f', {self.interface=}' \
               f', {self.match_value=}' \
               f', {self.minimum_matched=}' \
               f', {self.neighbors=}' \
               f', {self.number_predictions=}' \
               f', {self.prediction_model=}' \
               f', {self.term_constraint=}' \
               f', {self.use_gpu_relax=}' \
               ')'.replace('self.', '')


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


class DesignResidues(Base):
    """Has 4 columns"""
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
