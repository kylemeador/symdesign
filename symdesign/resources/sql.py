from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean, select, UniqueConstraint
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import column_property, declarative_base, relationship, Session
# from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, declarative_base  # Todo sqlalchemy 2.0
# from sqlalchemy import create_engine
# from sqlalchemy.dialects.sqlite import insert

from . import config
from symdesign.utils import symmetry
from .query import utils
# from symdesign import resources


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
                           order_by='DesignData.id', lazy='selectin')

    @property
    def number_of_designs(self) -> int:
        return len(self.designs)

    @property
    def number_of_entities(self) -> int:
        """Return the number of distinct entities (Gene/Protein products) found in the PoseMetadata"""
        return len(self.entity_data)

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
    symmetry_dimension = Column(Integer)
    """The result of the SymEntry"""
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
        """Format the instance for population of Structure metadata via the entity_info kwargs"""
        return {self.entity_id:
                dict(chains=[],
                     dbref=dict(accession=self.uniprot_ids, db=utils.UKB),
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

    # Todo set up the loading from database on these relationships
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
        """Format the instance for population of Structure metadata via the entity_info kwargs"""
        return self.meta.entity_info

    @property
    def transformation(self) -> dict:  # transformation_mapping
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
    def transformation(self) -> dict:  # Todo -> transformation_mapping:
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
    def transformation(self, transform):  # Todo transformation_mapping):
        if any((self.rotation_x, self.rotation_y, self.rotation_z,
                self.internal_translation_x, self.internal_translation_y, self.internal_translation_z,
                self.setting_matrix,
                self.external_translation_x, self.external_translation_y, self.external_translation_z)):
            raise RuntimeError("Can't set the transformation as this would disrupt the persistence of the table"
                               f"{EntityTransform.__tablename__}")

        if not isinstance(transform, dict):
            raise ValueError(f'The attribute transformation must be a Sequence of '
                             f'transformation_mapping, not {type(transform[0]).__name__}')
                             # f'{transformation_mapping.__name__}, not {type(transform[0]).__name__}')

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
            'ca_only', 'contiguous_ghosts', 'evolution_constraint',
            'initial_z_value', 'interface', 'match_value',
            'minimum_matched', 'neighbors', 'number_predictions',
            'prediction_model', 'term_constraint', 'use_gpu_relax',
            name='_pose_name_uc'
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
    initial_z_value = Column(Integer)  # dock
    interface = Column(Boolean)  # design
    match_value = Column(Integer)  # dock
    minimum_matched = Column(Integer)  # dock
    neighbors = Column(Boolean)  # design
    number_predictions = Column(Integer)  # structure-predict
    prediction_model = Column(String(5))  # structure-predict # ENUM - all, best, none
    term_constraint = Column(Boolean)  # design
    use_gpu_relax = Column(Boolean)  # structure-predict


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
    alphafold_model = Column(Float)


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


# This has 105 columns. Most of these don't get filled unless there is a Structure
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
    proteinmpnn_score_delta = Column(Float)
    proteinmpnn_score_delta_per_designed_residue = Column(Float)
    proteinmpnn_score_unbound = Column(Float)
    proteinmpnn_score_unbound_per_designed_residue = Column(Float)
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
    interaction_energy_per_residue = Column(Float)
    interface_separation = Column(Float)
    rmsd_complex = Column(Float)
    rosetta_reference_energy = Column(Float)
    shape_complementarity = Column(Float)
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
    spatial_aggregation_propensity = Column(Float)
    spatial_aggregation_propensity_unbound = Column(Float)
    # Alphafold metrics
    plddt = Column(Float)
    plddt_deviation = Column(Float)
    predicted_aligned_error = Column(Float)
    predicted_aligned_error_deviation = Column(Float)
    predicted_aligned_interface = Column(Float)
    predicted_aligned_interface_deviation = Column(Float)
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
    predicted_aligned_interface = Column(Float)  # entity_ is in config.metrics
    predicted_aligned_interface_deviation = Column(Float)  # entity_ is in config.metrics
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


# This has 54 columns, most of which rely on Structures
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

