from __future__ import annotations

import os
from itertools import combinations
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
# from mysql.connector import MySQLConnection, Error
from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean, select, UniqueConstraint
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import declarative_base, relationship, Session, column_property
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


class PoseMetadata(Base):
    __tablename__ = 'pose_data'
    id = Column(Integer, primary_key=True)

    __table_args__ = (UniqueConstraint('project', 'name', name='_project_name_uc'),
                      )
    project = Column(String, nullable=False)  # Todo? , index=True)
    name = Column(String, nullable=False, index=True)
    # This isn't a column in the __table__, but is an attribute of Class and derived instances
    pose_identifier = column_property(project + os.sep + name)
    # pose_identifier = column_property(f'{project}{os.sep}{name}')

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
    source_path = Column(String)
    # Symmetry
    sym_entry_number = Column(Integer)
    symmetry = Column(String)  # Result
    symmetry_dimension = Column(Integer)
    """The result of the SymEntry"""
    # symmetry_groups = relationship('SymmetryGroup')
    sym_entry_specification = Column(String)  # RESULT:{SUBSYMMETRY1}{SUBSYMMETRY2}...

    # @property
    # def symmetry_groups(self) -> list[str]:
    #     return [entity.symmetry for entity in self.entity_data]
    #
    # @property
    # def entity_names(self) -> list[str]:
    #     """Provide the names of all Entity instances mapped to the Pose"""
    #     return [entity.entity_name for entity in self.entity_data]


class PoseMetrics(Base):
    __tablename__ = 'pose_metrics'

    id = Column(Integer, primary_key=True)
    # name = Column(String, nullable=False, index=True)  # String(60)
    # project = Column(String)  # , nullable=False)  # String(60)
    # Set up one-to-one relationship with pose_data table
    pose_id = Column(ForeignKey('pose_data.id'), nullable=False, unique=True)
    pose = relationship('PoseJob', back_populates='metrics')

    # design_ids = relationship('DesignMetrics', back_populates='pose')
    number_of_designs = Column(Integer)
    # Dock features
    proteinmpnn_dock_cross_entropy_loss = Column(Float)
    proteinmpnn_dock_cross_entropy_per_residue = Column(Float)
    proteinmpnn_v_design_probability_cross_entropy_loss = Column(Float)
    proteinmpnn_v_design_probability_cross_entropy_per_residue = Column(Float)
    proteinmpnn_v_evolution_probability_cross_entropy_loss = Column(Float)
    proteinmpnn_v_evolution_probability_cross_entropy_per_residue = Column(Float)
    proteinmpnn_v_fragment_probability_cross_entropy_loss = Column(Float)
    proteinmpnn_v_fragment_probability_cross_entropy_per_residue = Column(Float)
    dock_collapse_significance_by_contact_order_z_mean = Column(Float)
    dock_collapse_increased_z_mean = Column(Float)
    dock_collapse_sequential_peaks_z_mean = Column(Float)
    dock_collapse_sequential_z_mean = Column(Float)
    dock_collapse_deviation_magnitude = Column(Float)
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
    # Fragment features
    nanohedra_score_normalized = Column(Float)
    nanohedra_score_center_normalized = Column(Float)
    nanohedra_score = Column(Float)
    nanohedra_score_center = Column(Float)
    number_fragment_residues_total = Column(Integer)
    number_fragment_residues_center = Column(Integer)
    multiple_fragment_ratio = Column(Float)
    percent_fragment_helix = Column(Float)
    percent_fragment_strand = Column(Float)
    percent_fragment_coil = Column(Float)
    number_of_fragments = Column(Integer)
    percent_residues_fragment_interface_total = Column(Float)
    percent_residues_fragment_interface_center = Column(Float)
    percent_residues_non_fragment_interface = Column(Float)
    number_interface_residues_non_fragment = Column(Float)
    # design_dimension = Column(Integer)
    # entity_max_radius = Column(Float)  # Has a # after entity LIST
    # entity_min_radius = Column(Float)  # Has a # after entity LIST
    # entity_name = Column(String(30))  # Has a # after entity LIST
    # entity_number_of_residues = Column(Integer)  # Has a # after entity LIST
    # entity_radius = Column(Float)  # Has a # after entity LIST
    # entity_symmetry_group = Column(String(4))  # Has a # after entity LIST
    # entity_n_terminal_helix = Column(Boolean)  # Has a # after entity LIST
    # entity_c_terminal_helix = Column(Boolean)  # Has a # after entity LIST
    # entity_n_terminal_orientation = Column(Boolean)  # Has a # after entity LIST
    # entity_c_terminal_orientation = Column(Boolean)  # Has a # after entity LIST
    # entity_thermophilicity = Column(Boolean)  # Has a # after entity LIST
    # interface1_secondary_structure_fragment_count = Column(Integer)  # Added after the fact
    # interface1_secondary_structure_fragment_topology = Column(String)  # Added after the fact
    # interface1_secondary_structure_count = Column(Integer)  # Added after the fact
    # interface1_secondary_structure_topology = Column(String)  # Added after the fact
    # interface2_secondary_structure_fragment_count = Column(Integer)  # Added after the fact
    # interface2_secondary_structure_fragment_topology = Column(String)  # Added after the fact
    # interface2_secondary_structure_count = Column(Integer)  # Added after the fact
    # interface2_secondary_structure_topology = Column(String)  # Added after the fact
    # entity_radius_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    # entity_min_radius_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    # entity_max_radius_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    # entity_number_of_residues_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    entity_max_radius_average_deviation = Column(Float)
    entity_min_radius_average_deviation = Column(Float)
    entity_number_of_residues_average_deviation = Column(Float)
    entity_radius_average_deviation = Column(Float)
    interface_b_factor_per_residue = Column(Float)
    interface_secondary_structure_fragment_topology = Column(String(120))
    interface_secondary_structure_fragment_count = Column(Integer)
    interface_secondary_structure_topology = Column(String(120))
    interface_secondary_structure_count = Column(Integer)
    minimum_radius = Column(Float)
    maximum_radius = Column(Float)
    number_interface_residues = Column(Integer)
    # number_design_residues = Column(Integer)
    # sequence = Column(String(config.MAXIMUM_SEQUENCE))
    pose_length = Column(Integer)
    pose_thermophilicity = Column(Float)
    """Thermophilicity implies this is a spectrum, while thermophilic implies binary"""


interface_pose_metrics = dict(
    interface_secondary_structure_fragment_count=Integer,
    interface_secondary_structure_fragment_topology=String,
    interface_secondary_structure_count=Integer,
    interface_secondary_structure_topology=String
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
    uniprot = relationship('UniProtEntity', back_populates='_protein_metadata')
    protein = relationship('ProteinMetadata', back_populates='_uniprot_entities')
    position = Column(Integer)


class ProteinMetadata(Base):
    """Used for hold fixed metadata of protein structures, typically pulled from PDB API"""
    __tablename__ = 'protein_metadata'
    id = Column(Integer, primary_key=True)

    entity_id = Column(String, nullable=False, index=True, unique=True)  # entity_name is used in config.metrics
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
                                     order_by=UniProtProteinAssociation.position
                                     )
    uniprot_entities = association_proxy('_uniprot_entities', 'uniprot',
                                         creator=lambda _unp_ent: UniProtProteinAssociation(uniprot=_unp_ent))
    # Set up one-to-many relationship with entity_data table
    entity_data = relationship('EntityData', back_populates='meta')

    model_source = Column(String)
    refined = Column(Boolean, default=True)
    loop_modeled = Column(Boolean, default=True)
    reference_sequence = Column(String)
    # number_of_residues = Column(Integer)  # entity_ is used in config.metrics
    n_terminal_helix = Column(Boolean)  # entity_ is used in config.metrics
    c_terminal_helix = Column(Boolean)  # entity_ is used in config.metrics
    thermophilicity = Column(Boolean)  # entity_ is used in config.metrics
    # Symmetry parameters
    symmetry_group = Column(String)  # entity_ is used in config.metrics
    # symmetry = Column(ForeignKey('symmetry_groups.id'))

    @property
    def uniprot_ids(self) -> dict[str, dict[str, Any]]:
        """Access the UniProtID's associated with this instance"""
        return tuple(entity.id for entity in self.uniprot_entities)

    @property
    def entity_info(self) -> dict[str, dict[str, Any]]:
        """Format the instance for population of Structure metadata via the entity_info kwargs"""
        return {self.entity_id:
                dict(chains=[],
                     dbref=dict(accession=self.uniprot_ids, db=utils.UKB),
                     reference_sequence=self.reference_sequence,
                     thermophilicity=self.thermophilicity)
                }


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
                        lazy='joined', innerjoin=True)
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


class EntityMetrics(Base):
    __tablename__ = 'entity_metrics'
    id = Column(Integer, primary_key=True)

    # # Set up many-to-one relationship with pose_data table
    # pose_id = Column(ForeignKey('pose_data.id'))
    # pose = relationship('PoseMetadata', back_populates='entity_metrics')
    # Set up one-to-one relationship with entity_data table
    entity_id = Column(ForeignKey('entity_data.id'))
    entity = relationship('EntityData', back_populates='metrics')

    number_of_residues = Column(Integer)  # entity_ is used in config.metrics
    max_radius = Column(Float)  # entity_ is used in config.metrics
    min_radius = Column(Float)  # entity_ is used in config.metrics
    radius = Column(Float)  # entity_ is used in config.metrics
    n_terminal_orientation = Column(Integer)  # entity_ is used in config.metrics
    c_terminal_orientation = Column(Integer)  # entity_ is used in config.metrics
    interface_secondary_structure_fragment_topology = Column(String(60))  # entity_ is used in config.metrics
    interface_secondary_structure_topology = Column(String(60))  # entity_ is used in config.metrics


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


# Add metrics which are dependent on multiples. Initialize Column() when setattr() is called to get correct column name
# entity_transformation_metrics = dict(
#     rotation=Float,
#     setting_matrix=Integer,
#     internal_translation=Float,
#     external_translation_x=Float,
#     external_translation_y=Float,
#     external_translation_z=Float,
# )
# for idx in range(1, 1 + config.MAXIMUM_ENTITIES):
#     for metric, value in entity_transformation_metrics.items():
#         setattr(PoseMetrics, f'{metric}{idx}', Column(value))

# entity_pose_metrics = dict(
#     entity_max_radius=Float,
#     entity_min_radius=Float,
#     entity_name=String(30),
#     entity_number_of_residues=Integer,
#     entity_radius=Float,
#     entity_symmetry_group=String(4),
#     entity_n_terminal_helix=Boolean,
#     entity_c_terminal_helix=Boolean,
#     entity_n_terminal_orientation=Integer,
#     entity_c_terminal_orientation=Integer,
#     entity_thermophilicity=Boolean,
#     entity_interface_secondary_structure_fragment_topology=String(60),
#     entity_interface_secondary_structure_topology=String(60),
# )
# for idx in range(1, 1 + config.MAXIMUM_ENTITIES):
#     for metric, value in entity_pose_metrics.items():
#         setattr(PoseMetrics, metric.replace('entity', f'entity{idx}'), Column(value))

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


class DesignProtocol(Base):
    __tablename__ = 'design_protocol'

    # class ProtocolMetadata(Base):
    #     __tablename__ = 'protocol_metadata'
    id = Column(Integer, primary_key=True)

    protocol = Column(String, nullable=False)
    # # Set up many-to-one relationship with protocols table
    # protocol_id = Column(ForeignKey('protocols.id'), nullable=False)
    # protocol = relationship('Protocol', back_populates='designs')
    # Set up many-to-one relationship with design_data table
    design_id = Column(ForeignKey('design_data.id'))
    design = relationship('DesignData', back_populates='protocols')
    temperature = Column(Float)
    file = Column(String)


class DesignData(Base):
    """Account for design metadata created from pose metadata"""
    __tablename__ = 'design_data'
    id = Column(Integer, primary_key=True)
    __table_args__ = (UniqueConstraint('pose_id', 'name', name='_pose_name_uc'),
                      )
    name = Column(String, nullable=False)  # , unique=True)
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
    residues = relationship('ResidueMetrics', back_populates='design')

    structure_path = Column(String)
    sequence = association_proxy('metrics', 'sequence')

    def __str__(self):
        return self.name


class DesignMetrics(Base):
    __tablename__ = 'design_metrics'
    id = Column(Integer, primary_key=True)

    # name = Column(String, nullable=False)  # String(60)
    # pose = Column(String, nullable=False)  # String(60)
    # pose_name = Column(String, nullable=False)  # String(60)
    # Set up one-to-one relationship with design_data table
    design_id = Column(ForeignKey('design_data.id'), nullable=False, unique=True)
    design = relationship('DesignData', back_populates='metrics')

    # Pose features
    # number_interface_residues = Column(Integer)
    contact_order = Column(Float)
    # Design metrics
    number_design_residues = Column(Integer)  # ResidueMetrics sum 'design_residue', nullable=False)
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
    # Sequence metrics
    percent_mutations = Column(Float)
    number_of_mutations = Column(Integer)  # ResidueMetrics sum 'mutation'
    # SymDesign metrics
    interface_local_density = Column(Float)
    interface_composition_similarity = Column(Float)
    interface_area_to_residue_surface_ratio = Column(Float)
    sequence = Column(String(config.MAXIMUM_SEQUENCE))
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
    number_of_hbonds = Column(Integer)
    # column name is changed
    area_hydrophobic_complex = Column(Float)
    area_hydrophobic_unbound = Column(Float)
    area_polar_complex = Column(Float)
    area_polar_unbound = Column(Float)
    area_total_complex = Column(Float)
    area_total_unbound = Column(Float)
    interface_area_polar = Column(Float)
    interface_area_hydrophobic = Column(Float)
    interface_area_total = Column(Float)
    # Done column name is changed
    coordinate_constraint = Column(Float)
    residue_favored = Column(Float)
    # SymDesign measurements
    errat_deviation = Column(Float)
    # sasa_hydrophobic_complex = Column(Float)
    # sasa_polar_complex = Column(Float)
    # NOT taken sasa_relative_complex = Column(Float)
    # sasa_hydrophobic_bound = Column(Float)
    # sasa_polar_bound = Column(Float)
    # NOT taken sasa_relative_bound = Column(Float)
    interior = Column(Float)
    surface = Column(Float)
    support = Column(Float)
    rim = Column(Float)
    core = Column(Float)
    percent_interface_area_hydrophobic = Column(Float)
    percent_interface_area_polar = Column(Float)
    percent_core = Column(Float)
    percent_rim = Column(Float)
    percent_support = Column(Float)
    # Collapse measurements
    # NOT taken hydrophobic_collapse = Column(Float)
    hydrophobicity = Column(Float)
    collapse_deviation_magnitude = Column(Float)
    collapse_increase_significance_by_contact_order_z = Column(Float)
    collapse_increased_z = Column(Float)
    collapse_new_positions = Column(Integer)
    collapse_new_position_significance = Column(Float)
    collapse_sequential_peaks_z = Column(Float)
    collapse_sequential_z = Column(Float)
    collapse_significance_by_contact_order_z = Column(Float)
    collapse_variance = Column(Float)
    collapse_violation = Column(Float)
    collapse_significance_by_contact_order_z_mean = Column(Float)
    collapse_increased_z_mean = Column(Float)
    collapse_sequential_peaks_z_mean = Column(Float)
    collapse_sequential_z_mean = Column(Float)
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
    # Direct coupling analysis energy
    dca_energy = Column(Float)
    # -----------------------
    # Alphafold metrics
    plddt = Column(Float)
    predicted_aligned_error = Column(Float)
    predicted_aligned_interface = Column(Float)
    predicted_interface_template_modeling_score = Column(Float)
    predicted_template_modeling_score = Column(Float)
    rmsd_predicted_models = Column(Float)
    # def __repr__(self):
    #     return f"Trajectory(id={self.id!r}, pose={self.pose!r}, name={self.name!r})"


interface_design_metrics = dict(
    buried_unsatisfied_hbonds_unbound=Integer
)
for idx in range(1, 1 + config.MAXIMUM_INTERFACES):
    for metric, value in interface_design_metrics.items():
        setattr(DesignMetrics, f'{metric}{idx}', Column(value))


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
    number_of_mutations = Column(Integer)  # entity_ is in config.metrics. ResidueMetrics sum 'mutation'
    # Alphafold metrics
    plddt = Column(Float)  # entity_ is in config.metrics
    predicted_aligned_error = Column(Float)  # entity_ is in config.metrics
    predicted_aligned_interface = Column(Float)  # entity_ is in config.metrics
    predicted_interface_template_modeling_score = Column(Float)  # entity_ is in config.metrics
    predicted_template_modeling_score = Column(Float)  # entity_ is in config.metrics
    rmsd_oligomer = Column(Float)  # entity_ is in config.metrics
    rmsd_predicted_models = Column(Float)  # entity_ is in config.metrics


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
    design = relationship('DesignData', back_populates='residues')

    # Residue index (surrogate for residue number) and type information
    index = Column(Integer, nullable=False)
    type = Column(String(1), nullable=False)
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
    errat_deviation = Column(Float)
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
    # Direct coupling analysis energy
    dca_energy = Column(Float)


# class Mysql:
#
#     __instance   = None
#     __host       = None
#     __user       = None
#     __password   = None
#     __database   = None
#     __session    = None
#     __connection = None
#     __dictionary = False
#
#     def __new__(cls, *args, **kwargs):
#         if not cls.__instance or not cls.__database:
#             cls.__instance = super(Mysql, cls).__new__(cls)  # , *args, **kwargs)
#         return cls.__instance
#
#     def __init__(self, host='localhost', user='root', password='', database=''):
#         self.__host     = host
#         self.__user     = user
#         self.__password = password
#         self.__database = database
#
#     def __open(self):
#         cnx = None
#         try:
#             cnx = MySQLConnection(host=self.__host, user=self.__user, password=self.__password, database=self.__database)
#             self.__connection = cnx
#             self.__session    = cnx.cursor(dictionary=self.__dictionary)
#         except Error as e:
#             print('Error %d: %s' % (e.args[0], e.args[1]))
#
#     def __close(self):
#         self.__session.close()
#         self.__connection.close()
#
#     def dictionary_on(self):
#         self.__dictionary = True
#
#     def dictionary_off(self):
#         self.__dictionary = False
#
#     def concatenate_table_columns(self, source, columns):
#         table_columns = []
#         for column in columns:
#             table_columns.append(source + '.' + column)
#         return tuple(table_columns)
#
#     def select(self, table, where=None, *args, **kwargs):
#         result = None
#         query = 'SELECT '
#         if not args:
#             print('TRUE')
#             keys = ['*']
#         else:
#             keys = args
#         print(keys)
#         values = tuple(kwargs.values())
#         l = len(keys) - 1
#
#         for i, key in enumerate(keys):
#             query += key
#             if i < l:
#                 query += ","
#
#         query += ' FROM %s' % table
#
#         if where:
#             query += " WHERE %s" % where
#
#         print(query)
#         self.__open()
#         self.__session.execute(query, values)
# #         number_rows = self.__session.rowcount
# #         number_columns = len(self.__session.description)
# #         print(number_rows, number_columns)
# #         if number_rows >= 1 and number_columns > 1:
#         result = [item for item in self.__session.fetchall()]
# #         else:
# #             print('if only')
# #             result = [item[0] for item in self.__session.fetchall()]
#         self.__close()
#
#         return result
#
#     def join(self, table1, table2, where=None, join_type='inner'):
#         """SELECT
#             m.member_id,
#             m.name member,
#             c.committee_id,
#             c.name committee
#         FROM
#             members m
#         INNER JOIN committees c USING(name);"""
#
#     def insert_into(self, table, columns, source=None, where=None, equivalents=None, *args, **kwargs):
#         query = "INSERT INTO %s SET " % table
#         table_columns = self.concatenate_table_columns(table, columns)
#
#         if source:
#             column_equivalents = self.concatenate_table_columns(source, columns)
#             # column_and_equivalents = [(columns[i], column_equivalents[i]) for i in range(len(columns))]
#             pre_query_columns = [column + ' = %s' for column in table_columns]
#             pre_query_columns = [pre_query_columns[i] % column_equivalents[i] for i in range(len(column_equivalents))]
#             # print(pre_query_columns)
#             query += ", ".join(pre_query_columns) # + ") VALUES (" + ", ".join(["%s"] * len(columns)) + ")"
#
#         if kwargs:
#             keys   = kwargs.keys()
#             values = tuple(kwargs.values()) + tuple(args)
#             l = len(keys) - 1
#             for i, key in enumerate(keys):
#                 query += key+ " = %s"
#                 if i < l:
#                     query += ","
#                 ## End if i less than 1
#             ## End for keys
#         if where:
#             query += " WHERE %s" % where
#
# #         print(query)
# #         stop = columns[30]
#         self.__open()
#         self.__session.execute(query) #, values)
#         self.__connection.commit()
#
#         # Obtain rows affected
#         update_rows = self.__session.rowcount
#         self.__close()
#
#         return update_rows
#
#     def insert(self, table, columns, values, *args, **kwargs):
#         # values = None
#         query = "INSERT INTO %s " % table
#         query += "(" + ",".join(["%s"] * len(columns)) % tuple(columns) + ") VALUES (" + ",".join(["%s"] * len(columns)) + ")"
#
#         if kwargs:
#             keys = kwargs.keys()
#             values = tuple(kwargs.values())
#             query += "(" + ",".join(["%s"] * len(keys)) % tuple(keys) + ") VALUES (" + ",".join(["%s"] * len(values)) + ")"
#         elif args:
#             values = args
#             query += " VALUES(" + ",".join(["%s"] * len(values[0][0])) + ")"
#
# #         print(query)
# #         print(values)
#
#         self.__open()
#         self.__session.execute(query, values)
#         self.__connection.commit()
#         self.__close()
#
#         return self.__session.lastrowid
#
#     def insert_multiple(self, table, columns, values, *args, **kwargs):
#         # values = None
#         query = "INSERT INTO %s " % table
#         query += "(" + ",".join(["`%s`"] * len(columns)) % tuple(columns) + ") VALUES (" + ",".join(["%s"] * len(columns)) + ")"
#
#         if kwargs:
#             keys = kwargs.keys()
#             values = tuple(kwargs.values())
#             query += "(" + ",".join(["`%s`"] * len(keys)) % tuple(keys) + ") VALUES (" + ",".join(["%s"] * len(values)) + ")"
#         elif args:
#             query += "(" + ",".join(["`%s`"] * len(columns)) % tuple(columns) + ")"
#             query += " VALUES(" + ",".join(["%s"] * len(args)) + ")"
#
#         # print(query)
#         # print(values[0][0])
#         self.__open()
#         self.__session.executemany(query, values)
#         self.__connection.commit()
#         self.__close()
#
#         return self.__session.lastrowid
#
#     def delete(self, table, where=None, *args):
#         query = "DELETE FROM %s" % table
#         if where:
#             query += ' WHERE %s' % where
#
#         values = tuple(args)
#
#         self.__open()
#         self.__session.execute(query, values)
#         self.__connection.commit()
#
#         # Obtain rows affected
#         delete_rows = self.__session.rowcount
#         self.__close()
#
#         return delete_rows
#
#     def select_advanced(self, sql, *args):
#         od = OrderedDict(args)
#         query  = sql
#         values = tuple(od.values())
#         self.__open()
#         self.__session.execute(query, values)
#         number_rows = self.__session.rowcount
#         number_columns = len(self.__session.description)
#
#         if number_rows >= 1 and number_columns > 1:
#             result = [item for item in self.__session.fetchall()]
#         else:
#             print('if only')
#             result = [item[0] for item in self.__session.fetchall()]
#
#         self.__close()
#
#         return result
