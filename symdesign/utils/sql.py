from __future__ import annotations

from collections import OrderedDict
from itertools import combinations

from mysql.connector import MySQLConnection, Error
from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean, select, Table
from sqlalchemy.orm import declarative_base, relationship, Session
# from sqlalchemy.orm import Mapped, mapped_column, declarative_base  # Todo sqlalchemy 2.0
# from sqlalchemy import create_engine
# from sqlalchemy.dialects.sqlite import insert

from symdesign.resources import config


# class Base(DeclarativeBase):  # Todo sqlalchemy 2.0
class _Base:

    # def next_primary_key(self, session: Session):
    #     stmt = select(self).order_by(tuple(key.desc() for key in self.__table__.primary_key)).limit(1)
    def next_key(self, session: Session) -> int:
        stmt = select(self).order_by(self.id.desc()).limit(1)
        return session.scalars(stmt).first() + 1


Base = declarative_base(cls=_Base)


class SymmetryGroup(Base):
    __tablename__ = 'symmetry_groups'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    rotation = Column(Float, nullable=False)


class PoseMetadata(Base):
    __tablename__ = 'pose_metadata'
    id = Column(Integer, primary_key=True)

    name = Column(String, nullable=False, index=True)  # String(60)
    project = Column(String, nullable=False)  # String(60)
    pose_identifier = column_property(f'{project}{os.sep}{name}')
    # Set up one-to-many relationship with design_metadata table
    designs = relationship('DesignMetadata', back_populates='pose')
    # # Set up one-to-many relationship with entity_metadata table
    # Set up many-to-many relationship with entity_metadata table
    entity_metadata = relationship('EntityMetadata', secondary='pose_entity_association',
                                   back_populates='poses')
    # Set up one-to-many relationship with residue_metrics table
    residues = relationship('ResidueMetrics', back_populates='pose')

    # # Set up many-to-one relationship with entity_metadata table
    # entity_metadata = Column(ForeignKey('entity_metadata.id'))

    # Set up one-to-many relationship with entity_metrics table
    entity_metrics = relationship('EntityMetrics', back_populates='pose')
    # Set up one-to-one relationship with pose_metrics table
    metrics = relationship('PoseMetrics', back_populates='pose', uselist=False)

    # State
    _pre_refine = Column('pre_refine', Boolean, default=True)
    _pre_loop_model = Column('pre_loop_model', Boolean, default=True)
    # Symmetry
    sym_entry_number = Column(Integer)
    symmetry = Column(String)  # Result
    symmetry_dimension = Column(Integer)
    """The result of the SymEntry"""
    # symmetry_groups = relationship('SymmetryGroup')
    sym_entry_specification = Column(String)  # RESULT:{SUBSYMMETRY1}{SUBSYMMETRY2}...

    # @classmethod
    # def insert_pose(cls, session: Session, name: str, project: str) -> int:
    #     """Insert a new PoseJob instance into the database and return the Database id"""
    #     # with session() as session:
    #     stmt = insert(cls).returning(cls.id)
    #     result = session.scalars(stmt,  # execute(stmt).inserted_primary_key  # fetchone()
    #                              dict(name=name, project=project))
    #     session.commit()
    #
    #     return result.one()
    #
    # def get_design_number(self, session: Session, number: int) -> int:
    #     return self.number_of_designs
    #
    # @staticmethod
    # def increment_design_number(cls, session: Session, number: int):
    #     return None
    # OLD ^
    # NEW v
    # @property
    # def symmetry_groups(self) -> list[str]:
    #     return [entity.symmetry for entity in self.entity_metadata]
    #
    # @property
    # def entity_names(self) -> list[str]:
    #     """Provide the names of all Entity instances mapped to the Pose"""
    #     return [entity.entity_name for entity in self.entity_metadata]
    #
    # @property
    # def pose_transformation(self) -> list[transformation_mapping]:
    #     """Provide the names of all Entity instances mapped to the Pose"""
    #     return [dict(
    #         rotation=scipy.spatial.transform.Rotation.from_rotvec([0., 0., entity.rotation], degrees=True).as_matrix(),
    #         translation=np.array([0., 0., entity.internal_translation]),
    #         rotation2=utils.symmetry.setting_matrices[entity.setting_matrix],
    #         translation2=np.array([entity.external_translation_x,
    #                                entity.external_translation_y,
    #                                entity.external_translation_z]),
    #     ) for entity in self.entity_metadata]
    #
    # @pose_transformation.setter
    # def pose_transformation(self, transform: Sequence[transformation_mapping]):
    #     for idx, (entity, operation_set) in enumerate(zip(self.entity_metadata, transform)):
    #         if not isinstance(operation_set, dict):
    #             try:
    #                 raise ValueError(f'The attribute pose_transformation must be a Sequence of '
    #                                  f'{transformation_mapping.__name__}, not {type(transform[0]).__name__}')
    #             except TypeError:  # Not a Sequence
    #                 raise TypeError(f'The attribute pose_transformation must be a Sequence of '
    #                                 f'{transformation_mapping.__name__}, not {type(transform).__name__}')
    #         for operation_type, operation in operation_set:
    #             if operation_type == 'translation':
    #                 _, _, entity.internal_translation = operation
    #             elif operation_type == 'rotation':
    #                 entity.setting_matrix = \
    #                     scipy.spatial.transform.Rotation.from_matrix(operation).as_rotvec(degrees=True)
    #             elif operation_type == 'rotation2':
    #                 entity.setting_matrix = self.sym_entry.setting_matrices_numbers[idx]
    #             elif operation_type == 'translation2':
    #                 entity.external_translation_x, \
    #                     entity.external_translation_y, \
    #                     entity.external_translation_z = operation
    #
    #             # self._pose_transformation = self.info['pose_transformation'] = list(transform)


class PoseMetrics(Base):
    __tablename__ = 'pose_metrics'

    id = Column(Integer, primary_key=True)
    # name = Column(String, nullable=False, index=True)  # String(60)
    # project = Column(String)  # , nullable=False)  # String(60)
    # Set up one-to-one relationship with pose_metadata table
    pose_id = Column(ForeignKey('pose_metadata.id'), nullable=False)
    pose = relationship('PoseMetadata', back_populates='metrics')

    # design_ids = relationship('DesignMetrics', back_populates='pose')
    number_of_designs = Column(Integer)  # , nullable=False)
    # Dock features
    proteinmpnn_v_design_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    proteinmpnn_v_design_probability_cross_entropy_per_residue = Column(Float)  # , nullable=False)
    proteinmpnn_v_evolution_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    proteinmpnn_v_evolution_probability_cross_entropy_per_residue = Column(Float)  # , nullable=False)
    proteinmpnn_v_fragment_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    proteinmpnn_v_fragment_probability_cross_entropy_per_residue = Column(Float)  # , nullable=False)
    dock_collapse_significance_by_contact_order_z_mean = Column(Float)  # , nullable=False)
    dock_collapse_increased_z_mean = Column(Float)  # , nullable=False)
    dock_collapse_sequential_peaks_z_mean = Column(Float)  # , nullable=False)
    dock_collapse_sequential_z_mean = Column(Float)  # , nullable=False)
    dock_collapse_deviation_magnitude = Column(Float)  # , nullable=False)
    dock_collapse_increase_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    dock_collapse_increased_z = Column(Float)  # , nullable=False)
    dock_collapse_new_positions = Column(Integer)  # , nullable=False)
    dock_collapse_new_position_significance = Column(Float)  # , nullable=False)
    dock_collapse_sequential_peaks_z = Column(Float)  # , nullable=False)
    dock_collapse_sequential_z = Column(Float)  # , nullable=False)
    dock_collapse_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    dock_collapse_variance = Column(Float)  # , nullable=False)
    dock_collapse_violation = Column(Boolean)  # , nullable=False)
    dock_hydrophobicity = Column(Float)  # , nullable=False)
    # Fragment features
    nanohedra_score_normalized = Column(Float)  # , nullable=False)
    nanohedra_score_center_normalized = Column(Float)  # , nullable=False)
    nanohedra_score = Column(Float)  # , nullable=False)
    nanohedra_score_center = Column(Float)  # , nullable=False)
    number_fragment_residues_total = Column(Integer)  # , nullable=False)
    number_fragment_residues_center = Column(Integer)  # , nullable=False)
    multiple_fragment_ratio = Column(Float)  # , nullable=False)
    percent_fragment_helix = Column(Float)  # , nullable=False)
    percent_fragment_strand = Column(Float)  # , nullable=False)
    percent_fragment_coil = Column(Float)  # , nullable=False)
    number_of_fragments = Column(Integer)  # , nullable=False)
    percent_residues_fragment_interface_total = Column(Float)  # , nullable=False)
    percent_residues_fragment_interface_center = Column(Float)  # , nullable=False)
    number_interface_residues_non_fragment = Column(Float)  # , nullable=False)
    # Pose features
    # design_dimension = Column(Integer)  # , nullable=False)
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
    # entity_thermophile = Column(Boolean)  # Has a # after entity LIST
    interface1_secondary_structure_fragment_topology = Column(String)
    interface2_secondary_structure_topology = Column(String)
    # entity_radius_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    # entity_min_radius_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    # entity_max_radius_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    # entity_number_of_residues_ratio_v = Column(Float)  # Has a #v# after ratio LIST
    entity_max_radius_average_deviation = Column(Float)  # , nullable=False)
    entity_min_radius_average_deviation = Column(Float)  # , nullable=False)
    entity_number_of_residues_average_deviation = Column(Float)  # , nullable=False)
    entity_radius_average_deviation = Column(Float)  # , nullable=False)
    interface_b_factor_per_residue = Column(Float)  # , nullable=False)
    interface_secondary_structure_fragment_topology = Column(String(120))  # , nullable=False)
    interface_secondary_structure_fragment_count = Column(Integer)  # , nullable=False)
    interface_secondary_structure_topology = Column(String(120))  # , nullable=False)
    interface_secondary_structure_count = Column(Integer)  # , nullable=False)
    minimum_radius = Column(Float)  # , nullable=False)
    maximum_radius = Column(Float)  # , nullable=False)
    number_interface_residues = Column(Integer)  # , nullable=False)
    # number_design_residues = Column(Integer)  # , nullable=False)
    sequence = Column(String(config.MAXIMUM_SEQUENCE))  # , nullable=False)
    pose_length = Column(Integer)  # , nullable=False)
    pose_thermophilicity = Column(Float)  # , nullable=False)


# class PoseEntityAssociation(Base):
pose_entity_association = Table(
    'pose_entity_association',
    Base.metadata,
    Column('pose_id', ForeignKey('pose_metadata.id'), primary_key=True),  # Use the sqlalchemy.Column construct not mapped_column()
    Column('entity_id', ForeignKey('entity_metadata.id'), primary_key=True)  # Use the sqlalchemy.Column construct
)


class EntityMetadata(Base):
    __tablename__ = 'entity_metadata'
    id = Column(Integer, primary_key=True)

    # Set up many-to-many relationship with pose_metadata table
    poses = relationship('PoseMetadata', secondary='pose_entity_association',
                         back_populates='entity_metadata')

    name = Column(String, nullable=False, index=True)  # entity_ is used in config.metrics
    # number_of_residues = Column(Integer)  # entity_ is used in config.metrics
    n_terminal_helix = Column(Boolean)  # entity_ is used in config.metrics
    c_terminal_helix = Column(Boolean)  # entity_ is used in config.metrics
    thermophile = Column(Boolean)  # entity_ is used in config.metrics
    # Symmetry parameters
    symmetry_group = Column(String(4))  # entity_ is used in config.metrics
    symmetry = Column(ForeignKey('symmetry_groups.id'))


class EntityMetrics(Base):
    __tablename__ = 'entity_metrics'
    id = Column(Integer, primary_key=True)

    # Set up many-to-one relationship with pose_metadata table
    pose_id = Column(ForeignKey('pose_metadata.id'))
    pose = relationship('PoseMetadata', back_populates='entity_metrics')

    # Todo new-style EntityMetrics
    number_of_residues = Column(Integer)  # entity_ is used in config.metrics
    max_radius = Column(Float)  # entity_ is used in config.metrics
    min_radius = Column(Float)  # entity_ is used in config.metrics
    radius = Column(Float)  # entity_ is used in config.metrics
    n_terminal_orientation = Column(Integer)  # entity_ is used in config.metrics
    c_terminal_orientation = Column(Integer)  # entity_ is used in config.metrics
    interface_secondary_structure_fragment_topology = Column(String(60))  # entity_ is used in config.metrics
    interface_secondary_structure_topology = Column(String(60))  # entity_ is used in config.metrics
    # Transformation parameters
    rotation = Column(Float)
    setting_matrix = Column(Integer)
    internal_translation = Column(Float)
    external_translation_x = Column(Float)
    external_translation_y = Column(Float)
    external_translation_z = Column(Float)
    # Todo new-style EntityMetrics


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
ratio_design_metrics = dict(
    entity_radius_ratio_v=Float,
    entity_min_radius_ratio_v=Float,
    entity_max_radius_ratio_v=Float,
    entity_number_of_residues_ratio_v=Float,
)
for idx1, idx2 in combinations(range(1, 1 + config.MAXIMUM_ENTITIES), 2):
    for metric, value in ratio_design_metrics.items():
        setattr(PoseMetrics, metric.replace('_v', f'_{idx1}v{idx2}'), Column(value))
# Todo remove for new-style EntityMetrics
entity_pose_metrics = dict(
    entity_max_radius=Float,
    entity_min_radius=Float,
    entity_name=String(30),
    entity_number_of_residues=Integer,
    entity_radius=Float,
    entity_symmetry_group=String(4),
    entity_n_terminal_helix=Boolean,
    entity_c_terminal_helix=Boolean,
    entity_n_terminal_orientation=Integer,
    entity_c_terminal_orientation=Integer,
    entity_thermophile=Boolean,
    entity_interface_secondary_structure_fragment_topology=String(60),
    entity_interface_secondary_structure_topology=String(60),
)
for idx in range(1, 1 + config.MAXIMUM_ENTITIES):
    for metric, value in entity_transformation_metrics.items():
        setattr(Poses, f'{metric}{idx}', Column(value))

class ProtocolMetadata(Base):
    __tablename__ = 'protocol_metadata'
    id = Column(Integer, primary_key=True)

    protocol = Column(String, nullable=False)
    # Set up many-to-one relationship with design_metadata table
    design_id = Column(ForeignKey('design_metadata.id'), nullable=False)
    design = relationship('DesignMetadata', back_populates='protocols')  # , nullable=False)
    temperature = Column(Float)
    file = Column(String)


class DesignMetadata(Base):
    """Account for design metadata created from pose metadata"""
    __tablename__ = 'design_metadata'
    id = Column(Integer, primary_key=True)

    name = Column(String, nullable=False)  # String(60)
    # Set up many-to-one relationship with pose_metadata table
    pose_id = Column(ForeignKey('pose_metadata.id'), nullable=False)
    pose = relationship('PoseMetadata', back_populates='designs')
    # Set up one-to-many relationship with protocol_metadata table
    protocols = relationship('ProtocolMetadata', back_populates='design')
    # Set up one-to-one relationship with design_metrics table
    metrics = relationship('DesignMetrics', back_populates='design', uselist=False)
    # Set up one-to-many relationship with residue_metrics table
    residues = relationship('ResidueMetrics', back_populates='design')


class DesignMetrics(Base):
    __tablename__ = 'design_metrics'
    id = Column(Integer, primary_key=True)

    # name = Column(String, nullable=False)  # String(60)
    # pose = Column(String, nullable=False)  # String(60)
    # pose_name = Column(String, nullable=False)  # String(60)
    # Set up one-to-one relationship with design_metadata table
    design_id = Column(ForeignKey('design_metadata.id'), nullable=False)
    design = relationship('DesignMetadata', back_populates='metrics')

    # Pose features
    number_interface_residues = Column(Integer)  # , nullable=False)
    contact_order = Column(Float)  # , nullable=False)
    # Design metrics
    number_design_residues = Column(Integer)  # ResidueMetrics sum 'design_residue', nullable=False)
    # Rosetta metrics
    buns_complex = Column(Integer)  # , nullable=False)
    # buns_unbound = Column(Integer)  # Has a # after buns LIST
    buried_unsatisfied_hbonds = Column(Integer)  # , nullable=False)
    buried_unsatisfied_hbond_density = Column(Float)  # , nullable=False)
    contact_count = Column(Integer)  # , nullable=False)
    # entity_interface_connectivity = Column(Float)  # Has a # after entity LIST
    favor_residue_energy = Column(Float)  # , nullable=False)
    interaction_energy_complex = Column(Float)  # , nullable=False)
    interface_energy_density = Column(Float)  # , nullable=False)
    interface_bound_activation_energy = Column(Float)  # , nullable=False)
    interface_solvation_energy = Column(Float)  # , nullable=False)
    interface_solvation_energy_activation = Column(Float)  # , nullable=False)
    interaction_energy_per_residue = Column(Float)  # , nullable=False)
    interface_separation = Column(Float)  # , nullable=False)
    rmsd_complex = Column(Float)  # , nullable=False)
    rosetta_reference_energy = Column(Float)  # , nullable=False)
    shape_complementarity = Column(Float)  # , nullable=False)
    # solvation_energy = Column(Float)  # , nullable=False)
    # solvation_energy_complex = Column(Float)  # , nullable=False)
    # Sequence metrics
    percent_mutations = Column(Float)  # , nullable=False)
    number_of_mutations = Column(Integer)  # ResidueMetrics sum 'mutation', nullable=False)
    # entity_percent_mutations = Column(Float)  # Has a # after entity LIST
    # entity_number_of_mutations = Column(Integer)  # Has a # after entity LIST
    # SymDesign metrics
    interface_local_density = Column(Float)  # , nullable=False)
    interface_composition_similarity = Column(Float)  # , nullable=False)
    interface_area_to_residue_surface_ratio = Column(Float)  # , nullable=False)
    sequence = Column(String(config.MAXIMUM_SEQUENCE))  # , nullable=False)
    # Summed ResidueMetrics metrics
    # -----------------------
    # column name is changed from Rosetta energy values
    interface_energy_complex = Column(Float)  # , nullable=False)
    interface_energy_bound = Column(Float)  # , nullable=False)
    interface_energy_unbound = Column(Float)  # , nullable=False)
    interface_energy = Column(Float)  # , nullable=False)
    interface_solvation_energy_complex = Column(Float)  # , nullable=False)
    interface_solvation_energy_bound = Column(Float)  # , nullable=False)
    interface_solvation_energy_unbound = Column(Float)  # , nullable=False)
    number_of_hbonds = Column(Integer)  # , nullable=False)
    # column name is changed
    area_hydrophobic_complex = Column(Float)  # , nullable=False)
    area_hydrophobic_unbound = Column(Float)  # , nullable=False)
    area_polar_complex = Column(Float)  # , nullable=False)
    area_polar_unbound = Column(Float)  # , nullable=False)
    area_total_complex = Column(Float)  # , nullable=False)
    area_total_unbound = Column(Float)  # , nullable=False)
    interface_area_polar = Column(Float)  # , nullable=False)
    interface_area_hydrophobic = Column(Float)  # , nullable=False)
    interface_area_total = Column(Float)  # , nullable=False)
    # Done column name is changed
    coordinate_constraint = Column(Float)  # , nullable=False)
    residue_favored = Column(Float)  # , nullable=False)
    # SymDesign measurements
    errat_deviation = Column(Float)  # , nullable=False)
    sasa_hydrophobic_complex = Column(Float)  # , nullable=False)
    sasa_polar_complex = Column(Float)  # , nullable=False)
    # NOT taken sasa_relative_complex = Column(Float)  # , nullable=False)
    sasa_hydrophobic_bound = Column(Float)  # , nullable=False)
    sasa_polar_bound = Column(Float)  # , nullable=False)
    # NOT taken sasa_relative_bound = Column(Float)  # , nullable=False)
    interior = Column(Float)  # , nullable=False)
    surface = Column(Float)  # , nullable=False)
    support = Column(Float)  # , nullable=False)
    rim = Column(Float)  # , nullable=False)
    core = Column(Float)  # , nullable=False)
    percent_interface_area_hydrophobic = Column(Float)  # , nullable=False)
    percent_interface_area_polar = Column(Float)  # , nullable=False)
    percent_core = Column(Float)  # , nullable=False)
    percent_rim = Column(Float)  # , nullable=False)
    percent_support = Column(Float)  # , nullable=False)
    # Collapse measurements
    # NOT taken hydrophobic_collapse = Column(Float)  # , nullable=False)
    hydrophobicity = Column(Float)  # , nullable=False)
    collapse_deviation_magnitude = Column(Float)  # , nullable=False)
    collapse_increase_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    collapse_increased_z = Column(Float)  # , nullable=False)
    collapse_new_positions = Column(Integer)  # , nullable=False)
    collapse_new_position_significance = Column(Float)  # , nullable=False)
    collapse_sequential_peaks_z = Column(Float)  # , nullable=False)
    collapse_sequential_z = Column(Float)  # , nullable=False)
    collapse_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    collapse_variance = Column(Float)  # , nullable=False)
    collapse_violation = Column(Float)  # , nullable=False)
    collapse_significance_by_contact_order_z_mean = Column(Float)  # , nullable=False)
    collapse_increased_z_mean = Column(Float)  # , nullable=False)
    collapse_sequential_peaks_z_mean = Column(Float)  # , nullable=False)
    collapse_sequential_z_mean = Column(Float)  # , nullable=False)
    # ProteinMPNN score terms
    proteinmpnn_loss_complex = Column(Float)  # , nullable=False)
    proteinmpnn_loss_unbound = Column(Float)  # , nullable=False)
    proteinmpnn_score_complex = Column(Float)
    proteinmpnn_score_complex_per_designed_residue = Column(Float)
    proteinmpnn_score_delta = Column(Float)
    proteinmpnn_score_delta_per_designed_residue = Column(Float)
    proteinmpnn_score_unbound = Column(Float)
    proteinmpnn_score_unbound_per_designed_residue = Column(Float)
    # Sequence loss terms
    sequence_loss_design = Column(Float)  # , nullable=False)
    sequence_loss_design_per_residue = Column(Float)  # , nullable=False)
    sequence_loss_evolution = Column(Float)  # , nullable=False)
    sequence_loss_evolution_per_residue = Column(Float)  # , nullable=False)
    sequence_loss_fragment = Column(Float)  # , nullable=False)
    sequence_loss_fragment_per_residue = Column(Float)  # , nullable=False)
    # Observed in profile measurements
    observed_design = Column(Float)  # , nullable=False)
    observed_evolution = Column(Float)  # , nullable=False)
    observed_fragment = Column(Float)  # , nullable=False)
    observed_interface = Column(Float)  # , nullable=False)
    # Direct coupling analysis energy
    dca_energy = Column(Float)  # , nullable=False)
    # -----------------------

    # def __repr__(self):
    #     return f"Trajectory(id={self.id!r}, pose={self.pose!r}, name={self.name!r})"


# Add metrics which are dependent on multiples. Initialize Column() when setattr() is called to get correct column name
entity_design_metrics = dict(
    entity_interface_connectivity=Float,
    entity_percent_mutations=Float,
    entity_number_of_mutations=Integer,
)
for idx in range(1, 1 + config.MAXIMUM_ENTITIES):
    for metric, value in entity_design_metrics.items():
        setattr(DesignMetrics, metric.replace('entity', f'entity{idx}'), Column(value))
interface_design_metrics = dict(
    buns_unbound=Integer
)
for idx in range(1, 1 + config.MAXIMUM_INTERFACES):
    for metric, value in interface_design_metrics.items():
        setattr(DesignMetrics, metric.replace('buns', f'buns{idx}'), Column(value))


class ResidueMetrics(Base):
    __tablename__ = 'residue_metrics'
    id = Column(Integer, primary_key=True)

    # Set up many-to-one relationship with design_metadata table
    pose_id = Column(ForeignKey('pose_metadata.id'))
    pose = relationship('PoseMetadata', back_populates='residues')
    # Set up many-to-one relationship with design_metadata table
    design_id = Column(ForeignKey('design_metadata.id'))
    design = relationship('DesignMetadata', back_populates='residues')

    # pose = Column(String, nullable=False)  # String(60)
    # design = Column(String, nullable=False)  # String(60)
    # pose_name = Column(String, nullable=False)  # String(60)
    # pose_id = Column(ForeignKey('pose_metadata.id'))  # String, nullable=False)  # String(60)
    # design_name = Column(String, nullable=False)

    # Residue position (surrogate for residue number) and type information
    index = Column(Integer, nullable=False)
    type = Column(String(1), nullable=False)
    design_residue = Column(Boolean)
    interface_residue = Column(Boolean)
    mutation = Column(Boolean)
    # Rosetta energy values
    complex = Column(Float)  # , nullable=False)
    bound = Column(Float)  # , nullable=False)
    unbound = Column(Float)  # , nullable=False)
    energy_delta = Column(Float)  # , nullable=False)
    solv_complex = Column(Float)  # , nullable=False)
    solv_bound = Column(Float)  # , nullable=False)
    solv_unbound = Column(Float)  # , nullable=False)
    hbond = Column(Boolean)  # , nullable=False)
    coordinate_constraint = Column(Float)  # , nullable=False)
    residue_favored = Column(Float)  # , nullable=False)
    # SymDesign measurements
    contact_order = Column(Float)  # , nullable=False)
    errat_deviation = Column(Float)  # , nullable=False)
    sasa_hydrophobic_complex = Column(Float)  # , nullable=False)
    sasa_polar_complex = Column(Float)  # , nullable=False)
    sasa_relative_complex = Column(Float)  # , nullable=False)
    sasa_hydrophobic_bound = Column(Float)  # , nullable=False)
    sasa_polar_bound = Column(Float)  # , nullable=False)
    sasa_relative_bound = Column(Float)  # , nullable=False)
    bsa_hydrophobic = Column(Float)  # , nullable=False)
    bsa_polar = Column(Float)  # , nullable=False)
    bsa_total = Column(Float)  # , nullable=False)
    sasa_total_bound = Column(Float)  # , nullable=False)
    sasa_total_complex = Column(Float)  # , nullable=False)
    interior = Column(Float)  # , nullable=False)
    surface = Column(Float)  # , nullable=False)
    support = Column(Float)  # , nullable=False)
    rim = Column(Float)  # , nullable=False)
    core = Column(Float)  # , nullable=False)
    # Collapse measurements
    collapse_deviation_magnitude = Column(Float)  # , nullable=False)
    collapse_increase_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    collapse_increased_z = Column(Float)  # , nullable=False)
    collapse_new_positions = Column(Boolean)  # , nullable=False)
    collapse_new_position_significance = Column(Float)  # , nullable=False)
    collapse_sequential_peaks_z = Column(Float)  # , nullable=False)
    collapse_sequential_z = Column(Float)  # , nullable=False)
    collapse_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    hydrophobic_collapse = Column(Float)  # , nullable=False)
    # ProteinMPNN score terms
    proteinmpnn_loss_complex = Column(Float)  # , nullable=False)
    proteinmpnn_loss_unbound = Column(Float)  # , nullable=False)
    sequence_loss_design = Column(Float)  # , nullable=False)
    sequence_loss_evolution = Column(Float)  # , nullable=False)
    sequence_loss_fragment = Column(Float)  # , nullable=False)
    proteinmpnn_v_design_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    proteinmpnn_v_evolution_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    proteinmpnn_v_fragment_probability_cross_entropy_loss = Column(Float)  # , nullable=False)
    dock_collapse_deviation_magnitude = Column(Float)  # , nullable=False)
    dock_collapse_increase_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    dock_collapse_increased_z = Column(Float)  # , nullable=False)
    dock_collapse_new_positions = Column(Boolean)  # , nullable=False)
    dock_collapse_new_position_significance = Column(Float)  # , nullable=False)
    dock_collapse_sequential_peaks_z = Column(Float)  # , nullable=False)
    dock_collapse_sequential_z = Column(Float)  # , nullable=False)
    dock_collapse_significance_by_contact_order_z = Column(Float)  # , nullable=False)
    dock_hydrophobic_collapse = Column(Float)  # , nullable=False)
    # Observed in profile measurements
    observed_design = Column(Boolean)  # , nullable=False)
    observed_evolution = Column(Boolean)  # , nullable=False)
    observed_fragment = Column(Boolean)  # , nullable=False)
    observed_interface = Column(Boolean)  # , nullable=False)
    # Direct coupling analysis energy
    dca_energy = Column(Float)  # , nullable=False)

    # user = relationship("User", back_populates="addresses")

    # def __repr__(self):
    #     return f"ResidueMetrics(id={self.id!r}, index={self.index!r})"


# def start_db(db: AnyStr):
#     """"""
#     # engine = create_engine('sqlite+pysqlite:///:memory:', echo=True, future=True)
#     engine = create_engine(f'sqlite:///{db}', echo=True, future=True)
#     Base.metadata.create_all(engine)
#
#     return engine


# db = '/absolute/path/to/foo.db'
# engine = start_db(db)
#
# with Session(engine) as session:
#
#     session.add_all([spongebob, sandy, patrick])
#
#     session.commit()
#
#
# # Example of starting a Session and changing data in a table
# session = Session(engine)
# stmt = select(ResidueMetrics).where(ResidueMetrics.index.in_(["spongebob", "sandy"]))
#
# for residue in session.scalars(stmt):
#     print(residue)
#     residue.complex = 3.1
#
# session.commit()


class Mysql:

    __instance   = None
    __host       = None
    __user       = None
    __password   = None
    __database   = None
    __session    = None
    __connection = None
    __dictionary = False

    def __new__(cls, *args, **kwargs):
        if not cls.__instance or not cls.__database:
            cls.__instance = super(Mysql, cls).__new__(cls)  # , *args, **kwargs)
        return cls.__instance

    def __init__(self, host='localhost', user='root', password='', database=''):
        self.__host     = host
        self.__user     = user
        self.__password = password
        self.__database = database

    def __open(self):
        cnx = None
        try:
            cnx = MySQLConnection(host=self.__host, user=self.__user, password=self.__password, database=self.__database)
            self.__connection = cnx
            self.__session    = cnx.cursor(dictionary=self.__dictionary)
        except Error as e:
            print('Error %d: %s' % (e.args[0], e.args[1]))

    def __close(self):
        self.__session.close()
        self.__connection.close()
        
    def dictionary_on(self):
        self.__dictionary = True
        
    def dictionary_off(self):
        self.__dictionary = False
        
    def concatenate_table_columns(self, source, columns):
        table_columns = []
        for column in columns:
            table_columns.append(source + '.' + column)
        return tuple(table_columns)

    def select(self, table, where=None, *args, **kwargs):
        result = None
        query = 'SELECT '
        if not args:
            print('TRUE')
            keys = ['*']
        else:
            keys = args
        print(keys)
        values = tuple(kwargs.values())
        l = len(keys) - 1

        for i, key in enumerate(keys):
            query += key
            if i < l:
                query += ","

        query += ' FROM %s' % table

        if where:
            query += " WHERE %s" % where

        print(query)
        self.__open()
        self.__session.execute(query, values)
#         number_rows = self.__session.rowcount
#         number_columns = len(self.__session.description)
#         print(number_rows, number_columns)
#         if number_rows >= 1 and number_columns > 1:
        result = [item for item in self.__session.fetchall()]
#         else:
#             print('if only')
#             result = [item[0] for item in self.__session.fetchall()]
        self.__close()

        return result
    
    def join(self, table1, table2, where=None, join_type='inner'):
        """SELECT
            m.member_id, 
            m.name member, 
            c.committee_id, 
            c.name committee
        FROM
            members m
        INNER JOIN committees c USING(name);"""

    def insert_into(self, table, columns, source=None, where=None, equivalents=None, *args, **kwargs):           
        query = "INSERT INTO %s SET " % table
        table_columns = self.concatenate_table_columns(table, columns)
        
        if source:
            column_equivalents = self.concatenate_table_columns(source, columns)
            # column_and_equivalents = [(columns[i], column_equivalents[i]) for i in range(len(columns))]
            pre_query_columns = [column + ' = %s' for column in table_columns]
            pre_query_columns = [pre_query_columns[i] % column_equivalents[i] for i in range(len(column_equivalents))]
            # print(pre_query_columns)
            query += ", ".join(pre_query_columns) # + ") VALUES (" + ", ".join(["%s"] * len(columns)) + ")"
        
        if kwargs:
            keys   = kwargs.keys()
            values = tuple(kwargs.values()) + tuple(args)
            l = len(keys) - 1
            for i, key in enumerate(keys):
                query += key+ " = %s"
                if i < l:
                    query += ","
                ## End if i less than 1
            ## End for keys
        if where:
            query += " WHERE %s" % where
            
#         print(query)
#         stop = columns[30]
        self.__open()
        self.__session.execute(query) #, values)
        self.__connection.commit()

        # Obtain rows affected
        update_rows = self.__session.rowcount
        self.__close()

        return update_rows

    def insert(self, table, columns, values, *args, **kwargs):
        # values = None
        query = "INSERT INTO %s " % table
        query += "(" + ",".join(["%s"] * len(columns)) % tuple(columns) + ") VALUES (" + ",".join(["%s"] * len(columns)) + ")"

        if kwargs:
            keys = kwargs.keys()
            values = tuple(kwargs.values())
            query += "(" + ",".join(["%s"] * len(keys)) % tuple(keys) + ") VALUES (" + ",".join(["%s"] * len(values)) + ")"
        elif args:
            values = args
            query += " VALUES(" + ",".join(["%s"] * len(values[0][0])) + ")"
        
#         print(query)
#         print(values)

        self.__open()
        self.__session.execute(query, values)
        self.__connection.commit()
        self.__close()
        
        return self.__session.lastrowid
    
    def insert_multiple(self, table, columns, values, *args, **kwargs):
        # values = None
        query = "INSERT INTO %s " % table
        query += "(" + ",".join(["`%s`"] * len(columns)) % tuple(columns) + ") VALUES (" + ",".join(["%s"] * len(columns)) + ")"
        
        if kwargs:
            keys = kwargs.keys()
            values = tuple(kwargs.values())
            query += "(" + ",".join(["`%s`"] * len(keys)) % tuple(keys) + ") VALUES (" + ",".join(["%s"] * len(values)) + ")"
        elif args:
            query += "(" + ",".join(["`%s`"] * len(columns)) % tuple(columns) + ")"
            query += " VALUES(" + ",".join(["%s"] * len(args)) + ")"
        
        # print(query)
        # print(values[0][0])
        self.__open()
        self.__session.executemany(query, values)
        self.__connection.commit()
        self.__close()
        
        return self.__session.lastrowid

    def delete(self, table, where=None, *args):
        query = "DELETE FROM %s" % table
        if where:
            query += ' WHERE %s' % where

        values = tuple(args)

        self.__open()
        self.__session.execute(query, values)
        self.__connection.commit()

        # Obtain rows affected
        delete_rows = self.__session.rowcount
        self.__close()

        return delete_rows

    def select_advanced(self, sql, *args):
        od = OrderedDict(args)
        query  = sql
        values = tuple(od.values())
        self.__open()
        self.__session.execute(query, values)
        number_rows = self.__session.rowcount
        number_columns = len(self.__session.description)

        if number_rows >= 1 and number_columns > 1:
            result = [item for item in self.__session.fetchall()]
        else:
            result = [item[0] for item in self.__session.fetchall()]

        self.__close()
        
        return result
