from __future__ import annotations

import logging
from time import time

import numpy as np
import pandas as pd
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, UniqueConstraint, inspect
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, relationship

from symdesign.resources.config import MAXIMUM_SEQUENCE
from symdesign.resources.sql import Base, DesignResidues

logger = logging.getLogger(__name__)


class PoseMetrics(Base):
    """Has 52 measures"""
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
MAXIMUM_INTERFACES = 2
for idx in range(1, 1 + MAXIMUM_INTERFACES):
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


class EntityMetrics(Base):
    """Has 9 measurements"""
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


# Todo
#  class StructureDesignMetrics(Base):
#  class SequenceDesignMetrics(Base):
class DesignMetrics(Base):
    """This has 105 columns. Most of these don't get filled unless there is a Pose structure associated
    3 always
    38 for Sequence
    59 for Structure
    12 for Alphafold
    """
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
    sequence = Column(String(MAXIMUM_SEQUENCE))
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
    contact_order = Column(Float)
    # Todo Duplicated of Pose, unless number of residues/coordinates change
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
for idx in range(1, 1 + MAXIMUM_INTERFACES):
    for metric, value in interface_design_metrics.items():
        setattr(DesignMetrics, f'{metric}{idx}', Column(value))


class DesignEntityMetrics(Base):
    """This has 18 columns:
    3 always
    3 for Design
    12 for AlphaFold
    """
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


class PoseResidueMetrics(Base):
    """Has 15 columns"""
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


class ResidueMetrics(Base):
    """This has 54 columns, most of which rely on structures.
    None (besides design_residue) has a use in the processing of modules... just used to produce DesignMetrics
    """
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


def _insert_dataframe(session: Session, table: Base, df: pd.DataFrame, mysql: bool = False, **kwargs):
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


def _upsert_dataframe(session: Session, table: Base, df: pd.DataFrame, mysql: bool = False, **kwargs):
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


# Ensure that the df.stack works across versions


def _format_residues_df_for_write(df: pd.DataFrame) -> pd.DataFrame:
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


# Suppress this warning
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
        dataframe_function = _upsert_dataframe
    else:
        dataframe_function = _insert_dataframe

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
        df = _format_residues_df_for_write(design_residues).replace(replace_values).reset_index()
        table = DesignResidues
        dataframe_function(session, table=table, df=df, **transaction_kwargs)
        logger.info(f'Wrote {table.__tablename__} to Database')

    if residues is not None and not residues.empty:
        # warn_multiple_update_results()
        # warn = True
        df = _format_residues_df_for_write(residues).replace(replace_values).reset_index()
        table = ResidueMetrics
        dataframe_function(session, table=table, df=df, **transaction_kwargs)
        logger.info(f'Wrote {table.__tablename__} to Database')

    if pose_residues is not None and not pose_residues.empty:
        # warn_multiple_update_results()
        # warn = True
        df = _format_residues_df_for_write(pose_residues).replace(replace_values).reset_index()
        table = PoseResidueMetrics
        dataframe_function(session, table=table, df=df, **transaction_kwargs)
        logger.info(f'Wrote {table.__tablename__} to Database')
