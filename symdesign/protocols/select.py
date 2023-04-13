from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from glob import glob
from itertools import repeat, count
from typing import Any, Iterable, Sequence

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select
from tqdm import tqdm

from . import cluster
from .pose import PoseJob
import symdesign.utils.path as putils
from symdesign import flags, metrics, utils
from symdesign.resources import sql, config
from symdesign.resources.job import job_resources_factory
from symdesign.resources.query.utils import input_string, boolean_choice, validate_input
from symdesign.structure.model import Model
from symdesign.sequence import constants, optimize_protein_sequence, write_sequences, expression, find_orf_offset, \
    generate_mutations, protein_letters_alph1

logger = logging.getLogger(__name__)


def load_total_dataframe(pose_jobs: Iterable[PoseJob], pose: bool = False) -> pd.DataFrame:
    """Return a pandas DataFrame with the trajectories of every PoseJob loaded and formatted according to the
    design directory and design on the index

    Args:
        pose_jobs: The PoseJob instances for which metrics are desired
        pose: Whether the total dataframe should contain the mean metrics from the pose or each individual design
    """
    all_dfs = []  # None for design in pose_jobs]
    for idx, pose_job in enumerate(pose_jobs):
        try:
            all_dfs.append(pd.read_csv(pose_job.designs_metrics_csv, index_col=0, header=[0]))
        except FileNotFoundError:  # as error
            # results[idx] = error
            logger.warning(f'{pose_job}: No trajectory analysis file found. Skipping')

    if pose:
        for pose_job, df in zip(pose_jobs, all_dfs):
            df.fillna(0., inplace=True)  # Shouldn't be necessary if saved files were formatted correctly
            # try:
            df.drop([index for index in df.index.tolist() if isinstance(index, float)], inplace=True)
            # Get rid of all individual trajectories and std, not mean
            pose_name = pose_job.name
            df.drop([index for index in df.index.tolist() if pose_name in index or 'std' in index], inplace=True)
            # except TypeError:
            #     for index in df.index.tolist():
            #         print(index, type(index))
    else:  # designs
        for pose_job, df in zip(pose_jobs, all_dfs):
            # Get rid of all statistic entries, mean, std, etc.
            pose_name = pose_job.name
            df.drop([index for index in df.index.tolist() if pose_name not in index], inplace=True)

    # Add pose directory str as MultiIndex
    try:
        df = pd.concat(all_dfs, keys=[str(pose_job) for pose_job in pose_jobs])
    except ValueError:  # No objects to concatenate
        raise RuntimeError(f"Didn't find any trajectory information in the provided PoseDirectory instances")
    df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)

    return df


def load_and_format(session: Session, stmt: Select, selected_column_names: Iterable[str]) -> pd.DataFrame:
    """From a SELECTable query, fetch the requested columns/attributes from the database, load into a DataFrame,
    and clean

    Args:
        session: A currently open transaction within sqlalchemy
        stmt: The SELECTable query statement
        selected_column_names: The column names to use during DataFrame construction
    Returns:
        The specified columns/attributes formatted as DataFrame.columns and their rows as the DataFrame.index
    """
    # Apply join condition(s) between each element to resolve.
    df = pd.DataFrame.from_records(session.execute(stmt).all(), columns=selected_column_names)
    logger.debug(f'Loaded DataFrame with primary_id keys: '
                 f'{[key for key in selected_column_names if "id" in key and "residue" not in key]}')

    # Format the dataframe and set the index
    # df = df.sort_index(axis=1).set_index('design_id')
    # Remove completely empty columns such as obs_interface
    df.dropna(how='all', inplace=True, axis=1)
    df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)

    return df


def load_sql_all_metrics_dataframe(session: Session, pose_ids: Iterable[int] = None,
                                   design_ids: Iterable[int] = None) -> pd.DataFrame:
    """Load and format every PoseJob instance's, PoseMetrics, EntityMetrics, DesignMetrics, and DesignEntityMetrics for
    each associated design

    Optionally limit those loaded to certain PoseJob.id's and DesignData.id's

    Args:
        session: A currently open transaction within sqlalchemy
        pose_ids: PoseJob instance identifiers for which metrics are desired
        design_ids: DesignData instance identifiers for which metrics are desired
    Returns:
        A DataFrame formatted with every metric in PoseMetrics, EntityMetrics, and DesignMetrics. The final DataFrame
            will have an as many entries corresponding to each Entity in EntityData for a total of DesignData's X number
            of Entities entries
    """
    pm_c = [c for c in sql.PoseMetrics.__table__.columns if not c.primary_key]
    pm_names = [c.name for c in pm_c]
    dm_c = [c for c in sql.DesignMetrics.__table__.columns if not c.primary_key]
    dm_names = [c.name for c in dm_c]
    # entity_metadata_c = [sql.ProteinMetadata.n_terminal_helix,
    #                      sql.ProteinMetadata.c_terminal_helix,
    #                      sql.ProteinMetadata.thermophilicity]
    em_c = [c for c in (*sql.EntityMetrics.__table__.columns,
                        # *entity_metadata_c,
                        *sql.DesignEntityMetrics.__table__.columns)
            if not c.primary_key]
    # Remove design_id (its duplicated?)
    em_c.pop(em_c.index(sql.DesignEntityMetrics.design_id))
    # Remove entity_id, it's duplicated
    em_c.pop(em_c.index(sql.DesignEntityMetrics.entity_id))
    em_names = [f'entity_{c.name}' if c.name != 'entity_id' else c.name for c in em_c]
    selected_columns = (*pm_c, *dm_c, *em_c)
    selected_column_names = (*pm_names, *dm_names, *em_names)
    # # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    # join_stmt = select(selected_columns).select_from(sql.PoseMetrics)\
    #     .join(sql.EntityData, sql.EntityData.pose_id == sql.PoseMetrics.pose_id)\
    #     .join(sql.EntityMetrics, sql.EntityMetrics.entity_id == sql.EntityData.id)\
    #     .join(sql.DesignData, sql.DesignData.pose_id == sql.PoseMetrics.pose_id, ).join(sql.DesignMetrics)\
    #     .join(sql.DesignEntityMetrics, sql.DesignEntityMetrics.design_id == sql.DesignData.id)

    join_stmt = select(selected_columns).select_from(sql.EntityData) \
        .join(sql.PoseMetrics, sql.PoseMetrics.pose_id == sql.EntityData.pose_id) \
        .join(sql.EntityMetrics, sql.EntityMetrics.entity_id == sql.EntityData.id) \
        .join(sql.DesignEntityMetrics, sql.DesignEntityMetrics.entity_id == sql.EntityData.id) \
        .join(sql.DesignMetrics, sql.DesignMetrics.design_id == sql.DesignEntityMetrics.design_id)

    if pose_ids:
        stmt = join_stmt.where(sql.PoseMetrics.pose_id.in_(pose_ids))
    else:
        stmt = join_stmt

    if design_ids:
        stmt = stmt.where(sql.DesignMetrics.design_id.in_(design_ids))
    else:
        stmt = stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_poses_dataframe(session: Session, pose_ids: Iterable[int] = None) -> pd.DataFrame:
    # , design_ids: Iterable[int] = None
    """Load and format every PoseJob instance's, PoseMetrics and EntityMetrics

    Optionally limit those loaded to certain PoseJob.id's

    Args:
        session: A currently open transaction within sqlalchemy
        pose_ids: PoseJob instance identifiers for which metrics are desired
    Returns:
        A DataFrame formatted with every metric in PoseMetrics and EntityMetrics. The final DataFrame will have an entry
            corresponding to each Entity in EntityData for a total of PoseJob's X number of Entities entries
    """
    #     design_ids: DesignData instance identifiers for which metrics are desired
    # Accessing only the PoseMetrics and EntityMetrics
    pm_c = [c for c in sql.PoseMetrics.__table__.columns if not c.primary_key]
    pm_names = [c.name for c in pm_c]
    # entity_metadata_c = [sql.ProteinMetadata.n_terminal_helix,
    #                      sql.ProteinMetadata.c_terminal_helix,
    #                      sql.ProteinMetadata.thermophilicity]
    # em_c = [c for c in (*sql.EntityMetrics.__table__.columns, *entity_metadata_c) if not c.primary_key]
    em_c = [c for c in sql.EntityMetrics.__table__.columns if not c.primary_key]
    em_names = [f'entity_{c.name}' if c.name != 'entity_id' else c.name for c in em_c]
    # em_c = [c for c in sql.EntityMetrics.__table__.columns if not c.primary_key]
    # em_names = [f'entity_{c.name}' if c.name != 'entity_id' else c.name for c in em_c]
    selected_columns = (*pm_c, *em_c)
    selected_column_names = (*pm_names, *em_names)

    # Construct the SQL query
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(sql.EntityData) \
        .join(sql.PoseMetrics, sql.PoseMetrics.pose_id == sql.EntityData.pose_id) \
        .join(sql.EntityMetrics, sql.EntityMetrics.entity_id == sql.EntityData.id)

    if pose_ids:
        stmt = join_stmt.where(sql.PoseMetrics.pose_id.in_(pose_ids))
    else:
        stmt = join_stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_pose_metrics_dataframe(session: Session, pose_ids: Iterable[int] = None) -> pd.DataFrame:
    """Load and format every PoseJob instance's, PoseMetrics

    Optionally limit those loaded to certain PoseJob.id's

    Args:
        session: A currently open transaction within sqlalchemy
        pose_ids: PoseJob instance identifiers for which metrics are desired
    Returns:
        A DataFrame formatted with every metric in PoseMetrics. The final DataFrame will have an entry corresponding to
            each PoseJob
    """
    # Accessing only the PoseMetrics
    pm_c = [c for c in sql.PoseMetrics.__table__.columns if not c.primary_key]
    # pm_names = [c.name for c in pm_c]
    selected_columns = (*pm_c,)
    selected_column_names = [c.name for c in selected_columns]  # (*pm_names,)

    # Construct the SQL query
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(sql.PoseMetrics)
    if pose_ids:
        stmt = join_stmt.where(sql.PoseMetrics.pose_id.in_(pose_ids))
    else:
        stmt = join_stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_entity_metrics_dataframe(session: Session, pose_ids: Iterable[int] = None,
                                      design_ids: Iterable[int] = None) -> pd.DataFrame:
    """Load and format every PoseJob instance's, EntityMetrics/DesignEntityMetrics

    Optionally limit those loaded to certain PoseJob.id's

    Args:
        session: A currently open transaction within sqlalchemy
        pose_ids: PoseJob instance identifiers for which metrics are desired
        design_ids: DesignData instance identifiers for which metrics are desired
    Returns:
        A DataFrame formatted with the pose_id, EntityMetrics, and DesignEntityMetrics. The final DataFrame will have an
            entry corresponding to each Entity in EntityData for a total of PoseJob's X number of entities entries
    """
    # Accessing only the PoseJob.id and EntityMetrics
    pose_id_c = sql.EntityData.pose_id
    # entity_metadata_c = [sql.ProteinMetadata.n_terminal_helix,
    #                      sql.ProteinMetadata.c_terminal_helix,
    #                      sql.ProteinMetadata.thermophilicity]
    # em_c = [c for c in (*sql.EntityMetrics.__table__.columns, *entity_metadata_c) if not c.primary_key]
    em_c = [c for c in (*sql.EntityMetrics.__table__.columns,
                        # *entity_metadata_c,
                        *sql.DesignEntityMetrics.__table__.columns)
            if not c.primary_key]
    # # Remove design_id
    # em_c.pop(em_c.index(sql.DesignEntityMetrics.design_id))
    # Remove entity_id as entity_id is duplicated
    em_c.pop(em_c.index(sql.DesignEntityMetrics.entity_id))
    em_names = [f'entity_{c.name}' if c.name not in ['entity_id', 'design_id'] else c.name for c in em_c]
    selected_columns = (pose_id_c, *em_c,)
    selected_column_names = (pose_id_c.name, *em_names,)

    # Construct the SQL query
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(sql.EntityData)\
        .join(sql.EntityMetrics, sql.EntityMetrics.entity_id == sql.EntityData.id) \
        .join(sql.DesignEntityMetrics, sql.DesignEntityMetrics.entity_id == sql.EntityData.id)

    if pose_ids:
        stmt = join_stmt.where(sql.EntityData.pose_id.in_(pose_ids))
    else:
        stmt = join_stmt

    if design_ids:
        stmt = stmt.where(sql.DesignEntityMetrics.design_id.in_(design_ids))
    else:
        stmt = stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_design_metrics_dataframe(session: Session, pose_ids: Iterable[int] = None, design_ids: Iterable[int] = None) \
        -> pd.DataFrame:
    """Load and format DesignMetrics/DesignEntityMetrics for each design associated with the PoseJob

    Optionally limit those loaded to certain PoseJob.id's and DesignData.id's

    Args:
        session: A currently open transaction within sqlalchemy
        pose_ids: PoseJob instance identifiers for which metrics are desired
        design_ids: DesignData instance identifiers for which metrics are desired
    Returns:
        A pandas DataFrame formatted with every metric in DesignMetrics/DesignEntityMetrics. The final DataFrame will
            have an entry for each DesignEntity for each DesignData
    """
    # dd_c = [sql.DesignData.pose_id, sql.DesignData.design_id]
    dd_c = (sql.DesignData.pose_id,)
    dd_names = [c.name for c in dd_c]
    dm_c = [c for c in sql.DesignMetrics.__table__.columns if not c.primary_key]
    dm_names = [c.name for c in dm_c]
    # em_c = [c for c in sql.DesignEntityMetrics.__table__.columns if not c.primary_key]
    # # Remove design_id
    # em_c.pop(em_c.index(sql.DesignEntityMetrics.design_id))
    # em_names = [f'entity_{c.name}' if c.name != 'entity_id' else c.name for c in em_c]
    selected_columns = (*dd_c, *dm_c)  # , *em_c)
    selected_column_names = (*dd_names, *dm_names)  # , *em_names)

    # Construct the SQL query
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(sql.DesignData)\
        .join(sql.DesignMetrics)
    if pose_ids:
        stmt = join_stmt.where(sql.DesignData.pose_id.in_(pose_ids))
    else:
        stmt = join_stmt

    if design_ids:
        stmt = stmt.where(sql.DesignData.id.in_(design_ids))
    else:
        stmt = stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_design_entities_dataframe(session: Session, pose_ids: Iterable[int] = None,
                                       design_ids: Iterable[int] = None) -> pd.DataFrame:
    """Load and format DesignEntityMetrics for each design associated with the PoseJob

    Optionally limit those loaded to certain PoseJob.id's and DesignData.id's

    Args:
        session: A currently open transaction within sqlalchemy
        pose_ids: PoseJob instance identifiers for which metrics are desired
        design_ids: DesignData instance identifiers for which metrics are desired
    Returns:
        A pandas DataFrame formatted with every metric in DesignMetrics/DesignEntityMetrics. The final DataFrame will
            have an entry for each DesignEntity for each DesignData
    """
    # dd_c = [sql.DesignData.pose_id, sql.DesignData.design_id]
    dd_c = (sql.DesignData.pose_id,)
    dd_names = [c.name for c in dd_c]
    # dm_c = [c for c in sql.DesignMetrics.__table__.columns if not c.primary_key]
    # dm_names = [c.name for c in dm_c]
    em_c = [c for c in sql.DesignEntityMetrics.__table__.columns if not c.primary_key]
    # Remove design_id
    em_c.pop(em_c.index(sql.DesignEntityMetrics.design_id))
    em_names = [f'entity_{c.name}' if c.name != 'entity_id' else c.name for c in em_c]
    selected_columns = (*dd_c, *em_c)  # *dm_c,
    selected_column_names = (*dd_names, *em_names)  # *dm_names,

    # Construct the SQL query
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(sql.DesignData)\
        .join(sql.DesignEntityMetrics)  # .join(PoseJob)
    if pose_ids:
        stmt = join_stmt.where(sql.DesignData.pose_id.in_(pose_ids))
    else:
        stmt = join_stmt

    if design_ids:
        stmt = stmt.where(sql.DesignData.id.in_(design_ids))
    else:
        stmt = stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_pose_metadata_dataframe(session: Session, pose_ids: Iterable[int] = None,
                                     design_ids: Iterable[int] = None) -> pd.DataFrame:
    """Load and format every PoseJob instance associated metadata including protocol information

    Optionally limit those loaded to certain PoseJob.id's and DesignData.id's

    Args:
        session: A currently open transaction within sqlalchemy
        pose_ids: PoseJob instance identifiers for which metrics are desired
        design_ids: Not used, but here for API. DesignData instance identifiers for which metrics are desired
    Returns:
        The pandas DataFrame formatted with the every metric in DesignMetrics. The final DataFrame will
            have an entry for each DesignData
    """
    selected_columns = PoseJob.__table__.columns
    selected_column_names = ['pose_id' if c.name == 'id' else c.name for c in selected_columns]

    # Construct the SQL query
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(PoseJob)
    if pose_ids:
        stmt = join_stmt.where(PoseJob.id.in_(pose_ids))
    else:
        stmt = join_stmt

    # if design_ids:
    #     stmt = stmt.where(sql.DesignData.id.in_(design_ids))
    # else:
    #     stmt = stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_design_metadata_dataframe(session: Session, pose_ids: Iterable[int] = None,
                                       design_ids: Iterable[int] = None) -> pd.DataFrame:
    """Load and format requested identifiers DesignData/DesignProtocol

    Optionally limit those loaded to certain PoseJob.id's and DesignData.id's

    Args:
        session: A currently open transaction within sqlalchemy
        pose_ids: PoseJob instance identifiers for which metrics are desired
        design_ids: DesignData instance identifiers for which metrics are desired
    Returns:
        The pandas DataFrame formatted with the every metric in DesignMetrics. The final DataFrame will
            have an entry for each DesignData
    """
    dd_c = [c for c in sql.DesignData.__table__.columns if not c.primary_key]
    dd_c.pop(dd_c.index(sql.DesignData.name))
    # dd_names = [c.name for c in dd_c]
    # name REMOVE
    # pose_id NEED
    # design_parent_id
    # structure_path
    # sequence
    dp_c = [c for c in sql.DesignProtocol.__table__.columns if not c.primary_key]
    # dp_names = [c.name for c in dp_c]
    # protocol
    # job_id JOIN
    # design_id NEED
    # file
    # temperature
    # alphafold_model

    job_c = [c for c in sql.JobProtocol.__table__.columns if not c.primary_key]
    # job_names = [c.name for c in job_c]
    selected_columns = (*dp_c, *dd_c, *job_c)
    # selected_column_names = (*dp_names, *dd_names, *job_names)
    selected_column_names = [c.name for c in selected_columns]

    # Construct the SQL query
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(sql.DesignData) \
        .join(sql.DesignProtocol).join(sql.JobProtocol)
    if pose_ids:
        stmt = join_stmt.where(sql.DesignData.pose_id.in_(pose_ids))
    else:
        stmt = join_stmt

    if design_ids:
        stmt = stmt.where(sql.DesignData.id.in_(design_ids))
    else:
        stmt = stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_entity_metadata_dataframe(session: Session, pose_ids: Iterable[int] = None) -> pd.DataFrame:
                                       # design_ids: Iterable[int] = None
    """Load and format every PoseJob instance associated metadata including protocol information

    Optionally limit those loaded to certain PoseJob.id's and DesignData.id's

    Args:
        session: A currently open transaction within sqlalchemy
        pose_ids: PoseJob instance identifiers for which metrics are desired
        # design_ids: DesignData instance identifiers for which metrics are desired
    Returns:
        The pandas DataFrame formatted with the every metric in DesignMetrics. The final DataFrame will
            have an entry for each DesignData
    """
    # pj_c = [PoseJob.id]
    # pj_names = [c.name for c in pj_c]
    pose_id_c = [sql.EntityData.pose_id, sql.EntityData.id]
    pose_id_name = ['entity_id' if c.name == 'id' else c.name for c in pose_id_c]
    em_c = [sql.ProteinMetadata.n_terminal_helix,
            sql.ProteinMetadata.c_terminal_helix,
            sql.ProteinMetadata.entity_id,  # The name of the Entity
            sql.ProteinMetadata.symmetry_group,
            sql.ProteinMetadata.refined,
            sql.ProteinMetadata.loop_modeled,
            # sql.ProteinMetadata.uniprot_ids,  # This is a property...
            sql.ProteinMetadata.thermophilicity]
    # This named entity_name as the external access is marked as entity_name while database access uses entity_id
    em_names = [f'entity_{c.name}' if c.name != 'entity_id' else 'entity_name' for c in em_c]
    uni_c = [sql.UniProtProteinAssociation.uniprot_id]
    uni_names = [c.name for c in uni_c]

    selected_columns = (*pose_id_c, *em_c, *uni_c)
    selected_column_names = (*pose_id_name, *em_names, *uni_names)
    # Construct the SQL query
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(sql.EntityData) \
        .join(sql.ProteinMetadata) \
        .join(sql.UniProtProteinAssociation)

    if pose_ids:
        stmt = join_stmt.where(sql.EntityData.pose_id.in_(pose_ids))
    else:
        stmt = join_stmt

    # if design_ids:
    #     stmt = stmt.where(sql.DesignData.id.in_(design_ids))  # Maybe can optimize joins
    # else:
    #     stmt = stmt

    return load_and_format(session, stmt, selected_column_names)


def poses(pose_jobs: Iterable[PoseJob]) -> list[PoseJob]:
    """Select PoseJob instances based on filters and weighting of all design summary metrics

    Args:
        pose_jobs: The PoseJob instances for which selection is desired
    Returns:
        The matching PoseJob instances
    """
    job = job_resources_factory.get()
    default_weight_metric = config.default_weight_parameter[job.design.method]

    if job.specification_file:  # Figure out poses from a specification file, filters, and weights
        loc_result = [(pose_job, design) for pose_job in pose_jobs
                      for design in pose_job.current_designs]
        total_df = load_total_dataframe(pose_jobs, pose=True)
        selected_poses_df = \
            metrics.prioritize_design_indices(total_df.loc[loc_result, :], filters=job.filter, weights=job.weight,
                                              protocols=job.protocol, default_weight=default_weight_metric,
                                              function=job.weight_function)
        # Remove excess pose instances
        number_chosen = 0
        selected_indices, selected_poses = [], set()
        for pose_job, design in selected_poses_df.index.tolist():
            if pose_job not in selected_poses:
                selected_poses.add(pose_job)
                selected_indices.append((pose_job, design))
                number_chosen += 1
                if number_chosen == job.select_number:
                    break

        # Specify the result order according to any filtering and weighting
        # Drop the specific design for the dataframe. If they want the design, they should run select_sequences
        save_poses_df = \
            selected_poses_df.loc[selected_indices, :].droplevel(-1)  # .droplevel(0, axis=1).droplevel(0, axis=1)
        # # convert selected_poses to PoseJob objects
        # selected_poses = [pose_job for pose_job in pose_jobs if pose_job_name in selected_poses]
    else:  # if job.total:  # Figure out poses from file/directory input, filters, and weights
        total_df = load_total_dataframe(pose_jobs, pose=True)
        if job.protocol:  # Todo adapt to protocol column not in Trajectories right now...
            group_df = total_df.groupby(putils.protocol)
            df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
                           keys=list(zip(group_df.groups, repeat('mean'))))
        else:
            df = pd.concat([total_df], axis=1, keys=['pose', 'metric'])
        # Figure out designs from dataframe, filters, and weights
        selected_poses_df = metrics.prioritize_design_indices(df, filters=job.filter, weights=job.weight,
                                                              protocols=job.protocol,
                                                              default_weight=default_weight_metric,
                                                              function=job.weight_function)
        # Remove excess pose instances
        number_chosen = 0
        selected_indices, selected_poses = [], set()
        for pose_job, design in selected_poses_df.index.tolist():
            if pose_job not in selected_poses:
                selected_poses.add(pose_job)
                selected_indices.append((pose_job, design))
                number_chosen += 1
                if number_chosen == job.select_number:
                    break

        # Specify the result order according to any filtering and weighting
        # Drop the specific design for the dataframe. If they want the design, they should run select_sequences
        save_poses_df = \
            selected_poses_df.loc[selected_indices, :].droplevel(-1)  # .droplevel(0, axis=1).droplevel(0, axis=1)
        # # convert selected_poses to PoseJob objects
        # selected_poses = [pose_job for pose_job in pose_jobs if pose_job_name in selected_poses]
    # elif job.dataframe:  # Figure out poses from a pose dataframe, filters, and weights
    #     if not pose_jobs:  # not job.directory:
    #         logger.critical(f'If using a --{flags.dataframe} for selection, you must include the directory where '
    #                         f'the designs are located in order to properly select designs. Please specify '
    #                         f'-d/--{flags.directory} with your command')
    #         sys.exit(1)
    #
    #     total_df = pd.read_csv(job.dataframe, index_col=0, header=[0, 1, 2])
    #     total_df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)
    #
    #     selected_poses_df = metrics.prioritize_design_indices(total_df, filter=job.filter, weight=job.weight,
    #                                                           protocol=job.protocol, function=job.weight_function)
    #     # Only drop excess columns as there is no MultiIndex, so no design in the index
    #     save_poses_df = selected_poses_df.droplevel(0, axis=1).droplevel(0, axis=1)
    #     program_root = job.program_root
    #     selected_poses = [PoseJob.from_directory(pose, root=job.projects)
    #                       for pose in save_poses_df.index.tolist()]
    # else:  # Generate design metrics on the spot
    #     raise NotImplementedError('This functionality is currently broken')
    #     selected_poses, selected_poses_df, total_df = [], pd.DataFrame(), pd.DataFrame()
    #     logger.debug('Collecting designs to sort')
    #     # if job.metric == 'score':
    #     #     metric_design_dir_pairs = [(pose_job.score, pose_job.path) for pose_job in pose_jobs]
    #     # elif job.metric == 'fragments_matched':
    #     #     metric_design_dir_pairs = [(pose_job.number_fragments_interface, pose_job.path)
    #     #                                for pose_job in pose_jobs]
    #     # else:  # This is impossible with the argparse options
    #     #     raise NotImplementedError(f'The metric "{job.metric}" is not supported!')
    #     #     metric_design_dir_pairs = []
    #
    #     logger.debug(f'Sorting designs according to "{job.metric}"')
    #     metric_design_dir_pairs = [(score, path) for score, path in metric_design_dir_pairs if score]
    #     sorted_metric_design_dir_pairs = sorted(metric_design_dir_pairs, key=lambda pair: (pair[0] or 0),
    #                                             reverse=True)
    #     top_designs_string = \
    #         f'Top ranked Designs according to {job.metric}:\n\t{job.metric.title()}\tDesign\n\t%s'
    #     results_strings = ['%.2f\t%s' % tup for tup in sorted_metric_design_dir_pairs]
    #     logger.info(top_designs_string % '\n\t'.join(results_strings[:500]))
    #     if len(pose_jobs) > 500:
    #         design_source = f'top_{job.metric}'
    #         # default_output_tuple = (utils.starttime, job.module, design_source)
    #         putils.make_path(job.all_scores)
    #         designs_file = \
    #             os.path.join(job.all_scores, f'{utils.starttime}_{job.module}_{design_source}_pose.scores')
    #         with open(designs_file, 'w') as f:
    #             f.write(top_designs_string % '\n\t'.join(results_strings))
    #         logger.info(f'Stdout performed a cutoff of ranked Designs at ranking 500. See the output design file '
    #                     f'"{designs_file}" for the remainder')
    #     sys.exit()  # terminate(output=False)
    # else:
    #     logger.critical('Missing a required method to provide or find metrics from %s. If you meant to gather '
    #                     'metrics from every pose in your input specification, ensure you include the --global '
    #                     'argument' % putils.program_output)
    #     sys.exit()

    # Format selected poses for output
    putils.make_path(job.output_directory)
    logger.info(f'Relevant files will be saved in the output directory: {job.output_directory}')

    if job.save_total:
        total_df = total_df[~total_df.index.duplicated()]
        total_df_filename = os.path.join(job.output_directory, 'TotalPoseMetrics.csv')
        total_df.to_csv(total_df_filename)
        logger.info(f'Total Pose DataFrame was written to: {total_df_filename}')

    logger.info(f'{len(save_poses_df)} Poses were selected')
    if len(save_poses_df) != len(total_df):
        if job.filter or job.weight:
            new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-{"Filtered" if job.filter else ""}'
                                                               f'{"Weighted" if job.weight else ""}PoseMetrics.csv')
        else:
            new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-PoseMetrics.csv')
        save_poses_df.to_csv(new_dataframe)
        logger.info(f'New DataFrame with selected poses was written to: {new_dataframe}')

    # # Select by clustering analysis
    # if job.cluster:
    # Sort results according to clustered poses if clustering exists
    if job.cluster.map:
        if os.path.exists(job.cluster.map):
            cluster_map = utils.unpickle(job.cluster.map)
        else:
            raise FileNotFoundError(f'No --{flags.cluster_map} "{job.cluster.map}" file was found')

        # Make the selected_poses into strings
        selected_pose_strs = list(map(str, selected_poses))
        # Check if the cluster map is stored as PoseDirectories or strings and convert
        representative_representative = next(iter(cluster_map))
        if not isinstance(representative_representative, PoseJob):
            # Make the cluster map based on strings
            for representative in list(cluster_map.keys()):
                # Remove old entry and convert all arguments to pose_id strings, saving as pose_id strings
                cluster_map[str(representative)] = [str(member) for member in cluster_map.pop(representative)]

        final_pose_indices = select_from_cluster_map(selected_pose_strs, cluster_map, number=job.cluster.number)
        final_poses = [selected_poses[idx] for idx in final_pose_indices]
        logger.info(f'Selected {len(final_poses)} poses after clustering')
    else:  # Try to generate the cluster_map?
        # raise utils.InputError(f'No --{flags.cluster_map} was provided. To cluster poses, specify:'
        logger.info(f'No --{flags.cluster_map} was provided. To {flags.cluster_poses}, specify:'
                    f'"{putils.program_command} {flags.cluster_poses}" or '
                    f'"{putils.program_command} {flags.protocol} '
                    f'--{flags.modules} {flags.cluster_poses} {flags.select_poses}"')
        logger.info('Grabbing all selected poses')
        final_poses = selected_poses
        # cluster_map: dict[str | PoseJob, list[str | PoseJob]] = {}
        # # {pose_string: [pose_string, ...]} where key is representative, values are matching designs
        # # OLD -> {composition: {pose_string: cluster_representative}, ...}
        # compositions: dict[tuple[str, ...], list[PoseJob]] = \
        #     protocols.cluster.group_compositions(selected_poses)
        # if job.multi_processing:
        #     mp_results = utils.mp_map(protocols.cluster.cluster_pose_by_transformations, compositions.values(),
        #                               processes=job.cores)
        #     for result in mp_results:
        #         cluster_map.update(result.items())
        # else:
        #     for composition_group in compositions.values():
        #         cluster_map.update(protocols.cluster.cluster_pose_by_transformations(composition_group))
        #
        # cluster_map_file = \
        #     os.path.join(job.clustered_poses, putils.default_clustered_pose_file.format(utils.starttime, location))
        # pose_cluster_file = utils.pickle_object(cluster_map, name=cluster_map_file, out_path='')
        # logger.info(f'Found {len(cluster_map)} unique clusters from {len(pose_jobs)} pose inputs. '
        #             f'All clusters stored in {pose_cluster_file}')
    # else:
    #     logger.info('Grabbing all selected poses')
    #     final_poses = selected_poses

    if len(final_poses) > job.select_number:
        final_poses = final_poses[:job.select_number]
        logger.info(f'Found {len(final_poses)} poses after applying your select-number criteria')

    return final_poses


def select_from_cluster_map(selected_pose_jobs: Sequence[Any], cluster_map: dict[Any, list[Any]], number: int = 1) \
        -> list[int]:
    """From a mapping of cluster representatives to their members, select members based on their ranking in the
    selected_members sequence

    Args:
        selected_pose_jobs: A sorted list of members that are members of the cluster_map
        cluster_map: A mapping of cluster representatives to their members
        number: The number of members to select
    Returns:
        The indices of selected_members, trimmed and retrieved according to cluster_map membership
    """
    # Make the selected_poses into strings
    selected_pose_identifiers = list(map(str, selected_pose_jobs))
    # Check if the cluster map is stored as PoseDirectories or strings and convert
    representative_representative = next(iter(cluster_map))
    if not isinstance(representative_representative, PoseJob):
        # Make the cluster map based on strings
        for representative in list(cluster_map.keys()):
            # Remove old entry and convert all arguments to pose_id strings, saving as pose_id strings
            cluster_map[str(representative)] = [str(member) for member in cluster_map.pop(representative)]

    membership_representative_map = cluster.invert_cluster_map(cluster_map)
    representative_found: dict[Any, list[Any]] = defaultdict(list)
    not_found = []
    for idx, member in enumerate(selected_pose_identifiers):
        try:
            cluster_representative = membership_representative_map[member]
        except KeyError:
            not_found.append(idx)
        else:
            representative_found[cluster_representative].append(idx)

    # Only include the highest ranked pose in the output as it provides info on all occurrences
    final_member_indices = []
    for member_indices in representative_found.values():
        final_member_indices.extend(member_indices[:number])

    if not_found:
        logger.warning(f"Couldn't locate the following members:\n\t%s\nAdding all of these to your selection..." %
                       '\n\t'.join(selected_pose_identifiers[idx] for idx in not_found))
        # 'Was {flags.cluster_poses} only run on a subset of the poses that were selected?
        final_member_indices.extend(not_found)

    return final_member_indices


def designs(pose_jobs: Iterable[PoseJob]) -> list[PoseJob]:
    """Select PoseJob instances based on filters and weighting of all design summary metrics

    Args:
        pose_jobs: The PoseJob instances for which selection is desired
    Returns:
        The matching PoseJob instances mapped to design name
    """
    job = job_resources_factory.get()
    default_weight_metric = config.default_weight_parameter[job.design.method]
    if job.specification_file:  # Figure out designs from a specification file, filters, and weights
        loc_result = [(pose_job, design) for pose_job in pose_jobs
                      for design in pose_job.current_designs]
        total_df = load_total_dataframe(pose_jobs)
        selected_poses_df = \
            metrics.prioritize_design_indices(total_df.loc[loc_result, :], filters=job.filter, weights=job.weight,
                                              protocols=job.protocol, default_weight=default_weight_metric,
                                              function=job.weight_function)
        # Specify the result order according to any filtering, weighting, and number
        selected_poses = {}
        for pose_job, design in selected_poses_df.index.tolist()[:job.select_number]:
            _designs = selected_poses.get(pose_job, None)
            if _designs:
                _designs.append(design)
            else:
                selected_poses[pose_job] = [design]

        # results = selected_poses
        save_poses_df = selected_poses_df.droplevel(0)  # .droplevel(0, axis=1).droplevel(0, axis=1)
        # Convert to PoseJob objects
        # results = {pose_job: results[str(pose_job)] for pose_job in pose_jobs
        #            if str(pose_job) in results}
    else:  # if job.total:
        total_df = load_total_dataframe(pose_jobs)
        if job.protocol:
            group_df = total_df.groupby(putils.protocol)
            df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
                           keys=list(zip(group_df.groups, repeat('mean'))))
        else:
            df = pd.concat([total_df], axis=1, keys=['pose', 'metric'])
        # Figure out designs from dataframe, filters, and weights
        selected_poses_df = metrics.prioritize_design_indices(df, filters=job.filter, weights=job.weight,
                                                              protocols=job.protocol,
                                                              default_weight=default_weight_metric,
                                                              function=job.weight_function)
        selected_designs = selected_poses_df.index.tolist()
        job.select_number = \
            len(selected_designs) if len(selected_designs) < job.select_number else job.select_number
        # if job.allow_multiple_poses:
        #     logger.info(f'Choosing {job.select_number} designs, from the top ranked designs regardless of pose')
        #     loc_result = selected_designs[:job.select_number]
        #     results = {pose_job: design for pose_job, design in loc_result}
        # else:  # elif job.designs_per_pose:
        designs_per_pose = job.designs_per_pose
        logger.info(f'Choosing up to {job.select_number} Designs, with {designs_per_pose} Design(s) per pose')
        number_chosen = count(1)
        selected_poses = {}
        for pose_job, design in selected_designs:
            _designs = selected_poses.get(pose_job, None)
            if _designs:
                if len(_designs) >= designs_per_pose:
                    # We already have too many, continue with search. No need to check as no addition
                    continue
                _designs.append(design)
            else:
                selected_poses[pose_job] = [design]

            if next(number_chosen) == job.select_number:
                break

        # results = selected_poses
        loc_result = [(pose_job, design) for pose_job, _designs in selected_poses.items() for design in _designs]

        # Include only the found index names to the saved dataframe
        save_poses_df = selected_poses_df.loc[loc_result, :]  # .droplevel(0).droplevel(0, axis=1).droplevel(0, axis=1)
        # Convert to PoseJob objects
        # results = {pose_job: results[str(pose_job)] for pose_job in pose_jobs
        #            if str(pose_job) in results}
    # else:  # Select designed sequences from each PoseJob.pose provided
    #     from . import select_sequences
    #     sequence_metrics = []  # Used to get the column headers
    #     sequence_filters = sequence_weights = None
    #
    #     if job.filter or job.weight:
    #         try:
    #             representative_pose_job = next(iter(pose_jobs))
    #         except StopIteration:
    #             raise RuntimeError('Missing the required argument pose_jobs. It must be passed to continue')
    #         example_trajectory = representative_pose_job.designs_metrics_csv
    #         trajectory_df = pd.read_csv(example_trajectory, index_col=0, header=[0])
    #         sequence_metrics = set(trajectory_df.columns.get_level_values(-1).tolist())
    #
    #     if job.filter == list():
    #         sequence_filters = metrics.query_user_for_metrics(sequence_metrics, mode='filter', level='sequence')
    #
    #     if job.weight == list():
    #         sequence_weights = metrics.query_user_for_metrics(sequence_metrics, mode='weight', level='sequence')
    #
    #     results: dict[PoseJob, list[str]]
    #     if job.multi_processing:
    #         # sequence_weights = {'buns_per_ang': 0.2, 'observed_evolution': 0.3, 'shape_complementarity': 0.25,
    #         #                     'int_energy_res_summary_delta': 0.25}
    #         zipped_args = zip(pose_jobs, repeat(sequence_filters), repeat(sequence_weights),
    #                           repeat(job.designs_per_pose), repeat(job.protocol))
    #         # result_mp = zip(*utils.mp_starmap(select_sequences, zipped_args, processes=job.cores))
    #         result_mp = utils.mp_starmap(select_sequences, zipped_args, processes=job.cores)
    #         results = {pose_job: _designs for pose_job, _designs in zip(pose_jobs, result_mp)}
    #     else:
    #         results = {pose_job: select_sequences(
    #                              pose_job, filters=sequence_filters, weights=sequence_weights,
    #                              number=job.designs_per_pose, protocols=job.protocol)
    #                    for pose_job in pose_jobs}
    #
    #     # Todo there is no sort here so the number isn't really doing anything
    #     results = dict(list(results.items())[:job.select_number])
    #     loc_result = [(pose_job, design) for pose_job, _designs in results.items() for design in _designs]
    #     total_df = load_total_dataframe(pose_jobs)
    #     save_poses_df = total_df.loc[loc_result, :].droplevel(0).droplevel(0, axis=1).droplevel(0, axis=1)

    logger.info(f'{len(selected_poses)} Poses were selected')
    logger.info(f'{len(save_poses_df)} Designs were selected')
    # Format selected sequences for output
    putils.make_path(job.output_directory)
    logger.info(f'Relevant files will be saved in the output directory: {job.output_directory}')

    if job.save_total:
        total_df = total_df[~total_df.index.duplicated()]
        total_df_filename = os.path.join(job.output_directory, 'TotalDesignMetrics.csv')
        total_df.to_csv(total_df_filename)
        logger.info(f'Total Pose/Designs DataFrame was written to: {total_df}')

    if job.filter or job.weight:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-{"Filtered" if job.filter else ""}'
                                                           f'{"Weighted" if job.weight else ""}DesignMetrics.csv')
    else:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-DesignMetrics.csv')
    save_poses_df.to_csv(new_dataframe)
    logger.info(f'New DataFrame with selected designs was written to: {new_dataframe}')

    # Create new output of designed PDB's  # Todo attach the state to these files somehow for further use
    exceptions = []
    for pose_job, _designs in selected_poses.items():
        pose_job.current_designs = _designs
        for design in _designs:
            file_path = os.path.join(pose_job.designs_path, f'*{design}*')
            file = sorted(glob(file_path))
            if not file:  # Add to exceptions
                exceptions.append((pose_job, f'No file found for "{file_path}"'))
                continue
            out_path = os.path.join(job.output_directory, f'{pose_job}_design_{design}.pdb')
            if not os.path.exists(out_path):
                shutil.copy(file[0], out_path)  # [i])))
                # shutil.copy(pose_job.designs_metrics_csv,
                #     os.path.join(outdir_traj, os.path.basename(pose_job.designs_metrics_csv)))
                # shutil.copy(pose_job.residues_metrics_csv,
                #     os.path.join(outdir_res, os.path.basename(pose_job.residues_metrics_csv)))
        # try:
        #     # Create symbolic links to the output PDB's
        #     os.symlink(file[0], os.path.join(job.output_directory,
        #                                      '%s_design_%s.pdb' % (str(pose_job), design)))  # [i])))
        #     os.symlink(pose_job.designs_metrics_csv,
        #                os.path.join(outdir_traj, os.path.basename(pose_job.designs_metrics_csv)))
        #     os.symlink(pose_job.residues_metrics_csv,
        #                os.path.join(outdir_res, os.path.basename(pose_job.residues_metrics_csv)))
        # except FileExistsError:
        #     pass

    return list(selected_poses.keys())


def sequences(pose_jobs: list[PoseJob]) -> list[PoseJob]:
    """Perform design selection followed by sequence formatting on those designs

    Args:
        pose_jobs: The PoseJob instances for which selection is desired
    Returns:
        The matching PoseJob instances
    """
    from symdesign.third_party.DnaChisel.dnachisel.DnaOptimizationProblem.NoSolutionError import NoSolutionError
    job = job_resources_factory.get()
    results = designs(pose_jobs)
    # Set up output_file pose_jobs for __main__.terminate()
    return_pose_jobs = list(results.keys())
    job.output_file = os.path.join(job.output_directory, 'SelectedDesigns.poses')

    if job.multicistronic:
        intergenic_sequence = job.multicistronic_intergenic_sequence
    else:
        intergenic_sequence = ''

    # Format sequences for expression
    tag_sequences, final_sequences, inserted_sequences, nucleotide_sequences = {}, {}, {}, {}
    codon_optimization_errors = {}
    for pose_job, _designs in results.items():
        pose_job.load_pose()
        number_of_entities = pose_job.number_of_entities
        number_of_tags, tag_index = solve_tags(number_of_entities, job.tag_entities)
        # Todo do I need to modify chains?
        pose_job.pose.rename_chains()
        for design in _designs:
            file_glob = f'{pose_job.designs_path}{os.sep}*{design}*'
            file = sorted(glob(file_glob))
            if not file:
                logger.error(f'No file found for {file_glob}')
                continue
            design_pose = Model.from_file(file[0], log=pose_job.log, entity_names=pose_job.entity_names)
            designed_atom_sequences = [entity.sequence for entity in design_pose.entities]

            # Container of booleans whether each Entity has been tagged
            missing_tags = [1 for _ in range(number_of_entities)]
            prior_offset = 0
            # all_missing_residues = {}
            # mutations = []
            sequences_and_tags = {}
            entity_termini_availability, entity_helical_termini = {}, {}
            for idx, (source_entity, design_entity) in enumerate(zip(pose_job.pose.entities, design_pose.entities)):
                # source_entity.retrieve_info_from_api()
                # source_entity.reference_sequence
                sequence_id = f'{pose_job}_{source_entity.name}'
                # design_string = '%s_design_%s_%s' % (pose_job, design, source_entity.name)  # [i])), pdb_code)
                design_string = f'{design}_{source_entity.name}'
                termini_availability = pose_job.pose.get_termini_accessibility(source_entity)
                logger.debug(f'Designed Entity {sequence_id} has the following termini accessible for tags: '
                             f'{termini_availability}')
                if job.avoid_tagging_helices:
                    termini_helix_availability = \
                        pose_job.pose.get_termini_accessibility(source_entity, report_if_helix=True)
                    logger.debug(f'Designed Entity {sequence_id} has the following helical termini available: '
                                 f'{termini_helix_availability}')
                    termini_availability = {'n': termini_availability['n'] and not termini_helix_availability['n'],
                                            'c': termini_availability['c'] and not termini_helix_availability['c']}
                    entity_helical_termini[design_string] = termini_helix_availability
                logger.debug(f'The termini {termini_availability} are available for tagging')
                entity_termini_availability[design_string] = termini_availability
                true_termini = [term for term, is_true in termini_availability.items() if is_true]

                # Find sequence specified attributes required for expression formatting
                # disorder = generate_mutations(source_entity.sequence, source_entity.reference_sequence,
                #                               only_gaps=True)
                # disorder = source_entity.disorder
                source_offset = source_entity.offset_index
                indexed_disordered_residues = {res_number + source_offset + prior_offset: mutation
                                               for res_number, mutation in source_entity.disorder.items()}
                # Todo, moved below indexed_disordered_residues on 7/26, ensure correct!
                prior_offset += len(indexed_disordered_residues)
                # Generate the source TO design mutations before any disorder handling
                mutations = generate_mutations(source_entity.sequence, design_entity.sequence, offset=False)
                # Insert the disordered residues into the design pose
                for residue_number, mutation in indexed_disordered_residues.items():
                    logger.debug(f'Inserting {mutation["from"]} into position {residue_number} on chain '
                                 f'{source_entity.chain_id}')
                    design_pose.insert_residue_type(mutation['from'], index=residue_number,
                                                    chain_id=source_entity.chain_id)
                    # adjust mutations to account for insertion
                    for mutation_index in sorted(mutations.keys(), reverse=True):
                        if mutation_index < residue_number:
                            break
                        else:  # mutation should be incremented by one
                            mutations[mutation_index + 1] = mutations.pop(mutation_index)

                # Check for expression tag addition to the designed sequences after disorder addition
                inserted_design_sequence = design_entity.sequence
                selected_tag = {}
                available_tags = expression.find_expression_tags(inserted_design_sequence)
                if available_tags:  # look for existing tag to remove from sequence and save identity
                    tag_names, tag_termini, _ = \
                        zip(*[(tag['name'], tag['termini'], tag['sequence']) for tag in available_tags])
                    try:
                        preferred_tag_index = tag_names.index(job.preferred_tag)
                        if tag_termini[preferred_tag_index] in true_termini:
                            selected_tag = available_tags[preferred_tag_index]
                    except ValueError:
                        pass
                    pretag_sequence = expression.remove_terminal_tags(inserted_design_sequence, tag_names)
                else:
                    pretag_sequence = inserted_design_sequence
                logger.debug(f'The pretag sequence is:\n{pretag_sequence}')

                # Find the open reading frame offset using the structure sequence after insertion
                offset = find_orf_offset(pretag_sequence, mutations)
                formatted_design_sequence = pretag_sequence[offset:]
                logger.debug(f'The open reading frame offset index is {offset}')
                logger.debug(f'The formatted_design sequence is:\n{formatted_design_sequence}')

                if number_of_tags == 0:  # Don't solve tags
                    sequences_and_tags[design_string] = {'sequence': formatted_design_sequence, 'tag': {}}
                    continue

                if not selected_tag:
                    # Find compatible tags from matching PDB observations
                    possible_matching_tags = []
                    for uniprot_id in source_entity.uniprot_ids:
                        uniprot_id_matching_tags = tag_sequences.get(uniprot_id, None)
                        if uniprot_id_matching_tags is None:
                            uniprot_id_matching_tags = \
                                expression.find_matching_expression_tags(uniprot_id=uniprot_id)
                            tag_sequences[uniprot_id] = uniprot_id_matching_tags
                        possible_matching_tags.extend(uniprot_id_matching_tags)

                    if possible_matching_tags:
                        tag_names, tag_termini, _ = \
                            zip(*[(tag['name'], tag['termini'], tag['sequence'])
                                  for tag in possible_matching_tags])
                    else:
                        tag_names, tag_termini, _ = [], [], []

                    iteration = 0
                    while iteration < len(tag_names):
                        try:
                            preferred_tag_index_2 = tag_names[iteration:].index(job.preferred_tag)
                            if tag_termini[preferred_tag_index_2] in true_termini:
                                selected_tag = uniprot_id_matching_tags[preferred_tag_index_2]
                                break
                        except ValueError:
                            selected_tag = \
                                expression.select_tags_for_sequence(sequence_id,
                                                                    uniprot_id_matching_tags,
                                                                    preferred=job.preferred_tag,
                                                                    **termini_availability)
                            break
                        iteration += 1

                if selected_tag.get('name'):
                    missing_tags[idx] = 0
                    logger.debug(f'The pre-existing, identified tag is:\n{selected_tag}')
                sequences_and_tags[design_string] = {'sequence': formatted_design_sequence, 'tag': selected_tag}

            # After selecting all tags, consider tagging the design as a whole
            if number_of_tags > 0:
                number_of_found_tags = number_of_entities - sum(missing_tags)
                if number_of_tags > number_of_found_tags:
                    print(f'There were {number_of_tags} requested tags for {pose_job} design {design.name} and '
                          f'{number_of_found_tags} were found')
                    current_tag_options = \
                        '\n\t'.join([f'{i} - {entity_name}\n'
                                     f'\tAvailable Termini: {entity_termini_availability[entity_name]}'
                                     f'\n\t\t   TAGS: {tag_options["tag"]}'
                                     for i, (entity_name, tag_options) in enumerate(sequences_and_tags.items(), 1)])
                    print(f'Current Tag Options:\n\t{current_tag_options}')
                    if job.avoid_tagging_helices:
                        print('Helical Termini:\n\t%s'
                              % '\n\t'.join(f'{entity_name}\t{availability}'
                                            for entity_name, availability in entity_helical_termini.items()))
                    satisfied = input('If this is acceptable, enter "continue", otherwise, '
                                      f'you can modify the tagging options with any other input.{input_string}')
                    if satisfied == 'continue':
                        number_of_found_tags = number_of_tags

                    iteration_idx = 0
                    while number_of_tags != number_of_found_tags:
                        if iteration_idx == len(missing_tags):
                            print(f'You have seen all options, but the number of requested tags ({number_of_tags}) '
                                  f"doesn't equal the number selected ({number_of_found_tags})")
                            satisfied = input('If you are satisfied with this, enter "continue", otherwise enter '
                                              'anything and you can view all remaining options starting from the '
                                              f'first entity{input_string}')
                            if satisfied == 'continue':
                                break
                            else:
                                iteration_idx = 0
                        for idx, entity_missing_tag in enumerate(missing_tags[iteration_idx:]):
                            sequence_id = f'{pose_job}_{pose_job.pose.entities[idx].name}'
                            if entity_missing_tag and tag_index[idx]:  # isn't tagged but could be
                                print(f'Entity {sequence_id} is missing a tag. Would you like to tag this entity?')
                                if not boolean_choice():
                                    continue
                            else:
                                continue
                            if job.preferred_tag:
                                tag = job.preferred_tag
                                while True:
                                    termini = input('Your preferred tag will be added to one of the termini. Which '
                                                    f'termini would you prefer? [n/c]{input_string}')
                                    if termini.lower() in ['n', 'c']:
                                        break
                                    else:
                                        print(f'"{termini}" is an invalid input. One of "n" or "c" is required')
                            else:
                                while True:
                                    tag_input = input('What tag would you like to use? Enter the number of the '
                                                      f'below options.\n\t%s\n{input_string}' %
                                                      '\n\t'.join(
                                                          [f'{i} - {tag}' for i, tag in enumerate(expression.tags, 1)]))
                                    if tag_input.isdigit():
                                        tag_input = int(tag_input)
                                        if tag_input <= len(expression.tags):
                                            tag = list(expression.tags.keys())[tag_input - 1]
                                            break
                                    print("Input doesn't match available options. Please try again")
                                while True:
                                    termini = input('Your tag will be added to one of the termini. Which termini '
                                                    f'would you prefer? [n/c]{input_string}')
                                    if termini.lower() in ['n', 'c']:
                                        break
                                    else:
                                        print(f'"{termini}" is an invalid input. One of "n" or "c" is required')

                            selected_entity = list(sequences_and_tags.keys())[idx]
                            if termini == 'n':
                                new_tag_sequence = \
                                    expression.tags[tag] + 'SG' \
                                    + sequences_and_tags[selected_entity]['sequence'][:12]
                            else:  # termini == 'c'
                                new_tag_sequence = \
                                    sequences_and_tags[selected_entity]['sequence'][-12:] \
                                    + 'GS' + expression.tags[tag]
                            sequences_and_tags[selected_entity]['tag'] = {'name': tag, 'sequence': new_tag_sequence}
                            missing_tags[idx] = 0
                            break

                        iteration_idx += 1
                        number_of_found_tags = number_of_entities - sum(missing_tags)

                elif number_of_tags < number_of_found_tags:  # when more than the requested number of tags were id'd
                    print(f'There were only {number_of_tags} requested tags for design {pose_job} and '
                          f'{number_of_found_tags} were found')
                    while number_of_tags != number_of_found_tags:
                        tag_input = input(f'Which tag would you like to remove? Enter the number of the currently '
                                          'configured tag option that you would like to remove. If you would like '
                                          f'to keep all, specify "keep"\n\t%s\n{input_string}'
                                          % '\n\t'.join([f'{i} - {entity_name}\n\t\t{tag_options["tag"]}'
                                                         for i, (entity_name, tag_options)
                                                         in enumerate(sequences_and_tags.items(), 1)]))
                        if tag_input == 'keep':
                            break
                        elif tag_input.isdigit():
                            tag_input = int(tag_input)
                            if tag_input <= len(sequences_and_tags):
                                missing_tags[tag_input - 1] = 1
                                selected_entity = list(sequences_and_tags.keys())[tag_input - 1]
                                sequences_and_tags[selected_entity]['tag'] = \
                                    {'name': None, 'termini': None, 'sequence': None}
                                # tag = list(expression.tags.keys())[tag_input - 1]
                                break
                            else:
                                print("Input doesn't match an integer from the available options. Please try again")
                        else:
                            print(f'"{tag_input}" is an invalid input. Try again')
                        number_of_found_tags = number_of_entities - sum(missing_tags)

            # Apply all tags to the sequences
            # Todo indicate the linkers that will be used!
            #  Request a new one if not ideal!
            cistronic_sequence = ''
            for idx, (design_string, sequence_tag) in enumerate(sequences_and_tags.items()):
                tag, sequence = sequence_tag['tag'], sequence_tag['sequence']
                # print('TAG:\n', tag.get('sequence'), '\nSEQUENCE:\n', sequence)
                design_sequence = expression.add_expression_tag(tag.get('sequence'), sequence)
                if tag.get('sequence') and design_sequence == sequence:  # tag exists and no tag added
                    tag_sequence = expression.tags[tag.get('name')]
                    if tag.get('termini') == 'n':
                        if design_sequence[0] == 'M':  # remove existing Met to append tag to n-term
                            design_sequence = design_sequence[1:]
                        design_sequence = tag_sequence + 'SG' + design_sequence
                    else:  # termini == 'c'
                        design_sequence = design_sequence + 'GS' + tag_sequence

                # If no MET start site, include one
                if design_sequence[0] != 'M':
                    design_sequence = f'M{design_sequence}'

                # If there is an unrecognized amino acid, modify
                if 'X' in design_sequence:
                    logger.critical(f'An unrecognized amino acid was specified in the sequence {design_string}. '
                                    'This requires manual intervention!')
                    # idx = 0
                    seq_length = len(design_sequence)
                    while True:
                        idx = design_sequence.find('X')
                        if idx == -1:  # Todo clean
                            break
                        idx_range = (idx - 6 if idx - 6 > 0 else 0,
                                     idx + 6 if idx + 6 < seq_length else seq_length)
                        while True:
                            new_amino_acid = input('What amino acid should be swapped for "X" in this sequence '
                                                   f'context?\n\t{idx_range[0] + 1}'
                                                   f'{" " * (len(range(*idx_range)) - (len(str(idx_range[0])) + 1))}'
                                                   f'{idx_range[1] + 1}'
                                                   f'\n\t{design_sequence[idx_range[0]:idx_range[1]]}'
                                                   f'{input_string}').upper()
                            if new_amino_acid in protein_letters_alph1:
                                design_sequence = design_sequence[:idx] + new_amino_acid + design_sequence[idx + 1:]
                                break
                            else:
                                print(f"{new_amino_acid} doesn't match a single letter canonical amino acid. "
                                      "Please try again")

                # For a final manual check of sequence generation, find sequence additions compared to the design
                # model and save to view where additions lie on sequence. Cross these additions with design
                # structure to check if insertions are compatible
                all_insertions = {residue: {'to': aa} for residue, aa in enumerate(design_sequence, 1)}
                all_insertions.update(generate_mutations(design_sequence, designed_atom_sequences[idx],
                                                         keep_gaps=True))
                # Reduce to sequence only
                inserted_sequences[design_string] = \
                    f'{"".join([res["to"] for res in all_insertions.values()])}\n{design_sequence}'
                logger.info(f'Formatted sequence comparison:\n{inserted_sequences[design_string]}')
                final_sequences[design_string] = design_sequence
                if job.nucleotide:
                    try:
                        nucleotide_sequence = \
                            optimize_protein_sequence(design_sequence, species=job.optimize_species)
                    except NoSolutionError:  # add the protein sequence?
                        logger.warning(f'Optimization of {design_string} was not successful!')
                        codon_optimization_errors[design_string] = design_sequence
                        break

                    if job.multicistronic:
                        if idx > 0:
                            cistronic_sequence += intergenic_sequence
                        cistronic_sequence += nucleotide_sequence
                    else:
                        nucleotide_sequences[design_string] = nucleotide_sequence
            if job.multicistronic:
                nucleotide_sequences[str(pose_job)] = cistronic_sequence

    # Report Errors
    if codon_optimization_errors:
        # Todo utilize errors
        error_file = \
            write_sequences(codon_optimization_errors, csv=job.csv,
                            file_name=os.path.join(job.output_directory,
                                                   f'OptimizationErrorProteinSequences'))
    # Write output sequences to fasta file
    seq_file = write_sequences(final_sequences, csv=job.csv,
                               file_name=os.path.join(job.output_directory, 'SelectedSequences'))
    logger.info(f'Final Design protein sequences written to: {seq_file}')
    seq_comparison_file = \
        write_sequences(inserted_sequences, csv=job.csv,
                        file_name=os.path.join(job.output_directory, 'SelectedSequencesExpressionAdditions'))
    logger.info(f'Final Expression sequence comparison to Design sequence written to: {seq_comparison_file}')
    # check for protein or nucleotide output
    if job.nucleotide:
        nucleotide_sequence_file = \
            write_sequences(nucleotide_sequences, csv=job.csv,
                            file_name=os.path.join(job.output_directory, 'SelectedSequencesNucleotide'))
        logger.info(f'Final Design nucleotide sequences written to: {nucleotide_sequence_file}')

    return return_pose_jobs


def format_save_df(session: Session, designs_df: pd.DataFrame, pose_ids: Iterable[int],
                   design_ids: Iterable[int] = None) -> pd.DataFrame:
    """Given a DataFrame with Pose/Design information, clean Pose and Entity information for readable output

    Args:
        session: A currently open transaction within sqlalchemy
        designs_df: A DataFrame with design metrics. Must contain a column corresponding to PoseJob.id named "pose_id"
        pose_ids: PoseJob instance identifiers for which metrics are desired
        design_ids: DesignData instance identifiers for which metrics are desired
    Returns:
        A DataFrame formatted with the PoseMetrics, EntityMetrics, and DesignEntityMetrics. The final DataFrame will
            have an entry for each PoseJob with separate metric columns grouped by 'structure_entity', i.e. Pose and
            Entity metrics
    """
    structure_entity = 'structure_entity'
    pose_id = 'pose_id'
    entity_id = 'entity_id'
    pose_metrics_df = load_sql_pose_metrics_dataframe(session, pose_ids=pose_ids)
    pose_metrics_df.set_index(pose_id, inplace=True)
    logger.debug(f'pose_metrics_df:\n{pose_metrics_df}')
    # save_df = pose_metrics_df.join(designs_df)  # , on='pose_id')

    pose_metadata_df = load_sql_pose_metadata_dataframe(session, pose_ids=pose_ids)
    pose_metrics_df = pose_metrics_df.join(pose_metadata_df.set_index(pose_id), rsuffix='_DROP')
    logger.debug(f'pose_metrics_df after metadata join:\n{pose_metrics_df}')

    # Join the designs_df (which may not have pose_id as index, but must have pose_id as a column)
    # with the pose_id indexed pose_metrics_df. This keeps the designs_df index in save_df
    save_df = designs_df.join(pose_metrics_df, on=pose_id, rsuffix='_DROP')
    save_df.drop(save_df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)
    save_df.columns = pd.MultiIndex.from_product([['pose'], save_df.columns.tolist()],
                                                 names=[structure_entity, 'metric'])
    logger.debug(f'save_df:\n{save_df}')
    # Get EntityMetrics
    entity_metrics_df = load_sql_entity_metrics_dataframe(session, pose_ids=pose_ids, design_ids=design_ids)
    logger.debug(f'entity_metrics_df:\n{entity_metrics_df}')
    # entity_metrics_df.set_index(pose_id, inplace=True)
    # Manipulate to combine with Pose data for the final format:
    # structure_entity        1        2 |    pose
    # metric            go fish  go fish | go fish
    # pose_id1           3    4   3    3 |  6  3.5
    # pose_id2           5    3   3    3 |  8    3
    # ...
    entity_metadata_df = load_sql_entity_metadata_dataframe(session, pose_ids=pose_ids)
    logger.debug(f'entity_metadata_df:\n{entity_metadata_df}')
    # entity_metadata_df.set_index(pose_id, inplace=True)
    # entity_metrics_df = entity_metrics_df.join(entity_metadata_df.set_index(pose_id), on=pose_id, rsuffix='_DROP')
    if entity_metrics_df.empty:
        # In the case there are no designs and therefore no design_entity_metrics entries
        entity_metrics_df = entity_metadata_df
    else:
        entity_metrics_df = entity_metrics_df.join(entity_metadata_df.set_index([pose_id, entity_id]),
                                                   on=[pose_id, entity_id], rsuffix='_DROP')
    entity_metrics_df.drop(entity_metrics_df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)
    logger.debug(f'entity_metrics_df after metadata.join:\n{entity_metrics_df}')
    # Get the first return from factorize since we just care about the unique "code" values
    entity_metrics_df[structure_entity] = \
        entity_metrics_df.groupby(pose_id).entity_id.transform(lambda x: pd.factorize(x)[0]) + 1
    # Todo add numeric_only=True? to groupby ops
    # entity_metrics_df[structure_entity] = entity_metrics_df.groupby(pose_id).entity_id.cumcount() + 1
    # entity_metrics_df[structure_entity] = \
    #     (entity_metrics_df.groupby('pose_id').entity_id.cumcount() + 1).apply(lambda x: f'entity_{x}')
    entity_metrics_df = entity_metrics_df.drop_duplicates([pose_id, structure_entity])
    logger.debug(f'entity_metrics_df AFTER factorize and deduplication:\n{entity_metrics_df}')
    # Make the stacked entity df and use the pose_id index to join with the above df
    pose_oriented_entity_df = entity_metrics_df.set_index([pose_id, structure_entity]).unstack().swaplevel(axis=1)
    # pose_oriented_entity_df.index = pd.MultiIndex.from_product([['pose'], pose_oriented_entity_df.index])
    # pose_oriented_entity_df = entity_metrics_df.unstack().swaplevel(axis=1)
    logger.debug(f'pose_oriented_entity_df:\n{pose_oriented_entity_df}')
    save_df = save_df.join(pose_oriented_entity_df, on=[('pose', pose_id)])  # , rsuffix='_DROP')  # , on=pose_id
    # save_df.drop(save_df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)
    logger.debug(f'Final save_df:\n{save_df}')

    return save_df


def load_pose_job_from_id(session: Session, ids: Sequence[int]) -> list[PoseJob]:
    """Given pose identifiers, either directory strings, or database ids, load PoseJobs

    Args:
        session:
        ids:
    Returns:
        The matching PoseJobs
    """
    # if job.module in flags.select_modules:
    #     pose_job_stmt = select(PoseJob).options(
    #         lazyload(PoseJob.entity_data),
    #         lazyload(PoseJob.metrics))
    # else:  # Load all attributes
    pose_job_stmt = select(PoseJob)
    try:  # To convert the identifier to an integer
        int(ids[0])
    except ValueError:  # Can't convert to integer, identifiers_are_database_id = False
        fetch_jobs_stmt = pose_job_stmt.where(PoseJob.pose_identifier.in_(ids))
    else:
        fetch_jobs_stmt = pose_job_stmt.where(PoseJob.id.in_(ids))

    return session.scalars(fetch_jobs_stmt).all()


def load_pose_identifier_from_id(session: Session, ids: Iterable[int]) -> dict[int, str]:
    pose_id_stmt = select((PoseJob.id, PoseJob.project, PoseJob.name)).where(PoseJob.id.in_(ids))
    return {id_: PoseJob.convert_pose_identifier(project, name)
            for id_, project, name in session.execute(pose_id_stmt)}


def load_design_identifier_from_id(session: Session, ids: Iterable[int]) -> list[tuple[int, int, str]]:
    design_id_stmt = select((sql.DesignData.pose_id, sql.DesignData.id, sql.DesignData.name)) \
        .where(sql.DesignData.id.in_(ids))
    return session.execute(design_id_stmt).all()
    # return dict(session.execute(design_id_stmt).all())


def sql_poses(pose_jobs: Iterable[PoseJob]) -> list[PoseJob]:
    """Select PoseJob instances based on filters and weighting of all design summary metrics

    Args:
        pose_jobs: The PoseJob instances for which selection is desired
    Returns:
        The selected PoseJob instances
    """
    job = job_resources_factory.get()
    default_weight_metric = config.default_weight_parameter[job.design.method]

    # Select poses from a starting pool and provided filters and weights
    pose_ids = [pose_job.id for pose_job in pose_jobs]
    # design_ids = [design.id for pose_job in pose_jobs for design in pose_job.current_designs]
    #     total_df = load_sql_poses_dataframe(session, pose_ids=pose_ids, design_ids=design_ids)
    #     selected_poses_df = \
    #         metrics.prioritize_design_indices(total_df, filters=job.filter, weights=job.weight,
    #                                           protocols=job.protocol, function=job.weight_function)
    #     # Remove excess pose instances
    #     number_chosen = 0
    #     selected_indices, selected_poses = [], set()
    #     for pose_job, design in selected_poses_df.index.tolist():
    #         if pose_job not in selected_poses:
    #             selected_poses.add(pose_job)
    #             selected_indices.append((pose_job, design))
    #             number_chosen += 1
    #             if number_chosen == job.select_number:
    #                 break
    #
    #     # Specify the result order according to any filtering and weighting
    #     # Drop the specific design for the dataframe. If they want the design, they should run select_sequences
    #     save_poses_df = \
    #         selected_poses_df.loc[selected_indices, :].droplevel(-1)  # .droplevel(0, axis=1).droplevel(0, axis=1)
    #     # # convert selected_poses to PoseJob objects
    #     # selected_poses = [pose_job for pose_job in pose_jobs if pose_job_name in selected_poses]
    # else:  # if job.total:  # Figure out poses from file/directory input, filters, and weights
    #     pose_ids = design_ids = None
    #     # total_df = load_sql_poses_dataframe(session)
    #
    #     # if job.protocol:  # Todo adapt to protocol column not in Trajectories right now...
    #     #     group_df = total_df.groupby(putils.protocol)
    #     #     df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
    #     #                    keys=list(zip(group_df.groups, repeat('mean'))))
    #     # else:
    #     #     df = pd.concat([total_df], axis=1, keys=['pose', 'metric'])

    pose_id = 'pose_id'
    entity_id = 'entity_id'
    design_id = 'design_id'
    with job.db.session(expire_on_commit=False) as session:
        # Figure out designs from dataframe, filters, and weights
        total_df = load_sql_poses_dataframe(session, pose_ids=pose_ids)  # , design_ids=design_ids)
        # # Todo
        # job_metadata_df = load_sql_pose_job_metadata_dataframe(session, pose_ids=pose_ids)
        pose_metadata_df = load_sql_pose_metadata_dataframe(session, pose_ids=pose_ids)
        entity_metadata_df = load_sql_entity_metadata_dataframe(session, pose_ids=pose_ids)
        logger.debug(f'entity_metadata_df:\n{entity_metadata_df}')
        total_df = total_df.join(pose_metadata_df.set_index(pose_id), on=pose_id, rsuffix='_DROP')
        total_df = \
            total_df.join(entity_metadata_df.set_index([pose_id, entity_id]), on=[pose_id, entity_id], rsuffix='_DROP')
        total_df.drop(total_df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)
        # logger.debug(f'total_df: {total_df.columns.tolist()}')
        if total_df.empty:
            raise utils.MetricsError(
                f"For the input PoseJobs, there aren't metrics collected. Use the '{flags.analysis}' module or perform "
                "some design module before selection")
        designs_df = load_sql_design_metrics_dataframe(session, pose_ids=pose_ids)  # , design_ids=design_ids)
        if designs_df.empty:
            pose_designs_mean_df = pd.DataFrame()
            # print(total_df)
            # raise NotImplementedError(f"Can't proceed without at least the PoseJob.pose_source")
        else:
            # designs_df has a multiplicity of number_of_entities from DesignEntityMetrics table join
            # Use the pose_id index to join to the total_df
            # Todo ensure non-numeric are here as well
            designs_df.drop(design_id, axis=1, inplace=True)
            pose_designs_mean_df = designs_df.groupby(pose_id).mean(numeric_only=True)
            total_df = total_df.join(pose_designs_mean_df, on=pose_id, rsuffix='_DROP')

            # # Todo JobMetadata
            # design_ids = total_df[design_id].unique().tolist()
            # design_metadata_df = load_sql_design_metadata_dataframe(session, design_ids=design_ids)
            # total_df = total_df.join(design_metadata_df.set_index(design_id), on=design_id, rsuffix='_DROP')

            entity_designs_df = load_sql_design_entities_dataframe(session, pose_ids=pose_ids)  # design_ids=design_ids)
            # logger.debug(f'entity_designs_df: {entity_designs_df}')
            pose_design_entities_mean_df = entity_designs_df.groupby([pose_id, entity_id]).mean(numeric_only=True)
            logger.debug(f'pose_design_entities_mean_df: {pose_design_entities_mean_df}')
            # # Drop unused designs columns
            # entity_columns = \
            #     [c.name for c in sql.DesignEntityMetrics.__table__.columns if c.name in designs_df.columns]
            # entity_designs_df = designs_df.loc[:, ['pose_id'] + entity_columns]
            # designs_df.drop([design_id] + entity_columns, axis=1, inplace=True)
            # This will create a total_df that is the number_of_entities X larger than the number of poses
            total_df = total_df.join(pose_design_entities_mean_df, on=[pose_id, entity_id], rsuffix='_DROP')
            total_df.drop(total_df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)
            logger.debug(f'total_df:\n{total_df}')

        if job.filter or job.protocol:
            entity_multiplicity = len(entity_metadata_df) / len(pose_metadata_df)
            # Todo still not accurate, got 13914 from 4241 designs
            logger.warning('Filtering statistics have an increased representation due to included Entity metrics. '
                           f'Values reported for each filter will be {entity_multiplicity}x over those actually '
                           f'present')
        # Ensure the pose_id is the index to prioritize
        total_df.set_index(pose_id, inplace=True)
        # Perform selection using provided arguments
        if not job.filter and not job.weight and not job.protocol and default_weight_metric not in total_df.columns:
            # Nothing to filter/weight
            selected_poses_df = total_df
        else:  # Filter/weight
            selected_poses_df = \
                metrics.prioritize_design_indices(total_df, filters=job.filter, weights=job.weight,
                                                  protocols=job.protocol, default_weight=default_weight_metric,
                                                  function=job.weight_function)
        # Remove excess pose instances
        selected_pose_ids = utils.remove_duplicates(selected_poses_df.index.tolist())[:job.select_number]

        # Select by clustering analysis
        if job.cluster_selection or job.cluster.map:
            pose_jobs = load_pose_job_from_id(session, selected_pose_ids)
            if job.cluster_selection:
                pose_jobs = cluster.cluster_poses(pose_jobs)

            # Sort results according to clustered poses
            # cluster_map: dict[str | PoseJob, list[str | PoseJob]] = {}
            if os.path.exists(job.cluster.map):
                cluster_map = utils.unpickle(job.cluster.map)
            else:
                raise FileNotFoundError(
                    f'No {flags.format_args(flags.cluster_map_args)} "{job.cluster.map}" file was found')

            final_pose_indices = select_from_cluster_map(pose_jobs, cluster_map, number=job.cluster.number)
            final_poses = [pose_jobs[idx] for idx in final_pose_indices]
            logger.info(f'Selected {len(final_poses)} poses after clustering')
            selected_pose_ids = [pose_job.id for pose_job in final_poses]

        if len(selected_pose_ids) > job.select_number:
            selected_pose_ids = selected_pose_ids[:job.select_number]
            logger.info(f'Found {len(selected_pose_ids)} Poses after applying your --select-number criteria')

        # Format selected PoseJob ids for output, including all additional metrics/metadata
        if not pose_designs_mean_df.empty:
            save_poses_df = pose_designs_mean_df.loc[selected_pose_ids].reset_index()
        else:
            save_poses_df = pd.DataFrame(zip(selected_pose_ids, range(len(selected_pose_ids))),
                                         columns=[pose_id, 'idx_DROP'])
            # save_poses_df.index = pd.Index(selected_pose_ids, name=pose_id)
            # save_poses_df = pd.Series(selected_pose_ids, name=pose_id).to_frame()
        save_poses_df = format_save_df(session, save_poses_df, selected_pose_ids)

        putils.make_path(job.output_directory)
        logger.info(f'Relevant files will be saved in the output directory: {job.output_directory}')
        if job.output_structures:
            logger.info(f'Copying Pose files...')
            # Create new output of designed PDB's
            final_pose_id_to_identifier = {}
            for pose_id_ in tqdm(selected_pose_ids):
                pose_job = session.get(PoseJob, pose_id_)
                final_pose_id_to_identifier[pose_id_] = pose_job.pose_identifier
                structure_path = pose_job.get_pose_file()
                if structure_path and os.path.exists(structure_path):
                    out_path = os.path.join(job.output_directory, f'{pose_job.project}-{pose_job.name}.pdb')
                    # Todo attach the program state to these files for downstream use?
                    shutil.copy(structure_path, out_path)
                else:
                    pose_job.log.error(f"Expected file '{structure_path}' wasn't found for {pose_job.pose_identifier}")
        else:
            final_pose_id_to_identifier = load_pose_identifier_from_id(session, selected_pose_ids)

        if job.save_total:
            out_total_df = total_df[~total_df[pose_id].duplicated()]
            total_pose_ids = out_total_df[pose_id].tolist()
            total_pose_id_to_identifier = load_pose_identifier_from_id(session, total_pose_ids)
            # Map the names to existing identifiers
            out_total_df['pose_identifier'] = out_total_df[pose_id].map(total_pose_id_to_identifier)
            out_total_df.set_index('pose_identifier', inplace=True)
            out_total_df.index.rename('pose_identifier', inplace=True)
            # Write
            total_df_filename = os.path.join(job.output_directory, 'TotalPoseMetrics.csv')
            out_total_df.to_csv(total_df_filename)
            logger.info(f'Total Pose DataFrame written to: {total_df_filename}')
            del out_total_df
    # End session

    # No need to rename as the index aren't design_id
    # save_poses_df.reset_index(col_fill='pose', col_level=-1, inplace=True)
    # Rename the identifiers to human-readable names
    save_poses_df.set_index(
        save_poses_df[('pose', pose_id)].map(final_pose_id_to_identifier).rename('pose_identifier'), inplace=True)

    # Format selected poses for output
    logger.info(f'{len(save_poses_df)} Poses were selected')
    if job.filter or job.weight:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-{"Filtered" if job.filter else ""}'
                                                           f'{"Weighted" if job.weight else ""}PoseMetrics.csv')
    else:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-PoseMetrics.csv')
    save_poses_df.to_csv(new_dataframe)
    logger.info(f'New DataFrame with selected poses written to: {new_dataframe}')

    return final_pose_id_to_identifier.values()
    # return final_poses


def sql_designs(pose_jobs: Iterable[PoseJob], return_pose_jobs: bool = False) -> list[PoseJob]:
    """Select PoseJob instances based on filters and weighting of all design summary metrics

    Args:
        pose_jobs: The PoseJob instances for which selection is desired
        return_pose_jobs: Whether to force the creation of PoseJob instances and load selected designs into
            PoseJob.current_designs
    Returns:
        The selected PoseJob instances with selected designs stored in the .current_designs attribute
    """
    job = job_resources_factory.get()
    default_weight_metric = config.default_weight_parameter[job.design.method]

    # Select designs from a starting pool and provided filters and weights
    pose_ids = [pose_job.id for pose_job in pose_jobs]
    design_ids = [design.id for pose_job in pose_jobs for design in pose_job.current_designs]
    #     total_df = load_sql_design_metrics_dataframe(session, pose_ids=pose_ids, design_ids=design_ids)
    #     selected_poses_df = \
    #         metrics.prioritize_design_indices(total_df, filters=job.filter, weights=job.weight,
    #                                           protocols=job.protocol, function=job.weight_function)
    #     # Specify the result order according to any filtering, weighting, and number
    #     results = {}
    #     for pose_id, design in selected_poses_df.index.tolist()[:job.select_number]:
    #         if pose_id in results:
    #             results[pose_id].append(design)
    #         else:
    #             results[pose_id] = [design]
    #
    #     save_designs_df = selected_poses_df.droplevel(0)  # .droplevel(0, axis=1).droplevel(0, axis=1)
    #     # Convert to PoseJob objects
    #     # results = {pose_id: results[str(pose_id)] for pose_id in pose_jobs
    #     #            if str(pose_id) in results}
    # else:  # if job.total:  # Figure out poses from file/directory input, filters, and weights
    #     pose_ids = design_ids = None
    #     # total_df = load_total_dataframe(pose_jobs)
    #     total_df = load_sql_design_metrics_dataframe(session)
    #     if job.protocol:
    #         group_df = total_df.groupby('protocol')
    #         df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
    #                        keys=list(zip(group_df.groups, repeat('mean'))))
    #     else:
    #         df = pd.concat([total_df], axis=1, keys=['pose', 'metric'])
    #     # Figure out designs from dataframe, filters, and weights
    #     selected_poses_df = metrics.prioritize_design_indices(df, filters=job.filter, weights=job.weight,
    #                                                           protocols=job.protocol, function=job.weight_function)
    #     selected_designs = selected_poses_df.index.tolist()
    #     job.select_number = \
    #         len(selected_designs) if len(selected_designs) < job.select_number else job.select_number
    #
    #     # Include only the found index names to the saved dataframe
    #     save_designs_df = selected_poses_df.loc[loc_result, :]  # droplevel(0).droplevel(0, axis=1).droplevel(0, axis=1)
    #     # Convert to PoseJob objects
    #     # results = {pose_id: results[str(pose_id)] for pose_id in pose_jobs
    #     #            if str(pose_id) in results}

    pose_id = 'pose_id'
    entity_id = 'entity_id'
    design_id = 'design_id'
    with job.db.session(expire_on_commit=False) as session:
        # Figure out designs from dataframe, filters, and weights
        total_df = load_sql_all_metrics_dataframe(session, pose_ids=pose_ids, design_ids=design_ids)
        design_ids = total_df[design_id].unique().tolist()
        design_metadata_df = load_sql_design_metadata_dataframe(session, design_ids=design_ids)
        pose_metadata_df = load_sql_pose_metadata_dataframe(session, pose_ids=pose_ids)
        entity_metadata_df = load_sql_entity_metadata_dataframe(session, pose_ids=pose_ids)
        logger.debug(f'entity_metadata_df:\n{entity_metadata_df}')
        total_df = total_df.join(design_metadata_df.set_index(design_id), on=design_id, rsuffix='_DROP')
        total_df = total_df.join(pose_metadata_df.set_index(pose_id), on=pose_id, rsuffix='_DROP')
        total_df = \
            total_df.join(entity_metadata_df.set_index([pose_id, entity_id]), on=[pose_id, entity_id], rsuffix='_DROP')
        total_df.drop(total_df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)
        logger.debug(f'total_df:\n{total_df}')
        # logger.debug(f'total_df: {total_df.columns.tolist()}')
        if total_df.empty:
            raise utils.MetricsError(
                f"For the input PoseJobs, there aren't metrics collected. Use the '{flags.analysis}' module or perform "
                f"some design module before {job.module}")
        if job.filter or job.protocol:
            entity_multiplicity = len(entity_metadata_df) / len(pose_metadata_df)
            logger.warning('Filtering statistics have an increased representation due to included Entity metrics. '
                           f'Values reported for each filter will be {entity_multiplicity}x over those actually '
                           f'present')
        # Ensure the design_id is the index to prioritize, though both pose_id and design_id are grabbed below
        total_df.set_index(design_id, inplace=True)
        # Perform selection using provided arguments
        if not job.filter and not job.weight and not job.protocol and default_weight_metric not in total_df.columns:
            # Nothing to filter/weight
            selected_designs_df = total_df
        else:  # Filter/weight
            selected_designs_df = \
                metrics.prioritize_design_indices(total_df, filters=job.filter, weights=job.weight,
                                                  protocols=job.protocol, default_weight=default_weight_metric,
                                                  function=job.weight_function)

        # Drop duplicated values keeping the order of the DataFrame
        selected_designs_df = selected_designs_df[~selected_designs_df.index.duplicated()]

        # Select by clustering analysis
        if job.cluster_selection or job.cluster.map:
            selected_pose_ids = selected_designs_df[pose_id].tolist()
            pose_jobs = load_pose_job_from_id(session, selected_pose_ids)
            if job.cluster_selection:
                pose_jobs = cluster.cluster_poses(pose_jobs)

            # Sort results according to clustered poses
            # cluster_map: dict[str | PoseJob, list[str | PoseJob]] = {}
            if os.path.exists(job.cluster.map):
                cluster_map = utils.unpickle(job.cluster.map)
            else:
                raise FileNotFoundError(
                    f'No {flags.format_args(flags.cluster_map_args)} "{job.cluster.map}" file was found')

            final_pose_indices = select_from_cluster_map(pose_jobs, cluster_map, number=job.cluster.number)
            final_poses = [pose_jobs[idx] for idx in final_pose_indices]
            logger.info(f'Selected {len(final_poses)} poses after clustering')
            selected_pose_ids = [pose_job.id for pose_job in final_poses]
            selected_designs_df = selected_designs_df[selected_designs_df[pose_id].isin(selected_pose_ids)]

        # Specify the result order according to any filtering, weighting, and number
        number_selected = len(selected_designs_df)
        job.select_number = number_selected if number_selected < job.select_number else job.select_number
        designs_per_pose = job.designs_per_pose
        logger.info(f'Choosing up to {job.select_number} Designs, with {designs_per_pose} Design(s) per Pose')

        # Get the pose_id and the design_id for each found design
        selected_design_ids = selected_designs_df.index.tolist()
        selected_pose_ids = selected_designs_df[pose_id].tolist()
        selected_designs = list(zip(selected_pose_ids, selected_design_ids))
        selected_designs_iter = iter(selected_designs)
        number_chosen = count()
        chosen = next(number_chosen)
        # selected_pose_id_to_design_ids = defaultdict(list)  # Alt way
        selected_pose_id_to_design_ids = {}
        try:
            while chosen < job.select_number:
                pose_id_, design_id_ = next(selected_designs_iter)
                # Alt way, but doesn't count designs_per_pose
                # selected_pose_id_to_design_ids[pose_id].append(design_id)
                _designs = selected_pose_id_to_design_ids.get(pose_id_, None)
                if _designs:
                    if len(_designs) < designs_per_pose:
                        _designs.append(design_id_)
                    else:  # Number of designs already satisfied for this pose
                        continue
                else:
                    selected_pose_id_to_design_ids[pose_id_] = [design_id_]
                chosen = next(number_chosen)
        except StopIteration:  # We exhausted selected_designs_iter
            pass

        logger.info(f'{len(selected_pose_id_to_design_ids)} Poses were selected')
        putils.make_path(job.output_directory)
        logger.info(f'Relevant files will be saved in the output directory: {job.output_directory}')
        if job.save_total:
            # Remove duplicate entries
            out_total_df = total_df[~total_df.index.duplicated()].copy()
            total_design_ids = out_total_df.index.tolist()
            total_pose_ids, total_design_ids, total_design_identifier = \
                zip(*load_design_identifier_from_id(session, total_design_ids))
            total_design_id_to_identifier = dict(zip(total_design_ids, total_design_identifier))
            total_pose_id_to_identifier = load_pose_identifier_from_id(session, set(total_pose_ids))
            # Map the names to existing identifiers
            out_total_df['pose_identifier'] = out_total_df[pose_id].map(total_pose_id_to_identifier)
            # Put the design_ids to a column
            out_total_df.reset_index(inplace=True)
            out_total_df['design_name'] = out_total_df[design_id].map(total_design_id_to_identifier)
            out_total_df.set_index(['pose_identifier', 'design_name'], inplace=True)
            out_total_df.index.rename(['pose_identifier', 'design_name'], inplace=True)
            # Write
            total_df_filename = os.path.join(job.output_directory, 'TotalDesignMetrics.csv')
            out_total_df.to_csv(total_df_filename)
            logger.info(f'Total Pose/Designs DataFrame written to: {total_df_filename}')
            del out_total_df

        # Format selected designs for output
        selected_design_ids = []
        for design_ids in selected_pose_id_to_design_ids.values():
            selected_design_ids.extend(design_ids)

        if job.output_structures:
            logger.info(f'Copying Design files...')
            # Create new output of designed PDB's
            pose_id_to_identifier = {}
            design_id_to_identifier = {}
            results = []
            for pose_id_, design_ids in tqdm(selected_pose_id_to_design_ids.items()):
                pose_job = session.get(PoseJob, pose_id_)
                pose_id_to_identifier[pose_id_] = pose_job.pose_identifier
                current_designs = []
                for design_id_ in design_ids:
                    design = session.get(sql.DesignData, design_id_)
                    design_name = design.name
                    design_id_to_identifier[design_id_] = design_name
                    design_structure_path = design.structure_path
                    if design_structure_path:
                        out_path = os.path.join(job.output_directory, f'{pose_job.project}-{design_name}.pdb')
                        if os.path.exists(design_structure_path):
                            # Todo attach the program state to these files for downstream use?
                            shutil.copy(design_structure_path, out_path)
                        else:
                            pose_job.log.error(f"Expected file '{design_structure_path}' wasn't found for "
                                               f"{design_structure_path}")
                        continue
                    else:
                        pose_job.log.error(f'No structure found for "{design}"')

                    current_designs.append(design)

                pose_job.current_designs = current_designs
                results.append(pose_job)
        else:
            pose_ids, design_ids, design_identifier = zip(*load_design_identifier_from_id(session, selected_design_ids))
            design_id_to_identifier = dict(zip(design_ids, design_identifier))
            unique_pose_ids = utils.remove_duplicates(pose_ids)
            pose_id_to_identifier = load_pose_identifier_from_id(session, unique_pose_ids)
            if return_pose_jobs:
                pose_job_stmt = select(PoseJob).where(PoseJob.id.in_(unique_pose_ids))
                results = session.scalars(pose_job_stmt).all()
                pose_id_to_design_ids = defaultdict(list)
                for idx, pose_id_ in enumerate(pose_ids):
                    pose_id_to_design_ids[pose_id_].append(design_ids[idx])

                for pose_job in results:
                    pose_job.current_designs = pose_id_to_design_ids[pose_job.id]
            else:
                results = pose_id_to_identifier.values()

        # Todo incorporate design_metadata_df
        design_metadata_df = load_sql_design_metadata_dataframe(session, design_ids=selected_design_ids)
        design_metrics_df = load_sql_design_metrics_dataframe(session, design_ids=selected_design_ids)
        # designs_df has a multiplicity of number_of_entities from DesignEntityMetrics table join
        design_metrics_df = \
            design_metadata_df.join(design_metrics_df.set_index(design_id), on=design_id, rsuffix='_DROP')
        # Format selected PoseJob with metrics for output
        # save_designs_df = selected_designs_df
        save_designs_df = format_save_df(session, design_metrics_df,
                                         selected_pose_id_to_design_ids.keys(),
                                         design_ids=selected_design_ids
                                         )
    # End session

    # No need to rename as the index aren't design_id
    # save_designs_df.reset_index(col_fill='pose', col_level=-1, inplace=True)
    # Rename the identifiers to human-readable names
    save_designs_df[('pose', 'design_name')] = save_designs_df[('pose', design_id)].map(design_id_to_identifier)
    # print('AFTER design_name', save_designs_df)
    save_designs_df[('pose', 'pose_identifier')] = save_designs_df[('pose', pose_id)].map(pose_id_to_identifier)
    # print('AFTER pose_identifier', save_designs_df)
    save_designs_df.set_index([('pose', 'pose_identifier'), ('pose', 'design_name')], inplace=True)
    save_designs_df.index.rename(['pose_identifier', 'design_name'], inplace=True)
    # print('AFTER set_index', save_designs_df)

    if job.filter or job.weight:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-{"Filtered" if job.filter else ""}'
                                                           f'{"Weighted" if job.weight else ""}DesignMetrics.csv')
    else:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-DesignMetrics.csv')
    save_designs_df.to_csv(new_dataframe)
    logger.info(f'New DataFrame with selected designs written to: {new_dataframe}')

    return results  # , exceptions


def solve_tags(n_of_tags: int, tag_entities: flags.tagging_literal = None) -> tuple[int, list[bool]]:
    """Set up mechanism to solve sequence tagging preferences

    Args:
        n_of_tags: The number of taggable entities
        tag_entities: A specification of how the tagable entities could be tagged
    Returns:
        The number of tags requested and the indices in which the tags could be applied
    """
    if tag_entities is None:
        boolean_tags = [False for _ in range(n_of_tags)]
        n_of_tags = 0
    elif tag_entities == 'all':
        boolean_tags = [True for _ in range(n_of_tags)]
    elif tag_entities == 'single':
        boolean_tags = [True for _ in range(n_of_tags)]
        n_of_tags = 1
    else:
        boolean_tags = []
        for tag_specification in map(str.strip, job.tag_entities.split(',')):
            # Remove non-numeric stuff
            if tag_specification == '':  # Probably a trailing ',' ...
                continue
            else:
                tag_specification.translate(utils.keep_digit_table)

            try:  # To convert to an integer
                boolean_tags.append(True if int(tag_specification) == 1 else False)
            except ValueError:  # Not an integer False
                boolean_tags.append(False)

        # Add any missing arguments to the tagging scheme
        for _ in range(n_of_tags - len(boolean_tags)):
            boolean_tags.append(False)
        n_of_tags = sum(boolean_tags)

    return n_of_tags, boolean_tags


def sql_sequences(pose_jobs: list[PoseJob]) -> list[PoseJob]:
    """Perform design selection followed by sequence formatting on those designs

    Args:
        pose_jobs: The PoseJob instances for which selection is desired
    Returns:
        The matching PoseJob instances
    """
    from symdesign.third_party.DnaChisel.dnachisel.DnaOptimizationProblem.NoSolutionError import NoSolutionError
    job = job_resources_factory.get()
    pose_jobs = sql_designs(pose_jobs, return_pose_jobs=True)
    # Ensure each design has relevant database features loaded
    with job.db.session(expire_on_commit=False) as session:
        session.add_all(pose_jobs)
        for pose_job in pose_jobs:
            # for entity_data in pose_job.entity_data:
            #     entity_data.metrics.number_of_residues
            session.add_all(pose_job.current_designs)
            for design in pose_job.current_designs:
                design.metrics.sequence
    # Set up output_file pose_jobs for __main__.terminate()
    job.output_file = os.path.join(job.output_directory, 'SelectedDesigns.poses')

    if job.multicistronic:
        intergenic_sequence = job.multicistronic_intergenic_sequence
    else:
        intergenic_sequence = ''
    if job.tag_entities:
        if job.tag_linker:
            tag_linker = job.tag_linker
        else:
            tag_linker = constants.default_tag_linker
        logger.info(f"Using the sequence '{tag_linker}' to link each protein sequence and the specified tag")

    # Format sequences for expression
    alignment_length = 40
    metrics_sequences = {}
    tag_sequences = {}
    final_sequences = {}
    inserted_sequences = {}
    nucleotide_sequences = {}
    codon_optimization_errors = {}
    for pose_job in pose_jobs:
        pose_job.load_pose()
        # Create the source_gap_mutations which provide mutation style dict for each gep
        # from the reference to the structure sequence
        entity_sequences = [entity.sequence for entity in pose_job.pose.entities]
        source_gap_mutations = [generate_mutations(entity.reference_sequence, entity.sequence,
                                                   zero_index=True, only_gaps=True)
                                for entity in pose_job.pose.entities]
        number_of_entities = pose_job.number_of_entities
        number_of_tags_requested, entity_taggable_indices = solve_tags(number_of_entities, job.tag_entities)

        # Find termini data
        logger.info('Searching for solvent accessible termini')
        entity_termini_availability = []
        entity_helical_termini = []
        entity_true_termini = []
        for entity in pose_job.pose.entities:
            pose_entity_id = f'{pose_job}_{entity.name}'
            termini_availability = pose_job.pose.get_termini_accessibility(entity)
            logger.debug(f'Designed Entity {pose_entity_id} has the accessible termini: {termini_availability}')
            if job.avoid_tagging_helices:
                termini_helix_availability = \
                    pose_job.pose.get_termini_accessibility(entity, report_if_helix=True)
                logger.debug(f'Designed Entity {pose_entity_id} has helical termini available: '
                             f'{termini_helix_availability}')
                termini_availability = {'n': termini_availability['n'] and not termini_helix_availability['n'],
                                        'c': termini_availability['c'] and not termini_helix_availability['c']}
                entity_helical_termini.append(termini_helix_availability)

            # Report and finalize for this Entity
            logger.debug(f'Designed Entity {pose_entity_id} has the termini available for tagging: '
                         f'{termini_availability}')
            entity_termini_availability.append(termini_availability)
            entity_true_termini.append([term for term, is_true in termini_availability.items() if is_true])

        metrics_sequences[pose_job] = []
        for design in pose_job.current_designs:
            design_sequence = design.metrics.sequence
            entity_number_residues_begin = entity_number_residues_end = 0
            designed_atom_sequences = []
            # Todo ensure can use without structure and Pose incase of sequence length change
            # for entity_data in pose_job.entity_data:
            for entity in pose_job.pose.entities:
                # entity_number_residues_end += entity_data.metrics.number_of_residues
                entity_number_residues_end += entity.number_of_residues
                designed_atom_sequences.append(design_sequence[entity_number_residues_begin:entity_number_residues_end])
                entity_number_residues_begin = entity_number_residues_end

            # Loop over each Entity
            entity_names = []
            entity_sequence_and_tags = []
            # Container of booleans, initialized where each Entity is missing a tag
            entity_missing_tags = [True for _ in range(number_of_entities)]
            for entity_idx, (data, source_sequence, design_sequence) in \
                    enumerate(zip(pose_job.entity_data, entity_sequences, designed_atom_sequences)):
                # Generate the design TO source mutations before any disorder handling
                # This will place design sequence identities in the 'from' position of mutations dictionary
                entity_name = data.name
                entity_names.append(entity_name)
                mutations = generate_mutations(''.join(design_sequence), source_sequence, zero_index=True)
                logger.debug(f'Found mutations: {mutations}')
                # Make sequence as list instead of string to use list.insert()
                inserted_design_sequence = list(design_sequence)
                # Insert the disordered residues into the design sequence
                for residue_index, mutation in source_gap_mutations[entity_idx].items():
                    # residue_index is zero indexed
                    new_aa_type = mutation['from']
                    logger.debug(f'Inserting {new_aa_type} into index {residue_index} on Entity {entity_name}')
                    # design_pose.insert_residue_type(new_aa_type, index=residue_index,
                    #                                 chain_id=entity.chain_id)
                    inserted_design_sequence.insert(residue_index, new_aa_type)
                    # Adjust mutations to account for insertion
                    for mutation_index in sorted(mutations.keys(), reverse=True):
                        if mutation_index < residue_index:
                            break
                        else:  # Mutation should be incremented by one
                            mutations[mutation_index + 1] = mutations.pop(mutation_index)

                # Check for expression tag addition to the designed sequences after disorder addition
                inserted_design_sequence = ''.join(inserted_design_sequence)
                logger.debug(f'The inserted design sequence is:\n{inserted_design_sequence}')
                selected_tag = {}
                available_tags = expression.find_expression_tags(inserted_design_sequence)
                if available_tags:
                    # Look for existing tags, save and possibly select a tag
                    tag_names, tag_termini, _ = \
                        zip(*[(tag['name'], tag['termini'], tag['sequence']) for tag in available_tags])
                    try:
                        preferred_tag_index = tag_names.index(job.preferred_tag)
                    except ValueError:
                        pass
                    else:
                        if tag_termini[preferred_tag_index] in entity_true_termini[entity_idx]:
                            selected_tag = available_tags[preferred_tag_index]
                    # Remove existing tags from sequence
                    pretag_sequence = expression.remove_terminal_tags(inserted_design_sequence, tag_names)
                    logger.debug(f'The sequence cleaned of tags is:\n{pretag_sequence}')
                else:
                    pretag_sequence = inserted_design_sequence
                    logger.debug(f'The pre-tagged sequence is the same as the inserted design sequence')

                # Find the open reading frame offset using the structure sequence after insertion
                offset = find_orf_offset(pretag_sequence, mutations)
                logger.debug(f'The open reading frame offset index is {offset}')
                if offset >= 0:
                    formatted_design_sequence = pretag_sequence[offset:]
                    logger.debug(f'The formatted_design sequence is:\n{formatted_design_sequence}')
                else:  # Subtract the offset from the mutations
                    # for mutation_index in sorted(mutations.keys(), reverse=True):
                    #     mutations[mutation_index + offset] = mutations.pop(mutation_index)
                    logger.debug('The offset is negative indicating non-reference sequence (such as tag linker '
                                 'residues), were added to the n-termini')
                    formatted_design_sequence = pretag_sequence

                # Figure out tagging specification
                if number_of_tags_requested == 0:  # Don't solve tags
                    selected_tag = {}
                # elif job.preferred_tag:
                # else:
                #     if not selected_tag:
                #         # Find compatible tags from matching PDB observations
                #         possible_matching_tags = []
                #         id_matching_tags = tag_sequences.get(entity_name)
                #         if id_matching_tags is None:
                #             tag_sequences[entity_name] = id_matching_tags = \
                #                 expression.find_matching_expression_tags(entity_id=entity_name,
                #                                                          alignment_length=alignment_length)
                #         possible_matching_tags.extend(id_matching_tags)
                #
                #         for uniprot_id in data.uniprot_ids:
                #             id_matching_tags = tag_sequences.get(uniprot_id)
                #             if id_matching_tags is None:
                #                 tag_sequences[uniprot_id] = id_matching_tags = \
                #                     expression.find_matching_expression_tags(uniprot_id=uniprot_id,
                #                                                              alignment_length=alignment_length)
                #             possible_matching_tags.extend(id_matching_tags)
                #
                #         if possible_matching_tags:
                #             tag_names, tag_termini, _ = \
                #                 zip(*[(tag['name'], tag['termini'], tag['sequence'])
                #                       for tag in possible_matching_tags])
                #         else:
                #             tag_names, tag_termini, _ = [], [], []
                #
                #         while True:
                #             # Using the while loop to enable break and avoid expression.select_tags_for_sequence()
                #             try:
                #                 preferred_tag_index_2 = tag_names.index(job.preferred_tag)
                #             except ValueError:  # job.preferred_tag not indexed
                #                 pass
                #             else:
                #                 if tag_termini[preferred_tag_index_2] in entity_true_termini[entity_idx]:
                #                     selected_tag = possible_matching_tags[preferred_tag_index_2]
                #                     break
                #             design_entity_id = f'{design.name}-{entity_name}'
                #             selected_tag = \
                #                 expression.select_tags_for_sequence(design_entity_id,
                #                                                     possible_matching_tags,
                #                                                     preferred=job.preferred_tag,
                #                                                     **entity_termini_availability[entity_idx])
                #             break
                #
                #     if selected_tag.get('name'):
                #         entity_missing_tags[entity_idx] = False
                #         logger.debug(f'The pre-existing, identified tag is:\n{selected_tag}')
                entity_sequence_and_tags.append({'sequence': formatted_design_sequence, 'tag': selected_tag})

            # After selecting individual Entity tags, consider tagging the whole Design
            if number_of_tags_requested > 0:
                number_of_found_tags = number_of_entities - sum(entity_missing_tags)
                # When fewer than the requested number of tags were identified
                if number_of_tags_requested > number_of_found_tags:
                    print(f'There were {number_of_tags_requested} requested tags for {pose_job} Design {design.name} '
                          f'and {number_of_found_tags} were found')
                    header = 'Index', 'Name', 'Selected tag', 'Available termini'
                    if job.avoid_tagging_helices:
                        header += ('Helical termini',)
                        helical_info = [(','.join(term for term, available in term_availablity.items() if available),)
                                        for term_availablity in entity_helical_termini]
                    else:
                        helical_info = tuple()
                    current_tag_options = \
                        '\n\t'.join(utils.pretty_format_table(
                            [(idx + 1, entity_names[idx],
                              seq_tag_options['tag'] if seq_tag_options['tag'].get('name') else None,
                              ','.join(term for term, available in entity_termini_availability[idx].items()
                                       if available)) + helical_info[idx]
                             for idx, seq_tag_options in enumerate(entity_sequence_and_tags)],
                            header=header))
                    print(f'Existing Entity tagging options:\n\t{current_tag_options}')
                    satisfied = utils.validate_input(
                        "If this tagging scheme is acceptable, enter 'p' (proceed), otherwise enter 'c' (configure) "
                        'to configure tags', ['p', 'c'])
                    if satisfied == 'p':
                        number_of_found_tags = number_of_tags_requested

                    iteration = count()
                    while number_of_tags_requested != number_of_found_tags:
                        iteration_idx = next(iteration)
                        if iteration_idx == number_of_entities:
                            print("You've seen all options, but the number of tags requested, "
                                  f'{number_of_tags_requested} != {number_of_found_tags}, the number of tags found')
                            satisfied = utils.validate_input(
                                "If you are satisfied with this scheme, enter 'p' (proceed), otherwise enter 'c' "
                                '(configure), and you can view all options again starting with the first entity',
                                ['p', 'c'])
                            if satisfied == 'p':
                                break
                            else:  # Start over
                                iteration = count()
                                continue
                        for entity_idx, entity_missing_tag in enumerate(entity_missing_tags[iteration_idx:]):
                            entity_name = entity_names[entity_idx]
                            if entity_missing_tag and entity_taggable_indices[entity_idx]:  # Isn't tagged but could be
                                print(f'Entity {pose_job}_{entity_name} is missing a tag. '
                                      f'Would you like to tag this entity?')
                                if not boolean_choice():
                                    continue
                            else:
                                continue
                            # Solve by preferred_tag or user input
                            if job.preferred_tag:
                                tag = job.preferred_tag
                            else:
                                print('Tag options include:\n\t%s' %
                                      '\n\t'.join([f'{idx} - {tag}' for idx, tag in enumerate(expression.tags, 1)]))
                                tag_input = validate_input('Which of the above tags would you like to use? Enter the '
                                                           'number of your preferred option',
                                                           list(map(str, range(1, 1 + len(expression.tags)))))
                                # Adjust for python indexing
                                tag_index = int(tag_input) - 1
                                tag = list(expression.tags.keys())[tag_index]
                            # termini = validate_input(f"Which termini should the selected tag '{tag}', be added to?",
                            #                          ['n', 'c'])
                            # Find compatible tags from matching PDB observations
                            possible_matching_tags = []
                            id_matching_tags = tag_sequences.get(entity_name)
                            if id_matching_tags is None:
                                tag_sequences[entity_name] = id_matching_tags = \
                                    expression.find_matching_expression_tags(entity_id=entity_name,
                                                                             alignment_length=alignment_length)
                            possible_matching_tags.extend(id_matching_tags)

                            for uniprot_id in pose_job.entity_data[entity_idx].uniprot_ids:
                                id_matching_tags = tag_sequences.get(uniprot_id)
                                if id_matching_tags is None:
                                    tag_sequences[uniprot_id] = id_matching_tags = \
                                        expression.find_matching_expression_tags(uniprot_id=uniprot_id,
                                                                                 alignment_length=alignment_length)
                                possible_matching_tags.extend(id_matching_tags)
                            termini = expression.report_termini_availability(possible_matching_tags,
                                                                             **entity_termini_availability[entity_idx])
                            if termini == 'skip':
                                continue

                            selected_sequence_and_tag = entity_sequence_and_tags[entity_idx]
                            if termini == 'n':
                                new_tag_sequence = expression.tags[tag] \
                                    + tag_linker + selected_sequence_and_tag['sequence'][:alignment_length]
                            else:  # termini == 'c'
                                new_tag_sequence = selected_sequence_and_tag['sequence'][-alignment_length:] \
                                    + tag_linker + expression.tags[tag]
                            selected_sequence_and_tag['tag'] = \
                                {'name': tag, 'termini': termini, 'sequence': new_tag_sequence}
                            entity_missing_tags[entity_idx] = False
                            break

                        number_of_found_tags = number_of_entities - sum(entity_missing_tags)
                # When more than the requested number of tags were identified
                elif number_of_tags_requested < number_of_found_tags:
                    print(f'There were only {number_of_tags_requested} requested tags for design {pose_job}, however, '
                          f'{number_of_found_tags} were found')
                    print('Configured tags:')
                    print('\t%s' % '\n\t'.join([f'{idx + 1} - {entity_names[idx]}\n\t\t{tag_options["tag"]}'
                                                for idx, tag_options in enumerate(entity_sequence_and_tags)]))
                    while number_of_tags_requested != number_of_found_tags:
                        tag_input = utils.validate_input(
                            'Which tag would you like to remove? Enter a number from the above tag options or, if you '
                            "would like to keep all, specify 'keep'",
                            list(map(str, range(1, 1 + number_of_found_tags))) + ['keep'])
                        if tag_input == 'keep':
                            break
                        else:  # if tag_input.isdigit():
                            tag_index = int(tag_input) - 1
                            # if tag_input <= len(entity_sequence_and_tags):
                            # Set that this entity is now missing a tag
                            entity_missing_tags[tag_index] = True
                            entity_sequence_and_tags[tag_index]['tag'] = \
                                {'name': None, 'termini': None, 'sequence': None}

                        number_of_found_tags = number_of_entities - sum(entity_missing_tags)

            # Apply all tags to the sequences
            cistronic_sequence = ''
            sequences_for_metrics = []
            for idx, (entity_name, sequence_tag) in enumerate(zip(entity_names, entity_sequence_and_tags)):
                design_string = f'{design.name}_{entity_name}'
                sequence = sequence_tag['sequence']
                tag = sequence_tag['tag']
                chimeric_tag_sequence = tag.get('sequence')

                # tagged_sequence = expression.add_expression_tag(chimeric_tag_sequence, sequence)
                if chimeric_tag_sequence:  # A tag exists
                    # if tagged_sequence == sequence:  # No tag added
                    #     tag_sequence = expression.tags[tag['name']]
                    #     if tag.get('termini') == 'n':
                    #         if tagged_sequence[0] == 'M':  # Remove existing n-term Met to append tag to n-term
                    #             tagged_sequence = tagged_sequence[1:]
                    #         tagged_sequence = tag_sequence + tag_linker + tagged_sequence
                    #     else:  # termini == 'c'
                    #         tagged_sequence = tagged_sequence + tag_linker + tag_sequence
                    # else:
                    logger.debug(f'Cleaning chimeric tag sequence: {chimeric_tag_sequence}')
                    tag_termini = tag['termini']
                    chimeric_tag_sequence = expression.remove_terminal_tags(chimeric_tag_sequence, termini=tag_termini)
                    tag_sequence = expression.tags[tag['name']]
                    if tag_termini == 'n':
                        chimeric_tag_sequence = tag_sequence + chimeric_tag_sequence
                    else:
                        chimeric_tag_sequence += tag_sequence

                    logger.debug(f'Applying cleaned chimeric tag sequence: {chimeric_tag_sequence}')
                    if tag_linker:  # and tag_linker not in chimeric_tag_sequence:
                        # Add the linker between the tag and designed sequence
                        tag_insert_index = chimeric_tag_sequence.find(tag_sequence)
                        if tag_termini == 'n':
                            # Insert the index from the c-term side
                            tag_insert_index += len(tag_sequence)
                        chimeric_tag_sequence = chimeric_tag_sequence[:tag_insert_index] + tag_linker \
                            + chimeric_tag_sequence[tag_insert_index:]
                        logger.debug(f'Formatted the chimeric tag sequence with the specified linker:'
                                     f' {chimeric_tag_sequence}')

                    tagged_sequence = expression.add_expression_tag(chimeric_tag_sequence, sequence)
                else:
                    tag_name = tag.get('name')
                    if tag_name:
                        tag_sequence = expression.tags[tag_name]
                        if tag['termini'] == 'n':
                            if sequence[0] == 'M':  # Remove existing n-term Met to append tag to n-term
                                sequence = sequence[1:]
                            tagged_sequence = tag_sequence + tag_linker + sequence
                        else:  # termini == 'c'
                            tagged_sequence = sequence + tag_linker + tag_sequence
                    else:
                        tagged_sequence = sequence

                # If no MET start site, include one
                if tagged_sequence[0] != 'M':
                    tagged_sequence = f'M{tagged_sequence}'

                # If there is an unrecognized amino acid, modify
                unknown_char = 'X'
                if unknown_char in tagged_sequence:
                    logger.critical(f'An unrecognized amino acid was specified in the sequence {design_string}. '
                                    'This requires manual intervention!')
                    # idx = 0
                    seq_length = len(tagged_sequence)
                    while True:
                        missing_idx = tagged_sequence.find(unknown_char)
                        if missing_idx == -1:
                            break
                        low_idx = missing_idx - 6 if missing_idx - 6 > 0 else 0
                        high_idx = missing_idx + 6 if missing_idx + 6 < seq_length else seq_length
                        print(f'Which amino acid should be swapped for "{unknown_char}" in this sequence context?\n'
                              f'\t{low_idx + 1}{" " * (missing_idx-low_idx-len(str(low_idx)))}|'
                              f'{" " * (high_idx-missing_idx-2)}{high_idx + 1}'  # Subtract 2 for slicing and high_idx
                              # f'{" " * (high_idx-low_idx - (len(str(low_idx))+1))}{high_idx + 1}'
                              f'\n\t{tagged_sequence[low_idx:high_idx]}')
                        new_amino_acid = validate_input(input_string, protein_letters_alph1)
                        tagged_sequence = tagged_sequence[:missing_idx] \
                            + new_amino_acid + tagged_sequence[missing_idx + 1:]

                # For a final manual check of sequence generation, find sequence additions compared to the design
                # model and save to view where additions lie on sequence. Cross these additions with design
                # structure to check if insertions are compatible
                # all_insertions = {residue: {'to': aa} for residue, aa in enumerate(tagged_sequence)}
                # all_insertions.update(generate_mutations(design_sequence, ''.join(designed_atom_sequences[idx]),
                #                                          keep_gaps=True))
                # generated_insertion_mutations = \
                #     generate_mutations(tagged_sequence, ''.join(designed_atom_sequences[idx]),
                #                        keep_gaps=True, zero_index=True)
                # logger.debug(f'generated_insertion_mutations: {generated_insertion_mutations}')
                # all_insertions.update(generated_insertion_mutations)
                # formatted_comparison = {}
                # for mutation_index in sorted(all_insertions.keys()):
                generated_insertion_mutations = \
                    generate_mutations(tagged_sequence, designed_atom_sequences[idx],
                                       return_all=True, keep_gaps=True, zero_index=True)
                # for mutations in generated_insertion_mutations.values():
                #     reference = mutations['from']
                #     query = mutations['to']

                # Reduce to sequence only
                inserted_sequences[design_string] = \
                    f'Expressed: {"".join([res["from"] for res in generated_insertion_mutations.values()])}\n' \
                    f'Designed : {"".join([res["to"] for res in generated_insertion_mutations.values()])}'
                # # Reduce to sequence only
                # inserted_sequences[design_string] = \
                #     f'{"".join([res["to"] for res in all_insertions.values()])}\n{tagged_sequence}'
                logger.info(f'Formatted sequence comparison:\n{inserted_sequences[design_string]}')
                final_sequences[design_string] = tagged_sequence
                sequences_for_metrics.append(tagged_sequence)
                if job.nucleotide:
                    try:
                        nucleotide_sequence = \
                            optimize_protein_sequence(tagged_sequence, species=job.optimize_species)
                    except NoSolutionError:  # Add the protein sequence?
                        logger.warning(f'Optimization of {design_string} was not successful!')
                        codon_optimization_errors[design_string] = tagged_sequence
                        break

                    if job.multicistronic:
                        if idx > 0:
                            cistronic_sequence += intergenic_sequence
                        cistronic_sequence += nucleotide_sequence
                    else:
                        nucleotide_sequences[design_string] = nucleotide_sequence
            # Finish processing for the design
            metrics_sequences[pose_job].append(sequences_for_metrics)
            if job.multicistronic:
                nucleotide_sequences[str(pose_job)] = cistronic_sequence
        # Clear memory of the PoseJob
        pose_job.clear_state()

    # Format expression sequence metrics
    sequence_metrics = {}
    for pose_job, designs_sequences in metrics_sequences.items():
        for design, design_sequences in zip(pose_job.current_designs, designs_sequences):
            # Iterate over each Entity
            pose_sequence = ''
            for entity_idx, sequence in enumerate(design_sequences, 1):
                entity_sequence_features = expression.get_sequence_features(sequence)
                sequence_metrics[(pose_job.pose_identifier, design.name, entity_idx)] = entity_sequence_features
                pose_sequence += sequence
            pose_sequence_features = expression.get_sequence_features(pose_sequence)
            sequence_metrics[(pose_job.pose_identifier, design.name, 'pose')] = pose_sequence_features
    # Format DataFrame and save metrics
    sequence_metrics_df = pd.DataFrame(sequence_metrics.values(),
                                       index=pd.MultiIndex.from_tuples(sequence_metrics.keys()))
    sequence_metrics_df = sequence_metrics_df.unstack(-1).swaplevel(axis=1)
    sequence_metrics_filename = os.path.join(job.output_directory, 'SequenceExpressionMetrics.csv')
    sequence_metrics_df.to_csv(sequence_metrics_filename)
    logger.info(f'Biochemical protein sequence metrics written to: {sequence_metrics_filename}')

    # Report Errors
    if codon_optimization_errors:
        # Todo utilize errors
        error_file = \
            write_sequences(codon_optimization_errors, csv=job.csv,
                            file_name=os.path.join(job.output_directory, 'OptimizationErrorProteinSequences'))
    # Write output sequences to fasta file
    seq_file = write_sequences(final_sequences, csv=job.csv,
                               file_name=os.path.join(job.output_directory, 'SelectedSequences'))
    logger.info(f'Protein designed sequences written to: {seq_file}')
    seq_comparison_file = \
        write_sequences(inserted_sequences, csv=job.csv,
                        file_name=os.path.join(job.output_directory, 'SelectedSequencesExpressionAdditions'))
    logger.info(f'Protein expression sequence comparison to designed sequences written to: {seq_comparison_file}')
    # check for protein or nucleotide output
    if job.nucleotide:
        nucleotide_sequence_file = \
            write_sequences(nucleotide_sequences, csv=job.csv,
                            file_name=os.path.join(job.output_directory, 'SelectedSequencesNucleotide'))
        logger.info(f'Nucleotide designed sequences written to: {nucleotide_sequence_file}')

    return pose_jobs
