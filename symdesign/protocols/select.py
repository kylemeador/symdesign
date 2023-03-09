from __future__ import annotations

import logging
import os
import shutil
from glob import glob
from itertools import repeat, count
from typing import Any, Iterable, Sequence

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from . import cluster
from .pose import PoseJob
import symdesign.utils.path as putils
from symdesign import flags, metrics, utils
from symdesign.resources import sql, config
from symdesign.resources.job import job_resources_factory
from symdesign.resources.query.utils import input_string, boolean_choice, validate_input
from symdesign.structure.model import Model
from symdesign.sequence import optimize_protein_sequence, write_sequences, expression, find_orf_offset, \
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
            df.drop([index for index in df.index.to_list() if isinstance(index, float)], inplace=True)
            # Get rid of all individual trajectories and std, not mean
            pose_name = pose_job.name
            df.drop([index for index in df.index.to_list() if pose_name in index or 'std' in index], inplace=True)
            # except TypeError:
            #     for index in df.index.to_list():
            #         print(index, type(index))
    else:  # designs
        for pose_job, df in zip(pose_jobs, all_dfs):
            # Get rid of all statistic entries, mean, std, etc.
            pose_name = pose_job.name
            df.drop([index for index in df.index.to_list() if pose_name not in index], inplace=True)

    # Add pose directory str as MultiIndex
    try:
        df = pd.concat(all_dfs, keys=[str(pose_job) for pose_job in pose_jobs])
    except ValueError:  # No objects to concatenate
        raise RuntimeError(f"Didn't find any trajectory information in the provided PoseDirectory instances")
    df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)

    return df


def load_and_format(session: Session, stmt: Select, selected_column_names: Iterable[str]) -> pd.DataFrame:
    # Todo modify the upstream stmt to avoid this... if a problem
    # SAWarning: SELECT statement has a cartesian product between FROM element(s) "pose_metrics", "pose_data",
    # "entity_data", "design_data", "design_metrics", "entity_metrics" and FROM element "protein_metadata".
    # Apply join condition(s) between each element to resolve.
    # SAWarning: SELECT statement has a cartesian product between FROM element(s) "entity_data", "pose_data",
    # "entity_metrics" and FROM element "protein_metadata".
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


def load_sql_metrics_dataframe(session: Session, pose_ids: Iterable[int] = None, design_ids: Iterable[int] = None) \
        -> pd.DataFrame:
    """Load and format every PoseJob instance's, PoseMetrics, EntityMetrics, and DesignMetrics for each associated
    design

    Optionally limit those loaded to certain PoseJob.id's and DesignData.id's

    Args:
        session: A session object to complete the transaction
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
    entity_metadata_c = [sql.ProteinMetadata.n_terminal_helix,
                         sql.ProteinMetadata.c_terminal_helix,
                         sql.ProteinMetadata.thermophilicity]
    em_c = [c for c in (*sql.EntityMetrics.__table__.columns, *entity_metadata_c,
                        *sql.DesignEntityMetrics.__table__.columns)
            if not c.primary_key]
    # Remove design_id
    em_c.pop(em_c.index(sql.DesignEntityMetrics.design_id))
    em_names = [f'entity_{c.name}' if c.name != 'entity_id' else c.name for c in em_c]
    selected_columns = (*pm_c, *dm_c, *em_c)
    selected_column_names = (*pm_names, *dm_names, *em_names)
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(PoseJob).join(sql.PoseMetrics).join(sql.EntityData).join(
        sql.EntityMetrics).join(sql.DesignData).join(sql.DesignMetrics).join(sql.DesignEntityMetrics)

    if pose_ids:
        stmt = join_stmt.where(PoseJob.id.in_(pose_ids))
    else:
        stmt = join_stmt

    if design_ids:
        stmt = stmt.where(sql.DesignData.id.in_(design_ids))
    else:
        stmt = stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_poses_dataframe(session: Session, pose_ids: Iterable[int] = None) -> pd.DataFrame:
    # , design_ids: Iterable[int] = None
    """Load and format every PoseJob instance's, PoseMetrics and EntityMetrics

    Optionally limit those loaded to certain PoseJob.id's

    Args:
        session: A session object to complete the transaction
        pose_ids: PoseJob instance identifiers for which metrics are desired
    Returns:
        A DataFrame formatted with every metric in PoseMetrics and EntityMetrics. The final DataFrame will have an entry
            corresponding to each Entity in EntityData for a total of PoseJob's X number of Entities entries
    """
    #     design_ids: DesignData instance identifiers for which metrics are desired
    # Accessing only the PoseMetrics and EntityMetrics
    pm_c = [c for c in sql.PoseMetrics.__table__.columns if not c.primary_key]
    pm_names = [c.name for c in pm_c]
    entity_metadata_c = [sql.ProteinMetadata.n_terminal_helix,
                         sql.ProteinMetadata.c_terminal_helix,
                         sql.ProteinMetadata.thermophilicity]
    em_c = [c for c in (*sql.EntityMetrics.__table__.columns, *entity_metadata_c) if not c.primary_key]
    em_names = [f'entity_{c.name}' if c.name != 'entity_id' else c.name for c in em_c]
    # em_c = [c for c in sql.EntityMetrics.__table__.columns if not c.primary_key]
    # em_names = [f'entity_{c.name}' if c.name != 'entity_id' else c.name for c in em_c]
    selected_columns = (*pm_c, *em_c)
    selected_column_names = (*pm_names, *em_names)

    # Construct the SQL query
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(PoseJob)\
        .join(sql.PoseMetrics).join(sql.EntityData).join(sql.EntityMetrics)
    if pose_ids:
        stmt = join_stmt.where(PoseJob.id.in_(pose_ids))
    else:
        stmt = join_stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_pose_metrics_dataframe(session: Session, pose_ids: Iterable[int] = None) -> pd.DataFrame:
    """Load and format every PoseJob instance's, PoseMetrics

    Optionally limit those loaded to certain PoseJob.id's

    Args:
        session: A session object to complete the transaction
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
    join_stmt = select(selected_columns).select_from(PoseJob)\
        .join(sql.PoseMetrics)
    if pose_ids:
        stmt = join_stmt.where(PoseJob.id.in_(pose_ids))
    else:
        stmt = join_stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_entity_metrics_dataframe(session: Session, pose_ids: Iterable[int] = None) -> pd.DataFrame:
    """Load and format every PoseJob instance's, EntityMetrics

    Optionally limit those loaded to certain PoseJob.id's

    Args:
        session: A session object to complete the transaction
        pose_ids: PoseJob instance identifiers for which metrics are desired
    Returns:
        A DataFrame formatted with the pose_id and EntityMetrics. The final DataFrame will have an entry corresponding
            to each Entity in EntityData for a total of PoseJob's X number of Entities entries
    """
    # Accessing only the PoseJob.id and EntityMetrics
    popse_id_c = sql.EntityData.pose_id
    entity_metadata_c = [sql.ProteinMetadata.n_terminal_helix,
                         sql.ProteinMetadata.c_terminal_helix,
                         sql.ProteinMetadata.thermophilicity]
    em_c = [c for c in (*sql.EntityMetrics.__table__.columns, *entity_metadata_c) if not c.primary_key]
    em_names = [f'entity_{c.name}' if c.name != 'entity_id' else c.name for c in em_c]
    selected_columns = (popse_id_c, *em_c,)
    selected_column_names = (popse_id_c.name, *em_names,)

    # Construct the SQL query
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(PoseJob)\
        .join(sql.EntityData).join(sql.EntityMetrics)
    if pose_ids:
        stmt = join_stmt.where(PoseJob.id.in_(pose_ids))
    else:
        stmt = join_stmt

    return load_and_format(session, stmt, selected_column_names)


def load_sql_designs_dataframe(session: Session, pose_ids: Iterable[int] = None, design_ids: Iterable[int] = None) \
        -> pd.DataFrame:
    """Load and format every PoseJob instance associated DesignMetrics for each design associated with the PoseJob

    Optionally limit those loaded to certain PoseJob.id's and DesignData.id's

    Args:
        session: A session object to complete the transaction
        pose_ids: PoseJob instance identifiers for which metrics are desired
        design_ids: DesignData instance identifiers for which metrics are desired
    Returns:
        The pandas DataFrame formatted with the every metric in DesignMetrics. The final DataFrame will
            have an entry for each DesignData
    """
    # dd_c = [sql.DesignData.pose_id, sql.DesignData.design_id]
    dd_c = (sql.DesignData.pose_id,)
    dd_names = [c.name for c in dd_c]
    dm_c = [c for c in sql.DesignMetrics.__table__.columns if not c.primary_key]
    dm_names = [c.name for c in dm_c]
    em_c = [c for c in sql.DesignEntityMetrics.__table__.columns if not c.primary_key]
    # Remove design_id
    em_c.pop(em_c.index(sql.DesignEntityMetrics.design_id))
    em_names = [f'entity_{c.name}' if c.name != 'entity_id' else c.name for c in em_c]
    selected_columns = (*dd_c, *dm_c, *em_c)
    selected_column_names = (*dd_names, *dm_names, *em_names)

    # Construct the SQL query
    # Todo CAUTION Deprecated API features detected for 2.0! # Error issued for the below line
    join_stmt = select(selected_columns).select_from(sql.DesignData)\
        .join(sql.DesignMetrics).join(PoseJob).join(sql.DesignEntityMetrics)
    if pose_ids:
        stmt = join_stmt.where(PoseJob.id.in_(pose_ids))
    else:
        stmt = join_stmt

    if design_ids:
        stmt = stmt.where(sql.DesignData.id.in_(design_ids))
    else:
        stmt = stmt

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
        for pose_job, design in selected_poses_df.index.to_list():
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
        for pose_job, design in selected_poses_df.index.to_list():
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
    #         exit(1)
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
    #                       for pose in save_poses_df.index.to_list()]
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
    #     exit()  # terminate(output=False)
    # else:
    #     logger.critical('Missing a required method to provide or find metrics from %s. If you meant to gather '
    #                     'metrics from every pose in your input specification, ensure you include the --global '
    #                     'argument' % putils.program_output)
    #     exit()

    # Format selected poses for output
    putils.make_path(job.output_directory)
    logger.info(f'Relevant files will be saved in the output directory: {job.output_directory}')

    if job.save_total:
        total_df_filename = os.path.join(job.output_directory, 'TotalPosesTrajectoryMetrics.csv')
        total_df.to_csv(total_df_filename)
        logger.info(f'Total Pose/Designs DataFrame was written to: {total_df_filename}')

    logger.info(f'{len(save_poses_df)} poses were selected')
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


def select_from_cluster_map(selected_members: Sequence[Any], cluster_map: dict[Any, list[Any]], number: int = 1) \
        -> list[int]:
    """From a mapping of cluster representatives to their members, select members based on their ranking in the
    selected_members sequence

    Args:
        selected_members: A sorted list of members that are members of the cluster_map
        cluster_map: A mapping of cluster representatives to their members
        number: The number of members to select
    Returns:
        The indices of selected_members, trimmed and retrieved according to cluster_map membership
    """
    membership_representative_map = cluster.invert_cluster_map(cluster_map)
    representative_found: dict[Any, list[Any]] = {}
    not_found = []
    for idx, member in enumerate(selected_members):
        cluster_representative = membership_representative_map.get(member, None)
        if cluster_representative:
            if cluster_representative not in representative_found:
                # Include. This representative hasn't been identified
                representative_found[cluster_representative] = [idx]  # [member]
            else:
                # This cluster has already been found, and it was identified again. Report and only
                # include the highest ranked pose in the output as it provides info on all occurrences
                representative_found[cluster_representative].append(idx)  # member)
        else:
            not_found.append(idx)  # member)

    final_member_indices = []
    for member_indices in representative_found.values():
        final_member_indices.extend(member_indices[:number])

    if not_found:
        logger.warning(f"Couldn't locate the following members:\n\t%s\nAdding all of these to your selection..." %
                       '\n\t'.join(map(str, [selected_members[idx] for idx in not_found])))
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
        for pose_job, design in selected_poses_df.index.to_list()[:job.select_number]:
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
        selected_designs = selected_poses_df.index.to_list()
        job.select_number = \
            len(selected_designs) if len(selected_designs) < job.select_number else job.select_number
        # if job.allow_multiple_poses:
        #     logger.info(f'Choosing {job.select_number} designs, from the top ranked designs regardless of pose')
        #     loc_result = selected_designs[:job.select_number]
        #     results = {pose_job: design for pose_job, design in loc_result}
        # else:  # elif job.designs_per_pose:
        designs_per_pose = job.designs_per_pose
        logger.info(f'Choosing up to {job.select_number} designs, with {designs_per_pose} designs per pose')
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
    #         sequence_metrics = set(trajectory_df.columns.get_level_values(-1).to_list())
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
        total_df_filename = os.path.join(job.output_directory, 'TotalPosesTrajectoryMetrics.csv')
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

    # Set up mechanism to solve sequence tagging preferences
    def solve_tags(n_of_tags: int) -> list[bool]:
        if job.tag_entities is None:
            boolean_tags = [False for _ in range(n_of_tags)]
        elif job.tag_entities == 'all':
            boolean_tags = [True for _ in range(n_of_tags)]
        elif job.tag_entities == 'single':
            boolean_tags = [True for _ in range(n_of_tags)]
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

        return boolean_tags

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
        tag_index = solve_tags(number_of_entities)
        number_of_tags = sum(tag_index)
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


def format_save_df(session: Session, pose_ids: Iterable[int], designs_df: pd.DataFrame) -> pd.DataFrame:
    """"""
    structure_type = 'structure_entity'
    pose_id = 'pose_id'
    pose_metrics_df = load_sql_pose_metrics_dataframe(session, pose_ids=pose_ids)
    pose_metrics_df.set_index(pose_id, inplace=True)
    logger.debug(f'pose_metrics_df: {pose_metrics_df}')  # Todo remove
    # save_df = pose_metrics_df.join(designs_df)  # , on='pose_id')
    # Join the designs_df (which may not have pose_id as index) with the pose_id indexed pose_metrics_df
    save_df = designs_df.join(pose_metrics_df, on=pose_id, rsuffix='_DROP')
    save_df.drop(save_df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)
    save_df.columns = pd.MultiIndex.from_product([['pose'], save_df.columns.tolist()],
                                                 names=[structure_type, 'metric'])
    logger.debug(f'save_df: {save_df}')  # Todo remove
    # Get the EntityMetrics and unstack in the format
    # structure_entity  1          2
    # metric            go fish    go fish
    # pose_id1          3   4      3    3
    # pose_id2          5   3      3    3
    # ...
    entity_metrics_df = load_sql_entity_metrics_dataframe(session, pose_ids=pose_ids)
    logger.debug(f'entity_metrics_df: {entity_metrics_df}')  # Todo remove
    # Get the first return from factorize since we just care about the unique "code" values
    entity_metrics_df[structure_type] = \
        entity_metrics_df.groupby(pose_id).entity_id.transform(lambda x: pd.factorize(x)[0]) + 1
    # entity_metrics_df[structure_type] = entity_metrics_df.groupby(pose_id).entity_id.cumcount() + 1
    # entity_metrics_df[structure_type] = \
    #     (entity_metrics_df.groupby('pose_id').entity_id.cumcount() + 1).apply(lambda x: f'entity_{x}')
    entity_metrics_df = entity_metrics_df.drop_duplicates([pose_id, structure_type])
    logger.debug(f'entity_metrics_df AFTER factorize and deduplication: {entity_metrics_df}')  # Todo remove
    # Make the stacked entity df and use the pose_id index to join with the above df
    pose_oriented_entity_df = entity_metrics_df.set_index([pose_id, structure_type]).unstack().swaplevel(axis=1)
    # pose_oriented_entity_df = entity_metrics_df.unstack().swaplevel(axis=1)
    logger.debug(f'pose_oriented_entity_df: {pose_oriented_entity_df}')  # Todo remove
    save_df = save_df.join(pose_oriented_entity_df, rsuffix='_DROP')  # , on=pose_id
    # save_df.drop(save_df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)
    logger.debug(f'Final save_df: {save_df}')  # Todo remove

    return save_df


def sql_poses(pose_jobs: Iterable[PoseJob]) -> list[PoseJob]:
    """Select PoseJob instances based on filters and weighting of all design summary metrics

    Args:
        pose_jobs: The PoseJob instances for which selection is desired
    Returns:
        The selected PoseJob instances
    """
    job = job_resources_factory.get()
    default_weight_metric = config.default_weight_parameter[job.design.method]
    session = job.current_session
    # Figure out poses from a specification file, filters, and weights
    # if job.specification_file:
    pose_ids = [pose_job.id for pose_job in pose_jobs]
    # design_ids = [design.id for pose_job in pose_jobs for design in pose_job.current_designs]
    #     total_df = load_sql_poses_dataframe(session, pose_ids=pose_ids, design_ids=design_ids)
    #     selected_poses_df = \
    #         metrics.prioritize_design_indices(total_df, filters=job.filter, weights=job.weight,
    #                                           protocols=job.protocol, function=job.weight_function)
    #     # Remove excess pose instances
    #     number_chosen = 0
    #     selected_indices, selected_poses = [], set()
    #     for pose_job, design in selected_poses_df.index.to_list():
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

    # Figure out designs from dataframe, filters, and weights
    total_df = load_sql_poses_dataframe(session, pose_ids=pose_ids)  # , design_ids=design_ids)
    if total_df.empty:
        raise utils.MetricsError(
            f"For the input PoseJobs, there aren't metrics collected. Use the '{flags.analysis}' module or perform some"
            " design module before selection")
    designs_df = load_sql_designs_dataframe(session, pose_ids=pose_ids)  # , design_ids=design_ids)
    if not designs_df.empty:
        # designs_df has a multiplicity of number_of_entities from DesignEntityMetrics table join
        # columns_to_keep = [c.name for c in sql.DesignMetrics.numeric_columns()]
        # print(designs_df.columns.tolist())
        # designs_df = designs_df.loc[:, [col for col in columns_to_keep if col in designs_df.columns]]
        # print(designs_df.columns.tolist())
        pose_designs_mean_df = designs_df.groupby('pose_id').mean(numeric_only=True)
        # Use the pose_id index to join to the total_df
        # This will create a total_df that is the number_of_entities X larger than the number of poses
        total_df = total_df.join(pose_designs_mean_df, rsuffix='_DROP')
        total_df.drop(total_df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)

    if job.filter or job.protocol:
        df_multiplicity = len(pose_jobs[0].entity_data)
        logger.warning('Filtering statistics have an increased representation due to included Entity metrics. '
                       f'Typically, values reported for each filter will be ~{df_multiplicity}x over those actually '
                       'present')
    selected_poses_df = \
        metrics.prioritize_design_indices(total_df, filters=job.filter, weights=job.weight, protocols=job.protocol,
                                          default_weight=default_weight_metric, function=job.weight_function)
    # Remove excess pose instances
    selected_pose_ids = utils.remove_duplicates(selected_poses_df['pose_id'].tolist())[:job.select_number]
    selected_poses = [session.get(PoseJob, id_) for id_ in selected_pose_ids]

    # # Select by clustering analysis
    # if job.cluster:
    # Sort results according to clustered poses if clustering exists
    if job.cluster.map:
        # cluster_map: dict[str | PoseJob, list[str | PoseJob]] = {}
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

    if len(final_poses) > job.select_number:
        final_poses = final_poses[:job.select_number]
        logger.info(f'Found {len(final_poses)} Poses after applying your select-number criteria')

    final_pose_id_to_identifier = {pose_job.id: pose_job.pose_identifier for pose_job in final_poses}

    # Format selected PoseJob with metrics for output
    # save_poses_df = format_save_df(session, selected_pose_ids, pose_designs_mean_df)
    save_poses_df = format_save_df(session, final_pose_id_to_identifier.keys(), pose_designs_mean_df)
    # Rename the identifiers to human-readable names
    save_poses_df.reset_index(inplace=True)
    save_poses_df['pose_identifier'] = save_poses_df['pose_id'].map(final_pose_id_to_identifier)
    save_poses_df.set_index('pose_identifier', inplace=True)

    # Format selected poses for output
    putils.make_path(job.output_directory)
    logger.info(f'Relevant files will be saved in the output directory: {job.output_directory}')

    if job.save_total:
        total_df_filename = os.path.join(job.output_directory, 'TotalPosesTrajectoryMetrics.csv')
        total_df.to_csv(total_df_filename)
        logger.info(f'Total Pose/Designs DataFrame written to: {total_df_filename}')

    logger.info(f'{len(save_poses_df)} Poses were selected')
    # if len(save_poses_df) != len(total_df):
    if job.filter or job.weight:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-{"Filtered" if job.filter else ""}'
                                                           f'{"Weighted" if job.weight else ""}PoseMetrics.csv')
    else:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-PoseMetrics.csv')
    save_poses_df.to_csv(new_dataframe)
    logger.info(f'New DataFrame with selected poses written to: {new_dataframe}')

    return final_poses


def sql_designs(pose_jobs: Iterable[PoseJob]) -> list[PoseJob]:
    """Select PoseJob instances based on filters and weighting of all design summary metrics

    Args:
        pose_jobs: The PoseJob instances for which selection is desired
    Returns:
        The selected PoseJob instances with selected designs stored in the .current_designs attribute
    """
    # global exceptions  # nonlocal exceptions
    job = job_resources_factory.get()
    default_weight_metric = config.default_weight_parameter[job.design.method]
    session = job.current_session

    # Figure out poses from a specification file, filters, and weights
    # if job.specification_file:
    pose_ids = [pose_job.id for pose_job in pose_jobs]
    design_ids = [design.id for pose_job in pose_jobs for design in pose_job.current_designs]
    #     total_df = load_sql_designs_dataframe(session, pose_ids=pose_ids, design_ids=design_ids)
    #     selected_poses_df = \
    #         metrics.prioritize_design_indices(total_df, filters=job.filter, weights=job.weight,
    #                                           protocols=job.protocol, function=job.weight_function)
    #     # Specify the result order according to any filtering, weighting, and number
    #     results = {}
    #     for pose_id, design in selected_poses_df.index.to_list()[:job.select_number]:
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
    #     total_df = load_sql_designs_dataframe(job.current_session)
    #     if job.protocol:
    #         group_df = total_df.groupby('protocol')
    #         df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
    #                        keys=list(zip(group_df.groups, repeat('mean'))))
    #     else:
    #         df = pd.concat([total_df], axis=1, keys=['pose', 'metric'])
    #     # Figure out designs from dataframe, filters, and weights
    #     selected_poses_df = metrics.prioritize_design_indices(df, filters=job.filter, weights=job.weight,
    #                                                           protocols=job.protocol, function=job.weight_function)
    #     selected_designs = selected_poses_df.index.to_list()
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
    # Figure out designs from dataframe, filters, and weights
    # designs_df = load_sql_designs_dataframe(session, pose_ids=pose_ids, design_ids=design_ids)
    # pose_designs_mean_df = designs_df.groupby('pose_id').mean()
    metrics_df = load_sql_metrics_dataframe(session, pose_ids=pose_ids, design_ids=design_ids)
    if metrics_df.empty:
        raise utils.MetricsError(
            f"For the input PoseJobs, there aren't metrics collected. Use the '{flags.analysis}' module or perform some"
            " design module before selection")
    if job.filter or job.protocol:
        df_multiplicity = len(pose_jobs[0].entity_data)**2
        logger.warning('Filtering statistics have an increased representation due to included Entity metrics. '
                       f'Typically, values reported for each filter will be ~{df_multiplicity}x over those actually '
                       'present')
    total_df = metrics_df
    selected_designs_df = \
        metrics.prioritize_design_indices(total_df, filters=job.filter, weights=job.weight, protocols=job.protocol,
                                          default_weight=default_weight_metric, function=job.weight_function)
    # # Groupby the design_id to remove extra instances of 'entity_id'
    # # This will turn these values into average which is fine since we just want the order
    # pose_designs_mean_df = selected_designs_df.groupby('design_id').mean()
    # Drop duplicated values keeping the order of the DataFrame
    selected_designs_df.drop_duplicates(subset=design_id, inplace=True)
    # Get the pose_id and the design_id for each found design
    # # Set the index according to 'pose_id', 'design_id'
    # selected_designs_df.set_index(['pose_id', 'design_id'], inplace=True)
    # selected_designs = selected_designs_df.index.tolist()
    # selected_designs = selected_designs_df[['pose_id', 'design_id']].values.tolist()
    selected_design_ids = selected_designs_df[design_id].tolist()
    selected_pose_ids = selected_designs_df[pose_id].tolist()
    selected_designs = list(zip(selected_pose_ids, selected_design_ids))

    # Specify the result order according to any filtering, weighting, and number
    number_selected = len(selected_designs_df)
    job.select_number = number_selected if number_selected < job.select_number else job.select_number
    designs_per_pose = job.designs_per_pose
    logger.info(f'Choosing up to {job.select_number} Designs, with {designs_per_pose} Design(s) per Pose')
    selected_designs_iter = iter(selected_designs)
    number_chosen = count(0)
    # selected_pose_id_to_design_ids = defaultdict(list)  # Alt way
    selected_pose_id_to_design_ids = {}
    try:
        while next(number_chosen) <= job.select_number:
            pose_id, design_id = next(selected_designs_iter)
            # Alt way, but doesn't count designs_per_pose
            # selected_pose_id_to_design_ids[pose_id].append(design_id)
            _designs = selected_pose_id_to_design_ids.get(pose_id, None)
            if _designs:
                if len(_designs) >= designs_per_pose:
                    # We already have too many for this pose, continue with search
                    continue
                _designs.append(design_id)
            else:
                selected_pose_id_to_design_ids[pose_id] = [design_id]
    except StopIteration:  # We exhausted selected_designs_iter
        pass

    logger.info(f'{len(selected_pose_id_to_design_ids)} Poses were selected')
    # Format selected designs for output
    putils.make_path(job.output_directory)
    logger.info(f'Relevant files will be saved in the output directory: {job.output_directory}')

    # Create new output of designed PDB's  # Todo attach the program state to these files for downstream use?
    pose_id_to_identifier = {}
    design_id_to_identifier = {}
    results = []
    selected_design_ids = []
    for pose_id, design_ids in selected_pose_id_to_design_ids.items():
        pose_job = session.get(PoseJob, pose_id)
        pose_id_to_identifier[pose_id] = pose_job.pose_identifier
        selected_design_ids.extend(design_ids)
        current_designs = []
        for design_id in design_ids:
            design = session.get(sql.DesignData, design_id)
            design_name = design.name
            design_id_to_identifier[design_id] = design_name
            design_structure_path = design.structure_path
            if design_structure_path:
                out_path = os.path.join(job.output_directory, f'{pose_job.project}-{design_name}.pdb')
                if os.path.exists(design_structure_path):
                    shutil.copy(design_structure_path, out_path)  # [i])))
                else:
                    pose_job.log.error(f'No file found for "{design_structure_path}"')
                # exceptions.append(utils.ReportException(f'No file found for "{file_path}"'))
                continue
            # Tod0 is this grabbing the structure everytime? We need to ensure no other files exist or refine this glob
            #  with an extension
            # file_path = os.path.join(pose_job.designs_path, f'*{design_name}*')
            # # print('file_path', file_path)
            # file = sorted(glob(file_path))
            # # print('files', file)
            # if not file:  # Add to exceptions
            #     pose_job.log.error(f'No file found for "{file_path}"')
            #     # exceptions.append(utils.ReportException(f'No file found for "{file_path}"'))
            #     continue
            # out_path = os.path.join(job.output_directory, f'{pose_job.output_modifier}-{design_name}.pdb')
            # if not os.path.exists(out_path):
            #     shutil.copy(file[0], out_path)  # [i])))

            current_designs.append(design)

        pose_job.current_designs = current_designs
        results.append(pose_job)

    design_metrics_df = load_sql_designs_dataframe(session, design_ids=selected_design_ids)
    # designs_df has a multiplicity of number_of_entities from DesignEntityMetrics table join
    design_metrics_df.set_index('design_id', inplace=True)
    # Format selected PoseJob with metrics for output
    # save_poses_df = selected_designs_df
    save_poses_df = format_save_df(session,
                                   selected_pose_id_to_design_ids.keys(),
                                   # # Remove the 'design_id' index
                                   # selected_designs_df.droplevel(-1, axis=0))
                                   design_metrics_df)
    #                                selected_designs_df.set_index('design_id').loc[selected_design_ids, :])
    # Rename the identifiers to human-readable names
    save_designs_df.reset_index(inplace=True, col_level=-1, col_fill='pose')
    save_designs_df[('pose', 'design_name')] = save_designs_df[('pose', design_id)].map(design_id_to_identifier)
    # print('AFTER design_name', save_designs_df)
    save_designs_df[('pose', 'pose_identifier')] = save_designs_df[('pose', pose_id)].map(pose_id_to_identifier)
    # print('AFTER pose_identifier', save_designs_df)
    save_designs_df.set_index([('pose', 'pose_identifier'), ('pose', 'design_name')], inplace=True)
    save_designs_df.index.rename(['pose_identifier', 'design_name'], inplace=True)
    # print('AFTER set_index', save_designs_df)

    if job.save_total:
        total_df_filename = os.path.join(job.output_directory, 'TotalPosesTrajectoryMetrics.csv')
        total_df.to_csv(total_df_filename)
        logger.info(f'Total Pose/Designs DataFrame written to: {total_df_filename}')

    if job.filter or job.weight:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-{"Filtered" if job.filter else ""}'
                                                           f'{"Weighted" if job.weight else ""}DesignMetrics.csv')
    else:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-DesignMetrics.csv')
    save_designs_df.to_csv(new_dataframe)
    logger.info(f'New DataFrame with selected designs written to: {new_dataframe}')

    return results  # , exceptions


def sql_sequences(pose_jobs: list[PoseJob]) -> list[PoseJob]:
    """Perform design selection followed by sequence formatting on those designs

    Args:
        pose_jobs: The PoseJob instances for which selection is desired
    Returns:
        The matching PoseJob instances
    """
    from symdesign.third_party.DnaChisel.dnachisel.DnaOptimizationProblem.NoSolutionError import NoSolutionError
    job = job_resources_factory.get()
    results = sql_designs(pose_jobs)
    # Set up output_file pose_jobs for __main__.terminate()
    return_pose_jobs = results
    job.output_file = os.path.join(job.output_directory, 'SelectedDesigns.poses')

    # Set up mechanism to solve sequence tagging preferences
    def solve_tags(n_of_tags: int) -> list[bool]:
        if job.tag_entities is None:
            boolean_tags = [False for _ in range(n_of_tags)]
        elif job.tag_entities == 'all':
            boolean_tags = [True for _ in range(n_of_tags)]
        elif job.tag_entities == 'single':
            boolean_tags = [True for _ in range(n_of_tags)]
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

        return boolean_tags

    if job.multicistronic:
        intergenic_sequence = job.multicistronic_intergenic_sequence
    else:
        intergenic_sequence = ''

    # Format sequences for expression
    tag_sequences, final_sequences, inserted_sequences, nucleotide_sequences = {}, {}, {}, {}
    codon_optimization_errors = {}
    for pose_job in results:
        pose_job.load_pose()
        # Create the source_gap_mutations which provide mutation style dict for each gep
        # from the reference to the structure sequence
        entity_sequences = [entity.sequence for entity in pose_job.pose.entities]
        source_gap_mutations = [generate_mutations(entity.reference_sequence, entity.sequence,
                                                   zero_index=True, only_gaps=True)
                                for entity in pose_job.pose.entities]
        number_of_entities = pose_job.number_of_entities
        entity_taggable_indices = solve_tags(number_of_entities)
        number_of_tags_requested = sum(entity_taggable_indices)

        # Find termini data
        # entity_termini_availability, entity_helical_termini = {}, {}
        entity_termini_availability = []
        entity_helical_termini = []
        entity_true_termini = []
        for entity in pose_job.pose.entities:
            # entity.retrieve_info_from_api()
            # entity.reference_sequence
            pose_entity_id = f'{pose_job}_{entity.name}'
            termini_availability = pose_job.pose.get_termini_accessibility(entity)
            logger.debug(f'Designed Entity {pose_entity_id} has the termini accessible: '
                         f'{termini_availability}')
            if job.avoid_tagging_helices:
                termini_helix_availability = \
                    pose_job.pose.get_termini_accessibility(entity, report_if_helix=True)
                logger.debug(f'Designed Entity {pose_entity_id} has helical termini available: '
                             f'{termini_helix_availability}')
                termini_availability = {'n': termini_availability['n'] and not termini_helix_availability['n'],
                                        'c': termini_availability['c'] and not termini_helix_availability['c']}
                entity_helical_termini.append(termini_helix_availability)

            # Report and finalize for this Entity
            logger.info(f'Designed Entity {pose_entity_id} has the termini available for tagging: '
                        f'{termini_availability}')
            entity_termini_availability.append(termini_availability)
            entity_true_termini.append([term for term, is_true in termini_availability.items() if is_true])

        for design in pose_job.current_designs:
            design_id_str = f'{pose_job} Design {design.name}'
            design_sequence = design.metrics.sequence
            entity_number_residues_begin = entity_number_residues_end = 0

            # Make sequence as list instead of string so can use list.insert()
            designed_atom_sequences = []
            for entity_data in pose_job.entity_data:
                entity_number_residues_end += entity_data.metrics.number_of_residues
                designed_atom_sequences.append(
                    list(design_sequence[entity_number_residues_begin:
                                         entity_number_residues_end]))
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
                design_entity_id = f'{design.name}-{entity_name}'
                entity_names.append(entity_name)
                mutations = generate_mutations(''.join(design_sequence), source_sequence, zero_index=True)
                logger.debug(f'Found mutations: {mutations}')
                # Insert the disordered residues into the design sequence
                for residue_index, mutation in source_gap_mutations[entity_idx].items():
                    # residue_index is zero indexed
                    new_aa_type = mutation['from']
                    logger.debug(f'Inserting {new_aa_type} into index {residue_index} on Entity {entity_name}')
                    # design_pose.insert_residue_type(new_aa_type, index=residue_index,
                    #                                 chain_id=entity.chain_id)
                    design_sequence.insert(residue_index, new_aa_type)
                    # Adjust mutations to account for insertion
                    for mutation_index in sorted(mutations.keys(), reverse=True):
                        if mutation_index < residue_index:
                            break
                        else:  # Mutation should be incremented by one
                            mutations[mutation_index + 1] = mutations.pop(mutation_index)

                # Check for expression tag addition to the designed sequences after disorder addition
                inserted_design_sequence = ''.join(design_sequence)
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
                else:
                    pretag_sequence = inserted_design_sequence
                logger.debug(f'The inserted sequence sequence is:\n{inserted_design_sequence}')
                logger.debug(f'The pretag sequence is:\n{pretag_sequence}')

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
                    entity_sequence_and_tags.append({'sequence': formatted_design_sequence,
                                                     'tag': {}})
                else:
                    if not selected_tag:
                        # Find compatible tags from matching PDB observations
                        possible_matching_tags = []
                        for uniprot_id in data.uniprot_ids:
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

                        iteration = count()
                        while next(iteration) < len(tag_names):
                            try:
                                preferred_tag_index_2 = tag_names[iteration:].index(job.preferred_tag)
                                if tag_termini[preferred_tag_index_2] in entity_true_termini[entity_idx]:
                                    selected_tag = possible_matching_tags[preferred_tag_index_2]
                                    break
                            except ValueError:
                                selected_tag = \
                                    expression.select_tags_for_sequence(design_entity_id,
                                                                        possible_matching_tags,
                                                                        preferred=job.preferred_tag,
                                                                        **entity_termini_availability[entity_idx])
                                break

                    if selected_tag.get('name'):
                        entity_missing_tags[entity_idx] = False
                        logger.debug(f'The pre-existing, identified tag is:\n{selected_tag}')
                    entity_sequence_and_tags.append({'sequence': formatted_design_sequence,
                                                     'tag': selected_tag})

            # After selecting individual Entity tags, consider tagging the whole Design
            if number_of_tags_requested > 0:
                number_of_found_tags = number_of_entities - sum(entity_missing_tags)
                # When fewer than the requested number of tags were identified
                if number_of_tags_requested > number_of_found_tags:
                    print(f'There were {number_of_tags_requested} requested tags for {design_id_str} and '
                          f'{number_of_found_tags} were found')
                    current_tag_options = \
                        '\n\t'.join([f'{idx + 1} - {entity_names[idx]}\n'
                                     f'\tAvailable Termini: {entity_termini_availability[idx]}'
                                     f'\n\t\t   TAGS: {seq_tag_options["tag"]}'
                                     for idx, seq_tag_options in enumerate(entity_sequence_and_tags)])
                    print(f'Current Tag Options:\n\t{current_tag_options}')
                    if job.avoid_tagging_helices:
                        print('Helical Termini:\n\t%s'
                              % '\n\t'.join(f'{entity_name}\t{availability}'
                                            for entity_name, availability in zip(entity_names, entity_helical_termini)))
                    satisfied = input('If this is acceptable, enter "continue", otherwise, '
                                      f'you can modify the tagging options with any other input.{input_string}')
                    if satisfied == 'continue':
                        number_of_found_tags = number_of_tags_requested

                    iteration = count()
                    while number_of_tags_requested != number_of_found_tags:
                        iteration_idx = next(iteration)
                        if iteration_idx == number_of_entities:
                            print("You've seen all options, but the number of tags requested, "
                                  f'{number_of_tags_requested} != {number_of_found_tags}, the number of tags found')
                            satisfied = input('If you are satisfied with this, enter "C" (continue), otherwise enter '
                                              'anything and you can view all remaining options starting from the '
                                              f'first entity{input_string}')
                            if satisfied == 'C':
                                break
                            else:  # Start over
                                iteration = count()
                                continue
                        for entity_idx, entity_missing_tag in enumerate(entity_missing_tags[iteration_idx:]):
                            if entity_missing_tag and entity_taggable_indices[entity_idx]:  # Isn't tagged but could be
                                # pose_entity_id = f'{pose_job}_{entity_names[entity_idx]}'
                                print(f'Entity {pose_job}_{entity_names[entity_idx]} is missing a tag. '
                                      f'Would you like to tag this entity?')
                                if not boolean_choice():
                                    continue
                            else:
                                continue
                            # Solve by preferred_tag or user input
                            if job.preferred_tag:
                                tag = job.preferred_tag
                                termini = validate_input(f'Your preferred tag {tag} will be added to one of the termini'
                                                         '. Which termini would you prefer?', ['n', 'c'])
                            else:
                                print('Tag options include:\n\t%s' %
                                      '\n\t'.join([f'{idx} - {tag}' for idx, tag in enumerate(expression.tags, 1)]))
                                tag_input = \
                                    validate_input('Which of the above tags would you like to use? Enter the '
                                                   'number of your preferred option',
                                                   list(map(str, range(1, 1 + len(expression.tags)))))
                                # if tag_input.isdigit():
                                # Adjust for python indexing
                                tag_index = int(tag_input) - 1
                                tag = list(expression.tags.keys())[tag_index]
                                termini = validate_input(f'Your tag {tag} will be added to one of the termini'
                                                         '. Which termini would you prefer?', ['n', 'c'])
                            selected_sequence_and_tag = entity_sequence_and_tags[entity_idx]
                            if termini == 'n':
                                new_tag_sequence = expression.tags[tag] \
                                    + 'SG' + selected_sequence_and_tag['sequence'][:12]
                            else:  # termini == 'c'
                                new_tag_sequence = selected_sequence_and_tag['sequence'][-12:] \
                                    + 'GS' + expression.tags[tag]
                            selected_sequence_and_tag['tag'] = {'name': tag, 'sequence': new_tag_sequence}
                            entity_missing_tags[entity_idx] = False
                            break

                        number_of_found_tags = number_of_entities - sum(entity_missing_tags)
                # When more than the requested number of tags were identified
                elif number_of_tags_requested < number_of_found_tags:
                    print(f'There were only {number_of_tags_requested} requested tags for design {pose_job} and '
                          f'{number_of_found_tags} were found')
                    while number_of_tags_requested != number_of_found_tags:
                        tag_input = input(f'Which tag would you like to remove? Enter the number of the currently '
                                          'configured tag option that you would like to remove. If you would like '
                                          f'to keep all, specify "keep"\n\t%s\n{input_string}'
                                          % '\n\t'.join([f'{idx + 1} - {entity_names[idx]}\n\t\t{tag_options["tag"]}'
                                                         for idx, tag_options in enumerate(entity_sequence_and_tags)]))
                        if tag_input == 'keep':
                            break
                        elif tag_input.isdigit():
                            tag_input = int(tag_input)
                            if tag_input <= len(entity_sequence_and_tags):
                                # Set that this entity is now missing a tag
                                entity_missing_tags[tag_input - 1] = True
                                selected_sequence_and_tag = entity_sequence_and_tags[tag_input - 1]
                                selected_sequence_and_tag['tag'] = \
                                    {'name': None, 'termini': None, 'sequence': None}
                                # tag = list(expression.tags.keys())[tag_input - 1]
                                break
                            else:
                                print("Input doesn't match an integer from the available options. Please try again")
                        else:
                            print(f'"{tag_input}" is an invalid input. Try again')
                        number_of_found_tags = number_of_entities - sum(entity_missing_tags)

            # Apply all tags to the sequences
            # Todo indicate the linkers that will be used!
            #  Request a new one if not ideal!
            cistronic_sequence = ''
            for idx, (entity_name, sequence_tag) in enumerate(zip(entity_names, entity_sequence_and_tags)):
                design_string = f'{design.name}_{entity_name}'
                tag, sequence = sequence_tag['tag'], sequence_tag['sequence']
                # print('TAG:\n', tag.get('sequence'), '\nSEQUENCE:\n', sequence)
                chimeric_tag_sequence = tag.get('sequence')
                tagged_sequence = expression.add_expression_tag(chimeric_tag_sequence, sequence)
                if chimeric_tag_sequence and tagged_sequence == sequence:  # tag exists and no tag added
                    tag_sequence = expression.tags[tag.get('name')]
                    if tag.get('termini') == 'n':
                        if tagged_sequence[0] == 'M':  # Remove existing n-term Met to append tag to n-term
                            tagged_sequence = tagged_sequence[1:]
                        tagged_sequence = tag_sequence + 'SG' + tagged_sequence
                    else:  # termini == 'c'
                        tagged_sequence = tagged_sequence + 'GS' + tag_sequence

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
                    generate_mutations(tagged_sequence, ''.join(designed_atom_sequences[idx]),
                                       return_all=True, keep_gaps=True, zero_index=True)
                # for mutations in generated_insertion_mutations.values():
                #     reference = mutations['from']
                #     query = mutations['to']

                # Reduce to sequence only
                inserted_sequences[design_string] = \
                    f'{"".join([res["to"] for res in generated_insertion_mutations.values()])}\n' \
                    f'{"".join([res["from"] for res in generated_insertion_mutations.values()])}'
                # # Reduce to sequence only
                # inserted_sequences[design_string] = \
                #     f'{"".join([res["to"] for res in all_insertions.values()])}\n{tagged_sequence}'
                logger.info(f'Formatted sequence comparison:\n{inserted_sequences[design_string]}')
                final_sequences[design_string] = tagged_sequence
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
            if job.multicistronic:
                nucleotide_sequences[str(pose_job)] = cistronic_sequence

    # Report Errors
    if codon_optimization_errors:
        # Todo utilize errors
        error_file = \
            write_sequences(codon_optimization_errors, csv=job.csv,
                            file_name=os.path.join(job.output_directory, 'OptimizationErrorProteinSequences'))
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
