import logging
import os
from itertools import repeat
from typing import Any, Iterable

import pandas as pd

from symdesign import flags, protocols, utils
from symdesign.protocols import metrics  # , PoseDirectory
from symdesign.resources.job import job_resources_factory
import symdesign.utils.path as putils

logger = logging.getLogger(__name__)


def poses(pose_directories: Iterable[protocols.protocols.PoseDirectory]) -> list[protocols.protocols.PoseDirectory]:
    job = job_resources_factory.get()

    if job.specification_file:  # Figure out poses from a specification file, filters, and weights
        loc_result = [(pose_directory, design) for pose_directory in pose_directories
                      for design in pose_directory.specific_designs]
        total_df = protocols.load_total_dataframe(pose_directories, pose=True)
        selected_poses_df = \
            metrics.prioritize_design_indices(total_df.loc[loc_result, :], filter=job.filter, weight=job.weight,
                                              protocol=job.protocol, function=job.weight_function)
        # Remove excess pose instances
        number_chosen = 0
        selected_indices, selected_poses = [], set()
        for pose_directory, design in selected_poses_df.index.to_list():
            if pose_directory not in selected_poses:
                selected_poses.add(pose_directory)
                selected_indices.append((pose_directory, design))
                number_chosen += 1
                if number_chosen == job.number:
                    break

        # Specify the result order according to any filtering and weighting
        # Drop the specific design for the dataframe. If they want the design, they should run select_sequences
        save_poses_df = \
            selected_poses_df.loc[selected_indices, :].droplevel(-1)  # .droplevel(0, axis=1).droplevel(0, axis=1)
        # # convert selected_poses to PoseDirectory objects
        # selected_poses = [pose_directory for pose_directory in pose_directories if pose_dir_name in selected_poses]
    elif job.total:  # Figure out poses from file/directory input, filters, and weights
        total_df = protocols.load_total_dataframe(pose_directories, pose=True)
        if job.protocol:  # Todo adapt to protocol column not in Trajectories right now...
            group_df = total_df.groupby(putils.protocol)
            df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
                           keys=list(zip(group_df.groups, repeat('mean'))))
        else:
            df = pd.concat([total_df], axis=1, keys=['pose', 'metric'])
        # Figure out designs from dataframe, filters, and weights
        selected_poses_df = metrics.prioritize_design_indices(df, filter=job.filter, weight=job.weight,
                                                              protocol=job.protocol, function=job.weight_function)
        # Remove excess pose instances
        number_chosen = 0
        selected_indices, selected_poses = [], set()
        for pose_directory, design in selected_poses_df.index.to_list():
            if pose_directory not in selected_poses:
                selected_poses.add(pose_directory)
                selected_indices.append((pose_directory, design))
                number_chosen += 1
                if number_chosen == job.number:
                    break

        # Specify the result order according to any filtering and weighting
        # Drop the specific design for the dataframe. If they want the design, they should run select_sequences
        save_poses_df = \
            selected_poses_df.loc[selected_indices, :].droplevel(-1)  # .droplevel(0, axis=1).droplevel(0, axis=1)
        # # convert selected_poses to PoseDirectory objects
        # selected_poses = [pose_directory for pose_directory in pose_directories if pose_dir_name in selected_poses]
    elif job.dataframe:  # Figure out poses from a pose dataframe, filters, and weights
        if not pose_directories:  # not job.directory:
            logger.critical(f'If using a --{flags.dataframe} for selection, you must include the directory where '
                            f'the designs are located in order to properly select designs. Please specify '
                            f'-d/--{flags.directory} with your command')
            exit(1)

        total_df = pd.read_csv(job.dataframe, index_col=0, header=[0, 1, 2])
        total_df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)

        selected_poses_df = metrics.prioritize_design_indices(total_df, filter=job.filter, weight=job.weight,
                                                              protocol=job.protocol, function=job.weight_function)
        # Only drop excess columns as there is no MultiIndex, so no design in the index
        save_poses_df = selected_poses_df.droplevel(0, axis=1).droplevel(0, axis=1)
        program_root = job.program_root
        selected_poses = [protocols.PoseDirectory.from_pose_id(pose, root=program_root)
                          for pose in save_poses_df.index.to_list()]
    else:  # Generate design metrics on the spot
        raise NotImplementedError('This functionality is currently broken')
        selected_poses, selected_poses_df, total_df = [], pd.DataFrame(), pd.DataFrame()
        logger.debug('Collecting designs to sort')
        if job.metric == 'score':
            metric_design_dir_pairs = [(des_dir.score, des_dir.path) for des_dir in pose_directories]
        elif job.metric == 'fragments_matched':
            metric_design_dir_pairs = [(des_dir.number_of_fragments, des_dir.path)
                                       for des_dir in pose_directories]
        else:  # This is impossible with the argparse options
            raise NotImplementedError(f'The metric "{job.metric}" is not supported!')
            metric_design_dir_pairs = []

        logger.debug(f'Sorting designs according to "{job.metric}"')
        metric_design_dir_pairs = [(score, path) for score, path in metric_design_dir_pairs if score]
        sorted_metric_design_dir_pairs = sorted(metric_design_dir_pairs, key=lambda pair: (pair[0] or 0),
                                                reverse=True)
        top_designs_string = \
            f'Top ranked Designs according to {job.metric}:\n\t{job.metric.title()}\tDesign\n\t%s'
        results_strings = ['%.2f\t%s' % tup for tup in sorted_metric_design_dir_pairs]
        logger.info(top_designs_string % '\n\t'.join(results_strings[:500]))
        if len(pose_directories) > 500:
            design_source = f'top_{job.metric}'
            # default_output_tuple = (utils.starttime, job.module, design_source)
            putils.make_path(job.all_scores)
            designs_file = \
                os.path.join(job.all_scores, f'{utils.starttime}_{job.module}_{design_source}_pose.scores')
            with open(designs_file, 'w') as f:
                f.write(top_designs_string % '\n\t'.join(results_strings))
            logger.info(f'Stdout performed a cutoff of ranked Designs at ranking 500. See the output design file '
                        f'"{designs_file}" for the remainder')

        terminate(output=False)
    # else:
    #     logger.critical('Missing a required method to provide or find metrics from %s. If you meant to gather '
    #                     'metrics from every pose in your input specification, ensure you include the --global '
    #                     'argument' % putils.program_output)
    #     exit()

    if job.total and job.save_total:
        total_df_filename = os.path.join(program_root, 'TotalPosesTrajectoryMetrics.csv')
        total_df.to_csv(total_df_filename)
        logger.info(f'Total Pose/Designs DataFrame was written to: {total_df_filename}')

    logger.info(f'{len(save_poses_df)} poses were selected')
    if len(save_poses_df) != len(total_df):
        if job.filter or job.weight:
            new_dataframe = os.path.join(program_root, f'{utils.starttime}-{"Filtered" if job.filter else ""}'
                                                       f'{"Weighted" if job.weight else ""}PoseMetrics.csv')
        else:
            new_dataframe = os.path.join(program_root, f'{utils.starttime}-PoseMetrics.csv')
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

        final_poses = select_from_cluster_map(selected_poses, cluster_map, number=job.cluster.number)
        logger.info(f'Selected {len(final_poses)} poses after clustering')
    else:  # Try to generate the cluster_map?
        # raise utils.InputError(f'No --{flags.cluster_map} was provided. To cluster poses, specify:'
        logger.info(f'No {flags.cluster_map} was provided. To cluster poses, specify:'
                    f'"{putils.program_command} {flags.cluster_poses}" or '
                    f'"{putils.program_command} {flags.protocol} '
                    f'--{flags.modules} {flags.cluster_poses} {flags.select_poses}')
        logger.info('Grabbing all selected poses')
        final_poses = selected_poses
        # cluster_map: dict[str | protocols.PoseDirectory, list[str | protocols.PoseDirectory]] = {}
        # # {pose_string: [pose_string, ...]} where key is representative, values are matching designs
        # # OLD -> {composition: {pose_string: cluster_representative}, ...}
        # compositions: dict[tuple[str, ...], list[protocols.PoseDirectory]] = \
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
        # logger.info(f'Found {len(cluster_map)} unique clusters from {len(pose_directories)} pose inputs. '
        #             f'All clusters stored in {pose_cluster_file}')
    # else:
    #     logger.info('Grabbing all selected poses')
    #     final_poses = selected_poses

    if len(final_poses) > job.number:
        final_poses = final_poses[:job.number]
        logger.info(f'Found {len(final_poses)} poses after applying your number selection criteria')

    return final_poses


def select_from_cluster_map(selected_members: Iterable[Any], cluster_map: dict[Any, list[Any]], number: int = 1) \
        -> list[Any]:
    """

    Args:
        cluster_map: A mapping of cluster representatives to their members
        selected_members: A sorted list of members that are members of the cluster_map
        number: The number of members to select
    Returns:
        The selected_members, trimmed and retrieved according to cluster_map membership
    """
    membership_representative_map = protocols.cluster.invert_cluster_map(cluster_map)
    representative_found: dict[Any, list[Any]] = {}
    not_found = []
    for idx, member in enumerate(selected_members):
        cluster_representative = membership_representative_map.get(member, None)
        if cluster_representative:
            if cluster_representative not in representative_found:
                # Include. This representative hasn't been identified
                representative_found[cluster_representative] = [member]
            else:
                # This cluster has already been found, and it was identified again. Report and only
                # include the highest ranked pose in the output as it provides info on all occurrences
                representative_found[cluster_representative].append(member)
        else:
            not_found.append(member)

    final_members = []
    for members in representative_found.values():
        final_members.extend(members[:number])

    if not_found:
        logger.warning(f"Couldn't locate the following members:\n\t%s\nAdding all of these to your selection..." %
                       '\n\t'.join(not_found))
        # 'Was {flags.cluster_poses} only run on a subset of the poses that were selected?
        final_members.extend(not_found)

    return final_members
