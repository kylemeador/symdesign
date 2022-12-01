import logging
from itertools import repeat

import pandas as pd

from symdesign import flags, metrics, protocols
from symdesign.resources.job import job_resources_factory
from symdesign.resources.query.utils import bool_d, invalid_string
import symdesign.utils.path as putils

logger = logging.getLogger(__name__)


def select_poses(pose_directories):
    job = job_resources_factory.get()

    if job.specification_file:  # Figure out poses from a specification file, filters, and weights
        loc_result = [(pose_directory, design) for pose_directory in pose_directories
                      for design in pose_directory.specific_designs]
        df = protocols.load_total_dataframe(pose=True)
        selected_poses_df = \
            metrics.prioritize_design_indices(df.loc[loc_result, :], filter=job.filter, weight=job.weight,
                                              protocol=job.protocol, function=job.weight_function)
        # Remove excess pose instances
        number_chosen = 0
        selected_indices, selected_poses = [], set()
        for pose_directory, design in selected_poses_df.index.to_list():
            if pose_directory not in selected_poses:
                selected_poses.add(pose_directory)
                selected_indices.append((pose_directory, design))
                number_chosen += 1
                if number_chosen == job.select_number:
                    break

        # Specify the result order according to any filtering and weighting
        # Drop the specific design for the dataframe. If they want the design, they should run select_sequences
        save_poses_df = \
            selected_poses_df.loc[selected_indices, :].droplevel(-1)  # .droplevel(0, axis=1).droplevel(0, axis=1)
        # # convert selected_poses to PoseDirectory objects
        # selected_poses = [pose_directory for pose_directory in pose_directories if pose_dir_name in selected_poses]
    elif job.total:  # Figure out poses from file/directory input, filters, and weights
        df = protocols.load_total_dataframe(pose=True)
        if job.protocol:  # Todo adapt to protocol column not in Trajectories right now...
            group_df = df.groupby(putils.protocol)
            df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
                           keys=list(zip(group_df.groups, repeat('mean'))))
        else:
            df = pd.concat([df], axis=1, keys=['pose', 'metric'])
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
                if number_chosen == job.select_number:
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

        df = pd.read_csv(job.dataframe, index_col=0, header=[0, 1, 2])
        df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)

        selected_poses_df = metrics.prioritize_design_indices(df, filter=job.filter, weight=job.weight,
                                                              protocol=job.protocol, function=job.weight_function)
        # Only drop excess columns as there is no MultiIndex, so no design in the index
        save_poses_df = selected_poses_df.droplevel(0, axis=1).droplevel(0, axis=1)
        program_root = job.program_root
        selected_poses = [protocols.PoseDirectory.from_pose_id(pose, root=program_root)
                          for pose in save_poses_df.index.to_list()]
    else:  # Generate design metrics on the spot
        raise NotImplementedError('This functionality is currently broken')
        selected_poses, selected_poses_df, df = [], pd.DataFrame(), pd.DataFrame()
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
        total_df = os.path.join(program_root, 'TotalPosesTrajectoryMetrics.csv')
        df.to_csv(total_df)
        logger.info(f'Total Pose/Designs DataFrame was written to: {total_df}')

    logger.info(f'{len(save_poses_df)} poses were selected')
    if len(save_poses_df) != len(df):
        if job.filter or job.weight:
            new_dataframe = os.path.join(program_root, f'{utils.starttime}-{"Filtered" if job.filter else ""}'
                                                       f'{"Weighted" if job.weight else ""}PoseMetrics.csv')
        else:
            new_dataframe = os.path.join(program_root, f'{utils.starttime}-PoseMetrics.csv')
        save_poses_df.to_csv(new_dataframe)
        logger.info(f'New DataFrame with selected poses was written to: {new_dataframe}')

    # Sort results according to clustered poses if clustering exists
    if job.cluster_map:
        cluster_map = job.cluster_map
    else:
        cluster_map = \
            os.path.join(job.clustered_poses, putils.default_clustered_pose_file.format(utils.starttime, location))

    if os.path.exists(cluster_map):
        pose_cluster_map = utils.unpickle(cluster_map)
    else:  # Try to generate the cluster_map
        logger.info(f'No cluster pose map was found at {cluster_map}. Clustering similar poses may eliminate '
                    f'redundancy from the final design selection. To cluster poses broadly, '
                    f'run "{putils.program_command} {flags.cluster_poses}"')
        while True:
            # Todo add option to provide the path to an existing file
            confirm = input(f'Would you like to {flags.cluster_poses} on the subset of designs '
                            f'({len(selected_poses)}) located so far? [y/n]{input_string}')
            if confirm.lower() in bool_d:
                break
            else:
                print(f'{invalid_string} {confirm} is not a valid choice')

        pose_cluster_map: dict[str | PoseDirectory, list[str | PoseDirectory]] = {}
        # {pose_string: [pose_string, ...]} where key is representative, values are matching designs
        # OLD -> {composition: {pose_string: cluster_representative}, ...}

        if bool_d[confirm.lower()] or confirm.isspace():  # The user wants to separate poses
            compositions: dict[tuple[str, ...], list[PoseDirectory]] = \
                protocols.cluster.group_compositions(selected_poses)
            if job.multi_processing:
                mp_results = utils.mp_map(protocols.cluster.cluster_transformations, compositions.values(),
                                          processes=job.cores)
                for result in mp_results:
                    pose_cluster_map.update(result.items())
            else:
                for composition_group in compositions.values():
                    pose_cluster_map.update(protocols.cluster.cluster_transformations(composition_group))

            pose_cluster_file = utils.pickle_object(pose_cluster_map, name=cluster_map, out_path='')
            logger.info(f'Found {len(pose_cluster_map)} unique clusters from {len(pose_directories)} pose inputs. '
                        f'All clusters stored in {pose_cluster_file}')

    if pose_cluster_map:
        pose_cluster_membership_map = protocols.cluster.invert_cluster_map(pose_cluster_map)
        pose_clusters_found, pose_not_found = {}, []
        # Convert all the selected poses to their string representation
        # Todo this assumes the pose_cluster_map was not saved with job.as_object
        for idx, pose_directory in enumerate(map(str, selected_poses)):
            cluster_membership = pose_cluster_membership_map.get(pose_directory, None)
            if cluster_membership:
                if cluster_membership not in pose_clusters_found:
                    # Include as this pose hasn't been identified
                    pose_clusters_found[cluster_membership] = [pose_directory]
                else:
                    # This cluster has already been found, and it was identified again. Report and only
                    # include the highest ranked pose in the output as it provides info on all occurrences
                    pose_clusters_found[cluster_membership].append(pose_directory)
            else:
                pose_not_found.append(pose_directory)

        # Todo report the clusters and the number of instances
        final_poses = [members[0] for members in pose_clusters_found.values()]
        if pose_not_found:
            logger.warning(f"Couldn't locate the following poses:\n\t%s\nWas {flags.cluster_poses} only run on a "
                           'subset of the poses that were selected? Adding all of these to your final poses...'
                           % '\n\t'.join(pose_not_found))
            final_poses.extend(pose_not_found)
        logger.info(f'Found {len(final_poses)} poses after clustering')
    else:
        logger.info('Grabbing all selected poses')
        final_poses = selected_poses

    if len(final_poses) > job.select_number:
        final_poses = final_poses[:job.select_number]
        logger.info(f'Found {len(final_poses)} poses after applying your select_number selection criteria')

    return final_poses