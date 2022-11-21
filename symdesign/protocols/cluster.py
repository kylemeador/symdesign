from __future__ import annotations

import logging
import os
from itertools import combinations, repeat

from symdesign.protocols import protocols
from symdesign.resources.job import job_resources_factory
from symdesign import utils
from symdesign.utils import path as putils

logger = logging.getLogger(__name__)


def run(pose_directories: list[protocols.PoseDirectory]) -> \
        dict[str | protocols.PoseDirectory, list[str | protocols.PoseDirectory]] | None:
    job = job_resources_factory.get()
    pose_cluster_map: dict[str | protocols.PoseDirectory, list[str | protocols.PoseDirectory]] = {}
    """Mapping which takes the format:
    {pose_string: [pose_string, ...]} where keys are representatives, values are matching designs
    """
    results = []
    if job.mode == 'ialign':
        # Measure the alignment of all selected pose_directories
        # all_files = [design.source_file for design in pose_directories]

        # Need to change directories to prevent issues with the path length being passed to ialign
        prior_directory = os.getcwd()
        os.chdir(job.data)  # os.path.join(job.data, 'ialign_output'))
        temp_file_dir = os.path.join(os.getcwd(), 'temp')
        putils.make_path(temp_file_dir)

        # Save the interface for each design to the temp directory
        design_interfaces = []
        for pose_dir in pose_directories:
            pose_dir.identify_interface()  # calls design.load_pose()
            # Todo this doesn't work for asymmetric Poses
            interface = pose_dir.pose.get_interface()
            design_interfaces.append(
                # interface.write(out_path=os.path.join(temp_file_dir, f'{pose_dir.name}_interface.pdb')))  # Todo reinstate
                interface.write(out_path=os.path.join(temp_file_dir, f'{pose_dir.name}.pdb')))

        design_directory_pairs = list(combinations(pose_directories, 2))
        if job.multi_processing:
            results = utils.mp_starmap(utils.cluster.ialign, combinations(design_interfaces, 2), processes=job.cores)
        else:
            for idx, (interface_file1, interface_file2) in enumerate(combinations(design_interfaces, 2)):
                # is_score = utils.cluster.ialign(design1.source, design2.source, out_path='ialign')
                results.append(utils.cluster.ialign(interface_file1, interface_file2))
                #                                     out_path=os.path.join(job.data, 'ialign_output'))

        if results:
            # Separate interfaces which fall below a threshold
            is_threshold = 0.4  # 0.5  # TODO
            pose_pairs = []
            for idx, is_score in enumerate(results):
                if is_score > is_threshold:
                    pose_pairs.append(set(design_directory_pairs[idx]))
                    # pose_pairs.append({design1, design2})

            # Cluster all those designs together that are in alignment
            # Add both orientations to the pose_cluster_map
            for pose1, pose2 in pose_pairs:
                cluster1 = pose_cluster_map.get(pose1)
                try:
                    cluster1.append(pose2)
                except AttributeError:
                    pose_cluster_map[pose1] = [pose2]

                cluster2 = pose_cluster_map.get(pose2)
                try:
                    cluster2.append(pose1)
                except AttributeError:
                    pose_cluster_map[pose2] = [pose1]

        # Return to prior directory
        os.chdir(prior_directory)
    elif job.mode == 'transform':
        # First, identify the same compositions
        compositions: dict[tuple[str, ...], list[protocols.PoseDirectory]] = \
            utils.cluster.group_compositions(pose_directories)
        if job.multi_processing:
            results = utils.mp_map(utils.cluster.cluster_transformations, compositions.values(), processes=job.cores)
        else:
            for composition_group in compositions.values():
                results.append(utils.cluster.cluster_transformations(composition_group))

        # Add all clusters to the pose_cluster_map
        for result in results:
            pose_cluster_map.update(result.items())
    elif job.mode == 'rmsd':
        logger.critical(f"The mode {job.mode} hasn't been thoroughly debugged")
        # First, identify the same compositions
        compositions: dict[tuple[str, ...], list[protocols.PoseDirectory]] = \
            utils.cluster.group_compositions(pose_directories)
        # pairs_to_process = [grouping for entity_tuple, pose_directories in compositions.items()
        #                     for grouping in combinations(pose_directories, 2)]
        # composition_pairings = [combinations(pose_directories, 2) for entity_tuple, pose_directories in compositions.items()]
        # Find the rmsd between a pair of poses
        if job.multi_processing:
            results = utils.mp_map(utils.cluster.pose_pair_by_rmsd, compositions.items(), processes=job.cores)
        else:
            for entity_tuple, _pose_directories in compositions.items():
                results.append(utils.cluster.pose_pair_by_rmsd(_pose_directories))

        # Add all clusters to the pose_cluster_map
        for result in results:
            pose_cluster_map.update(result.items())
    else:
        exit(f"{job.mode} isn't a viable mode")

    if pose_cluster_map:
        if job.as_objects:
            pass  # They are by default objects
        else:
            for representative in list(pose_cluster_map.keys()):
                # remove old entry and convert all arguments to pose_id strings, saving as pose_id strings
                pose_cluster_map[str(representative)] = \
                    [str(member) for member in pose_cluster_map.pop(representative)]

        return pose_cluster_map
    else:
        logger.info('No significant clusters were located. Clustering ended')
        return None
