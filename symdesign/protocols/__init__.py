import logging
from typing import Iterable

import pandas as pd

from . import cluster, fragdock, protocols, select
# from ..resources.job import JobResources, job_resources_factory

logger = logging.getLogger(__name__)
orient = protocols.PoseDirectory.orient
expand_asu = protocols.PoseDirectory.expand_asu
rename_chains = protocols.PoseDirectory.rename_chains
check_clashes = protocols.PoseDirectory.check_clashes
generate_fragments = protocols.PoseDirectory.generate_interface_fragments
interface_metrics = protocols.PoseDirectory.interface_metrics
optimize_designs = protocols.PoseDirectory.optimize_designs
refine = protocols.PoseDirectory.refine
interface_design = protocols.PoseDirectory.interface_design
analysis = protocols.PoseDirectory.interface_design_analysis
nanohedra = fragdock.fragment_dock
cluster_poses = cluster.cluster_poses
select_poses = select.poses
# select_sequences = select.sequences


def load_total_dataframe(pose_directories: Iterable[protocols.PoseDirectory], pose: bool = False) -> pd.DataFrame:
    """Return a pandas DataFrame with the trajectories of every pose_directory loaded and formatted according to the
    design directory and design on the index

    Args:
        pose_directories: The pose_directories for which metrics are desired
        pose: Whether the total dataframe should contain the mean metrics from the pose or each individual design
    """
    all_dfs = []  # None for design in pose_directories]
    for idx, pose_dir in enumerate(pose_directories):
        try:
            all_dfs.append(pd.read_csv(pose_dir.trajectories, index_col=0, header=[0]))
        except FileNotFoundError:  # as error
            # results[idx] = error
            logger.warning(f'{pose_dir}: No trajectory analysis file found. Skipping')

    if pose:
        for idx, df in enumerate(all_dfs):
            df.fillna(0., inplace=True)  # Shouldn't be necessary if saved files were formatted correctly
            # try:
            df.drop([index for index in df.index.to_list() if isinstance(index, float)], inplace=True)
            # Get rid of all individual trajectories and std, not mean
            pose_name = pose_directories[idx].name
            df.drop([index for index in df.index.to_list() if pose_name in index or 'std' in index], inplace=True)
            # except TypeError:
            #     for index in df.index.to_list():
            #         print(index, type(index))
    else:  # designs
        for idx, df in enumerate(all_dfs):
            # Get rid of all statistic entries, mean, std, etc.
            pose_name = pose_directories[idx].name
            df.drop([index for index in df.index.to_list() if pose_name not in index], inplace=True)

    # Add pose directory str as MultiIndex
    df = pd.concat(all_dfs, keys=[str(pose_dir) for pose_dir in pose_directories])
    df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)

    return df


# def run(pose_jobs, job: JobResources = job_resources_factory.get()):
#     # job = job_resources_factory.get()
#
#
# @run
# def orient():
#     """Run the orient protocol over the pose_jobs"""
#     nonlocal pose_jobs, job
#
#     _orient = protocols.PoseDirectory.orient
#     if processes > 1:
#         results = mp.map(_orient, pose_jobs, processes=processes)
#     else:
#         results = []
#         for pose_job in pose_jobs:
#             results.append(_orient)
#
#
# @run
# def expand_asu():
#     """Run the expand_asu protocol over the pose_jobs"""
#     nonlocal pose_jobs, job
#
#     _expand_asu = protocols.PoseDirectory.expand_asu
#     if processes > 1:
#         results = mp.map(_expand_asu, pose_jobs, processes=processes)
#     else:
#         results = []
#         for pose_job in pose_jobs:
#             results.append(_expand_asu)
#
#
# @run
# def rename_chains():
#     """Run the rename_chains protocol over the pose_jobs"""
#     nonlocal pose_jobs, job
#
#     _rename_chains = protocols.PoseDirectory.rename_chains
#     if processes > 1:
#         results = mp.map(_rename_chains, pose_jobs, processes=processes)
#     else:
#         results = []
#         for pose_job in pose_jobs:
#             results.append(_rename_chains)
#
#
# @run
# def check_clashes():
#     """Run the check_clashes protocol over the pose_jobs"""
#     nonlocal pose_jobs, job
#
#     _check_clashes = protocols.PoseDirectory.check_clashes
#     if processes > 1:
#         results = mp.map(_check_clashes, pose_jobs, processes=processes)
#     else:
#         results = []
#         for pose_job in pose_jobs:
#             results.append(_check_clashes)
#
#
# @run
# def generate_fragments():
#     """Run the generate_interface_fragments protocol over the pose_jobs"""
#     nonlocal pose_jobs, job
#
#     _generate_interface_fragments = protocols.PoseDirectory.generate_interface_fragments
#     if processes > 1:
#         results = mp.map(_generate_interface_fragments, pose_jobs, processes=processes)
#     else:
#         results = []
#         for pose_job in pose_jobs:
#             results.append(_generate_interface_fragments)
#
#
# @run
# def interface_metrics():
#     """Run the interface_metrics protocol over the pose_jobs"""
#     nonlocal pose_jobs, job
#
#     _interface_metrics = protocols.PoseDirectory.interface_metrics
#     if processes > 1:
#         results = mp.map(_interface_metrics, pose_jobs, processes=processes)
#     else:
#         results = []
#         for pose_job in pose_jobs:
#             results.append(_interface_metrics)
#
#
# @run
# def optimize_designs():
#     """Run the optimize_designs protocol over the pose_jobs"""
#     nonlocal pose_jobs, job
#
#     _optimize_designs = protocols.PoseDirectory.optimize_designs
#     if processes > 1:
#         results = mp.map(_optimize_designs, pose_jobs, processes=processes)
#     else:
#         results = []
#         for pose_job in pose_jobs:
#             results.append(_optimize_designs)
#
#
# @run
# def refine():
#     """Run the refine protocol over the pose_jobs"""
#     nonlocal pose_jobs, job
#
#     _refine = protocols.PoseDirectory.refine
#     if processes > 1:
#         results = mp.map(_refine, pose_jobs, processes=processes)
#     else:
#         results = []
#         for pose_job in pose_jobs:
#             results.append(_refine)
#
#
# @run
# def interface_design():
#     """Run the interface_design protocol over the pose_jobs"""
#     nonlocal pose_jobs, job
#
#     _interface_design = protocols.PoseDirectory.interface_design
#     if processes > 1:
#         results = mp.map(_interface_design, pose_jobs, processes=processes)
#     else:
#         results = []
#         for pose_job in pose_jobs:
#             results.append(_interface_design)
#
#
# @run
# def analysis():
#     """Run the interface_design_analysis protocol over the pose_jobs"""
#     nonlocal pose_jobs, job
#
#     _interface_design_analysis = protocols.PoseDirectory.interface_design_analysis
#     if processes > 1:
#         results = mp.map(_interface_design_analysis, pose_jobs, processes=processes)
#     else:
#         results = []
#         for pose_job in pose_jobs:
#             results.append(_interface_design_analysis)
#
#
# @run
# def nanohedra():
#     fragdock.fragment_dock
#
#
# @run
# def cluster_poses():
#     cluster.cluster_poses
#
#
# @run
# def select_poses():
#     select.poses
#
#
# @run
# def select_sequences():
#     select.sequences
