import logging
from typing import Iterable

import pandas as pd

from .protocols import PoseDirectory
from .fragdock import fragment_dock
from .cluster import cluster_poses

logger = logging.getLogger(__name__)
orient = PoseDirectory.orient
expand_asu = PoseDirectory.expand_asu
rename_chains = PoseDirectory.rename_chains
check_clashes = PoseDirectory.check_clashes
generate_fragments = PoseDirectory.generate_interface_fragments
nanohedra = fragment_dock
interface_metrics = PoseDirectory.interface_metrics
optimize_designs = PoseDirectory.optimize_designs
refine = PoseDirectory.refine
interface_design = PoseDirectory.interface_design
analysis = PoseDirectory.interface_design_analysis
cluster_poses = cluster_poses


def load_total_dataframe(pose_directories: Iterable[PoseDirectory], pose: bool = False) -> pd.DataFrame:
    """Return a pandas DataFrame with the trajectories of every pose_directory loaded and formatted according to the
    design directory and design on the index

    Args:
        pose_directories: The pose_directories for which metrics are desired
        pose: Whether the total dataframe should contain the mean metrics from the pose or each individual design
    """
    # global pose_directories
    # global results
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
