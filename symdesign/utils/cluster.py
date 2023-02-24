from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import sklearn

from symdesign import metrics

# Globals
logger = logging.getLogger(__name__)


# def pose_rmsd_mp(pose_jobs: list[PoseJob], cores: int = 1):
#     """Map the RMSD for a Nanohedra output based on building block directory (ex 1abc_2xyz)
#
#     Args:
#         pose_jobs: List of relevant design directories
#         cores: Number of multiprocessing cores to run
#     Returns:
#         (dict): {composition: {pair1: {pair2: rmsd, ...}, ...}, ...}
#     """
#     pose_map = {}
#     pairs_to_process = []
#     singlets = {}
#     for pair1, pair2 in combinations(pose_jobs, 2):
#         if pair1.composition == pair2.composition:
#             singlets.pop(pair1.composition, None)
#             pairs_to_process.append((pair1, pair2))
#         else:
#             # add all individual poses to a singles pool. pair2 is included in pair1, no need to add additional
#             singlets[pair1.composition] = pair1
#     compositions: dict[tuple[str, ...], list[PoseJob]] = group_compositions(pose_jobs)
#     pairs_to_process = [grouping for entity_tuple, pose_jobs in compositions.items()
#                         for grouping in combinations(pose_jobs, 2)]
#     # find the rmsd between a pair of poses.  multiprocessing to increase throughput
#     _results = mp_map(pose_pair_rmsd, pairs_to_process, processes=cores)
#
#     # Make dictionary with all pairs
#     for pair, pair_rmsd in zip(pairs_to_process, _results):
#         protein_pair_path = os.path.basename(pair[0].building_blocks)
#         # protein_pair_path = pair[0].composition
#         # pose_map[result[0]] = result[1]
#         if protein_pair_path in pose_map:
#             # # {composition: {(pair1, pair2): rmsd, ...}, ...}
#             # {composition: {pair1: {pair2: rmsd, ...}, ...}, ...}
#             if str(pair[0]) in pose_map[protein_pair_path]:
#                 pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
#                 if str(pair[1]) not in pose_map[protein_pair_path]:
#                     pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}  # add the pair with itself
#         else:
#             pose_map[protein_pair_path] = {str(pair[0]): {str(pair[0]): 0.0}}  # add the pair with itself
#             pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
#             pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}  # add the pair with itself
#
#     # Add all singlets (poses that are missing partners) to the map
#     for protein_pair in singlets:
#         protein_path = os.path.basename(protein_pair)
#         if protein_path in pose_map:
#             # This logic is impossible??
#             pose_map[protein_path][str(singlets[protein_pair])] = {str(singlets[protein_pair]): 0.0}
#         else:
#             pose_map[protein_path] = {str(singlets[protein_pair]): {str(singlets[protein_pair]): 0.0}}
#
#     return pose_map
#
#
# def pose_rmsd_s(all_pose_job_pairs):
#     pose_map = {}
#     for pair in combinations(all_pose_job_pairs, 2):
#         if pair[0].composition == pair[1].composition:
#             protein_pair_path = pair[0].composition
#             # Grab designed resides from the pose_job_pair
#             pair_rmsd = pose_pair_rmsd(pair)
#             # des_residue_list = [pose.info['des_residues'] for pose in pair]
#             # # could use the union as well...
#             # des_residue_set = index_intersection({pair[n]: set(pose_residues)
#             #                                               for n, pose_residues in enumerate(des_residue_list)})
#             # if des_residue_set == list():  # when the two structures are not significantly overlapped
#             #     pair_rmsd = np.nan
#             # else:
#             #     pdb_parser = PDBParser(QUIET=True)
#             #     # pdb = parser.get_structure(pdb_name, filepath)
#             #     pair_structures = [pdb_parser.get_structure(str(pose), pose.asu) for pose in pair]
#             #     # returns a list with all ca atoms from a structure
#             #     # pair_atoms = utils.get_rmsd_atoms([pair[0].asu, pair[1].asu], utils.get_biopdb_ca)
#             #     # pair_atoms = utils.get_rmsd_atoms([pair[0].path, pair[1].path], utils.get_biopdb_ca)
#             #
#             #     # pair should be a structure...
#             #     # for structure in pair_structures:
#             #     #     for residue in structure.residues:
#             #     #         print(residue)
#             #     #         print(residue[0])
#             #     rmsd_residue_list = [[residue for residue in structure.residues  # residue.get_id()[1] is res number
#             #                           if residue.get_id()[1] in des_residue_set] for structure in pair_structures]
#             #
#             #     # rmsd_residue_list = [[residue for residue in structure.residues
#             #     #                       if residue.get_id()[1] in des_residue_list[n]]
#             #     #                      for n, structure in enumerate(pair_structures)]
#             #
#             #     # print(rmsd_residue_list)
#             #     pair_atom_list = [[atom for atom in unfold_entities(entity_list, 'A') if atom.get_id() == 'CA']
#             #                       for entity_list in rmsd_residue_list]
#             #     # [atom for atom in structure.get_atoms if atom.get_id() == 'CA']
#             #     # pair_atom_list = utils.get_rmsd_atoms(rmsd_residue_list, utils.get_biopdb_ca)
#             #     # pair_rmsd = utils.superimpose(pair_atoms, threshold)
#             #
#             #     pair_rmsd = utils.superimpose(pair_atom_list)  # , threshold)
#             # if not pair_rmsd:
#             #     continue
#             if protein_pair_path in pose_map:
#                 # {composition: {(pair1, pair2): rmsd, ...}, ...}
#                 if str(pair[0]) in pose_map[protein_pair_path]:
#                     pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
#                     if str(pair[1]) not in pose_map[protein_pair_path]:
#                         pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}
#                     # else:
#                     #     print('\n' * 6 + 'NEVER ACCESSED' + '\n' * 6)
#                     #     pose_map[pair[0].composition][str(pair[1])][str(pair[1])] = 0.0
#                 # else:
#                 #     print('\n' * 6 + 'ACCESSED' + '\n' * 6)
#                 #     pose_map[pair[0].composition][str(pair[0])] = {str(pair[1]): pair_rmsd}
#                 #     pose_map[pair[0].composition][str(pair[0])][str(pair[0])] = 0.0
#                 # pose_map[pair[0].composition][(str(pair[0]), str(pair[1]))] = pair_rmsd[2]
#             else:
#                 pose_map[protein_pair_path] = {str(pair[0]): {str(pair[0]): 0.0}}
#                 pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
#                 pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}
#                 # pose_map[pair[0].composition] = {(str(pair[0]), str(pair[1])): pair_rmsd[2]}
#
#     return pose_map


def predict_best_pose_from_transformation_cluster(train_trajectories_file, training_clusters):  # UNUSED
    """From full training Nanohedra, Rosetta Sequecnce Design analyzed trajectories, train a linear model to select the
    best trajectory from a group of clustered poses given only the Nanohedra Metrics

    Args:
        train_trajectories_file (str): Location of a Cluster Trajectory Analysis .csv with complete metrics for cluster
        training_clusters (dict): Mapping of cluster representative to cluster members

    Returns:
        (sklearn.linear_model)
    """
    possible_lin_reg = {'MultiTaskLassoCV': sklearn.linear_model.MultiTaskLassoCV,
                        'LassoCV': sklearn.linear_model.LassoCV,
                        'MultiTaskElasticNetCV': sklearn.linear_model.MultiTaskElasticNetCV,
                        'ElasticNetCV': sklearn.linear_model.ElasticNetCV}
    idx_slice = pd.IndexSlice
    trajectory_df = pd.read_csv(train_trajectories_file, index_col=0, header=[0, 1, 2])
    # 'dock' category is synonymous with nanohedra metrics
    trajectory_df = trajectory_df.loc[:, idx_slice[['pose', 'no_constraint'],
                                                   ['mean', 'dock', 'seq_design'], :]].droplevel(1, axis=1)
    # scale the data to a standard gaussian distribution for each trajectory independently
    # Todo ensure this mechanism of scaling is correct for each cluster individually
    scaler = sklearn.preprocessing.StandardScaler()
    train_traj_df = pd.concat([scaler.fit_transform(trajectory_df.loc[cluster_members, :])
                               for cluster_members in training_clusters.values()], keys=list(training_clusters.keys()),
                              axis=0)

    # standard_scale_traj_df[train_traj_df.columns] = standard_scale.transform(train_traj_df)

    # select the metrics which the linear model should be trained on
    nano_traj = train_traj_df.loc[:, metrics.fragment_metrics]

    # select the Rosetta metrics to train model on
    # potential_training_metrics = set(train_traj_df.columns).difference(nanohedra_metrics)
    # rosetta_select_metrics = query_user_for_metrics(potential_training_metrics, mode='design', level='pose')
    rosetta_metrics = {'shape_complementarity': sklearn.preprocessing.StandardScaler(),  # I think a gaussian dist is preferable to MixMax
                       # 'protocol_energy_distance_sum': 0.25,  This will select poses by evolution
                       'int_composition_similarity': sklearn.preprocessing.StandardScaler(),  # gaussian preferable to MixMax
                       'interface_energy': sklearn.preprocessing.StandardScaler(),  # gaussian preferable to MaxAbsScaler,
                       # 'observed_evolution': 0.25}  # also selects by evolution
                       }
    # assign each metric a weight proportional to it's share of the total weight
    rosetta_select_metrics = {item: 1 / len(rosetta_metrics) for item in rosetta_metrics}
    # weighting scheme inherently standardizes the weights between [0, 1] by taking a linear combination of the metrics
    targets = metrics.prioritize_design_indices(train_trajectories_file, weight=rosetta_select_metrics)  # weight=True)

    # for proper MultiTask model training, must scale the selected metrics. This is performed on trajectory_df above
    # targets2d = train_traj_df.loc[:, rosetta_select_metrics.keys()]
    pose_traj_df = train_traj_df.loc[:, idx_slice['pose', 'int_composition_similarity']]
    no_constraint_traj_df = \
        train_traj_df.loc[:, idx_slice['no_constraint',
                                       set(rosetta_metrics.keys()).difference('int_composition_similarity')]]
    targets2d = pd.concat([pose_traj_df, no_constraint_traj_df])

    # split training and test dataset
    trajectory_train, trajectory_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(nano_traj, targets, random_state=42)
    trajectory_train2d, trajectory_test2d, target_train2d, target_test2d = \
        sklearn.model_selection.train_test_split(nano_traj, targets2d, random_state=42)
    # calculate model performance with cross-validation, alpha tuning
    alphas = np.logspace(-10, 10, 21)  # Todo why log space here?
    # then compare between models based on various model scoring parameters
    reg_scores, mae_scores = [], []
    for lin_reg, model in possible_lin_reg.items():
        if lin_reg.startswith('MultiTask'):
            trajectory_train, trajectory_test = trajectory_train2d, trajectory_test2d
            target_train, target_test = target_train2d, target_test2d
        # else:
        #     target = target_train
        test_reg = model(alphas=alphas).fit(trajectory_train, target_train)
        reg_scores.append(test_reg.score(trajectory_train, target_train))
        target_test_prediction = test_reg.predict(trajectory_test, target_test)
        mae_scores.append(sklearn.metrics.median_absolute_error(target_test, target_test_prediction))


def chose_top_pose_from_model(test_trajectories_file, clustered_poses, model):  # UNUSED
    """
    Args:
        test_trajectories_file (str): Location of a Nanohedra Trajectory Analysis .csv with Nanohedra metrics
        clustered_poses (dict): A set of clustered poses that share similar transformational parameters
    Returns:

    """
    test_docking_df = pd.read_csv(test_trajectories_file, index_col=0, header=[0, 1, 2])

    for cluster_representative, designs in clustered_poses.items():
        trajectory_df = test_docking_df.loc[designs, metrics.fragment_metrics]
        trajectory_df['model_predict'] = model.predict(trajectory_df)
        trajectory_df.sort_values('model_predict')
