import sys
import os
import numpy as np
from itertools import chain
from Bio.PDB import PDBParser, PDBIO, Superimposer
import FragUtils as Frag


# Globals
module = 'Cluster All to All RMSD of Individual Fragments:'


# def get_biopdb_ca(biopdb_structure):
#     ca_atoms = []
#     for atom in biopdb_structure.get_atoms():
#         if atom.get_id() == 'CA':
#             ca_atoms.append(atom)
#
#     return ca_atoms
#
#
# def center(bio_pdb):
#     ca_atoms = get_biopdb_ca(bio_pdb)
#
#     # Get Central Residue (5 Residue Fragment => 3rd Residue) CA Coordinates
#     center_ca_atom = ca_atoms[2]
#     center_ca_coords = center_ca_atom.get_coord()
#
#     # Center Such That Central Residue CA is at Origin
#     for atom in bio_pdb.get_atoms():
#         atom.set_coord(np.add(atom.get_coord(), -center_ca_coords))
#
#
# def cluster_fragment_rmsds(rmsd_file_path):
#     # Get All to All RMSD File
#     with open(rmsd_file_path, 'r') as rmsd_file:
#         rmsd_file_lines = rmsd_file.readlines()
#
#     # Create Dictionary Containing Structure Name as Key and a List of Neighbors within RMSD Threshold as Values
#     rmsd_dict = {}
#     for line in rmsd_file_lines:
#         line = line.rstrip()
#         line = line.split()
#
#         if line[0] in rmsd_dict:
#             rmsd_dict[line[0]].append(line[1])
#         else:
#             rmsd_dict[line[0]] = [line[1]]
#
#         if line[1] in rmsd_dict:
#             rmsd_dict[line[1]].append(line[0])
#         else:
#             rmsd_dict[line[1]] = [line[0]]
#
#     print(module, 'Finished Creating RMSD Dictionary with a total of', len(rmsd_dict), 'Clusters')
#
#     # Cluster
#     return_clusters = []
#     flattened_query = list(chain.from_iterable(rmsd_dict.values()))
#
#     while flattened_query != list():
#         # Find Structure With Most Neighbors within RMSD Threshold
#         max_neighbor_structure = None
#         max_neighbor_count = 0
#         for query_structure in rmsd_dict:
#             neighbor_count = len(rmsd_dict[query_structure])
#             if neighbor_count > max_neighbor_count:
#                 max_neighbor_structure = query_structure
#                 max_neighbor_count = neighbor_count
#
#         # Create Cluster Containing Max Neighbor Structure (Cluster Representative) and its Neighbors
#         cluster = rmsd_dict[max_neighbor_structure]
#         return_clusters.append((max_neighbor_structure, cluster))
#
#         # Remove Claimed Structures from rmsd_dict
#         claimed_structures = [max_neighbor_structure] + cluster
#         updated_dict = {}
#         for query_structure in rmsd_dict:
#             if query_structure not in claimed_structures:
#                 tmp_list = []
#                 for idx in rmsd_dict[query_structure]:
#                     if idx not in claimed_structures:
#                         tmp_list.append(idx)
#                 updated_dict[query_structure] = tmp_list
#             else:
#                 updated_dict[query_structure] = []
#
#         rmsd_dict = updated_dict
#         flattened_query = list(chain.from_iterable(rmsd_dict.values()))
#
#     return return_clusters


def main():
    print(module, 'Beginning')
    # Fragment DB Directory
    frag_db_dir = os.path.join(os.getcwd(), 'all_individual_frags')

    # Outdir
    aligned_clusters_outdir = os.path.join(os.getcwd(), 'i_clusters')
    if not os.path.exists(aligned_clusters_outdir):
        os.makedirs(aligned_clusters_outdir)

    # Get All to All RMSD File
    rmsd_file_path = os.path.join(os.getcwd(), 'all_individual_frags', 'all_to_all_rmsd.txt')

    return_clusters = Frag.cluster_fragment_rmsds(rmsd_file_path)
    print(module, 'Clustering Finished, Creating Representatives')

    # Align all Cluster Members to Cluster Representative
    cluster_count = 1
    for cluster in return_clusters:
        if len(cluster[1]) >= 10:
            cluster_rep = cluster[0]
            cluster_rep_pdb_path = os.path.join(frag_db_dir, cluster[0] + '.pdb')

            parser = PDBParser()
            cluster_rep_biopdb = parser.get_structure(cluster_rep, cluster_rep_pdb_path)

            cluster_outdir = os.path.join(aligned_clusters_outdir, str(cluster_count))
            if not os.path.exists(cluster_outdir):
                os.makedirs(cluster_outdir)
            Frag.center(cluster_rep_biopdb)

            io1 = PDBIO()
            io1.set_structure(cluster_rep_biopdb)
            io1.save(os.path.join(cluster_outdir, cluster_rep + '_representative.pdb'))

            for structure_idx in cluster[1]:
                structure_path = os.path.join(frag_db_dir, structure_idx + '.pdb')

                parser = PDBParser()
                idx_biopdb = parser.get_structure(structure_idx, structure_path)

                sup = Superimposer()
                sup.set_atoms(Frag.get_biopdb_ca(cluster_rep_biopdb), Frag.get_biopdb_ca(idx_biopdb))
                sup.apply(idx_biopdb)

                io2 = PDBIO()
                io2.set_structure(idx_biopdb)
                io2.save(os.path.join(cluster_outdir, structure_idx + '_aligned.pdb'))
            cluster_count += 1

    print(module, 'Finished')


if __name__ == '__main__':
    main()
