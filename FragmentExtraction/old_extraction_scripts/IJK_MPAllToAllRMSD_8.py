# import os
# from Bio.PDB import *
# import math
import IJ_MPAllToAllRMSD_5 as Frag5

# Globals
module = 'Map Fragment Pairs to Representative:'


# def guide_atom_rmsd(biopdb_guide_atoms_tup):
#     biopdb_1_guide_atoms = biopdb_guide_atoms_tup[0]
#     biopdb_2_guide_atoms = biopdb_guide_atoms_tup[1]
#
#     biopdb_1_id = biopdb_1_guide_atoms[0].get_full_id()[0]
#     biopdb_2_id = biopdb_2_guide_atoms[0].get_full_id()[0]
#
#     # Calculate RMSD
#     e1 = (biopdb_1_guide_atoms[0] - biopdb_2_guide_atoms[0]) ** 2
#     e2 = (biopdb_1_guide_atoms[1] - biopdb_2_guide_atoms[1]) ** 2
#     e3 = (biopdb_1_guide_atoms[2] - biopdb_2_guide_atoms[2]) ** 2
#     s = e1 + e2 + e3
#     m = s / float(3)
#     r = math.sqrt(m)
#
#     return biopdb_1_id, biopdb_2_id, r
#
#
# def get_all_base_root_paths(directory):
#     dir_paths = []
#     for root, dirs, files in os.walk(directory):
#         if not dirs:
#             dir_paths.append(root)
#
#     return dir_paths
#
#
# def get_all_pdb_file_paths(pdb_dir):
#     filepaths = []
#     for root, dirs, files in os.walk(pdb_dir):
#         for file in files:
#             if file.endswith('.pdb'):
#                 filepaths.append(os.path.join(pdb_dir, file))
#
#     return filepaths
#
#
# def get_guide_atoms_biopdb(biopdb_structure):
#     guide_atoms = []
#     for atom in biopdb_structure.get_atoms():
#         if atom.get_full_id()[2] == '9':
#             guide_atoms.append(atom)
#     return guide_atoms
#
#
# def mp_all_to_all_rmsd(ijk_clustered_dir):
#     # Cluster Directory Paths
#     cluster_dir_paths = Frag.get_all_base_root_paths(ijk_clustered_dir)
#
#     for cluster_dir in cluster_dir_paths:
#         # Get Fragment Guide Atoms for all Fragments in Cluster
#         frag_db_biopdb_guide_atoms = []
#         frag_paths = get_all_pdb_file_paths(cluster_dir)
#
#         for frag_path in frag_paths:
#             frag_pdb_name = os.path.splitext(os.path.basename(frag_path))[0]
#             parser = PDBParser()
#             frag_pdb_biopdb = parser.get_structure(frag_pdb_name, frag_path)
#             guide_atoms = Frag.get_guide_atoms_biopdb(frag_pdb_biopdb)
#             if len(guide_atoms) == 3:
#                 frag_db_biopdb_guide_atoms.append(guide_atoms)
#
#         # All to All Cluster Fragment Guide Atom RMSD (no redundant calculations) Processes Argument
#         if frag_db_biopdb_guide_atoms != list():
#             outfile_prefix = cluster_dir.split("/")[-1]
#             outfile_path = cluster_dir + "/" + outfile_prefix + "_all_to_all_guide_atom_rmsd.txt"
#             outfile = open(outfile_path, "w")
#             for i in range(len(frag_db_biopdb_guide_atoms) - 1):
#                 for j in range(i + 1, len(frag_db_biopdb_guide_atoms)):
#                     result = guide_atom_rmsd((frag_db_biopdb_guide_atoms[i], frag_db_biopdb_guide_atoms[j]))
#                     outfile.write("%s %s %s\n" %(result[0], result[1], str(result[2])))
#             outfile.close()
#
#
# def main():
#     print(module, 'Beginning')
#     ijk_clustered_dir = "/home/kmeador/yeates/interface_extraction/ijk_clustered_xtal_fragDB_1A"
#
#     cluster_dir_paths = Frag.get_all_base_root_paths(ijk_clustered_dir)
#     for cluster_dir in cluster_dir_paths:
#         print(module, 'Starting', cluster_dir)
#         frag_filepaths = Frag.get_all_pdb_file_paths(cluster_dir)
#         if len(frag_filepaths) > cluster_size_limit:
#             print(module, 'Number of Fragment:', len(frag_filepaths), '>', cluster_size_limit,
#                   '(cluster size limit). Sampled', cluster_size_limit, 'times instead.')
#             shuffle(frag_filepaths)
#             frag_filepaths = frag_filepaths[0:cluster_size_limit]
#         else:
#             print(module, 'Number of Fragments:', len(frag_filepaths))
#
#         # Get Fragment Guide Atoms for all Fragments in Cluster
#         frag_db_guide_atoms = Frag.get_rmsd_atoms(frag_filepaths, Frag.get_guide_atoms_biopdb)
#         print(module, 'Got Cluster', os.path.basename(cluster_dir), 'Guide Atoms')
#
#         # All to All Cluster Fragment Guide Atom RMSD (no redundant calculations) Processes Argument
#         if frag_db_biopdb_guide_atoms != list():
#             outfile_prefix = cluster_dir.split("/")[-1]
#             outfile_path = cluster_dir + "/" + outfile_prefix + "_all_to_all_guide_atom_rmsd.txt"
#             outfile = open(outfile_path, "w")
#             for i in range(len(frag_db_biopdb_guide_atoms) - 1):
#                 for j in range(i + 1, len(frag_db_biopdb_guide_atoms)):
#                     result = guide_atom_rmsd((frag_db_biopdb_guide_atoms[i], frag_db_biopdb_guide_atoms[j]))
#                     outfile.write("%s %s %s\n" % (result[0], result[1], str(result[2])))
#             outfile.close()
#
#     print(module, 'Finished')


if __name__ == '__main__':
    rmsd_thresh = 1
    frag_db_dir = os.path.join(os.getcwd(), 'ijk_clusters', 'db_' + str(rmsd_thresh))
    threads = 6
    Frag5.main(module, frag_db_dir, threads)
