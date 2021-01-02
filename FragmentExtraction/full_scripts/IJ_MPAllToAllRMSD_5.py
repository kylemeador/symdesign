import sys
import os
import math
import multiprocessing as mp
from itertools import combinations
from Bio.PDB import PDBParser
from random import shuffle
import FragUtils as Frag

# Globals
module = 'Limited Random All to All RMSD Guide Atoms:'


# def guide_atom_rmsd(biopdb_guide_atoms_tup):
#     rmsd_thresh = 1.0
#
#     biopdb_1_guide_atoms = biopdb_guide_atoms_tup[0]
#     biopdb_2_guide_atoms = biopdb_guide_atoms_tup[1]
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
#     if r <= rmsd_thresh:
#         return biopdb_1_id, biopdb_2_id, r
#     else:
#         return None
#
#
# def get_guide_atoms_biopdb(biopdb_structure):
#     guide_atoms = []
#     for atom in biopdb_structure.get_atoms():
#         if atom.get_full_id()[2] == '9':
#             guide_atoms.append(atom)
#     if len(guide_atoms) == 3:
#         return guide_atoms
#     else:
#         return None
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
# def get_rmsd_atoms(filepaths, function):
#     all_rmsd_atoms = []
#     for filepath in filepaths:
#         pdb_name = os.path.splitext(os.path.basename(filepath))[0]
#         parser = PDBParser()
#         pdb = parser.get_structure(pdb_name, filepath)
#         all_rmsd_atoms.append(function(pdb))
#
#     return all_rmsd_atoms
#
#
# def mp_function(function, processes_arg, threads):
#     p = mp.Pool(processes=threads)
#     results = p.map(function, processes_arg)
#     p.close()
#     p.join()
#
#     return results


def main(mod, directory, thread_count, cluster_size_limit=20000):
    print(mod, 'Beginning')

    # Cluster Directory Paths
    cluster_dir_paths = Frag.get_all_base_root_paths(directory)
    for cluster_dir in cluster_dir_paths:
        print(mod, 'Starting', cluster_dir)
        frag_filepaths = Frag.get_all_pdb_file_paths(cluster_dir)
        if len(frag_filepaths) > cluster_size_limit:
            print(mod, 'Number of Fragment:', len(frag_filepaths), '>', cluster_size_limit,
                  '(cluster size limit). Sampled', cluster_size_limit, 'times instead.')
            shuffle(frag_filepaths)
            frag_filepaths = frag_filepaths[0:cluster_size_limit]
        else:
            print(mod, 'Number of Fragments:', len(frag_filepaths))

        # Get Fragment Guide Atoms for all Fragments in Cluster
        frag_db_guide_atoms = Frag.get_rmsd_atoms(frag_filepaths, Frag.get_guide_atoms_biopdb)
        print(mod, 'Got Cluster', os.path.basename(cluster_dir), 'Guide Atoms')

        if frag_db_guide_atoms != list():
            args = combinations(frag_db_guide_atoms, 2)
        else:
            print(mod, 'Couldn\'t find any atoms from', cluster_dir, '\nEnsure proper input!!')
            break

        print(mod, 'Processing RMSD calculation on', thread_count, 'cores. This may take awhile...')
        results = Frag.mp_function(Frag.mp_guide_atom_rmsd, args, thread_count)

        outfile_path = os.path.join(cluster_dir, 'all_to_all_guide_atom_rmsd.txt')
        with open(outfile_path, 'w') as outfile:
            for result in results:
                if result is not None:
                    outfile.write('%s %s %s\n' % (result[0], result[1], result[2]))

    print(mod, 'Finished')


if __name__ == '__main__':
    frag_db_dir = os.path.join(os.getcwd(), 'ij_mapped_paired_frags')
    _mod = module
    threads = 6
    main(_mod, frag_db_dir, threads)
