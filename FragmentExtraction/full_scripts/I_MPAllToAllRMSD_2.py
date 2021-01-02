import sys
import os
import math
import multiprocessing as mp
from itertools import combinations
from Bio.PDB import PDBParser
from random import shuffle
import FragUtils as Frag

# Globals
module = 'Individual Fragment All to All RMSD:'


# def get_biopdb_ca(biopdb_structure):
#     ca_atoms = []
#     for atom in biopdb_structure.get_atoms():
#         if atom.get_id() == 'CA':
#             ca_atoms.append(atom)
#     if len(ca_atoms) == 5:
#         return ca_atoms
#     else:
#         return None
#
#
# def superimpose(atoms):
#     rmsd_thresh = 0.75
#     biopdb_1_id = atoms1[0].get_full_id()[0]
#     biopdb_2_id = atoms2[0].get_full_id()[0]
#
#     sup = Superimposer()
#     sup.set_atoms(atoms[0], atoms[1])
#     if sup.rms <= rmsd_thresh:
#         return biopdb_1_id, biopdb_2_id, sup.rms
#     else:
#         return None
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


if __name__ == '__main__':
    print(module, 'Beginning')
    frag_db_dir = os.path.join(os.getcwd(), 'all_individual_frags')
    outfile_path = os.path.join(frag_db_dir, 'all_to_all_rmsd.txt')

    thread_count = 6
    cluster_size_limit = 20000

    all_paths = Frag.get_all_pdb_file_paths(frag_db_dir)
    if len(all_paths) > cluster_size_limit:
        print(module, 'Cluster number:', len(all_paths), '>', cluster_size_limit, ' cluster size limit. Sampled',
              cluster_size_limit, 'times instead')
        shuffle(all_paths)
        all_paths = all_paths[0:cluster_size_limit]
    else:
        print(module, 'Cluster number:', len(all_paths))

    all_atoms = Frag.get_rmsd_atoms(all_paths, Frag.get_biopdb_ca)

    if all_atoms != list():
        arg = args = combinations(all_atoms, 2)
    else:
        print(module, 'Couldn\'t find any atoms from fragments. Ensure proper input')
        sys.exit()

    print(module, 'Processing RMSD calculation on', thread_count, 'cores. This may take awhile...')
    mp_result = Frag.mp_function(Frag.superimpose, arg, thread_count)

    with open(outfile_path, 'w') as outfile:
        for result in mp_result:
            if result is not None:
                outfile.write('%s %s %s\n' % (result[0], result[1], result[2]))

    print(module, 'Finished')
