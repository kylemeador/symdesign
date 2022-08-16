import os
from itertools import combinations, repeat
from random import shuffle
import FragUtils as Frag
from I_MPAllToAllRMSD_2 import main as frag2main


# Globals
module = 'Limited Random All to All RMSD Guide Atoms:'


def main(directory, num_threads, cluster_size_limit, rmsd_thresh):
    print('%s Beginning' % module)

    # Cluster Directory Paths
    cluster_dir_paths = Frag.get_all_base_root_paths(directory)
    for cluster_dir in cluster_dir_paths:
        print('%s Starting %s' % (module, cluster_dir))
        frag2main(cluster_dir, num_threads, cluster_size_limit, rmsd_thresh, rmsd_source=Frag.get_guide_atoms_biopdb)

        # frag_filepaths = Frag.get_file_paths_recursively(cluster_dir, extension='.pdb')
        # if len(frag_filepaths) > cluster_size_limit:
        #     print('%s Number of Fragments: %d > %d (size limit). Sampled %d times instead.' % (module,
        #                                                                                        len(frag_filepaths),
        #                                                                                        cluster_size_limit,
        #                                                                                        cluster_size_limit))
        #     shuffle(frag_filepaths)
        #     frag_filepaths = frag_filepaths[0:cluster_size_limit]
        # else:
        #     print('%s Number of Fragments: %d' % (module, len(frag_filepaths)))
        #
        # # Get Fragment Guide Atoms for all Fragments in Cluster
        # frag_db_guide_atoms = Frag.get_rmsd_atoms(frag_filepaths, Frag.get_guide_atoms_biopdb)
        # print(module, 'Got Cluster', os.path.basename(cluster_dir), 'Guide Atoms')
        #
        # if frag_db_guide_atoms != list():
        #     args = combinations(frag_db_guide_atoms, 2)
        #     zipped_args = zip(args, repeat(rmsd_thresh))
        # else:
        #     print('%s Couldn\'t find any atoms from %s \nEnsure proper input!!' % (module, cluster_dir))
        #     break
        #
        # print('%s Processing RMSD calculation on %d cores. This may take awhile...' % (module, num_threads))
        # results = Frag.mp_starmap(Frag.mp_guide_atom_rmsd, zipped_args, num_threads)
        # # results = Frag.mp_function(Frag.mp_guide_atom_rmsd, rmsd_thresh, args, num_threads)
        #
        # outfile_path = os.path.join(cluster_dir, 'all_to_all_guide_atom_rmsd.txt')
        # with open(outfile_path, 'w') as outfile:
        #     for result in results:
        #         if result is not None:
        #             outfile.write('%s %s %s\n' % (result[0], result[1], result[2]))

    print('%s Finished' % module)


if __name__ == '__main__':
    ij_clustered_fragdir = os.path.join(os.getcwd(), 'ij_mapped_paired_frags')
    ijk_rmsd_thresh = 1
    threads = 4
    size_limit = 20000

    main(ij_clustered_fragdir, threads, size_limit, ijk_rmsd_thresh)
