import sys
import os
from itertools import combinations, repeat
from random import shuffle

import FragUtils as Frag
from SymDesignUtils import get_all_file_paths

# Globals
module = 'Individual Fragment All to All RMSD:'


def main(directory, num_threads, cluster_size_limit, rmsd_thresh, rmsd_source=Frag.get_biopdb_ca):
    print('%s Beginning' % module)

    all_paths = get_all_file_paths(directory, extension='.pdb')
    if len(all_paths) > cluster_size_limit:
        print('%s Number of Fragment: %d > %d (size limit). Sampled %d times instead.' % (module, len(all_paths),
                                                                                          cluster_size_limit,
                                                                                          cluster_size_limit))
        shuffle(all_paths)
        all_paths = all_paths[0:cluster_size_limit]
    else:
        print('%s Number of Fragments: %d' % (module, len(all_paths)))

    all_atoms = Frag.get_rmsd_atoms(all_paths, rmsd_source)
    print('%s Got Guide Atoms from \'%s\'' % (module, os.path.basename(directory)))

    if all_atoms != list():
        args = combinations(all_atoms, 2)
        zipped_args = zip(args, repeat(rmsd_thresh))
    else:
        print('%s Couldn\'t find any atoms from %s \nEnsure proper input!!' % (module, directory))
        sys.exit()

    print('%s Processing RMSD calculation on %d cores. This may take awhile...' % (module, num_threads))
    results = Frag.mp_starmap(Frag.mp_guide_atom_rmsd, zipped_args, num_threads)
    # mp_result = Frag.mp_function(Frag.superimpose, arg, num_threads, thresh=rmsd_thresh)

    outfile = os.path.join(directory, 'all_to_all_guide_atom_rmsd.txt')
    with open(outfile, 'w') as outfile:
        for result in results:
            if result is not None:
                outfile.write('%s %s %s\n' % (result[0], result[1], result[2]))

    # print('%s Finished' % module)


if __name__ == '__main__':
    indv_frag_outdir = os.path.join(os.getcwd(), 'all_individual_frags')
    # RMSD Threshold to Match I Clusters
    i_clust_rmsd_thresh = 0.75
    thread_count = 4
    size_limit = 20000

    main(indv_frag_outdir, thread_count, size_limit, i_clust_rmsd_thresh)
