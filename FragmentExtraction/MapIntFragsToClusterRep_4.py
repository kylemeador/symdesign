"""Map Paired Fragments to IJ Cluster and Add Guide Atoms"""
import os
from copy import deepcopy
from itertools import repeat

from Bio.PDB import PDBParser, Superimposer, PDBIO
from FragUtils import get_biopdb_ca, add_guide_atoms
from SymDesignUtils import get_all_file_paths, mp_starmap, start_log

# Globals
logger = start_log(name=__name__)


def ij_sort(int_frag_path, cent_split_frag_rep_biopdb, i_cluster_dict, outdir, clust_rmsd_thresh):
    try:
        parser = PDBParser()
        frag_biopdb = parser.get_structure(os.path.splitext(os.path.basename(int_frag_path))[0], int_frag_path)
        int_frag_chains_info = []
        # Bio.PDB 1.76 retrieves chains in the order in which they are found in the .pdb file. This will iterate
        # through the first chain then the second chain. Therefore, the fragments that are mapped then output
        # will be centered on the I fragment of the first chain
        for chain in frag_biopdb[0]:
            min_rms = float('inf')
            min_rms_biopdb = None
            min_sup = None

            # Getting CA Atoms From Single Fragment of Interface
            chain_frag_ca = get_biopdb_ca(chain)

            # Find the Cluster/Fragment Type that Interface Fragment Belongs to. If any...
            for cent_i_frag_biopdb in cent_split_frag_rep_biopdb:
                # Getting CA Atoms From Centered Cluster Representative Fragment
                cent_i_frag_rep_ca = get_biopdb_ca(cent_i_frag_biopdb)

                # Comparing Interface Fragment Chain to Cluster Representatives
                sup = Superimposer()
                sup.set_atoms(cent_i_frag_rep_ca, chain_frag_ca)

                if sup.rms < min_rms and sup.rms < clust_rmsd_thresh:
                    min_rms = sup.rms
                    min_rms_biopdb = cent_i_frag_biopdb
                    min_sup = sup

            if not min_rms_biopdb:
                break
            # Getting Cluster Representative Fragment Type
            frag_type = i_cluster_dict[min_rms_biopdb.id]

            # Aligning Interface Fragment Chain to Cluster Representative
            int_frag_biopdb_copy = deepcopy(frag_biopdb)
            min_sup.apply(int_frag_biopdb_copy)
            int_frag_chains_info.append((int_frag_biopdb_copy, chain.id, frag_type, deepcopy(min_rms_biopdb)))
            # below the indices are     :                 [0]       [1]        [2]                      [3]
            # with the first index referring to the first chain and the second index referring to the second

        if len(int_frag_chains_info) == 2:
            chain1_alignment, chain2_alignment = int_frag_chains_info
            subtype_dir_1 = os.path.join(outdir, chain1_alignment[2],
                                         '%s_%s' % (chain1_alignment[2], chain2_alignment[2]))
            if not os.path.exists(subtype_dir_1):
                os.makedirs(subtype_dir_1)

            # superimpose CA atoms from fragment, second chain_id, with second fragment min_rms_biopdb
            # this way we can map the guide coordinates to the second chain, fragment using a transformation
            sup_1 = Superimposer()
            sup_1.set_atoms(get_biopdb_ca(chain1_alignment[0][0][chain2_alignment[1]]),
                            get_biopdb_ca(chain2_alignment[3]))
            add_guide_atoms(chain2_alignment[3])
            sup_1.apply(chain2_alignment[3])
            guide_moved_atom_chain_1 = chain2_alignment[3][0]['9']
            chain1_alignment[0][0].add(guide_moved_atom_chain_1)

            io_1 = PDBIO()
            io_1.set_structure(chain1_alignment[0])
            io_1_path = os.path.join(subtype_dir_1, '%s_mapch_%s_i%s_pairch_%s_j%s.pdb'
                                     % (chain1_alignment[0].id, chain1_alignment[1], chain1_alignment[2],
                                        chain2_alignment[1], chain2_alignment[2]))
            io_1.save(io_1_path)
            # Also save the inverse orientation
            subtype_dir_2 = os.path.join(outdir, chain2_alignment[2],
                                         '%s_%s' % (chain2_alignment[2], chain1_alignment[2]))
            if not os.path.exists(subtype_dir_2):
                os.makedirs(subtype_dir_2)

            # superimpose CA atoms from fragment, first chain_id, with first fragment min_rms_biopdb
            # this way we can map the guide coordinates to the first chain, fragment using a transformation
            sup_2 = Superimposer()
            sup_2.set_atoms(get_biopdb_ca(chain2_alignment[0][0][chain1_alignment[1]]),
                            get_biopdb_ca(chain1_alignment[3]))
            add_guide_atoms(chain1_alignment[3])
            sup_2.apply(chain1_alignment[3])
            guide_moved_atom_chain_2 = chain1_alignment[3][0]['9']
            chain2_alignment[0][0].add(guide_moved_atom_chain_2)

            io_2 = PDBIO()
            io_2.set_structure(chain2_alignment[0])
            io_2_path = os.path.join(subtype_dir_2, chain2_alignment[0].id + '_mapch_' +
                                     chain2_alignment[1] + '_i' + chain2_alignment[2] +
                                     '_pairch_' + chain1_alignment[1] + '_j' +
                                     chain1_alignment[2] + '.pdb')
            io_2.save(io_2_path)

    except Exception as ex:
        logger.error('Failed to map %s to a cluster representative\nException: %s'
                     % (os.path.splitext(os.path.basename(int_frag_path))[0], str(ex)))


def main(cent_i_frag, paired_frag_dir, outdir, i_frag_limit, clust_rmsd_thresh, multi=False, num_threads=4):
    logger.info('Beginning')

    # Load Centered Cluster Representative Fragments, Create Bio.PDB Objects and Cluster Directories in Out Directory
    cent_split_frag_rep_biopdb = []
    i_cluster_dict = {}
    while len(cent_split_frag_rep_biopdb) <= i_frag_limit:
        for root1, dirs1, files1 in os.walk(cent_i_frag):
            if not dirs1:
                if int(os.path.basename(root1)) <= i_frag_limit:
                    for file1 in files1:
                        if file1.endswith('_i_representative.pdb'):
                            parser = PDBParser()
                            i_frag_biopdb = parser.get_structure(os.path.splitext(file1)[0], os.path.join(root1, file1))
                            cent_split_frag_rep_biopdb.append(i_frag_biopdb)
                            i_cluster_dict[os.path.splitext(file1)[0]] = os.path.basename(root1)

                            if not os.path.exists(os.path.join(outdir, os.path.basename(root1))):
                                os.makedirs(os.path.join(outdir, os.path.basename(root1)))
    logger.info('Finished Setting Up I Directories, Fetched I Fragments')

    # Get Paths for Paired Interface Fragment PDB files
    paired_frag_paths = get_all_file_paths(paired_frag_dir, extension='.pdb')

    logger.info('Mapping Pairs to I Fragments, Setting Up Guide Atoms, and Making J Directories. This may take some '
                'time...')
    if multi:
        zipped_fragment_paths = zip(paired_frag_paths, repeat(cent_split_frag_rep_biopdb), repeat(i_cluster_dict),
                                    repeat(outdir), repeat(clust_rmsd_thresh))
        result = mp_starmap(ij_sort, zipped_fragment_paths, num_threads)
        # Frag.report_errors(result)
    else:
        for int_frag_path in paired_frag_paths:
            ij_sort(int_frag_path, cent_split_frag_rep_biopdb, i_cluster_dict, outdir, clust_rmsd_thresh)

    logger.info('Finished')


if __name__ == '__main__':
    # Match RMSD Threshold
    i_clust_rmsd_thresh = 0.75
    number_fragment_types = 5
    thread_count = 8

    # Centered Split Fragment Representatives Directory
    i_clusters_outdir = os.path.join(os.getcwd(), 'i_clusters')

    # Paired Fragment Directory
    pair_frag_outdir = os.path.join(os.getcwd(), 'all_paired_frags')

    # Out Directory - Where one Chain is Mapped onto a Corresponding Centered I Fragment Representative. J directory is
    # the Corresponding Partner Fragment I Representative
    ij_clustered_fragdir = os.path.join(os.getcwd(), 'ij_mapped_paired_frags')
    if not os.path.exists(ij_clustered_fragdir):
        os.makedirs(ij_clustered_fragdir)

    main(i_clusters_outdir, pair_frag_outdir, ij_clustered_fragdir, number_fragment_types, i_clust_rmsd_thresh,
         num_threads=thread_count)
