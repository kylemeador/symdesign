import os
import copy
from Bio.PDB import PDBParser, Superimposer, PDBIO
import FragUtils as Frag

# Globals
module = 'Map Paired Fragments to IJ Cluster and Add Guide Atoms:'


def ij_sort(int_frag_path, cent_split_frag_rep_biopdb, i_cluster_dict, outdir, clust_rmsd_thresh):
    # for int_frag_path in paired_frag_paths:
    try:
        parser = PDBParser()
        frag_biopdb = parser.get_structure(os.path.splitext(os.path.basename(int_frag_path))[0], int_frag_path)
        int_frag_chains_info = []
        # Bio.PDB 1.76 retrieves chains in the order in which they are found in the .pdb file. This will iterate
        # through the first chain then the second chain. Therefore, the fragments that are mapped then output will
        # be centered on the I fragment of the first chain
        for chain in frag_biopdb[0]:
            min_rms = float('inf')
            min_rms_biopdb = None
            min_sup = None

            # Getting CA Atoms From Single Fragment of Interface
            chain_frag_ca = Frag.get_biopdb_ca(chain)

            # Find the Cluster/Fragment Type that Interface Fragment Belongs to. If any...
            for cent_i_frag_biopdb in cent_split_frag_rep_biopdb:
                # Getting CA Atoms From Centered Cluster Representative Fragment
                cent_i_frag_rep_ca = Frag.get_biopdb_ca(cent_i_frag_biopdb)

                # Comparing Interface Fragment Chain to Cluster Representatives
                sup = Superimposer()
                sup.set_atoms(cent_i_frag_rep_ca, chain_frag_ca)

                if sup.rms < min_rms:
                    min_rms = sup.rms
                    min_rms_biopdb = cent_i_frag_biopdb
                    min_sup = sup

            if min_rms_biopdb is not None and min_rms <= clust_rmsd_thresh:
                # Getting Cluster Representative Fragment Type
                frag_type = i_cluster_dict[min_rms_biopdb.id]

                # Aligning Interface Fragment Chain to Cluster Representative
                int_frag_biopdb_copy = copy.deepcopy(frag_biopdb)
                min_sup.apply(int_frag_biopdb_copy)
                int_frag_chains_info.append((int_frag_biopdb_copy, chain.id, frag_type,
                                             copy.deepcopy(min_rms_biopdb)))
            else:
                break

        if len(int_frag_chains_info) == 2:
            subtype_dir_1 = os.path.join(outdir, int_frag_chains_info[0][2], int_frag_chains_info[0][2] + '_'
                                         + int_frag_chains_info[1][2])
            if not os.path.exists(subtype_dir_1):
                os.makedirs(subtype_dir_1)

            # superimpose CA atoms from fragment, second chain_id, with second fragment min_rms_biopdb
            sup_1 = Superimposer()
            sup_1.set_atoms(Frag.get_biopdb_ca(int_frag_chains_info[0][0][0][int_frag_chains_info[1][1]]),
                            Frag.get_biopdb_ca(int_frag_chains_info[1][3]))
            Frag.add_guide_atoms(int_frag_chains_info[1][3])
            sup_1.apply(int_frag_chains_info[1][3])
            guide_moved_atom_chain_1 = int_frag_chains_info[1][3][0]['9']
            int_frag_chains_info[0][0][0].add(guide_moved_atom_chain_1)

            io_1 = PDBIO()
            io_1.set_structure(int_frag_chains_info[0][0])
            io_1_path = os.path.join(subtype_dir_1, int_frag_chains_info[0][0].id + '_mapch_' +
                                     int_frag_chains_info[0][1] + '_i' + int_frag_chains_info[0][2] +
                                     '_pairch_' + int_frag_chains_info[1][1] + '_j' +
                                     int_frag_chains_info[1][2] + '.pdb')
            io_1.save(io_1_path)

    except Exception as ex:
        print('\n')
        print('Error mapping %s to a cluster representative' % os.path.splitext(os.path.basename(int_frag_path))[0])
        print('Exception: %s' % str(ex))
        print('\n')

    return 0


def main(cent_i_frag, paired_frag_dir, outdir, i_frag_limit, clust_rmsd_thresh, multi=False, num_threads=4):
    print('%s Beginning' % module)

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
    print('%s Finished Setting Up I Directories, Fetched I Fragments' % module)

    # Get Paths for Paired Interface Fragment PDB files
    paired_frag_paths = Frag.get_all_pdb_file_paths(paired_frag_dir)
    # for root2, dirs2, files2 in os.walk(paired_frag_dir):
    #     for file2 in files2:
    #         if file2.endswith('.pdb'):
    #             paired_frag_paths.append(os.path.join(paired_frag_dir, file2))

    print('%s Mapping Pairs to I Fragments, Setting Up Guide Atoms, and Making J Directories. '
          'This may take some time...' % module)
    if multi:
        zipped_fragment_paths = zip(paired_frag_paths, repeat(cent_split_frag_rep_biopdb), repeat(i_cluster_dict),
                                    repeat(outdir), repeat(clust_rmsd_thresh))
        result = Frag.mp_starmap(ij_sort, zipped_fragment_paths, num_threads)
        # Frag.report_errors(result)
    else:
        for int_frag_path in paired_frag_paths:
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
                    chain_frag_ca = Frag.get_biopdb_ca(chain)

                    # Find the Cluster/Fragment Type that Interface Fragment Belongs to. If any...
                    for cent_i_frag_biopdb in cent_split_frag_rep_biopdb:
                        # Getting CA Atoms From Centered Cluster Representative Fragment
                        cent_i_frag_rep_ca = Frag.get_biopdb_ca(cent_i_frag_biopdb)

                        # Comparing Interface Fragment Chain to Cluster Representatives
                        sup = Superimposer()
                        sup.set_atoms(cent_i_frag_rep_ca, chain_frag_ca)

                        if sup.rms < min_rms:
                            min_rms = sup.rms
                            min_rms_biopdb = cent_i_frag_biopdb
                            min_sup = sup

                    if min_rms_biopdb is not None and min_rms <= clust_rmsd_thresh:
                        # Getting Cluster Representative Fragment Type
                        frag_type = i_cluster_dict[min_rms_biopdb.id]

                        # Aligning Interface Fragment Chain to Cluster Representative
                        int_frag_biopdb_copy = copy.deepcopy(frag_biopdb)
                        min_sup.apply(int_frag_biopdb_copy)
                        int_frag_chains_info.append((int_frag_biopdb_copy, chain.id, frag_type,
                                                     copy.deepcopy(min_rms_biopdb)))
                    else:
                        break

                if len(int_frag_chains_info) == 2:
                    subtype_dir_1 = os.path.join(outdir, int_frag_chains_info[0][2], int_frag_chains_info[0][2] + '_'
                                                 + int_frag_chains_info[1][2])
                    if not os.path.exists(subtype_dir_1):
                        os.makedirs(subtype_dir_1)

                    # superimpose CA atoms from fragment, second chain_id, with second fragment min_rms_biopdb
                    sup_1 = Superimposer()
                    sup_1.set_atoms(Frag.get_biopdb_ca(int_frag_chains_info[0][0][0][int_frag_chains_info[1][1]]),
                                    Frag.get_biopdb_ca(int_frag_chains_info[1][3]))
                    Frag.add_guide_atoms(int_frag_chains_info[1][3])
                    sup_1.apply(int_frag_chains_info[1][3])
                    guide_moved_atom_chain_1 = int_frag_chains_info[1][3][0]['9']
                    int_frag_chains_info[0][0][0].add(guide_moved_atom_chain_1)

                    io_1 = PDBIO()
                    io_1.set_structure(int_frag_chains_info[0][0])
                    io_1_path = os.path.join(subtype_dir_1, int_frag_chains_info[0][0].id + '_mapch_' +
                                             int_frag_chains_info[0][1] + '_i' + int_frag_chains_info[0][2] +
                                             '_pairch_' + int_frag_chains_info[1][1] + '_j' +
                                             int_frag_chains_info[1][2] + '.pdb')
                    # io_1_path = os.path.join(subtype_dir_1, int_frag_chains_info[0][0].id + '_mappedchain_' +
                    #                          int_frag_chains_info[0][1] + '_fragtype_' + int_frag_chains_info[0][2] +
                    #                          '_partnerchain_' + int_frag_chains_info[1][1] + '_fragtype_' +
                    #                          int_frag_chains_info[1][2] + '.pdb')
                    io_1.save(io_1_path)

            except Exception as ex:
                print('\n')
                print('Error mapping %s to a cluster representative' % os.path.splitext(os.path.basename(int_frag_path))
                      [0])
                print('Exception: %s' % str(ex))
                print('\n')

    print('%s Finished' % module)


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
