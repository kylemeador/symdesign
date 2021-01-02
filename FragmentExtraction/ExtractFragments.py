import sys
import os
import argparse
import ExtractIntFrag_1 as ExFrag1
import I_MPAllToAllRMSD_2 as ExFrag2
import ClusterAllToAllRMSDAndCenter_3 as ExFrag3
import MapIntFragsToClusterRep_4 as ExFrag4
import IJ_MPAllToAllRMSD_5 as ExFrag5
import GetIJKRepresentatives_6 as ExFrag6
import MapIJToIJKRepresentative_7 as ExFrag7
import MakeIJKClusterStats_8 as ExFrag8
import FindIJKWeightStatistics_9 as ExFrag9


def main(fragment_length, interface_distance, i_clust_rmsd_thresh, size_limit, thread_count, number_fragment_types,
         ijk_rmsd_thresh, minimum_ijk_cluster_size, side_chain_contact_dist, skip, intra_chain, multi_processing):  #,
         # file_list):
    # Check if any modules should be skipped
    module = {str(i): True for i in range(1, 10)}
    if skip:
        skips = skip.split(',')
        for mod in skips:
            mod_skip = mod.strip()
            module[mod_skip] = False
        print('Modules', ', '.join([k for k in module if not module[k]]), 'will be skipped.')

    # if file_list:
    #     pdbs_of_interest = Frag.pdb_codes_to_iterable(file_list)
    # else:
    #     pdbs_of_interest = []

    # 1 Extract all Individual and Paired Interface Fragments
    # Natural Interface PDB File Paths
    clean_pdb_dir = os.path.join(os.getcwd(), 'all_clean_pdbs')
    # Individual Fragment Directory
    indv_frag_outdir = os.path.join(os.getcwd(), 'all_individual_frags')
    if not os.path.exists(indv_frag_outdir):
        os.makedirs(indv_frag_outdir)
    # Paired Fragment Directory
    pair_frag_outdir = os.path.join(os.getcwd(), 'all_paired_frags')
    if not os.path.exists(pair_frag_outdir):
        os.makedirs(pair_frag_outdir)

    # fragment_length = 5
    # interface_distance = 8
    if module['1']:
        ExFrag1.main(clean_pdb_dir, indv_frag_outdir, pair_frag_outdir, fragment_length, interface_distance, intra_chain
                     , multi_processing, thread_count)

    # 2 Process All to All RMSD on Individual Fragments
    i_rmsd_outfile_path = os.path.join(indv_frag_outdir, 'all_to_all_rmsd.txt')
    # RMSD Threshold to Match I Clusters
    # i_clust_rmsd_thresh = 0.75
    # thread_count = 6
    # size_limit = 20000

    if module['2']:
        ExFrag2.main(indv_frag_outdir, i_rmsd_outfile_path, thread_count, size_limit, i_clust_rmsd_thresh)

    # 3 Cluster, Center, and Sort By Occurrence I Fragments. These are I Fragment Representatives
    # Centered I Clusters Directory
    i_clusters_outdir = os.path.join(os.getcwd(), 'i_clusters')
    if not os.path.exists(i_clusters_outdir):
        os.makedirs(i_clusters_outdir)

    if module['3']:
        ExFrag3.main(indv_frag_outdir, i_clusters_outdir, i_rmsd_outfile_path)

    # 4 Map Both Fragments of Paired Interface Fragments to Centered I Fragment Representatives. Create Guide Atoms
    # number_fragment_types = 5

    # IJ Clustered Interface Fragment Directory.
    # First Chain Mapped onto Centered I Fragment. Second Chain Also Mapped to I Fragment Called J Fragment
    ij_clustered_fragdir = os.path.join(os.getcwd(), 'ij_mapped_paired_frags')
    if not os.path.exists(ij_clustered_fragdir):
        os.makedirs(ij_clustered_fragdir)

    if module['4']:
        ExFrag4.main(i_clusters_outdir, pair_frag_outdir, ij_clustered_fragdir, number_fragment_types,
                     i_clust_rmsd_thresh, multi_processing, thread_count)

    # 5 Process All to All RMSD on Paired Fragment Guide Atoms
    if module['5']:
        ExFrag5.main(ij_clustered_fragdir, thread_count, size_limit, ijk_rmsd_thresh)

    # IJK Cluster Directory, Database, and Info Directory
    ijk_outdir = os.path.join(os.getcwd(), 'ijk_clusters')
    if not os.path.exists(ijk_outdir):
        os.makedirs(ijk_outdir)
    ijk_db = os.path.join(ijk_outdir, 'db_' + str(ijk_rmsd_thresh))
    if not os.path.exists(ijk_db):
        os.makedirs(ijk_db)

    # 6 Cluster and Sort By Occurrence IJK Fragments. These are IJK Fragment Representatives
    # Match Guide Atom RMSD Threshold
    # ijk_rmsd_thresh = 1
    # minimum_ijk_cluster_size = 4

    if module['6']:
        ExFrag6.main(ij_clustered_fragdir, ijk_db, minimum_ijk_cluster_size, multi_processing, thread_count)

    # 7 Map IJ Fragments onto IJK Representative
    if module['7']:
        ExFrag7.main(ij_clustered_fragdir, ijk_db, ijk_rmsd_thresh, multi_processing, thread_count)

    # 8 Get IJK Cluster Statistics including Frequency and Residue Contact Weight
    # side_chain_contact_dist = 5
    info_db = os.path.join(ijk_outdir, 'info_' + str(ijk_rmsd_thresh))
    if not os.path.exists(info_db):
        os.makedirs(info_db)

    if module['8']:
        ExFrag8.main(ijk_db, info_db, fragment_length, side_chain_contact_dist, multi_processing, thread_count)

    # 9 Get I, IJ, Index Statistics
    if module['9']:
        ExFrag9.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract peptide fragments from a specific PDB Library\n '
                                                 '1 Extract all Individual and Paired Interface Fragments\n '
                                                 '2 Process All to All RMSD on Individual Fragments\n '
                                                 '3 Cluster, Center, and Sort By Occurrence I Fragments. '
                                                 'These are I Fragment Representatives\n '
                                                 '4 Map Both Fragments of Paired Interface Fragments to Centered I '
                                                 'Fragment Representatives. Create Guide Atoms\n '
                                                 '5 Process All to All RMSD on Paired Fragment Guide Atoms\n '
                                                 '6 Cluster and Sort By Occurrence IJK Fragments. These are '
                                                 'IJK Fragment Representatives\n '
                                                 '7 Map IJ Fragments onto IJK Representative\n '
                                                 '8 Get IJK Cluster Statistics including Frequency and Residue Contact '
                                                 'Weight\n '
                                                 '9 Get I, IJ, Index Statistics\n')
    # 12 possible variables
    parser.add_argument('-l', '--fragment_length', type=int, help='The length of fragments to extract', default=5)
    parser.add_argument('-d', '--interface_distance', type=int, help='The distance between CB Atoms for extraction '
                                                                     '(Angstroms)', default=8)
    parser.add_argument('-i', '--i_clust_rmsd_thresh', type=int, help='RMSD Threshold for single fragments',
                        default=0.75)
    parser.add_argument('-s', '--size_limit', type=int, help='The size limit for All to All Calculations',
                        default=20000)
    parser.add_argument('-t', '--thread_count', type=int, help='The number of computer cores to utilize during '
                                                               'Multi-Processing', default=4)
    parser.add_argument('-n', '--number_fragment_types', type=int, help='Number of Fragment Types to Cluster',
                        default=5)
    parser.add_argument('-k', '--ijk_rmsd_thresh', type=int, help='RMSD Threshold for final paired fragments',
                        default=1)
    parser.add_argument('-m', '--minimum_ijk_cluster_size', type=int, help='RMSD Threshold for final paired fragments',
                        default=4)
    parser.add_argument('-c', '--side_chain_contact_dist', type=int, help='Distance between side chain atoms for '
                                                                          'weighting (Angstroms)', default=5)
    parser.add_argument('-p', '--pass_modules', type=str, help='Skip any steps?', default=None)
    parser.add_argument('-a', '--intra', type=bool, help='Process Intrachain Fragments?', default=False)
    parser.add_argument('-z', '--multi', type=bool, help='Use Multiprocessing?', default=False)
    # parser.add_argument('-f', '--file', type=str, help='Filter PDB\'s by defined list?', default=None)

    args = parser.parse_args()
    main(args.fragment_length, args.interface_distance, args.i_clust_rmsd_thresh, args.size_limit, args.thread_count,
         args.number_fragment_types, args.ijk_rmsd_thresh, args.minimum_ijk_cluster_size, args.side_chain_contact_dist,
         args.pass_modules, args.intra, args.multi)  #, args.file)
