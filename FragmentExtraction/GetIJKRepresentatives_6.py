import os
import shutil
from itertools import repeat
import FragUtils as Frag

# Globals
module = 'Get IJK Cluster Representatives:'


def ijk_reps(root, db_outdir, minimum_clust_size):
    print('%s Starting on' % os.path.basename(root))
    rmsd_file_path = os.path.join(root, 'all_to_all_guide_atom_rmsd.txt')
    return_clusters = Frag.cluster_fragment_rmsds(rmsd_file_path)
    print('%s \'%s\' Clustering Finished, Creating Representatives...' % (module, root))

    i_dir = os.path.basename(root).split('_')[0]
    ij_dir = os.path.basename(root)
    ij_outdir = os.path.join(db_outdir, i_dir, ij_dir)

    if not os.path.exists(os.path.join(db_outdir, i_dir)):
        os.makedirs(os.path.join(db_outdir, i_dir))
    if not os.path.exists(ij_outdir):
        os.makedirs(ij_outdir)

    # Get Cluster Representatives
    cluster_count = 1
    for cluster in return_clusters:
        if len(cluster[1]) >= minimum_clust_size:
            cluster_rep = cluster[0] + '.pdb'
            cluster_rep_pdb_path = os.path.join(root, cluster_rep)

            k_cluster_dirname = os.path.basename(ij_outdir) + '_' + str(cluster_count)
            cluster_outdir = os.path.join(ij_outdir, k_cluster_dirname)
            if not os.path.exists(cluster_outdir):
                os.makedirs(cluster_outdir)

            cluster_rep_name = cluster[0] + '_k_' + str(cluster_count) + '_representative.pdb'
            shutil.copyfile(cluster_rep_pdb_path, os.path.join(cluster_outdir, cluster_rep_name))
            cluster_count += 1

    return 0


def main(ij_mapped_dir, db_outdir, minimum_clust_size, multi=False, num_threads=4):
    print('%s Beginning' % module)

    # Cluster
    rmsd_paths = []
    for root, dirs, files in os.walk(ij_mapped_dir):
        if not dirs:
            rmsd_paths.append(root)
    if multi:
        zipped_args = zip(rmsd_paths, repeat(db_outdir), repeat(minimum_clust_size))
        result = Frag.mp_starmap(ijk_reps, zipped_args, num_threads)
        # Frag.report_errors(result)
    else:
        for path in rmsd_paths:
            rmsd_file_path = os.path.join(path, 'all_to_all_guide_atom_rmsd.txt')
            return_clusters = Frag.cluster_fragment_rmsds(rmsd_file_path)
            print('%s \'%s\' Clustering Finished, Creating Representatives...' % (module, path))

            i_dir = os.path.basename(path).split('_')[0]
            ij_dir = os.path.basename(path)
            ij_outdir = os.path.join(db_outdir, i_dir, ij_dir)

            if not os.path.exists(os.path.join(db_outdir, i_dir)):
                os.makedirs(os.path.join(db_outdir, i_dir))
            if not os.path.exists(ij_outdir):
                os.makedirs(ij_outdir)

            # Get Cluster Representatives
            cluster_count = 1
            for cluster in return_clusters:
                if len(cluster[1]) >= minimum_clust_size:
                    cluster_rep = cluster[0] + '.pdb'
                    cluster_rep_pdb_path = os.path.join(path, cluster_rep)

                    k_cluster_dirname = os.path.basename(ij_outdir) + '_' + str(cluster_count)
                    cluster_outdir = os.path.join(ij_outdir, k_cluster_dirname)
                    if not os.path.exists(cluster_outdir):
                        os.makedirs(cluster_outdir)

                    cluster_rep_name = cluster[0] + '_k' + str(cluster_count) + '_representative.pdb'
                    shutil.copyfile(cluster_rep_pdb_path, os.path.join(cluster_outdir, cluster_rep_name))
                    cluster_count += 1

    print('%s Finished' % module)


if __name__ == '__main__':
    minimum_ijk_cluster_size = 4
    ijk_rmsd_thresh = 1

    # IJ Clustered Directory
    ij_clustered_fragdir = os.path.join(os.getcwd(), 'ij_mapped_paired_frags')

    # Output Directories
    outdir = os.path.join(os.getcwd(), 'ijk_clusters')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ijk_db = os.path.join(outdir, 'db_' + str(ijk_rmsd_thresh))
    if not os.path.exists(ijk_db):
        os.makedirs(ijk_db)

    main(ij_clustered_fragdir, ijk_db, minimum_ijk_cluster_size)  # ijk_rmsd_thresh)
