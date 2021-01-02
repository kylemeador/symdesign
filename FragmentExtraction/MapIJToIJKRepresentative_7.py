import os
import shutil
from itertools import repeat
from Bio.PDB import PDBParser
import FragUtils as Frag

# Globals
module = 'Map Fragment Pairs to Representative:'


def ijk_map(j_dir, ijk_cluster_rep_guide_atom_dict, ijk_db_dir, rmsd_thresh):
    ij_outliers = []
    i_cluster = os.path.basename(j_dir).split('_')[0]
    j_cluster = os.path.basename(j_dir).split('_')[1]
    outliers = os.path.join(ijk_db_dir, i_cluster, i_cluster + j_cluster, 'outliers.log')
    print('%s Finding Matching K Cluster for Fragments in Cluster %s_%s' % (module, i_cluster, j_cluster))
    for file in os.listdir(j_dir):
        if file.endswith('.pdb'):
            ij_cluster_path = os.path.join(j_dir, file)
            parser = PDBParser()
            ij_cluster_frag_biopdb = parser.get_structure(os.path.splitext(file)[0], ij_cluster_path)
            ij_cluster_frag_guide_atoms = Frag.get_guide_atoms_biopdb(ij_cluster_frag_biopdb)

            min_rmsd = float('inf')
            min_rmsd_key = None
            for ijk_rep_key in ijk_cluster_rep_guide_atom_dict:
                if ijk_rep_key[0] == i_cluster and ijk_rep_key[1] == j_cluster:
                    ijk_cluster_rep_guide_atoms = ijk_cluster_rep_guide_atom_dict[ijk_rep_key]
                    rmsd = Frag.guide_atom_rmsd_biopdb(ij_cluster_frag_guide_atoms, ijk_cluster_rep_guide_atoms)
                    if rmsd < min_rmsd:
                        min_rmsd = rmsd
                        min_rmsd_key = ijk_rep_key

            if min_rmsd <= rmsd_thresh:
                new_i_cluster_path = os.path.join(ijk_db_dir, min_rmsd_key[0])
                new_ij_cluster_path = os.path.join(new_i_cluster_path, min_rmsd_key[0] + '_' + min_rmsd_key[1])
                new_ijk_cluster_path = os.path.join(new_ij_cluster_path,
                                                    min_rmsd_key[0] + '_' + min_rmsd_key[1] + '_' + min_rmsd_key[2])

                if not os.path.exists(new_i_cluster_path):
                    os.makedirs(new_i_cluster_path)
                if not os.path.exists(new_ij_cluster_path):
                    os.makedirs(new_ij_cluster_path)
                if not os.path.exists(new_ijk_cluster_path):
                    os.makedirs(new_ijk_cluster_path)

                new_filename = ij_cluster_frag_biopdb.get_full_id()[0] + '_k_' + min_rmsd_key[2] + '.pdb'
                shutil.copy(os.path.join(j_dir, file), os.path.join(new_ijk_cluster_path, new_filename))
            else:
                ij_outliers.append((os.path.join(ij_mapped_dir, min_rmsd_key[0], min_rmsd_key[0] + min_rmsd_key[1],
                                                 ij_cluster_frag_biopdb.get_full_id()[0]),
                                    outliers[:-4] + '_of_' + min_rmsd_key[2]))

    with open(outliers, 'w') as f:
        for line in ij_outliers:
            f.write(line[0] + ' ' + line[1])

    return 0


def main(ij_mapped_dir, ijk_db_dir, rmsd_thresh, multi=False, num_threads=4):
    print('%s Beginning' % module)

    # Read IJK Cluster Representatives, Create Bio.PDB Objects and Store Guide Atoms in IJK Cluster Dictionary
    ijk_cluster_rep_guide_atom_dict = {}
    for root, dirs, files in os.walk(ijk_db_dir):
        if not dirs:
            for file in files:
                if file.endswith('_representative.pdb'):
                    ijk_rep_path = os.path.join(root, files[0])
                    parser = PDBParser()
                    ijk_rep_biopdb = parser.get_structure(os.path.basename(ijk_rep_path), ijk_rep_path)

                    ijk_name = str(os.path.basename(root))
                    ijk_key = (ijk_name.split('_')[0], ijk_name.split('_')[1], ijk_name.split('_')[2])
                    ijk_cluster_rep_guide_atom_dict[ijk_key] = Frag.get_guide_atoms_biopdb(ijk_rep_biopdb)

    print('%s All Guide Atoms Retrieved' % module)

    # Get Directory Paths for IJ Clustered Interface Fragment PDB files
    j_directory_paths = Frag.get_all_base_root_paths(ij_mapped_dir)
    if multi:
        zipped_args = zip(j_directory_paths, repeat(ijk_cluster_rep_guide_atom_dict), repeat(ij_mapped_dir),
                          repeat(rmsd_thresh))
        result = Frag.mp_starmap(ijk_map, zipped_args, num_threads)
        # Frag.report_errors(result)
    else:
        for j_dir in j_directory_paths:
            ij_outliers = []
            i_cluster = os.path.basename(j_dir).split('_')[0]
            j_cluster = os.path.basename(j_dir).split('_')[1]
            outliers = os.path.join(ijk_db_dir, i_cluster, i_cluster + j_cluster, 'outliers.log')
            print('%s Finding Matching K Cluster for Fragments in Cluster %s_%s' %(module, i_cluster, j_cluster))
            for file in os.listdir(j_dir):
                if file.endswith('.pdb'):
                    ij_cluster_path = os.path.join(j_dir, file)
                    parser = PDBParser()
                    ij_cluster_frag_biopdb = parser.get_structure(os.path.splitext(file)[0], ij_cluster_path)
                    ij_cluster_frag_guide_atoms = Frag.get_guide_atoms_biopdb(ij_cluster_frag_biopdb)

                    min_rmsd = float('inf')
                    min_rmsd_key = None
                    for ijk_rep_key in ijk_cluster_rep_guide_atom_dict:
                        if ijk_rep_key[0] == i_cluster and ijk_rep_key[1] == j_cluster:
                            ijk_cluster_rep_guide_atoms = ijk_cluster_rep_guide_atom_dict[ijk_rep_key]
                            rmsd = Frag.guide_atom_rmsd_biopdb(ij_cluster_frag_guide_atoms, ijk_cluster_rep_guide_atoms)
                            if rmsd < min_rmsd:
                                min_rmsd = rmsd
                                min_rmsd_key = ijk_rep_key

                    if min_rmsd <= rmsd_thresh:
                        new_i_cluster_path = os.path.join(ijk_db_dir, min_rmsd_key[0])
                        new_ij_cluster_path = os.path.join(new_i_cluster_path, min_rmsd_key[0] + '_' + min_rmsd_key[1])
                        new_ijk_cluster_path = os.path.join(new_ij_cluster_path, min_rmsd_key[0] + '_' + min_rmsd_key[1] + '_' + min_rmsd_key[2])

                        if not os.path.exists(new_i_cluster_path):
                            os.makedirs(new_i_cluster_path)
                        if not os.path.exists(new_ij_cluster_path):
                            os.makedirs(new_ij_cluster_path)
                        if not os.path.exists(new_ijk_cluster_path):
                            os.makedirs(new_ijk_cluster_path)

                        new_filename = ij_cluster_frag_biopdb.get_full_id()[0] + '_k_' + min_rmsd_key[2] + '.pdb'
                        shutil.copy(os.path.join(j_dir, file), os.path.join(new_ijk_cluster_path, new_filename))
                    else:
                        ij_outliers.append((os.path.join(ij_mapped_dir, min_rmsd_key[0], min_rmsd_key[0] + min_rmsd_key[1],
                                                         ij_cluster_frag_biopdb.get_full_id()[0]),
                                            outliers[:-4] + '_of_' + min_rmsd_key[2]))
            with open(outliers, 'w') as f:
                for line in ij_outliers:
                    f.write(line[0] + ' ' + line[1])

    print('%s Finished' % module)


if __name__ == '__main__':
    # Match Guide Atom RMSD Threshold
    ijk_rmsd_thresh = 1

    # IJ Clustered Interface Fragment Directory
    ij_clustered_fragdir = os.path.join(os.getcwd(), 'ij_mapped_paired_frags')
    # IJK Cluster Directory & Database Directory
    outdir = os.path.join(os.getcwd(), 'ijk_clusters')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ijk_db = os.path.join(outdir, 'db_' + str(ijk_rmsd_thresh))
    if not os.path.exists(ijk_db):
        os.makedirs(ijk_db)

    main(ij_clustered_fragdir, ijk_db, ijk_rmsd_thresh)
