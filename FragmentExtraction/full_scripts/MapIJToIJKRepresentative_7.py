import sys
import os
import shutil
import math
from Bio.PDB import PDBParser
import FragUtils as Frag

# Globals
module = 'Map Fragment Pairs to Representative:'


# def guide_atom_rmsd_biopdb(biopdb_1_guide_atoms, biopdb_2_guide_atoms):
#     # Calculate RMSD
#     e1 = (biopdb_1_guide_atoms[0] - biopdb_2_guide_atoms[0]) ** 2
#     e2 = (biopdb_1_guide_atoms[1] - biopdb_2_guide_atoms[1]) ** 2
#     e3 = (biopdb_1_guide_atoms[2] - biopdb_2_guide_atoms[2]) ** 2
#     s = e1 + e2 + e3
#     m = s / float(3)
#     r = math.sqrt(m)
#
#     return r
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
# def get_all_base_root_paths(directory):
#     dir_paths = []
#     for root, dirs, files in os.walk(directory):
#         if not dirs:
#             dir_paths.append(root)
#
#     return dir_paths


def main():
    print(module, 'Beginning')
    # Match Guide Atom RMSD Threshold
    rmsd_thresh = 1

    # IJ Clustered Interface Fragment Directory
    ij_mapped_dir = os.path.join(os.getcwd(), 'ij_mapped_paired_frags')
    # IJK Cluster Directory & Database Directory
    outdir = os.path.join(os.getcwd(), 'ijk_clusters')
    ijk_db_dir = os.path.join(outdir, 'db_' + str(rmsd_thresh))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(ijk_db_dir):
        os.makedirs(db_outdir)

    # Read IJK Cluster Representatives, Create Bio.PDB Objects and Store Guide Atoms in IJK Cluster Dictionary
    ijk_cluster_rep_guide_atom_dict = {}
    for root, dirs, files in os.walk(ijk_db_dir):
        if not dirs:
            for file in files:
                if file.endswith('_representative.pdb'):
                    ijk_rep_path = os.path.join(root, files[0])
                    parser = PDBParser()
                    ijk_rep_biopdb = parser.get_structure(os.path.basename(ijk_rep_path), ijk_rep_path)

                    ijk_name = os.path.basename(root)
                    ijk_key = (ijk_name.split('_')[0], ijk_name.split('_')[1], ijk_name.split('_')[2])
                    ijk_cluster_rep_guide_atom_dict[ijk_key] = Frag.get_guide_atoms_biopdb(ijk_rep_biopdb)

    print(module, 'All Guide Atoms Retrieved')

    # Get Directory Paths for IJ Clustered Interface Fragment PDB files
    j_directory_paths = Frag.get_all_base_root_paths(ij_mapped_dir)

    for j_dir in j_directory_paths:
        i_cluster = os.path.basename(j_dir).split('_')[0]
        j_cluster = os.path.basename(j_dir).split('_')[1]
        print(module, 'Finding Matching K Cluster for Fragments in Cluster', i_cluster, j_cluster)
        for file in os.listdir(j_dir):
            if file.endswith('.pdb'):
                ij_cluster_path = os.path.join(j_dir, file)
                parser = PDBParser()
                ij_cluster_frag_biopdb = parser.get_structure(file, ij_cluster_path)
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

                    new_filename = file + '_orientation_' + min_rmsd_key[2] + '.pdb'
                    shutil.copy(os.path.join(j_dir, file), os.path.join(new_ijk_cluster_path, new_filename))
    print(module, 'Finished')


if __name__ == '__main__':
    main()
