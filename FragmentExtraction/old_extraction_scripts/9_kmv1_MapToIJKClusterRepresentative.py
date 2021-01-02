import sys
import os
import Bio.PDB
import shutil
import math


def guide_atom_rmsd(biopdb_guide_atoms_tup):
    biopdb_1_guide_atoms = biopdb_guide_atoms_tup[0]
    biopdb_2_guide_atoms = biopdb_guide_atoms_tup[1]

    # Calculate RMSD
    e1 = (biopdb_1_guide_atoms[0] - biopdb_2_guide_atoms[0]) ** 2
    e2 = (biopdb_1_guide_atoms[1] - biopdb_2_guide_atoms[1]) ** 2
    e3 = (biopdb_1_guide_atoms[2] - biopdb_2_guide_atoms[2]) ** 2
    s = e1 + e2 + e3
    m = s / float(3)
    r = math.sqrt(m)

    return r


def get_guide_atoms_biopdb(biopdb_structure):
    guide_atoms = []
    for atom in biopdb_structure.get_atoms():
        if atom.get_full_id()[2] == "9":
            guide_atoms.append(atom)
    return guide_atoms


def main():

    # Match Guide Atom RMSD Threshold
    clust_rmsd_thresh = 1.0

    # IJK Cluster Representative Directory - "I_J_K_ClusterRepresentatives"
    # ijk_cluster_representative_dir = sys.argv[1]
    ijk_cluster_representative_dir = "/home/kmeador/yeates/interface_extraction/ijk_cluster_reps"

    # IJ Clustered Interface Fragment Directory - "MappedIntFragsWithGuideAtoms"
    # ij_clustered_intfrags_dir = sys.argv[2]
    ij_clustered_intfrags_dir = "/home/kmeador/yeates/interface_extraction/mapped_paired_xtal_frags"

    # Out Directory - Final IJK Clustered Interface Fragment Directory - "IJK_ClusteredInterfaceFragmentDB"
    # of the Corresponding Fragment Type Cluster Representative - "MappedIntFragsWithGuideAtoms"
    # outdir = sys.argv[3]
    outdir = "/home/kmeador/yeates/interface_extraction/ijk_clustered_xtal_fragDB_1A"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Read In IJK Cluster Representative Fragments. Create Bio.PDB Structure Objects. Store Guide Atoms in IJK Cluster Dictionary
    ijk_cluster_representatives_guide_atoms_dict = {}
    for dirpath, dirnames, filenames in os.walk(ijk_cluster_representative_dir):
        if not dirnames:
            ijk_cluster_representative_path = dirpath + "/" + filenames[0]
            ijk_cluster_representative_parser = Bio.PDB.PDBParser()
            ijk_cluster_representative_biopdb = ijk_cluster_representative_parser.get_structure(os.path.basename(ijk_cluster_representative_path), ijk_cluster_representative_path)

            ijk_cluster_name = dirpath.split("/")[-1]
            ijk_cluster_key = (ijk_cluster_name.split("_")[0], ijk_cluster_name.split("_")[1], ijk_cluster_name.split("_")[2])
            ijk_cluster_representatives_guide_atoms_dict[ijk_cluster_key] = get_guide_atoms_biopdb(ijk_cluster_representative_biopdb)

    # Get Directory Paths for IJ Clustered Interface Fragment PDB files
    j_directory_paths = []
    for dirpath, dirnames, filenames in os.walk(ij_clustered_intfrags_dir):
        if not dirnames:
            j_directory_paths.append(dirpath)

    for j_dir in j_directory_paths:

        i_cluster = j_dir.split("/")[-1].split("_")[0]  # ensure '_vs_' (9_km_) or '_' (9_kmv1_)
        j_cluster = j_dir.split("/")[-1].split("_")[1]  # ensure '_vs_' or '_'

        for root, dirs, files in os.walk(j_dir):
            for filename in files:
                if filename.endswith(".pdb"):
                    ij_clustered_intfrag_path = j_dir + "/" + filename
                    ij_clustered_intfrag_parser = Bio.PDB.PDBParser()
                    ij_clustered_intfrag_biopdb = ij_clustered_intfrag_parser.get_structure(filename, ij_clustered_intfrag_path)
                    ij_clustered_intfrag_guide_atoms = get_guide_atoms_biopdb(ij_clustered_intfrag_biopdb)

                    min_rmsd = sys.maxint
                    min_rmsd_ijk_cluster = None
                    for ijk_rep_key in ijk_cluster_representatives_guide_atoms_dict:
                        print ijk_rep_key
			if ijk_rep_key[0] == i_cluster and ijk_rep_key[1] == j_cluster:
                            ijk_cluster_representatives_guide_atoms = ijk_cluster_representatives_guide_atoms_dict[ijk_rep_key]
                            rmsd = guide_atom_rmsd((ij_clustered_intfrag_guide_atoms, ijk_cluster_representatives_guide_atoms))
			    print rmsd
                            if rmsd < min_rmsd:
                                min_rmsd = rmsd
                                min_rmsd_ijk_cluster = ijk_rep_key
                                # print min_rmsd_ijk_cluster
                    if min_rmsd <= clust_rmsd_thresh:
                        new_i_cluster_directory_path = outdir + "/" + min_rmsd_ijk_cluster[0]
                        new_ij_cluster_directory_path = new_i_cluster_directory_path + "/" + min_rmsd_ijk_cluster[0] + "_" + min_rmsd_ijk_cluster[1]
                        new_ijk_cluster_directory_path = new_ij_cluster_directory_path + "/" + min_rmsd_ijk_cluster[0] + "_" + min_rmsd_ijk_cluster[1] + "_" + min_rmsd_ijk_cluster[2]

                        if not os.path.exists(new_i_cluster_directory_path):
                            os.makedirs(new_i_cluster_directory_path)

                        if not os.path.exists(new_ij_cluster_directory_path):
                            os.makedirs(new_ij_cluster_directory_path)

                        if not os.path.exists(new_ijk_cluster_directory_path):
                            os.makedirs(new_ijk_cluster_directory_path)

                        new_filename = os.path.basename(filename) + "_orientation_" + min_rmsd_ijk_cluster[2] + ".pdb"
                        shutil.copy(j_dir + "/" + filename, new_ijk_cluster_directory_path + "/" + new_filename)


if __name__ == '__main__':
    main()
