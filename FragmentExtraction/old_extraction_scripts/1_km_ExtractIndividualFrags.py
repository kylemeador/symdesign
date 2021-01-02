import sys
import os
import numpy as np
from PDB import PDB
from sklearn.neighbors import BallTree

# Globals
module = 'Extract Individual Fragments:'


def main():
    print(module, 'Starting')
    # Get Natural Interface PDB File Paths
    int_db_dir = os.path.join(os.getcwd(), 'all_clean_pdbs')
    int_db_pdb_filepaths = []
    for root, dirs, files in os.walk(int_db_dir):
        for filename in files:
            if filename.endswith(".pdb"):
                int_db_pdb_filepaths.append(int_db_dir + "/" + filename)

    frag_out_dir = os.path.join(os.getcwd(), 'all_individual_frags')
    if not os.path.exists(frag_out_dir):
        os.makedirs(frag_out_dir)

    for pdb_path in int_db_pdb_filepaths:
        # Reading In PDB Structure
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0][0:4]
        pdb = PDB()
        pdb.readfile(pdb_path, remove_alt_location=True)

        # Creating PDB instance for Ch1 and Ch2
        pdb_ch1_id = pdb.chain_id_list[0]
        pdb_ch1 = PDB()
        pdb_ch1.read_atom_list(pdb.chain(pdb.chain_id_list[0]))

        pdb_ch2_id = pdb.chain_id_list[1]
        pdb_ch2 = PDB()
        pdb_ch2.read_atom_list(pdb.chain(pdb.chain_id_list[1]))

        # Getting Ch1 and Ch2 Coordinates
        ch1_cb_coords = pdb_ch1.extract_CB_coords(InclGlyCA=True)
        ch2_cb_coords = pdb_ch2.extract_CB_coords(InclGlyCA=True)
        ch1_cb_coords = np.array(ch1_cb_coords)
        ch2_cb_coords = np.array(ch2_cb_coords)

        # Constructing CB Tree for Ch1
        ch1_cb_tree = BallTree(ch1_cb_coords)

        # Query Tree for all Ch2 CB Atoms within 8A of Ch1 CB Atoms
        query = ch1_cb_tree.query_radius(ch2_cb_coords, 8)

        # Map Coordinates to Atoms
        ch1_cb_indices = pdb_ch1.get_cb_indices(InclGlyCA=True)
        ch2_cb_indices = pdb_ch2.get_cb_indices(InclGlyCA=True)

        interacting_pairs = []
        for ch2_query_index in range(len(query)):
            if query[ch2_query_index].tolist() != list():
                ch2_cb_res_num = pdb_ch2.all_atoms[ch2_cb_indices[ch2_query_index]].residue_number
                for ch1_query_index in query[ch2_query_index]:
                    ch1_cb_res_num = pdb_ch1.all_atoms[ch1_cb_indices[ch1_query_index]].residue_number
                    interacting_pairs.append((ch1_cb_res_num, ch2_cb_res_num))

        ch1_central_res_num_used = []
        ch2_central_res_num_used = []
        for pair in interacting_pairs:
            int_frag_out_ch1 = PDB()
            int_frag_out_ch2 = PDB()
            int_frag_out_atom_list_ch1 = []
            int_frag_out_atom_list_ch2 = []

            ch1_center_res = pair[0]
            ch2_center_res = pair[1]

            ch1_res_num_list = [ch1_center_res - 2, ch1_center_res - 1, ch1_center_res, ch1_center_res + 1, ch1_center_res + 2]
            ch2_res_num_list = [ch2_center_res - 2, ch2_center_res - 1, ch2_center_res, ch2_center_res + 1, ch2_center_res + 2]

            frag1_ca_count = 0
            for atom in pdb_ch1.all_atoms:
                if atom.residue_number in ch1_res_num_list:
                    int_frag_out_atom_list_ch1.append(atom)
                    if atom.is_CA():
                        frag1_ca_count += 1

            frag2_ca_count = 0
            for atom in pdb_ch2.all_atoms:
                if atom.residue_number in ch2_res_num_list:
                    int_frag_out_atom_list_ch2.append(atom)
                    if atom.is_CA():
                        frag2_ca_count += 1

            if frag1_ca_count == 5 and frag2_ca_count == 5:
                if ch1_center_res not in ch1_central_res_num_used:
                    int_frag_out_ch1.read_atom_list(int_frag_out_atom_list_ch1)
                    int_frag_out_ch1.write(os.path.join(frag_out_dir, pdb_id + '_' + pdb_ch1_id + pdb_ch2_id + '_frag_' + str(ch1_center_res) + '_' + pdb_ch1_id + '.pdb'))
                    ch1_central_res_num_used.append(ch1_center_res)

                if ch2_center_res not in ch2_central_res_num_used:
                    int_frag_out_ch2.read_atom_list(int_frag_out_atom_list_ch2)
                    int_frag_out_ch2.write(os.path.join(frag_out_dir, pdb_id + '_' + pdb_ch1_id + pdb_ch2_id + '_frag_' + str(ch2_center_res) + '_' + pdb_ch2_id + '.pdb'))
                    ch2_central_res_num_used.append(ch2_center_res)
    print(module, 'Finished')


if __name__ == '__main__':
    main()
