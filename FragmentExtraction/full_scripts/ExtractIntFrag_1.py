import os

import FragUtils as Frag

from PDB import PDB

# Globals
module = 'Extract Interface Fragments:'
interface_distance = 8


# def construct_cb_atom_tree(pdb1, pdb2, distance):
#     # Get CB Atom Coordinates
#     pdb1_coords = np.array(pdb1.extract_CB_coords())
#     pdb2_coords = np.array(pdb2.extract_CB_coords())
#
#     # Construct CB Tree for PDB1
#     pdb1_tree = sklearn.neighbors.BallTree(pdb1_coords)
#
#     # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
#     query = pdb1_tree.query_radius(pdb2_coords, distance)
#
#     # Map Coordinates to Atoms
#     pdb1_cb_indices = pdb1.get_cb_indices()
#     pdb2_cb_indices = pdb2.get_cb_indices()
#
#     return query, pdb1_cb_indices, pdb2_cb_indices
#
#
# def find_interface_pairs(pdb1, pdb2):
#     # Get Queried CB Tree for all PDB2 Atoms within 8A of PDB1 CB Atoms
#     query, pdb1_cb_indices, pdb2_cb_indices = construct_cb_atom_tree(pdb1, pdb2, interface_distance)
#
#     # Map Coordinates to Residue Numbers
#     interface_pairs = []
#     for pdb2_index in range(len(query)):
#         if query[pdb2_index].tolist() != list():
#             pdb2_res_num = pdb2.atoms[pdb2_cb_indices[pdb2_index]].residue_number
#             for pdb1_index in query[pdb2_index]:
#                 pdb1_res_num = pdb1.atoms[pdb1_cb_indices[pdb1_index]].residue_number
#                 interface_pairs.append((pdb1_res_num, pdb2_res_num))
#
#     return interface_pairs


def main():
    print(module, 'Beginning')
    # Get Natural Interface PDB File Paths
    int_db_dir = os.path.join(os.getcwd(), 'all_clean_pdbs')
    int_db_filepaths = Frag.get_all_pdb_file_paths(int_db_dir)

    frag_outdir = os.path.join(os.getcwd(), 'all_individual_frags')
    paired_frag_outdir = os.path.join(os.getcwd(), 'all_paired_frags')
    if not os.path.exists(frag_outdir):
        os.makedirs(frag_outdir)
    if not os.path.exists(paired_frag_outdir):
        os.makedirs(paired_frag_outdir)

    print(module, 'Creating Neighbor CB Atom Trees at', interface_distance, 'Angstroms Distance')
    print(module, 'Saving Paired and Individual Fragments')
    for pdb_path in int_db_filepaths:
        # Reading In PDB Structure
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
        pdb = PDB()
        pdb.readfile(pdb_path)

        # Creating PDB instance for Ch1 and Ch2
        pdb_ch1 = PDB()
        pdb_ch2 = PDB()
        pdb_ch1_id = pdb.chain_id_list[0]
        pdb_ch2_id = pdb.chain_id_list[1]
        pdb_ch1.read_atom_list(pdb.get_chain_atoms(pdb_ch1_id))
        pdb_ch2.read_atom_list(pdb.get_chain_atoms(pdb_ch2_id))

        # Find Pairs of Interacting Residues
        interacting_pairs = Frag.find_interface_pairs(pdb_ch1, pdb_ch2)

        ch1_central_res_num_used = []
        ch2_central_res_num_used = []
        for pair in interacting_pairs:
            int_frag_out = PDB()
            int_frag_out_ch1 = PDB()
            int_frag_out_ch2 = PDB()
            int_frag_out_atom_list_ch1 = []
            int_frag_out_atom_list_ch2 = []

            ch1_center_res = pair[0]
            ch2_center_res = pair[1]

            ch1_res_num_list = [ch1_center_res - 2, ch1_center_res - 1, ch1_center_res, ch1_center_res + 1, ch1_center_res + 2]
            ch2_res_num_list = [ch2_center_res - 2, ch2_center_res - 1, ch2_center_res, ch2_center_res + 1, ch2_center_res + 2]

            frag1_ca_count = 0
            for atom in pdb_ch1.atoms():
                if atom.residue_number in ch1_res_num_list:
                    int_frag_out_atom_list_ch1.append(atom)
                    if atom.is_CA():
                        frag1_ca_count += 1

            frag2_ca_count = 0
            for atom in pdb_ch2.atoms():
                if atom.residue_number in ch2_res_num_list:
                    int_frag_out_atom_list_ch2.append(atom)
                    if atom.is_CA():
                        frag2_ca_count += 1

            if frag1_ca_count == 5 and frag2_ca_count == 5:
                if ch1_center_res not in ch1_central_res_num_used:
                    int_frag_out_ch1.read_atom_list(int_frag_out_atom_list_ch1)
                    int_frag_out_ch1.write(os.path.join(frag_outdir, pdb_id + '_' + pdb_ch1_id + pdb_ch2_id + '_frag_'
                                                        + str(ch1_center_res) + '_' + pdb_ch1_id + '.pdb'))
                    ch1_central_res_num_used.append(ch1_center_res)

                if ch2_center_res not in ch2_central_res_num_used:
                    int_frag_out_ch2.read_atom_list(int_frag_out_atom_list_ch2)
                    int_frag_out_ch2.write(os.path.join(frag_outdir, pdb_id + '_' + pdb_ch1_id + pdb_ch2_id + '_frag_'
                                                        + str(ch2_center_res) + '_' + pdb_ch2_id + '.pdb'))
                    ch2_central_res_num_used.append(ch2_center_res)

                int_frag_out.read_atom_list(int_frag_out_atom_list_ch1 + int_frag_out_atom_list_ch2)
                int_frag_out.write(os.path.join(paired_frag_outdir, pdb_id + '_' + pdb_ch1_id + pdb_ch2_id + '_frag_'
                                                + str(ch1_center_res) + '_' + str(ch2_center_res) + '.pdb'))

    print(module, 'Finished')


if __name__ == '__main__':
    main()
