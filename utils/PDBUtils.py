import numpy as np
import sklearn.neighbors

from classes.Fragment import GhostFragment
from classes.Fragment import MonoFragment
from classes.PDB import *
from utils.GeneralUtils import rot_txint_set_txext_frag_coord_sets


def rot_txint_set_txext_pdb(pdb, rot_mat=None, internal_tx_vec=None, set_mat=None, ext_tx_vec=None):  # Todo to PDB
    # pdb_coords = np.array(pdb.extract_coords())  # TODO
    pdb_coords = np.array(pdb.extract_all_coords())

    if pdb_coords.size != 0:

        # Rotate coordinates if rotation matrix is provided
        if rot_mat is not None:
            rot_mat_T = np.transpose(rot_mat)
            pdb_coords = np.matmul(pdb_coords, rot_mat_T)

        # Translate coordinates if internal translation vector is provided
        if internal_tx_vec is not None:
            pdb_coords = pdb_coords + internal_tx_vec

        # Set coordinates if setting matrix is provided
        if set_mat is not None:
            set_mat_T = np.transpose(set_mat)
            pdb_coords = np.matmul(pdb_coords, set_mat_T)

        # Translate coordinates if external translation vector is provided
        if ext_tx_vec is not None:
            pdb_coords = pdb_coords + ext_tx_vec

        transformed_pdb = PDB()
        transformed_atoms = []
        atom_index = 0
        for atom in pdb.get_atoms():
            x_transformed = pdb_coords[atom_index][0]
            y_transformed = pdb_coords[atom_index][1]
            z_transformed = pdb_coords[atom_index][2]
            atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
                                    atom.get_residue_type(), atom.get_chain(), atom.get_residue_number(),
                                    atom.get_code_for_insertion(), x_transformed, y_transformed, z_transformed,
                                    atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                    atom.get_atom_charge())
            transformed_atoms.append(atom_transformed)
            atom_index += 1

        transformed_pdb.set_all_atoms(transformed_atoms)
        transformed_pdb.set_chain_id_list(pdb.get_chain_id_list())
        transformed_pdb.set_filepath(pdb.get_filepath())

        return transformed_pdb

    else:
        return []


def get_contacting_asu(pdb1, pdb2, contact_dist=8):
    pdb1_ca_coords_chain_dict = {}
    for atom in pdb1.get_atoms():
        if atom.chain not in pdb1_ca_coords_chain_dict:
            pdb1_ca_coords_chain_dict[atom.chain] = [atom.coords()]
        else:
            pdb1_ca_coords_chain_dict[atom.chain].append(atom.coords())

    pdb2_ca_coords_chain_dict = {}
    for atom in pdb2.get_atoms():
        if atom.chain not in pdb2_ca_coords_chain_dict:
            pdb2_ca_coords_chain_dict[atom.chain] = [atom.coords()]
        else:
            pdb2_ca_coords_chain_dict[atom.chain].append(atom.coords())

    max_contact_count = 0
    max_contact_chain1 = None
    max_contact_chain2 = None
    for chain1 in pdb1_ca_coords_chain_dict:
        for chain2 in pdb2_ca_coords_chain_dict:
            pdb1_ca_coords = pdb1_ca_coords_chain_dict[chain1]
            pdb2_ca_coords = pdb2_ca_coords_chain_dict[chain2]

            pdb1_ca_coords_kdtree = sklearn.neighbors.BallTree(np.array(pdb1_ca_coords))
            contact_count = pdb1_ca_coords_kdtree.two_point_correlation(pdb2_ca_coords, [contact_dist])[0]

            if contact_count > max_contact_count:
                max_contact_count = contact_count
                max_contact_chain1 = chain1
                max_contact_chain2 = chain2

    if max_contact_count > 0 and max_contact_chain1 is not None and max_contact_chain2 is not None:
        pdb1_asu = PDB()
        pdb1_asu.read_atom_list(pdb1.chain(max_contact_chain1))
        # pdb1_asu.read_atom_list(pdb1.get_chain_atoms(max_contact_chain1))

        pdb2_asu = PDB()
        pdb2_asu.read_atom_list(pdb2.chain(max_contact_chain2))
        # pdb2_asu.read_atom_list(pdb2.get_chain_atoms(max_contact_chain2))

        return pdb1_asu, pdb2_asu

    else:
        return None, None


def get_interface_fragments(pdb1, pdb2, cb_distance=9.0):
    interface_fragments_pdb1 = []
    interface_fragments_pdb2 = []

    pdb1_cb_coords, pdb1_cb_indices = pdb1.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)
    pdb2_cb_coords, pdb2_cb_indices = pdb2.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)

    pdb1_cb_kdtree = sklearn.neighbors.BallTree(np.array(pdb1_cb_coords))

    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    query = pdb1_cb_kdtree.query_radius(pdb2_cb_coords, cb_distance)

    # Get ResidueNumber, ChainID for all Interacting PDB1 CB, PDB2 CB Pairs
    interacting_pairs = []
    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].tolist() != list():
            pdb2_cb_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].residue_number
            pdb2_cb_chain_id = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].chain
            for pdb1_query_index in query[pdb2_query_index]:
                pdb1_cb_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].residue_number
                pdb1_cb_chain_id = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].chain
                interacting_pairs.append(((pdb1_cb_res_num, pdb1_cb_chain_id), (pdb2_cb_res_num, pdb2_cb_chain_id)))

    pdb1_central_resnum_chainid_used = []
    pdb2_central_resnum_chainid_used = []
    for pair in interacting_pairs:
        int_frag_out_pdb1 = PDB()
        int_frag_out_pdb2 = PDB()
        int_frag_out_atom_list_pdb1 = []
        int_frag_out_atom_list_pdb2 = []

        pdb1_central_res_num = pair[0][0]
        pdb1_central_chain_id = pair[0][1]
        pdb2_central_res_num = pair[1][0]
        pdb2_central_chain_id = pair[1][1]

        pdb1_res_num_list = [pdb1_central_res_num - 2, pdb1_central_res_num - 1, pdb1_central_res_num,
                             pdb1_central_res_num + 1, pdb1_central_res_num + 2]
        pdb2_res_num_list = [pdb2_central_res_num - 2, pdb2_central_res_num - 1, pdb2_central_res_num,
                             pdb2_central_res_num + 1, pdb2_central_res_num + 2]

        frag1_ca_count = 0
        for atom in pdb1.all_atoms:
            if atom.chain == pdb1_central_chain_id:
                if atom.residue_number in pdb1_res_num_list:
                    int_frag_out_atom_list_pdb1.append(atom)
                    if atom.is_CA():
                        frag1_ca_count += 1

        frag2_ca_count = 0
        for atom in pdb2.all_atoms:
            if atom.chain == pdb2_central_chain_id:
                if atom.residue_number in pdb2_res_num_list:
                    int_frag_out_atom_list_pdb2.append(atom)
                    if atom.is_CA():
                        frag2_ca_count += 1

        if frag1_ca_count == 5 and frag2_ca_count == 5:
            if (pdb1_central_res_num, pdb1_central_chain_id) not in pdb1_central_resnum_chainid_used:
                int_frag_out_pdb1.read_atom_list(int_frag_out_atom_list_pdb1)
                interface_fragments_pdb1.append(int_frag_out_pdb1)
                pdb1_central_resnum_chainid_used.append((pdb1_central_res_num, pdb1_central_chain_id))

            if (pdb2_central_res_num, pdb2_central_chain_id) not in pdb2_central_resnum_chainid_used:
                int_frag_out_pdb2.read_atom_list(int_frag_out_atom_list_pdb2)
                interface_fragments_pdb2.append(int_frag_out_pdb2)
                pdb2_central_resnum_chainid_used.append((pdb2_central_res_num, pdb2_central_chain_id))

    return interface_fragments_pdb1, interface_fragments_pdb2


def get_interface_ghost_surf_frags(pdb1, pdb2, pdb1_ghost_frag_list, pdb2_surf_frag_list, rot_mat1, rot_mat2,
                                   internal_tx_vec1, internal_tx_vec2, set_mat1, set_mat2, ext_tx_vec1, ext_tx_vec2,
                                   cb_distance=9.0):
    """Calculate all the residues within a cb_distance between two oligomers, identify associated ghost and surface
    fragments on each, by the chain name and residue number, translated the selected fragments to the oligomers using
    symmetry specific rotation matrix, internal translation vector, setting matrix, and external translation vector then
    return copies of these translated fragments

    Returns:
        (tuple): transformed ghost fragments, transformed surface fragments, transformed ghost guide corrdinates,
        transformed surface guide coordinates, number of interface residues on pdb1 where fragments are possible, number
        on pdb2 where fragments are possible
    """
    # print "Length of Complete PDB1 Ghost Frag List: " + str(len(pdb1_ghost_frag_list)) + "\n"
    # print "Length of Complete PDB2 Surf Frag List: " + str(len(pdb2_surf_frag_list)) + "\n"

    interface_ghost_frag_list = []
    interface_ghost_frag_transformed_list = []

    interface_ghost_frag_pdb_coords_list = []
    interface_ghost_frag_pdb_coords_list_transformed = []
    interface_ghost_frag_guide_coords_list_transformed = []

    interface_surf_frag_list = []
    interface_surf_frag_transformed_list = []

    interface_surf_frag_pdb_coords_list = []
    interface_surf_frag_pdb_coords_list_transformed = []
    interface_surf_frag_guide_coords_list = []
    interface_surf_frag_guide_coords_list_transformed = []

    pdb1_cb_coords, pdb1_cb_indices = pdb1.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)
    pdb2_cb_coords, pdb2_cb_indices = pdb2.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)

    pdb1_cb_kdtree = sklearn.neighbors.BallTree(np.array(pdb1_cb_coords))

    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    query = pdb1_cb_kdtree.query_radius(pdb2_cb_coords, cb_distance)

    # Get ResidueNumber, ChainID for all Interacting PDB1 CB, PDB2 CB Pairs
    interacting_pairs = []
    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].tolist() != list():
            pdb2_cb_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].residue_number
            pdb2_cb_chain_id = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].chain
            for pdb1_query_index in query[pdb2_query_index]:
                pdb1_cb_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].residue_number
                pdb1_cb_chain_id = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].chain
                interacting_pairs.append(((pdb1_cb_res_num, pdb1_cb_chain_id), (pdb2_cb_res_num, pdb2_cb_chain_id)))

    pdb1_central_resnum_chainid_unique_list = []
    pdb2_central_resnum_chainid_unique_list = []
    for pair in interacting_pairs:

        pdb1_central_res_num = pair[0][0]
        pdb1_central_chain_id = pair[0][1]
        pdb2_central_res_num = pair[1][0]
        pdb2_central_chain_id = pair[1][1]

        pdb1_res_num_list = [pdb1_central_res_num - 2, pdb1_central_res_num - 1, pdb1_central_res_num,
                             pdb1_central_res_num + 1, pdb1_central_res_num + 2]
        pdb2_res_num_list = [pdb2_central_res_num - 2, pdb2_central_res_num - 1, pdb2_central_res_num,
                             pdb2_central_res_num + 1, pdb2_central_res_num + 2]

        frag1_ca_count = 0
        for atom in pdb1.all_atoms:
            if atom.chain == pdb1_central_chain_id:
                if atom.residue_number in pdb1_res_num_list:
                    if atom.is_CA():
                        frag1_ca_count += 1

        frag2_ca_count = 0
        for atom in pdb2.all_atoms:
            if atom.chain == pdb2_central_chain_id:
                if atom.residue_number in pdb2_res_num_list:
                    if atom.is_CA():
                        frag2_ca_count += 1

        if frag1_ca_count == 5 and frag2_ca_count == 5:
            if (pdb1_central_chain_id, pdb1_central_res_num) not in pdb1_central_resnum_chainid_unique_list:
                pdb1_central_resnum_chainid_unique_list.append((pdb1_central_chain_id, pdb1_central_res_num))

            if (pdb2_central_chain_id, pdb2_central_res_num) not in pdb2_central_resnum_chainid_unique_list:
                pdb2_central_resnum_chainid_unique_list.append((pdb2_central_chain_id, pdb2_central_res_num))

    # ghost_lookup_time_start = time.time()
    for ghost_frag in pdb1_ghost_frag_list:
        if ghost_frag.get_aligned_surf_frag_central_res_tup() in pdb1_central_resnum_chainid_unique_list:
            interface_ghost_frag_list.append(ghost_frag)
            interface_ghost_frag_pdb_coords_list.append(ghost_frag.get_pdb_coords())
    # ghost_lookup_time_end = time.time()
    # ghost_lookup_time = ghost_lookup_time_end - ghost_lookup_time_start
    # print "Ghost Lookup Time: " + str(ghost_lookup_time) + "\n"

    # surf_lookup_time_start = time.time()
    for surf_frag in pdb2_surf_frag_list:
        if surf_frag.get_central_res_tup() in pdb2_central_resnum_chainid_unique_list:
            interface_surf_frag_list.append(surf_frag)
            interface_surf_frag_pdb_coords_list.append(surf_frag.get_pdb_coords())
            interface_surf_frag_guide_coords_list.append(surf_frag.get_guide_coords())
    # surf_lookup_time_end = time.time()
    # surf_lookup_time = surf_lookup_time_end - surf_lookup_time_start
    # print "Surf Lookup Time: " + str(surf_lookup_time) + "\n"

    # Rotate, Translate and Set Ghost Fragment Guide Coordinates

    # print "Interface Ghost Fragment PDB Coords List Length: " + str(np.vstack(interface_ghost_frag_pdb_coords_list).size) + "\n"
    # ghost_pdb_transform_time_start = time.time()
    interface_ghost_frag_pdb_coords_list_transformed = rot_txint_set_txext_frag_coord_sets(
        interface_ghost_frag_pdb_coords_list, rot_mat=rot_mat1, internal_tx_vec=internal_tx_vec1, set_mat=set_mat1,
        ext_tx_vec=ext_tx_vec1)
    # ghost_pdb_transform_time_end = time.time()
    # ghost_pdb_transform_time = ghost_pdb_transform_time_end - ghost_pdb_transform_time_start
    # print "Ghost PDB Transform Time: " + str(ghost_pdb_transform_time) + "\n"

    # ghost_copy_time_start = time.time()
    # print "Number of Interface Ghost Fragments: " + str(len(interface_ghost_frag_list)) + "\n"
    for int_ghost_frag_index in range(len(interface_ghost_frag_list)):
        int_ghost_frag = interface_ghost_frag_list[int_ghost_frag_index]
        int_ghost_frag_pdb = int_ghost_frag.get_pdb()
        int_ghost_frag_pdb_atoms = int_ghost_frag_pdb.get_atoms()

        int_ghost_frag_transformed_pdb_coords = interface_ghost_frag_pdb_coords_list_transformed[int_ghost_frag_index]
        int_ghost_frag_pdb_transformed = PDB()
        int_ghost_frag_pdb_transformed_atoms = []
        for atom_index in range(len(int_ghost_frag_pdb_atoms)):
            atom = int_ghost_frag_pdb_atoms[atom_index]
            x_transformed = int_ghost_frag_transformed_pdb_coords[atom_index][0]
            y_transformed = int_ghost_frag_transformed_pdb_coords[atom_index][1]
            z_transformed = int_ghost_frag_transformed_pdb_coords[atom_index][2]
            atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
                                    atom.get_residue_type(), atom.get_chain(), atom.get_residue_number(),
                                    atom.get_code_for_insertion(), x_transformed, y_transformed, z_transformed,
                                    atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                    atom.get_atom_charge())
            int_ghost_frag_pdb_transformed_atoms.append(atom_transformed)
        int_ghost_frag_pdb_transformed.set_all_atoms(int_ghost_frag_pdb_transformed_atoms)

        int_ghost_frag_transformed = GhostFragment(int_ghost_frag_pdb_transformed, int_ghost_frag.get_i_frag_type(),
                                                   int_ghost_frag.get_j_frag_type(), int_ghost_frag.get_k_frag_type(),
                                                   int_ghost_frag.get_rmsd(),
                                                   # int_ghost_frag.get_central_res_tup(),
                                                   int_ghost_frag.get_aligned_surf_frag_central_res_tup(),
                                                   # guide_atoms=int_ghost_frag_pdb_transformed_atoms[-3:],
                                                   guide_coords=int_ghost_frag_transformed_pdb_coords[-3:],
                                                   pdb_coords=int_ghost_frag_transformed_pdb_coords)

        interface_ghost_frag_transformed_list.append(int_ghost_frag_transformed)
        interface_ghost_frag_guide_coords_list_transformed.append(int_ghost_frag_transformed.get_guide_coords())
    # ghost_copy_time_end = time.time()
    # ghost_copy_time = ghost_copy_time_end - ghost_copy_time_start
    # print "Ghost Copy Time: " + str(ghost_copy_time) + "\n"

    # Rotate, Translate and Set Surface Fragment Guide Coordinates
    # surf_pdb_guidecoords_transform_time_start = time.time()
    interface_surf_frag_pdb_coords_list_transformed = rot_txint_set_txext_frag_coord_sets(
        interface_surf_frag_pdb_coords_list, rot_mat=rot_mat2, internal_tx_vec=internal_tx_vec2, set_mat=set_mat2,
        ext_tx_vec=ext_tx_vec2)
    interface_surf_frag_guide_coords_list_transformed = rot_txint_set_txext_frag_coord_sets(
        interface_surf_frag_guide_coords_list, rot_mat=rot_mat2, internal_tx_vec=internal_tx_vec2, set_mat=set_mat2,
        ext_tx_vec=ext_tx_vec2)
    # surf_pdb_guidecoords_transform_time_end = time.time()
    # surf_pdb_guidecoords_transform_time = surf_pdb_guidecoords_transform_time_end - surf_pdb_guidecoords_transform_time_start
    # print "Surf Transform Time: " + str(surf_pdb_guidecoords_transform_time) + "\n"

    # print "Number of Interface Surface Fragments: " + str(len(interface_surf_frag_list)) + "\n"
    # surf_copy_time_start = time.time()
    for int_surf_frag_index in range(len(interface_surf_frag_list)):
        int_surf_frag = interface_surf_frag_list[int_surf_frag_index]

        int_surf_frag_pdb = int_surf_frag.get_pdb()
        int_surf_frag_pdb_transformed = PDB()
        int_surf_frag_transformed_pdb_coords = interface_surf_frag_pdb_coords_list_transformed[int_surf_frag_index]
        int_surf_frag_transformed_pdb_atoms = []
        int_surf_frag_pdb_atoms = int_surf_frag_pdb.get_atoms()
        for atom_index in range(len(int_surf_frag_pdb_atoms)):
            atom = int_surf_frag_pdb_atoms[atom_index]
            x_transformed = int_surf_frag_transformed_pdb_coords[atom_index][0]
            y_transformed = int_surf_frag_transformed_pdb_coords[atom_index][1]
            z_transformed = int_surf_frag_transformed_pdb_coords[atom_index][2]
            atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
                                    atom.get_residue_type(), atom.get_chain(), atom.get_residue_number(),
                                    atom.get_code_for_insertion(), x_transformed, y_transformed, z_transformed,
                                    atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                    atom.get_atom_charge())
            int_surf_frag_transformed_pdb_atoms.append(atom_transformed)
        int_surf_frag_pdb_transformed.set_all_atoms(int_surf_frag_transformed_pdb_atoms)

        int_surf_frag_guide_coords_transformed = interface_surf_frag_guide_coords_list_transformed[int_surf_frag_index]

        int_surf_frag_transformed = MonoFragment(int_surf_frag_pdb_transformed, type=int_surf_frag.get_type(),
                                                 guide_coords=int_surf_frag_guide_coords_transformed,
                                                 central_res_num=int_surf_frag.get_central_res_num(),
                                                 central_res_chain_id=int_surf_frag.get_central_res_chain_id())
        #                                        ,pdb_coords=int_surf_frag_transformed_pdb_coords)

        interface_surf_frag_transformed_list.append(int_surf_frag_transformed)
    # surf_copy_time_end = time.time()
    # surf_copy_time = surf_copy_time_end - surf_copy_time_start
    # print "Surf Copy Time: " + str(surf_copy_time) + "\n\n"

    return interface_ghost_frag_transformed_list, interface_surf_frag_transformed_list, interface_ghost_frag_guide_coords_list_transformed, interface_surf_frag_guide_coords_list_transformed, len(
        pdb1_central_resnum_chainid_unique_list), len(pdb2_central_resnum_chainid_unique_list)
