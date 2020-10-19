import Bio.PDB.Superimposer
from Bio.PDB.Atom import Atom as BioPDBAtom
from classes.PDB import *
import os
import numpy as np
import sys
from glob import glob
import sklearn.neighbors
from copy import deepcopy
from itertools import permutations
import warnings
from Bio.PDB.Atom import PDBConstructionWarning
import SymDesignUtils as SDUtils
import PathUtils as PUtils
warnings.simplefilter('ignore', PDBConstructionWarning)


################################################ BioPDB Superimposer ###################################################
def biopdb_superimposer(atoms_fixed, atoms_moving):

    biopdb_atom_fixed = []
    for atom in atoms_fixed:
        biopdb_atom_fixed.append(
            BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ, atom.alt_location,
                       " %s " % atom.type, atom.number, element=atom.element_symbol))

    biopdb_atom_moving = []
    for atom in atoms_moving:
        biopdb_atom_moving.append(
            BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ, atom.alt_location,
                       " %s " % atom.type, atom.number, element=atom.element_symbol))

    sup = Bio.PDB.Superimposer()
    sup.set_atoms(biopdb_atom_fixed, biopdb_atom_moving)

    rmsd = sup.rms
    rot = np.transpose(sup.rotran[0]).tolist()
    tx = sup.rotran[1].tolist()

    return rmsd, rot, tx
########################################################################################################################


########################################## STANDARDIZE OLIGOMER CHAIN LENGTHS ##########################################
def standardize_oligomer_chain_lengths(oligomer1_pdb, oligomer2_pdb):
    # This function takes 2 PDB objects as input: oligomer1_pdb and oligomer2_pdb.
    # Both input PDBs are required to have the same residue numbering and the same number of chains.
    # The function identifies residue numbers that are present in all chains (within an oligomer and between oligomers).
    # The function returns both input PDB objects without residues that are not present in all chains (within an
    # oligomer and between oligomers).

    oligomer1_resnums_by_chain_dict = {}
    for atom1 in oligomer1_pdb.get_all_atoms():
        if atom1.is_CA():
            if atom1.get_chain() not in oligomer1_resnums_by_chain_dict:
                oligomer1_resnums_by_chain_dict[atom1.get_chain()] = [atom1.get_residue_number()]
            else:
                oligomer1_resnums_by_chain_dict[atom1.get_chain()].append(atom1.get_residue_number())

    oligomer2_resnums_by_chain_dict = {}
    for atom2 in oligomer2_pdb.get_all_atoms():
        if atom2.is_CA():
            if atom2.get_chain() not in oligomer2_resnums_by_chain_dict:
                oligomer2_resnums_by_chain_dict[atom2.get_chain()] = [atom2.get_residue_number()]
            else:
                oligomer2_resnums_by_chain_dict[atom2.get_chain()].append(atom2.get_residue_number())

    oligomers_resnums_lists = oligomer1_resnums_by_chain_dict.values() + oligomer2_resnums_by_chain_dict.values()
    oligomers_resnums_sets = map(set, oligomers_resnums_lists)
    resnums_in_common = list(set.intersection(*oligomers_resnums_sets))

    oligomer1_pdb_standardized = PDB()
    oligomer1_pdb_standardized_atom_list = []
    oligomer1_pdb_standardized_chid_list = []
    for atom1 in oligomer1_pdb.get_all_atoms():
        if atom1.get_residue_number() in resnums_in_common:
            oligomer1_pdb_standardized_atom_list.append(atom1)
            oligomer1_pdb_standardized_chid_list.append(atom1.get_chain())
    oligomer1_pdb_standardized_chid_list = list(set(oligomer1_pdb_standardized_chid_list))
    oligomer1_pdb_standardized.set_all_atoms(oligomer1_pdb_standardized_atom_list)
    oligomer1_pdb_standardized.set_chain_id_list(oligomer1_pdb_standardized_chid_list)

    oligomer2_pdb_standardized = PDB()
    oligomer2_pdb_standardized_atom_list = []
    oligomer2_pdb_standardized_chid_list = []
    for atom2 in oligomer2_pdb.get_all_atoms():
        if atom2.get_residue_number() in resnums_in_common:
            oligomer2_pdb_standardized_atom_list.append(atom2)
            oligomer2_pdb_standardized_chid_list.append(atom2.get_chain())
    oligomer2_pdb_standardized_chid_list = list(set(oligomer2_pdb_standardized_chid_list))
    oligomer2_pdb_standardized.set_all_atoms(oligomer2_pdb_standardized_atom_list)
    oligomer2_pdb_standardized.set_chain_id_list(oligomer2_pdb_standardized_chid_list)

    return oligomer1_pdb_standardized, oligomer2_pdb_standardized
########################################################################################################################


############################################## ROTATED AND TRANSLATED PDB ##############################################
def rotated_translated_atoms(atom_list, rot, tx):

    coordinates = [[atom.get_x(), atom.get_y(), atom.get_z()] for atom in atom_list]

    coordinates_rot = np.matmul(coordinates, np.transpose(rot))
    coordinates_rot_tx = coordinates_rot + tx

    transformed_atom_list = []
    atom_count = 0
    for atom in atom_list:
        x_transformed = coordinates_rot_tx[atom_count][0]
        y_transformed = coordinates_rot_tx[atom_count][1]
        z_transformed = coordinates_rot_tx[atom_count][2]
        atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
                                atom.get_residue_type(), atom.get_chain(),
                                atom.get_residue_number(),
                                atom.get_code_for_insertion(), x_transformed, y_transformed,
                                z_transformed,
                                atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                atom.get_atom_charge())
        transformed_atom_list.append(atom_transformed)
        atom_count += 1

    return transformed_atom_list


def rotated_translated_pdb(pdb, rot, tx):
    atoms = pdb.get_all_atoms()

    rot_tx_atoms = rotated_translated_atoms(atoms, rot, tx)

    pdb_rot_tx = PDB()
    pdb_rot_tx.set_all_atoms(rot_tx_atoms)

    return pdb_rot_tx
########################################################################################################################


############################################### RMSD CALCULATION TOOLS #################################################
def euclidean_squared_3d(coordinates_1, coordinates_2):
    if len(coordinates_1) != 3 or len(coordinates_2) != 3:
        raise ValueError("len(coordinate list) != 3")

    elif type(coordinates_1) is not list or type(coordinates_2) is not list:
        raise TypeError("input parameters are not of type list")

    else:
        x1, y1, z1 = coordinates_1[0], coordinates_1[1], coordinates_1[2]
        x2, y2, z2 = coordinates_2[0], coordinates_2[1], coordinates_2[2]
        return (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2


def calc_rmsd_3d(coord_list1, coord_list2):
    assert len(coord_list1) == len(coord_list2), "len(coord_list1) != len(coord_list2)\n"

    if len(coord_list1) > 0 and len(coord_list2) > 0:

        err_sum = 0
        for i in range(len(coord_list1)):
            err = euclidean_squared_3d(coord_list1[i], coord_list2[i])
            err_sum += err

        mean = err_sum / float(len(coord_list1))
        rmsd = math.sqrt(mean)

        return rmsd

    else:
        raise ZeroDivisionError("calc_rmsd: Can't divide by 0\n")


def calc_rmsd_3d_atoms(atom_list1, atom_list2):
    assert len(atom_list1) == len(atom_list2), "len(atom_list1) != len(atom_list2)\n"

    coord_list1 = []
    coord_list2 = []
    for i in range(len(atom_list1)):
        coord_list1.append([atom_list1[i].get_x(), atom_list1[i].get_y(), atom_list1[i].get_z()])
        coord_list2.append([atom_list2[i].get_x(), atom_list2[i].get_y(), atom_list2[i].get_z()])

    return calc_rmsd_3d(coord_list1, coord_list2)
########################################################################################################################


################################################### RANKING FUNCTION ###################################################
def res_lev_sum_score_rank(all_design_directories):
    # designid_metric_tup_list = []
    designid_metric_tup_list = [(str(des_dir), SDUtils.gather_fragment_metrics(des_dir, score=True))
                                 for des_dir in all_design_directories]
    # for root1, dirs1, files1 in os.walk(master_design_dirpath):
    #     for file1 in files1:
    #         if "frag_match_info_file.txt" in file1:
    #             info_file_filepath = root1 + "/" + file1
    #
    #             tx_filepath = os.path.dirname(root1)
    #             rot_filepath = os.path.dirname(tx_filepath)
    #             degen_filepath = os.path.dirname(rot_filepath)
    #             design_filepath = os.path.dirname(degen_filepath)
    #
    #             tx_filename = tx_filepath.split("/")[-1]
    #             rot_filename = rot_filepath.split("/")[-1]
    #             degen_filename = degen_filepath.split("/")[-1]
    #             design_filename = design_filepath.split("/")[-1]
    #
    #             # design_path = "/" + design_filename + "/" + degen_filename + "/" + rot_filename + "/" + tx_filename
    #             design_id = degen_filename + "_" + rot_filename + "_" + tx_filename
    #
    #             info_file = open(info_file_filepath, 'r')
    #             for line in info_file.readlines():
    #                 if line.startswith("Residue-Level Summation Score:"):
    #                     score = float(line[30:].rstrip())
    #                     designid_metric_tup_list.append((design_id, score))
    #                     break
    #             info_file.close()

    designid_metric_tup_list_sorted = sorted(designid_metric_tup_list, key=lambda tup: tup[1], reverse=True)
    designid_metric_rank_dict = {d: (m, r) for r, (d, m) in enumerate(designid_metric_tup_list_sorted, 1)}
    # r = 0  # rank count
    # for r, (d, m) in enumerate(designid_metric_tup_list_sorted, 1):
        # r += 1
        # designid_metric_rank_dict[d] = (m, r)

    return designid_metric_rank_dict
########################################################################################################################


######################### FUNCTION TO RETRIEVE INTERFACE CHAIN IDS AND RESIDUE NUMBERS #################################
def interface_chains_and_resnums(pdb1, pdb2, cb_distance=9.0):

    pdb1_cb_coords, pdb1_cb_indices = pdb1.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)
    pdb2_cb_coords, pdb2_cb_indices = pdb2.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)

    pdb1_cb_kdtree = sklearn.neighbors.BallTree(np.array(pdb1_cb_coords))

    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    query = pdb1_cb_kdtree.query_radius(pdb2_cb_coords, cb_distance)

    # Get Chain ID and Residue Number for all Residues At The Interface Formed Between PDB1 and PDB2
    pdb1_int_chids_resnums_dict = {}
    pdb2_int_chids_resnums_dict = {}
    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].tolist() != list():
            pdb2_cb_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].residue_number
            pdb2_cb_chain_id = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].chain

            if pdb2_cb_chain_id not in pdb2_int_chids_resnums_dict:
                pdb2_int_chids_resnums_dict[pdb2_cb_chain_id] = [pdb2_cb_res_num]
            elif pdb2_cb_res_num not in pdb2_int_chids_resnums_dict[pdb2_cb_chain_id]:
                pdb2_int_chids_resnums_dict[pdb2_cb_chain_id].append(pdb2_cb_res_num)

            for pdb1_query_index in query[pdb2_query_index]:
                pdb1_cb_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].residue_number
                pdb1_cb_chain_id = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].chain

                if pdb1_cb_chain_id not in pdb1_int_chids_resnums_dict:
                    pdb1_int_chids_resnums_dict[pdb1_cb_chain_id] = [pdb1_cb_res_num]
                elif pdb1_cb_res_num not in pdb1_int_chids_resnums_dict[pdb1_cb_chain_id]:
                    pdb1_int_chids_resnums_dict[pdb1_cb_chain_id].append(pdb1_cb_res_num)

    return pdb1_int_chids_resnums_dict, pdb2_int_chids_resnums_dict
########################################################################################################################


###################################### FUNCTION TO MAP AND ALIGN INTERFACE CHAINS ######################################
def map_align_interface_chains(pdb1, pdb2, ref_pdb1, ref_pdb2, ref_pdb1_int_chids_resnums_dict, ref_pdb2_int_chids_resnums_dict, e=3.0):

    # This function requires pdb1 and ref_pdb1 to have the same: residue numbering, number of chains and number of
    # equivalent CA atoms. Same for pdb2 and ref_pdb2.
    # All input PDBs are assumed to be homo-oligomers with cyclic symmetry.
    # pdb1 and ref_pdb1 are required to have the name number of CA atoms (in total and per chain).
    # pdb2 and ref_pdb2 are required to have the name number of CA atoms (in total and per chain).
    # All chains within a given input PDB must also have the same number of CA atoms.

    # The interface formed between reference pdb1 and reference pdb2 is referred to as: 'reference interface'.

    # The input keyword argument 'e' is the max RMSD threshold in A for the overlap of chain(s) that belong to one
    # reference pdb and that are involved in the 'reference interface' with chain(s) in the corresponding pdb given a
    # specific chain mapping. This attempts to prevent overlaps that disrupt the internal structural integrity of an
    # oligomer. Default 'e' value is set to 3 A. Default is not set to 0 A because 'symmetry related' subunits might
    # slightly differ structurally.

    # 'tot_bio_perms' is the total number of biologically plausible chain mapping permutations i.e. chain mappings that
    # do not disrupt the internal structural integrity of a cyclic oligomer.
    # 'tot_tested_perms' is the total number of interface RMSD values that were calculated.
    # An error message is printed out if tot_tested_perms < tot_bio_perms.
    # This could mean that the 'e' threshold is set too low and that 'symmetry related' subunits differ slightly more
    # structurally.
    # An error message can also be printed out in the event that tot_tested_perms > tot_bio_perms.
    # This could mean that the 'e' threshold is set too high and that interface RMSD values could have been calculated
    # for biologically implausible chain mappings.
    tot_bio_perms = len(ref_pdb1.get_chain_id_list()) * len(ref_pdb2.get_chain_id_list())
    tot_tested_perms = 0

    # Min Interface RMSD
    min_irmsd = sys.maxint
    min_irot = None
    min_itx = None

    # get chain id's for all reference pdb1 and reference pdb2 chains that participate in the 'reference interface'
    ref_pdb1_int_chids = ref_pdb1_int_chids_resnums_dict.keys()
    ref_pdb2_int_chids = ref_pdb2_int_chids_resnums_dict.keys()

    # create a list for both ref_pdb1 and ref_pdb2 that stores all CA atoms that belong to a chain that participates
    # in the 'reference interface'. The atoms in the list are ordered in the same order the atoms(/chains) appear in
    # ref_pdb1 / ref_pdb2
    # create a list for both ref_pdb1 and ref_pdb2 that store chain ids for chains that participate in the
    # 'reference interface' and ordered such that the chain ids appear in the same order as they appear in
    # ref_pdb1 / ref_pdb2
    # create a list 'ref_int_ca_atoms' containing all reference interface CA atoms (from ref_pdb1 and ref_pdb2)
    ref_pdb1_ca_int_ch_atoms = []
    ref_pdb1_int_chids_ordered = []
    ref_pdb1_int_ca_atoms = []
    for ref_pdb1_atom in ref_pdb1.chains(ref_pdb1_int_chids):
        if ref_pdb1_atom.get_chain() not in ref_pdb1_int_chids_ordered:
            ref_pdb1_int_chids_ordered.append(ref_pdb1_atom.get_chain())
        if ref_pdb1_atom.is_CA():
            ref_pdb1_ca_int_ch_atoms.append(ref_pdb1_atom)

            if ref_pdb1_atom.get_residue_number() in ref_pdb1_int_chids_resnums_dict[ref_pdb1_atom.get_chain()]:
                ref_pdb1_int_ca_atoms.append(ref_pdb1_atom)

    ref_pdb2_ca_int_ch_atoms = []
    ref_pdb2_int_chids_ordered = []
    ref_pdb2_int_ca_atoms = []
    for ref_pdb2_atom in ref_pdb2.chains(ref_pdb2_int_chids):
        if ref_pdb2_atom.get_chain() not in ref_pdb2_int_chids_ordered:
            ref_pdb2_int_chids_ordered.append(ref_pdb2_atom.get_chain())
        if ref_pdb2_atom.is_CA():
            ref_pdb2_ca_int_ch_atoms.append(ref_pdb2_atom)

            if ref_pdb2_atom.get_residue_number() in ref_pdb2_int_chids_resnums_dict[ref_pdb2_atom.get_chain()]:
                ref_pdb2_int_ca_atoms.append(ref_pdb2_atom)

    ref_int_ca_atoms = ref_pdb1_int_ca_atoms + ref_pdb2_int_ca_atoms

    # get pdb1 and pdb2 full chain id lists
    pdb1_chids = list(set(pdb1.get_chain_id_list()))
    pdb2_chids = list(set(pdb2.get_chain_id_list()))

    # construct a dictionary for both pdb1 and pdb2 that stores their CA atoms by chain id
    pdb1_chid_ca_atom_dict = {}
    for pdb1_chid in pdb1_chids:
        pdb1_chid_ca_atom_dict[pdb1_chid] = [a for a in pdb1.chain(pdb1_chid) if a.is_CA()]
    pdb2_chid_ca_atom_dict = {}
    for pdb2_chid in pdb2_chids:
        pdb2_chid_ca_atom_dict[pdb2_chid] = [a for a in pdb2.chain(pdb2_chid) if a.is_CA()]

    # construct lists of all possible chain id permutations for pdb1 and for pdb2
    # that could map onto reference pdb1 and reference pdb2 interface chains respectively
    pdb1_chids_perms = list(permutations(pdb1_chids, len(ref_pdb1_int_chids)))
    pdb2_chids_perms = list(permutations(pdb2_chids, len(ref_pdb2_int_chids)))

    for pdb1_perm in pdb1_chids_perms:

        pdb1_perm_ca_atoms = []
        for pdb1_ch in pdb1_perm:
            pdb1_perm_ca_atoms.extend(pdb1_chid_ca_atom_dict[pdb1_ch])

        rmsd_1, rot_1, tx_1 = biopdb_superimposer(pdb1_perm_ca_atoms, ref_pdb1_ca_int_ch_atoms)  # fixed, moving

        if rmsd_1 < e:

            for pdb2_perm in pdb2_chids_perms:

                pdb2_perm_ca_atoms = []
                for pdb2_ch in pdb2_perm:
                    pdb2_perm_ca_atoms.extend(pdb2_chid_ca_atom_dict[pdb2_ch])

                rmsd_2, rot_2, tx_2 = biopdb_superimposer(pdb2_perm_ca_atoms, ref_pdb2_ca_int_ch_atoms)  # fixed, moving

                if rmsd_2 < e:

                    # get chain id mapping from pdb1_perm to ref_pdb1_int_chids_ordered
                    chid_map_dict_1 = dict(zip(pdb1_perm, ref_pdb1_int_chids_ordered))

                    # get chain id mapping from pdb2_perm to ref_pdb2_int_chids_ordered
                    chid_map_dict_2 = dict(zip(pdb2_perm, ref_pdb2_int_chids_ordered))

                    # create a list of pdb1_perm atoms that map to reference pdb1 interface CA atoms
                    # ==> pdb1_perm_int_ca_atoms
                    pdb1_perm_int_ca_atoms = []
                    for atom in pdb1_perm_ca_atoms:
                        if atom.get_residue_number() in ref_pdb1_int_chids_resnums_dict[chid_map_dict_1[atom.get_chain()]]:
                            pdb1_perm_int_ca_atoms.append(atom)

                    # create a list of pdb2_perm atoms that map to reference pdb2 interface CA atoms
                    # ==> pdb2_perm_int_ca_atoms
                    pdb2_perm_int_ca_atoms = []
                    for atom in pdb2_perm_ca_atoms:
                        if atom.get_residue_number() in ref_pdb2_int_chids_resnums_dict[chid_map_dict_2[atom.get_chain()]]:
                            pdb2_perm_int_ca_atoms.append(atom)

                    # create a single list containing both pdb1_perm and pdb2_perm CA atoms that map to reference
                    # interface CA atoms by concatenating pdb1_perm_int_ca_atoms and pdb2_perm_int_ca_atoms lists
                    perm_int_ca_atoms = pdb1_perm_int_ca_atoms + pdb2_perm_int_ca_atoms

                    if len(perm_int_ca_atoms) != len(ref_int_ca_atoms):
                        raise Exception("cannot calculate irmsd: number of ref_pdb1/ref_pdb2 reference interface CA atoms != number of pdb1/pdb2 CA atoms that map to reference interface CA atoms\n")

                    irmsd, irot, itx = biopdb_superimposer(perm_int_ca_atoms, ref_int_ca_atoms)  # fixed, moving

                    tot_tested_perms += 1

                    if irmsd < min_irmsd:
                        min_irmsd = irmsd
                        min_irot = irot
                        min_itx = itx

    if tot_tested_perms < tot_bio_perms:
        ex_line_1 = "number of iRMSD values calculated (%s) < number of biologically plausible chain mappings (%s)\n" % (str(tot_tested_perms), str(tot_bio_perms))
        ex_line_2 = "this could mean that the 'e' (%s) threshold is set too low and that 'symmetry related' subunits differ more structurally\n" % str(e)
        raise Exception("%s%s" % (ex_line_1, ex_line_2))

    elif tot_tested_perms > tot_bio_perms:
        ex_line_1 = "number of iRMSD values calculated (%s) > number of biologically plausible chain mappings (%s)\n" % (str(tot_tested_perms), str(tot_bio_perms))
        ex_line_2 = "this could mean that the 'e' (%s) threshold is set too high and that interface RMSD values could have been calculated for biologically implausible chain mappings\n" % str(e)
        raise Exception("%s%s" % (ex_line_1, ex_line_2))

    else:
        # Create a new PDB object that includes both reference pdb1 and reference pdb2
        # rotated and translated using min_rot and min_tx
        ref_pdb1_rot_tx = rotated_translated_pdb(ref_pdb1, min_irot, min_itx)
        ref_pdb2_rot_tx = rotated_translated_pdb(ref_pdb2, min_irot, min_itx)
        ref_pdbs_rot_tx = PDB()
        ref_pdbs_rot_tx.set_all_atoms(ref_pdb1_rot_tx.get_all_atoms() + ref_pdb2_rot_tx.get_all_atoms())
        return ref_pdbs_rot_tx, min_irmsd
########################################################################################################################


############################################### Crystal VS Docked ######################################################
def get_docked_pdb_pairs(all_design_directories):

    docked_pdb_pairs = []
    for des_dir in all_design_directories:
        docked_pdbs_d = []
        for building_block in os.path.basename(des_dir.building_blocks).split('_'):
            docked_pdb = PDB()
            docked_pdb.readfile(glob(os.path.join(des_dir.path, building_block + '_tx_*.pdb'))[0])
            docked_pdbs_d.append(docked_pdb)
        docked_pdb_pairs.append((str(des_dir), tuple(docked_pdbs_d)))

    return docked_pdb_pairs


def crystal_vs_docked_irmsd(xtal_pdb1, xtal_pdb2, all_design_directories):

    return_list = []

    # get all (docked_pdb1, docked_pdb2) pairs
    docked_pdb_pairs = get_docked_pdb_pairs(all_design_directories)

    for (design_id, (docked_pdb1, docked_pdb2)) in docked_pdb_pairs:
        # standardize oligomer chain lengths such that every 'symmetry related' subunit in an oligomer has the same number
        # of CA atoms and only contains residues (based on residue number) that are present in all 'symmetry related'
        # subunits. Also, standardize oligomer chain lengths such that oligomers being compared have the same number of CA
        # atoms and only contain residues (based on residue number) that are present in all chains of both oligomers.
        stand_docked_pdb1, stand_xtal_pdb_1 = standardize_oligomer_chain_lengths(docked_pdb1, xtal_pdb1)
        stand_docked_pdb2, stand_xtal_pdb_2 = standardize_oligomer_chain_lengths(docked_pdb2, xtal_pdb2)

        # store residue number(s) of amino acid(s) that constitute the interface between xtal_pdb_1 and xtal_pdb_2
        # (i.e. 'reference interface') by their chain id in two dictionaries. One for xtal_pdb_1 and one for xtal_pdb_2.
        # {'chain_id': [residue_number(s)]}
        xtal1_int_chids_resnums_dict, xtal2_int_chids_resnums_dict = interface_chains_and_resnums(stand_xtal_pdb_1,
                                                                                                  stand_xtal_pdb_2,
                                                                                                  cb_distance=9.0)

        # find correct chain mapping between crystal structure and docked pose
        # perform a structural alignment of xtal_pdb_1 onto docked_pdb1 using correct chain mapping
        # transform xtal_pdb_2 using the rotation and translation obtained from the alignment above
        # calculate RMSD between xtal_pdb_2 and docked_pdb2 using only 'reference interface' CA atoms from xtal_pdb_2
        # and corresponding mapped CA atoms in docked_pdb2 ==> interface RMSD or iRMSD
        aligned_xtal_pdb, irmsd = map_align_interface_chains(stand_docked_pdb1, stand_docked_pdb2, stand_xtal_pdb_1,
                                                                stand_xtal_pdb_2, xtal1_int_chids_resnums_dict,
                                                                xtal2_int_chids_resnums_dict)

        return_list.append((design_id, aligned_xtal_pdb, irmsd))

    return return_list
########################################################################################################################


def main():

    ############################################## INPUT PARAMETERS ####################################################

    xtal_pdb1_path = sys.argv[1]
    xtal_pdb2_path = sys.argv[2]
    docked_poses_dirpath = sys.argv[3]
    outdir = sys.argv[4]

    ####################################################################################################################

    # read in crystal structure oligomer 1 and oligomer 2 PDB files
    # get the PDB file names without '.pdb' extension
    # create name for combined crystal structure oligomers
    xtal_pdb1 = PDB()
    xtal_pdb1.readfile(xtal_pdb1_path, remove_alt_location=True)
    xtal_pdb1_name = os.path.splitext(os.path.basename(xtal_pdb1_path))[0]

    xtal_pdb2 = PDB()
    xtal_pdb2.readfile(xtal_pdb2_path, remove_alt_location=True)
    xtal_pdb2_name = os.path.splitext(os.path.basename(xtal_pdb2_path))[0]

    xtal_pdb_name = xtal_pdb1_name + "_" + xtal_pdb2_name

    # retrieve Residue Level Summation Score and Scoring Rank for all docked poses
    all_poses, location = SDUtils.collect_directories(docked_poses_dirpath)  # , file=args.file)
    # assert all_poses != list(), print 'No %s directories found within \'%s\'! Please ensure correct location' % \
    #                                   (PUtils.nano.title(), location)
    all_design_directories = SDUtils.set_up_directory_objects(all_poses)  # , symmetry=args.design_string)

    designid_score_dict = res_lev_sum_score_rank(all_design_directories)  # {design_id: (score, score_rank)}

    # align crystal structure to docked poses and calculate interface RMSD
    aligned_xtal_pdbs = crystal_vs_docked_irmsd(xtal_pdb1, xtal_pdb2, all_design_directories)  # [(design_id, aligned_xtal_pdb, irmsd)]

    # sort by RMSD value from lowest to highest
    aligned_xtal_pdbs_sorted = sorted(aligned_xtal_pdbs, key=lambda tup: tup[2], reverse=False)

    # output a PDB file for all of the aligned crystal structures
    # and a text file containing all of the corresponding:
    # iRMSD values, Residue Level Summation Scores and Score Rankings
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfile = open(outdir + "/crystal_vs_docked_irmsd.txt", "w")
    for design_id, aligned_xtal_pdb, irmsd in aligned_xtal_pdbs_sorted:
        # aligned_xtal_pdb.write(outdir + "/%s_AlignedTo_%s.pdb" % (xtal_pdb_name, design_id))
        design_score, design_score_rank = designid_score_dict[design_id]
        out_str = "{:35s} {:8.3f} {:8.3f} {:10d}\n".format(design_id, irmsd, design_score, design_score_rank)
        outfile.write(out_str)
    outfile.close()


if __name__ == "__main__":
    main()
