import sys
import warnings
from itertools import permutations, combinations

import Bio.PDB.Superimposer
import numpy as np
import sklearn.neighbors
from Bio.PDB.Atom import PDBConstructionWarning

import DesignDirectory
import PathUtils as PUtils
import SymDesignUtils as SDUtils
from utils.PDBUtils import biopdb_superimposer
from PDB import PDB
from Structure import Atom

warnings.simplefilter('ignore', PDBConstructionWarning)


########################################## STANDARDIZE OLIGOMER CHAIN LENGTHS ##########################################
def standardize_oligomer_chain_lengths(oligomer1_pdb, oligomer2_pdb):
    # This function takes 2 PDB objects as input: oligomer1_pdb and oligomer2_pdb.
    # Both input PDBs are required to have the same residue numbering and the same number of chains.
    # The function identifies residue numbers that are present in all chains (within an oligomer and between oligomers).
    # The function returns both input PDB objects without residues that are not present in all chains (within an
    # oligomer and between oligomers).

    oligomer1_resnums_by_chain_dict = {}
    for atom1 in oligomer1_pdb.atoms:
        if atom1.is_CA():
            if atom1.get_chain() not in oligomer1_resnums_by_chain_dict:
                oligomer1_resnums_by_chain_dict[atom1.get_chain()] = [atom1.get_residue_number()]
            else:
                oligomer1_resnums_by_chain_dict[atom1.get_chain()].append(atom1.get_residue_number())

    oligomer2_resnums_by_chain_dict = {}
    for atom2 in oligomer2_pdb.atoms:
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
    for atom1 in oligomer1_pdb.atoms:
        if atom1.get_residue_number() in resnums_in_common:
            oligomer1_pdb_standardized_atom_list.append(atom1)
            oligomer1_pdb_standardized_chid_list.append(atom1.get_chain())
    oligomer1_pdb_standardized_chid_list = list(set(oligomer1_pdb_standardized_chid_list))
    oligomer1_pdb_standardized.set_all_atoms(oligomer1_pdb_standardized_atom_list)
    oligomer1_pdb_standardized.chain_id_list = oligomer1_pdb_standardized_chid_list

    oligomer2_pdb_standardized = PDB()
    oligomer2_pdb_standardized_atom_list = []
    oligomer2_pdb_standardized_chid_list = []
    for atom2 in oligomer2_pdb.atoms:
        if atom2.get_residue_number() in resnums_in_common:
            oligomer2_pdb_standardized_atom_list.append(atom2)
            oligomer2_pdb_standardized_chid_list.append(atom2.get_chain())
    oligomer2_pdb_standardized_chid_list = list(set(oligomer2_pdb_standardized_chid_list))
    oligomer2_pdb_standardized.set_all_atoms(oligomer2_pdb_standardized_atom_list)
    oligomer2_pdb_standardized.chain_id_list = oligomer2_pdb_standardized_chid_list

    return oligomer1_pdb_standardized, oligomer2_pdb_standardized


def standardize_intra_oligomer_chain_lengths(oligomer1_pdb):
    # This function takes 1 PDB object as input: oligomer_pdb
    # Input is assumed to be a homo-oligomer
    # It is assumed that all subunits have the same residue numbering
    # The function identifies residue numbers that are present in all chains of the homo-oligomer
    # The function returns the input PDB object without residues that are not present in all chains of the homo-oligomer

    oligomer1_resnums_by_chain_dict = {}
    for atom1 in oligomer1_pdb.atoms:
        if atom1.is_CA():
            if atom1.get_chain() not in oligomer1_resnums_by_chain_dict:
                oligomer1_resnums_by_chain_dict[atom1.get_chain()] = [atom1.get_residue_number()]
            else:
                oligomer1_resnums_by_chain_dict[atom1.get_chain()].append(atom1.get_residue_number())

    oligomers_resnums_lists = oligomer1_resnums_by_chain_dict.values()
    oligomers_resnums_sets = map(set, oligomers_resnums_lists)
    resnums_in_common = list(set.intersection(*oligomers_resnums_sets))

    oligomer1_pdb_standardized = PDB()
    oligomer1_pdb_standardized_atom_list = []
    oligomer1_pdb_standardized_chid_list = []
    for atom1 in oligomer1_pdb.atoms:
        if atom1.get_residue_number() in resnums_in_common:
            oligomer1_pdb_standardized_atom_list.append(atom1)
            oligomer1_pdb_standardized_chid_list.append(atom1.get_chain())
    oligomer1_pdb_standardized_chid_list = list(set(oligomer1_pdb_standardized_chid_list))
    oligomer1_pdb_standardized.set_all_atoms(oligomer1_pdb_standardized_atom_list)
    oligomer1_pdb_standardized.chain_id_list = oligomer1_pdb_standardized_chid_list

    return oligomer1_pdb_standardized
########################################################################################################################


############################################## ROTATED AND TRANSLATED PDB ##############################################
def rotated_translated_atoms(atom_list, rot, tx):

    coordinates = [[atom.coords] for atom in atom_list]
    # coordinates = [[atom.get_x(), atom.get_y(), atom.get_z()] for atom in atom_list]

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
    atoms = pdb.atoms

    rot_tx_atoms = rotated_translated_atoms(atoms, rot, tx)

    pdb_rot_tx = PDB()
    pdb_rot_tx.set_all_atoms(rot_tx_atoms)

    return pdb_rot_tx
########################################################################################################################


######################### FUNCTION TO RETRIEVE INTERFACE CHAIN IDS AND RESIDUE NUMBERS #################################
def interface_chains_and_resnums(pdb1, pdb2, cb_distance=9.0):
    pdb1_cb_coords = pdb1.get_cb_coords()
    pdb1_cb_indices = pdb1.cb_indices
    pdb2_cb_coords = pdb2.get_cb_coords()
    pdb2_cb_indices = pdb2.cb_indices

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
def map_align_interface_chains(pdb1, pdb2, ref_pdb1, ref_pdb2, ref_pdb1_int_chids_resnums_dict,
                               ref_pdb2_int_chids_resnums_dict, e=3.0, return_aligned_ref_pdbs=True):

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
    tot_bio_perms = len(ref_pdb1.chain_id_list) * len(ref_pdb2.chain_id_list)
    tot_tested_perms = 0

    # Min Interface RMSD
    min_irmsd = float('inf')
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
    for ref_pdb1_atom in ref_pdb1.chain(ref_pdb1_int_chids).atoms:
        if ref_pdb1_atom.get_chain() not in ref_pdb1_int_chids_ordered:
            ref_pdb1_int_chids_ordered.append(ref_pdb1_atom.get_chain())
        if ref_pdb1_atom.is_CA():
            ref_pdb1_ca_int_ch_atoms.append(ref_pdb1_atom)

            if ref_pdb1_atom.get_residue_number() in ref_pdb1_int_chids_resnums_dict[ref_pdb1_atom.get_chain()]:
                ref_pdb1_int_ca_atoms.append(ref_pdb1_atom)

    ref_pdb2_ca_int_ch_atoms = []
    ref_pdb2_int_chids_ordered = []
    ref_pdb2_int_ca_atoms = []
    for ref_pdb2_atom in ref_pdb2.chain(ref_pdb2_int_chids).atoms:
        if ref_pdb2_atom.get_chain() not in ref_pdb2_int_chids_ordered:
            ref_pdb2_int_chids_ordered.append(ref_pdb2_atom.get_chain())
        if ref_pdb2_atom.is_CA():
            ref_pdb2_ca_int_ch_atoms.append(ref_pdb2_atom)

            if ref_pdb2_atom.get_residue_number() in ref_pdb2_int_chids_resnums_dict[ref_pdb2_atom.get_chain()]:
                ref_pdb2_int_ca_atoms.append(ref_pdb2_atom)

    ref_int_ca_atoms = ref_pdb1_int_ca_atoms + ref_pdb2_int_ca_atoms

    # get pdb1 and pdb2 full chain id lists
    pdb1_chids = list(set(pdb1.chain_id_list))
    pdb2_chids = list(set(pdb2.chain_id_list))

    # construct a dictionary for both pdb1 and pdb2 that stores their CA atoms by chain id
    pdb1_chid_ca_atom_dict = {}
    for pdb1_chid in pdb1_chids:
        pdb1_chid_ca_atom_dict[pdb1_chid] = [a for a in pdb1.chain(pdb1_chid).atoms if a.is_CA()]
    pdb2_chid_ca_atom_dict = {}
    for pdb2_chid in pdb2_chids:
        pdb2_chid_ca_atom_dict[pdb2_chid] = [a for a in pdb2.chain(pdb2_chid).atoms if a.is_CA()]

    # construct lists of all possible chain id permutations for pdb1 and for pdb2
    # that could map onto reference pdb1 and reference pdb2 interface chains respectively
    # KM this is an excess of what would need to be tested. pdb2_perm inner loop doesn't need to be done for each
    # pdb1_perm and should be brought to the outer loop. This will cut down run time
    pdb1_chids_perms = list(permutations(pdb1_chids, len(ref_pdb1_int_chids)))
    pdb2_chids_perms = list(permutations(pdb2_chids, len(ref_pdb2_int_chids)))

    for pdb1_perm in pdb1_chids_perms:

        pdb1_perm_ca_atoms = []
        for pdb1_ch in pdb1_perm:
            pdb1_perm_ca_atoms.extend(pdb1_chid_ca_atom_dict[pdb1_ch])

        rmsd_1, rot_1, tx_1 = biopdb_superimposer(pdb1_perm_ca_atoms, ref_pdb1_ca_int_ch_atoms)  # fixed, moving
        # rot_1, tx_1 not used
        if rmsd_1 < e:

            for pdb2_perm in pdb2_chids_perms:

                pdb2_perm_ca_atoms = []
                for pdb2_ch in pdb2_perm:
                    pdb2_perm_ca_atoms.extend(pdb2_chid_ca_atom_dict[pdb2_ch])

                rmsd_2, rot_2, tx_2 = biopdb_superimposer(pdb2_perm_ca_atoms, ref_pdb2_ca_int_ch_atoms)  # fixed, moving
                # rot_2, tx_2 not used
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
        if return_aligned_ref_pdbs:
            # Create a new PDB object that includes both reference pdb1 and reference pdb2
            # rotated and translated using min_rot and min_tx
            ref_pdb1_rot_tx = rotated_translated_pdb(ref_pdb1, np.transpose(min_irot), min_itx)
            ref_pdb2_rot_tx = rotated_translated_pdb(ref_pdb2, np.transpose(min_irot), min_itx)
            ref_pdbs_rot_tx = PDB()
            ref_pdbs_rot_tx.set_all_atoms(ref_pdb1_rot_tx.atoms + ref_pdb2_rot_tx.atoms)
            return ref_pdbs_rot_tx, min_irmsd
        else:
            return min_irmsd

########################################################################################################################
def map_align_interface_chains_km_mp(pdb1, pdb2, ref_pdb1, ref_pdb2, id_1, id_2, task, ref_pdb1_int_chids_resnums_dict,
                                     ref_pdb2_int_chids_resnums_dict):
    try:
        irmsd = map_align_interface_chains_km(pdb1, pdb2, ref_pdb1, ref_pdb2,  id_1, id_2, task,
                                              ref_pdb1_int_chids_resnums_dict, ref_pdb2_int_chids_resnums_dict)
        # print('returning', irmsd, None, 'inside _mp')
        return irmsd, None
    except (Bio.PDB.PDBExceptions.PDBException, Exception) as e:
        return None, ((pdb1.filepath, pdb2.filepath), e)


def map_align_interface_chains_km(pdb1, pdb2, ref_pdb1, ref_pdb2,  id_1, id_2, task, ref_pdb1_int_chids_resnums_dict,
                                  ref_pdb2_int_chids_resnums_dict, e=3.0, return_aligned_ref_pdbs=False):
    print('Process:\t%s\tTask\t%s' % (os.getpid(), task))
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

    # tot_bio_perms = len(ref_pdb1.chain_id_list) * len(ref_pdb2.chain_id_list)
    # tot_tested_perms = 0

    # get chain id's for all reference pdb1 and reference pdb2 chains that participate in the 'reference interface'
    ref_pdb1_int_chids = ref_pdb1_int_chids_resnums_dict.keys()
    ref_pdb2_int_chids = ref_pdb2_int_chids_resnums_dict.keys()

    # create a list for both ref_pdb1 and ref_pdb2 that stores all CA atoms that belong to a chain that participates
    # in the 'reference interface'. The atoms in the list are ordered in the same order the atoms(/chains) appear in
    # ref_pdb1 / ref_pdb2 <- KM comment: WHY??
    # create a list for both ref_pdb1 and ref_pdb2 that store chain ids for chains that participate in the
    # 'reference interface' and ordered such that the chain ids appear in the same order as they appear in
    # ref_pdb1 / ref_pdb2
    # create a list 'ref_ca_int_atoms' containing all reference interface CA atoms (from ref_pdb1 and ref_pdb2)
    ref_pdb1_ca_int_ch_atoms = []
    # ref_pdb1_int_chids_ordered = []  # unnecessary in python 3.6+
    ref_pdb1_ca_int_atoms = []
    for ref_pdb1_atom in ref_pdb1.chain(ref_pdb1_int_chids).atoms:
        # if ref_pdb1_atom.get_chain() not in ref_pdb1_int_chids_ordered:
        #     ref_pdb1_int_chids_ordered.append(ref_pdb1_atom.get_chain())
        if ref_pdb1_atom.is_CA():
            ref_pdb1_ca_int_ch_atoms.append(ref_pdb1_atom)

            if ref_pdb1_atom.get_residue_number() in ref_pdb1_int_chids_resnums_dict[ref_pdb1_atom.get_chain()]:
                ref_pdb1_ca_int_atoms.append(ref_pdb1_atom)

    ref_pdb2_ca_int_ch_atoms = []
    # ref_pdb2_int_chids_ordered = []  # unnecessary in python 3.6+
    ref_pdb2_ca_int_atoms = []
    for ref_pdb2_atom in ref_pdb2.chain(ref_pdb2_int_chids).atoms:
        # if ref_pdb2_atom.get_chain() not in ref_pdb2_int_chids_ordered:
        #     ref_pdb2_int_chids_ordered.append(ref_pdb2_atom.get_chain())
        if ref_pdb2_atom.is_CA():
            ref_pdb2_ca_int_ch_atoms.append(ref_pdb2_atom)

            if ref_pdb2_atom.get_residue_number() in ref_pdb2_int_chids_resnums_dict[ref_pdb2_atom.get_chain()]:
                ref_pdb2_ca_int_atoms.append(ref_pdb2_atom)

    ref_ca_int_atoms = ref_pdb1_ca_int_atoms + ref_pdb2_ca_int_atoms

    # get pdb1 and pdb2 full chain id lists
    pdb1_chids = list(set(pdb1.chain_id_list))
    pdb2_chids = list(set(pdb2.chain_id_list))

    # construct a dictionary for both pdb1 and pdb2 that stores their CA atoms by chain id
    pdb1_chid_ca_atom_dict = {}
    for pdb1_chid in pdb1_chids:
        pdb1_chid_ca_atom_dict[pdb1_chid] = [a for a in pdb1.chain(pdb1_chid).atoms if a.is_CA()]
    pdb2_chid_ca_atom_dict = {}
    for pdb2_chid in pdb2_chids:
        pdb2_chid_ca_atom_dict[pdb2_chid] = [a for a in pdb2.chain(pdb2_chid).atoms if a.is_CA()]

    # construct lists of all possible chain id permutations for pdb1 and for pdb2
    # that could map onto reference pdb1 and reference pdb2 interface chains respectively
    # KM this is an excess of what would need to be tested. pdb2_perm inner loop doesn't need to be done for each
    # pdb1_perm and should be brought to the outer loop. This will cut down run time
    pdb1_chids_perms = list(permutations(pdb1_chids, len(ref_pdb1_int_chids)))
    pdb2_chids_perms = list(permutations(pdb2_chids, len(ref_pdb2_int_chids)))

    allowed_perms1, allowed_perms2 = {}, {}
    for pdb1_perm in pdb1_chids_perms:

        pdb1_perm_ca_atoms = []
        for pdb1_ch in pdb1_perm:
            pdb1_perm_ca_atoms.extend(pdb1_chid_ca_atom_dict[pdb1_ch])

        rmsd_1, rot_1, tx_1 = biopdb_superimposer(pdb1_perm_ca_atoms, ref_pdb1_ca_int_ch_atoms)  # fixed, moving
        # rot_1, tx_1 not used
        if rmsd_1 < e:
            allowed_perms1[pdb1_perm] = pdb1_perm_ca_atoms

    for pdb2_perm in pdb2_chids_perms:

        pdb2_perm_ca_atoms = []
        for pdb2_ch in pdb2_perm:
            pdb2_perm_ca_atoms.extend(pdb2_chid_ca_atom_dict[pdb2_ch])
        # try:
        rmsd_2, rot_2, tx_2 = biopdb_superimposer(pdb2_perm_ca_atoms, ref_pdb2_ca_int_ch_atoms)  # fixed, moving
        # except Bio.PDB.PDBExceptions.PDBException:
        #
        #     raise Exception('reference2 (%s) and query2 (%s) have different atom lengths ref2 (chains:%s, len:%d) '
        #                     '!= query2 (chains:%s, len:%d)' % (pdb2.filepath, ref_pdb2.filepath, pdb2_perm,
        #                                                        len(pdb2_perm_ca_atoms), ref_pdb2_int_chids,
        #                                                        len(ref_pdb2_ca_int_ch_atoms)))
        # rot_2, tx_2 not used
        if rmsd_2 < e:
            allowed_perms2[pdb2_perm] = pdb2_perm_ca_atoms

    # there should be the same number of allowed perms that there is symmetry, it's not required to test so many,
    # but we can't be sure about grabbing the right atoms. I calculated this using an atom invarient measure with pose
    # numbering for the P432 clustering exercise
    # Min Interface RMSD
    min_irmsd = float('inf')
    min_irot = None
    min_itx = None
    for pdb1_perm in allowed_perms1:
        # get chain id mapping from pdb1_perm to ref_pdb1_int_chids_ordered
        chid_map_dict_1 = dict(zip(pdb1_perm, ref_pdb1_int_chids))

        # create a list of pdb1_perm atoms that map to reference pdb1 interface CA atoms
        # ==> pdb1_perm_int_ca_atoms
        pdb1_perm_int_ca_atoms = []
        for atom in allowed_perms1[pdb1_perm]:
            if atom.get_residue_number() in ref_pdb1_int_chids_resnums_dict[chid_map_dict_1[atom.get_chain()]]:
                pdb1_perm_int_ca_atoms.append(atom)

        for pdb2_perm in allowed_perms2:
            # get chain id mapping from pdb2_perm to ref_pdb2_int_chids_ordered
            chid_map_dict_2 = dict(zip(pdb2_perm, ref_pdb2_int_chids))

            # create a list of pdb2_perm atoms that map to reference pdb2 interface CA atoms
            # ==> pdb2_perm_int_ca_atoms
            pdb2_perm_int_ca_atoms = []
            for atom in allowed_perms2[pdb2_perm]:
                if atom.get_residue_number() in ref_pdb2_int_chids_resnums_dict[chid_map_dict_2[atom.get_chain()]]:
                    pdb2_perm_int_ca_atoms.append(atom)

            # create a single list containing both pdb1_perm and pdb2_perm CA atoms that map to reference
            # interface CA atoms by concatenating pdb1_perm_int_ca_atoms and pdb2_perm_int_ca_atoms lists
            perm_int_ca_atoms = pdb1_perm_int_ca_atoms + pdb2_perm_int_ca_atoms

            if len(perm_int_ca_atoms) != len(ref_ca_int_atoms):
                raise Exception("cannot calculate irmsd: number of ref_pdb1/ref_pdb2 reference interface CA atoms != number of pdb1/pdb2 CA atoms that map to reference interface CA atoms\n")

            irmsd, irot, itx = biopdb_superimposer(perm_int_ca_atoms, ref_ca_int_atoms)  # fixed, moving

            # tot_tested_perms += 1

            if irmsd < min_irmsd:
                min_irmsd = irmsd
                min_irot = irot
                min_itx = itx

    # if tot_tested_perms < tot_bio_perms:
    #     ex_line_1 = "number of iRMSD values calculated (%s) < number of biologically plausible chain mappings (%s)\n" % (str(tot_tested_perms), str(tot_bio_perms))
    #     ex_line_2 = "this could mean that the 'e' (%s) threshold is set too low and that 'symmetry related' subunits differ more structurally\n" % str(e)
    #     raise Exception("%s%s" % (ex_line_1, ex_line_2))
    #
    # elif tot_tested_perms > tot_bio_perms:
    #     ex_line_1 = "number of iRMSD values calculated (%s) > number of biologically plausible chain mappings (%s)\n" % (str(tot_tested_perms), str(tot_bio_perms))
    #     ex_line_2 = "this could mean that the 'e' (%s) threshold is set too high and that interface RMSD values could have been calculated for biologically implausible chain mappings\n" % str(e)
    #     raise Exception("%s%s" % (ex_line_1, ex_line_2))

    else:
        if not return_aligned_ref_pdbs:
            # print('returning min_irmsd inside _km inside _mp')
            return id_1, id_2, min_irmsd
        else:
            # Create a new PDB object that includes both reference pdb1 and reference pdb2
            # rotated and translated using min_rot and min_tx
            ref_pdb1_rot_tx = rotated_translated_pdb(ref_pdb1, min_irot, min_itx)
            ref_pdb2_rot_tx = rotated_translated_pdb(ref_pdb2, min_irot, min_itx)
            ref_pdbs_rot_tx = PDB()
            ref_pdbs_rot_tx.set_all_atoms(ref_pdb1_rot_tx.atoms + ref_pdb2_rot_tx.atoms)
            return ref_pdbs_rot_tx, min_irmsd

########################################################################################################################


############################################### Crystal VS Docked ######################################################
# def get_docked_pdb_pairs(docked_poses_dirpath):
#
#     docked_pdb_pairs = []
#
#     for root1, dirs1, files1 in os.walk(docked_poses_dirpath):
#         for file1 in files1:
#             if "frag_match_info_file.txt" in file1:
#                 info_file_filepath = root1 + "/" + file1
#
#                 tx_filepath = os.path.dirname(root1)
#                 rot_filepath = os.path.dirname(tx_filepath)
#                 degen_filepath = os.path.dirname(rot_filepath)
#                 design_filepath = os.path.dirname(degen_filepath)
#
#                 tx_filename = tx_filepath.split("/")[-1]
#                 rot_filename = rot_filepath.split("/")[-1]
#                 degen_filename = degen_filepath.split("/")[-1]
#                 design_filename = design_filepath.split("/")[-1]
#
#                 # design_path = "/" + design_filename + "/" + degen_filename + "/" + rot_filename + "/" + tx_filename
#                 design_id = degen_filename + "_" + rot_filename + "_" + tx_filename
#
#                 docked_pdb1_filepath = None
#                 docked_pdb2_filepath = None
#                 info_file = open(info_file_filepath, 'r')
#                 for line in info_file.readlines():
#                     if line.startswith("Original PDB 1 Path:"):
#                         docked_pdb1_filename = os.path.splitext(os.path.basename(line))[0] + "_%s.pdb" % tx_filename
#                         docked_pdb1_filepath = tx_filepath + "/" + docked_pdb1_filename
#                     if line.startswith("Original PDB 2 Path:"):
#                         docked_pdb2_filename = os.path.splitext(os.path.basename(line))[0] + "_%s.pdb" % tx_filename
#                         docked_pdb2_filepath = tx_filepath + "/" + docked_pdb2_filename
#                 info_file.close()
#
#                 if docked_pdb1_filepath is None or docked_pdb2_filepath is None:
#                     raise Exception('cannot find docked pdb file path(s)\n')
#
#                 elif not os.path.exists(docked_pdb1_filepath) or not os.path.exists(docked_pdb2_filepath):
#                     raise Exception('docked pdb file path(s) do not exist\n')
#
#                 else:
#                     docked_pdb1 = PDB()
#                     docked_pdb1.readfile(docked_pdb1_filepath)
#
#                     docked_pdb2 = PDB()
#                     docked_pdb2.readfile(docked_pdb2_filepath)
#
#                     docked_pdb_pairs.append((design_id, docked_pdb1, docked_pdb2))
#
#     return docked_pdb_pairs


# def crystal_vs_docked_irmsd(xtal_pdb1, xtal_pdb2, docked_poses_dirpath):
#     # get all (docked_pdb1, docked_pdb2) pairs
#     docked_pdb_pairs = get_docked_pdb_pairs(docked_poses_dirpath)
#
#     return_list = []
#     for (design_id, docked_pdb1, docked_pdb2) in docked_pdb_pairs:
#
#         # standardize oligomer chain lengths such that every 'symmetry related' subunit in an oligomer has the same number
#         # of CA atoms and only contains residues (based on residue number) that are present in all 'symmetry related'
#         # subunits. Also, standardize oligomer chain lengths such that oligomers being compared have the same number of CA
#         # atoms and only contain residues (based on residue number) that are present in all chains of both oligomers.
#         stand_docked_pdb1, stand_xtal_pdb_1 = standardize_oligomer_chain_lengths(docked_pdb1, xtal_pdb1)
#         stand_docked_pdb2, stand_xtal_pdb_2 = standardize_oligomer_chain_lengths(docked_pdb2, xtal_pdb2)
#
#         # store residue number(s) of amino acid(s) that constitute the interface between xtal_pdb_1 and xtal_pdb_2
#         # (i.e. 'reference interface') by their chain id in two dictionaries. One for xtal_pdb_1 and one for xtal_pdb_2.
#         # {'chain_id': [residue_number(s)]}
#         xtal1_int_chids_resnums_dict, xtal2_int_chids_resnums_dict = interface_chains_and_resnums(stand_xtal_pdb_1,
#                                                                                                   stand_xtal_pdb_2,
#                                                                                                   cb_distance=9.0)
#
#         # find correct chain mapping between crystal structure and docked pose
#         # perform a structural alignment of xtal_pdb_1 onto docked_pdb1 using correct chain mapping
#         # transform xtal_pdb_2 using the rotation and translation obtained from the alignment above
#         # calculate RMSD between xtal_pdb_2 and docked_pdb2 using only 'reference interface' CA atoms from xtal_pdb_2
#         # and corresponding mapped CA atoms in docked_pdb2 ==> interface RMSD or iRMSD
#         aligned_xtal_pdb, irmsd = map_align_interface_chains(stand_docked_pdb1, stand_docked_pdb2, stand_xtal_pdb_1,
#                                                              stand_xtal_pdb_2, xtal1_int_chids_resnums_dict,
#                                                              xtal2_int_chids_resnums_dict)
#
#         return_list.append((design_id, aligned_xtal_pdb, irmsd))
#
#     return return_list
########################################################################################################################


############################################## ALL TO ALL DOCKED POSES IRMSD ###########################################
def get_docked_pdb1_pdb2_filepaths(docked_poses_dirpath, top_ranked_ids):

    docked_pdb1_pdb2_filepaths = []

    for root1, dirs1, files1 in os.walk(docked_poses_dirpath):
        for file1 in files1:
            if "frag_match_info_file.txt" in file1:
                info_file_filepath = root1 + "/" + file1

                tx_filepath = os.path.dirname(root1)
                rot_filepath = os.path.dirname(tx_filepath)
                degen_filepath = os.path.dirname(rot_filepath)
                design_filepath = os.path.dirname(degen_filepath)

                tx_filename = tx_filepath.split("/")[-1]
                rot_filename = rot_filepath.split("/")[-1]
                degen_filename = degen_filepath.split("/")[-1]
                design_filename = design_filepath.split("/")[-1]

                # design_path = "/" + design_filename + "/" + degen_filename + "/" + rot_filename + "/" + tx_filename
                design_id = degen_filename + "_" + rot_filename + "_" + tx_filename

                if design_id in top_ranked_ids:

                    docked_pdb1_filepath = None
                    docked_pdb2_filepath = None
                    info_file = open(info_file_filepath, 'r')
                    for line in info_file.readlines():
                        if line.startswith("Original PDB 1 Path:"):
                            docked_pdb1_filename = os.path.splitext(os.path.basename(line))[0] + "_%s.pdb" % tx_filename
                            docked_pdb1_filepath = tx_filepath + "/" + docked_pdb1_filename
                        if line.startswith("Original PDB 2 Path:"):
                            docked_pdb2_filename = os.path.splitext(os.path.basename(line))[0] + "_%s.pdb" % tx_filename
                            docked_pdb2_filepath = tx_filepath + "/" + docked_pdb2_filename
                    info_file.close()

                    if docked_pdb1_filepath is None or docked_pdb2_filepath is None:
                        raise Exception('cannot find docked pdb file path(s)\n')

                    elif not os.path.exists(docked_pdb1_filepath) or not os.path.exists(docked_pdb2_filepath):
                        raise Exception('docked pdb file path(s) do not exist\n')

                    else:
                        docked_pdb1_pdb2_filepaths.append((design_id, docked_pdb1_filepath, docked_pdb2_filepath))

    return docked_pdb1_pdb2_filepaths


# def all_to_all_docked_poses_irmsd(docked_poses_dirpath, top_ranked_ids):
def all_to_all_docked_poses_irmsd_mp(design_directories, threads):

    # populate a list with (docked_pose_id, docked_pdb1_filepath, docked_pdb2_filepath) tuples
    # for all top scoring docked poses
    # docked_pdb1_pdb2_filepaths = get_docked_pdb1_pdb2_filepaths(docked_poses_dirpath, top_ranked_ids)
    # n = len(docked_pdb1_pdb2_filepaths)

    # get the chain and interface residue numbers for each docked configuration in the entire docked configuration
    # reference1_chains_and_residues_d, reference2_chains_and_residues_d = {}, {}
    reference_chains_and_residues_d = {}
    for i, des_dir in enumerate(design_directories):
        des_dir.get_oligomers()
        # docked_pdb1 = des_dir.oligomers[des_dir.entity_names[0]]
        docked_pdb1 = des_dir.oligomers[0]
        # docked_pdb2 = des_dir.oligomers[des_dir.entity_names[1]]
        docked_pdb2 = des_dir.oligomers[1]
        reference_chains_and_residues_d[str(des_dir)] = interface_chains_and_resnums(docked_pdb1, docked_pdb2,
                                                                                     cb_distance=9.0)
        # ref1_int_chain_residue_d, ref2_int_chain_residue_d = interface_chains_and_resnums(docked_pdb1, docked_pdb2,
        #                                                                                   cb_distance=9.0)
        # reference1_chains_and_residues_d[i] = ref1_int_chain_residue_d
        # reference2_chains_and_residues_d[i] = ref2_int_chain_residue_d

    # For debuggin multiprocessing hang
    # for des_dir in design_directories:
        des_dir.oligomers[des_dir.entity_names[0]] = standardize_intra_oligomer_chain_lengths(docked_pdb1)
        des_dir.oligomers[des_dir.entity_names[1]] = standardize_intra_oligomer_chain_lengths(docked_pdb2)
    # standardized_pdbs1, standardized_pdbs2 = {}, {}
    # for i in range(n):
    #     ref_pose_id, ref_pose_pdb1_filepath, ref_pose_pdb2_filepath = docked_pdb1_pdb2_filepaths[i]
    #     ref_pose_pdb1 = PDB()
    #     ref_pose_pdb1.readfile(ref_pose_pdb1_filepath, remove_alt_location=True)
    #     ref_pose_pdb2 = PDB()
    #     ref_pose_pdb2.readfile(ref_pose_pdb2_filepath, remove_alt_location=True)
    #
    #     # standardize oligomer chain lengths such that every 'symmetry related' subunit in an oligomer has the same
    #     # number of CA atoms and only contains residues (based on residue number) that are present in all
    #     # 'symmetry related' subunits.
    #     KM for all oriented files, this routine is already done
    #     standardized_pdbs1[i] = standardize_intra_oligomer_chain_lengths(ref_pose_pdb1)
    #     standardized_pdbs2[i] = standardize_intra_oligomer_chain_lengths(ref_pose_pdb2)

    # # ORIGINAL PATCH
    # irmsds = []  # , directory_pairs = [], []
    # for pair in combinations(design_directories, 2):
    #     try:
    #         irmsds.append(map_align_interface_chains_km(pair[1].oligomers[pair[1].entity_names[0]],
    #                                                     pair[1].oligomers[pair[1].entity_names[1]],
    #                                                     pair[0].oligomers[pair[0].entity_names[0]],
    #                                                     pair[0].oligomers[pair[0].entity_names[1]], str(pair[0]),
    #                                                     str(pair[1]),
    #                                                     *reference_chains_and_residues_d[str(pair[0])]))
    #     except (Bio.PDB.PDBExceptions.PDBException, Exception):
    #         pass
    #     # directory_pairs.append((str(pair[0]), str(pair[1])))
    #
    # # print(directory_pairs)
    # irmsds, errors = zip(*SDUtils.mp_starmap(map_align_interface_chains_km_mp, zipped_args, threads=threads))
    # irmsds = list(irmsds)
    # errors = list(errors)

    # MULTIPROCESSING
    zipped_args = []  # , directory_pairs = [], []
    for i, pair in enumerate(combinations(design_directories, 2)):
        zipped_args.append((pair[1].oligomers[pair[1].entity_names[0]], pair[1].oligomers[pair[1].entity_names[1]],
                            pair[0].oligomers[pair[0].entity_names[0]], pair[0].oligomers[pair[0].entity_names[1]],
                            str(pair[0]), str(pair[1]), i, *reference_chains_and_residues_d[str(pair[0])]))
        # directory_pairs.append((str(pair[0]), str(pair[1])))

    # print(directory_pairs)
    irmsds, errors = zip(*SDUtils.mp_starmap(map_align_interface_chains_km_mp, zipped_args, threads=threads))
    irmsds = list(irmsds)
    errors = list(errors)
    for i, error in enumerate(errors):
        if error:
            print('ERROR: ', error[1], '\nFiles: ', error[0][0], 'AND', error[0][1])
            # directory_pairs.pop(i)
            irmsds.pop(i)
    # irmsds = SDUtils.mp_starmap(map_align_interface_chains_km, zipped_args, threads=threads)
    # for i in range(n-1):
    #     # # obtain id, oligomer 1 pdb file path and oligomer 2 pdb file path for reference pose
    #     ref_pose_id, ref_pose_pdb1_filepath, ref_pose_pdb2_filepath = docked_pdb1_pdb2_filepaths[i]
    #     #
    #     # # read in pdb files for both reference pose oligomers
    #     # ref_pose_pdb1 = PDB()
    #     # ref_pose_pdb1.readfile(ref_pose_pdb1_filepath, remove_alt_location=True)
    #     # ref_pose_pdb2 = PDB()
    #     # ref_pose_pdb2.readfile(ref_pose_pdb2_filepath, remove_alt_location=True)
    #     #
    #     # # standardize oligomer chain lengths such that every 'symmetry related' subunit in an oligomer has the same
    #     # # number of CA atoms and only contains residues (based on residue number) that are present in all
    #     # # 'symmetry related' subunits.
    #     # stand_ref_pose_pdb1 = standardize_intra_oligomer_chain_lengths(ref_pose_pdb1)
    #     # stand_ref_pose_pdb2 = standardize_intra_oligomer_chain_lengths(ref_pose_pdb2)
    #     #
    #     # # store residue number(s) of amino acid(s) that constitute the interface between stand_ref_pose_pdb1 and
    #     # # stand_ref_pose_pdb2 (i.e. 'reference interface') by their chain id in two dictionaries.
    #     # # One for stand_ref_pose_pdb1 and one for stand_ref_pose_pdb2.
    #     # # {'chain_id': [residue_number(s)]}
    #     ref_pdb1_int_chids_resnums_dict, ref_pdb2_int_chids_resnums_dict = \
    #         interface_chains_and_resnums(stand_ref_pose_pdb1, stand_ref_pose_pdb2, cb_distance=9.0)
    #
    #     for j in range(i+1, n):
    #         # # obtain id, oligomer 1 pdb file path and oligomer 2 pdb file path for query pose
    #         query_pose_id, query_pose_pdb1_filepath, query_pose_pdb2_filepath = docked_pdb1_pdb2_filepaths[j]
    #         #
    #         # # read in pdb files for both query pose oligomers
    #         # query_pose_pdb1 = PDB()
    #         # query_pose_pdb1.readfile(query_pose_pdb1_filepath, remove_alt_location=True)
    #         # query_pose_pdb2 = PDB()
    #         # query_pose_pdb2.readfile(query_pose_pdb2_filepath, remove_alt_location=True)
    #         #
    #         # # standardize oligomer chain lengths such that every 'symmetry related' subunit in an oligomer has the same
    #         # # number of CA atoms and only contains residues (based on residue number) that are present in all
    #         # # 'symmetry related' subunits.
    #         # stand_query_pose_pdb1 = standardize_intra_oligomer_chain_lengths(query_pose_pdb1)
    #         # stand_query_pose_pdb2 = standardize_intra_oligomer_chain_lengths(query_pose_pdb2)
    #         #
    #         # # find correct chain mapping between reference pose and query pose
    #         # # align reference pose CA interface atoms to corresponding mapped CA atoms in the query pose
    #         # # calculate interface CA RMSD between reference pose and query pose
    #         irmsd = map_align_interface_chains(standardized_pdbs1[j], standardized_pdbs2[j],
    #                                            standardized_pdbs1[i], standardized_pdbs2[i],
    #                                            ref_pdb1_int_chids_resnums_dict,
    #                                            ref_pdb2_int_chids_resnums_dict, e=3.0, return_aligned_ref_pdbs=False)
    #
    #         irmsds.append((ref_pose_id, query_pose_id, irmsd))

    return irmsds
    # return zip(*zip(*directory_pairs), irmsds)
########################################################################################################################


def main():
    ############################################## INPUT PARAMETERS ####################################################
    top_scoring = 2000
    docked_poses_dirpath = sys.argv[1]  # nanohedra output directory
    rankfile_path = sys.argv[2]  # path to text file containing: reference structure vs nanohedra poses irmsd values, scores, ranks
    outdir = os.path.dirname(rankfile_path)  # sys.argv[3]  # output directory
    num_threads = 128
    ####################################################################################################################

    # make output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # get docked pose id for top scoring poses from the ranking file
    with open(rankfile_path, 'r') as rankfile:
        top_ranked_ids = []
        for rankfile_line in rankfile.readlines():
            rankfile_line = rankfile_line.rstrip().split()
            pose_id = rankfile_line[0]
            pose_score_rank = int(rankfile_line[3])
            if pose_score_rank <= top_scoring:
                top_ranked_ids.append(pose_id)

    # retrieve all poses and filter for those ID's in consideration
    all_poses, location = SDUtils.collect_designs(directory=docked_poses_dirpath)  # , file=args.file)
    assert all_poses != list(), 'No %s directories found within \'%s\'! Please ensure correct location' \
                                % (PUtils.nano.title(), location)
    all_design_directories = DesignDirectory.set_up_directory_objects(all_poses)  # , symmetry=args.design_string)
    # return only directories for which an id is matched
    top_design_directories = [des_dir for des_dir in all_design_directories if des_dir in top_ranked_ids]  # Todo test
    # top_design_directories = SDUtils.get_pose_by_id(all_design_directories, top_ranked_ids)

    # obtain an irmsd value for all possible pairs of top scoring docked poses
    irmsds = all_to_all_docked_poses_irmsd_mp(top_design_directories, num_threads)  # returns ref_pose_id, query_pose_id, irmsd

    with open(outdir + "/top%s_all_to_all_docked_poses_irmsd.txt" % str(top_scoring), 'w') as outfile:
        # for ref_pose_id, query_pose_id, irmsd in irmsds:
        outfile.write('\n'.join("{:35s} {:35s} {:8.3f}".format(*irmsd) for irmsd in irmsds))


if __name__ == "__main__":
    main()
