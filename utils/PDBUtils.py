import warnings

import numpy as np
from sklearn.neighbors import BallTree
import Bio.PDB
from Bio.PDB.Atom import Atom as BioPDBAtom, PDBConstructionWarning
from PDB import PDB
from SymDesignUtils import start_log


# Globals
warnings.simplefilter('ignore', PDBConstructionWarning)
logger = start_log(name=__name__)
# def rot_txint_set_txext_pdb(pdb, rot_mat=None, internal_tx_vec=None, set_mat=None, ext_tx_vec=None):
#     # pdb_coords = np.array(pdb.extract_coords())
#     pdb_coords = np.array(pdb.extract_all_coords())
#
#     if pdb_coords.size != 0:
#
#         # Rotate coordinates if rotation matrix is provided
#         if rot_mat is not None:
#             rot_mat_T = np.transpose(rot_mat)
#             pdb_coords = np.matmul(pdb_coords, rot_mat_T)
#
#         # Translate coordinates if internal translation vector is provided
#         if internal_tx_vec is not None:
#             pdb_coords = pdb_coords + internal_tx_vec
#
#         # Set coordinates if setting matrix is provided
#         if set_mat is not None:
#             set_mat_T = np.transpose(set_mat)
#             pdb_coords = np.matmul(pdb_coords, set_mat_T)
#
#         # Translate coordinates if external translation vector is provided
#         if ext_tx_vec is not None:
#             pdb_coords = pdb_coords + ext_tx_vec
#
#         transformed_pdb = PDB()
#         transformed_atoms = []
#         atom_index = 0
#         for atom in pdb.get_atoms():
#             x_transformed = pdb_coords[atom_index][0]
#             y_transformed = pdb_coords[atom_index][1]
#             z_transformed = pdb_coords[atom_index][2]
#             atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
#                                     atom.get_residue_type(), atom.get_chain(), atom.get_residue_number(),
#                                     atom.get_code_for_insertion(), x_transformed, y_transformed, z_transformed,
#                                     atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
#                                     atom.get_atom_charge())
#             transformed_atoms.append(atom_transformed)
#             atom_index += 1
#
#         transformed_pdb.set_all_atoms(transformed_atoms)
#         transformed_pdb.chain_ids = pdb.chain_ids
#         transformed_pdb.filepath = pdb.filepath
#
#         return transformed_pdb
#
#     else:
#         return []


def get_contacting_asu(pdb1, pdb2, contact_dist=8, **kwargs):
    max_contact_count = 0
    max_contact_chain1, max_contact_chain2 = None, None
    for chain1 in pdb1.chains:
        pdb1_cb_coords_kdtree = BallTree(chain1.get_cb_coords())
        for chain2 in pdb2.chains:
            contact_count = pdb1_cb_coords_kdtree.two_point_correlation(chain2.get_cb_coords(), [contact_dist])[0]

            if contact_count > max_contact_count:
                max_contact_count = contact_count
                max_contact_chain1, max_contact_chain2 = chain1, chain2

    if max_contact_count > 0:
        return PDB.from_chains([max_contact_chain1, max_contact_chain2], name='asu', entities=True, **kwargs)
    else:
        return


def get_interface_residues(pdb1, pdb2, cb_distance=9.0):
    """Calculate all the residues within a cb_distance between two oligomers, identify associated ghost and surface
    fragments on each, by the chain name and residue number, translated the selected fragments to the oligomers using
    symmetry specific rotation matrix, internal translation vector, setting matrix, and external translation vector then
    return copies of these translated fragments

    Returns:
        (tuple[list[tuple], list[tuple]]): interface chain/residues on pdb1, interface chain/residues on pdb2
    """
    pdb1_cb_indices = pdb1.cb_indices
    pdb2_cb_indices = pdb2.cb_indices
    pdb1_coords_indexed_residues = pdb1.coords_indexed_residues
    pdb2_coords_indexed_residues = pdb2.coords_indexed_residues

    pdb1_cb_kdtree = BallTree(pdb1.get_cb_coords())

    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    query = pdb1_cb_kdtree.query_radius(pdb2.get_cb_coords(), cb_distance)

    # Get ResidueNumber, ChainID for all Interacting PDB1 CB, PDB2 CB Pairs
    # interacting_pairs = [(pdb1_residue.number, pdb1_residue.chain, pdb2_residue.number, pdb2_residue.chain)
    #                      for pdb2_query_index, pdb1_query in enumerate(query) for pdb1_query_index in pdb1_query]
    interacting_pairs = []
    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].size > 0:
            # pdb2_atom = pdb2.atoms[pdb2_cb_indices[pdb2_query_index]]
            pdb2_residue = pdb2_coords_indexed_residues[pdb2_cb_indices[pdb2_query_index]]
            # pdb2_cb_chain_id = pdb2.atoms[pdb2_cb_indices[pdb2_query_index]].chain
            for pdb1_query_index in query[pdb2_query_index]:
                # pdb1_atom = pdb1.atoms[pdb1_cb_indices[pdb1_query_index]]
                pdb1_residue = pdb1_coords_indexed_residues[pdb1_cb_indices[pdb1_query_index]]
                # pdb1_cb_res_num = pdb1.atoms[pdb1_cb_indices[pdb1_query_index]].residue_number
                # pdb1_cb_chain_id = pdb1.atoms[pdb1_cb_indices[pdb1_query_index]].chain
                interacting_pairs.append((pdb1_residue.number, pdb1_residue.chain, pdb2_residue.number,
                                          pdb2_residue.chain))

    pdb1_unique_chain_central_resnums, pdb2_unique_chain_central_resnums = [], []
    for pdb1_central_res_num, pdb1_central_chain_id, pdb2_central_res_num, pdb2_central_chain_id in interacting_pairs:
        pdb1_res_num_list = [pdb1_central_res_num + i for i in range(-2, 3)]  # Todo parameterize by frag length
        pdb2_res_num_list = [pdb2_central_res_num + i for i in range(-2, 3)]

        frag1_length = len(pdb1.chain(pdb1_central_chain_id).get_residues(numbers=pdb1_res_num_list))
        frag2_length = len(pdb2.chain(pdb2_central_chain_id).get_residues(numbers=pdb2_res_num_list))

        if frag1_length == 5 and frag2_length == 5:
            if (pdb1_central_chain_id, pdb1_central_res_num) not in pdb1_unique_chain_central_resnums:
                pdb1_unique_chain_central_resnums.append((pdb1_central_chain_id, pdb1_central_res_num))

            if (pdb2_central_chain_id, pdb2_central_res_num) not in pdb2_unique_chain_central_resnums:
                pdb2_unique_chain_central_resnums.append((pdb2_central_chain_id, pdb2_central_res_num))

    return pdb1_unique_chain_central_resnums, pdb2_unique_chain_central_resnums


def biopdb_aligned_chain(pdb_fixed, pdb_moving, chain_id_moving):
    # for atom in pdb_fixed.chain(chain_id_fixed).ca_atoms:
    biopdb_atom_fixed = [BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ, atom.alt_location,
                                    " %s " % atom.type, atom.number, element=atom.element_symbol)
                         for atom in pdb_fixed.ca_atoms]
    biopdb_atom_moving = [BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ, atom.alt_location,
                                     " %s " % atom.type, atom.number, element=atom.element_symbol)
                          for atom in pdb_moving.chain(chain_id_moving).ca_atoms]
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(biopdb_atom_fixed, biopdb_atom_moving)  # Todo remove Bio.PDB
    rot, tr = sup.rotran
    # return np.transpose(rot), tr
    # transpose rotation matrix as Bio.PDB.Superimposer() returns correct matrix to rotate using np.matmul
    return pdb_moving.return_transformed_copy(rotation=np.transpose(rot), translation=tr)


def biopdb_superimposer(atoms_fixed, atoms_moving):
    """

    Args:
        atoms_fixed:
        atoms_moving:

    Returns:
        (tuple[float, numpy.ndarray, numpy.ndarray]): RMSD, Rotation matrix(BioPDB format), Translation vector
    """
    biopdb_atom_fixed = [BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ, atom.alt_location,
                                    " %s " % atom.type, atom.number, element=atom.element_symbol)
                         for atom in atoms_fixed]
    biopdb_atom_moving = [BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ, atom.alt_location,
                                     " %s " % atom.type, atom.number, element=atom.element_symbol)
                          for atom in atoms_moving]

    sup = Bio.PDB.Superimposer()
    sup.set_atoms(biopdb_atom_fixed, biopdb_atom_moving)

    # rmsd = sup.rms
    # rot = np.transpose(sup.rotran[0]).tolist()
    # tx = sup.rotran[1].tolist()

    return (sup.rms, *sup.rotran)
