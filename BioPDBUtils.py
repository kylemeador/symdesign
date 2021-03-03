import warnings

import Bio.PDB.Superimposer
import numpy as np
from Bio.PDB.Atom import Atom as BioPDBAtom
from Bio.PDB.Atom import PDBConstructionWarning
from PDB import PDB
from Structure import Atom

warnings.simplefilter('ignore', PDBConstructionWarning)

# OLD


def biopdb_aligned_chain_old(pdb_fixed, chain_id_fixed, pdb_moving, chain_id_moving):
    biopdb_atom_fixed = []
    biopdb_atom_moving = []

    for atom in pdb_fixed.get_ca_atoms():
        if atom.chain == chain_id_fixed:
            biopdb_atom_fixed.append(
                BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ, atom.alt_location,
                           " %s " % atom.type, atom.number, element=atom.element_symbol))

    # pdb_moving_coords = []
    # for atom in pdb_moving.atoms():
    for atom in pdb_moving.get_ca_atoms():
        # pdb_moving_coords.append([atom.coords])
        # pdb_moving_coords.append([atom.get_x(), atom.get_y(), atom.get_z()])
        # if atom.is_CA():
        if atom.chain == chain_id_moving:
            biopdb_atom_moving.append(BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ,
                                                 atom.alt_location, " %s " % atom.type, atom.number,
                                                 element=atom.element_symbol))

    sup = Bio.PDB.Superimposer()
    sup.set_atoms(biopdb_atom_fixed, biopdb_atom_moving)
    # no need to transpose rotation matrix as Bio.PDB.Superimposer() generates correct matrix to rotate using np.matmul
    rot, tr = sup.rotran[0], sup.rotran[1]

    pdb_moving_coords_rot = np.matmul(pdb_moving.coords, rot)
    pdb_moving_coords_rot_tx = pdb_moving_coords_rot + tr

    # pdb_moving_copy = PDB()
    # pdb_moving_copy.set_chain_id_list(pdb_moving.get_chain_id_list())
    pdb_moving_copy_atom_list = []
    # atom_count = 0
    for atom in pdb_moving.atoms():
        # x_transformed = pdb_moving_coords_rot_tx[atom_count][0]
        # y_transformed = pdb_moving_coords_rot_tx[atom_count][1]
        # z_transformed = pdb_moving_coords_rot_tx[atom_count][2]
        atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
                                atom.get_residue_type(), atom.get_chain(),
                                atom.get_residue_number(),
                                atom.get_code_for_insertion(),  # x_transformed, y_transformed, z_transformed,
                                atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                atom.get_atom_charge())
        pdb_moving_copy_atom_list.append(atom_transformed)
        # atom_count += 1

    pdb_moving_copy = PDB.from_atoms(atoms=pdb_moving_copy_atom_list, coords=pdb_moving_coords_rot_tx)

    return pdb_moving_copy


# def biopdb_aligned_chain(pdb_fixed, chain_id_fixed, pdb_moving, chain_id_moving):
def biopdb_aligned_chain(pdb_fixed, pdb_moving, chain_id_moving):
    # for atom in pdb_fixed.chain(chain_id_fixed).get_ca_atoms():
    biopdb_atom_fixed = [BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ, atom.alt_location,
                                    " %s " % atom.type, atom.number, element=atom.element_symbol)
                         for atom in pdb_fixed.get_ca_atoms()]
    biopdb_atom_moving = [BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ, atom.alt_location,
                                     " %s " % atom.type, atom.number, element=atom.element_symbol)
                          for atom in pdb_moving.chain(chain_id_moving).get_ca_atoms()]
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(biopdb_atom_fixed, biopdb_atom_moving)  # Todo remove Bio.PDB
    rot, tr = sup.rotran
    # return np.transpose(rot), tr
    # transpose rotation matrix as Bio.PDB.Superimposer() returns correct matrix to rotate using np.matmul
    return pdb_moving.return_transformed_copy(rotation=np.transpose(rot), translation=tr)


# def biopdb_aligned_chain(pdb_fixed, chain_id_fixed, pdb_moving, chain_id_moving):
def biopdb_align_atom_lists(atoms_fixed, atoms_moving):
    # for atom in pdb_fixed.chain(chain_id_fixed).get_ca_atoms():
    biopdb_atom_fixed = [BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ, atom.alt_location,
                                    " %s " % atom.type, atom.number, element=atom.element_symbol)
                         for atom in atoms_fixed]
    biopdb_atom_moving = [BioPDBAtom(atom.type, (atom.x, atom.y, atom.z), atom.temp_fact, atom.occ, atom.alt_location,
                                     " %s " % atom.type, atom.number, element=atom.element_symbol)
                          for atom in atoms_moving]
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(biopdb_atom_fixed, biopdb_atom_moving)  # Todo remove Bio.PDB
    rot, tr = sup.rotran
    # return np.transpose(rot), tr
    # transpose rotation matrix as Bio.PDB.Superimposer() returns correct matrix to rotate using np.matmul
    return np.transpose(rot), tr
    # return pdb_moving.return_transformed_copy(rotation=np.transpose(rot), translation=tr)


def biopdb_superimposer(atoms_fixed, atoms_moving):
    """

    Args:
        atoms_fixed:
        atoms_moving:

    Returns:
        (tuple[float, numpy.ndarray, numpy.ndarray])
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
