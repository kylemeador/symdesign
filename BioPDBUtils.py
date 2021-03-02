import warnings

import Bio.PDB.Superimposer
import numpy as np
from Bio.PDB.Atom import Atom as BioPDBAtom
from Bio.PDB.Atom import PDBConstructionWarning

warnings.simplefilter('ignore', PDBConstructionWarning)


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
