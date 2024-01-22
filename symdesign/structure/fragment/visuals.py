import os
from typing import AnyStr

import pandas as pd

from . import GhostFragment
from ..utils import chain_id_generator

idx_slice = pd.IndexSlice


def write_fragment_pairs_as_accumulating_states(
        ghost_frags: list[GhostFragment], file_name: AnyStr = os.getcwd()) -> AnyStr:
    """Write GhostFragments as an MultiModel trajectory where each successive Model has one more Fragment

    Args:
        ghost_frags: An iterable of GhostFragments
        file_name: The path that the file should be written to
    Returns:
        The file_name
    """
    if '.pdb' not in file_name:
        file_name += '.pdb'

    with open(file_name, 'w') as f:
        atom_iterator = 0
        residue_iterator = 1
        chain_generator = chain_id_generator()
        mapped_chain_id = next(chain_generator)
        model_number = 1
        # Write the monofrag that ghost frags are paired against
        f.write(f'MODEL    {model_number:>4d}\n')
        ghost_frag_init = ghost_frags[0]
        frag_model, frag_paired_chain = ghost_frag_init.fragment_db.paired_frags[ghost_frag_init.ijk]
        trnsfmd_fragment = frag_model.get_transformed_copy(*ghost_frag_init.transformation)
        # Set mapped_chain to A
        mapped_chain = trnsfmd_fragment.chain(
            tuple(set(frag_model.chain_ids).difference({frag_paired_chain, '9'}))[0])
        mapped_chain.chain_id = mapped_chain_id
        # Renumber residues
        trnsfmd_fragment.renumber_residues(at=residue_iterator)
        # Write
        f.write('%s\n' % mapped_chain.get_atom_record(atom_offset=atom_iterator))
        # Iterate atom/residue numbers
        atom_iterator += mapped_chain.number_of_atoms
        residue_iterator += mapped_chain.number_of_residues
        f.write('ENDMDL\n')

        # Write all subsequent models, stacking each subsequent model on the previous
        fragment_lines = []
        for model_number, ghost_frag in enumerate(ghost_frags, model_number + 1):
            f.write(f'MODEL    {model_number:>4d}\n')
            frag_model, frag_paired_chain = ghost_frag.fragment_db.paired_frags[ghost_frag.ijk]
            trnsfmd_fragment = frag_model.get_transformed_copy(*ghost_frag.transformation)
            # Iterate only the paired chain with new chainID
            trnsfmd_fragment.chain(frag_paired_chain).chain_id = next(chain_generator)
            # Set the mapped chain to the single mapped_chain_id
            trnsfmd_fragment.chain(tuple(
                set(frag_model.chain_ids).difference({frag_paired_chain, '9'})
            )[0]).chain_id = mapped_chain_id
            trnsfmd_fragment.renumber_residues(at=residue_iterator)
            fragment_lines.append(trnsfmd_fragment.get_atom_record(atom_offset=atom_iterator))
            # trnsfmd_fragment.write(file_handle=f)
            f.write('%s\n' % '\n'.join(fragment_lines))
            atom_iterator += frag_model.number_of_atoms
            residue_iterator += frag_model.number_of_residues
            f.write('ENDMDL\n')

    return file_name


def write_fragments_as_multimodel(ghost_frags: list[GhostFragment], file_name: AnyStr = os.getcwd()) -> AnyStr:
    """Write GhostFragments as one MultiModel

    Args:
        ghost_frags: An iterable of GhostFragments
        file_name: The path that the file should be written to
    Returns:
        The file_name
    """
    if '.pdb' not in file_name:
        file_name += '.pdb'

    with open(file_name, 'w') as f:
        atom_iterator = 0
        residue_iterator = 1
        chain_generator = chain_id_generator()
        mapped_chain_id = next(chain_generator)
        paired_chain_id = next(chain_generator)
        model_number = 1

        # Write all models
        for model_number, ghost_frag in enumerate(ghost_frags, model_number):
            f.write(f'MODEL    {model_number:>4d}\n')
            frag_model, frag_paired_chain = ghost_frag.fragment_db.paired_frags[ghost_frag.ijk]
            trnsfmd_fragment = frag_model.get_transformed_copy(*ghost_frag.transformation)
            # Set the paired chain to the single paired chainID
            trnsfmd_fragment.chain(frag_paired_chain).chain_id = paired_chain_id
            # Set the mapped chain to the single mapped chainID
            trnsfmd_fragment.chain(tuple(
                set(frag_model.chain_ids).difference({frag_paired_chain, '9'})
            )[0]).chain_id = mapped_chain_id
            trnsfmd_fragment.renumber_residues(at=residue_iterator)
            trnsfmd_fragment.write(file_handle=f, atom_offset=atom_iterator)
            atom_iterator += frag_model.number_of_atoms
            residue_iterator += frag_model.number_of_residues
            f.write('ENDMDL\n')

    return file_name
