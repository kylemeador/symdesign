import os
from typing import AnyStr

import pandas as pd
# import seaborn as sns

from symdesign import structure

idx_slice = pd.IndexSlice


def write_fragment_pairs_as_accumulating_states(ghost_frags: list[structure.fragment.GhostFragment],
                                                file_name: AnyStr = os.getcwd()) -> AnyStr:
    """Write GhostFragments as an MultiModel Structure trajectory where each successive Model has one more Fragment

    Args:
        ghost_frags: An iterable of GhostFragments
        file_name: The desired filename
    Returns:
        The file_name
    """
    if '.pdb' not in file_name:
        file_name += '.pdb'

    with open(file_name, 'w') as f:
        atom_iterator = 0
        residue_iterator = 1
        chain_generator = structure.utils.chain_id_generator()
        mapped_chain_id = next(chain_generator)
        model_number = 1
        # Write the monofrag that ghost frags are paired against
        f.write(f'MODEL    {model_number:>4d}\n')
        ghost_frag_init = ghost_frags[0]
        frag_model, frag_paired_chain = ghost_frag_init.fragment_db.paired_frags[ghost_frag_init.ijk]
        trnsfmd_fragment = frag_model.get_transformed_copy(*ghost_frag_init.transformation)
        # Set mapped_chain to A
        mapped_chain = trnsfmd_fragment.chain(tuple(set(frag_model.chain_ids)
                                                    .difference({frag_paired_chain, '9'}))[0])
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
            trnsfmd_fragment.chain(tuple(set(frag_model.chain_ids).difference({frag_paired_chain, '9'})
                                         )[0]).chain_id = mapped_chain_id
            trnsfmd_fragment.renumber_residues(at=residue_iterator)
            fragment_lines.append(trnsfmd_fragment.get_atom_record(atom_offset=atom_iterator))
            # trnsfmd_fragment.write(file_handle=f)
            f.write('%s\n' % '\n'.join(fragment_lines))
            atom_iterator += frag_model.number_of_atoms
            residue_iterator += frag_model.number_of_residues
            f.write('ENDMDL\n')


def write_fragments_as_multimodel(ghost_frags: list[structure.fragment.GhostFragment],
                                  file_name: AnyStr = os.getcwd()) -> AnyStr:
    """Write GhostFragments as one MultiModel Structure

    Args:
        ghost_frags: An iterable of GhostFragments
        file_name: The desired filename
    Returns:
        The file_name
    """
    if '.pdb' not in file_name:
        file_name += '.pdb'

    with open(file_name, 'w') as f:
        atom_iterator = 0
        residue_iterator = 1
        chain_generator = structure.utils.chain_id_generator()
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
            trnsfmd_fragment.chain(tuple(set(frag_model.chain_ids).difference({frag_paired_chain, '9'})
                                         )[0]).chain_id = mapped_chain_id
            trnsfmd_fragment.renumber_residues(at=residue_iterator)
            trnsfmd_fragment.write(file_handle=f, atom_offset=atom_iterator)
            atom_iterator += frag_model.number_of_atoms
            residue_iterator += frag_model.number_of_residues
            f.write('ENDMDL\n')


metrics_of_interest = [
    'proteinmpnn_score_designed_delta',
    'proteinmpnn_score_designed_complex',
    'proteinmpnn_score_designed_unbound',
    'proteinmpnn_v_fragment_cross_entropy_designed_mean',
    'proteinmpnn_v_evolution_cross_entropy_designed_mean',
    'proteinmpnn_loss_evolution',
    'proteinmpnn_loss_fragment',
    'designed_residues_total',
    'collapse_violation_design_residues',
    'nanohedra_score_normalized',
    'interface_b_factor',
    'percent_residues_fragment_center',
    'interface_energy',
    'multiple_fragment_ratio',
    'number_fragments_interface',
    'percent_mutations',  # 'number_mutations',
]


# def plot_df_correlation(df: pd.DataFrame, metrics_of_interest: Sequence[str]):
#     """From a DataFrame, plot the correlation between all points in the DataFrame for selected metrics
#
#     Args:
#         df: This is assumed to be a three column dataframe with 'pose' and 'dock' present in levels 0 and 1
#         metrics_of_interest: The metrics one is interested in correlating
#     Returns:
#         None
#     """
#     _ = sns.pairplot(df.loc[:, idx_slice['pose', 'dock', metrics_of_interest]]
#                      .droplevel(0, axis=1).droplevel(0, axis=1), kind='reg', diag_kind='kde')


if __name__ == '__main__':
    current_file = '/home/kylemeador/T33-Good-2gtr-3m6n_docked_poses_Trajectories.csv'
    nano_df = pd.read_csv(current_file, header=[0, 1, 2])
    metrics_of_interest = [
        # 'proteinmpnn_score_designed_delta',
        'proteinmpnn_score_designed_complex',
        # 'proteinmpnn_score_designed_unbound',
        'proteinmpnn_v_fragment_cross_entropy_designed_mean',
        'proteinmpnn_v_evolution_cross_entropy_designed_mean',
        'proteinmpnn_loss_evolution',
        'proteinmpnn_loss_fragment',
        'designed_residues_total',
        # 'collapse_violation_design_residues',
        'nanohedra_score_normalized',
        # 'interface_b_factor',
        'percent_residues_fragment_center',
        # 'interface_energy',
        'multiple_fragment_ratio',
        'number_fragments_interface',
        # 'percent_mutations',  # 'number_mutations',
    ]
    plot_df_correlation(nano_df, metrics_of_interest)