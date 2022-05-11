import argparse
import os

# try:
#     from Bio.Alphabet import generic_protein  # , IUPAC
# except ImportError:
# from Bio.Align import substitution_matrices
# generic_protein = None

import PathUtils as PUtils
from SymDesignUtils import start_log, filter_dictionary_keys, set_logging_to_debug
from PDB import PDB
from SequenceProfile import position_specific_jsd, weave_mutation_dict, \
    SequenceProfile, jensen_shannon_divergence, rank_possibilities

# Globals
logger = start_log(name=__name__)
db = PUtils.biological_fragmentDB


def calculate_sequence_metrics(des_dir, alignment, residues=None):  # Unused Todo SequenceProfile.py
    if residues:
        keep_residues = residues
        mutation_probabilities = filter_dictionary_keys(alignment.frequencies, keep_residues)
    else:
        mutation_probabilities = alignment.frequencies
    #     design_flags = SDUtils.parse_flags_file(des_dir.path, name='design')
    #     keep_residues = SDUtils.get_interface_residues(design_flags, zero=True)

    ranked_frequencies = rank_possibilities(mutation_probabilities)

    # Calculate Jensen Shannon Divergence from DSSM using the occurrence data in col 2 and design Mutations
    dssm = SequenceProfile.parse_pssm(os.path.join(des_dir.path, PUtils.dssm))
    residue_divergence_values = position_specific_jsd(mutation_probabilities, dssm)

    interface_bkgd = SequenceProfile.get_db_aa_frequencies(db)
    interface_divergence_values = jensen_shannon_divergence(mutation_probabilities, interface_bkgd)

    if os.path.exists(os.path.join(des_dir.path, PUtils.pssm)):
        pssm = SequenceProfile.parse_pssm(os.path.join(des_dir.path, PUtils.pssm))
    else:
        pssm = SequenceProfile.parse_pssm(os.path.join(des_dir.composition, PUtils.pssm))
    evolution_divergence_values = position_specific_jsd(mutation_probabilities, pssm)

    final_mutation_dict = weave_mutation_dict(ranked_frequencies, mutation_probabilities, evolution_divergence_values,
                                              interface_divergence_values)


# def mutate_wildtype_sequences(sequence_dir_files, wild_type_file):  # UNUSED
#     """Take a directory with PDB files and compare to a Wild-type PDB"""
#     wt_seq_dict = get_pdb_sequences(wild_type_file)
#     return generate_sequences(wt_seq_dict, generate_all_design_mutations(sequence_dir_files, wild_type_file))


# def get_pdb_sequences(pdb, chain=None, source='atom'):
#     """Return all sequences or those specified by a chain from a PDB file
#
#     Args:
#         pdb (str or PDB): Location on disk of a reference .pdb file or PDB object
#     Keyword Args:
#         chain=None (str): If a particular chain is desired, specify it
#         source='atom' (str): One of 'atom' or 'seqres'
#     Returns:
#         (dict): {chain: sequence, ...}
#     """
#     print('get_pdb_sequences is using pdb parameter: %s' % pdb)
#     if not isinstance(pdb, PDB):
#         pdb = PDB.from_file(pdb, log=start_log(handler=3), entities=False)
#
#     if source == 'atom':
#         seq_dict = pdb.atom_sequences
#     else:
#         seq_dict = pdb.reference_sequence
#     # for _chain in pdb.chain_ids:
#     #     seq_dict[_chain] =
#     if chain:
#         seq_dict = filter_dictionary_keys(seq_dict, chain)
#
#     return seq_dict


def find_gapped_columns(alignment_dict):  # UNUSED
    target_seq_index = []
    n = 1
    for aa in alignment_dict['meta']['query']:
        if aa != '-':
            target_seq_index.append(n)
        n += 1

    return target_seq_index


def update_alignment_meta(alignment_dict):  # UNUSED UNFINISHED
    all_meta = []
    for alignment in alignment_dict:
        all_meta.append(alignment_dict[alignment]['meta'])

    meta_strings = ['' for i in range(len(next(all_meta)))]
    for meta in all_meta:
        j = 0
        for data in meta:
            meta_strings[j] += meta[data]

    return alignment_dict


def modify_index(count_dict, index_start=0):  # UNUSED NOT Working
    return {i + index_start: count_dict[i] for i in count_dict}


def modify_alignment_dict_index(alignment_dict, index=0):  # UNUSED UNFINISHED
    alignment_dict['counts'] = modify_index(alignment_dict['counts'], index_start=index)
    alignment_dict['rep'] = modify_index(alignment_dict['rep'], index_start=index)

    return alignment_dict


def merge_alignment_dicts(alignment_merge):  # UNUSED UNFINISHED
    length = [0]
    for i, alignment in enumerate(alignment_merge):
        alignment_dict = modify_alignment_dict_index(alignment_merge[alignment], index=length[i])
        length.append(len(alignment_merge[alignment]['meta']['query']))
        merged_alignment_dict = {'meta': update_alignment_meta(alignment)} # alignment_dict
    for alignment in alignment_merge:
        merged_alignment_dict.update(alignment_merge[alignment])

    return merged_alignment_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='%s\nAnalyze mutations compared to a wild_type protein. Requires a '
                                                 'directory with \'mutated\' PDB files and a wild-type PDB reference.'
                                                 % __name__)
    parser.add_argument('-d', '--directory', type=str, help='Where is the design PDB directory located?',
                        default=os.getcwd())
    parser.add_argument('-w', '--wildtype', type=str, help='Where is the wild-type PDB located?', default=None)
    parser.add_argument('-p', '--print', action='store_true', help='Print the output the the console? Default=False')
    parser.add_argument('-s', '--score', type=str, help='Where is the score file located?', default=None)
    parser.add_argument('--debug', action='store_true', help='Debug all steps to standard out? Default=False')

    args = parser.parse_args()
    # Start logging output
    if args.debug:
        logger = start_log(name='main', level=1)
        set_logging_to_debug()
        logger.debug('Debug mode. Produces verbose output and not written to any .log files')
    else:
        logger = start_log(name='main', propagate=True)

    logger.info('Starting %s with options:\n%s' %
                (__name__, '\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    design_directory = DesignDirectory.DesignDirectory(args.directory)

    logger.warning('If you are running into issues with locating files, the problem is not you, it is me. '
                   'I have limited capacity to locate specific files given the scope of my creation.')
    if os.path.basename(args.directory).startswith('tx_'):
        logger.info('Design directory specified, using standard method and disregarding additional inputs '
                    '(-s, -score) and (-w, --wildtype).')
        analyze_mutations(design_directory, mutate_wildtype_sequences(args.directory, args.wildtype),
                          print_results=True)  # args.print)
    else:
        if args.directory and args.wildtype and args.score:
            path_object = DesignDirectory.set_up_pseudo_design_dir(args.wildtype, args.directory, args.score)
            analyze_mutations(design_directory, mutate_wildtype_sequences(args.directory, args.wildtype),
                              print_results=True)  # args.print)
        else:
            logger.critical('Must pass all three, wildtype, directory, and score if using non-standard %s '
                            'directory structure' % PUtils.program_name)
            exit()


