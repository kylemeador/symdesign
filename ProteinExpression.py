"""Add expression tags onto the termini of specific designs"""
import csv
# from itertools import chain as iter_chain  # combinations,
import numpy as np
from Bio.Data.IUPACData import protein_letters

import PathUtils as PUtils
import SymDesignUtils as SDUtils
# from PDB import PDB
# import Pose
import SequenceProfile
from Query.PDB import get_entity_reference_sequence, pdb_id_matching_uniprot_id
from Query.utils import input_string
from dependencies.DnaChisel.dnachisel import DnaOptimizationProblem, CodonOptimize, reverse_translate, AvoidHairpins, \
    EnforceGCContent, AvoidPattern, AvoidRareCodons, UniquifyAllKmers, EnforceTranslation, EnforceMeltingTemperature

# Globals
logger = SDUtils.start_log(name=__name__)
uniprot_pdb_d = SDUtils.unpickle(PUtils.uniprot_pdb_map)
with open(PUtils.affinity_tags, 'r') as f:
    expression_tags = {'_'.join(map(str.lower, row[0].split())): row[1] for row in csv.reader(f)}
ndeI_multicistronic_sequence = \
    'taatgcttaagtcgaacagaaagtaatcgtattgtacacggccgcataatcgaaat' \
    'taatacgactcactataggggaattgtgagcggataacaattccccatcttagtatattagttaagtataagaaggagatatacat'  # ATG
#    ^ Start of T7 promoter                   ^ last nucleotide of LacO         ^ S-D stop        ^ NdeI end
#                       ^^ start of LacO                                 ^ Shine Dalgarno start
#                                                                                       ^ NdeI start
ncoI_multicistronic_sequence = \
    'taatgcttaagtcgaacagaaagtaatcgtattgtacacggccgcataatcgaaat' \
    'taatacgactcactataggggaattgtgagcggataacaattccccatcttagtatattagttaagtataagaaggagatatacc'  # ATGG
#    ^ Start of T7 promoter                   ^ last nucleotide of LacO         ^ S-D stop        ^ NcoI end
#                       ^^ start of LacO                                 ^ Shine Dalgarno start
#                                                                                       ^ NcoI start
default_multicistronic_sequence = ncoI_multicistronic_sequence
# Retrieved from https://www.chem.ucalgary.ca/courses/351/Carey5th/Ch27/ch27-1-4-2.html 12/22/21
isoelectric_point_table = \
    {'A': {'c': 2.34, 'n': 9.69, 'sc': None, 'pi': 6.00},
     'C': {'c': 1.96, 'n': 8.18, 'sc': None, 'pi': 5.07},
     'D': {'c': 1.88, 'n': 9.60, 'sc': 3.65, 'pi': 2.77},
     'E': {'c': 2.19, 'n': 9.67, 'sc': 4.25, 'pi': 3.22},
     'F': {'c': 1.83, 'n': 9.13, 'sc': None, 'pi': 5.48},
     'G': {'c': 2.34, 'n': 9.60, 'sc': None, 'pi': 5.97},
     'H': {'c': 1.82, 'n': 9.17, 'sc': 6.00, 'pi': 7.59},
     'I': {'c': 2.36, 'n': 9.60, 'sc': None, 'pi': 6.02},
     'K': {'c': 2.18, 'n': 8.95, 'sc': 10.53, 'pi': 9.74},
     'L': {'c': 2.36, 'n': 9.60, 'sc': None, 'pi': 5.98},
     'M': {'c': 2.28, 'n': 9.21, 'sc': None, 'pi': 5.74},
     'N': {'c': 2.02, 'n': 8.80, 'sc': None, 'pi': 5.41},
     'P': {'c': 1.99, 'n': 10.60, 'sc': None, 'pi': 6.30},
     'Q': {'c': 2.17, 'n': 9.13, 'sc': None, 'pi': 5.65},
     'R': {'c': 2.17, 'n': 9.04, 'sc': 12.48, 'pi': 10.76},
     'S': {'c': 2.21, 'n': 9.15, 'sc': None, 'pi': 5.68},
     'T': {'c': 2.09, 'n': 9.10, 'sc': None, 'pi': 5.60},
     'V': {'c': 2.32, 'n': 9.62, 'sc': None, 'pi': 5.96},
     'W': {'c': 2.83, 'n': 9.39, 'sc': None, 'pi': 5.89},
     'Y': {'c': 2.20, 'n': 9.11, 'sc': None, 'pi': 5.66}}
# these values are for individual amino acids (benchling), subtract h20_mass to get weight in a polymer
# aa_molecular_weights = \
#     {'A': 89.09,
#      'B': 132.65,
#      'C': 121.15,
#      'D': 133.1,
#      'E': 147.13,
#      'F': 165.19,
#      'G': 75.07,
#      'H': 155.16,
#      'I': 131.17,
#      'J': 131.2,
#      'K': 146.19,
#      'L': 131.17,
#      'M': 149.21,
#      'N': 132.12,
#      'O': 255.31,
#      'P': 115.13,
#      'Q': 146.15,
#      'R': 174.2,
#      'S': 105.09,
#      'T': 119.12,
#      'U': 168.06,
#      'V': 117.15,
#      'W': 204.23,
#      'X': 143.7,
#      'Y': 181.19,
#      'Z': 146.75}
aa_molecular_weights = np.array(
    [89.09,  # A
     121.15,  # C
     133.1,  # D
     147.13,  # E
     165.19,  # F
     75.07,  # G
     155.16,  # H
     131.17,  # I
     146.19,  # K
     131.17,  # L
     149.21,  # M
     132.12,  # N
     115.13,  # P
     146.15,  # Q
     174.2,  # R
     105.09,  # S
     119.12,  # T
     117.15,  # V
     204.23,  # W
     181.19])  # y
# these values are for amino acids in a polypeptide. Add the weight of one water molecule to get the correct mass
# average used here  monoisotopic    average isotopic
h2o_mass = 18.0152
# aa_polymer_molecular_weights = \
#     {'A': 71.0788,   #   71.03711  	    71.0788
#      'B': 114.6348,  # calculated from individual values above
#      'C': 103.1388,  #   103.00919  	103.1388
#      'D': 115.0886,  # 	 115.02694  	115.0886
#      'E': 129.1155,  # 	 129.04259  	129.1155
#      'F': 147.1766,  # 	 147.06841  	147.1766
#      'G': 57.0519,   # 	 57.02146  	    57.0519
#      'H': 137.1411,  # 	 137.05891  	137.1411
#      'I': 113.1594,  # 	 113.08406  	113.1594
#      'J': 113.1848,  # calculated from individual values above
#      'K': 128.1741,  # 	 128.09496  	128.1741
#      'L': 113.1594,  # 	 113.08406  	113.1594
#      'M': 131.1926,  # 	 131.04049  	131.1926
#      'N': 114.1038,  #   114.04293  	114.1038
#      'O': 237.3018,  #   237.147727  	237.3018
#      'P': 97.1167,   # 	 97.05276  	    97.1167
#      'Q': 128.1307,  # 	 128.05858  	128.1307
#      'R': 156.1875,  # 	 156.10111  	156.1875
#      'S': 87.0782,   # 	 87.03203  	    87.0782
#      'T': 101.1051,  # 	 101.04768  	101.1051
#      'U': 150.0388,  #   150.953636  	150.0388
#      'V': 99.1326,   #   99.06841  	    99.1326
#      'W': 186.2132,  # 	 186.07931  	186.2132
#      'X': 125.6848,  # calculated from individual values above
#      'Y': 163.1760,  #   163.06333  	163.1760
#      'Z': 128.7348}  # calculated from individual values above

# aa 1 letter alphabetical order
# average used here  monoisotopic    average isotopic
aa_polymer_molecular_weights = np.array(
    [71.0788,  # A  71.03711    71.0788
     103.1388,  # C  103.00919   103.1388
     115.0886,  # D  115.02694   115.0886
     129.1155,  # E  129.04259   129.1155
     147.1766,  # F  147.06841   147.1766
     57.0519,   # G  57.02146    57.0519
     137.1411,  # H  137.05891   137.1411
     113.1594,  # I  113.08406   113.1594
     128.1741,  # K  128.09496   128.1741
     113.1594,  # L  113.08406   113.1594
     131.1926,  # M  131.04049   131.1926
     114.1038,  # N  114.04293   114.1038
     97.1167,   # P  97.05276    97.1167
     128.1307,  # Q  128.05858   128.1307
     156.1875,  # R  156.10111   156.1875
     87.0782,   # S  87.03203    87.0782
     101.1051,  # T  101.04768   101.1051
     99.1326,   # V  99.06841    99.1326
     186.2132,  # W  186.07931   186.2132
     163.1760])  # Y  163.06333   163.1760


def calculate_protein_molecular_weight(sequence):
    sequence = sequence.upper()
    # weight = 0.
    # for aa in sequence:
    #     weight += aa_polymer_molecular_weights[aa]
    #
    # return weight + h2o_mass

    # use this to input all sequences to SequenceProfile. This will form the basis for all sequence handling by array
    seq_index = [instability_order.index(aa) for aa in sequence]
    return aa_polymer_molecular_weights[seq_index].sum() + h2o_mass


#  biojava version
# order = ['W', 'C', 'M', 'H', 'Y', 'F', 'Q', 'N', 'I', 'R', 'D', 'P', 'T', 'K', 'E', 'V', 'S', 'G', 'A', 'L']
# instability_array = \
#     [[1., 1., 24.68, 24.68, 1., 1., 1., 13.34, 1., 1., 1., 1., -14.03, 1., 1., -7.49, 1., -9.37, -14.03, 13.34],
#      [24.68, 1., 33.6, 33.6, 1., 1., -6.54, 1., 1., 1., 20.26, 20.26, 33.6, 1., 1., -6.54, 1., 1., 1., 20.26],
#      [1., 1., -1.88, 58.28, 24.68, 1., -6.54, 1., 1., -6.54, 1., 44.94, -1.88, 1., 1., 1., 44.94, 1., 13.34, 1.],
#      [-1.88, 1., 1., 1., 44.94, -9.37, 1., 24.68, 44.94, 1., 1., -1.88, -6.54, 24.68, 1., 1., 1., -9.37, 1., 1.],
#      [-9.37, 1., 44.94, 13.34, 13.34, 1., 1., 1., 1., -15.91, 24.68, 13.34, -7.49, 1., -6.54, 1., 1., -7.49, 24.68, 1.],
#      [1., 1., 1., 1., 33.6, 1., 1., 1., 1., 1., 13.34, 20.26, 1., -14.03, 1., 1., 1., 1., 1., 1.],
#      [1., -6.54, 1., 1., -6.54, -6.54, 20.26, 1., 1., 1., 20.26, 20.26, 1., 1., 20.26, -6.54, 44.94, 1., 1., 1.],
#      [-9.37, -1.88, 1., 1., 1., -14.03, -6.54, 1., 44.94, 1., 1., -1.88, -7.49, 24.68, 1., 1., 1., -14.03, 1., 1.],
#      [1., 1., 1., 13.34, 1., 1., 1., 1., 1., 1., 1., -1.88, 1., -7.49, 44.94, -7.49, 1., 1., 1., 20.26],
#      [58.28, 1., 1., 20.26, -6.54, 1., 20.26, 13.34, 1., 58.28, 1., 20.26, 1., 1., 1., 1., 44.94, -7.49, 1., 1.],
#      [1., 1., 1., 1., 1., -6.54, 1., 1., 1., -6.54, 1., 1., -14.03, -7.49, 1., 1., 20.26, 1., 1., 1.],
#      [-1.88, -6.54, -6.54, 1., 1., 20.26, 20.26, 1., 1., -6.54, -6.54, 20.26, 1., 1., 18.38, 20.26, 20.26, 1., 20.26, 1.],
#      [-14.03, 1., 1., 1., 1., 13.34, -6.54, -14.03, 1., 1., 1., 1., 1., 1., 20.26, 1., 1., -7.49, 1., 1.],
#      [1., 1., 33.6, 1., 1., 1., 24.68, 1., -7.49, 33.6, 1., -6.54, 1., 1., 1., -7.49, 1., -7.49, 1., -7.49],
#      [-14.03, 44.94, 1., -6.54, 1., 1., 20.26, 1., 20.26, 1., 20.26, 20.26, 1., 1., 33.6, 1., 20.26, 1., 1., 1.],
#      [1., 1., 1., 1., -6.54, 1., 1., 1., 1., 1., -14.03, 20.26, -7.49, -1.88, 1., 1., 1., -7.49, 1., 1.],
#      [1., 33.6, 1., 1., 1., 1., 20.26, 1., 1., 20.26, 1., 44.94, 1., 1., 20.26, 1., 20.26, 1., 1., 1.],
#      [13.34, 1., 1., 1., -7.49, 1., 1., -7.49, -7.49, 1., 1., 1., -7.49, -7.49, -6.54, 1., 1., 13.34, -7.49, 1.],
#      [1., 44.94, 1., -7.49, 1., 1., 1., 1., 1., 1., -7.49, 20.26, 1., 1., 1., 1., 1., 1., 1., 1.],
#      [24.68, 1., 1., 1., 1., 1., 33.6, 1., 1., 20.26, 1., 20.26, 1., -7.49, 1., 1., 1., 1., 1., 1.]]
# 1. 24.68 1. -14.03 1. 13.34 -1.88 <- W test when reordered as below

# derived from
# Guruprasad, K., Reddy, B.V.B. and Pandit, M.W. (1990)
# Correlation between stability of a protein and its dipeptide composition: a novel approach for predicting in vivo
# stability of a protein from its primary sequence.
# Protein Eng. 4,155-161. Table III.
# KM reorganization according to aa 1 letter alphabetical
instability_order = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
instability_array = np.array(
    [[1., 44.94, -7.49, 1., 1., 1., -7.49, 1., 1., 1., 1., 1., 20.26, 1., 1., 1., 1., 1., 1., 1.], 
     [1., 1., 20.26, 1., 1., 1., 33.6, 1., 1., 20.26, 33.6, 1., 20.26, -6.54, 1., 1., 33.6, -6.54, 24.68, 1.], 
     [1., 1., 1., 1., -6.54, 1., 1., 1., -7.49, 1., 1., 1., 1., 1., -6.54, 20.26, -14.03, 1., 1., 1.], 
     [1., 44.94, 20.26, 33.6, 1., 1., -6.54, 20.26, 1., 1., 1., 1., 20.26, 20.26, 1., 20.26, 1., 1., -14.03, 1.], 
     [1., 1., 13.34, 1., 1., 1., 1., 1., -14.03, 1., 1., 1., 20.26, 1., 1., 1., 1., 1., 1., 33.6], 
     [-7.49, 1., 1., -6.54, 1., 13.34, 1., -7.49, -7.49, 1., 1., -7.49, 1., 1., 1., 1., -7.49, 1., 13.34, -7.49], 
     [1., 1., 1., 1., -9.37, -9.37, 1., 44.94, 24.68, 1., 1., 24.68, -1.88, 1., 1., 1., -6.54, 1., -1.88, 44.94], 
     [1., 1., 1., 44.94, 1., 1., 13.34, 1., -7.49, 20.26, 1., 1., -1.88, 1., 1., 1., 1., -7.49, 1., 1.], 
     [1., 1., 1., 1., 1., -7.49, 1., -7.49, 1., -7.49, 33.6, 1., -6.54, 24.68, 33.6, 1., 1., -7.49, 1., 1.], 
     [1., 1., 1., 1., 1., 1., 1., 1., -7.49, 1., 1., 1., 20.26, 33.6, 20.26, 1., 1., 1., 24.68, 1.], 
     [13.34, 1., 1., 1., 1., 1., 58.28, 1., 1., 1., -1.88, 1., 44.94, -6.54, -6.54, 44.94, -1.88, 1., 1., 24.68], 
     [1., -1.88, 1., 1., -14.03, -14.03, 1., 44.94, 24.68, 1., 1., 1., -1.88, -6.54, 1., 1., -7.49, 1., -9.37, 1.], 
     [20.26, -6.54, -6.54, 18.38, 20.26, 1., 1., 1., 1., 1., -6.54, 1., 20.26, 20.26, -6.54, 20.26, 1., 20.26, -1.88, 1.], 
     [1., -6.54, 20.26, 20.26, -6.54, 1., 1., 1., 1., 1., 1., 1., 20.26, 20.26, 1., 44.94, 1., -6.54, 1., -6.54], 
     [1., 1., 1., 1., 1., -7.49, 20.26, 1., 1., 1., 1., 13.34, 20.26, 20.26, 58.28, 44.94, 1., 1., 58.28, -6.54], 
     [1., 33.6, 1., 20.26, 1., 1., 1., 1., 1., 1., 1., 1., 44.94, 20.26, 20.26, 20.26, 1., 1., 1., 1.], 
     [1., 1., 1., 20.26, 13.34, -7.49, 1., 1., 1., 1., 1., -14.03, 1., -6.54, 1., 1., 1., 1., -14.03, 1.],
     [1., 1., -14.03, 1., 1., -7.49, 1., 1., -1.88, 1., 1., 1., 20.26, 1., 1., 1., -7.49, 1., 1., -6.54], 
     [-14.03, 1., 1., 1., 1., -9.37, 24.68, 1., 1., 13.34, 24.68, 13.34, 1., 1., 1., 1., -14.03, -7.49, 1., 1.], 
     [24.68, 1., 24.68, -6.54, 1., -7.49, 13.34, 1., 1., 1., 44.94, 1., 13.34, 1., -15.91, 1., -7.49, 1., -9.37, 13.34]])


def calculate_instability_index(sequence):
    sequence = sequence.upper()
    # dipeptide_stability_sum = 0.
    # for idx, aa in enumerate(sequence[:-2], 1):  # only want to iterate until the second to last amino acid
    #     # get the current amino acid and the next amino acid
    #     dipeptide_stability_sum += instability_array[instability_order.index(aa)][instability_order.index(sequence[idx])]
    #
    # return dipeptide_stability_sum

    # index1, index2 = [], []
    # for idx, aa in enumerate(sequence[:-2], 1):  # only want to iterate until the second to last amino acid
    #     index1.append(instability_order.index(aa))
    #     index2.append(instability_order.index(sequence[idx]))
    # return instability_array[index1, index2].sum()

    # index_pairs = list(zip(*((instability_order.index(aa), instability_order.index(sequence[idx])) for idx, aa in
    #                          enumerate(sequence[:-1], 1))))
    # return instability_array[index_pairs].sum()

    # use this to input all sequences to SequenceProfile. This will form the basis for all sequence handling by array
    seq_index = [instability_order.index(aa) for aa in sequence]
    return instability_array[seq_index[:-1], seq_index[1:]].sum()


# E(Prot) = Numb(Tyr) * Ext(Tyr) + Numb(Trp) * Ext(Trp) + Numb(Cystine) * Ext(Cystine)
# where(for proteins in water measured at 280 nm): Ext(Tyr) = 1490, Ext(Trp) = 5500, Ext(Cystine) = 125

def find_matching_expression_tags(uniprot_id=None, pdb_code=None, chain=None):
    """Take a pose and find expression tags from each PDB reference asking user for input on tag choice

    Args:
        uniprot_id=None (str): The uniprot_id to query tags from
        pdb_code=None (str): The pdb to query tags from. Requires chain argument as well
        chain=None (str): The chain to query tags from. Requires pdb argument as well
    Returns:
        (list[dict]): [{'name': 'his_tag', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]
    """
    #         (dict): {'n': {His Tag: 2}, 'c': {Spy Catcher: 1},
    #                  'matching_tags': [{'name': 'his_tag', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]}
    matching_pdb_tags = []
    if not uniprot_id:
        if not pdb_code or not chain:
            # raise AttributeError('One of uniprot_id or pdb_code AND chain is required')
            logger.error('One of uniprot_id OR pdb_code AND chain is required')
            return matching_pdb_tags
        uniprot_id = pull_uniprot_id_by_pdb(uniprot_pdb_d, pdb_code, chain=chain)

    # from PDB API
    partner_sequences = [get_entity_reference_sequence(entity_id=entity_id)
                         for entity_id in pdb_id_matching_uniprot_id(uniprot_id=uniprot_id)]
    # # from internal data storage
    # if uniprot_id not in uniprot_pdb_d:
    #     return {'name': None, 'seq': None}
    #     # return AnalyzeMutatedSequences.get_pdb_sequences(Pose.retrieve_pdb_file_path(pdb_code), chain=chain,
    #     #                                                  source='seqres')
    #
    # # {pdb: [{'A': 'MSGHHHHHHGKLKPNDLRI...'}, ...], ...}
    # # pdb_chain_d = {}
    # partner_sequences = []  # v in this dictionary 'all' gives PDB.Chain, 'unique' gives only PDB handle
    # for matching_pdb_chain in uniprot_pdb_d[uniprot_id]['all']:
    #     matching_pdb, chain = matching_pdb_chain.split('.')
    #     # pdb_chain_d[matching_pdb] = chain  # This is essentially a set as duplicates are overwritten
    # # # for matching_pdb, chain in pdb_chain_d.items():
    # #     partner_pdb = PDB.from_file(Pose.fetch_pdb_file(matching_pdb), log=None, pose_format=False, entities=False)
    # #     # partner_d = AnalyzeMutatedSequences.get_pdb_sequences(Pose.retrieve_pdb_file_path(matching_pdb),
    # #     #                                                       chain=pdb_chain_d[matching_pdb], source='seqres')
    # #     partner_sequences.append(partner_pdb.reference_sequence[chain])
    #     # api_info = get_pdb_info_by_entry(matching_pdb)
    #     # chain_entity = {chain: entity_idx for entity_idx, chains in api_info.get('entity').items() for ch in chains}
    #     partner_sequences.append(get_entity_reference_sequence(pdb=matching_pdb, chain=chain))

    # matching_pdb_tags = {idx: find_expression_tags(seq) for idx, seq in enumerate(partner_sequences)}
    # [[{'name': tag_name, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...], ...]
    # matching_pdb_tags = list(iter_chain.from_iterable(find_expression_tags(sequence) for sequence in partner_sequences))
    # reduce the iter of iterables for missing values. ^ can return empty lists
    for sequence in partner_sequences:
        matching_pdb_tags.extend(find_expression_tags(sequence))

    return matching_pdb_tags
    # # Next, align all the tags to the reference sequence and tally the tag location and type
    # pdb_tag_tally = {'n': {}, 'c': {}, 'matching_tags': matching_pdb_tags}
    # for partner_tag in matching_pdb_tags:
    #     # if partner_pdb_tags:
    #     #     for partner_tag in partner_pdb_tags:
    #     if partner_tag['name'] in pdb_tag_tally[partner_tag['termini']]:
    #         pdb_tag_tally[partner_tag['termini']][partner_tag['name']] += 1
    #     else:
    #         pdb_tag_tally[partner_tag['termini']][partner_tag['name']] = 1
    #
    # return pdb_tag_tally


def select_tags_for_sequence(sequence_id, matching_pdb_tags, preferred=None, n=True, c=True):
    """From a list of possible tags, solve for the tag with the most observations in the PDB. If there are
    discrepancies, query the user for a solution

    Args:
        sequence_id (str): The sequence identifier
        matching_pdb_tags (list[dict]): [{'name': His Tag, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]
    Keyword Args:
        preferred=None (str): The name of a preferred tag provided by the user
        n=True (bool): Whether the n-termini can be tagged
        c=True (bool): Whether the c-termini can be tagged
    Returns:
        (dict): {'name': 'his_tag', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}
    """
    # Next, align all the tags to the reference sequence and tally the tag location and type
    # {'n': {His Tag: 2}, 'c': {Spy Catcher: 1}}
    pdb_tag_tally = {'n': {}, 'c': {}}
    for partner_tag in matching_pdb_tags:
        # if partner_pdb_tags:
        #     for partner_tag in partner_pdb_tags:
        if partner_tag['name'] in pdb_tag_tally[partner_tag['termini']]:
            pdb_tag_tally[partner_tag['termini']][partner_tag['name']] += 1
        else:
            pdb_tag_tally[partner_tag['termini']][partner_tag['name']] = 1

    final_tag_sequence = {'name': None, 'termini': None, 'sequence': None}
    # n_term, c_term = 0, 0
    n_term = sum([pdb_tag_tally['n'][tag_name] for tag_name in pdb_tag_tally.get('n', {})])
    c_term = sum([pdb_tag_tally['c'][tag_name] for tag_name in pdb_tag_tally.get('c', {})])
    if n_term == 0 and c_term == 0:  # No tags found
        return final_tag_sequence
    if n_term > c_term and n or (n_term < c_term and n and not c):
        termini = 'n'
    elif n_term < c_term and c or (n_term > c_term and c and not n):
        termini = 'c'
    elif not c and not n:
        while True:
            termini = \
                input('For sequence target %s, NEITHER termini are available for tagging.\n\n'
                      'You can set up tags anyway and modify this sequence later, or skip tagging.\nThe tag options, '
                      'are as follows:\n  Termini: {tag name: count}}\n\t%s\nWhich termini would you prefer '
                      '[n/c]? To skip, input \'skip\'%s' %
                      (sequence_id, '\n\t'.join('%s: %s' % it for it in pdb_tag_tally.items()), input_string)).lower()
            if termini in ['n', 'c']:
                break
            elif termini == 'skip':
                return final_tag_sequence
            else:
                print('\'%s\' is an invalid input, one of \'n\', \'c\', or \'skip\' is required.' % termini)
    else:  # termini = 'Both'
        if c and not n:
            termini = 'c'
        elif not c and n:
            termini = 'n'
        else:
            while True:
                termini = \
                    input('For sequence target %s, BOTH termini are available and have the same number of matched tags.'
                          '\nThe tag options, are as follows:\n  Termini: {tag name: count}}\n\t%s\nWhich termini would'
                          ' you prefer [n/c]? To skip, input \'skip\'%s' %
                          (sequence_id, '\n\t'.join('%s: %s' % it for it in pdb_tag_tally.items()),
                           input_string)).lower()
                if termini in ['n', 'c']:
                    break
                elif termini == 'skip':
                    return final_tag_sequence
                else:
                    print('\'%s\' is an invalid input, one of \'n\', \'c\', or \'skip\' is required.' % termini)

    # Find the most common tag at the specific termini
    all_tags = []
    max_type, max_count = None, 0
    for tag_name in pdb_tag_tally[termini]:
        if pdb_tag_tally[termini][tag_name] > max_count:
            max_count = pdb_tag_tally[termini][tag_name]
            max_type = tag_name
    if max_type:
        all_tags.append(max_type)

    # Check if there are equally represented tags
    for tag_name in pdb_tag_tally[termini]:
        if pdb_tag_tally[termini][tag_name] == max_count and tag_name != max_type:
            all_tags.append(tag_name)

    # Finally report results to the user and solve ambiguous tags
    final_tags = {'termini': termini, 'name': all_tags}
    if not final_tags['name']:  # ensure list has at least one element
        return final_tag_sequence
    custom = False
    final_choice = {}
    while True:
        if preferred and preferred == final_tags['name'][0]:
            default = 'y'
        else:
            default = \
                input('For %s, the RECOMMENDED tag options are: Termini-%s Type-%s\nIf the Termini or Type is '
                      'undesired, you can see the underlying options by specifying \'options\'. Otherwise, \'%s\' will '
                      'be chosen.\nIf you would like to proceed with the RECOMMENDED options, enter \'y\'.%s'
                      % (sequence_id, final_tags['termini'], final_tags['name'][0], final_tags['name'][0],
                         input_string)).lower()
        if default == 'y':
            final_choice['name'] = final_tags['name'][0]
            final_choice['termini'] = final_tags['termini']
            break
        elif default == 'options':
            print('\nFor %s, all tag options are:\n\tTermini Tag:\tCount\n%s\nAll tags:\n%s\n'
                  % (sequence_id, '\n'.join('\t%s:\t%s' % item for item in pdb_tag_tally.items()), matching_pdb_tags))
            # Todo pretty_table_format on the .values() from each item in above list() ('name', 'termini', 'sequence')
            while True:
                termini_input = input('What termini would you like to use [n/c]? If no tag option is appealing, '
                                      'enter \'none\' or specify the termini and select \'custom\' at the next step %s'
                                      % input_string).lower()
                if termini in ['n', 'c']:
                    final_choice['termini'] = termini_input
                    break
                elif termini == 'none':
                    return final_tag_sequence
                else:
                    print('Input \'%s\' doesn\'t match available options. Please try again' % termini_input)
            while True:
                tag_input = input('What tag would you like to use? Enter the number of the below options.\n\t%s%s\n%s'
                                  % ('\n\t'.join(['%d - %s' % (i, tag)
                                                  for i, tag in enumerate(pdb_tag_tally[termini_input], 1)]),
                                     '\n\t%d - %s' % (len(pdb_tag_tally[termini_input]) + 1, 'CUSTOM'), input_string))
                if tag_input.isdigit():
                    tag_input = int(tag_input)
                    if tag_input <= len(pdb_tag_tally[termini_input]):
                        final_choice['name'] = list(pdb_tag_tally[termini_input].keys())[tag_input - 1]
                        break
                    elif tag_input == len(pdb_tag_tally[termini_input]) + 1:
                        custom = True
                        while True:
                            tag_input = input('What tag would you like to use? Enter the number of the below options.'
                                              '\n\t%s\n%s'
                                              % ('\n\t'.join(['%d - %s' % (i, tag)
                                                              for i, tag in enumerate(expression_tags, 1)]),
                                                 input_string))
                            if tag_input.isdigit():
                                tag_input = int(tag_input)
                            if tag_input <= len(expression_tags):
                                final_choice['name'] = list(expression_tags.keys())[tag_input - 1]
                                break
                            print('Input \'%s\' doesn\'t match available options. Please try again' % tag_input)
                        break
                print('Input \'%s\' doesn\'t match available options. Please try again' % tag_input)
            break
        print('Input \'%s\' doesn\'t match. Please try again' % default)

    final_tag_sequence['name'] = final_choice['name']
    final_tag_sequence['termini'] = final_choice['termini']
    all_matching_tags = []
    # [{'name': tag_name, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]
    for tag in matching_pdb_tags:
        # for tag in pdb_match:
        if final_choice['name'] == tag['name'] and final_choice['termini'] == tag['termini']:
            all_matching_tags.append(tag['sequence'])

    # TODO align multiple and choose the consensus
    # all_alignments = []
    # max_tag_idx, max_len = None, []  # 0
    # for idx, (tag1, tag2) in enumerate(combinations(all_matching_tags, 2)):
    #     alignment = SequenceProfile.generate_alignment(tag1, tag2)
    #     all_alignments.append(alignment)
    #     # if max_len < alignment[4]:  # the length of alignment
    #     max_len.append(alignment[4])
    #     # have to find the alignment with the max length, then find which one of the sequences has the max length for
    #     # multiple alignments, then need to select all alignments to this sequence to generate the MSA
    #
    # total_alignment = SequenceProfile.create_bio_msa({idx: tag for idx, tag in enumerate(all_matching_tags)})
    # tag_msa = SequenceProfile.generate_msa_dictionary(total_alignment)
    if custom:
        final_tag_sequence['sequence'] = expression_tags[final_choice['name']]
    else:
        final_tag_sequence['sequence'] = all_matching_tags[0]  # for now grab the first

    return final_tag_sequence


def add_expression_tag(tag, sequence):
    """Take a raw sequence and add expression tag by aligning a specified tag by PDB reference

    Args:
        tag (str):
        sequence (str):
    Returns:
        (str): The final sequence with the tag added
    """
    if not tag:
        return sequence
    alignment = SequenceProfile.generate_alignment(tag, sequence)
    # print('Expression TAG alignment:', alignment[0])
    tag_seq, seq, score, *_ = alignment
    # score = alignment[2]  # first alignment, grab score value
    # print('Expression TAG alignment score:', score)
    # if score == 0:  # TODO find the correct score for a bad alignment to indicate there was no productive alignment?
    #     # alignment[0][4]:  # the length of alignment
    #     # match_score = score / len(sequence)  # could also use which ever sequence is greater
    #     return sequence
    # print(alignment[0])
    # print(tag_seq)
    # print(seq)
    # starting_index_of_seq2 = seq.find(sequence[0])
    # i = -starting_index_of_seq2 + index_offset  # make 1 index so residue value starts at 1
    final_seq = ''
    for i, (seq1_aa, seq2_aa) in enumerate(zip(tag_seq, seq)):
        if seq2_aa == '-':
            if seq1_aa in protein_letters:
                final_seq += seq1_aa
        else:
            final_seq += seq2_aa

    return final_seq


def pull_uniprot_id_by_pdb(uniprot_pdb_d, pdb_code, chain=None):
    # uniprot_pdb_d = SDUtils.unpickle(PUtils.uniprot_pdb_map)
    source = 'unique_pdb'
    pdb_code = pdb_code.upper()
    if chain:
        # pdb_code = '%s.%s' % (pdb_code, chain)
        # source = 'all'
        dummy = 'TODO ensure that this works once the database is integrated'  # TODO

    for uniprot_id in uniprot_pdb_d:
        if pdb_code in uniprot_pdb_d[uniprot_id][source]:
            return uniprot_id
    return None


def find_expression_tags(sequence, alignment_length=12):
    """Find all expression_tags on an input sequence from a reference set of expression_tags. Returns the matching tag
    sequence with additional protein sequence context equal to the passed alignment_length

    Args:
        sequence (str): 'MSGHHHHHHGKLKPNDLRI...'
    Keyword Args:
        # tag_file=PathUtils.affinity_tags (list): List of tuples where tuple[0] is the name and tuple[1] is the string
        alignment_length=12 (int): length to perform the clipping of the native sequence in addition to found tag
    Returns:
        (list[dict]): [{'name': tag_name, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...], [] if none are found
    """
    matching_tags = []
    for tag, tag_sequence in expression_tags.items():
        tag_index = sequence.find(tag_sequence)
        if tag_index == -1:  # no match was found
            continue
        # save the tag name, the termini of the sequence it is closest to, and the source sequence context
        tag_length = len(tag_sequence)
        alignment_size = tag_length + alignment_length
        if tag_index == 0 or tag_index < len(sequence) / 2:
            termini = 'n'
            matching_sequence = sequence[tag_index:tag_index + alignment_size]
        else:
            termini = 'c'
            matching_sequence = sequence[tag_index - alignment_size:tag_index + tag_length]
        matching_tags.append({'name': tag, 'termini': termini, 'sequence': matching_sequence})

    return matching_tags


def remove_expression_tags(sequence, tags):
    """Remove all specified tags from an input sequence

    Args:
        sequence (str): 'MSGHHHHHHGKLKPNDLRI...'
        tags (list): A list with the sequences of found tags
    Returns:
        (str): 'MSGGKLKPNDLRI...' The modified sequence without the tag
    """
    for tag in tags:
        tag_index = sequence.find(tag)
        if tag_index == -1:  # no match was found
            continue
        sequence = sequence[:tag_index] + sequence[tag_index + len(tag):]

    return sequence


def optimize_protein_sequence(sequence, species='e_coli'):
    """Optimize a sequence for expression in a desired organism

    Args:
        sequence (str): The sequence of interest
    Keyword Args:
        species='e_coli' (str): The species context to optimize nucleotide sequence usage
    Returns:
        (str): The input sequence optimized to nucleotides for expression considerations
    """
    seq_length = len(sequence)
    problem = DnaOptimizationProblem(sequence=reverse_translate(sequence),  # max_random_iters=20000,
                                     objectives=[CodonOptimize(species=species)], logger=None,
                                     constraints=[EnforceGCContent(mini=0.25, maxi=0.65),  # twist required
                                                  EnforceGCContent(mini=0.35, maxi=0.65, window=50),  # twist required
                                                  AvoidHairpins(stem_size=20, hairpin_window=48),  # efficient translate
                                                  AvoidPattern('GGAGG', location=(1, seq_length, 1)),  # ribosome bind
                                                  AvoidPattern('TAAGGAG', location=(1, seq_length, 1)),  # ribosome bind
                                                  AvoidPattern('AAAAA', location=(1, seq_length, 0)),  # terminator
                                                  # AvoidPattern('TTTTT', location=(1, seq_length, 1)),  # terminator
                                                  AvoidPattern('GGGGGGGGGG', location=(1, seq_length, 0)),  # homopoly
                                                  # AvoidPattern('CCCCCCCCCC', location=(1, seq_length)),  # homopoly
                                                  UniquifyAllKmers(20),  # twist required
                                                  AvoidRareCodons(0.08, species=species),
                                                  EnforceTranslation(),
                                                  # EnforceMeltingTemperature(mini=10, maxi=62, location=(1, seq_length)),
                                                  ])

    # Solve constraints and solve with regards to the objective
    problem.max_random_iters = 20000
    problem.resolve_constraints()
    problem.optimize()

    # Display summaries of constraints that pass
    # print(problem.constraints_text_summary())
    # print(problem.objectives_text_summary())

    # Get final sequence as string
    final_sequence = problem.sequence
    # Get final sequene as BioPython record
    # final_record = problem.to_record(with_sequence_edits=True)

    return final_sequence
