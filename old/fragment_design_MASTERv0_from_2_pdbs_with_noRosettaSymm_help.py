#!/home/kmeador/miniconda3/bin/python
import os
import sys
import math
import subprocess
import pickle
import copy
import numpy as np
import sklearn.neighbors
# import collections
# import PDB
from PDB import PDB
import functions as fc
import pyrosetta as pyr
pyr.init()
'''
Before full run need to alleviate the following tags and incompleteness
-TEST - ensure working and functional against logic and coding errors
-TODO - Make this resource
-JOSH - coordinate on input style from Josh's output
'''

# TODO Modify all of these during individual users program set up
#### OBJECTS #### #TODO
interface_background = {}
# TODO find the interface background from some source. Initial thoughts are from fragment
# TODO libraries using Levy SASA conventions.

#### PATHS #### #TODO
# alignmentDB = #TODO Install ProteinNet on the users storage. Needs to update with new releases
'''Ex. user/symDock/ProteinNet/'''
# fragmentDB = #TODO set up to automatically on any users system after the install of fragment library
'''Ex. user/symDock/fragments/bio/1_2_24 or user/symDock/fragments/xtal/1_1_29'''
# rosettaDB = #TODO set up to automatically to get this path
'''Ex. user/rosetta_..._bundle'''
# TEST
alignmentdb = '/home/kmeador/local_programs/ncbi_databases/uniref90'
fragmentDB = '/home/kmeador/yeates/interface_extraction/ijk_clustered_xtal_fragDB_1A_info'
rosettaBIN = '/joule2/programs/rosetta/rosetta_src_2019.35.60890_bundle/main/source/bin/'
rosettaDB = '/joule2/programs/rosetta/rosetta_src_2019.35.60890_bundle/main/database/'

#### MODULE FUNCTIONS #### TODO
# fc.extract_aa_seq
# fc.alph_3_aa_list
# fc.write_fasta_file

#### CODE FORMATTING ####
# PSSM format is  dictating the corresponding file formatting for Rosetta
# aa_counts_dict and PSSM format are in different alphabetical orders (1v3 letter), this may be a problem down the line


# class RS:
#     def __init__(self):
#         self.script = ''
#         self.score_functions = []
#         self.task_operations = []
#         self.movers = []
#         self.filters = []
#         self.protocols = []
#
#     def add(self, **kwargs):
#         if **kwargs = 'score_functions', 'task_operations', 'movers', 'filters', 'protocols'
#             self.score_functions.append(argument
#                                         )
#     def assemble_rosetta_script(self):
#         self.script = '<ROSETTASCRIPTS>\n'
#         score_functions = list_objects(self, self.score_functions)
#         for i in score_functions:
#             self.script = '\t<SCOREFXNS>\n' + '\t<ScoreFunction name="%s" weights= "%s"' %(i[0], i[1])
#
#     def list_objects(self, function_list):
#         for i in function_list:
#
#             return
#
#     def run_rosetta_script(self):


def rosetta_score(pdb):
    # this will also format your output in rosetta numbering
    cmd = [rosettaBIN, 'score_jd2.default.linuxgccrelease', '-renumber_pdb', '-ignore_unrecognized_res', '-s', pdb,
           '-out:pdb']
    subprocess.Popen(cmd, start_new_session=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    return pdb + '_0001.pdb'


def make_pssm_psiblast(query, remote=False):
    # generate an position specific scoring matrix from the pdb homologs
    # the output from PSI-BLAST is sufficient for our purposes. I would like the background to come from Uniref90
    # instead of BLOSUM 62. This could be something to dive deeper later #TODO
    if query + '.pssm' in os.listdir(os.getcwd()):
        return query + '.pssm'

    cmd = ['psiblast', '-db', alignmentdb, '-query', query + '.fasta', '-out_ascii_pssm', query + '.pssm',
           '-save_pssm_after_last_round', '-evalue', '1e-6', '-num_iterations', '0']
    if remote:
        cmd.append('-remote')
    else:
        cmd.append('-num_threads')
        cmd.append('8')
    print(cmd)
    # + ' -outfmt "6 sseqid sseq" | awk \'BEGIN{FS="\t"; OFS="\n"}{gsub(/[-XZBU]/, "", $2)}; /\|/ {if(NR<=1999) print
    # ">"$1,$2}\' >> ' + query + '_homologs.fasta'
    # cmd = 'psiblast -db ' + local_db + options + ' -query ' + query + '.fasta' + ' -out_ascii_pssm ' + query + '.pssm'
    # + ' -outfmt "6 sseqid sseq" | awk \'BEGIN{FS="\t"; OFS="\n"}{gsub(/[-XZBU]/, "", $2)}; /\|/ {if(NR<=1999) print
    # ">"$1,$2}\' >> ' + query + '_homologs.fasta'
    subprocess.call(cmd)  # , stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    return query + '.pssm'


def get_pssm(query):
    # TODO Use HHblits to generate a PSSM
    return query


def modify_pssm_background(pssm, background):
    # TODO make a function which unpacks pssm and refactors the lod score with specifc background

    return mod_pssm


def parse_pssm(file, pose_dict):
    # take the contents of protein.pssm, parse file, and input data into pose_dict
    with open(file, 'r') as f:
        lines = f.readlines()
    pssm = []
    for line in lines:
        try:
            resi = int(line[:6].strip())
            resi_type = line[6:8].strip()
            log_odds = line[11:90].strip().split()
            int_odds = []
            for o in log_odds:
                int_odds.append(int(o))
            counts = line[91:171].strip().split()
            int_counts = []
            for i in counts:
                int_counts.append(int(i))
            info = float(line[173:177].strip())
            weight = float(line[178:182].strip())
            pssm.append((resi, resi_type, int_odds, int_counts, info, weight))
        except ValueError:
            continue

    # Get normalized counts for pose_dict
    for residue in pssm:
        resi = residue[0] - 1
        i = 0
        for aa in pose_dict[resi]:
            pose_dict[resi][aa] = residue[3][i]
            i += 1
        pose_dict[resi]['type'] = residue[1]
        pose_dict[resi]['log'] = residue[2]
        pose_dict[resi]['info'] = residue[4]
        pose_dict[resi]['weight'] = residue[5]

    return pose_dict


def combine_pssm(pssm1, pssm2):
    temp = {}
    new_key = len(pssm1)
    for old_key in pssm2:
        temp[new_key] = pssm2[old_key]
        new_key += 1
    pssm1.update(temp)

    return pssm1


def duplicate_ssm(pssm_dict, pdb_copies):
    duplicated_ssm = {}
    duplication_start = len(pssm_dict)
    for i in range(pdb_copies):
        if i == 0:
            offset = 0
        else:
            offset = duplication_start * i
        # j = 0
        for line in pssm_dict:
            duplicated_ssm[line + offset] = pssm_dict[line]
            # j += 1

    return duplicated_ssm


def make_pssm_file(pssm_dict, name):
    header = '\n\n            A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   A   R' \
             '   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V\n'
    footer = ''
    out_file = name + '.pssm'

    with open(out_file, 'w') as file:
        file.write(header)
        for res in pssm_dict:
            aa_type = pssm_dict[res]['type']
            log_string = ''
            for aa in pssm_dict[res]['log']:
                log_string += '{:>3d} '.format(aa)
            counts_string = ''
            for aa in fc.alph_3_aa_list:  # ensure the order never changes... this is true for python 3.6+
                counts_string += '{:>3d} '.format(pssm_dict[res][aa])
            info = pssm_dict[res]['info']
            weight = pssm_dict[res]['weight']
            line = '{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}''\n'.format(res + 1, aa_type, log_string,
                                                                             counts_string, round(info, 4),
                                                                             round(weight, 4))
            file.write(line)
        file.write(footer)

    return out_file


def parse_cluster_info(file):
    # TODO
    return None


def populate_design_dict(pose_length, frag_length):
    # generate a dictionary of dictionaries with n elements and m subelements,
    # where n is the number of residues in the pdb and m is the length of fragments in the design library
    design_dict = {residue: {i: {} for i in frag_length} for residue in range(pose_length)}

    return design_dictt


def get_all_clusters(residue_cluster_id_list):  # list version
    # generate an interface specific scoring matrix from the fragment library
    # JOSH assuming residue_cluster_list has form [(1_2_24, [78, 87]), ...]
    cluster_list = []
    for cluster in residue_cluster_id_list:
        cluster_loc = cluster[0].split('_')
        filename = os.path.join(fragmentDB, cluster_loc[0], cluster_loc[0] + '_' + cluster_loc[1], cluster_loc[0] +
                                '_' + cluster_loc[1] + '_' + cluster_loc[2], cluster[0] + '.pkl')
        with open(filename, 'rb') as f:
            cluster_list.append((cluster[1], pickle.load(f)))

    # OUTPUT format: [([78, 87], {'IJKClusterDict - such as 1_2_45'}), ([64, 87], {...}, ...]
    return cluster_list


def convert_to_rosetta_num(pdb, pose, interface_residue_list):  # TEST
    # INPUT format: interface_residue_list = [([78, 87], {'IJKClusterDict - such as 1_2_45'}), ([64, 87], {...}, ...]
    component_chains = [pdb.chain_id_list[0], pdb.chain_id_list[-1]]
    interface_residue_dict = {}
    for residue_dict_pair in interface_residue_list:
        residues = residue_dict_pair[0]
        dict = residue_dict_pair[1]
        new_key = []
        pair_index = 0
        for chain in component_chains:
            new_key.append(pose.pdb_info().pdb2pose(chain, residues[pair_index]))
            pair_index = 1
        hash = (new_key[0], new_key[1])

        interface_residue_dict[hash] = dict

    # OUTPUT format: interface_residue_dict = {[78, 256]: {'IJKClusterDict - such as 1_2_45'}, [64, 256]: {...}, ...}
    return interface_residue_dict


def deconvolve_clusters(full_cluster_dict, full_design_dict):
    # INPUT format: full_design_dict = {0: {-2: {},... }, ...}
    # INPUT format: full_cluster_dict = {[78, 256]: {'size': cluster_member_count, 'rmsd': mean_cluster_rmsd, 'rep':
    #                                                str(cluster_rep), 'mapped': mapped_freq_dict, 'paired':
    #                                                partner_freq_dict}
    # aa_freq format: mapped/paired_freq_dict = {-2: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37]}, -1:...}
    # where 'stats'[0] is total fragments in cluster, and 'stats'[1] is weight of fragment index
    offset = -1  # for indexing when design dict starts at 0 and cluster dict has actual residue numbers
    designed_residues = []
    for pair in full_cluster_dict:
        mapped = pair[0]
        paired = pair[1]
        mapped_dict = full_cluster_dict[pair]['mapped']
        paired_dict = full_cluster_dict[pair]['paired']
        for fragment_data in [(mapped_dict, mapped), (paired_dict, paired)]:
            for frag_index in fragment_data[0]:
                aa_freq = fragment_data[0][frag_index]  # This has weights in it...
                residue = fragment_data[1] + frag_index + offset
                # First, check to see if there are already fragments which share the same central residue
                if fragment_data[1] not in designed_residues:
                    # Add an instance of the aa_freq at the residue to the frag_index indicator
                    full_design_dict[residue][frag_index][0] = aa_freq
                else:
                    for i in range(1, 100):
                        try:
                            if full_design_dict[residue][frag_index][i]:  # != dict():  # TEST to see if exists, should be false if not
                                continue
                        except KeyError:
                            full_design_dict[residue][frag_index][i] = aa_freq
                            break
            designed_residues.append(fragment_data[1])

    return full_design_dict

    # for frag_index in mapped_dict:
    #     mapped_counts = mapped_dict[frag_index]
    #     residue = mapped + frag_index
    #     if mapped not in designed_residues:
    #         # add the aa type count at the residue to the frag_index (priority coefficient) indicator
    #         full_design_dict[residue][frag_index] = mapped_counts
    #     else:
    #         # If the residue is part of design fragment_data already, add counts to those observed already.
    #         # This is only for cases where the design fragment central residue contains multiple partners
    #         for aa in mapped_counts:
    #             if full_design_dict[residue][frag_index][aa]:
    #                 full_design_dict[residue][frag_index][aa] += mapped_counts[aa]
    #             else:
    #                 full_design_dict[residue][frag_index][aa] = mapped_counts[aa]


def flatten_for_issm(full_cluster_dict):
    # Take a multi-observation, mulit-fragment indexed design dict and reduce to single residue frequency for ISSM
    aa_counts_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
                      'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'stats': [0, 1]}
    no_design = []
    for residue in full_cluster_dict:
        total_residue_weight = 0
        for frag_index in full_cluster_dict[residue]:
            if full_cluster_dict[residue][frag_index] == dict():
                continue
            # else:
            try:
                total_obs_weight = 0
                for observation in full_cluster_dict[residue][frag_index]:
                    total_obs_weight += full_cluster_dict[residue][frag_index][observation]['stats'][1]
                if total_obs_weight == 0:
                    # Case where no weights associated with observations (side chain not structurally significant)
                    full_cluster_dict[residue][frag_index] = dict()
                    continue

                obs_aa_dict = copy.deepcopy(aa_counts_dict)
                obs_aa_dict['stats'][1] = total_obs_weight  # /2 <--TODO if fragments and their flip are present
                for observation in full_cluster_dict[residue][frag_index]:
                    obs_weight = full_cluster_dict[residue][frag_index][observation]['stats'][1]
                    for aa in full_cluster_dict[residue][frag_index][observation]:
                        if aa == 'stats':
                            continue
                        full_cluster_dict[residue][frag_index][observation][aa] *= (obs_weight / total_obs_weight)
                        # Add all occurrences to summed frequencies list
                        obs_aa_dict[aa] += full_cluster_dict[residue][frag_index][observation][aa]
                full_cluster_dict[residue][frag_index] = obs_aa_dict

            except TypeError:
                # TEST Is this code reachable? observations is always set to 0 for a single instance. is 0 iterable?
                # What does this do again??
                print('REACHED!')
                full_cluster_dict[residue][frag_index]['stats'][0] = 0
            total_residue_weight += full_cluster_dict[residue][frag_index]['stats'][1]
        if total_residue_weight == 0:
            # Add to list for removal from the design dict
            no_design.append(residue)
            continue

        res_aa_dict = copy.deepcopy(aa_counts_dict)
        res_aa_dict['stats'][1] = total_residue_weight
        for frag_index in full_cluster_dict[residue]:
            if full_cluster_dict[residue][frag_index] == dict():
                continue
            index_weight = full_cluster_dict[residue][frag_index]['stats'][1]
            for aa in full_cluster_dict[residue][frag_index]:
                if aa == 'stats':
                    continue
                full_cluster_dict[residue][frag_index][aa] *= (index_weight / total_residue_weight)
                # Add all occurrences to summed frequencies list
                res_aa_dict[aa] += full_cluster_dict[residue][frag_index][aa]
        full_cluster_dict[residue] = res_aa_dict

    # Remove missing data from dictionary
    for residue in no_design:
        full_cluster_dict.pop(residue)
    for residue in full_cluster_dict:
        remove_aa = []
        for aa in full_cluster_dict[residue]:
            if aa == 'stats':
                continue
            if full_cluster_dict[residue][aa] == 0.0:
                remove_aa.append(aa)
        for i in remove_aa:
            full_cluster_dict[residue].pop(i)

    return full_cluster_dict


def make_issm(cluster_freq_dict, background):
    for residue in cluster_freq_dict:
        for aa in cluster_freq_dict[residue]:
            cluster_freq_dict[residue][aa] = math.log((cluster_freq_dict[residue][aa] / background[aa]), 2)
    issm = []

    return issm


def combine_sm(pssm, issm):
    # combine the weights for the PSSM with the ISSM
    full_ssm = []

    return full_ssm


def read_pdb(file):
    pdb = PDB()
    pdb.readfile(file)

    return pdb


def combine_pdb(pdb_1, pdb_2, name):
    # Take two pdb objects and write them to the same file
    pdb_1.write(name)
    with open(name, 'a') as full_pdb:
        for atom in pdb_2.all_atoms:
            full_pdb.write(str(atom))  # .strip() + '\n')


def identify_interfaces(pdb1, pdb2):
    distance = 12  # Angstroms
    pdb1_chains = []
    pdb2_chains = []

    # Get CB Coordinates
    pdb1_cb_coords = pdb1.extract_CB_coords(InclGlyCA=True)
    pdb1_cb_coords = np.array(pdb1_cb_coords)
    pdb2_cb_coords = pdb2.extract_CB_coords(InclGlyCA=True)
    pdb2_cb_coords = np.array(pdb2_cb_coords)

    # Construct Atom Tree for PDB1 CB Atoms and query for all PDB2 CB Atoms within distance
    pdb1_cb_tree = sklearn.neighbors.BallTree(pdb1_cb_coords)
    query = pdb1_cb_tree.query_radius(pdb2_cb_coords, distance)

    # Map Coordinates to Atoms
    pdb1_cb_indices = pdb1.get_cb_indices(InclGlyCA=True)
    pdb2_cb_indices = pdb2.get_cb_indices(InclGlyCA=True)

    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].tolist() != list():
            pdb2_chains.append(pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].chain)
            for pdb1_query_index in query[pdb2_query_index]:
                pdb1_chains.append(pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].chain)

    pdb1_chains = list(set(pdb1_chains))
    pdb2_chains = list(set(pdb2_chains))

    return pdb1_chains, pdb2_chains


def reduce_pose_to_interface(pdb, chains):
    new_pdb = PDB()
    new_pdb.read_atom_list(pdb.chains(chains))

    return new_pdb


def find_jump_site(pdb, pose):
    chains = pdb.chain_id_list  # TODO Test and integrate the FoldTree
    jump_start = pose.pdb_info().pdb2pose(chains[0], pdb.getTermCAAtom('C', chains[0]).residue_number)
    jump_end = pose.pdb_info().pdb2pose(chains[1], pdb.getTermCAAtom('N', chains[1]).residue_number)
    jumps = (jump_start, jump_end)

    return jumps


def parameterize_frag_length(length):
    divide_2 = math.floor(length / 2)
    modulus = length % 2
    if modulus == 1:
        upper_bound = 0 + divide_2
        lower_bound = 0 - divide_2
    else:
        upper_bound = 0 + divide_2
        lower_bound = upper_bound - length + 1
    index_offset = -lower_bound + 1

    return lower_bound, upper_bound, index_offset


def main():
    # query = '/home/kmeador/yeates/xtal_designs/2nd_batch/seq_files/firsttest/1BVS.pdb1.fasta'
    design_directory = sys.argv[1]  # TODO parse input
    frag_size = 5
    lower_bound, upper_bound, index_offset = parameterize_frag_length(frag_size)

    # cluster_data = []
    template_pdbs = []
    frag_pdbs = []
    for root, dirs, files, in os.walk(design_directory):
        for file in files:
            if file.endswith('.pdb'):
                if len(file.strip('.pdb')) == 6:
                    template_pdbs.append(file)
                else:
                    frag_pdbs.append(file)
            if file == 'frag_match_info_file.txt':
                # TODO get output of Josh's program to make a text file of all the central residue numbers and their
                # TODO corresponding fragment cluster identity for each pdb
                # JOSH assuming file has form (1_2_24, [78, 87])\n(1_2_45, [64, 87])\n...
                with open(file, 'r') as f:
                    residue_cluster_list = f.readlines()
                    print(residue_cluster_list)
                    for i in range(len(residue_cluster_list)):
                        residue_cluster_list[i] = residue_cluster_list[i].strip()
                        residue_cluster_list[i] = (residue_cluster_list[i].split(',')[0], [int(residue_cluster_list[i].split(',')[1].strip().split('|')[0]), int(residue_cluster_list[i].split(',')[1].strip().split('|')[1])])
                    print(residue_cluster_list)

    # JOSH need to make this input from the docking program
    # TODO fill these in a specific order. Could use files with a designation 1ABC_1 and 1BCA_2 to designate 1 and 2
    pdb_name_1, pdb_name_2, pdb1, pdb2, pdb_sequence1, pdb_sequence2, pdb_sequence_file1, pdb_sequence_file2, \
        pdb_1_copies, pdb_2_copies = None, None, None, None, None, None, None, None, None, None
    for pdb in template_pdbs:
        oligomer = int(pdb.strip('.pdb').split('_')[1])
        if oligomer == 1:
            pdb_1 = pdb
            pdb_name_1 = pdb_1[:4]
            pdb1 = read_pdb(pdb_1)
            pdb_1_protomers = len(pdb1.chain_id_list)
            pdb_sequence1, errors1 = fc.extract_aa_seq(pdb1)
            pdb_sequence_file1 = fc.write_fasta_file(pdb_sequence1, pdb_name_1)
        if oligomer == 2:
            pdb_2 = pdb
            pdb_name_2 = pdb_2[:4]
            pdb2 = read_pdb(pdb_2)
            pdb_2_protomers = len(pdb2.chain_id_list)
            pdb_sequence2, errors2 = fc.extract_aa_seq(pdb2)
            pdb_sequence_file2 = fc.write_fasta_file(pdb_sequence2, pdb_name_2)
    if pdb_sequence_file1 and pdb_sequence_file2:
        pass
    else:
        print('Error parsing files')
        sys.exit()

    # Make PSSM from (TODO PSI-BLAST/Easel/ProteinNet) of PDB sequence
    pssm_file1 = make_pssm_psiblast(pdb_name_1)  # , True)
    pssm_file2 = make_pssm_psiblast(pdb_name_2)  # , True)
    config_pssm_dict1 = populate_design_dict(len(pdb_sequence1), fc.alph_3_aa_list)
    config_pssm_dict2 = populate_design_dict(len(pdb_sequence2), fc.alph_3_aa_list)
    pssm_dict1 = parse_pssm(pssm_file1, config_pssm_dict1)  # index is 0 start
    pssm_dict2 = parse_pssm(pssm_file2, config_pssm_dict2)  # index is 0 start

    # Find interacting chains and rename to avoid overlap
    dummy = []
    pdb1_interface_chains, pdb2_interface_chains = identify_interfaces(pdb1, pdb2)
    # print(pdb1_interface_chains, pdb2_interface_chains)
    pdb1_int_chain_num = len(pdb1_interface_chains)
    pdb2_int_chain_num = len(pdb2_interface_chains)
    pdb1_interface = reduce_pose_to_interface(pdb1, pdb1_interface_chains)
    pdb2_interface = reduce_pose_to_interface(pdb2, pdb2_interface_chains)
    pdb1_interface.reorder_chains(dummy)
    pdb2_interface.reorder_chains(pdb1_interface.chain_id_list)
    # print(pdb1_interface.chain_id_list, pdb2_interface.chain_id_list)

    # Generate multiplicative objects to account for object symmetry/chains in interface
    pssm_dict1 = duplicate_ssm(pssm_dict1, pdb1_int_chain_num)
    pssm_dict2 = duplicate_ssm(pssm_dict2, pdb2_int_chain_num)
    # new_pssm_1 = make_pssm_file(pssm_dict1, pdb_name_1 + '_' + str(pdb1_int_chain_num))
    # new_pssm_2 = make_pssm_file(pssm_dict2, pdb_name_2 + '_' + str(pdb2_int_chain_num))

    # Combine both 1 and 2 PDB_interface and PSSM objects into single instances of PDB and PSSM
    full_pdb_name = pdb_name_1 + '_' + pdb_name_2 + '_interface.pdb'  # puts oligomer1 first then oligomer2
    full_pdb_sequence = pdb_sequence1 * pdb1_int_chain_num + pdb_sequence2 * pdb2_int_chain_num # puts oligomer1 first then oligomer2
    full_pssm = combine_pssm(pssm_dict1, pssm_dict2)  # puts oligomer1 first then oligomer2
    full_pssm_file = make_pssm_file(full_pssm, full_pdb_name[:-4])
    combine_pdb(pdb1_interface, pdb2_interface, full_pdb_name)
    full_pdb = read_pdb(full_pdb_name)

    # Set up Rosetta Specific Information
    pose = pyr.pose_from_pdb(full_pdb_name)
    # jumps = find_jump_site(full_pdb, pose)
    # print(jumps)
    # pdbR = rosetta_score(full_pdb_name)

    # Parse IJK Cluster Dictionaries
    interface_residue_dict = get_all_clusters(residue_cluster_list)  # 191203 changed from dict to list.
    interface_residue_dict = convert_to_rosetta_num(full_pdb, pose, interface_residue_dict)
    # Conversion occurs because PSSM format being 1, 2, ..., N and Rosetta pose # is used instead of PDB # to match
    full_design_dictionary = populate_design_dict(len(full_pdb_sequence), [i for i in
                                                                           range(lower_bound, upper_bound + 1)])
    full_cluster_dict = deconvolve_clusters(interface_residue_dict, full_design_dictionary)
    flattened_cluster_dict = flatten_for_issm(full_cluster_dict)
    print(flattened_cluster_dict)
    sys.exit()
    # Make ISSM from Rosetta Numbered PDB format, fragment list, fragment library
    # TODO Is this function necessary? Can this be condensed into the combine_sm()
    full_issm = make_issm(flattened_cluster_dict, interface_background)

    # Take counts PSSM and counts ISSM and combine into a single PSSM
    final_pssm = combine_sm(full_pssm, full_issm)

    # Run RosettaScriipts
    script_cmd = []
    subprocess.call(script_cmd)


if __name__ == '__main__':
    main()
