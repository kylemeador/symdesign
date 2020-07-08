#!/home/kmeador/miniconda3/bin/python
'''
Before full run need to alleviate the following tags and incompleteness
-TEST - ensure working and functional against logic and coding errors
-TODO - Make this resource
-JOSH - coordinate on input style from Josh's output

Design Decisions:
-We need to have the structures relaxed before input into the design program. Should relaxation be done on individual
protein building blocks? This again gets into the SDF issues. But for rosetta scoring, we need a baseline to be relaxed.
-Should the sidechain info for each of the interface positions be stripped or made to GLY upon rosetta start up?

'''
import os
import sys
import math
import subprocess
import pickle
import copy
# import collections
import numpy as np
import sklearn.neighbors
import fnmatch
import PDB
import functions as fc
# from PDB import PDB
from Bio import pairwise2
# from Bio.SubsMat import MatrixInfo as matlist
# import pyrosetta as pyr
# pyr.init()

# TODO Modify all of these during individual users program set up
#### GLOBAL OBJECTS #### #TODO
# matrix = matlist.blosum62
# gap_penalty = -10
# gap_ext_penalty = -1
interface_background = {}
flags = ['-ex1', '-ex2', '-linmem_ig 10', '-ignore_unrecognized_res', '-ignore_zero_occupancy false', '-overwrite',
         '-extrachi_cutoff 5', '-out:path:pdb rosetta_pdbs/', '-out:path:score rosetta_scores/', '-no_his_his_pairE',
         '-chemical:exclude_patches LowerDNA UpperDNA Cterm_amidation SpecialRotamer ' 
         'VirtualBB ShoveBB VirtualDNAPhosphate VirtualNTerm CTermConnect sc_orbitals pro_hydroxylated_case1 '
         'pro_hydroxylated_case2 ser_phosphorylated thr_phosphorylated tyr_phosphorylated tyr_sulfated '
         'lys_dimethylated lys_monomethylated  lys_trimethylated lys_acetylated glu_carboxylated cys_acetylated '
         'tyr_diiodinated N_acetylated C_methylamidated MethylatedProteinCterm']

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
uniclustdb = '/home/kmeador/local_programs/hh-suite/build/uniclust30_2018_08/uniclust30_2018_08'
source = '/home/kmeador/yeates/symdesign'
fragmentDB = '/home/kmeador/yeates/fragment_database'
bio_fragmentDB = '/home/kmeador/yeates/fragment_database/bio'
xtal_fragmentDB = '/home/kmeador/yeates/fragment_database/xtal'
full_fragmentDB = '/home/kmeador/yeates/fragment_database/bio+xtal'
# TODO Make rosetta_scripts operating system/build independent
rosetta = '/joule2/programs/rosetta/rosetta_src_2019.35.60890_bundle/main'
xtal_protocol = (os.path.join(source, 'rosetta_scripts/xtal_refine.xml'),
                 os.path.join(source, 'rosetta_scripts/xtal_design.xml'))
layer_protocol = (os.path.join(source, 'rosetta_scripts/layer_refine.xml'),
                  os.path.join(source, 'rosetta_scripts/layer_design.xml'))
point_protocol = (os.path.join(source, 'rosetta_scripts/point_refine.xml'),
                  os.path.join(source, 'rosetta_scripts/point_design.xml'))

#### MODULE FUNCTIONS #### TODO
# fc.extract_aa_seq()
# fc.alph_3_aa_list
# fc.alph_aa_list
# fc.write_fasta_file()
# fc.distance() # use a numpy function

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
        cmd = ['echo', 'PSSM: ' + query + '.pssm already exists']
        p = subprocess.Popen(cmd)
        return query + '.pssm', p

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
    p = subprocess.Popen(cmd)  # , stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    return query + '.pssm', p


def make_pssm_hhblits(query):
    if query + '.hmm' in os.listdir(os.getcwd()):
        cmd = ['echo', 'PSSM: ' + query + '.hmm already exists']
        p = subprocess.Popen(cmd)
        return query + '.hmm', p

    cmd = ['hhblits', '-d', uniclustdb, '-i', query + '.fasta', '-ohhm', query + '.hmm', '-v', '1']
    cmd.append('-cpu')
    cmd.append('6')
    print(cmd)

    p = subprocess.Popen(cmd)

    return query + '.hmm', p


def modify_pssm_background(pssm, background):
    # TODO make a function which unpacks pssm and refactors the lod score with specifc background

    return mod_pssm


def populate_design_dict(n, m):
    # Generate a dictionary of dictionaries with n elements and m subelements,
    # Where n is the number of residues in the pdb and m is the length of library
    design_dict = {residue: {i: {} for i in m} for residue in range(n)}

    return design_dict


def parse_hhblits_pssm(file, pose_dict, null_background=True):
    # Take contents of protein.hmm, parse file and input into pose_dict
    dummy = 0.00
    null_bg = fc.distribution_dictionary['uniclust30']

    def to_freq(value):
        if value == '*':
            # When frequency is zero
            return 0.0001
        else:
            # Equation: value = -1000 * log_2(frequency)
            freq = 2 ** (-int(value)/1000)
            return freq

    def get_lod(aa_freq_dict, bg_dict):
        lods = []
        for a in aa_freq_dict:
            aa_lod = math.log(aa_freq_dict[a]/bg_dict[a], 2)
            aa_lod *= 2
            if aa_lod < -9:
                aa_lod = -9
            else:
                aa_lod = round(aa_lod)
            lods.append(aa_lod)
        return lods

    with open(file, 'r') as f:
        lines = f.readlines()

    read = False
    for line in lines:
        if not read:
            if line[0:1] == '#':
                read = True
        else:
            if line[0:4] == 'NULL':
                if null_background:
                    background = line.strip().split()
                    del background[0]
                    null_bg = copy.deepcopy(pose_dict[1])
                    i = 0
                    for aa in fc.alph_aa_list:
                        null_bg[aa] = to_freq(background[i])

            if len(line.split()) == 23:
                items = line.strip().split()
                resi = int(items[1]) - 1
                i = 2
                for aa in fc.alph_aa_list:
                    pose_dict[resi][aa] = to_freq(items[i])
                    i += 1

                pose_dict[resi]['log'] = get_lod(pose_dict[resi], null_bg)
                pose_dict[resi]['type'] = items[0]
                pose_dict[resi]['info'] = dummy
                pose_dict[resi]['weight'] = dummy
    # Output: {1: {'A': 0, 'R': 0, ..., 'log': [-5, -5, -6, ...], 'type': 'W', 'info': 3.20, 'weight': 0.73}, {...}}
    return pose_dict


def parse_psi_pssm(file, pose_dict):
    # take the contents of protein.pssm, parse file, and input into pose_dict
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
            info = float(line[172:177].strip())
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
        pose_dict[resi]['log'] = residue[2]
        pose_dict[resi]['type'] = residue[1]
        pose_dict[resi]['info'] = residue[4]
        pose_dict[resi]['weight'] = residue[5]

    # Output: {1: {'A': 0, 'R': 0, ..., 'log': [-5, -5, -6, ...], 'type': 'W', 'info': 3.20, 'weight': 0.73}, {...}}
    return pose_dict


def make_pssm_file(pssm_dict, name):
    header = '\n\n            A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   A   R' \
             '   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V\n'
    footer = ''
    out_file = name + '.pssm'
    if type(pssm_dict[0]['A']) == float:
        freq = True
    else:
        freq = False

    with open(out_file, 'w') as file:
        file.write(header)
        for res in pssm_dict:
            aa_type = pssm_dict[res]['type']
            log_string = ''
            for aa in pssm_dict[res]['log']:
                log_string += '{:>3d} '.format(aa)
            counts_string = ''
            if freq:
                for aa in fc.alph_3_aa_list:  # ensure the order never changes... this is true for python 3.6+
                    counts_string += '{:>3.0f} '.format(math.ceil(pssm_dict[res][aa] * 100))
            else:
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


def combine_pssm(pssm1, pssm2):
    temp = {}
    new_key = len(pssm1)
    for old_key in pssm2:
        temp[new_key] = pssm2[old_key]
        new_key += 1
    pssm1.update(temp)

    return pssm1


def duplicate_ssm(pssm_dict, copies):  # UNUSED
    duplicated_ssm = {}
    duplication_start = len(pssm_dict)
    for i in range(int(copies)):
        if i == 0:
            offset = 0
        else:
            offset = duplication_start * i
        # j = 0
        for line in pssm_dict:
            duplicated_ssm[line + offset] = pssm_dict[line]
            # j += 1

    return duplicated_ssm


def parse_cluster_info(file):
    # TODO
    return None


def get_all_clusters(pdb, residue_cluster_id_list, db=bio_fragmentDB):
    # generate an interface specific scoring matrix from the fragment library
    # assuming residue_cluster_id_list has form [(1_2_24, [78, 87]), ...]
    cluster_list = []
    for cluster in residue_cluster_id_list:
        cluster_loc = cluster[0].split('_')
        filename = os.path.join(db, cluster_loc[0], cluster_loc[0] + '_' + cluster_loc[1], cluster_loc[0] +
                                '_' + cluster_loc[1] + '_' + cluster_loc[2], cluster[0] + '.pkl')
        res1 = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[0], cluster[1][0]))
        res2 = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[1], cluster[1][1]))
        with open(filename, 'rb') as f:
            cluster_list.append([[res1.ca, res2.ca], pickle.load(f)])

    # OUTPUT format: [[[residue1_ca_atom, residue2_ca_atom], {'IJKClusterDict - such as 1_2_45'}], ...]
    # DEPRECIATED OUTPUT format: [[[78, 87], {'IJKClusterDict - such as 1_2_45'}], ...]
    return cluster_list


def convert_to_frag_dict(interface_residue_list):
    # Make PDB/ATOM objects and dictionary into dictionary
    # INPUT format: interface_residue_list = [[[residue1_ca_atom, residue2_ca_atom], {'IJKClusterDict - such as 1_2_45'}
    #                                          ], [[64, 87], {...}], ...]
    # component_chains = [pdb.chain_id_list[0], pdb.chain_id_list[-1]]
    interface_residue_dict = {}
    for residue_dict_pair in interface_residue_list:
        residues = residue_dict_pair[0]
        for i in range(len(residues)):
            residues[i] = residues[i].residue_number
        hash_ = (residues[0], residues[1])
        interface_residue_dict[hash_] = residue_dict_pair[1]

    # OUTPUT format: interface_residue_dict = {(78, 256): {'IJKClusterDict - such as 1_2_45'}, (64, 256): {...}, ...}
    return interface_residue_dict


def convert_to_rosetta_num(pdb, pose, interface_residue_list):
    # DEPRECIATED in favor of updating PDB/ATOM objects
    # INPUT format: interface_residue_list = [[[78, 87], {'IJKClusterDict - such as 1_2_45'}], [[64, 87], {...}], ...]
    component_chains = [pdb.chain_id_list[0], pdb.chain_id_list[-1]]
    interface_residue_dict = {}
    for residue_dict_pair in interface_residue_list:
        residues = residue_dict_pair[0]
        dict_ = residue_dict_pair[1]
        new_key = []
        pair_index = 0
        for chain in component_chains:
            new_key.append(pose.pdb_info().pdb2pose(chain, residues[pair_index]))
            pair_index = 1
        hash_ = (new_key[0], new_key[1])

        interface_residue_dict[hash_] = dict_

    # OUTPUT format: interface_residue_dict = {(78, 256): {'IJKClusterDict - such as 1_2_45'}, (64, 256): {...}, ...}
    return interface_residue_dict


def convert_to_rosetta_num_v2(pdb, residue_list):

    return None


def get_residue_ca_list(pdb, residue_list, chain=0):
    if chain == 0:
        chain = pdb.chain_id_list[0]
    residues = []
    for residue in residue_list:
        res_atoms = PDB.Residue(pdb.getResidueAtoms(chain, residue))
        res_ca = res_atoms.ca
        residues.append(res_ca)

    return residues


def deconvolve_clusters(full_cluster_dict, full_design_dict):
    # INPUT format: full_design_dict = {0: {-2: {},... }, ...}
    # [[[residue1_ca_atom, residue2_ca_atom], {'IJKClusterDict - such as 1_2_45'}], ...]
    # INPUT format: full_cluster_dict = {[residue1_ca_atom, residue2_ca_atom]: {'size': cluster_member_count, 'rmsd':
    #                                                                           mean_cluster_rmsd, 'rep':
    #                                                                           str(cluster_rep), 'mapped':
    #                                                                           mapped_freq_dict, 'paired':
    #                                                                           partner_freq_dict}
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
                            if full_design_dict[residue][frag_index][i]:
                                continue
                        except KeyError:
                            full_design_dict[residue][frag_index][i] = aa_freq
                            break
            designed_residues.append(fragment_data[1])

    return full_design_dict


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
            else:
                full_cluster_dict[residue][aa] = round(full_cluster_dict[residue][aa], 3)
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
    # PSSM Input - {1: {'A': 0, 'R': 0, 'N': 0, ..., 'type': 'W', 'log': [-5, -5, -6, ...], 'info': 3.20, 'weight': 0.73}, {...}}
    # ISSM Input - {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'stats': [0, 0.274]}, 50: {...}, ...}
    alpha = 0.5
    beta = 1 - alpha
    for entry in pssm:
        for aa in fc.alph_aa_list:
            pssm[entry][aa]
    full_ssm = []
    for entry in issm:
        fun = 2

    return full_ssm


def read_pdb(file):
    pdb = PDB.PDB()
    pdb.readfile(file)

    return pdb


def combine_pdb(pdb_1, pdb_2, name):  # UNUSED
    # Take two pdb objects and write them to the same file
    pdb_1.write(name)
    with open(name, 'a') as full_pdb:
        for atom in pdb_2.all_atoms:
            full_pdb.write(str(atom))  # .strip() + '\n')


def identify_interface_chains(pdb1, pdb2):  # UNUSED
    distance = 12  # Angstroms
    pdb1_chains = []
    pdb2_chains = []
    # Get Queried CB Tree for all PDB2 Atoms within 12A of PDB1 CB Atoms
    query = construct_cb_atom_tree(pdb1, pdb2, distance)

    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].tolist() != list():
            pdb2_chains.append(pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].chain)
            for pdb1_query_index in query[pdb2_query_index]:
                pdb1_chains.append(pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].chain)

    pdb1_chains = list(set(pdb1_chains))
    pdb2_chains = list(set(pdb2_chains))

    return pdb1_chains, pdb2_chains


def construct_cb_atom_tree(pdb1, pdb2, distance):
    # Get CB Atom Coordinates
    pdb1_coords = np.array(pdb1.extract_CB_coords())
    pdb2_coords = np.array(pdb2.extract_CB_coords())

    # Construct CB Tree for PDB1
    pdb1_tree = sklearn.neighbors.BallTree(pdb1_coords)

    # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
    query = pdb1_tree.query_radius(pdb2_coords, distance)

    # Map Coordinates to Atoms
    pdb1_cb_indices = pdb1.get_cb_indices()
    pdb2_cb_indices = pdb2.get_cb_indices()

    return query, pdb1_cb_indices, pdb2_cb_indices


def find_interface_residues(pdb1, pdb2):
    distance = 8
    # Get Queried CB Tree for all PDB2 Atoms within 8A of PDB1 CB Atoms
    query, pdb1_cb_indices, pdb2_cb_indices = construct_cb_atom_tree(pdb1, pdb2, distance)

    # Map Coordinates to Residue Numbers
    residues1 = []
    residues2 = []
    for pdb2_index in range(len(query)):
        if query[pdb2_index].tolist() != list():
            pdb2_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_index]].residue_number
            residues2.append(pdb2_res_num)
            for pdb1_index in query[pdb2_index]:
                pdb1_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_index]].residue_number
                residues1.append(pdb1_res_num)
    residues1 = sorted(set(residues1), key=int)
    residues2 = sorted(set(residues2), key=int)

    return residues1, residues2


def reduce_pose_to_interface(pdb, chains):  # UNUSED
    new_pdb = PDB.PDB()
    new_pdb.read_atom_list(pdb.chains(chains))

    return new_pdb


def find_jump_site(pdb):
    # gets the residue between chain 1 and chain 2
    chains = pdb.chain_id_list
    jump_start = pdb.getTermCAAtom('C', chains[0]).residue_number
    jump_end = pdb.getTermCAAtom('N', chains[1]).residue_number
    # jumps = (jump_start, jump_end)
    jump = jump_start

    return jump


def sdf_lookup(point_type):
    # TODO
    return '0'


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


def print_atoms(atom_list):
    for residue in atom_list:
        for atom in residue:
            print(str(atom))


def main(design_directory, frag_database, sym, debug=False):
    # Variable initialization
    start_cmd = ['echo', 'SymDesign: Starting Design of Directory "' + design_directory +
                 '"\nSymDesign: Symmetry Option ' + sym]
    refine_process = subprocess.Popen(start_cmd)
    script_cmd = [os.path.join(rosetta, 'source/bin/rosetta_scripts.python.linuxgccrelease'), '-database',
                  os.path.join(rosetta, 'database')]
    flag_variables = '-parser:script_vars'

    cst_value = 0.4
    frag_size = 5
    lower_bound, upper_bound, index_offset = parameterize_frag_length(frag_size)

    symm = sdf_lookup(sym)
    symmetry_variables = [['sdf', symm], ['-symmetry_definition', 'CRYST1']]
    if sym in ['3', '2']:
        if sym == '3':
            protocol = xtal_protocol
        elif sym == '2':
            protocol = layer_protocol
    else:
        protocol = point_protocol
        print('Not possible to input point groups just yet...')
        sys.exit()

    # Extract information from SymDock Output
    paths = []
    for root, dirs, files, in os.walk(design_directory):
        for file in files:
            if file.endswith('central_asu.pdb'):
                template_pdb_name = 'central_asu'
                template_pdb_path = root
                template_pdb = read_pdb(os.path.join(root, file))
                asu_com = template_pdb.center_of_mass()

            if fnmatch.fnmatch(file, '*_parsed_orient_tx_*'):
                paths.append((root, file))
                if len(paths) == 2:
                    oligomer1 = read_pdb(os.path.join(paths[0][0], paths[0][1]))
                    oligomer2 = read_pdb(os.path.join(paths[1][0], paths[1][1]))
                    oligomer1_name = oligomer1.filepath.split('/')[-1].split('_')[0]
                    oligomer2_name = oligomer2.filepath.split('/')[-1].split('_')[0]
                    oligomer1_com = oligomer1.center_of_mass()
                    oligomer2_com = oligomer2.center_of_mass()
                    int_residues1, int_residues2 = find_interface_residues(oligomer1, oligomer2)
                    # copies1 = len(oligomer1.chain_id_list)
                    # copies2 = len(oligomer2.chain_id_list)
                    olig_seq1, garbage1 = fc.extract_aa_seq(oligomer1)
                    olig_seq2, garbage2 = fc.extract_aa_seq(oligomer2)

            if file == 'frag_match_info_file.txt':
                # TODO Clean output/input?
                with open(os.path.join(root, file), 'r') as f:
                    frag_match_info_file = f.readlines()
                    residue_cluster_list = []
                    for line in frag_match_info_file:
                        if line[:12] == 'Cluster ID: ':
                            cluster = line[12:].strip()
                        if line[:43] == 'Surface Fragment Oligomer1 Residue Number: ':
                            res_chain1 = int(line[43:].strip())
                        if line[:43] == 'Surface Fragment Oligomer2 Residue Number: ':
                            res_chain2 = int(line[43:].strip())
                            residue_cluster_list.append((cluster, [res_chain1, res_chain2]))
                        if line[:15] == 'CRYST1 RECORD: ' and symmetry in ['2', '3']:
                            cryst = line[15:].strip()
                            for var in symmetry_variables[1]:
                                script_cmd.append(var)

    print('\nInput PDBs:', oligomer1_name.upper(), oligomer2_name.upper(), '\nPulling fragment info from clusters:',
          residue_cluster_list, '\nSymmetry information:', cryst)

    # TODO JOSH need to make this input from the docking program
    # TODO Coordinate with Josh on which chain comes first in ASU
    pdb_seq1, errors1 = fc.extract_aa_seq(template_pdb)
    pdb_seq2, errors2 = fc.extract_aa_seq(template_pdb, chain=template_pdb.chain_id_list[-1])

    # REMOVE if Josh can format upstream. Or get rid of fasta/PSSM name <- I think this is necessary to give good output
    alignment1 = pairwise2.align.globalxx(pdb_seq1, olig_seq1)
    alignment2 = pairwise2.align.globalxx(pdb_seq1, olig_seq2)
    if alignment1[0][2] > alignment2[0][2]:
        name1 = oligomer1_name
        name2 = oligomer2_name
        # switch = False
        # print(olig_seq1)
        # print(pdb_seq1)
    else:
        name1 = oligomer2_name
        name2 = oligomer1_name
        # switch = True
        # print(olig_seq1)
        # print(pdb_seq2)

    pdb_seq_file1 = fc.write_fasta_file(pdb_seq1, name1)
    pdb_seq_file2 = fc.write_fasta_file(pdb_seq2, name2)
    if pdb_seq_file1 and pdb_seq_file2:
        error_string = 'Sequence generation ran into the following residue errors: '
        if errors1:
            print(error_string, errors1)
        if errors2:
            print(error_string, errors2)
        full_pdb_sequence = pdb_seq1 + pdb_seq2  # chain1 (smaller oligomer) first then chain2 (bigger)
    else:
        print('Error parsing sequence files')
        sys.exit()

    # Get ASU distance parameters TODO
    com_asu_com_olig1_dist = fc.distance(asu_com, oligomer1_com)
    com_asu_com_olig2_dist = fc.distance(asu_com, oligomer2_com)
    if com_asu_com_olig1_dist >= com_asu_com_olig2_dist:
        dist = math.ceil(com_asu_com_olig1_dist)
    else:
        dist = math.ceil(com_asu_com_olig2_dist)
    dist = round(math.sqrt(dist), 0)  # added for first tests as this was a reasonable amount at 6A TEST TODO

    # Fetch IJK Cluster Dictionaries and Setup Residue Specific Data for Residue Number Conversion
    frag_residue_list = get_all_clusters(template_pdb, residue_cluster_list, frag_database)
    int_residue_ca1 = get_residue_ca_list(template_pdb, int_residues1)
    int_residue_ca2 = get_residue_ca_list(template_pdb, int_residues2, chain=template_pdb.chain_id_list[-1])
    # for item in frag_residue_list:
    #    print_atoms(item)
    # print_atoms(int_residue_atoms1)
    # print_atoms(int_residue_atoms2)

    # Convert to Rosetta Numbering because PSSM format (1, 2, ... N) works with Rosetta pose # instead of PDB #
    template_pdb.reorder_chains()
    template_pdb.renumber_residues()
    cleaned_pdb = 'clean_asu.pdb'
    template_pdb.write(cleaned_pdb, cryst1=cryst)
    # for item in frag_residue_list:
    #     print_atoms(item)
    # print_atoms(int_residue_atoms1)
    # print_atoms(int_residue_atoms2)

    # Set Up Interfaces and Jumps for Residue Specific Interface Design
    interface_residues = int_residue_ca1 + int_residue_ca2
    for i in range(len(interface_residues)):
        interface_residues[i] = interface_residues[i].residue_number
    jump = find_jump_site(template_pdb)

    print('\nInterface Residues:', interface_residues, '\nJump Site:', jump)

    # Prepare Command, and @flags file then run RosettaScripts Refine on Cleaned ASU #TEST
    refined_pdb = os.path.join(design_directory, 'rosetta_pdbs', 'refined_asu.pdb')
    refine_cmd = copy.deepcopy(script_cmd)
    _flags = copy.deepcopy(flags)
    refine_flags = ['-constrain_relax_to_start_coords', '-use_input_sc', '-relax:ramp_constraints false',
                    '-relax:coord_constrain_sidechains', '-relax:coord_cst_stdev 0.5', '-no_optH false', '-flip_HNQ',
                    '-nblist_autoupdate true', '-mute all', '-unmute protocols.rosetta_scripts.ParsedProtocol']
                     # try -relax:bb_move false
    for variable in refine_flags:
        _flags.append(variable)

    variables_for_flagsrefine_file = copy.copy(flag_variables)
    refine_variables = [('pdb_reference', cleaned_pdb), ('dist', dist)]
    for variable, value in refine_variables:
        variables_for_flagsrefine_file += ' ' + str(variable) + '=' + str(value)

    refine_flags.append(variables_for_flagsrefine_file)

    input_structure_refine = ['-in:file:s', cleaned_pdb, '@flagsRefine', '-parser:protocol', protocol[0], '-out:file:o',
                              refined_pdb]
    for variable in input_structure_refine:
        refine_cmd.append(variable)

    if not debug:
        with open('flagsRefine', 'w') as flag_file:
            for flag in refine_flags:
                flag_file.write(flag + '\n')

        print('Refine Command:', subprocess.list2cmdline(refine_cmd))
        refine_process = subprocess.Popen(refine_cmd)

    # Generate Constraints File
    # cst_file_name = template_pdb_name + '.cst'
    # cst_cmd = []
    # subprocess.call(cst_cmd)

    # Make PSSM of PDB sequence
    print('Getting PSSM files...')
    # pssm_file1, pssm_process1 = make_pssm_psiblast(name1)  # , True)
    # pssm_file2, pssm_process2 = make_pssm_psiblast(name2)  # , True)
    pssm_file1, pssm_process1 = make_pssm_hhblits(name1)
    pssm_file2, pssm_process2 = make_pssm_hhblits(name2)
    config_pssm_dict1 = populate_design_dict(len(pdb_seq1), fc.alph_3_aa_list)
    config_pssm_dict2 = populate_design_dict(len(pdb_seq2), fc.alph_3_aa_list)

    # Parse Fragment Clusters into usable Dictionaries and Flatten for Sequence Design
    frag_dict = convert_to_frag_dict(frag_residue_list)
    full_design_dict = populate_design_dict(len(full_pdb_sequence), [i for i in range(lower_bound, upper_bound + 1)])
    full_cluster_dict = deconvolve_clusters(frag_dict, full_design_dict)
    flattened_cluster_dict = flatten_for_issm(full_cluster_dict)
    print(flattened_cluster_dict)

    # WAIT for PSSM command to complete, then move on
    pssm_process1.communicate()
    pssm_dict1 = parse_hhblits_pssm(pssm_file1, config_pssm_dict1)
    # pssm_dict1 = parse_psi_pssm(pssm_file1, config_pssm_dict1)  # index is 0 start
    pssm_process2.communicate()
    pssm_dict2 = parse_hhblits_pssm(pssm_file2, config_pssm_dict2)
    # pssm_dict2 = parse_psi_pssm(pssm_file2, config_pssm_dict2)  # index is 0 start

    # Combine PSSM1, 2 into single PSSM
    full_pssm = combine_pssm(pssm_dict1, pssm_dict2)  # chain1 (smaller oligomer) first then chain2 (bigger)
    full_pssm_file = make_pssm_file(full_pssm, template_pdb_name)

    # Make ISSM from Rosetta Numbered PDB format, fragment list, fragment library
    # TODO Take counts PSSM and counts ISSM and combine into a single PSSM
    # Is this next function necessary? Can this be condensed into the combine_sm()
    # full_issm = make_issm(flattened_cluster_dict, interface_background)
    final_pssm = combine_sm(full_pssm, flattened_cluster_dict)

    # WAIT for Rosetta Refine command to complete, then move on
    refine_process.communicate()

    # Prepare Command, and @flags file then run RosettaScripts Design
    design_cmd = copy.deepcopy(script_cmd)
    design_flags = copy.deepcopy(flags)
    input_structure_design = ['-in:file:s', refined_pdb, '@flags', '-parser:protocol', protocol[1]]
    for variable in input_structure_design:
        design_cmd.append(variable)
    design_variables = [('interface_separator', jump), ('pdb_reference', refined_pdb), ('dist', dist),
                        ('pssm_file', full_pssm_file), ('interface', ','.join(str(i) for i in interface_residues)),
                        ('cst_value', cst_value)]
    # ('pdb_reference', template_pdb.filepath), ('cst_path', cst_file_name)]

    variables_for_flag_file = '-parser:script_vars'
    for variable, value in design_variables:
        variables_for_flag_file += ' ' + str(variable) + '=' + str(value)
    design_flags.append(variables_for_flag_file)

    print('Design Command:', subprocess.list2cmdline(design_cmd))
    print('Flags:', subprocess.list2cmdline(design_flags))

    sys.exit()

    if not debug:
        with open('flags', 'w') as flag_file:
            for flag in design_flags:
                flag_file.write(flag + '\n')

        subprocess.call(design_cmd)


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        directory = sys.argv[1]

        if sys.argv[2] == 'bio':
            database = bio_fragmentDB
        elif sys.argv[2] == 'xtal':
            database = xtal_fragmentDB
        elif sys.argv[2] == 'combined':
            database = full_fragmentDB
        else:
            database = bio_fragmentDB

        if sys.argv[3] not in ['0', '2', '3']:
            print('Please define type of symmetry')
            sys.exit() #TODO
        else:
            symmetry = sys.argv[3]
        try:
            if sys.argv[4]:
                debug_ = True
        except IndexError:
            debug_ = False

        main(directory, database, symmetry, debug_)
    else:
        print('USAGE fragment_design_MASTER.py design directory fragment_database_type[bio,xtal,combined] '
              'symmetry_type[0,2,3] debug')
        sys.exit()
