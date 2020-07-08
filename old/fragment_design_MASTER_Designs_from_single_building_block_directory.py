#!/home/kmeador/miniconda3/bin/python
'''
Before full run need to alleviate the following tags and incompleteness
-TEST - ensure working and functional against logic and coding errors
-OPTIMIZE - make the best protocol
-TODO - Make this resource
-JOSH - coordinate on input style from Josh's output

Design Decisions:
-We need to have the structures relaxed before input into the design program. Should relaxation be done on individual
protein building blocks? This again gets into the SDF issues. But for rosetta scoring, we need a baseline to be relaxed.
        -currently relaxed in unit cell
-Should the sidechain info for each of the interface positions be stripped or made to ALA upon rosetta start up?
        -currently YES

'''
import os
import sys
import math
import subprocess
import pickle
import shutil
import copy
import fnmatch
import numpy as np
import sklearn.neighbors
from Bio import pairwise2
import PDB
import functions as fc
# from PDB import PDB

# PATHS #
# TODO Make rosetta_scripts operating system/build independent
module = 'SymDesign |'
alignmentdb = '/home/kmeador/local_programs/ncbi_databases/uniref90'
uniclustdb = '/home/kmeador/local_programs/hh-suite/build/uniclust30_2018_08/uniclust30_2018_08'
source = '/home/kmeador/yeates/symdesign'
fragmentDB = '/home/kmeador/yeates/fragment_database'
bio_fragmentDB = '/home/kmeador/yeates/fragment_database/bio'
xtal_fragmentDB = '/home/kmeador/yeates/fragment_database/xtal'
full_fragmentDB = '/home/kmeador/yeates/fragment_database/bio+xtal'
rosetta = '/joule2/programs/rosetta/rosetta_src_2019.35.60890_bundle/main'
rosetta_pdbs_outdir = 'rosetta_pdbs'
scores_outdir = 'scores'
xtal_protocol = (os.path.join(source, 'rosetta_scripts/xtal_refine.xml'),
                 os.path.join(source, 'rosetta_scripts/xtal_design.xml'))
layer_protocol = (os.path.join(source, 'rosetta_scripts/layer_refine.xml'),
                  os.path.join(source, 'rosetta_scripts/layer_design.xml'))
point_protocol = (os.path.join(source, 'rosetta_scripts/point_refine.xml'),
                  os.path.join(source, 'rosetta_scripts/point_design.xml'))


# GLOBALS #
# TODO Modify all of these during individual users program set up
start_cmd = ['echo', module, 'Starting Design of Directory ' + os.getcwd()]
script_cmd = [os.path.join(rosetta, 'source/bin/rosetta_scripts.python.linuxgccrelease'), '-database',
              os.path.join(rosetta, 'database')]
flags = ['-ex1', '-ex2', '-linmem_ig 10', '-ignore_unrecognized_res', '-ignore_zero_occupancy false', '-overwrite',
         '-extrachi_cutoff 5', '-out:path:pdb rosetta_pdbs/', '-out:path:score scores/', '-no_his_his_pairE',
         '-chemical:exclude_patches LowerDNA UpperDNA Cterm_amidation SpecialRotamer ' 
         'VirtualBB ShoveBB VirtualDNAPhosphate VirtualNTerm CTermConnect sc_orbitals pro_hydroxylated_case1 '
         'pro_hydroxylated_case2 ser_phosphorylated thr_phosphorylated tyr_phosphorylated tyr_sulfated '
         'lys_dimethylated lys_monomethylated  lys_trimethylated lys_acetylated glu_carboxylated cys_acetylated '
         'tyr_diiodinated N_acetylated C_methylamidated MethylatedProteinCterm']
refine_suffix = '_refined'
refine_options = ['-constrain_relax_to_start_coords', '-use_input_sc', '-relax:ramp_constraints false',
                  '-relax:coord_constrain_sidechains', '-relax:coord_cst_stdev 0.5', '-no_optH false', '-flip_HNQ',
                  '-nblist_autoupdate true', '-mute all', '-unmute protocols.rosetta_scripts.ParsedProtocol',
                  '-out:suffix ' + refine_suffix, '-relax:bb_move false']
design_options = ['-mute all', '-unmute protocols.rosetta_scripts.ParsedProtocol', '-out:suffix _design']
                  # -holes:dalphaball
index_offset = -1
interface_background = {}
# TODO find the interface background from some source. Initial thoughts are from fragment


#### MODULE FUNCTIONS #### TODO
# PDB.mutate_to_ala()
# PDB.reorder_chains()
# PDB.renumber_residues()
# PDB.write()
# fc.extract_aa_seq()
# fc.alph_3_aa_list
# fc.alph_aa_list
# fc.write_fasta_file()
# np.linalg.norm()


#### CODE FORMATTING ####
# PSSM format is dictating the corresponding file formatting for Rosetta
# aa_counts_dict and PSSM format are in different alphabetical orders (1v3 letter), this may be a problem down the line


def reduce_pose_to_interface(pdb, chains):  # UNUSED
    new_pdb = PDB.PDB()
    new_pdb.read_atom_list(pdb.chains(chains))

    return new_pdb


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


def rosetta_score(pdb):  # UNUSED
    # this will also format your output in rosetta numbering
    cmd = [rosettaBIN, 'score_jd2.default.linuxgccrelease', '-renumber_pdb', '-ignore_unrecognized_res', '-s', pdb,
           '-out:pdb']
    subprocess.Popen(cmd, start_new_session=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    return pdb + '_0001.pdb'


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


def deconvolve_clusters_old(full_cluster_dict, full_design_dict):  # UNUSED DEPRECIATED
    # [[[residue1_ca_atom, residue2_ca_atom], {'IJKClusterDict - such as 1_2_45'}], ...]
    # full_cluster_dict FORMAT: = {[residue1_ca_atom, residue2_ca_atom]: {'size': cluster_member_count, 'rmsd':
    #                                                                           mean_cluster_rmsd, 'rep':
    #                                                                           str(cluster_rep), 'mapped':
    #                                                                           mapped_freq_dict, 'paired':
    #                                                                           partner_freq_dict}
    # aa_freq format: mapped/paired_freq_dict = {-2: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37]}, -1:...}
    # where 'stats'[0] is total fragments in cluster, and 'stats'[1] is weight of fragment index
    # full_design_dict FORMAT: = {0: {-2: {},... }, ...}

    designed_residues = []
    for pair in full_cluster_dict:
        mapped = pair[0]
        paired = pair[1]
        mapped_dict = full_cluster_dict[pair]['mapped']
        paired_dict = full_cluster_dict[pair]['paired']
        for fragment_data in [(mapped_dict, mapped), (paired_dict, paired)]:
            for frag_index in fragment_data[0]:
                aa_freq = fragment_data[0][frag_index]  # This has weights in it...
                residue = fragment_data[1] + frag_index + index_offset  # make zero index so design_dict starts at 0
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


def get_all_cluster(pdb, residue_cluster_id_list, db=bio_fragmentDB):  # UNUSED DEPRECIATED
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
    return cluster_list


def convert_to_frag_dict(interface_residue_list, cluster_dict):  #UNUSED
    # Make PDB/ATOM objects and dictionary into design dictionary
    # INPUT format: interface_residue_list = [[[Atom_ca.residue1, Atom_ca.residue2], '1_2_45'], ...]
    interface_residue_dict = {}
    for residue_dict_pair in interface_residue_list:
        residues = residue_dict_pair[0]
        for i in range(len(residues)):
            residues[i] = residues[i].residue_number
        hash_ = (residues[0], residues[1])
        interface_residue_dict[hash_] = cluster_dict[residue_dict_pair[1]]
    # OUTPUT format: interface_residue_dict = {(78, 256): {'IJKClusterDict - such as 1_2_45'}, (64, 256): {...}, ...}
    return interface_residue_dict


def convert_to_rosetta_num(pdb, pose, interface_residue_list):  # UNUSED
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


def get_residue_list_atom(pdb, residue_list, chain=None):  # UNUSED DEPRECIATED
    if chain is None:
        chain = pdb.chain_id_list[0]
    residues = []
    for residue in residue_list:
        res_atoms = PDB.Residue(pdb.getResidueAtoms(chain, residue))
        residues.append(res_atoms)

    return residues


def parse_cluster_info(file):  # UNUSED
    # TODO
    return file


def modify_pssm_background(pssm, background):  # UNUSED
    # TODO make a function which unpacks pssm and refactors the lod score with specifc background
    return pssm, background


def make_issm(cluster_freq_dict, background):  # UNUSED
    for residue in cluster_freq_dict:
        for aa in cluster_freq_dict[residue]:
            cluster_freq_dict[residue][aa] = round(2 * (math.log((cluster_freq_dict[residue][aa] / background[aa]), 2)))
    issm = []

    return issm


def print_atoms(atom_list):  # UNUSED DEBUG PERTINENT
    for residue in atom_list:
        for atom in residue:
            print(str(atom))


def populate_design_dict(n, m):
    # Generate a dictionary of dictionaries with n elements and m subelements, where n is the number of residues and m
    # is the length of library. 0 indexed.
    design_dict = {residue: {i: {} for i in m} for residue in range(n)}

    return design_dict


def return_cluster_id_string(cluster_rep, index_number=3):
    while len(cluster_rep) < 3:
        cluster_rep += '0'
    if len(cluster_rep.split('_')) != 3:
        index = [cluster_rep[:1], cluster_rep[1:2], cluster_rep[2:]]
    else:
        index = cluster_rep.split('_')

    info = []
    n = 0
    for i in range(index_number):
        info.append(index[i])
        n += 1
    while n < 3:
        info.append('0')
        n += 1

    return '_'.join(info)


def get_cluster_dicts(info_db=bio_fragmentDB, id_list=None):
    # generate an interface specific scoring matrix from the fragment library
    # assuming residue_cluster_id_list has form [(1_2_24, [78, 87]), ...]
    if id_list is None:
        directory_list = get_all_base_root_paths(info_db)
    else:
        directory_list = []
        for _id in id_list:
            c_id = _id.split('_')
            _dir = os.path.join(info_db, c_id[0], c_id[0] + '_' + c_id[1], c_id[0] + '_' + c_id[1] + '_' + c_id[2])
            directory_list.append(_dir)

    cluster_dict_dict = {}
    for cluster in directory_list:
        filename = os.path.join(cluster, os.path.basename(cluster) + '.pkl')
        with open(filename, 'rb') as f:
            cluster_dict_dict[os.path.basename(cluster)] = pickle.load(f)

    # OUTPUT format: {'1_2_45': {'size': ..., 'rmsd': ..., 'rep': ..., 'mapped': ..., 'paired': ...}, ...}
    return cluster_dict_dict


def make_residues_pdb_object(pdb, residue_dict):  # , cluster_dict):
    # Pair residues with cluster info
    # residue_dict format: {'1_2_24': [78, 87], ...}
    # cluster_dict format: {'1_2_45': {'size': ..., 'rmsd': ..., 'rep': ..., 'mapped': ..., 'paired': ...}, ...}
    cluster_list = []
    for cluster in residue_dict:
        res1 = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[0], residue_dict[cluster][0]))
        res2 = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[-1], residue_dict[cluster][1]))
        cluster_list.append([[res1.ca, res2.ca], cluster])

    # OUTPUT format: [[[residue1_ca_atom, residue2_ca_atom], '1_2_45'}], ...]
    return cluster_list


def get_db_statistics():
    for file in os.listdir(database):
        if file.endswith('statistics.pkl'):
            with open(os.path.join(database, file), 'rb') as f:
                stats = pickle.load(f)

    return stats


def convert_to_residue_cluster_map(interface_residue_list):
    # INPUT format: interface_residue_list = [[[Atom_ca.residue1, Atom_ca.residue2], '1_2_45'], ...]
    cluster_map = {}
    for residue_dict_pair in interface_residue_list:
        residues = residue_dict_pair[0]
        for i in range(len(residues)):  # 0, 1
            # if residues[i].residue_number not in map:
            #     map[residues[i].residue_number] = {}
            #     if i == 0:
            #         map[residues[i].residue_number]['chain'] = 'mapped'
            #     else:
            #         map[residues[i].residue_number]['chain'] = 'paired'
            #     map[residues[i].residue_number]['cluster'] = []
            # map[residues[i].residue_number]['cluster'].append(residue_dict_pair[1])
            # Next for each residue in map add the same cluster to -2 to 2 residue numbers
            residue_number = residues[i].residue_number + index_offset  # make zero index so residue starts at 0
            for j in range(-2, 3):
                if residue_number + j not in cluster_map:
                    cluster_map[residue_number + j] = {}
                    if i == 0:
                        cluster_map[residue_number + j]['chain'] = 'mapped'
                    else:
                        cluster_map[residue_number + j]['chain'] = 'paired'
                    cluster_map[residue_number + j]['cluster'] = []
                cluster_map[residue_number + j]['cluster'].append((j, residue_dict_pair[1]))

    # Cluster Mapping - {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
    return cluster_map


def deconvolve_clusters(full_cluster_dict, full_design_dict, cluster_map):
    # Map Format - {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
    # aa_freq format: mapped/paired_freq_dict = {-2: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37]}, -1:...}
    # where 'stats'[0] is total fragments in cluster, and 'stats'[1] is weight of fragment index
    # full_design_dict FORMAT: = {0: {-2: {},... }, ...}

    for resi in cluster_map:
        dict_type = cluster_map[resi]['chain']
        indices = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
        for index_cluster_pair in cluster_map[resi]['cluster']:
            aa_freq = full_cluster_dict[index_cluster_pair[1]][dict_type][index_cluster_pair[0]]  # This has weights in it...
            # Add the aa_freq at the residue/frag_index
            full_design_dict[resi][index_cluster_pair[0]][indices[index_cluster_pair[0]]] = aa_freq
            indices[index_cluster_pair[0]] += 1

    return full_design_dict


def flatten_for_issm(full_cluster_dict):
    # Take a multi-observation, mulit-fragment indexed design dict and reduce to single residue frequency for ISSM
    aa_counts_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
                      'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'stats': [0, 1]}
    no_design = []
    for res in full_cluster_dict:
        total_residue_weight = 0
        num_frag_weights_observed = 0
        for index in full_cluster_dict[res]:
            if full_cluster_dict[res][index] == dict():
                continue
            else:
                total_obs_weight = 0
                for obs in full_cluster_dict[res][index]:
                    total_obs_weight += full_cluster_dict[res][index][obs]['stats'][1]
                if total_obs_weight == 0:
                    # Case where no weights associated with observations (side chain not structurally significant)
                    full_cluster_dict[res][index] = dict()
                    continue

                obs_aa_dict = copy.deepcopy(aa_counts_dict)
                for obs in full_cluster_dict[res][index]:
                    num_frag_weights_observed += 1
                    obs_weight = full_cluster_dict[res][index][obs]['stats'][1]
                    for aa in full_cluster_dict[res][index][obs]:
                        if aa == 'stats':
                            continue
                        # Add all occurrences to summed frequencies list
                        obs_aa_dict[aa] += full_cluster_dict[res][index][obs][aa] * (obs_weight / total_obs_weight)
                obs_aa_dict['stats'][1] = total_obs_weight  # /2 <--TODO if fragments and their flip are present
                full_cluster_dict[res][index] = obs_aa_dict

            total_residue_weight += full_cluster_dict[res][index]['stats'][1]
        if total_residue_weight == 0:
            # Add to list for removal from the design dict
            no_design.append(res)
            continue

        res_aa_dict = copy.deepcopy(aa_counts_dict)
        res_aa_dict['stats'][1] = total_residue_weight
        res_aa_dict['stats'][0] = num_frag_weights_observed
        for index in full_cluster_dict[res]:
            if full_cluster_dict[res][index] == dict():
                continue
            index_weight = full_cluster_dict[res][index]['stats'][1]
            for aa in full_cluster_dict[res][index]:
                if aa == 'stats':
                    continue
                # Add all occurrences to summed frequencies list
                res_aa_dict[aa] += full_cluster_dict[res][index][aa] * (index_weight / total_residue_weight)
        full_cluster_dict[res] = res_aa_dict

    # Remove missing residues from dictionary
    for res in no_design:
        full_cluster_dict.pop(res)
    # for res in full_cluster_dict:
    #     remove_aa = []
    #     for aa in full_cluster_dict[res]:
    #         if aa == 'stats':
    #             continue
    #         if full_cluster_dict[res][aa] == 0.0:
    #             remove_aa.append(aa)
    #         else:
    #             full_cluster_dict[res][aa] = round(full_cluster_dict[res][aa], 3)
    #     for i in remove_aa:
    #         full_cluster_dict[res].pop(i)

    return full_cluster_dict


def combine_ssm(pssm, issm, cluster_mapping, weights=False):
    # combine the weights for the PSSM with the ISSM
    # HHblits PSSM Input: {0: {'A': 0.13, ..., 'log': [-5, ...], 'type': 'W', 'info': 0.00, 'weight': 0.00}, {...}}
    # PSIBLAST PSSM Input: {0: {'A': 13, ..., 'log': [-5, ...], 'type': 'W', 'info': 3.20, 'weight': 0.73}, {...}}
    # ISSM Input - {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'stats': [4, 0.274]}, 50: {...}, ...}
    # Cluster Mapping - {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
    # Stat Dict FORMAT  cluster_id mapped  paired  max_weight_counts
    # Stat Dictionary - {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]

    if weights:
        stat_dict = get_db_statistics()
    alpha = 0.5
    beta = 1 - alpha
    for entry in pssm:
        try:
            if weights:
                if cluster_mapping[entry]['chain'] == 'mapped':
                    i = 0
                else:
                    i = 1
                average_total, count = 0.0, 0
                for cluster in cluster_mapping[entry]['cluster']:
                    cluster_id = return_cluster_id_string(cluster[1], index_number=2)
                    average_total += stat_dict[cluster_id][0][i]
                    count += 1
                stats_average = average_total/count
                observations = issm[entry]['stats'][0]
                tot_entry_weight = issm[entry]['stats'][1]
                entry_ave_frag_weight = tot_entry_weight/count
                if entry_ave_frag_weight > stats_average:
                    alpha = 1
                else:
                    alpha = entry_ave_frag_weight/stats_average
                beta = 1 - alpha
            print(module, 'Residue', entry - index_offset, 'Alpha:', alpha)
            for aa in fc.alph_aa_list:
                pssm[entry][aa] = (alpha * issm[entry][aa]) + (beta * pssm[entry][aa])
        except KeyError as ex:
            # print(module, '[ERROR]', ex)
            continue

    return pssm


def make_pssm_psiblast(query, outpath=None, remote=False):
    # Generate an position specific scoring matrix using PSI-BLAST
    # I would like the background to come from Uniref90 instead of BLOSUM62 #TODO
    if outpath is not None:
        outfile_name = os.path.join(outpath, query + '.pssm')
        direct = outpath
    else:
        outfile_name = query + '.hmm'
        direct = os.getcwd()
    if query + '.pssm' in os.listdir(direct):
        cmd = ['echo', module, 'PSSM: ' + query + '.pssm already exists']
        p = subprocess.Popen(cmd)

        return outfile_name, p

    cmd = ['psiblast', '-db', alignmentdb, '-query', query + '.fasta', '-out_ascii_pssm', outfile_name,
           '-save_pssm_after_last_round', '-evalue', '1e-6', '-num_iterations', '0']
    if remote:
        cmd.append('-remote')
    else:
        cmd.append('-num_threads')
        cmd.append('8')
    print(module, query + 'Profile Command:', subprocess.list2cmdline(cmd))
    p = subprocess.Popen(cmd)

    return outfile_name, p


def make_pssm_hhblits(query, outpath=None):
    # Generate an position specific scoring matrix from HHblits using Hidden Markov Models
    if outpath is not None:
        outfile_name = os.path.join(outpath, query + '.hmm')
        direct = outpath
    else:
        outfile_name = query + '.hmm'
        direct = os.getcwd()
    if query + '.hmm' in os.listdir(direct):
        cmd = ['echo', module, 'PSSM: ' + query + '.hmm already exists']
        p = subprocess.Popen(cmd)

        return outfile_name, p

    cmd = ['hhblits', '-d', uniclustdb, '-i', query + '.fasta', '-ohhm', outfile_name, '-v', '1']
    cmd.append('-cpu')
    cmd.append('4')
    print(module, query + 'Profile Command:', subprocess.list2cmdline(cmd))
    p = subprocess.Popen(cmd)

    return outfile_name, p


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
        resi = residue[0] + index_offset
        i = 0
        for aa in pose_dict[resi]:
            pose_dict[resi][aa] = residue[3][i]
            i += 1
        pose_dict[resi]['log'] = residue[2]
        pose_dict[resi]['type'] = residue[1]
        pose_dict[resi]['info'] = residue[4]
        pose_dict[resi]['weight'] = residue[5]

    # Output: {0: {'A': 0, 'R': 0, ..., 'log': [-5, -5, -6, ...], 'type': 'W', 'info': 3.20, 'weight': 0.73}, {...}}
    return pose_dict


def parse_hhblits_pssm(file, pose_dict, null_background=True):
    # Take contents of protein.hmm, parse file and input into pose_dict. File is Single AA code alphabetical order
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
            aa_lod = (2 * math.log(aa_freq_dict[a]/bg_dict[a], 2))
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
                resi = int(items[1]) + index_offset  # make zero index so dict starts at 0
                i = 2
                for aa in fc.alph_aa_list:
                    pose_dict[resi][aa] = to_freq(items[i])
                    i += 1
                pose_dict[resi]['log'] = get_lod(pose_dict[resi], null_bg)
                pose_dict[resi]['type'] = items[0]
                pose_dict[resi]['info'] = dummy
                pose_dict[resi]['weight'] = dummy

    # Output: {0: {'A': 0.04, 'R': 0.12, ..., 'log': [-5, -5, ...], 'type': 'W', 'info': 3.20, 'weight': 0.73}, {...}}
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
            line = '{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}''\n'.format(res - index_offset, aa_type, log_string,
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


def read_pdb(file):
    pdb = PDB.PDB()
    pdb.readfile(file)

    return pdb


def get_residue_list_atoms(pdb, residue_list, chain=None):
    if chain is None:
        chain = pdb.chain_id_list[0]
    residues = []
    for residue in residue_list:
        res_atoms = pdb.getResidueAtoms(chain, residue)
        residues.append(res_atoms)

    return residues


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


def prepare_rosetta_flags(flag_options, flag_variables, stage):
    def make_flags_file(flag_list, subtype):
        with open('flags_' + subtype, 'w') as f:
            for flag in flag_list:
                f.write(flag + '\n')

        return 'flags_' + subtype

    _flags = copy.deepcopy(flags)
    for variable in flag_options:
        _flags.append(variable)

    variables_for_flag_file = '-parser:script_vars'
    for variable, value in flag_variables:
        variables_for_flag_file += ' ' + str(variable) + '=' + str(value)

    _flags.append(variables_for_flag_file)
    flag_file = make_flags_file(_flags, stage)

    print(module, 'Flags for', stage, 'stage:', subprocess.list2cmdline(_flags))

    return flag_file


def prepare_rosetta_scripts_cmd(input_cmds):
    _cmd = copy.deepcopy(script_cmd)
    for variable in input_cmds:
        _cmd.append(variable)

    return _cmd


def main(design_directory, frag_db, sym, debug=False):
    # Variable initialization
    refine_process = subprocess.Popen(start_cmd)
    # design_directory always has format building_blocks/DEGENX_X/ROT_Y/tx_X/. base_dir backs out to building_blocks
    base_dir = design_directory[:design_directory.find(design_directory.split('/')[-3]) - 1]
    seq_dir = os.path.join(base_dir, 'Sequence_Info')
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)

    cst_value = 0.4
    frag_size = 5
    lower_bound, upper_bound, offset = parameterize_frag_length(frag_size)
    symm = sdf_lookup(sym)
    symmetry_variables = [['sdf', symm], ['-symmetry_definition', 'CRYST1']]

    # Find design protocol based off symmetry
    if sym in ['3', '2']:
        if sym == '3':
            protocol = xtal_protocol
        elif sym == '2':
            protocol = layer_protocol
        for var in symmetry_variables[1]:
            script_cmd.append(var)
    else:
        protocol = point_protocol
        sym = symm
        print(module, '[ERROR] Not possible to input point groups just yet...')
        for var in symmetry_variables[0]:
            script_cmd.append(var)
        sys.exit()
    print(module, 'Symmetry Option:', sym)

    # Extract information from SymDock Output
    paths = []
    for root, dirs, files, in os.walk(design_directory):
        for file in files:
            if file.endswith('central_asu.pdb'):
                template_pdb_name = os.path.basename(file)
                template_pdb = read_pdb(os.path.join(root, file))

            if fnmatch.fnmatch(file, '*_parsed_orient_tx_*'):
                paths.append((root, file))
                if len(paths) == 2:
                    oligomer1 = read_pdb(os.path.join(paths[0][0], paths[0][1]))
                    oligomer1_name = os.path.basename(oligomer1.filepath).split('_')[0]
                    oligomer1_com = oligomer1.center_of_mass()
                    olig_seq1, error_dump1 = fc.extract_aa_seq(oligomer1)
                    oligomer2 = read_pdb(os.path.join(paths[1][0], paths[1][1]))
                    oligomer2_name = os.path.basename(oligomer2.filepath).split('_')[0]
                    oligomer2_com = oligomer2.center_of_mass()
                    olig_seq2, error_dump2 = fc.extract_aa_seq(oligomer2)
                    int_res_atoms1, int_res_atoms2 = find_interface_residues(oligomer1, oligomer2)

            if file == 'frag_match_info_file.txt':
                # JOSH needs to modify frag_match_info_file.txt to have cluster names as X_Y_Z TODO
                with open(os.path.join(root, file), 'r') as f:
                    frag_match_info_file = f.readlines()
                    residue_cluster_dict = {}
                    for line in frag_match_info_file:
                        if line[:12] == 'Cluster ID: ':
                            cluster = line[12:].strip()
                        if line[:43] == 'Surface Fragment Oligomer1 Residue Number: ':
                            # Always contains I fragment? #JOSH
                            res_chain1 = int(line[43:].strip())
                        if line[:43] == 'Surface Fragment Oligomer2 Residue Number: ':
                            # Always contains J fragment and Guide Atoms? #JOSH
                            res_chain2 = int(line[43:].strip())
                            residue_cluster_dict[cluster] = [res_chain1, res_chain2]
                        if line[:15] == 'CRYST1 RECORD: ' and symmetry in ['2', '3']:
                            cryst = line[15:].strip()

    print(module, 'Symmetry Information:', cryst)
    print(module, 'Input PDB\'s:', oligomer1_name.upper(), oligomer2_name.upper())
    print(module, 'Pulling fragment info from clusters:', ', '.join(residue_cluster_dict))

    # Prepare Output Directory/Files
    cleaned_pdb = 'clean_asu.pdb'
    ala_mut_refine_pdb = os.path.splitext(cleaned_pdb)[0] + '_for_refine.pdb'
    if not os.path.exists(os.path.join(design_directory, rosetta_pdbs_outdir)):
        os.makedirs(os.path.join(design_directory, rosetta_pdbs_outdir))
    if not os.path.exists(os.path.join(design_directory, scores_outdir)):
        os.makedirs(os.path.join(design_directory, scores_outdir))
    refined_pdb = os.path.join(design_directory, rosetta_pdbs_outdir, os.path.splitext(cleaned_pdb)[0] + refine_suffix
                               + '_0001.pdb')

    # Fetch IJK Cluster Dictionaries and Setup Interface Residues for Residue Number Conversion. MUST BE PRE-RENUMBER
    frag_residue_list = make_residues_pdb_object(template_pdb, residue_cluster_dict)
    int_res_atoms1 = get_residue_list_atoms(template_pdb, int_res_atoms1)
    int_res_atoms2 = get_residue_list_atoms(template_pdb, int_res_atoms2, chain=template_pdb.chain_id_list[-1])
    # Renumber PDB to Rosetta Numbering. PSSM format (1, 2, ... N) works best with Rosetta pose number
    template_pdb.reorder_chains()
    template_pdb.renumber_residues()
    template_pdb.write(cleaned_pdb, cryst1=cryst)
    # Set Up Interface Residues, Remove their Side Chain Atoms, and Find Jump Site
    int_res_numbers = [[], []]
    i = 0
    for chain in [int_res_atoms1, int_res_atoms2]:
        for residue_atoms in chain:
            int_res_numbers[i].append(residue_atoms[0].residue_number)
            template_pdb.mutate_to_ala(residue_atoms)
        i += 1

    int_res_numbers1 = int_res_numbers[0]
    int_res_numbers2 = int_res_numbers[1]
    int_res_numbers = np.concatenate(int_res_numbers)
    print(module, 'Interface Residues:', int_res_numbers)
    jump = template_pdb.getTermCAAtom('C', template_pdb.chain_id_list[0]).residue_number
    print(module, 'Jump Site:', jump)
    template_pdb.write(ala_mut_refine_pdb, cryst1=cryst)
    print(module, 'Cleaned PDB for Refine:', ala_mut_refine_pdb)

    # Get ASU distance parameters TODO FOR POINT GROUPS
    asu_com = template_pdb.center_of_mass()
    com_asu_com_olig1_dist = np.linalg.norm(np.array(asu_com) - np.array(oligomer1_com))
    com_asu_com_olig2_dist = np.linalg.norm(np.array(asu_com) - np.array(oligomer2_com))
    if com_asu_com_olig1_dist >= com_asu_com_olig2_dist:
        dist = math.ceil(com_asu_com_olig1_dist)
    else:
        dist = math.ceil(com_asu_com_olig2_dist)
    dist = round(math.sqrt(dist), 0)  # OPTIMIZE added for first tests as this was a reasonable amount at 6A
    print(module, 'Expand ASU by', dist, 'Angstroms')

    # REFINE: Prepare Command, and Flags File then run RosettaScripts on Cleaned ASU #OPTIMIZE
    refine_variables = [('pdb_reference', template_pdb_name), ('dist', dist)]

    if not debug:
        if not os.path.exists(refined_pdb):
            flagrefine = prepare_rosetta_flags(refine_options, refine_variables, 'refine')
            input_refine_cmds = ['-in:file:s', ala_mut_refine_pdb, '@' + flagrefine, '-parser:protocol', protocol[0]]
            refine_cmd = prepare_rosetta_scripts_cmd(input_refine_cmds)
            print(module, 'Refine Command:', subprocess.list2cmdline(refine_cmd))
            refine_process = subprocess.Popen(refine_cmd)

    # Extract/Format Sequence Information
    pdb_seq1, errors1 = fc.extract_aa_seq(template_pdb)
    pdb_seq2, errors2 = fc.extract_aa_seq(template_pdb, chain=template_pdb.chain_id_list[-1])

    # Check to see if other poses have already collected sequence info about design
    if template_pdb_name not in os.listdir(seq_dir):
        # Check alignment to assign correct PDB name to each sequence from ASU.pdb file
        alignment1 = pairwise2.align.globalxx(pdb_seq1, olig_seq1)
        alignment2 = pairwise2.align.globalxx(pdb_seq1, olig_seq2)
        if alignment1[0][2] > alignment2[0][2]:
            name[0] = 'A_' + oligomer1_name
            name[1] = 'B_' + oligomer2_name
        else:
            name[0] = 'A_' + oligomer2_name
            name[1] = 'B_' + oligomer1_name

        # Check to see if other combinations have already collected sequence info about both design candidates
        for file in os.listdir(design_sequence_fir):
            if fnmatch.fnmatch(file, name[0]):
                seq_files[0] = True
            if fnmatch.fnmatch(file, name[1]):
                seq_files[1] = True

        for i in range(2):
            if not seq_files:
                pdb_seq_file1 = fc.write_fasta_file(pdb_seq1, name1, outpath=seq_dir)
                pdb_seq_file2 = fc.write_fasta_file(pdb_seq2, name2, outpath=seq_dir)
                shutil.copy(template_pdb_name, seq_dir)

                if pdb_seq_file1 and pdb_seq_file2:
                    error_string = '[WARNING] Sequence generation ran into the following residue errors: '
                    if errors1:
                        print(module, error_string, errors1)
                    if errors2:
                        print(module, error_string, errors2)
                    full_pdb_sequence = pdb_seq1 + pdb_seq2  # chain1 (smaller oligomer) first then chain2 (bigger)
                else:
                    print(module, '[ERROR] Unable to parse sequence files. Check input PDB is valid. \nProgram Termination')
                    sys.exit()

                # Make PSSM of PDB sequence POST-SEQUENCE EXTRACTION
                print(module, 'PSSM: Getting PSSM files...')
                pssm_file1, pssm_process1 = make_pssm_hhblits(name1, outpath=seq_dir)
                pssm_file2, pssm_process2 = make_pssm_hhblits(name2, outpath=seq_dir)
                # WAIT for PSSM command to complete, then move on
                pssm_process1.communicate()
                pssm_process2.communicate()

                shutil.copy(pssm_file1, design_sequence_dir)
                shutil.copy(pssm_file2, design_sequence_dir)

    # Configure AA Dictionaries
    pssm_dict1 = populate_design_dict(len(pdb_seq1), fc.alph_3_aa_list)
    pssm_dict2 = populate_design_dict(len(pdb_seq2), fc.alph_3_aa_list)
    full_design_dict = populate_design_dict(len(full_pdb_sequence), [i for i in range(lower_bound, upper_bound + 1)])

    # Parse Fragment Clusters into usable Dictionaries and Flatten for Sequence Design
    cluster_dicts = get_cluster_dicts(info_db=frag_db, id_list=[i for i in residue_cluster_dict])
    residue_cluster_map = convert_to_residue_cluster_map(frag_residue_list)
    full_cluster_dict = deconvolve_clusters(cluster_dicts, full_design_dict, residue_cluster_map)
    final_issm = flatten_for_issm(full_cluster_dict)

    # Extract PSSM for each protein
    pssm_dict1 = parse_hhblits_pssm(pssm_file1, pssm_dict1)
    pssm_dict2 = parse_hhblits_pssm(pssm_file2, pssm_dict2)

    # Combine PSSM1, 2 into single PSSM
    full_pssm = combine_pssm(pssm_dict1, pssm_dict2)  # chain1 (smaller oligomer) first then chain2 (bigger)

    # Make single PSSM with cluster fragments, and evolutionary PSSM
    final_pssm = combine_ssm(full_pssm, final_issm, residue_cluster_map, weights=True)  # final_issm is 0 indexed
    final_pssm_file = make_pssm_file(final_pssm, template_pdb_name)

    # WAIT for Rosetta Refine command to complete, then move on
    refine_process.communicate()
    # DESIGN: Prepare Command, and Flags file then run RosettaScripts on Refined ASU
    design_variables = [('interface_separator', jump), ('pdb_reference', cleaned_pdb), ('dist', dist),
                        ('pssm_file', final_pssm_file), ('interface', ','.join(str(i) for i in int_res_numbers)),
                        ('interface1', ','.join(str(i) for i in int_res_numbers1)),
                        ('interface2', ','.join(str(i) for i in int_res_numbers2)), ('cst_value', cst_value)]
    # 'fix_res', residues_to_fix
    if not debug:
        flagdesign = prepare_rosetta_flags(design_options, design_variables, 'design')
        input_design_cmds = ['-in:file:s', refined_pdb, '-in:file:native', cleaned_pdb, '@' + flagdesign, '-parser:protocol', protocol[1]]
        design_cmd = prepare_rosetta_scripts_cmd(input_design_cmds)
        print(module, 'Design Command:', subprocess.list2cmdline(design_cmd))
        subprocess.call(design_cmd)
    # See if sequence information exists for pose members


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        directory = os.getcwd()

        if sys.argv[1] == 'bio':
            database = bio_fragmentDB
        elif sys.argv[1] == 'xtal':
            database = xtal_fragmentDB
        elif sys.argv[1] == 'combined':
            database = full_fragmentDB
        else:
            database = bio_fragmentDB

        if sys.argv[2] not in ['0', '2', '3']:  # TODO
            print(module, '[ERROR] Please define type of symmetry')
            sys.exit()
        else:
            symmetry = sys.argv[2]
        try:
            if sys.argv[3]:
                debug_ = True
        except IndexError:
            debug_ = False

        main(directory, database, symmetry, debug_)
    else:
        print('\n', module, 'NOTES: Must be run in specific ROT_AA_B/tx_CC directory\n', module,
              'USAGE: fragment_design_MASTER.py fragment_database_type[bio,xtal,combined] symmetry_type[0,2,3] [debug]')
        sys.exit()
