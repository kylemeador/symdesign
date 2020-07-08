import os
import sys
import math
import subprocess
import logging
import pickle
import copy
import numpy as np
import multiprocessing as mp
import sklearn.neighbors
from itertools import repeat
import PDB
from Bio.SeqUtils import IUPACData
from Bio.SubsMat import MatrixInfo
from Bio import pairwise2

import PathUtils as PUtils
import CmdUtils as CUtils
import AnalyzeOutput as AOut
logging.getLogger().setLevel(logging.INFO)

#### MODULE FUNCTIONS #### TODO
# PDB.mutate_to_ala()
# PDB.reorder_chains()
# PDB.renumber_residues()
# PDB.write()

index_offset = 1
alph_3_aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aa_counts_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
                  'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
aa_weight_counts_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0,
                         'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'stats': [0, 1]}


def scout_sdf_chains(pdb):
    # perl $SymDesign/dependencies/rosetta/sdf/scout_symmdef_file.pl
    # -p symdesign/P432/3l8r_1ho1/DEGEN_1_1/ROT_36_1/tx_4/1ho1_tx_4.pdb -i B C D E F G H
    num_chains = len(pdb.chain_id_list)
    chains_of_interest = pdb.chain_id_list[1:]
    scout_cmd = ['perl', PUtils.scout_symmdef, '-p', pdb.filepath, '-i'] + chains_of_interest
    print(subprocess.list2cmdline(scout_cmd))
    p = subprocess.run(scout_cmd, capture_output=True)
    lines = p.stdout.decode('utf-8').strip().split('\n')
    rotation_dict = {}
    max_sym, max_chain = 0, None
    for line in lines:
        chain = line[0]
        symmetry = int(line.split(':')[1][:5].rstrip('-fold'))
        rotation_dict[chain] = symmetry
        if symmetry > max_sym:
            max_sym = symmetry
            max_chain = chain
    if max_chain:
        chain_string = max_chain
    else:
        raise DesignError('%s: No symmetry found for SDF creation' % pdb.filepath)

    # Check for dihedral symmetry
    if num_chains / max_sym == 2:
        for chain in rotation_dict:
            if rotation_dict[chain] == 2:
                chain_string += ' ' + chain
                break

    return chain_string


def make_sdf(pdb):
    chains = scout_sdf_chains(pdb)
    dihedral = False
    if len(chains) > 1:
        dihedral = True
    sdf_file_name = os.path.join(os.path.dirname(pdb.filepath), pdb.name + '.sdf')
    sdf_cmd = ['perl', PUtils.make_symmdef, '-p', pdb.filepath, '-i', chains, '-q']
    print(subprocess.list2cmdline(sdf_cmd))
    with open(sdf_file_name, 'w') as file:
        p = subprocess.Popen(sdf_cmd, stdout=file, stderr=subprocess.DEVNULL)
        p.communicate()

    print(p.returncode)
    if p.returncode != 0:
        raise DesignError('%s: Oligomeric symmetry definition file failed to generate successfully' % pdb.filepath)

    return sdf_file_name


def sdf_lookup(point_type):
    # TODO
    for root, dirs, files in os.walk(PUtils.symmetry_def_files):
        placeholder = None
    symm = os.path.join(PUtils.symmetry_def_files, 'dummy.symm')
    return symm


def unpickle(filename):
    if os.path.getsize(filename) > 0:
        with open(filename, 'rb') as infile:
            new_object = pickle.load(infile)

        return new_object
    else:
        return None


def pickle_object(_object, name, out_path=os.getcwd()):
    with open(os.path.join(out_path, name + '.pkl'), 'wb') as f:
        pickle.dump(_object, f, pickle.HIGHEST_PROTOCOL)

    return os.path.join(out_path + name) + '.pkl'


def get_all_base_root_paths(directory):
    dir_paths = []
    for root, dirs, files in os.walk(directory):
        if not dirs:
            dir_paths.append(root)

    return dir_paths


def get_all_pdb_file_paths(pdb_dir):
    filepaths = []
    for root, dirs, files in os.walk(pdb_dir):
        for file in files:
            if file.endswith('.pdb'):
                filepaths.append(os.path.join(root, file))

    return filepaths


def get_directory_pdb_file_paths(pdb_dir):
    filepaths = []
    for file in os.listdir(pdb_dir):
        if file.endswith('.pdb'):
            filepaths.append(os.path.join(pdb_dir, file))

    return filepaths


def reduce_pose_to_interface(pdb, chains):  # UNUSED
    new_pdb = PDB.PDB()
    new_pdb.read_atom_list(pdb.chains(chains))

    return new_pdb


def combine_pdb(pdb_1, pdb_2, name):  # UNUSED
    """Take two pdb objects and write them to the same file

    Args:
        pdb_1 (PDB): First PDB to concatentate
        pdb_2 (PDB): Second PDB
        name (str): Name of the output file
    """
    pdb_1.write(name)
    with open(name, 'a') as full_pdb:
        for atom in pdb_2.all_atoms:
            full_pdb.write(str(atom))  # .strip() + '\n')


def identify_interface_chains(pdb1, pdb2):  # UNUSED
    distance = 12  # Angstroms
    pdb1_chains = []
    pdb2_chains = []
    # Get Queried CB Tree for all PDB2 Atoms within 12A of PDB1 CB Atoms
    query, pdb1_cb_indices, pdb2_cb_indices = construct_cb_atom_tree(pdb1, pdb2, distance)

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
    # design_cluster_dict FORMAT: = {[residue1_ca_atom, residue2_ca_atom]: {'size': cluster_member_count, 'rmsd':
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
                residue = fragment_data[1] + frag_index - index_offset  # make zero index so design_dict starts at 0
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


def get_all_cluster(pdb, residue_cluster_id_list, db=PUtils.bio_fragmentDB):  # UNUSED DEPRECIATED
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


def get_residue_atom_list(pdb, residue_list, chain=None):  # UNUSED
    if chain is None:
        chain = pdb.chain_id_list[0]
    residues = []
    for residue in residue_list:
        residues.append(pdb.getResidueAtoms(chain, residue))

    return residues


def make_issm(cluster_freq_dict, background):  # UNUSED
    for residue in cluster_freq_dict:
        for aa in cluster_freq_dict[residue]:
            cluster_freq_dict[residue][aa] = round(2 * (math.log2((cluster_freq_dict[residue][aa] / background[aa]))))
    issm = []

    return issm


def print_atoms(atom_list):  # UNUSED DEBUG PERTINENT
    for residue in atom_list:
        for atom in residue:
            print(str(atom))


def populate_design_dict(n, alph, counts=False):
    """Return a dictionary with n elements and alph subelements.

    Args:
        n (int): number of residues in a design
        alph (iter): alphabet of interest
    Keyword Args:
        counts=False (bool): If true include an integer placeholding for counting
     Returns:
         Custom length, 0 indexed dictionary with residue numbers as keys
     """
    if counts:
        return {residue: {i: 0 for i in alph} for residue in range(n)}
    else:
        return {residue: {i: dict() for i in alph} for residue in range(n)}


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


def get_cluster_dicts(info_db=PUtils.biological_fragmentDB, id_list=None):
    """Generate an interface specific scoring matrix from the fragment library

    Args:
    Keyword Args:
        info_db=PUtils.biological_fragmentDB
        id_list=None: [1_2_24, ...]
    Returns:
         cluster_dict: {'1_2_45': {'size': ..., 'rmsd': ..., 'rep': ..., 'mapped': ..., 'paired': ...}, ...}
    """
    if id_list is None:
        directory_list = get_all_base_root_paths(info_db)
    else:
        directory_list = []
        for _id in id_list:
            c_id = _id.split('_')
            _dir = os.path.join(info_db, c_id[0], c_id[0] + '_' + c_id[1], c_id[0] + '_' + c_id[1] + '_' + c_id[2])
            directory_list.append(_dir)

    cluster_dict = {}
    for cluster in directory_list:
        filename = os.path.join(cluster, os.path.basename(cluster) + '.pkl')
        with open(filename, 'rb') as f:
            cluster_dict[os.path.basename(cluster)] = pickle.load(f)

    return cluster_dict


def make_residues_pdb_object(pdb, residue_dict):  # TODO supplement with names info and pull out by names
    """Pair residues with cluster info.

    Args:
        pdb (PDB): PDB object to extract residues from
        residue_dict: residue_dict format: {'1_2_24': [(78, 87),], ...}. Clusters mapped to residue number pairs
    Returns:
        residue_dict - {'1_2_45': [(residue1_ca_atom, residue2_ca_atom), ...] ...}
    """
    for cluster in residue_dict:
        pairs = []
        for pair in range(len(residue_dict[cluster])):
            residue_pair = []
            for i, residue in enumerate(residue_dict[cluster][pair]):
                resi_object = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[i], residue)).ca
                if not resi_object:
                    raise DesignError('Residue \'%s\' is missing from PDB file \'%s\'' % (residue, pdb.filepath))
                residue_pair.append(resi_object)
            pairs.append(tuple(residue_pair))
        residue_dict[cluster] = pairs

    return residue_dict


def convert_to_residue_cluster_map(residue_cluster_dict, frag_range):
    """Make a residue and cluster/fragment index map

    Args:
        residue_cluster_dict (dict): {'1_2_45': [(residue1_ca_atom, residue2_ca_atom), ...] ...}
        frag_range (dict): A range of the fragment size to search over. Ex: (-2, 3) for fragments of length 5
    Returns:
        cluster_map (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
            Where the key is the 0 indexed residue id
    """
    cluster_map = {}
    for cluster in residue_cluster_dict:
        for pair in range(len(residue_cluster_dict[cluster])):
            for i, residue_atom in enumerate(residue_cluster_dict[cluster][pair]):
                # for each residue in map add the same cluster to the range of fragment residue numbers
                residue_num = residue_atom.residue_number - index_offset  # zero index
                for j in range(*frag_range):
                    if residue_num + j not in cluster_map:
                        if i == 0:
                            cluster_map[residue_num + j] = {'chain': 'mapped', 'cluster': []}
                        else:
                            cluster_map[residue_num + j] = {'chain': 'paired', 'cluster': []}
                    cluster_map[residue_num + j]['cluster'].append((j, cluster))

    return cluster_map


def deconvolve_clusters(cluster_dict, design_dict, cluster_map):
    """Add frequency information from a fragment database to a design dictionary

    The frequency information is added in a fragment index dependent manner. If multiple fragment indices are present in
    a single residue, a new observation is created for that fragment index.

    Args:
        cluster_map (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
        cluster_dict (dict): {1_1_54: {'mapped': {aa_freq}, 'paired': {aa_freq}}, ...}
            mapped/paired aa_dict = {-2: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37]}, -1: {}, ...}
                Where 'stats'[0] is total fragments in cluster, and 'stats'[1] is weight of fragment index
        design_dict (dict): {0: {-2: {'A': 0.1, 'C': 0.0, ...}, -1: {}, ... }, 1: {}, ...}
    """

    for resi in cluster_map:
        dict_type = cluster_map[resi]['chain']
        observation = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
        for index_cluster_pair in cluster_map[resi]['cluster']:
            aa_freq = cluster_dict[index_cluster_pair[1]][dict_type][index_cluster_pair[0]]
            # Add the aa_freq from cluster to the residue/frag_index/observation
            try:
                design_dict[resi][index_cluster_pair[0]][observation[index_cluster_pair[0]]] = aa_freq
            except KeyError:
                raise DesignError('Missing residue %d in %s.' % (resi, deconvolve_clusters.__name__))
            observation[index_cluster_pair[0]] += 1
    
    return design_dict


def flatten_for_issm(design_cluster_dict, keep_extras=True):
    """Take a multi-observation, mulit-fragment index, fragment frequency dictionary and flatten to single frequency

    Args:
        design_cluster_dict (dict): {0: {-2: {'A': 0.1, 'C': 0.0, ...}, -1: {}, ... }, 1: {}, ...}
            Dictionary containing fragment frequency and statistics across a design sequence
    Keyword Args:
        keep_extras=True (bool): If true, keep values for all design dictionary positions that are missing fragment data
    Returns:
        design_cluster_dict (dict): Weighted average design dictionary combining all fragment profile information at a
            single residue
    """
    no_design = []
    for res in design_cluster_dict:
        total_residue_weight = 0
        num_frag_weights_observed = 0
        for index in design_cluster_dict[res]:
            if design_cluster_dict[res][index] != dict():
                total_obs_weight = 0
                for obs in design_cluster_dict[res][index]:
                    total_obs_weight += design_cluster_dict[res][index][obs]['stats'][1]
                if total_obs_weight > 0:
                    total_residue_weight += total_obs_weight
                    obs_aa_dict = copy.deepcopy(aa_weight_counts_dict)
                    obs_aa_dict['stats'][1] = total_obs_weight
                    for obs in design_cluster_dict[res][index]:
                        num_frag_weights_observed += 1
                        obs_weight = design_cluster_dict[res][index][obs]['stats'][1]
                        for aa in design_cluster_dict[res][index][obs]:
                            if aa != 'stats':
                                # Add all occurrences to summed frequencies list
                                obs_aa_dict[aa] += design_cluster_dict[res][index][obs][aa] * (obs_weight /
                                                                                               total_obs_weight)
                    design_cluster_dict[res][index] = obs_aa_dict
                else:
                    # Case where no weights associated with observations (side chain not structurally significant)
                    design_cluster_dict[res][index] = dict()

        if total_residue_weight > 0:
            res_aa_dict = copy.deepcopy(aa_weight_counts_dict)
            res_aa_dict['stats'][1] = total_residue_weight
            res_aa_dict['stats'][0] = num_frag_weights_observed
            for index in design_cluster_dict[res]:
                if design_cluster_dict[res][index] != dict():
                    index_weight = design_cluster_dict[res][index]['stats'][1]
                    for aa in design_cluster_dict[res][index]:
                        if aa != 'stats':
                            # Add all occurrences to summed frequencies list
                            res_aa_dict[aa] += design_cluster_dict[res][index][aa] * (index_weight / total_residue_weight)
            design_cluster_dict[res] = res_aa_dict
        else:
            # Add to list for removal from the design dict
            no_design.append(res)

    # Remove missing residues from dictionary
    if not keep_extras:
        for res in no_design:
            design_cluster_dict.pop(res)
    else:
        for res in no_design:
            design_cluster_dict[res] = aa_weight_counts_dict

    return design_cluster_dict


def get_db_statistics(database):
    """Retrieve summary statistics for a specific fragment database

    Args:
        database (str): Disk location of a fragment database
    Returns:
        stats (dict): {cluster_id: [[mapped, paired, {max_weight_counts}, ...], ...}
            ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]
    """
    for file in os.listdir(database):
        if file.endswith('statistics.pkl'):
            with open(os.path.join(database, file), 'rb') as f:
                stats = pickle.load(f)

    return stats


def combine_ssm(pssm, issm, cluster_mapping, db=PUtils.biological_fragmentDB, favor_fragments=True, boltzmann=False):
    """Combine weights for profile PSSM and fragment SSM.

    All input are zero indexed
    Args:
        pssm (dict): HHblits - {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                         'weight': 0.00}, {...}}
              PSIBLAST -  {0: {'A': 0.13, 'R': 0.12, ..., 'lod': {'A': -5, 'R': 2, ...}, 'type': 'W', 'info': 3.20,
                          'weight': 0.73}, {...}} CURRENTLY IMPOSSIBLE, NEED TO CHANGE THE LOD SCORE IN PARSING
        issm (dict): {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'stats': [4, 0.274]}, 50: {...}, ...}
        cluster_mapping (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
    Keyword Args:
        db=PUtils.biological_fragmentDB: Disk location of fragment database
        favor_fragments=True (bool): Whether to favor fragment profile in the lod score of the resulting profile
        boltzmann=True (bool): Whether to weight the fragment profile by the Boltzmann probability. If false, residues
            are weighted by a local maximum over the residue scaled to a maximum provided in the standard Rosetta per
            residue reference weight.
    Returns:
        pssm (dict): Pssm dictionary with modified values based on input
        modified_residues (list(tuple)): String information on the percentage of modification at each residue position
    """
    # Residue reporting moved to main SymDesign protocol to log output
    _alpha, boltzman_energy = 0.5, 1
    favor_seqprofile_score_modifier = 0.2 * PUtils.reference_average_residue_weight
    stat_dict = get_db_statistics(db)

    def find_alpha(cluster_entry):
        _alpha_, average_total, count, i = copy.copy(_alpha), 0.0, 0, 0  # i = 0 corresponds to 'mapped' chain
        if cluster_entry['chain'] == 'paired':
            i = 1
        for residue_cluster_pair in cluster_entry['cluster']:
            cluster_id = return_cluster_id_string(residue_cluster_pair[1], index_number=2)
            average_total += stat_dict[cluster_id][0][i]
            count += 1
        stats_average = average_total / count
        tot_entry_weight = issm[entry]['stats'][1]
        entry_ave_frag_weight = tot_entry_weight / count
        if entry_ave_frag_weight < stats_average:
            _alpha_ *= (entry_ave_frag_weight / stats_average)

        return _alpha_

    # Find alpha parameter. Indicates how much contribution fragments provide to design profile. Fragment cap at 0.5
    alpha = {}
    for entry in pssm:
        if entry in cluster_mapping:
            alpha[entry] = find_alpha(cluster_mapping[entry])
        else:
            alpha[entry] = 0
    # Combine fragment and evolutionary probability profile (second set of values) according to alpha parameter
    modified_residues = []
    for entry in pssm:
        if alpha[entry] > 0:
            for aa in IUPACData.protein_letters:
                pssm[entry][aa] = (alpha[entry] * issm[entry][aa]) + ((1 - alpha[entry]) * pssm[entry][aa])
            modified_residues.append((entry + index_offset, 'Combined fragment and homology profile: %.0f%% fragment'
                                      % (alpha[entry] * 100)))

    if favor_fragments:
        # Modify lod ratios to weight fragments higher in design. Otherwise use evolutionary profile lod scores
        stat_dict_bkg = stat_dict['frequencies']
        null_residue = get_lod(stat_dict_bkg, stat_dict_bkg)
        for aa in null_residue:
            null_residue[aa] = float(null_residue[aa])

        for entry in pssm:
            if issm[entry]['stats'][0] > 0:
                pssm[entry]['lod'] = get_lod(issm[entry], stat_dict_bkg, round_lod=False)
                partition, max_lod = 0, 0.0
                for aa in pssm[entry]['lod']:
                    # for use with a boltzman probability weighting, Z = sum(exp(score / kT))
                    if boltzmann:
                        # for aa in pssm[entry]['lod']:
                        pssm[entry]['lod'][aa] = math.exp(pssm[entry]['lod'][aa] / boltzman_energy)
                        partition += pssm[entry]['lod'][aa]
                    # remove any lod penalty
                    elif pssm[entry]['lod'][aa] < 0:
                        pssm[entry]['lod'][aa] = 0
                    # find the maximum/residue (local) lod score
                    if pssm[entry]['lod'][aa] > max_lod:
                        max_lod = pssm[entry]['lod'][aa]
                modified_entry_alpha = (alpha[entry] / _alpha) * favor_seqprofile_score_modifier
                if boltzmann:
                    modifier = partition
                    modified_entry_alpha /= (max_lod / partition)
                else:
                    modifier = max_lod
                for aa in pssm[entry]['lod']:
                    pssm[entry]['lod'][aa] /= modifier
                    pssm[entry]['lod'][aa] *= modified_entry_alpha

                modified_residues.append((entry + index_offset,
                                          'Fragment lod ratio generated with alpha=%f' % (alpha[entry] / _alpha)))
            else:
                pssm[entry]['lod'] = null_residue

    return pssm, modified_residues


def gather_profile_info(pdb, des_dir, names, log_stream):
    """For a given PDB, find the chain wise profile (pssm) then combine into one continuous pssm

    Args:
        pdb (PDB): PDB to generate a profile from. Sequence is taken from the ATOM record
        des_dir (DesignDirectory): Location of which to write output files in the design tree
        names (dict): The pdb names and corresponding chain of each protomer in the pdb object
        log_stream (logging): Which log to pass logging directives to
    Returns:
        pssm_file (str): Location of the combined pssm file written to disk
        full_pssm (dict): A combined pssm with all chains concatenated in the same order as pdb sequence
    """
    pssm_files, pdb_seq, errors, pdb_seq_file, pssm_process = {}, {}, {}, {}, {}
    log_stream.debug('Fetching PSSM Files')

    # Extract/Format Sequence Information
    for n, name in enumerate(names):
        # if pssm_files[name] == dict():
        log_stream.debug('%s is chain %s in ASU' % (name, names[name](n)))
        pdb_seq[name], errors[name] = extract_aa_seq(pdb, chain=names[name](n))
        log_stream.debug('%s Sequence=%s' % (name, pdb_seq[name]))
        if errors[name]:
            log_stream.warning('Sequence generation ran into the following residue errors: %s'
                               % ', '.join(errors[name]))
        pdb_seq_file[name] = write_fasta_file(pdb_seq[name], name + '_' + os.path.basename(des_dir.path),
                                              outpath=des_dir.sequences)
        if not pdb_seq_file[name]:
            log_stream.error('Unable to parse sequence. Check if PDB \'%s\' is valid.' % name)
            raise DesignError('Unable to parse sequence in %s' % des_dir.path)
        # else:
        #     pdb_seq_file[name] = os.path.join(des_dir.sequences, name + '.fasta')

    # Make PSSM of PDB sequence POST-SEQUENCE EXTRACTION
    for name in names:
        # if pssm_files[name] == dict():
        log_stream.info('Generating PSSM file for %s' % name)
        pssm_files[name], pssm_process[name] = hhblits(pdb_seq_file[name], outpath=des_dir.sequences)
        log_stream.debug('%s seq file: %s' % (name, pdb_seq_file[name]))
        # elif pssm_files[name] == PUtils.temp:
        # log_stream.info('Waiting for profile generation...')

    # Wait for PSSM command to complete
    for name in names:
        pssm_process[name].communicate()

    # Extract PSSM for each protein and combine into single PSSM
    pssm_dict = {}
    for name in names:
        pssm_dict[name] = parse_hhblits_pssm(pssm_files[name])
    full_pssm = combine_pssm([pssm_dict[name] for name in pssm_dict])
    pssm_file = make_pssm_file(full_pssm, PUtils.msa_pssm, outpath=des_dir.path)

    return pssm_file, full_pssm


def psiblast(query, outpath=None, remote=False):  # UNUSED
    """Generate an position specific scoring matrix using PSI-BLAST subprocess

    Args:
        query (str): Basename of the sequence to use as a query, intended for use as pdb
    Keyword Args:
        outpath=None (str): Disk location where generated file should be written
        remote=False (bool): Whether to perform the serach locally (need blast installed locally) or perform search through web
    Returns:
        outfile_name (str): Name of the file generated by psiblast
        p (subprocess): Process object for monitoring progress of psiblast command
    """
    # I would like the background to come from Uniref90 instead of BLOSUM62 #TODO
    if outpath is not None:
        outfile_name = os.path.join(outpath, query + '.pssm')
        direct = outpath
    else:
        outfile_name = query + '.hmm'
        direct = os.getcwd()
    if query + '.pssm' in os.listdir(direct):
        cmd = ['echo', 'PSSM: ' + query + '.pssm already exists']
        p = subprocess.Popen(cmd)

        return outfile_name, p

    cmd = ['psiblast', '-db', PUtils.alignmentdb, '-query', query + '.fasta', '-out_ascii_pssm', outfile_name,
           '-save_pssm_after_last_round', '-evalue', '1e-6', '-num_iterations', '0']
    if remote:
        cmd.append('-remote')
    else:
        cmd.append('-num_threads')
        cmd.append('8')

    p = subprocess.Popen(cmd)

    return outfile_name, p


def hhblits(query, threads=CUtils.hhblits_threads, outpath=os.getcwd()):
    """Generate an position specific scoring matrix from HHblits using Hidden Markov Models

    Args:
        query (str): Basename of the sequence to use as a query, intended for use as pdb
        threads (int): Number of cpu's to use for the process
    Keyword Args:
        outpath=None (str): Disk location where generated file should be written
    Returns:
        outfile_name (str): Name of the file generated by hhblits
        p (subprocess): Process object for monitoring progress of hhblits command
    """

    outfile_name = os.path.join(outpath, os.path.splitext(os.path.basename(query))[0] + '.hmm')

    cmd = [PUtils.hhblits, '-d', PUtils.uniclustdb, '-i', query, '-ohhm', outfile_name, '-v', '1', '-cpu', str(threads)]
    # print('%s Profile Command: %s' % (query, subprocess.list2cmdline(cmd)))
    p = subprocess.Popen(cmd)

    return outfile_name, p


def parse_pssm(file):
    """Take the contents of a pssm file, parse and input into a pose profile dictionary.

    Args:
        file (str): The name/location of the file on disk
    Returns:
        pose_dict (dict): Dictionary containing residue indexed profile information
            Ex: {0: {'A': 0, 'R': 0, ..., 'lod': [-5, -5, ...], 'type': 'W', 'info': 3.20, 'weight': 0.73}, {...}}
    """
    with open(file, 'r') as f:
        lines = f.readlines()

    pose_dict = {}
    for line in lines:
        line_data = line.strip().split()
        if len(line_data) == 44:
            resi = int(line_data[0]) - index_offset
            pose_dict[resi] = copy.deepcopy(aa_counts_dict)
            for i, aa in enumerate(alph_3_aa_list, 22):  # pose_dict[resi], 22):
                # Get normalized counts for pose_dict
                pose_dict[resi][aa] = (int(line_data[i]) / 100.0)
            pose_dict[resi]['lod'] = {}
            for i, aa in enumerate(alph_3_aa_list, 2):
                pose_dict[resi]['lod'][aa] = line_data[i]
            pose_dict[resi]['type'] = line_data[1]
            pose_dict[resi]['info'] = float(line_data[42])
            pose_dict[resi]['weight'] = float(line_data[43])

    return pose_dict


def get_lod(aa_freq_dict, bg_dict, round_lod=True):
    """Get the lod scores for an aa frequency distribution compared to a background frequency
    Args:
        aa_freq_dict (dict): {'A': 0.10, 'C': 0.0, 'D': 0.04, ...}
        bg_dict (dict): {'A': 0.10, 'C': 0.0, 'D': 0.04, ...}
    Keyword Args:
        round_lod=True (bool): Whether or not to round the lod values to an integer
    Returns:
         lods (dict): {'A': 2, 'C': -9, 'D': -1, ...}
    """
    lods = {}
    iteration = 0
    for a in aa_freq_dict:
        if aa_freq_dict[a] == 0:
            lods[a] = -9
        elif a != 'stats':
            lods[a] = float((2.0 * math.log2(aa_freq_dict[a]/bg_dict[a])))  # + 0.0
            if lods[a] < -9:
                lods[a] = -9
            if round_lod:
                lods[a] = round(lods[a])
            iteration += 1
    # if iteration != (len(aa_freq_dict) - 1):
    #     print('frequencies: ', aa_freq_dict)
    #     print('lods: ', lods)

        # print('background: ', bg_dict)
    # print(type(lods[list(lods.keys())[0]]))

    return lods


def parse_hhblits_pssm(file, null_background=True):
    # Take contents of protein.hmm, parse file and input into pose_dict. File is Single AA code alphabetical order
    dummy = 0.00
    null_bg = {'A': 0.0835, 'C': 0.0157, 'D': 0.0542, 'E': 0.0611, 'F': 0.0385, 'G': 0.0669, 'H': 0.0228, 'I': 0.0534,
               'K': 0.0521, 'L': 0.0926, 'M': 0.0219, 'N': 0.0429, 'P': 0.0523, 'Q': 0.0401, 'R': 0.0599, 'S': 0.0791,
               'T': 0.0584, 'V': 0.0632, 'W': 0.0127, 'Y': 0.0287}  # 'uniclust30_2018_08'

    def to_freq(value):
        if value == '*':
            # When frequency is zero
            return 0.0001
        else:
            # Equation: value = -1000 * log_2(frequency)
            freq = 2 ** (-int(value)/1000)
            return freq

    with open(file, 'r') as f:
        lines = f.readlines()

    pose_dict = {}
    read = False
    for line in lines:
        if not read:
            if line[0:1] == '#':
                read = True
        else:
            if line[0:4] == 'NULL':
                if null_background:
                    # use the provided null background from the profile search
                    background = line.strip().split()
                    null_bg = {i: {} for i in alph_3_aa_list}
                    for i, aa in enumerate(alph_3_aa_list, 1):
                        null_bg[aa] = to_freq(background[i])

            if len(line.split()) == 23:
                items = line.strip().split()
                resi = int(items[1]) - index_offset  # make zero index so dict starts at 0
                pose_dict[resi] = {}
                for i, aa in enumerate(IUPACData.protein_letters, 2):
                    pose_dict[resi][aa] = to_freq(items[i])
                pose_dict[resi]['lod'] = get_lod(pose_dict[resi], null_bg)
                pose_dict[resi]['type'] = items[0]
                pose_dict[resi]['info'] = dummy
                pose_dict[resi]['weight'] = dummy

    # Output: {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
    # 'weight': 0.00}, {...}}
    return pose_dict


def make_pssm_file(pssm_dict, name, outpath=os.getcwd()):
    """Create a PSI-BLAST format PSSM file from a PSSM dictionary

    Args:
        pssm_dict (dict): A pssm dictionary which has the fields 'A', 'C', (all aa's), 'lod', 'type', 'info', 'weight'
        name (str): The name of the file
    Keyword Args:
        outpath=cwd (str): A specific location to write the .pssm file to
    Returns:
        out_file (str): Disk location of newly created .pssm file
    """
    lod_freq, counts_freq = False, False
    separation_string1, separation_string2 = 3, 3
    if type(pssm_dict[0]['lod']['A']) == float:
        lod_freq = True
        separation_string1 = 4
    if type(pssm_dict[0]['A']) == float:
        counts_freq = True

    header = '\n\n            ' + (' ' * separation_string1).join(aa for aa in alph_3_aa_list) \
             + ' ' * separation_string1 + (' ' * separation_string2).join(aa for aa in alph_3_aa_list) + '\n'
    footer = ''
    out_file = os.path.join(outpath, name)  # + '.pssm'
    with open(out_file, 'w') as f:
        f.write(header)
        for res in pssm_dict:
            aa_type = pssm_dict[res]['type']
            lod_string = ''
            if lod_freq:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    lod_string += '{:>4.2f} '.format(pssm_dict[res]['lod'][aa])
            else:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    lod_string += '{:>3d} '.format(pssm_dict[res]['lod'][aa])
            counts_string = ''
            if counts_freq:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    counts_string += '{:>3.0f} '.format(math.floor(pssm_dict[res][aa] * 100))
            else:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    counts_string += '{:>3d} '.format(pssm_dict[res][aa])
            info = pssm_dict[res]['info']
            weight = pssm_dict[res]['weight']
            line = '{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}''\n'.format(res + index_offset, aa_type, lod_string,
                                                                             counts_string, round(info, 4),
                                                                             round(weight, 4))
            f.write(line)
        f.write(footer)

    return out_file


def combine_pssm(pssms):
    """To a first pssm, append subsequent pssms incrementing the residue number in each additional pssm

    Args:
        pssms (list(dict)): List of pssm dictionaries to concatentate
    Returns:
        combined_pssm (dict): Concatentated PSSM
    """
    # requires python 3.6+ to maintain sorted dictionaries TODO add for old_key in sort(list(pssms[i].keys()))
    combined_pssm = {}
    new_key = 0
    for i in range(len(pssms)):
        for old_key in pssms[i]:
            combined_pssm[new_key] = pssms[i][old_key]
            new_key += 1

    return combined_pssm


def read_pdb(file):
    """Wrapper on the PDB __init__ and readfile functions
    Args:
        file (str): disk location of pdb file
    Returns:
        pdb (PDB): Initialized PDB object
    """
    pdb = PDB.PDB()
    pdb.readfile(file)

    return pdb


def construct_cb_atom_tree(pdb1, pdb2, distance=8, gly_ca=True):
    """Create a atom tree using CB atoms from two PDB's

    Args:
        pdb1 (PDB): First PDB to query against
        pdb2 (PDB): Second PDB which will be tested against pdb1
    Keyword Args:
        distance=8 (int): The distance to query in Angstroms
        gly_ca=True (bool): Whether glycine CA should be included in the tree
    Returns:
        query (list()): sklearn query object of pdb2 coordinates within dist of pdb1 coordinates
        pdb1_cb_indices (list): List of all CB indices from pdb1
        pdb2_cb_indices (list): List of all CB indices from pdb2
    """
    # Get CB Atom Coordinates including CA coordinates for Gly residues
    pdb1_coords = np.array(pdb1.extract_CB_coords(InclGlyCA=gly_ca))
    pdb2_coords = np.array(pdb2.extract_CB_coords(InclGlyCA=gly_ca))

    # Construct CB Tree for PDB1
    pdb1_tree = sklearn.neighbors.BallTree(pdb1_coords)

    # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
    query = pdb1_tree.query_radius(pdb2_coords, distance)

    # Map Coordinates to Atoms
    pdb1_cb_indices = pdb1.get_cb_indices(InclGlyCA=gly_ca)
    pdb2_cb_indices = pdb2.get_cb_indices(InclGlyCA=gly_ca)

    return query, pdb1_cb_indices, pdb2_cb_indices


def find_interface_residues(pdb1, pdb2, dist=8):
    """Get Queried CB Tree for all PDB2 Atoms within 8A of PDB1 CB Atoms

    Args:
        pdb1 (PDB): First pdb to measure interface between
        pdb2 (PDB): Second pdb to measure interface between
    Keyword Args:
        dist=8 (int): The distance to query in Angstroms
    Returns:
        residues1 (list): A sorted list of unique residue numbers from pdb1
        residues2 (list): A sorted list of unique residue numbers from pdb2
    """
    query, pdb1_cb_indices, pdb2_cb_indices = construct_cb_atom_tree(pdb1, pdb2, distance=dist)

    # Map Coordinates to Residue Numbers
    residues1, residues2 = [], []
    for pdb2_index in range(len(query)):
        if query[pdb2_index].tolist() != list():
            residues2.append(pdb2.all_atoms[pdb2_cb_indices[pdb2_index]].residue_number)
            for pdb1_index in query[pdb2_index]:
                residues1.append(pdb1.all_atoms[pdb1_cb_indices[pdb1_index]].residue_number)
    residues1 = sorted(set(residues1), key=int)
    residues2 = sorted(set(residues2), key=int)
    return residues1, residues2
    # return {pdb1.name: residues1, pdb2.name: residues2}


def parameterize_frag_length(length):
    """Generate fragment length range parameters for use in fragment functions"""
    _range = math.floor(length / 2)
    if length % 2 == 1:
        return 0 - _range, 0 + _range + index_offset
    else:
        print('%d is an even integer which is not symmetric about a single residue. '
              'Ensure this is what you want and modify %s' % (length, parameterize_frag_length.__name__))


def prepare_rosetta_flags(flag_variables, stage, outpath=os.getcwd()):
    """Prepare a protocol specific Rosetta flags file with program specific variables

    Args:
        flag_variables (list(tuple)): The variable value pairs to be filed in the RosettaScripts XML
        stage (str): The protocol stage or flag suffix to name the specific flags file
    Keyword Args:
        outpath=cwd (str): Disk location to write the flags file
    Returns:
        flag_file (str): Disk location of the written flags file
    """
    output_flags = ['-out:path:pdb ' + os.path.join(outpath, PUtils.pdbs_outdir),
                    '-out:path:score ' + os.path.join(outpath, PUtils.scores_outdir)]

    def make_flags_file(flag_list):
        with open(os.path.join(outpath, 'flags_' + stage), 'w') as f:
            for flag in flag_list:
                f.write(flag + '\n')

        return 'flags_' + stage

    _flags = copy.deepcopy(CUtils.flags)
    _flags += output_flags
    _options = CUtils.flag_options[stage]
    for variable in _options:
        _flags.append(variable)

    variables_for_flag_file = '-parser:script_vars'
    for variable, value in flag_variables:
        variables_for_flag_file += ' ' + str(variable) + '=' + str(value)

    _flags.append(variables_for_flag_file)
    flag_file = make_flags_file(_flags)

    return flag_file


def parse_design_flags(directory, flag_variable=None):
    parser_vars = '-parser:script_vars'
    with open(os.path.join(directory, 'flags_design'), 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            if line[:19] == parser_vars:
                variables = line.lstrip(parser_vars).strip().split()
                variable_dict = {}
                for variable in variables:
                    variable_dict[variable.split('=')[0]] = variable.split('=')[1]
                if flag_variable:
                    return variable_dict[flag_variable]
                else:
                    return variable_dict


def write_shell_script(command, stage=1, outpath=os.getcwd(), additional=None):
    with open(os.path.join(outpath, PUtils.stage[stage] + '.sh'), 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('%s\n' % command)
        if additional:
            f.write('%s\n' % additional)


def pdb_list_file(refined_pdb, total_pdbs=1, suffix='', loc=os.getcwd()):
    file_name = os.path.join(loc, 'pdb_file_list.txt')
    with open(file_name, 'w') as f:
        f.write(refined_pdb + '\n')  # run second round of metrics on input as well
        for i in range(index_offset, total_pdbs + index_offset):
            file_line = os.path.splitext(refined_pdb)[0] + suffix + '_' + str(i).zfill(4) + '.pdb\n'
            f.write(file_line)

    return file_name


def write_commands(command_list, name=PUtils.all_command_file, loc=os.getcwd()):
    file = os.path.join(loc, name + '.cmd')
    with open(file, 'w') as f:
        for command in command_list:
            f.write(command + '\n')

    return file


def mp_starmap(function, process_args, threads):
    """Maps an iterable list to a function for multiprocessing using 'spawn' context and multiprocessing.Pool

    Args:
        function (function): Which function should be executed
        process_args (list(tuple)): Arguments to be unpacked in the defined function, order specific
        threads (int): How many workers/threads should be spawned to handle function(arguments)?
    Returns:
        errors (DesignError): Any errors collected from the function
    """
    # TODO find out if I can include more exception clauses to capture without over catching
    results = 0
    errors = []
    with mp.get_context('spawn').Pool(processes=threads) as p:
    # with mp.Pool(processes=threads) as p:
        try:
            results = p.starmap(function, process_args)
        # except Exception as error:
        except (DesignError, AssertionError) as error:
            errors.append(error.message)
    p.join()

    return results, errors


def calculate_mp_threads(io_suspend):
    if io_suspend:
        # current cap for use with hhblits mp. Need to change to take into account specific memory constraints
        return int(mp.cpu_count() / (CUtils.hhblits_threads + 5))
    else:
        return int(mp.cpu_count() / CUtils.min_cores_per_job)


def start_log(name, handler=1, level=2, location=os.getcwd(), propagate=True):
    """Create a logger to handle program messages

    Args:
        name (str): The name of the logger
    Keyword Args:
        handler=1 (int): Whether to handle to stream (1-default) or a file (2)
        level=2 (int): What level of messages to emit (1-debug, 2-info (default), 3-warning, 4-error, 5-critical)
        location=os.getcwd() (str): If a FileHandler is used (handler=2) where should file be written?
            .log is appended to file
        propagate=True (bool): Whether to pass messages to parent level loggers
    Returns:
        logger (Logger): Logger object to handle messages
    """
    log_handler = {1: logging.StreamHandler(), 2: logging.FileHandler(location + '.log')}
    log_level = {1: logging.DEBUG, 2: logging.INFO, 3: logging.WARNING, 4: logging.ERROR, 5: logging.CRITICAL}
    log_format = logging.Formatter('[%(levelname)s]: %(message)s')

    logger = logging.getLogger(name)
    handler = log_handler[handler]
    handler.setLevel(log_level[level])
    handler.setFormatter(log_format)
    logger.addHandler(handler)
    if not propagate:
        logger.propagate = False

    return logger


def report_errors(results):  #UNUSED
    errors = []
    for result in results:
        if result != 0:
            errors.append(result)

    if errors != list():
        err_file = os.path.join(os.getcwd(), module[:-1] + '.errors')
        with open(err_file, 'w') as f:
            f.write(', '.join(errors))
        print('%s Errors written as %s' % (module, err_file))
    else:
        print('%s No errors detected' % module)


class DesignDirectory:

    def __init__(self, directory, auto_structure=True):
        self.symmetry = None         # design_symmetry (P432)
        self.sequences = None        # design_symmetry/sequences (P432/sequences)
        self.building_blocks = None  # design_symmetry/building_blocks (P432/4ftd_5tch)
        self.all_scores = None       # design_symmetry/building_blocks/scores (P432/4ftd_5tch/scores)
        self.path = directory        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/rosetta_pdbs (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2
        self.scores = None           # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/scores (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/scores)
        self.design_pdbs = None      # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/rosetta_pdbs (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/rosetta_pdbs)
        self.log = None
        if auto_structure:
            self.make_directory_structure()

    def __str__(self):
        if self.symmetry:
            return self.path.replace(self.symmetry + '/', '').replace('/', '-')
        else:
            # When is this relevant?
            return self.path.replace('/', '-')[1:]

    def make_directory_structure(self):
        # Prepare Output Directory/Files. path always has format:
        self.symmetry = self.path[:self.path.find(self.path.split('/')[-4]) - 1]
        self.sequences = os.path.join(self.symmetry, PUtils.sequence_info)
        self.building_blocks = self.path[:self.path.find(self.path.split('/')[-3]) - 1]
        self.all_scores = os.path.join(self.building_blocks, PUtils.scores_outdir)
        self.scores = os.path.join(self.path, PUtils.scores_outdir)
        self.design_pdbs = os.path.join(self.path, PUtils.pdbs_outdir)

        if not os.path.exists(self.sequences):
            os.makedirs(self.sequences)
        if not os.path.exists(self.all_scores):
            os.makedirs(self.all_scores)
        if not os.path.exists(self.scores):
            os.makedirs(self.scores)
        if not os.path.exists(self.design_pdbs):
            os.makedirs(self.design_pdbs)

    def start_log(self, name=None, level=2):
        _name = __name__ + '.' + str(self)
        if name:
            _name = name
        self.log = start_log(_name, handler=2, level=level, location=os.path.join(self.path, os.path.basename(self.path)))


def get_design_directories(directory):
    all_design_directories = []
    for design_root, design_dirs, design_files in os.walk(directory):
        if os.path.basename(design_root).startswith('tx_'):
            all_design_directories.append(design_root)
        for directory in design_dirs:
            if directory.startswith('tx_'):
                all_design_directories.append(os.path.join(design_root, directory))
    all_design_directories = set(all_design_directories)
    all_des_dirs = []
    for directory in all_design_directories:
        all_des_dirs.append(DesignDirectory(directory))

    return all_des_dirs


class DesignError(Exception):
    # pass

    def __init__(self, message):  # expression,
        # self.expression = expression
        self.message = message


def set_up_pseudo_design_dir(wildtype, directory, score):
    pseudo_dir = DesignDirectory(wildtype, auto_structure=False)
    pseudo_dir.path = os.path.dirname(wildtype)
    pseudo_dir.building_blocks = os.path.dirname(wildtype)
    pseudo_dir.design_pdbs = directory
    pseudo_dir.scores = os.path.dirname(score)
    pseudo_dir.all_scores = os.getcwd()

    return pseudo_dir


###################
# Sequence handling
###################


def write_fasta_file(sequence, name, outpath=os.getcwd(), multi_sequence=False):
    outfile_name = os.path.join(outpath, name + '.fasta')
    if type(sequence) is list:
        if multi_sequence:
            with open(outfile_name, 'w') as outfile:
                for seq in sequence:
                    header = '>' + seq[0] + '\n'
                    line = seq[2] + '\n'
                    outfile.write(header + line)

        with open(outfile_name, 'w') as outfile:
            if type(sequence[0]) is list:
                header = '>' + name + '\n'
                outfile.write(header)
                for aa in sequence:
                    outfile.write(aa + ' ')
            elif type(sequence[0]) is tuple:
                for seq in sequence:
                    header = seq[0]
                    line = seq[1]
                    outfile.write(header)
                    outfile.write(line)
            else:
                # print('Cannot parse data to make fasta')
                raise DesignError('Cannot parse data to make fasta')
    elif isinstance(sequence, str):
        header = '>' + name + '\n'
        with open(outfile_name, 'w') as outfile:
            outfile.write(header)
            outfile.write(sequence + '\n')
    else:
        # print('Cannot parse data to make fasta')
        raise DesignError('Cannot parse data to make fasta')

    return outfile_name


def extract_aa_seq(pdb, aa_code=1, source='atom', chain=0):
    # Extracts amino acid sequence from either ATOM or SEQRES record of PDB object
    if type(chain) == int:
        chain = pdb.chain_id_list[chain]
    final_sequence = None
    sequence_list = []
    failures = []
    aa_code = int(aa_code)

    if source == 'atom':
        # Extracts sequence from ATOM records
        if aa_code == 1:
            i = 1
            for atom in pdb.all_atoms:
                if atom.chain == chain and atom.type == 'N' and atom.alt_location == '':
                    try:
                        sequence_list.append(IUPACData.protein_letters_3to1[atom.residue_type.lower().capitalize()])
                    except KeyError:
                        sequence_list.append('X')
                        failures.append((atom.residue_number, atom.residue_type))
                        i += 1
            final_sequence = ''.join(sequence_list)
        elif aa_code == 3:
            for atom in pdb.all_atoms:
                if atom.chain == chain and atom.type == 'N' and atom.alt_location == '':
                    sequence_list.append(atom.residue_type)
            final_sequence = sequence_list
        else:
            print('\'%s\' is an incorrect argument for \'aa_code\' in %s' % (aa_code, extract_aa_seq.__name__))

    elif source == 'seqres':
        # Extract sequence from the SEQRES record
        sequence = pdb.sequence_dictionary[chain]
        if not sequence:
            pdb = PDB.PDB()
            sequence = pdb.readfile(pdb.filepath, coordinates_only=False)
        if aa_code == 1:
            final_sequence = sequence
            for i in range(len(sequence)):
                if sequence[i] == 'X':
                    failures.append((i, sequence[i]))
        elif aa_code == 3:
            i = 1
            for residue in sequence:
                sequence_list.append(IUPACData.protein_letters_1to3[residue])
                if residue == 'X':
                    failures.append((i, residue))
                i += 1
            final_sequence = sequence_list
        else:
            print('\'%s\' is an incorrect argument for \'aa_code\' in %s' % (aa_code, extract_aa_seq.__name__))

    else:
        # print('Invalid sequence input')
        raise DesignError('Invalid sequence input')

    return final_sequence, failures


def extract_sequence_from_pdb(pdb_class_dict, aa_code=1, seq_source='atom', mutations=False, output=False, outpath=None
                              ):  # chain_dict
    """Extract the sequence from PDB objects

    Args:
        pdb_class_dict (dict):
    Keyword Args:
        aa_code=1 (int): Whether to return sequence with one-letter or three-letter code [1,3]
        seq_source='atom' (str): Whether to return the ATOM or SEQRES record ['atom','seqres','compare']
        mutations=False (bool): Whether to return mutations from sequences compared to a reference specified by 'ref'
        output=False (bool): Whether or not to save the results to disk
        outpath=None (str): Where to save the results to disk
    Returns:
        mutation_dict (dict): IF mutations=True {pdb: {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...},
            ...}
        or
        sequence_dict (dict): ELSE {pdb: {chain_id: 'Sequence', ...}, ...}
    """
    reference_seq_dict = None
    if mutations:
        # If looking for mutations, the reference PDB object should be given as 'ref' in the dictionary
        if 'ref' not in pdb_class_dict:
            sys.exit('No reference sequence specified, but mutations requested. Include a key \'ref\' in PDB dict!')
        reference_seq_dict = {}
        fail_ref = []
        reference = pdb_class_dict['ref']
        # for pdb in pdb_class_dict:
        #     # if not reference_found:
        #     # TEST to see if the length check is necessary
        # if len(chain_dict['ref']) > 1:
        for chain in pdb_class_dict['ref'].chain_id_list:
            reference_seq_dict[chain], fail = extract_aa_seq(reference, aa_code, seq_source, chain)
            if fail != list():
                fail_ref.append((reference, chain, fail))
        # else:
        #     reference_seq_dict[chain_dict['ref']], fail = extract_aa_seq(reference, aa_code, seq_source,
        #                                                                  chain_dict['ref'])
        #     if fail != list():
        #         fail_ref.append((pdb, chain_dict[pdb], fail))
            # break  # Only grab the first sequence as reference
        if fail_ref:
            print('Errors grabbing reference sequence for mutational analysis:', fail_ref)

    if seq_source == 'compare':
        mutations = True

    file_list = []
    error_list = []
    sequence_dict = {}
    mutation_dict = {}

    def handle_extraction(pdb_code, _pdb, _aa, _source, _chain):
        if _source == 'compare':
            sequence1, failures1 = extract_aa_seq(_pdb, _aa, 'atom', _chain)
            sequence2, failures2 = extract_aa_seq(_pdb, _aa, 'seqres', _chain)
        else:
            sequence1, failures1 = extract_aa_seq(_pdb, _aa, _source, _chain)
            sequence2 = reference_seq_dict[_chain]
            sequence_dict[pdb_code][_chain] = sequence1
        if mutations:
            seq_mutations = generate_mutations_from_seq(sequence1, sequence2, offset=False, remove_blanks=False)
            mutation_dict[pdb_code][_chain] = seq_mutations
        if failures1:
            error_list.append((_pdb, _chain, failures1))

    for pdb in pdb_class_dict:
        sequence_dict[pdb] = {}
        mutation_dict[pdb] = {}
        # if len(chain_dict[pdb]) > 1:
        # for chain in chain_dict[pdb]:
        for chain in pdb_class_dict[pdb].chain_id_list:
            handle_extraction(pdb, pdb_class_dict[pdb], aa_code, seq_source, chain)

    if output:
        _file_list = []
        for pdb in sequence_dict:
            for chain in sequence_dict[pdb]:
                file = write_fasta_file(sequence_dict[pdb][chain], pdb + '_' + chain + '_' + seq_source, outpath)
                _file_list.append(file)
        filepath = concatenate_fasta_files(_file_list, pdb)
        file_list.append(filepath)
    # else:
    #     handle_extraction(pdb, pdb_class_dict[pdb], aa_code, seq_source, chain_dict[pdb])
    #     if output:
    #         filepath = write_fasta_file(sequence1, pdb + '_' + chain_dict[pdb] + '_' + seq_source, outpath)
    #         file_list.append(filepath)

    if error_list:
        print('Warning, the following residues were not extracted:')
        print(error_list)
    if file_list:
        print('The following files were written:')
        print(file_list)
    if mutations:
        return mutation_dict
    else:
        return sequence_dict


def make_sequences_from_mutations(wild_type, mutation_list, aligned=False, output=False):
    """Takes a list of sequence mutations and returns the mutated form on wildtype

    Args:
        wild_type (str): Sequence to mutate
        mutation_list (list): [(name, chain, mutation list), ...]
    Keyword Args:
        aligned=False (bool): Whether the input sequences are already aligned
        output=False (bool): Whether to make a fasta file of the sequence
    Returns:
        all_sequences (dict): {name: sequence, ...}
    """
    all_sequences = {}

    for mutations in mutation_list:
        sequence = make_mutations(wild_type, mutations[2], find_orf=not aligned)
        all_sequences[mutations[0]] = sequence
        chain = mutations[1]

    if output:
        filename = wild_type + '_' + chain
        write_fasta_file(chain_seqs, filename, multi_sequence=True)
        return filename
    else:
        return all_sequences


def make_mutations(seq, mutation_dict, find_orf=True):  # mutation_list
    """Modify a sequence to contain mutations specified by a mutation dictionary

    Args:
        seq (str): 'Wild-type' sequence to mutate
        mutation_dict (dict): Mutations with index as key and value of (From AA, To AA)
    Keyword Args:
        find_orf=True (bool): Whether or not to fing the correct ORF for the mutations and the seq
    Returns:
        seq (str): The mutated sequence
    """
    # Seq can be either list or string
    def find_orf_offset():
        lower_range = 50  # this corresponds to how far away the max seq start is from the ORF MET start site
        met_offset_list = []
        for i, aa in enumerate(seq):
            if aa == 'M':
                met_offset_list.append(i)
        if met_offset_list:
            # Weight potential MET offsets by finding the one which gives the highest number correct mutation sites
            which_met_offset_counts = []
            for index in met_offset_list:
                index -= index_offset
                s = 0
                for mut in mutation_dict:
                    try:
                        if seq[mut + index] == mutation_dict[mut][0]:
                            s += 1
                    except IndexError:
                        break
                which_met_offset_counts.append(s)
            max_count = np.max(which_met_offset_counts)
        else:
            max_count = 0

        # Check if likely ORF has been identified (count > number mutations/2). If not, MET is missing/not the ORF start
        offset_list = []
        if max_count < len(mutation_dict) / 2:
            for i in range(-lower_range, 1):
                s = 0
                for mut in mutation_dict:
                    try:
                        if seq[mut + i] == mutation_dict[mut][0]:
                            s += 1
                    except IndexError:
                        continue
                offset_list.append(s)
            max_count = np.max(offset_list)
            # find likely orf offset index
            orf_offset = offset_list.index(max_count) - lower_range  # + mut_index_correct
        else:
            orf_offset = met_offset_list[which_met_offset_counts.index(max_count)] - index_offset

        return orf_offset

    # mutation_dict = parse_mutations(mutation_list)
    if find_orf:
        print("finding ORF")
        offset = find_orf_offset()
    else:
        offset = -index_offset

    # zero index seq and 1 indexed mutation_dict
    index_errors = []
    for key in mutation_dict:
        if seq[key + offset] == mutation_dict[key][0]:  # adjust seq for zero index slicing
            seq = seq[:key + offset] + mutation_dict[key][1] + seq[key + offset + 1:]
        else:  # find correct offset, or mark mutation source as doomed
            index_errors.append(key)
    if index_errors:
        print('Index errors:', index_errors)

    return seq


def parse_mutations(mutation_list):  # UNUSED
    if isinstance(mutation_list, str):
        mutation_list = mutation_list.split(', ')

    # Takes a list of mutations in the form A37K and parses the index (37), the FROM aa (A), and the TO aa (K)
    # output looks like {37: ('A', 'K'), 440: ('K', 'Y'), ...}
    mutation_dict = {}
    for mutation in mutation_list:
        to_letter = mutation[-1]
        from_letter = mutation[0]
        index = int(mutation[1:-1])
        mutation_dict[index] = (from_letter, to_letter)

    return mutation_dict


def generate_alignment(seq1, seq2):
    """Use Biopython's pairwise2 to generate a local alignment. *Only use for generally similar sequences* """
    matrix = MatrixInfo.blosum62
    gap_penalty = -10
    gap_ext_penalty = -1
    # Create sequence alignment
    alignment = pairwise2.align.localds(seq1, seq2, matrix, gap_penalty, gap_ext_penalty)

    return alignment


def generate_mutations_from_seq(seq1, seq2, offset=True, remove_blanks=True):
    """Create mutations with format A5K, one-indexed

    For PDB file comparison, seq1 should be crystal sequence (ATOM), seq2 should be expression sequence (SEQRES)
    Args:
        seq1 (str): Mutant sequence
        seq2 (str): Wild-type sequence
    Keyword Args:
        offset=True (bool): Whether to calculate alignment offset
        remove_blanks=True (bool): Whether to remove all sequence that has zero index or missing residues
    Returns:
        mutations (dict): {index: {'from': 'A', 'to': 'K'}, ...}
    """
    if offset:
        alignment = generate_alignment(seq1, seq2)
        align_seq_1 = alignment[0][0]
        align_seq_2 = alignment[0][1]
    else:
        align_seq_1 = seq1
        align_seq_2 = seq2

    # Extract differences from the alignment
    starting_index_of_seq2 = align_seq_2.find(seq2[0])
    i = -starting_index_of_seq2 + index_offset  # make 1 index so residue value starts at 1
    mutations = {}
    for seq1_aa, seq2_aa in zip(align_seq_1, align_seq_2):
        if seq1_aa != seq2_aa:
            mutations[i] = {'from': seq2_aa, 'to': seq1_aa}
            # mutation_list.append(str(seq2_aa) + str(i) + str(seq1_aa))
        i += 1

    if remove_blanks:
        # Remove any blank mutations and negative/zero indices
        remove_mutation_list = []
        for entry in mutations:
            if entry > 0:
                # if mutations[entry].find('-') == -1:
                for index in mutations[entry]:
                    if mutations[entry][index] == '-':
                    # if mutations[entry][index] == '0':
                        remove_mutation_list.append(entry)

        for entry in remove_mutation_list:
            mutations.pop(entry)

    return mutations


# Alignments


def find_gapped_columns(alignment_dict):
    target_seq_index = []
    n = 1
    for aa in alignment_dict['meta']['query']:
        if aa != '-':
            target_seq_index.append(n)
        n += 1

    return target_seq_index


def update_alignment_meta(alignment_dict):  # UNUSED
    all_meta = []
    for alignment in alignment_dict:
        all_meta.append(alignment_dict[alignment]['meta'])

    meta_strings = ['' for i in range(len(next(all_meta)))]
    for meta in all_meta:
        j = 0
        for data in meta:
            meta_strings[j] += meta[data]


def modify_alignment_dict_index(alignment_dict, index=0):  # UNUSED
    alignment_dict['counts'] = modify_index(alignment_dict['counts'], index_start=index)
    alignment_dict['rep'] = modify_index(alignment_dict['rep'], index_start=index)

    return alignment_dict


def merge_alignment_dicts(alignment_merge):  # UNUSED
    length = [0]
    i = 0
    for alignment in alignment_merge:
        modify_alignment_dict_index(alignment_merge[alignment], index=length[i])
        length.append(len(alignment_merge[alignment]['meta']['query']))
        i += 1
    merged_alignment_dict = {'meta': update_alignment_meta(alignment_dict)}
    for alignment in alignment_merge:
        merged_alignment_dict.update(alignment_merge[alignment])

    return merged_alignment_dict


def modify_index(alignment, index_start=0):  # UNUSED
    for index in alignment:
        new_index = index + index_start
        alignment[new_index] = alignment.pop(index)

    return alignment


def clean_gapped_columns(alignment_dict, correct_index):
    # Cleans an alignment dictionary by revising key list with correctly indexed positions. 0 indexed
    clean_dict = {}
    # i = 0
    for i, index in enumerate(correct_index):
        clean_dict[i] = alignment_dict[index]
        # i += 1

    return clean_dict


def weight_sequences(msa_dict, alignment):
    # Measure "diversity/suprise" when comparing a single alignment entry to the rest of the alignment
    # The math for this is: SUM(1 / (column_j_aa_representation * aa_ij_count))
    # as was implemented in Capra and Singh, 2007 and was first described by Heinkoff and Heinkoff, 1994
    col_tot_aa_count_dict = {}
    for i in range(len(msa_dict)):
        s = 0  # column amino acid representation
        for aa in msa_dict[i]:
            if aa == '-':
                continue
            elif msa_dict[i][aa] > 0:
                s += 1
            else:
                continue
        col_tot_aa_count_dict[i] = s

    seq_weight_dict = {}
    # k = 1
    for k, record in enumerate(alignment):
        s = 0  # s = "diversity/suprise"
        j = 1  # column number
        for aa in record.seq:
            s += (1 / (col_tot_aa_count_dict[j] * msa_dict[j][aa]))
            j += 1
        seq_weight_dict[k] = s
        # k += 1
    # Format: sequence_in_MSA : sequence_weight_factor. Ex: { 1: 2.390, 2: 2.90, 3:5.33, 4: 1.123, ...}
    return seq_weight_dict


def generate_msa_dictionary(alignment, alphabet=IUPACData.protein_letters, weighted_seq_dict=None, weight=False):
    aligned_seq = str(alignment[0].seq)
    # Add Info to 'meta' record as needed
    alignment_dict = {'meta': {'num_sequences': len(alignment), 'query': aligned_seq.replace('-', ''),
                               'query_with_gaps': aligned_seq}}
    # Populate Counts Dictionary
    alignment_counts_dict = populate_design_dict(alignment.get_alignment_length(), alphabet, counts=True)
    if weight:
        # Procedure if weighted sequences are to be used
        for record in alignment:
            i = 0
            counting_factor = weighted_seq_dict[i]
            for aa in record.seq:
                alignment_counts_dict[i][aa] += counting_factor
                i += 1
    else:
        # Procedure when no weight is used
        for record in alignment:
            i = 0
            for aa in record.seq:
                alignment_counts_dict[i][aa] += 1
                i += 1
    alignment_dict['counts'] = alignment_counts_dict

    return alignment_dict


def add_alignment_representation(counts_dict, gaps=False):
    # Find sequence representation for each column in the alignment and add 'rep'. Gaps are not counted by default
    representation_dict = {i: sum_column_weight(counts_dict['counts'][i], gaps=gaps) for i in range(len(counts_dict['counts']))}
    counts_dict['rep'] = representation_dict

    return counts_dict


def sum_column_weight(column, gaps=False):
    # provide a dictionary with columns to weight and whether or not to count gaps ('-') in the total count
    s = 0
    if gaps:
        for key in column:
            s += column[key]
    else:
        for key in column:
            if key == '-':
                continue
            else:
                s += column[key]

    return s


def msa_to_prob_distribution(alignment_dict):
    # Turn Alignment dictionary into a probability distribution
    for residue in alignment_dict['counts']:
        total_weight_in_column = alignment_dict['rep'][residue]
        # if including -, add gaps=True which treats all column weights the same.
        # I am typically not, as this is inaccurate for scoring. Therefore all weights should be different
        # and reflect probability of residue(i) in the specific column
        if total_weight_in_column == 0:
            print('%s: There is a processing error... Downstream cannot divide by 0. Position = %s' %
                  (msa_to_prob_distribution.__name__, residue))
            sys.exit()  # TODO exception
        else:
            for aa in alignment_dict['counts'][residue]:
                alignment_dict['counts'][residue][aa] /= total_weight_in_column
                # cleaned_msa_dict[i][aa] = round(cleaned_msa_dict[i][aa], 3)

    return alignment_dict


def compute_jsd(msa_dict, bgd_matrix, jsd_lambda=0.5):
    divergence_dict = {}
    for i in range(len(msa_dict)):
        sum_prob1 = 0
        sum_prob2 = 0
        for aa in alph_aa_list_gap:
            if aa == '-':
                continue
            else:
                p = msa_dict[i][aa]
                q = bgd_matrix[aa]
                r = (jsd_lambda * p) + ((1 - jsd_lambda) * q)
                if r == 0:
                    continue
                if q != 0:
                    prob2 = (q * math.log2(q / r))
                    sum_prob2 += prob2
                if p != 0:
                    prob1 = (p * math.log2(p / r))
                    sum_prob1 += prob1
        divergence = jsd_lambda * sum_prob1 + (1 - jsd_lambda) * sum_prob2
        divergence_dict[i] = round(divergence, 3)

    return divergence_dict


def weight_gaps(divergence, representation, alignment_length):
    for i in range(len(divergence)):
        divergence[i] = divergence[i] * representation[i] / alignment_length

    return divergence


def window_score(score_dict, window_len, lam):
    # Modified from Capra and Singh 2007 code
    # " This function takes a list of scores and a length and transforms them
    # so that each position is a weighted average of the surrounding positions.
    # Positions with scores less than zero are not changed and are ignored in the
    # calculation. Here window_len is interpreted to mean window_len residues on
    # either side of the current residue. "
    if window_len == 0:

        return score_dict
    else:
        window_scores = {}
        length = len(score_dict)
        for i in range(length + 1):
            s = 0
            number_terms = 0
            if i <= window_len:
                for j in range(1, i + window_len + 1):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            elif i + window_len > length:
                for j in range(i - window_len, length + 1):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            else:
                for j in range(i - window_len, i + window_len + 1):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            window_scores[i] = (1 - lam) * (s / number_terms) + lam * score_dict[i]

        return window_scores


def rank_possibilities(probability_dict):
    # gather alternative residues and sort them by probability. Return the sorted aa probabilities residue by residue in
    # dictionary where the key is the residue number and the value is the sorted tuple of residue probabilities
    sorted_alternate_dict = {}
    for residue in probability_dict:
        residue_probability_list = []
        for aa in probability_dict[residue]:
            if probability_dict[residue][aa] > 0:
                temp_tuple = (aa, round(probability_dict[residue][aa], 5))
                residue_probability_list.append(temp_tuple)  # using tuples here instead of list
        residue_probability_list.sort(key=lambda tup: tup[1], reverse=True)
        sorted_alternate_dict[residue] = [aa[0] for aa in residue_probability_list]

    return sorted_alternate_dict


def process_alignment(bio_alignment_object):
    alignment_dict = generate_msa_dictionary(bio_alignment_object)
    alignment_dict = add_alignment_representation(alignment_dict)
    alignment_dict = msa_to_prob_distribution(alignment_dict)

    return alignment_dict
