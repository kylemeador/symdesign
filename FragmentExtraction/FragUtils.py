import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import math
import numpy as np
import multiprocessing as mp
from itertools import chain
from sklearn.neighbors import BallTree
from Bio.PDB import PDBParser, Atom, Residue, Chain, Superimposer
from PDB import PDB
from SymDesignUtils import DesignError

# Globals
module = 'Fragment Utilities:'


def construct_cb_atom_tree(pdb1, pdb2, distance):
    # Get CB Atom Coordinates
    pdb1_coords = np.array(pdb1.extract_cb_coords(InclGlyCA=True))
    pdb2_coords = np.array(pdb2.extract_cb_coords(InclGlyCA=True))

    # Construct CB Tree for PDB1
    pdb1_tree = BallTree(pdb1_coords)

    # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
    return pdb1_tree.query_radius(pdb2_coords, distance)


def find_interface_pairs(pdb1, pdb2, distance):
    # Get Queried CB Tree for all PDB2 Atoms within 8A of PDB1 CB Atoms
    query = construct_cb_atom_tree(pdb1, pdb2, distance)
    pdb1_cb_indices = pdb1.get_cb_indices()
    pdb2_cb_indices = pdb2.get_cb_indices()

    # Map Coordinates to Residue Numbers
    interface_pairs = []
    for pdb2_index in range(len(query)):
        if query[pdb2_index].tolist() != list():
            pdb2_res_num = pdb2.atoms[pdb2_cb_indices[pdb2_index]].residue_number
            for pdb1_index in query[pdb2_index]:
                pdb1_res_num = pdb1.atoms[pdb1_cb_indices[pdb1_index]].residue_number
                interface_pairs.append((pdb1_res_num, pdb2_res_num))

    return interface_pairs


def get_guide_atoms(frag_pdb):
    guide_atoms = []
    for atom in frag_pdb.atoms:
        if atom.chain == "9":
            guide_atoms.append(atom)
    if len(guide_atoms) == 3:
        return guide_atoms
    else:
        return None


def get_guide_atoms_biopdb(biopdb_structure):
    guide_atoms = []
    for atom in biopdb_structure.get_atoms():
        if atom.get_full_id()[2] == '9':
            guide_atoms.append(atom)
    return guide_atoms


def guide_atom_rmsd(guide_atom_list_1, guide_atom_list_2):
    # Calculate RMSD
    sq_e1 = guide_atom_list_1[0].distance_squared(guide_atom_list_2[0], intra=True)
    sq_e2 = guide_atom_list_1[1].distance_squared(guide_atom_list_2[1], intra=True)
    sq_e3 = guide_atom_list_1[2].distance_squared(guide_atom_list_2[2], intra=True)
    s = sq_e1 + sq_e2 + sq_e3
    mean = s / float(3)
    rmsd = math.sqrt(mean)

    return rmsd


def guide_atom_rmsd_biopdb(biopdb_1_guide_atoms, biopdb_2_guide_atoms):
    # Calculate RMSD
    e1 = (biopdb_1_guide_atoms[0] - biopdb_2_guide_atoms[0]) ** 2
    e2 = (biopdb_1_guide_atoms[1] - biopdb_2_guide_atoms[1]) ** 2
    e3 = (biopdb_1_guide_atoms[2] - biopdb_2_guide_atoms[2]) ** 2
    s = e1 + e2 + e3
    m = s / float(3)
    r = math.sqrt(m)

    return r


def mp_guide_atom_rmsd(biopdb_guide_atoms_tup, rmsd_thresh):
    biopdb_1_guide_atoms = biopdb_guide_atoms_tup[0]
    biopdb_2_guide_atoms = biopdb_guide_atoms_tup[1]
    biopdb_1_id = biopdb_1_guide_atoms[0].get_full_id()[0]
    biopdb_2_id = biopdb_2_guide_atoms[0].get_full_id()[0]

    # Calculate RMSD
    e1 = (biopdb_1_guide_atoms[0] - biopdb_2_guide_atoms[0]) ** 2
    e2 = (biopdb_1_guide_atoms[1] - biopdb_2_guide_atoms[1]) ** 2
    e3 = (biopdb_1_guide_atoms[2] - biopdb_2_guide_atoms[2]) ** 2
    s = e1 + e2 + e3
    m = s / float(3)
    r = math.sqrt(m)

    if r <= rmsd_thresh:
        return biopdb_1_id, biopdb_2_id, r
    else:
        return None


def superimpose(atoms, rmsd_thresh):
    biopdb_1_id = atoms[0][0].get_full_id()[0]
    biopdb_2_id = atoms[1][0].get_full_id()[0]

    sup = Superimposer()
    sup.set_atoms(atoms[0], atoms[1])
    if sup.rms <= rmsd_thresh:
        return biopdb_1_id, biopdb_2_id, sup.rms
    else:
        return None


def get_all_base_root_paths(directory):
    dir_paths = []
    for root, dirs, files in os.walk(directory):
        if not dirs:
            dir_paths.append(root)

    return dir_paths


def get_all_pdb_file_paths(pdb_dir):
    filepaths = []
    for root, dirs, files in os.walk(pdb_dir, followlinks=True):
        for file in files:
            if file.endswith('.pdb'):
                filepaths.append(os.path.join(root, file))

    return filepaths


def get_rmsd_atoms(filepaths, function):
    all_rmsd_atoms = []
    for filepath in filepaths:
        pdb_name = os.path.splitext(os.path.basename(filepath))[0]
        parser = PDBParser()
        pdb = parser.get_structure(pdb_name, filepath)
        all_rmsd_atoms.append(function(pdb))

    return all_rmsd_atoms


# def mp_function(function, process_args, threads, thresh=None):  #, dir1=None, dir2=None):
#     with mp.Pool(processes=threads) as p:
#         if thresh:
#             results = p.map(partial(function, rmsd_thresh=thresh), process_args)
#         # elif dir1 and dir2:
#         #     results = p.map(partial(function, dir1=dir1, dir2=dir2), process_args)
#         else:
#             print('mp_function is missing required arguments')
#             sys.exit()
#     p.join()
#
#     return results


def mp_starmap(function, process_args, threads):
    with mp.Pool(processes=threads) as p:
        results = p.starmap(function, process_args)
    p.join()

    return results


def get_biopdb_ca(structure):
    return [atom for atom in structure.get_atoms() if atom.get_id() == 'CA']


def center(bio_pdb):
    ca_atoms = get_biopdb_ca(bio_pdb)

    # Get Central Residue (5 Residue Fragment => 3rd Residue) CA Coordinates
    center_ca_atom = ca_atoms[2]
    center_ca_coords = center_ca_atom.get_coord()

    # Center Such That Central Residue CA is at Origin
    for atom in bio_pdb.atoms():  # Todo might be wrong
        atom.set_coord(np.add(atom.get_coord(), -center_ca_coords))


def cluster_fragment_rmsds(rmsd_file_path):
    # Get All to All RMSD File
    with open(rmsd_file_path, 'r') as rmsd_file:
        rmsd_file_lines = rmsd_file.readlines()

    # Create Dictionary Containing Structure Name as Key and a List of Neighbors within RMSD Threshold as Values
    rmsd_dict = {}
    for line in rmsd_file_lines:
        line = line.rstrip()
        line = line.split()

        if line[0] in rmsd_dict:
            rmsd_dict[line[0]].append(line[1])
        else:
            rmsd_dict[line[0]] = [line[1]]

        if line[1] in rmsd_dict:
            rmsd_dict[line[1]].append(line[0])
        else:
            rmsd_dict[line[1]] = [line[0]]

    # Cluster
    return_clusters = []
    flattened_query = list(chain.from_iterable(rmsd_dict.values()))

    while flattened_query != list():
        # Find Structure With Most Neighbors within RMSD Threshold
        max_neighbor_structure = None
        max_neighbor_count = 0
        for query_structure in rmsd_dict:
            neighbor_count = len(rmsd_dict[query_structure])
            if neighbor_count > max_neighbor_count:
                max_neighbor_structure = query_structure
                max_neighbor_count = neighbor_count

        # Create Cluster Containing Max Neighbor Structure (Cluster Representative) and its Neighbors
        cluster = rmsd_dict[max_neighbor_structure]
        return_clusters.append((max_neighbor_structure, cluster))

        # Remove Claimed Structures from rmsd_dict
        claimed_structures = [max_neighbor_structure] + cluster
        updated_dict = {}
        for query_structure in rmsd_dict:
            if query_structure not in claimed_structures:
                tmp_list = []
                for idx in rmsd_dict[query_structure]:
                    if idx not in claimed_structures:
                        tmp_list.append(idx)
                updated_dict[query_structure] = tmp_list
            else:
                updated_dict[query_structure] = []

        rmsd_dict = updated_dict
        flattened_query = list(chain.from_iterable(rmsd_dict.values()))

    return return_clusters


def add_guide_atoms(biopdb_structure):
    # Create Guide Atoms
    _a1 = Atom
    a1 = _a1.Atom("CA", (0.0, 0.0, 0.0), 20.00, 1.0, " ", " CA ", 1, element="C")
    _a2 = Atom
    a2 = _a2.Atom("N", (3.0, 0.0, 0.0), 20.00, 1.0, " ", " N  ", 2, element="N")
    _a3 = Atom
    a3 = _a3.Atom("O", (0.0, 3.0, 0.0), 20.00, 1.0, " ", " O  ", 3, element="O")

    # Create Residue for Guide Atoms
    _r = Residue
    r = _r.Residue((' ', 0, ' '), "GLY", "    ")
    # Create Chain for Guide Atoms
    _c = Chain
    c = _c.Chain("9")
    # Add Guide Atoms to Residue
    r.add(a1)
    r.add(a2)
    r.add(a3)
    # Add Residue to Chain
    c.add(r)
    # Add Chain to BioPDB Structure
    biopdb_structure[0].add(c)


def collect_frag_weights(pdb, mapped_chain, paired_chain, interaction_dist):
    num_bb_atoms = 4

    # Creating PDB instance for mapped and paired chains
    pdb_mapped = PDB.from_atoms(atoms=pdb.chain(mapped_chain).atoms)
    pdb_paired = PDB.from_atoms(atoms=pdb.chain(paired_chain).atoms)
    # pdb_mapped.read_atom_list(pdb.get_chain_atoms(mapped_chain))
    # pdb_paired.read_atom_list(pdb.get_chain_atoms(paired_chain))

    # Query Atom Tree for all Ch2 Atoms within interaction_distance of Ch1 Atoms
    query = construct_cb_atom_tree(pdb_mapped, pdb_paired, interaction_dist)

    # Map Coordinates to Atoms
    # pdb_map_cb_indices = pdb1.get_cb_indices()  # InclGlyCA=True)
    # pdb_partner_cb_indices = pdb2.get_cb_indices()  # InclGlyCA=True)

    # Map Coordinates to Atoms
    interacting_pairs = []
    for patner_index in range(len(query)):
        if query[patner_index].tolist() != list():
            if not pdb_paired.atoms[patner_index].is_backbone():
                paired_atom_num = pdb_paired.atoms[patner_index].residue_number
            else:
                # marks the atom number as backbone
                paired_atom_num = False
            for mapped_index in query[patner_index]:
                if not pdb_mapped.atoms[mapped_index].is_backbone():
                    mapped_atom_num = pdb_mapped.atoms[mapped_index].residue_number
                else:
                    # marks the atom number as backbone
                    mapped_atom_num = False
                interacting_pairs.append((mapped_atom_num, paired_atom_num))

    # Create dictionary and Count all atoms in each residue sidechain
    # ex. {'A': {32: (0, 9), 33: (0, 5), ...}, 'B':...}
    res_counts_dict = {'mapped': {residue.number: [0, residue.number_of_atoms - num_bb_atoms]
                                  for residue in pdb_mapped.residues},
                       'paired': {residue.number: [0, residue.number_of_atoms - num_bb_atoms]
                                  for residue in pdb_paired.residues}}
    # res_counts_dict = {'mapped': {i.residue_number: [0, len(pdb_mapped.get_residue_atoms(mapped_chain, i.residue_number))
    #                                                  - num_bb_atoms] for i in pdb_mapped.get_ca_atoms()},
    #                    'paired': {i.residue_number: [0, len(pdb_paired.get_residue_atoms(paired_chain, i.residue_number))
    #                                                  - num_bb_atoms] for i in pdb_paired.get_ca_atoms()}}
    # Count all residue/residue interactions that do not originate from a backbone atom. In this way, side-chain to
    # backbone are counted for the sidechain residue, indicating significance. However, backbones are (mostly)
    # identical, and therefore, their interaction should be conserved in each member of the cluster and not counted
    for res_pair in interacting_pairs:
        if res_pair[0]:
            res_counts_dict['mapped'][res_pair[0]][0] += 1
        if res_pair[1]:
            res_counts_dict['paired'][res_pair[1]][0] += 1

    # Add the value of the total residue involvement for single structure to overall cluster dictionary
    for chain in res_counts_dict:
        total_pose_score = 0
        for residue in res_counts_dict[chain]:
            if res_counts_dict[chain][residue][1] == 0:
                res_counts_dict[chain][residue] = 0.0
            else:
                res_normalized_score = res_counts_dict[chain][residue][0] / float(res_counts_dict[chain][residue][1])
                res_counts_dict[chain][residue] = res_normalized_score
                total_pose_score += res_normalized_score
        if total_pose_score == 0:
            # case where no atoms are within interaction distance
            for residue in res_counts_dict[chain]:
                res_counts_dict[chain][residue] = 0.0
            continue
        for residue in res_counts_dict[chain]:
            # Get percent of residue contribution to interaction over the entire pose interaction
            res_counts_dict[chain][residue] /= total_pose_score
            res_counts_dict[chain][residue] = round(res_counts_dict[chain][residue], 3)

    return res_counts_dict


def populate_aa_dictionary(low, up):
    aa_dict = {i: {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
                   'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'stats': [0, 1]}
               for i in range(low, up + 1)}
    # 'stats' are total (stats[0]) and weight (stats[1])
    # Weight starts as 1 to prevent removal during dictionary culling procedure
    return aa_dict


def freq_distribution(counts_dict, size):
    # turn the dictionary into a frequency distribution dictionary
    for residue in counts_dict:
        remove = []
        for aa in counts_dict[residue]:
            if aa != 'stats':
                # remove residues with no representation
                if counts_dict[residue][aa] == 0:
                    remove.append(aa)
                else:
                    counts_dict[residue][aa] = round(counts_dict[residue][aa] / size, 3)
        for null_aa in remove:
            counts_dict[residue].pop(null_aa)

    return counts_dict


def parameterize_frag_length(length):
    """Generate fragment length range parameters for use in fragment functions"""
    _range = math.floor(length / 2)
    if length % 2 == 1:
        return 0 - _range, 0 + _range + 1
    else:
        logger.critical('%d is an even integer which is not symmetric about a single residue. '
                        'Ensure this is what you want and modify %s' % (length, parameterize_frag_length.__name__))
        raise DesignError('Function not supported: Even fragment length \'%d\'' % length)


def report_errors(results):
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
