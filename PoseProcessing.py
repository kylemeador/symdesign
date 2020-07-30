"""
The main program for generating sequence profiles from evolutionary and fragment databases to constrain protein design

Before full run need to alleviate the following tags and incompleteness
-TODO - Make this resource
-OPTIMIZE - make the best protocol
-JOSH - input style from Josh's output

Object formatting
-PSSM format is dictating the output file format as it works best with Rosetta pose number
"""
import os
import sys
import subprocess
import time
import argparse
import shutil
import copy
from glob import glob, iglob
from itertools import repeat
import math
import numpy as np
# import pandas as pd
from Bio.SeqUtils import IUPACData
import SymDesignUtils as SDUtils
import PathUtils as PUtils
import CmdUtils as CUtils
from AnalyzeOutput import analyze_output


def cluster_distances():
    from itertools import chain
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


def cluster_poses(pose_map):
    for building_block in pose_map:
        building_block_rmsd_map = pd.DataFrame(pose_map[building_block])

        # PCA analysis of distances
        # pairwise_sequence_diff_mat = np.zeros((len(designs), len(designs)))
        # for k, dist in enumerate(pairwise_sequence_diff_np):
        #     i, j = SDUtils.condensed_to_square(k, len(designs))
        #     pairwise_sequence_diff_mat[i, j] = dist

        building_block_rmsd_matrix = SDUtils.sym(building_block_rmsd_map.values)

        building_block_rmsd_matrix = StandardScaler().fit_transform(building_block_rmsd_matrix)
        pca = PCA(PUtils.variance)
        building_block_rmsd_pc_np = pca.fit_transform(building_block_rmsd_matrix)
        pca_distance_vector = pdist(building_block_rmsd_pc_np)
        # epsilon = math.sqrt(seq_pca_distance_vector.mean()) * 0.5
        epsilon = pca_distance_vector.mean() * 0.5
        logger.info('Finding maximum neighbors within distance of %f' % epsilon)

    # Compute the highest density cluster using DBSCAN algorithm
    # seq_cluster = DBSCAN(eps=epsilon)
    # seq_cluster.fit(pairwise_sequence_diff_np)
    #
    # seq_pc_df = pd.DataFrame(seq_pc, index=designs,
    #                          columns=['pc' + str(x + SDUtils.index_offset) for x in range(len(seq_pca.components_))])
    # seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)


def pose_rmsd(all_des_dirs):
    from Bio.PDB import PDBParser
    # from Bio.PDB.Selection import unfold_entities
    from itertools import combinations
    threshold = 1.0  # TODO test

    pose_map = {}
    for pair in combinations(all_des_dirs, 2):
        if pair[0].building_blocks == pair[1].building_blocks:
            # returns a list with all ca atoms from a structure
            # pair_atoms = SDUtils.get_rmsd_atoms([pair[0].asu, pair[1].asu], SDUtils.get_biopdb_ca)
            pdb_parser = PDBParser()
            # pdb = parser.get_structure(pdb_name, filepath)
            pair_structures = [pdb_parser.get_structure(str(pose), pose.asu) for pose in pair]
            # pair_atoms = SDUtils.get_rmsd_atoms([pair[0].path, pair[1].path], SDUtils.get_biopdb_ca)
            logger.info(pair[0].info)
            # grabs stats['des_resides'] from the design_directory
            des_residue_list = [pose.info['des_residues'] for pose in pair]
            # des_residues = SDUtils.unpickle(os.path.join(des_dir.data, PUtils.des_residues))
            # # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/data
            # pair should be a structure...
            rmsd_residue_list = [residue for n, structure in enumerate(pair_structures)
                                 for residue in structure.get_residues() if residue in des_residue_list[n]]
            pair_atom_list = SDUtils.get_rmsd_atoms(rmsd_residue_list, SDUtils.get_biopdb_ca)
            # pair_rmsd = SDUtils.superimpose(pair_atoms, threshold)
            pair_rmsd = SDUtils.superimpose(pair_atom_list, threshold)
            if not pair_rmsd:
                continue
            if pair[0].building_blocks in pose_map:
                # {building_blocks: {(pair1, pair2): rmsd, ...}, ...}
                if str(pair[0]) in pose_map[pair[0].building_blocks]:
                    pose_map[pair[0].building_blocks][str(pair[0])][str(pair[1])] = pair_rmsd[2]  # 2 is the rmsd value
                else:
                    pose_map[pair[0].building_blocks][str(pair[0])] = {str(pair[1]): pair_rmsd[2]}
                # pose_map[pair[0].building_blocks][(str(pair[0]), str(pair[1]))] = pair_rmsd[2]
            else:
                pose_map[pair[0].building_blocks] = {str(pair[0]): {str(pair[1]): pair_rmsd[2]}}
                # pose_map[pair[0].building_blocks] = {(str(pair[0]), str(pair[1])): pair_rmsd[2]}

    return pose_map


@SDUtils.handle_errors(errors=(SDUtils.DesignError, AssertionError))
def initialization_s(des_dir, frag_db, sym, script=False, mpi=False, suspend=False, debug=False):
    return initialization(des_dir, frag_db, sym, script=script, mpi=mpi, suspend=suspend, debug=debug)


def initialization_mp(des_dir, frag_db, sym, script=False, mpi=False, suspend=False, debug=False):
    try:
        pose = initialization(des_dir, frag_db, sym, script=script, mpi=mpi, suspend=suspend, debug=debug)
        # return initialization(des_dir, frag_db, sym, script=script, mpi=mpi, suspend=suspend, debug=debug), None
        return pose, None
    except (SDUtils.DesignError, AssertionError) as e:
        return None, (des_dir.path, e)
    # finally:
    #     print('Error occurred in %s' % des_dir.path)


# @SDUtils.handle_errors((SDUtils.DesignError, AssertionError))
def initialization(des_dir, frag_db, sym, script=False, mpi=False, suspend=False, debug=False):
    # Variable initialization
    cst_value = round(0.2 * CUtils.reference_average_residue_weight, 2)
    frag_size = 5
    done_process = subprocess.Popen(['run="go"'], shell=True)

    # Log output
    if debug:
        # global logger
        logger = SDUtils.start_log(name=__name__, handler=2, level=1,
                                   location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))
    else:
        logger = SDUtils.start_log(name=__name__, handler=2, level=2,
                                   location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))
    logger.info('Processing directory \'%s\'' % des_dir.path)

    # Set up Rosetta command, files
    main_cmd = copy.deepcopy(CUtils.script_cmd)
    # cleaned_pdb = os.path.join(des_dir.path, PUtils.clean)
    # ala_mut_pdb = os.path.splitext(cleaned_pdb)[0] + '_for_refine.pdb'
    ala_mut_pdb = os.path.splitext(des_dir.asu)[0] + '_for_refine.pdb'
    consensus_pdb = os.path.splitext(des_dir.asu)[0] + '_for_consensus.pdb'
    consensus_design_pdb = os.path.join(des_dir.design_pdbs, os.path.splitext(des_dir.asu)[0] + '_for_consensus.pdb')
    refined_pdb = os.path.join(des_dir.design_pdbs, os.path.splitext(os.path.basename(ala_mut_pdb))[0] + '.pdb')
    # '_%s.pdb' % PUtils.stage[1]) TODO clean this stupid mechanism only for P432
    # if out:file:o works, could use, os.path.join(des_dir.design_pdbs, PUtils.stage[1] + '.pdb') but it won't register

    # Extract information from SymDock Output
    pdb_codes = str(os.path.basename(des_dir.building_blocks)).split('_')
    # cluster_residue_d, transformation_dict = SDUtils.gather_fragment_metrics(des_dir)

    # Fetch PDB object of each chain from PDBdb or PDB server # TODO set up pdb database
    # UNCOMMMENT WHEN DATABASE IS SET UP
    # oligomers = SDUtils.fetch_pdbs(des_dir, pdb_codes)
    #
    # for i, name in enumerate(oligomers):
    #     # oligomers[name].translate(oligomers[name].center_of_mass())
    #     # TODO get orient program into source, get symm files from Josh
    #     oligomers[name].orient(symm, PUtils.orient)
    #     oligomers[name].rotate_translate(transformation_dict[i]['rot/deg'], transformation_dict[i]['tx_int'])
    #     oligomers[name].rotate_translate(transformation_dict[i]['setting'], transformation_dict[i]['tx_ref'])
    #     # {1: {'rot/deg': [[], ...],'tx_int': [], 'setting': [[], ...], 'tx_ref': []}, ...}

    template_pdb = SDUtils.read_pdb(os.path.join(des_dir.path, PUtils.asu))
    num_chains = len(template_pdb.chain_id_list)

    # TODO JOSH Get rid of same chain ID problem....
    # if num_chains != 2:
    if num_chains != len(pdb_codes):
        oligomer_file = glob(os.path.join(des_dir.path, pdb_codes[0] + '_tx_*.pdb'))
        assert len(oligomer_file) == 1, 'More than one matching file found with %s' % pdb_codes[0] + '_tx_*.pdb'
        # assert len(oligomer_file) == 1, '%s: More than one matching file found with %s' % \
        #                                 (des_dir.path, pdb_codes[0] + '_tx_*.pdb')
        first_oligomer = SDUtils.read_pdb(oligomer_file[0])
        # find the number of ATOM records for template_pdb chain1 using the same oligomeric chain as model
        for atom_idx in range(len(first_oligomer.chain(template_pdb.chain_id_list[0]))):
            template_pdb.all_atoms[atom_idx].chain = 'x'
        template_pdb.chain_id_list = ['x', template_pdb.chain_id_list[0]]
        num_chains = len(template_pdb.chain_id_list)
        logger.warning('%s: Incorrect chain count: %d. Chains probably have the same id! Temporarily changing IDs\'s to'
                       ' %s' % (des_dir.path, num_chains, template_pdb.chain_id_list))

    assert len(pdb_codes) == num_chains, 'Number of chains \'%d\' in ASU doesn\'t match number of building blocks ' \
                                         '\'%d\'' % (num_chains, len(pdb_codes))
    # assert len(pdb_codes) == num_chains, '%s: Number of chains \'%d\' in ASU doesn\'t match number of building blocks '\
    #                                      '\'%d\'' % (des_dir.path, num_chains, len(pdb_codes))

    # Set up names object containing pdb id and chain info
    names = {}
    for c, chain in enumerate(template_pdb.chain_id_list):
        names[pdb_codes[c]] = template_pdb.get_chain_index
    logger.debug('Chain, Name Pairs: %s' % ', '.join(oligomer + ', ' + str(value(c)) for c, (oligomer, value) in
                                                     enumerate(names.items())))

    # Fetch PDB object of each chain individually from the design directory
    oligomer = {}
    sym_definition_files = {}
    for name in names:
        name_pdb_file = glob(os.path.join(des_dir.path, name + '_tx_*.pdb'))
        assert len(name_pdb_file) == 1, 'More than one matching file found with %s_tx_*.pdb' % name
        # assert len(name_pdb_file) == 1, '%s: More than one matching file found with %s' % \
        #                                 (des_dir.path, name + '_tx_*.pdb')
        oligomer[name] = SDUtils.read_pdb(name_pdb_file[0])
        oligomer[name].AddName(name)
        oligomer[name].reorder_chains()
        # TODO Chains must be symmetrized on input before SDF creation, currently raise DesignError
        sym_definition_files[name] = SDUtils.make_sdf(oligomer[name], modify_sym_energy=True)
    logger.debug('%s: %d matching oligomers found' % (des_dir.path, len(oligomer)))

    # TODO insert mechanism to Decorate and then gather my own fragment decoration statistics
    # TODO supplement with names info and pull out by names

    #         if line[:15] == 'CRYST1 RECORD: ' and sym in [2, 3]:  # TODO Josh providing in PDB now... can remove here?
    #             cryst = line[15:].strip()

    cluster_residue_d, transformation_dict = SDUtils.gather_fragment_metrics(des_dir, init=True)
    # vUsed for central pair fragment mapping of the biological interface generated fragments
    cluster_freq_tuple_d = {cluster: cluster_residue_d[cluster]['freq'] for cluster in cluster_residue_d}
    # cluster_freq_tuple_d = {cluster: {cluster_residue_d[cluster]['freq'][0]: cluster_residue_d[cluster]['freq'][1]}
    #                         for cluster in cluster_residue_d}

    # READY for all to all fragment incorporation once fragment library is of sufficient size # TODO all_frags
    cluster_freq_d = {cluster: SDUtils.format_frequencies(cluster_residue_d[cluster]['freq'])
                      for cluster in cluster_residue_d}  # orange mapped to cluster tag
    cluster_freq_twin_d = {cluster: SDUtils.format_frequencies(cluster_residue_d[cluster]['freq'], flip=True)
                           for cluster in cluster_residue_d}  # orange mapped to cluster tag
    cluster_residue_d = {cluster: cluster_residue_d[cluster]['pair'] for cluster in cluster_residue_d}

    # cryst = template_pdb.cryst_record

    # Set up protocol symmetry
    protocol = PUtils.protocol[sym]
    # sym_def_file = SDUtils.handle_symmetry(cryst)  # TODO
    sym_def_file = SDUtils.sdf_lookup(sym)  # TODO currently grabbing dummy.symm
    if sym > 1:
        main_cmd += ['-symmetry_definition', 'CRYST1']
    else:
        logger.error('Not possible to input point groups just yet...')
        sys.exit()

    # logger.info('Symmetry Information: %s' % cryst)
    logger.info('Symmetry Option: %s' % protocol)
    logger.info('Input PDBs: %s' % ', '.join(name for name in names))
    logger.info('Pulling fragment info from clusters: %s' % ', '.join(cluster_residue_d))
    for j, pdb_id in enumerate(names):
        logger.info('Fragments identified: Oligomer %s, residues: %s' %
                    (pdb_id, ', '.join(str(cluster_residue_d[cluster][k][j]) for cluster in cluster_residue_d
                                       for k, pair in enumerate(cluster_residue_d[cluster]))))

    # Fetch IJK Cluster Dictionaries and Setup Interface Residues for Residue Number Conversion. MUST BE PRE-RENUMBER
    frag_residue_object_d = SDUtils.residue_number_to_object(template_pdb, cluster_residue_d)
    logger.debug('Fragment Residue Object Dict: %s' % str(frag_residue_object_d))
    # TODO Make chain number independent. Low priority
    int_residues = SDUtils.find_interface_residues(oligomer[pdb_codes[0]], oligomer[pdb_codes[1]])

    # Get residue numbers as Residue objects to map across chain renumbering
    int_residue_objects = {}
    for k, name in enumerate(names):
        int_residue_objects[name] = []
        for residue in int_residues[k]:
            try:
                int_residue_objects[name].append(template_pdb.get_residue(names[name](k), residue))
            except IndexError:
                raise SDUtils.DesignError('Oligomeric and ASU chains do not match. Interface likely involves '
                                          'missing density at oligomer \'%s\', chain \'%s\', residue \'%d\'. Resolve '
                                          'this error and make sure that all input oligomers are symmetrized for '
                                          'optimal script performance.' % (name, names[name](k), residue))

    # Renumber PDB to Rosetta Numbering
    logger.info('Converting to standard Rosetta numbering. 1st residue of chain A is 1, 1st residue of chain B is last '
                'residue in chain A + 1, etc')
    template_pdb.reorder_chains()
    template_pdb.renumber_residues()
    jump = template_pdb.getTermCAAtom('C', template_pdb.chain_id_list[0]).residue_number
    template_residues = template_pdb.get_all_residues()
    logger.info('Last residue of first oligomer %s, chain %s is %d' %
                (list(names.keys())[0], names[list(names.keys())[0]](0), jump))
    logger.info('Total number of residues is %d' % len(template_residues))
    template_pdb.write(des_dir.asu)

    # Mutate all design positions to Ala
    mutated_pdb = copy.deepcopy(template_pdb)
    logger.debug('Cleaned PDB: \'%s\'' % des_dir.asu)

    # Set Up Interface Residues after renumber, remove Side Chain Atoms to Ala NECESSARY for all chains to ASU chain map
    total_int_residue_objects = []
    int_res_numbers = {}
    for c, name in enumerate(names):  # int_residue_objects):
        int_res_numbers[name] = []
        for residue_obj in int_residue_objects[name]:
            total_int_residue_objects.append(residue_obj)
            int_res_numbers[name].append(residue_obj.ca.residue_number)  # must use .ca.residue_number,.number is static
            mutated_pdb.mutate_to(names[name](c), residue_obj.ca.residue_number)

    # Construct CB Tree for full interface atoms to map residue residue contacts
    # total_int_residue_objects = [res_obj for chain in names for res_obj in int_residue_objects[chain]] Now above
    interface = SDUtils.fill_pdb([atom for residue in total_int_residue_objects for atom in residue.atom_list])
    interface_tree = SDUtils.residue_interaction_graph(interface)
    interface_cb_indices = interface.get_cb_indices(InclGlyCA=True)

    interface_residue_edges = {}
    for idx, residue_contacts in enumerate(interface_tree):
        if interface_tree[idx].tolist() != list():
            residue = interface.all_atoms[interface_cb_indices[idx]].residue_number
            contacts = {interface.all_atoms[interface_cb_indices[contact_idx]].residue_number
                        for contact_idx in interface_tree[idx]}
            interface_residue_edges[residue] = contacts - {residue}
    # ^ {78: [14, 67, 87, 109], ...}  green

    # Old residue numbering system, keep for backwards compatibility
    # logger.info('Interface Residues: %s' % ', '.join(str(n) for name in int_res_numbers
    #                                                    for n in int_res_numbers[name]))
    logger.info('Interface Residues: %s' % ', '.join(str(n) + names[name](c) for c, name in enumerate(names)
                                                     for n in int_res_numbers[name]))
    mutated_pdb.write(ala_mut_pdb)
    # mutated_pdb.write(ala_mut_pdb, cryst1=cryst)
    logger.debug('Cleaned PDB for Refine: \'%s\'' % ala_mut_pdb)

    # Get ASU distance parameters
    asu_oligomer_com_dist = []
    for d, name in enumerate(names):
        asu_oligomer_com_dist.append(np.linalg.norm(np.array(template_pdb.center_of_mass())
                                                    - np.array(oligomer[name].center_of_mass())))
    max_com_dist = 0
    for com_dist in asu_oligomer_com_dist:
        if com_dist > max_com_dist:
            max_com_dist = com_dist
    dist = round(math.sqrt(math.ceil(max_com_dist)), 0)
    logger.info('Expanding ASU by %f Angstroms' % dist)

    # Check to see if other poses have collected design sequence info and grab PSSM
    temp_file = os.path.join(des_dir.building_blocks, PUtils.temp)
    rerun = False
    if PUtils.clean not in os.listdir(des_dir.building_blocks):
        shutil.copy(des_dir.asu, des_dir.building_blocks)
        with open(temp_file, 'w') as f:
            f.write('Still fetching data. Process will resume once data is gathered\n')

        pssm_files, pdb_seq, errors, pdb_seq_file, pssm_process = {}, {}, {}, {}, {}
        logger.debug('Fetching PSSM Files')

        # Check if other design combinations have already collected sequence info about design candidates
        for name in names:
            for seq_file in iglob(os.path.join(des_dir.sequences, name + '.*')):
                if seq_file == name + '.hmm':
                    pssm_files[name] = os.path.join(des_dir.sequences, seq_file)
                    logger.debug('%s PSSM Files=%s' % (name, pssm_files[name]))
                    break
                elif seq_file == name + '.fasta':
                    pssm_files[name] = PUtils.temp
            if name not in pssm_files:
                pssm_files[name] = {}
                logger.debug('%s PSSM File not created' % name)

        # Extract/Format Sequence Information
        for n, name in enumerate(names):
            if pssm_files[name] == dict():
                logger.debug('%s is chain %s in ASU' % (name, names[name](n)))
                pdb_seq[name], errors[name] = SDUtils.extract_aa_seq(template_pdb, chain=names[name](n))
                logger.debug('%s Sequence=%s' % (name, pdb_seq[name]))
                if errors[name]:
                    logger.warning('%s: Sequence generation ran into the following residue errors: %s'
                                   % (des_dir.path, ', '.join(errors[name])))
                pdb_seq_file[name] = SDUtils.write_fasta_file(pdb_seq[name], name, outpath=des_dir.sequences)
                if not pdb_seq_file[name]:
                    logger.critical('%s: Unable to parse sequence. Check if PDB \'%s\' is valid' % (des_dir.path, name))
                    raise SDUtils.DesignError('Unable to parse sequence')
                    # raise SDUtils.DesignError('%s: Unable to parse sequence' % des_dir.path)
            else:
                pdb_seq_file[name] = os.path.join(des_dir.sequences, name + '.fasta')

        # Make PSSM of PDB sequence POST-SEQUENCE EXTRACTION
        for name in names:
            if pssm_files[name] == dict():
                logger.info('Generating PSSM file for %s' % name)
                pssm_files[name], pssm_process[name] = SDUtils.hhblits(pdb_seq_file[name], outpath=des_dir.sequences)
                logger.debug('%s seq file: %s' % (name, pdb_seq_file[name]))
            elif pssm_files[name] == PUtils.temp:
                logger.info('Waiting for profile generation...')
                while True:
                    time.sleep(20)
                    if os.path.exists(os.path.join(des_dir.sequences, name + '.hmm')):
                        pssm_files[name] = os.path.join(des_dir.sequences, name + '.hmm')
                        pssm_process[name] = done_process
                        break
            else:
                logger.info('Found PSSM file for %s' % name)
                pssm_process[name] = done_process

        # Wait for PSSM command to complete
        for name in names:
            pssm_process[name].communicate()
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # Extract PSSM for each protein and combine into single PSSM
        pssm_dict = {}
        for name in names:
            pssm_dict[name] = SDUtils.parse_hhblits_pssm(pssm_files[name])
        full_pssm = SDUtils.combine_pssm([pssm_dict[name] for name in pssm_dict])
        pssm_file = SDUtils.make_pssm_file(full_pssm, PUtils.msa_pssm, outpath=des_dir.building_blocks)
    else:
        time.sleep(1)
        while True:
            if os.path.exists(temp_file):
                logger.info('Waiting for profile generation...')
                time.sleep(20)
                continue
            break
        # Check to see if specific profile has been made for the pose.
        if os.path.exists(os.path.join(des_dir.path, PUtils.msa_pssm)):
            pssm_file = os.path.join(des_dir.path, PUtils.msa_pssm)
        else:
            pssm_file = os.path.join(des_dir.building_blocks, PUtils.msa_pssm)
        full_pssm = SDUtils.parse_pssm(pssm_file)

    # Check Pose and Profile for equality before proceeding
    second = False
    while True:
        if len(full_pssm) != len(template_residues):
            logger.warning('%s: Profile and Pose sequences are different lengths!\nProfile=%d, Pose=%d. Generating new '
                           'profile' % (des_dir.path, len(full_pssm), len(template_residues)))
            rerun = True

        if not rerun:
            # Check sequence from Pose and PSSM to compare identity before proceeding
            pssm_res, pose_res = {}, {}
            for res in range(len(template_residues)):
                pssm_res[res] = full_pssm[res]['type']
                pose_res[res] = IUPACData.protein_letters_3to1[template_residues[res].type.title()]
                if pssm_res[res] != pose_res[res]:
                    logger.warning('%s: Profile and Pose sequences are different!\nResidue %d: Profile=%s, Pose=%s. '
                                   'Generating new profile' % (des_dir.path, res + SDUtils.index_offset, pssm_res[res],
                                                               pose_res[res]))
                    rerun = True
                    break

        if rerun:
            if second:
                logger.error('%s: Profile Generation got stuck, design aborted' % des_dir.path)
                raise SDUtils.DesignError('Profile Generation got stuck, design aborted')
                # raise SDUtils.DesignError('%s: Profile Generation got stuck, design aborted' % des_dir.path)
            pssm_file, full_pssm = SDUtils.gather_profile_info(template_pdb, des_dir, names)
            rerun, second = False, True
        else:
            break
        logger.debug('Position Specific Scoring Matrix: %s' % str(full_pssm))

    # Parse Fragment Clusters into usable Dictionaries and Flatten for Sequence Design
    fragment_range = SDUtils.parameterize_frag_length(frag_size)
    full_design_dict = SDUtils.populate_design_dict(len(full_pssm), [j for j in range(*fragment_range)])
    residue_cluster_map = SDUtils.convert_to_residue_cluster_map(frag_residue_object_d, fragment_range)
    # ^cluster_map (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
    #             Where the key is the 0 indexed residue id

    # # TODO all_frags
    cluster_residue_pose_d = SDUtils.residue_object_to_number(frag_residue_object_d)
    # logger.debug('Cluster residues pose number:\n%s' % cluster_residue_pose_d)
    # # ^{cluster: [(78, 87, ...), ...]...}
    residue_freq_map = {residue_set: cluster_freq_d[cluster] for cluster in cluster_freq_d
                        for residue_set in cluster_residue_pose_d[cluster]}  # blue
    # ^{(78, 87, ...): {'A': {'S': 0.02, 'T': 0.12}, ...}, ...}
    # make residue_freq_map inverse pair frequencies with cluster_freq_twin_d
    residue_freq_map.update({tuple(residue for residue in reversed(residue_set)): cluster_freq_twin_d[cluster]
                             for cluster in cluster_freq_twin_d for residue_set in residue_freq_map})

    # remove entries which don't exist on protein because of fragment_index +- residues
    not_available = []
    for residue in residue_cluster_map:
        if residue >= len(full_design_dict) or residue < 0:
            not_available.append(residue)
            logger.warning('In \'%s\', residue %d is represented by a fragment but there is no Atom record for it. '
                           'Fragment index will be deleted.' % (des_dir.path, residue + SDUtils.index_offset))
    for residue in not_available:
        residue_cluster_map.pop(residue)
    logger.debug('Residue Cluster Map: %s' % str(residue_cluster_map))

    cluster_dicts = SDUtils.get_cluster_dicts(db=frag_db, id_list=[j for j in cluster_residue_d])
    full_cluster_dict = SDUtils.deconvolve_clusters(cluster_dicts, full_design_dict, residue_cluster_map)
    final_issm = SDUtils.flatten_for_issm(full_cluster_dict, keep_extras=False)  # =False added for pickling 6/14/20
    interface_data_file = SDUtils.pickle_object(final_issm, frag_db + PUtils.frag_type, out_path=des_dir.data)
    logger.debug('Fragment Specific Scoring Matrix: %s' % str(final_issm))

    # Make DSSM by combining fragment and evolutionary profile
    fragment_alpha = SDUtils.find_alpha(final_issm, residue_cluster_map, db=frag_db)
    dssm = SDUtils.combine_ssm(full_pssm, final_issm, fragment_alpha, db=frag_db, boltzmann=True)
    dssm_file = SDUtils.make_pssm_file(dssm, PUtils.dssm, outpath=des_dir.path)
    # logger.debug('Design Specific Scoring Matrix: %s' % dssm)

    # # Set up consensus design # TODO all_frags
    # # Combine residue fragment information to find residue sets for consensus
    # # issm_weights = {residue: final_issm[residue]['stats'] for residue in final_issm}
    final_issm = SDUtils.offset_index(final_issm)  # change so it is one-indexed
    frag_overlap = SDUtils.fragment_overlap(final_issm, interface_residue_edges, residue_freq_map)  # all one-indexed
    logger.debug('Residue frequency map:\n%s' % residue_freq_map)
    logger.debug('Residue interface edges:\n%s' % interface_residue_edges)  # This is perfect for Bale 2016 int connect
    logger.debug('Residue fragment overlap:\n%s' % frag_overlap)
    #
    # for pair in residue_freq_map:
    #     for res in pair:
    #         if frag_overlap[res] == set():
    #             consensus = False  # if no amino acids are possible for residue, quit
    #             break
    #     if consensus:
    #         for pair in cluster_freq_tuple_d:
    #
    # consensus_residues = SDUtils.overlap_consensus(final_issm, frag_overlap)
    # # ^ {23: 'T', 29: 'A', ...}

    # residue_cluster_map
    # ^{48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
    #             Where the key is the 0 indexed residue id
    # cluster_freq_tuple_d
    # ^{1_1_54: [(('A', 'G'), 0.2963), (('G', 'A'), 0.1728), (('G', 'G'), 0.1235), (('G', 'S'), 0.0741), (('G', 'M'),
    #            0.0741), (('G', 'L'), 0.0741), (('H', 'G'), 0.0741), (('A', 'A'), 0.0247), (('G', 'T'), 0.0247),
    #            (('G', 'N'), 0.0247), (('G', 'I'), 0.0247), (('G', 'Q'), 0.0123)], ...}
    # frag_overlap
    # {55: {'A', 'G', 'H'}, 56: set(), 223: set(), 224: {'I', 'N', 'L', 'A', 'G', 'T', 'M', 'Q', 'S'}, 225: set()}
    #
    # residue_freq_map
    # ^{(78, 87, ...): {'A': {'S': 0.02, 'T': 0.12}, ...}, ...}

    consensus_residues = {}
    all_pose_fragment_pairs = list(residue_freq_map.keys())
    residue_cluster_map = SDUtils.offset_index(residue_cluster_map)  # change so it is one-indexed
    # for residue in residue_cluster_map:
    for residue, partner in all_pose_fragment_pairs:
        for idx, cluster in residue_cluster_map[residue]['cluster']:
            if idx == 0:  # check if the fragment index is 0. No current information for other pairs 07/24/20
                for idx_p, cluster_p in residue_cluster_map[partner]['cluster']:
                    if idx_p == 0:  # check if the fragment index is 0. No current information for other pairs 07/24/20
                        if residue_cluster_map[residue]['chain'] == 'mapped':
                            # choose first AA from AA tuple in residue frequency d
                            aa_i, aa_j = 0, 1
                        else:  # choose second AA from AA tuple in residue frequency d
                            aa_i, aa_j = 1, 0
                        for pair_freq in cluster_freq_tuple_d[cluster]:
                            # if cluster_freq_tuple_d[cluster][k][0][aa_i] in frag_overlap[residue]:
                            if residue in frag_overlap:  # edge case where fragment has no weight but it is center res
                                if pair_freq[0][aa_i] in frag_overlap[residue]:
                                    # if cluster_freq_tuple_d[cluster][k][0][aa_j] in frag_overlap[partner]:
                                    if partner in frag_overlap:
                                        if pair_freq[0][aa_j] in frag_overlap[partner]:
                                            consensus_residues[residue] = pair_freq[0][aa_i]
                                            break  # because pair_freq's are sorted we end at the highest matching pair

    consensus = {residue: dssm[residue]['type'] for residue in dssm}
    # ^{0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00, 'weight': 0.00}, ...}}
    consensus.update(consensus_residues)
    consensus = SDUtils.offset_index(consensus)
    # consensus = SDUtils.consensus_sequence(dssm)
    logger.debug('Consensus Residues only:\n%s' % consensus_residues)
    logger.debug('Consensus:\n%s' % consensus)
    for n, name in enumerate(names):
        for residue in int_res_numbers[name]:  # one-indexed
            mutated_pdb.mutate_to(names[name](n), residue, IUPACData.protein_letters_1to3[consensus[residue]].upper())
    mutated_pdb.write(consensus_pdb)
    # mutated_pdb.write(consensus_pdb, cryst1=cryst)

    # Update DesignDirectory with design information
    des_dir.info['pssm'] = pssm_file
    des_dir.info['issm'] = interface_data_file
    des_dir.info['dssm'] = dssm_file
    des_dir.info['db'] = frag_db
    des_dir.info['des_residues'] = [j for name in names for j in int_res_numbers[name]]
    info_pickle = SDUtils.pickle_object(des_dir.info, 'info', out_path=des_dir.data)

    # RELAX: Prepare command and flags file
    refine_variables = [('pdb_reference', des_dir.asu), ('scripts', PUtils.rosetta_scripts),
                        ('sym_score_patch', PUtils.sym_weights), ('symmetry', protocol), ('sdf', sym_def_file),
                        ('dist', dist), ('cst_value', cst_value), ('cst_value_sym', (cst_value / 2))]
    for k, name in enumerate(names):
        refine_variables.append(('interface' + names[name](k), ','.join(str(j) + names[name](k)
                                                                        for j in int_res_numbers[name])))
    # Old method using residue index. Keep until all backwards compatibility is cleared TODO remove
    # for k, name in enumerate(int_res_numbers, 1):
    #     refine_variables.append(('interface' + str(k), ','.join(str(j) for j in int_res_numbers[name])))

    flags_refine = SDUtils.prepare_rosetta_flags(refine_variables, PUtils.stage[1], outpath=des_dir.path)
    relax_cmd = main_cmd + \
                ['@' + os.path.join(des_dir.path, flags_refine), '-scorefile', os.path.join(des_dir.scores, PUtils.scores_file),
                 '-parser:protocol', os.path.join(PUtils.rosetta_scripts, PUtils.stage[1] + '.xml')]
    refine_cmd = relax_cmd + ['-in:file:s', ala_mut_pdb,  '-parser:script_vars', 'switch=%s' % PUtils.stage[1]]
    consensus_cmd = relax_cmd + ['-in:file:s', consensus_pdb, '-parser:script_vars', 'switch=%s' % PUtils.stage[5]]

    # Create executable/Run FastRelax on Clean ASU/Consensus ASU with RosettaScripts
    if script:
        SDUtils.write_shell_script(subprocess.list2cmdline(refine_cmd), name=PUtils.stage[1], outpath=des_dir.path,
                                   additional=[subprocess.list2cmdline(consensus_cmd)])
        SDUtils.write_shell_script(subprocess.list2cmdline(consensus_cmd), name=PUtils.stage[5], outpath=des_dir.path)
    else:
        if not suspend:
            logger.info('Refine Command: %s' % subprocess.list2cmdline(relax_cmd))
            refine_process = subprocess.Popen(relax_cmd)
            # Wait for Rosetta Refine command to complete
            refine_process.communicate()
            logger.info('Consensus Command: %s' % subprocess.list2cmdline(consensus_cmd))
            consensus_process = subprocess.Popen(consensus_cmd)
            # Wait for Rosetta Consensus command to complete
            consensus_process.communicate()

    # DESIGN: Prepare command and flags file
    design_variables = copy.deepcopy(refine_variables)
    design_variables.append(('pssm_file', dssm_file))  # TODO change name to dssm_file after P432
    flags_design = SDUtils.prepare_rosetta_flags(design_variables, PUtils.stage[2], outpath=des_dir.path)
    # TODO back out nstruct label to command distribution
    design_cmd = main_cmd + \
                 ['-in:file:s', refined_pdb, '-in:file:native', des_dir.asu, '-nstruct', str(PUtils.nstruct),
                  '@' + os.path.join(des_dir.path, flags_design), '-in:file:pssm', pssm_file, '-parser:protocol',
                  os.path.join(PUtils.rosetta_scripts, PUtils.stage[2] + '.xml'),
                  '-scorefile', os.path.join(des_dir.scores, PUtils.scores_file)]

    # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
    # TODO if nstruct is backed out, create pdb_list for metrics distribution
    pdb_list_file = SDUtils.pdb_list_file(refined_pdb, total_pdbs=PUtils.nstruct, suffix='_' + PUtils.stage[2],
                                          loc=des_dir.design_pdbs, additional=[consensus_design_pdb, ])
    design_variables += [('sdfA', sym_definition_files[pdb_codes[0]]), ('sdfB', sym_definition_files[pdb_codes[1]])]

    flags_metric = SDUtils.prepare_rosetta_flags(design_variables, PUtils.stage[3], outpath=des_dir.path)
    metric_cmd = main_cmd + \
                 ['-in:file:l', pdb_list_file, '-in:file:native', refined_pdb, '@' + os.path.join(des_dir.path, flags_metric),
                  '-out:file:score_only', os.path.join(des_dir.scores, PUtils.scores_file),
                  '-parser:protocol', os.path.join(PUtils.rosetta_scripts, PUtils.stage[3] + '.xml')]

    if mpi:
        design_cmd = CUtils.run_cmds[PUtils.rosetta_extras] + design_cmd
        metric_cmd = CUtils.run_cmds[PUtils.rosetta_extras] + metric_cmd
    metric_cmds = {name: metric_cmd + ['-parser:script_vars', 'chain=%s' % names[name](n)]
                   for n, name in enumerate(names)}
    # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics on Designs
    if script:
        SDUtils.write_shell_script(subprocess.list2cmdline(design_cmd), name=PUtils.stage[2], outpath=des_dir.path)
        SDUtils.write_shell_script(subprocess.list2cmdline(metric_cmds[pdb_codes[0]]), name=PUtils.stage[3],
                                   outpath=des_dir.path, additional=[subprocess.list2cmdline(metric_cmds[name])
                                                                     for n, name in enumerate(names) if n > 0])
    else:
        if not suspend:
            logger.info('Design Command: %s' % subprocess.list2cmdline(design_cmd))
            design_process = subprocess.Popen(design_cmd)
            # Wait for Rosetta Design command to complete
            design_process.communicate()
            for name in metric_cmds:
                logger.info('Metrics Command: %s' % subprocess.list2cmdline(metric_cmds[name]))
                metrics_process = subprocess.Popen(metric_cmds[name])
                metrics_process.communicate()

    # ANALYSIS: each output from the Design process based on score, Analyze Sequence Variation
    if script:
        analysis_cmd = 'python %s -d %s' % (PUtils.filter_designs, des_dir.path)
        SDUtils.write_shell_script(analysis_cmd, name=PUtils.stage[4], outpath=des_dir.path)
    else:
        if not suspend:
            pose_s = analyze_output(des_dir)
            outpath = os.path.join(des_dir.all_scores, PUtils.analysis_file)
            _header = False
            if not os.path.exists(outpath):
                _header = True
            pose_s.to_csv(outpath, mode='a', header=_header)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=PUtils.program_name +
                                     '\nGather output from %s and format it for input into Rosetta for interface'
                                     ' design. Design is constrained by fragment profiles extracted from the PDB and '
                                     'evolutionary profiles of homologous sequences' % PUtils.nano)
    parser.add_argument('-d', '--directory', type=str, help='Directory where %s output is located. Default=CWD'
                                                            % PUtils.nano, default=os.getcwd())
    parser.add_argument('-f', '--file', type=str, help='File with location(s) of %s output.' % PUtils.nano,
                        default=None)
    # TODO, function for selecting the appropriate interface library given a viable input
    parser.add_argument('-i', '--fragment_database', type=str, help='Database to match fragments for interface specific'
                                                                    ' scoring matrices. Default=biological_interfaces',
                        default=PUtils.biological_fragmentDB)
    parser.add_argument('symmetry_group', type=int, help='What type of symmetry group does your design belong too? '
                                                         'One of 0-Point Group, 2-Plane Group, or 3-Space Group')
    parser.add_argument('-c', '--command_only', action='store_true', help='Should commands be written but not executed?'
                                                                          '\nDefault=False')
    parser.add_argument('-m', '--multi_processing', action='store_true', help='Should job be run with multiprocessing?'
                                                                              '\nDefault=False')
    parser.add_argument('-x', '--suspend', action='store_true', help='Should Rosetta design trajectory be suspended?\n'
                                                                     'Default=False')
    parser.add_argument('-p', '--mpi', action='store_true', help='Should job be set up for cluster submission?\n'
                                                                 'Default=False')
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out?\nDefault=False')
    args = parser.parse_args()
    extras = ''

    # Start logging output
    if args.debug:
        logger = SDUtils.start_log(name=os.path.basename(__file__), level=1)
        logger.debug('Debug mode. Verbose output')
    else:
        logger = SDUtils.start_log(name=os.path.basename(__file__), level=2)

    logger.info('Starting %s with options:\n%s' %
                (os.path.basename(__file__),
                 '\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    assert args.symmetry_group in PUtils.protocol, logger.critical(
        'Symmetry group \'%s\' is not available. Please choose from %s' %
        (args.symmetry_group, ', '.join(sym for sym in PUtils.protocol)))

    # Collect all designs to be processed
    all_designs, location = SDUtils.collect_designs(args.directory, file=args.file)
    assert all_designs != list(), logger.critical('No %s directories found within \'%s\' input! Please ensure correct '
                                                  'location.' % (PUtils.nano, location))
    logger.info('%d Poses found in \'%s\'' % (len(all_designs), location))
    all_design_dirs = SDUtils.set_up_directory_objects(all_designs)
    logger.info('All pose specific logs are located in their corresponding directories.\nEx: \'%s\'' %
                os.path.join(all_design_dirs[0].path, os.path.basename(all_design_dirs[0].path) + '.log'))

    if args.mpi:
        args.command_only = True
        extras = ' mpi %d' % CUtils.mpi
        logger.info('Setting job up for submission to MPI capable computer. Pose trajectories will run in parallel, '
                    '%s at a time. This will speed up pose processing %f-fold.' %
                    (CUtils.mpi - 1, PUtils.nstruct / CUtils.mpi - 1))
    if args.command_only:
        args.suspend = True
        logger.info('Writing modelling commands out to file only, no modelling will occur until commands are executed')

    # Start Pose processing and preparation for Rosetta
    if args.multi_processing:
        # Calculate the number of threads to use depending on computer resources
        mp_threads = SDUtils.calculate_mp_threads(mpi=args.mpi, maximum=True, no_model=args.suspend)
        logger.info('Starting multiprocessing %s threads' % str(mp_threads))
        zipped_args = zip(all_design_dirs, repeat(args.fragment_database), repeat(args.symmetry_group),
                          repeat(args.command_only), repeat(args.mpi), repeat(args.suspend),
                          repeat(args.debug))  # repeat(args.prioritize_frags)
        results, exceptions = SDUtils.mp_try_starmap(initialization_mp, zipped_args, mp_threads)
        if exceptions:
            logger.warning('\nThe following exceptions were thrown. Design for these directories is inaccurate.')
            for exception in exceptions:
                logger.warning(exception)
    else:
        logger.info('Starting processing. If single process is taking awhile, use -m during submission')
        for des_directory in all_design_dirs:
            initialization_s(des_directory, args.fragment_database, args.symmetry_group, script=args.command_only,
                             mpi=args.mpi, suspend=args.suspend, debug=args.debug)

    if args.command_only:
        all_commands = [[] for s in PUtils.stage_f]
        command_files = [[] for s in PUtils.stage_f]
        for des_directory in all_design_dirs:
            for i, stage in enumerate(PUtils.stage_f):
                all_commands[i].append(os.path.join(des_directory.path, stage + '.sh'))
        for i, stage in enumerate(PUtils.stage_f):
            if i > 3:  # No consensus
                break
            command_files[i] = SDUtils.write_commands(all_commands[i], name=stage, loc=args.directory)
            logger.info('All \'%s\' commands were written to \'%s\'' % (stage, command_files[i]))
            logger.info('\nTo process all commands in correct order, execute:\ncd %s\n%s\n%s\n%s' %
                        (args.directory, 'python ' + __file__ + ' distribute -s refine',
                         'python ' + __file__ + ' distribute -s design',
                         'python ' + __file__ + ' distribute -s metrics'))
