import argparse
import copy
import math
import os
import shutil
import subprocess
import time
from glob import glob, iglob
from itertools import repeat

import numpy as np
from Bio.Data.IUPACData import protein_letters_1to3, protein_letters_3to1
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from scipy.spatial.distance import euclidean, pdist

import CommandDistributer
import PathUtils as PUtils
import SymDesignUtils as SDUtils
from DesignMetrics import analyze_output
from PDB import PDB
import SequenceProfile
from SequenceProfile import SequenceProfile, residue_number_to_object, extract_aa_seq
from SymDesignUtils import start_log, write_fasta_file

logger = start_log(name=__name__)


# @SDUtils.handle_design_errors(errors=(SDUtils.DesignError, AssertionError))
# def initialization_s(des_dir, frag_db, sym, script=False, mpi=False, suspend=False, debug=False):
#     return initialization(des_dir, frag_db, sym, script=script, mpi=mpi, suspend=suspend, debug=debug)
#
#
# def initialization_mp(des_dir, frag_db, sym, script=False, mpi=False, suspend=False, debug=False):
#     try:
#         pose = initialization(des_dir, frag_db, sym, script=script, mpi=mpi, suspend=suspend, debug=debug)
#         # return initialization(des_dir, frag_db, sym, script=script, mpi=mpi, suspend=suspend, debug=debug), None
#         return pose, None
#     except (SDUtils.DesignError, AssertionError) as e:
#         return None, (des_dir.path, e)
#     # finally:
#     #     print('Error occurred in %s' % des_dir.path)


@SDUtils.handle_errors((SDUtils.DesignError, AssertionError))
def initialization(des_dir, frag_db, sym, script=False, mpi=False, suspend=False, debug=False):
    # Variable initialization
    cst_value = round(0.2 * CommandDistributer.reference_average_residue_weight, 2)
    frag_size = 5
    done_process = subprocess.Popen(['run="go"'], shell=True)  # Todo remove shell=True

    # Log output
    if debug:
        # global logger
        logger = SDUtils.start_log(name=__name__, handler=2, level=1,
                                   location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))
    else:
        logger = SDUtils.start_log(name=__name__, handler=2, propagate=True,
                                   location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))
    logger.info('Processing directory \'%s\'' % des_dir.path)

    # Set up Rosetta command, files
    main_cmd = copy.deepcopy(CommandDistributer.script_cmd)
    # cleaned_pdb = os.path.join(des_dir.path, PUtils.clean_asu)
    # ala_mut_pdb = os.path.splitext(cleaned_pdb)[0] + '_for_refine.pdb'
    # TODO no mut if glycine...
    # ala_mut_pdb = os.path.splitext(des_dir.asu)[0] + '_for_refine.pdb'
    # consensus_pdb = os.path.splitext(des_dir.asu)[0] + '_for_consensus.pdb'
    # consensus_design_pdb = os.path.join(des_dir.designs, os.path.splitext(des_dir.asu)[0] + '_for_consensus.pdb')
    # refined_pdb = os.path.join(des_dir.designs, os.path.basename(des_dir.refine_pdb))
    # refined_pdb = os.path.join(des_dir.designs, os.path.splitext(os.path.basename(ala_mut_pdb))[0] + '.pdb')
    # '_%s.pdb' % PUtils.stage[1]) TODO clean this stupid mechanism only for P432
    # if out:file:o works, could use, os.path.join(des_dir.designs, PUtils.stage[1] + '.pdb') but it won't register

    # Extract information from SymDock Output
    des_dir.gather_docking_metrics()
    des_dir.retrieve_pose_metrics_from_file()
    pdb_codes = str(os.path.basename(des_dir.composition)).split('_')
    # cluster_residue_d, transformation_dict = SDUtils.gather_fragment_metrics(des_dir)

    # Fetch PDB object of each chain from PDBdb or PDB server # TODO set up pdb database
    # UNCOMMMENT WHEN DATABASE IS SET UP
    # # oligomers = SDUtils.fetch_pdbs(des_dir, pdb_codes)
    # oligomers = {pdb_code: PDB(file=SDUtils.fetch_pdbs(pdb_code)) for pdb_code in pdb_codes}
    #
    # for i, name in enumerate(oligomers):
    #     # oligomers[name].translate(oligomers[name].center_of_mass())
    #     oligomers[name].orient(symm, PUtils.orient)
    #     oligomers[name].rotate_translate(transformation_dict[i]['rot/deg'], transformation_dict[i]['tx_int'])
    #     oligomers[name].rotate_translate(transformation_dict[i]['setting'], transformation_dict[i]['tx_ref'])
    #     # {1: {'rot/deg': [[], ...],'tx_int': [], 'setting': [[], ...], 'tx_ref': []}, ...}

    template_pdb = PDB.from_file(des_dir.source)
    num_chains = len(template_pdb.chain_ids)

    # if num_chains != 2:
    if num_chains != len(pdb_codes):
        oligomer_file = glob(os.path.join(des_dir.path, pdb_codes[0] + '_tx_*.pdb'))
        assert len(oligomer_file) == 1, 'More than one matching file found with %s_tx_*.pdb' % pdb_codes[0]
        # assert len(oligomer_file) == 1, '%s: More than one matching file found with %s' % \
        #                                 (des_dir.path, pdb_codes[0] + '_tx_*.pdb')
        first_oligomer = PDB.from_file(oligomer_file[0])
        # first_oligomer = SDUtils.read_pdb(oligomer_file[0])
        # find the number of ATOM records for template_pdb chain1 using the same oligomeric chain as model
        for atom_idx in range(len(first_oligomer.chain(template_pdb.chain_ids[0]))):
            template_pdb.atoms[atom_idx].chain = template_pdb.chain_ids[0].lower()
        template_pdb.chain_ids = [template_pdb.chain_ids[0].lower(), template_pdb.chain_ids[0]]
        num_chains = len(template_pdb.chain_ids)
        logger.warning('%s: Incorrect chain count: %d. Chains probably have the same id! Temporarily changing IDs\'s to'
                       ' %s' % (des_dir.path, num_chains, template_pdb.chain_ids))
        # Save the renamed chain PDB to central_asu.pdb
        template_pdb.write(out_path=des_dir.source)

    assert len(pdb_codes) == num_chains, 'Number of chains \'%d\' in ASU doesn\'t match number of building blocks ' \
                                         '\'%d\'' % (num_chains, len(pdb_codes))
    # assert len(pdb_codes) == num_chains, '%s: Number of chains \'%d\' in ASU doesn\'t match number of building blocks '\
    #                                      '\'%d\'' % (des_dir.path, num_chains, len(pdb_codes))

    # Set up names object containing pdb id and chain info
    names = {}
    for c, chain in enumerate(template_pdb.chain_ids):
        names[pdb_codes[c]] = template_pdb.get_chain_index
    logger.debug('Chain, Name Pairs: %s' % ', '.join(oligomer + ', ' + str(value(c)) for c, (oligomer, value) in
                                                     enumerate(names.items())))

    # Fetch PDB object of each chain individually from the design directory TODO replaced above by database retrieval
    oligomer = {}
    sym_definition_files = {}
    for name in names:
        name_pdb_file = glob(os.path.join(des_dir.path, name + '_tx_*.pdb'))
        assert len(name_pdb_file) == 1, 'More than one matching file found with %s_tx_*.pdb' % name
        # assert len(name_pdb_file) == 1, '%s: More than one matching file found with %s' % \
        #                                 (des_dir.path, name + '_tx_*.pdb')
        oligomer[name] = PDB.from_file(name_pdb_file[0])
        # oligomer[name] = SDUtils.read_pdb(name_pdb_file[0])
        oligomer[name].name = name
        # TODO Chains must be symmetrized on input before SDF creation, currently raise DesignError
        sdf_file_name = os.path.join(os.path.dirname(oligomer[name].filepath), '%s.sdf' % oligomer[name].name)
        sym_definition_files[name] = oligomer[name].make_sdf(out_path=sdf_file_name, modify_sym_energy=True)
        oligomer[name].reorder_chains()
    logger.debug('%s: %d matching oligomers found' % (des_dir.path, len(oligomer)))

    # TODO insert mechanism to Decorate and then gather my own fragment decoration statistics
    # TODO supplement with names info and pull out by names

    cluster_residue_d = des_dir.pose_fragments()
    transformation_dict = des_dir.pose_transformation()
    # cluster_residue_d, transformation_dict = Pose.gather_fragment_metrics(des_dir, init=True)
    # v Used for central pair fragment mapping of the biological interface generated fragments
    cluster_freq_tuple_d = {cluster: cluster_residue_d[cluster]['freq'] for cluster in cluster_residue_d}
    # cluster_freq_tuple_d = {cluster: {cluster_residue_d[cluster]['freq'][0]: cluster_residue_d[cluster]['freq'][1]}
    #                         for cluster in cluster_residue_d}

    # READY for all to all fragment incorporation once fragment library is of sufficient size # TODO all_frags
    cluster_freq_d = {cluster: SequenceProfile.format_frequencies(cluster_residue_d[cluster]['freq'])
                      for cluster in cluster_residue_d}  # orange mapped to cluster tag
    cluster_freq_twin_d = {cluster: SequenceProfile.format_frequencies(cluster_residue_d[cluster]['freq'], flip=True)
                           for cluster in cluster_residue_d}  # orange mapped to cluster tag
    cluster_residue_d = {cluster: cluster_residue_d[cluster]['pair'] for cluster in cluster_residue_d}

    # Set up protocol symmetry
    # sym_entry_number, oligomer_symmetry_1, oligomer_symmetry_2, design_symmetry = des_dir.symmetry_parameters()
    # sym = SDUtils.handle_symmetry(sym_entry_number)  # This makes the process dependent on the PUtils.master_log file
    protocol = PUtils.protocol[des_dir.design_dimension]
    if des_dir.design_dimension > 0:  # layer or space
        sym_def_file = SDUtils.sdf_lookup(None)  # grabs dummy.sym
        main_cmd += ['-symmetry_definition', 'CRYST1']
    else:  # point
        sym_def_file = SDUtils.sdf_lookup(des_dir.sym_entry_number)
        main_cmd += ['-symmetry_definition', sym_def_file]

    # logger.info('Symmetry Information: %s' % cryst)
    logger.info('Symmetry Option: %s' % protocol)
    logger.info('Input PDBs: %s' % ', '.join(name for name in names))
    logger.info('Pulling fragment info from clusters: %s' % ', '.join(cluster_residue_d))
    for j, pdb_id in enumerate(names):
        logger.info('Fragments identified: Oligomer %s, residues: %s' %
                    (pdb_id, ', '.join(str(cluster_residue_d[cluster][k][j]) for cluster in cluster_residue_d
                                       for k, pair in enumerate(cluster_residue_d[cluster]))))

    # Fetch IJK Cluster Dictionaries and Setup Interface Residues for Residue Number Conversion. MUST BE PRE-RENUMBER
    frag_residue_object_d = residue_number_to_object(template_pdb, cluster_residue_d)
    logger.debug('Fragment Residue Object Dict: %s' % str(frag_residue_object_d))
    # TODO Make chain number independent. Low priority
    int_residues = Pose.find_interface_residues(oligomer[pdb_codes[0]], oligomer[pdb_codes[1]])
    # Get full assembly coordinates. Works for every possible symmetry even if template_pdb.get_uc_dimensions() is None
    # symmetrized_model = Model(expand_asu(template_pdb, des_dir.design_symmetry, uc_dimensions=template_pdb.get_uc_dimensions()))
    symmetrized_model_chain1 = symmetrized_model.select_chain(oligomer[pdb_codes[0]])
    symmetrized_model_chain1_coords = symmetrized_model_chain1.extract_cb_coords_chain(oligomer[pdb_codes[0]], InclGlyCA=True)
    symmetrized_model_chain2 = symmetrized_model.select_chain(oligomer[pdb_codes[1]])
    # should I split this into the oligomeric component parts?
    oligomer_symmetry_int_residues = Pose.find_interface_residues(oligomer[pdb_codes[0]], symmetrized_model_chain1_coords)

    # Get residue numbers as Residue objects to map across chain renumbering
    int_residue_objects = {}
    for k, name in enumerate(names):
        int_residue_objects[name] = []
        for residue in int_residues[k]:
            try:
                int_residue_objects[name].append(template_pdb.chain(names[name](k)).get_residue(residue))
                # int_residue_objects[name].append(template_pdb.get_residue(names[name](k), residue))
            except IndexError:
                raise SDUtils.DesignError('Oligomeric and ASU chains do not match. Interface likely involves '
                                          'missing density at oligomer \'%s\', chain \'%s\', residue \'%d\'. Resolve '
                                          'this error and make sure that all input oligomers are symmetrized for '
                                          'optimal script performance.' % (name, names[name](k), residue))

    # Renumber PDB to Pose Numbering
    logger.info('Converting to standard Rosetta numbering. 1st residue of chain A is 1, 1st residue of chain B is last '
                'residue in chain A + 1, etc')
    template_pdb.reorder_chains()

    # # TODO Insert loops identified by comparison of SEQRES and ATOM
    # pdb_atom_seq = get_pdb_sequences(template_pdb)
    # pose_offset_d = Ams.pdb_to_pose_num(pdb_atom_seq)
    # if template_pdb.atom_sequences:
    #     missing_termini_d = {chain: generate_mutations_from_seq(pdb_atom_seq[chain],
    #                                                             template_pdb.atom_sequences[chain], offset=True,
    #                                                             termini=True) for chain in template_pdb.chain_ids}
    #     gapped_residues_d = {chain: generate_mutations_from_seq(pdb_atom_seq[chain], template_pdb.atom_sequences,
    #                                                             offset=True, reference_gaps=True)
    #                          for chain in template_pdb.chain_ids}
    #     all_missing_residues_d = {chain: generate_mutations_from_seq(pdb_atom_seq[chain],
    #                                                                  template_pdb.atom_sequences,
    #                                                                  offset=True, only_gaps=True)
    #                               for chain in template_pdb.chain_ids}
    #
    #     # Modify residue indices to pose numbering
    #     all_missing_residues_d = {chain: {residue + pose_offset_d[chain]: all_missing_residues_d[chain][residue]
    #                                   for residue in all_missing_residues_d[chain]} for chain in all_missing_residues_d}
    #
    #     for chain in gapped_residues_d:
    #         for residue in gapped_residues_d[chain]:
    #
    #             if residue not in loops_file[loop]:
    #             template_pdb.insert_residue(chain, residue, gapped_residues_d[chain][residue]['from'])

    template_pdb.renumber_residues()
    jump = template_pdb.chain(template_pdb.chain_ids[0]).c_terminal_residue.number
    template_residues = template_pdb.residues
    logger.info('Last residue of first oligomer %s, chain %s is %d' %
                (list(names.keys())[0], names[list(names.keys())[0]](0), jump))
    logger.info('Total number of residues is %d' % len(template_residues))

    # Save renumbered PDB to clean_asu.pdb
    template_pdb.write(out_path=des_dir.asu)
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
            int_res_numbers[name].append(residue_obj.number)  # Todo ensure .number is accessor to residue.ca Atom obj
            mutated_pdb.mutate_residue(number=residue_obj.number)
            # mutated_pdb.mutate_to(names[name](c), residue_obj.number)
            # Todo no mutation from GLY to ALA

    # Construct CB Tree for full interface atoms to map residue residue contacts
    # total_int_residue_objects = [res_obj for chain in names for res_obj in int_residue_objects[chain]] Now above
    interface = PDB.from_atoms([atom for residue in total_int_residue_objects for atom in residue.atoms])
    interface_tree = SDUtils.residue_interaction_graph(interface)
    interface_cb_indices = interface.cb_indices

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
    mutated_pdb.write(out_path=des_dir.refine_pdb)
    # mutated_pdb.write(ala_mut_pdb)
    # mutated_pdb.write(ala_mut_pdb, cryst1=cryst)
    logger.debug('Cleaned PDB for Refine: \'%s\'' % des_dir.refine_pdb)

    # Get ASU distance parameters
    asu_oligomer_com_dist = []
    for d, name in enumerate(names):
        asu_oligomer_com_dist.append(np.linalg.norm(np.array(template_pdb.get_center_of_mass())
                                                    - np.array(oligomer[name].get_center_of_mass())))
    max_com_dist = 0
    for com_dist in asu_oligomer_com_dist:
        if com_dist > max_com_dist:
            max_com_dist = com_dist
    dist = round(math.sqrt(math.ceil(max_com_dist)), 0)
    logger.info('Expanding ASU by %f Angstroms' % dist)

    # Check to see if other poses have collected design sequence info and grab PSSM
    temp_file = os.path.join(des_dir.composition, PUtils.temp)
    rerun = False
    if PUtils.clean_asu not in os.listdir(des_dir.composition):
        shutil.copy(des_dir.asu, des_dir.composition)
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
                logger.debug('%s PSSM File not yet created' % name)

        # Extract/Format Sequence Information
        for n, name in enumerate(names):
            if pssm_files[name] == dict():
                logger.debug('%s is chain %s in ASU' % (name, names[name](n)))
                pdb_seq[name], errors[name] = extract_aa_seq(template_pdb, chain=names[name](n))
                logger.debug('%s Sequence=%s' % (name, pdb_seq[name]))
                if errors[name]:
                    logger.warning('%s: Sequence generation ran into the following residue errors: %s'
                                   % (des_dir.path, ', '.join(errors[name])))
                pdb_seq_file[name] = write_fasta_file(pdb_seq[name], name, out_path=des_dir.sequences)
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
                pssm_files[name], pssm_process[name] = SequenceProfile.hhblits(pdb_seq_file[name], outpath=des_dir.sequences)
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
        pssm_dict = {name: SequenceProfile.parse_hhblits_pssm(pssm_files[name]) for name in names}
        full_pssm = SequenceProfile.combine_pssm([pssm_dict[name] for name in pssm_dict])  # requires python3.6 or greater
        pssm_file = SequenceProfile.make_pssm_file(full_pssm, PUtils.pssm, outpath=des_dir.composition)
    else:
        time.sleep(1)
        while os.path.exists(temp_file):
            logger.info('Waiting for profile generation...')
            time.sleep(20)
        # Check to see if specific profile has been made for the pose.
        if os.path.exists(os.path.join(des_dir.path, PUtils.pssm)):
            pssm_file = os.path.join(des_dir.path, PUtils.pssm)
        else:
            pssm_file = os.path.join(des_dir.composition, PUtils.pssm)
        full_pssm = SequenceProfile.parse_pssm(pssm_file)

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
                pose_res[res] = protein_letters_3to1[template_residues[res].type.title()]
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
            pssm_file, full_pssm = gather_profile_info(template_pdb, des_dir, names)
            rerun, second = False, True
        else:
            break
        logger.debug('Position Specific Scoring Matrix: %s' % str(full_pssm))

    # Parse Fragment Clusters into usable Dictionaries and Flatten for Sequence Design
    fragment_range = SequenceProfile.parameterize_frag_length(frag_size)
    full_design_dict = SequenceProfile.populate_design_dict(len(full_pssm), [j for j in range(*fragment_range)])
    residue_cluster_map = SequenceProfile.convert_to_residue_cluster_map(frag_residue_object_d, fragment_range)
    # ^cluster_map (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
    #             Where the key is the 0 indexed residue id

    # # TODO all_frags
    cluster_residue_pose_d = SequenceProfile.residue_object_to_number(frag_residue_object_d)
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

    cluster_dicts = SequenceProfile.get_cluster_dicts(db=frag_db, id_list=[j for j in cluster_residue_d])
    full_cluster_dict = SequenceProfile.deconvolve_clusters(cluster_dicts, full_design_dict, residue_cluster_map)
    final_issm = SequenceProfile.flatten_for_issm(full_cluster_dict, keep_extras=False)  # =False added for pickling 6/14/20
    interface_data_file = SDUtils.pickle_object(final_issm, frag_db + PUtils.frag_profile, out_path=des_dir.data)
    logger.debug('Fragment Specific Scoring Matrix: %s' % str(final_issm))

    # Make DSSM by combining fragment and evolutionary profile
    fragment_alpha = SequenceProfile.find_alpha(final_issm, residue_cluster_map, db=frag_db)
    dssm = SequenceProfile.combine_ssm(full_pssm, final_issm, fragment_alpha, db=frag_db, boltzmann=True)
    dssm_file = SequenceProfile.make_pssm_file(dssm, PUtils.dssm, outpath=des_dir.path)
    # logger.debug('Design Specific Scoring Matrix: %s' % dssm)

    # # Set up consensus design # TODO all_frags
    # # Combine residue fragment information to find residue sets for consensus
    # # issm_weights = {residue: final_issm[residue]['stats'] for residue in final_issm}
    final_issm = SequenceProfile.offset_index(final_issm)  # change so it is one-indexed
    frag_overlap = SequenceProfile.fragment_overlap(final_issm, interface_residue_edges, residue_freq_map)  # all one-indexed
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
    residue_cluster_map = SequenceProfile.offset_index(residue_cluster_map)  # change so it is one-indexed
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
    consensus = SequenceProfile.offset_index(consensus)
    # consensus = SDUtils.consensus_sequence(dssm)
    logger.debug('Consensus Residues only:\n%s' % consensus_residues)
    logger.debug('Consensus:\n%s' % consensus)
    for n, name in enumerate(names):
        for residue in int_res_numbers[name]:  # one-indexed
            mutated_pdb.mutate_residue(number=residue, to=protein_letters_1to3[consensus[residue]].upper())
            # mutated_pdb.mutate_to(names[name](n), residue, res_id=protein_letters_1to3[consensus[residue]].upper())
    mutated_pdb.write(out_path=des_dir.consensus_pdb)
    # mutated_pdb.write(consensus_pdb)
    # mutated_pdb.write(consensus_pdb, cryst1=cryst)

    # Update DesignDirectory with design information
    des_dir.info['pssm'] = pssm_file
    des_dir.info['issm'] = interface_data_file
    des_dir.info['dssm'] = dssm_file
    des_dir.info['db'] = frag_db
    des_dir.info['des_residues'] = [j for name in names for j in int_res_numbers[name]]
    # TODO add oligomer data to .info
    info_pickle = SDUtils.pickle_object(des_dir.info, 'info', out_path=des_dir.data)

    # -----------------------------------------------------------------------------------------------------------------
    # Rosetta Execution formatting
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

    flags_refine = DesignDirectory.prepare_rosetta_flags(refine_variables, PUtils.stage[1], out_path=des_dir.path)
    relax_cmd = main_cmd + \
        ['@' + os.path.join(des_dir.path, flags_refine), '-scorefile', os.path.join(des_dir.scores, PUtils.scores_file),
         '-parser:protocol', os.path.join(PUtils.rosetta_scripts, PUtils.stage[1] + '.xml')]
    refine_cmd = relax_cmd + ['-in:file:s', des_dir.refine_pdb,  '-parser:script_vars', 'switch=%s' % PUtils.stage[1]]
    consensus_cmd = relax_cmd + ['-in:file:s', des_dir.consensus_pdb, '-parser:script_vars', 'switch=%s' % PUtils.stage[5]]

    # Create executable/Run FastRelax on Clean ASU/Consensus ASU with RosettaScripts
    if script:
        SDUtils.write_shell_script(subprocess.list2cmdline(refine_cmd), name=PUtils.stage[1], out_path=des_dir.path,
                                   additional=[subprocess.list2cmdline(consensus_cmd)])
        SDUtils.write_shell_script(subprocess.list2cmdline(consensus_cmd), name=PUtils.stage[5], out_path=des_dir.path)
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
    flags_design = DesignDirectory.prepare_rosetta_flags(design_variables, PUtils.stage[2], out_path=des_dir.path)
    # TODO back out nstruct label to command distribution
    design_cmd = main_cmd + \
        ['-in:file:s', des_dir.refined_pdb, '-in:file:native', des_dir.asu, '-nstruct', str(PUtils.nstruct),
         '@' + os.path.join(des_dir.path, flags_design), '-in:file:pssm', pssm_file, '-parser:protocol',
         os.path.join(PUtils.rosetta_scripts, PUtils.stage[2] + '.xml'),
         '-scorefile', os.path.join(des_dir.scores, PUtils.scores_file)]

    # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
    # TODO if nstruct is backed out, create pdb_list for metrics distribution
    pdb_list = SDUtils.pdb_list_file(des_dir.refined_pdb, total_pdbs=PUtils.nstruct, suffix='_' + PUtils.stage[2],
                                     out_path=des_dir.designs, additional=[des_dir.consensus_design_pdb, ])
    design_variables += [('sdfA', sym_definition_files[pdb_codes[0]]), ('sdfB', sym_definition_files[pdb_codes[1]])]

    flags_metric = DesignDirectory.prepare_rosetta_flags(design_variables, PUtils.stage[3], out_path=des_dir.path)
    metric_cmd = main_cmd + \
        ['-in:file:l', pdb_list, '-in:file:native', des_dir.refined_pdb, '@' + os.path.join(des_dir.path, flags_metric),
         '-out:file:score_only', os.path.join(des_dir.scores, PUtils.scores_file),
         '-parser:protocol', os.path.join(PUtils.rosetta_scripts, PUtils.stage[3] + '.xml')]

    if mpi:
        design_cmd = CommandDistributer.run_cmds[PUtils.rosetta_extras] + design_cmd
        metric_cmd = CommandDistributer.run_cmds[PUtils.rosetta_extras] + metric_cmd
    metric_cmds = {name: metric_cmd + ['-parser:script_vars', 'chain=%s' % names[name](n)]
                   for n, name in enumerate(names)}
    # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics on Designs
    if script:
        SDUtils.write_shell_script(subprocess.list2cmdline(design_cmd), name=PUtils.stage[2], out_path=des_dir.path)
        SDUtils.write_shell_script(subprocess.list2cmdline(metric_cmds[pdb_codes[0]]), name=PUtils.stage[3],
                                   out_path=des_dir.path, additional=[subprocess.list2cmdline(metric_cmds[name])
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
        # analysis_cmd = 'python %s -d %s' % (PUtils.filter_designs, des_dir.path)
        SDUtils.write_shell_script(analysis_cmd, name=PUtils.stage[4], out_path=des_dir.path)
    else:
        if not suspend:
            pose_s = analyze_output(des_dir)
            outpath = os.path.join(des_dir.all_scores, PUtils.analysis_file)
            _header = False
            if not os.path.exists(outpath):
                _header = True
            pose_s.to_csv(outpath, mode='a', header=_header)


def gather_profile_info(pdb, des_dir, names):
    """For a given PDB, find the chain wise profile (pssm) then combine into one continuous pssm

    Args:
        pdb (PDB): PDB to generate a profile from. Sequence is taken from the ATOM record
        des_dir (DesignDirectory): Location of which to write output files in the design tree
        names (dict): The pdb names and corresponding chain of each protomer in the pdb object
    Returns:
        (str): Location of the combined pssm file written to disk
        (dict): A combined pssm with all chains concatenated in the same order as pdb sequence
    """
    pssm_files, pdb_seq, errors, pdb_seq_file, pssm_process = {}, {}, {}, {}, {}
    logger.debug('Fetching PSSM Files')

    # Extract/Format Sequence Information
    for n, name in enumerate(names):
        # if pssm_files[name] == dict():
        logger.debug('%s is chain %s in ASU' % (name, names[name](n)))
        pdb_seq[name], errors[name] = extract_aa_seq(pdb, chain=names[name](n))
        logger.debug('%s Sequence=%s' % (name, pdb_seq[name]))
        if errors[name]:
            logger.warning('Sequence generation ran into the following residue errors: %s' % ', '.join(errors[name]))
        pdb_seq_file[name] = write_fasta_file(pdb_seq[name], name + '_' + os.path.basename(des_dir.path),
                                              out_path=des_dir.sequences)
        if not pdb_seq_file[name]:
            logger.error('Unable to parse sequence. Check if PDB \'%s\' is valid.' % name)
            raise SDUtils.DesignError('Unable to parse sequence in %s' % des_dir.path)

    # Make PSSM of PDB sequence POST-SEQUENCE EXTRACTION
    for name in names:
        logger.info('Generating PSSM file for %s' % name)
        pssm_files[name], pssm_process[name] = SequenceProfile.hhblits(pdb_seq_file[name], outpath=des_dir.sequences)
        logger.debug('%s seq file: %s' % (name, pdb_seq_file[name]))

    # Wait for PSSM command to complete
    for name in names:
        pssm_process[name].communicate()

    # Extract PSSM for each protein and combine into single PSSM
    pssm_dict = {}
    for name in names:
        pssm_dict[name] = SequenceProfile.parse_hhblits_pssm(pssm_files[name])
    full_pssm = SequenceProfile.combine_pssm([pssm_dict[name] for name in pssm_dict])
    pssm_file = SequenceProfile.make_pssm_file(full_pssm, PUtils.pssm, outpath=des_dir.path)

    return pssm_file, full_pssm


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
                        default='biological_interfaces')  # PUtils.biological_fragmentDB <- this is a path, need keyword
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
        SDUtils.set_logging_to_debug()
        logger.debug('Debug mode. Verbose output')
    else:
        logger = SDUtils.start_log(name=os.path.basename(__file__), propagate=True)

    logger.info('Starting %s with options:\n%s' %
                (os.path.basename(__file__),
                 '\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    assert args.symmetry_group in PUtils.protocol, 'Symmetry group \'%s\' is not available. Please choose from %s' % \
                                                   (args.symmetry_group, ', '.join(sym for sym in PUtils.protocol))

    # Collect all designs to be processed
    all_designs, location = SDUtils.collect_designs(files=args.file, directory=args.directory)
    assert all_designs != list(), 'No %s directories found within \'%s\' input! Please ensure correct location' \
                                  % (PUtils.nano, location)
    logger.info('%d Poses found in \'%s\'' % (len(all_designs), location))
    all_design_dirs = DesignDirectory.set_up_directory_objects(all_designs)
    logger.info('All pose specific logs are located in their corresponding directories.\nEx: \'%s\'' %
                os.path.join(all_design_dirs[0].path, os.path.basename(all_design_dirs[0].path) + '.log'))

    if args.mpi:
        args.command_only = True
        extras = ' mpi %d' % CommandDistributer.mpi
        logger.info('Setting job up for submission to MPI capable computer. Pose trajectories will run in parallel, '
                    '%s at a time. This will speed up pose processing %f-fold.' %
                    (CommandDistributer.mpi - 1, PUtils.nstruct / CommandDistributer.mpi - 1))
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
            command_files[i] = SDUtils.write_commands(all_commands[i], name=stage, out_path=args.directory)
            logger.info('All \'%s\' commands were written to \'%s\'' % (stage, command_files[i]))
            logger.info('\nTo process all commands in correct order, execute:\ncd %s\n%s\n%s\n%s' %
                        (args.directory, 'python ' + __file__ + ' distribute -s refine',
                         'python ' + __file__ + ' distribute -s design',
                         'python ' + __file__ + ' distribute -s metrics'))
