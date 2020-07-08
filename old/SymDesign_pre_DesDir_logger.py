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
import logging
import shutil
import copy
# import fnmatch
import glob
from itertools import repeat
import math
import numpy as np
from Bio.SeqUtils import IUPACData
import SymDesignUtils as SDUtils
import PathUtils as PUtils
import CmdUtils as CUtils
import AnalyzeOutput as AOut
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger(__name__).setLevel(logging.DEBUG)
logger = SDUtils.start_log(__name__, level=3)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARNING)
# logger.warning('This is a warn test!')
# logger.info('This is a info test!')


def design(des_dir, frag_db, sym, script=False, suspend=False, debug=False):
    # Variable initialization
    cst_value = round(0.2 * PUtils.reference_average_residue_weight, 2)
    frag_size = 5
    cmd = ['run="go"']
    done_process = subprocess.Popen(cmd, shell=True)
    refine_process = done_process

    # Log output
    if debug:
        global logger
    else:
        # global logger
        # logger.warning('This is a warn test!')
        # logger.info('This is a info test!')
        log_name = __name__ + '.' + str(des_dir)
        logger = SDUtils.start_log(log_name, handler=2, level=2,
                                   location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))
    logger.info('Starting Design of Directory \'%s\'' % des_dir.path)

    # Set up files, Rosetta commands
    cleaned_pdb = os.path.join(des_dir.path, PUtils.clean)
    ala_mut_pdb = os.path.splitext(cleaned_pdb)[0] + '_for_refine.pdb'
    refined_pdb = os.path.join(des_dir.design_pdbs,
                               os.path.splitext(os.path.basename(ala_mut_pdb))[0] + '_' + PUtils.stage[1] + '.pdb')
    # if out:file:o works, could use, but it won't register: os.path.join(des_dir.design_pdbs, PUtils.stage[1] + '.pdb')
    if script:
        # can forgo any special program calls as production is handled by SLURM
        prefix_cmd = copy.deepcopy(CUtils.script_cmd)
    else:
        prefix_cmd = CUtils.run_cmds[PUtils.rosetta_extras] + copy.deepcopy(CUtils.script_cmd)

    # Extract information from SymDock Output
    template_pdb = SDUtils.read_pdb(os.path.join(des_dir.path, PUtils.asu))
    num_chains = len(template_pdb.chain_id_list)
    pdb_codes = str(os.path.basename(des_dir.building_blocks)).split('_')

    # TODO JOSH Get rid of same chain ID problem....
    if num_chains != 2:
        logger.warning('Incorrect chain count: %d' % num_chains)
        oligomer_file = glob.glob(os.path.join(des_dir.path, pdb_codes[0] + '_tx_*.pdb'))
        assert len(oligomer_file) == 1, '%s: More than one matching file found with %s' % \
                                        (des_dir.path, pdb_codes[0] + '_tx_*.pdb')
        first_oligomer = SDUtils.read_pdb(oligomer_file[0])  # os.path.join(des_dir.path, file))
        # find the number of ATOM records for template_pdb chain1 using the oligomeric chain as model
        for atom_idx in range(len(first_oligomer.chain(template_pdb.chain_id_list[0]))):
            template_pdb.all_atoms[atom_idx].chain = 'x'
        # for file in os.listdir(des_dir.path):
        #     if fnmatch.fnmatch(file, pdb_codes[0] + '*'):
        #         first_oligomer = SDUtils.read_pdb(os.path.join(des_dir.path, file))
        #         for atom_idx in range(len(first_oligomer.chain(template_pdb.chain_id_list[0]))):
        #             template_pdb.all_atoms[atom_idx].chain = 'x'
        #         break
        template_pdb.chain_id_list = ['x', template_pdb.chain_id_list[0]]
        num_chains = len(template_pdb.chain_id_list)
        logger.warning(
            'Chains probably have the same name! Changing IDs temporarily to %s' % template_pdb.chain_id_list)

    assert len(pdb_codes) == num_chains, 'Number of chains \'%d\' in ASU doesn\'t match number of building blocks ' \
                                         '\'%d\'' % (num_chains, len(pdb_codes))
    # if len(pdb_codes) != num_chains:
    #     logger.critical('Number of chains \'%d\' in ASU doesn\'t match number of building blocks \'%d\''
    #                         % (num_chains, len(pdb_codes)))
    #     logger.critical('Number of chains \'%d\' in ASU doesn\'t match number of building blocks \'%d\''
    #                     % (num_chains, len(pdb_codes)))
    #     raise SDUtils.DesignError('Chain error in %s' % des_dir.path)

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
        name_pdb_file = glob.glob(os.path.join(des_dir.path, name + '_tx_*.pdb'))
        assert len(name_pdb_file) == 1, '%s: More than one matching file found with %s' % \
                                        (des_dir.path, name + '_tx_*.pdb')
        oligomer[name] = SDUtils.read_pdb(name_pdb_file[0])
        oligomer[name].AddName(name)
        oligomer[name].reorder_chains()
        # Chains must be symmetrized before sdf creation TODO on input, currently raise DesignError
        sym_definition_files[name] = SDUtils.make_sdf(oligomer[name])
        # for file in os.listdir(des_dir.path):
        #     if fnmatch.fnmatch(file, name + '*'):
        #         oligomer[name] = SDUtils.read_pdb(os.path.join(des_dir.path, file))
        #         oligomer[name].AddName(name)
        #         oligomer[name].reorder_chains()
        #         # Chains must be symmetrized before sdf creation TODO on input, currently raise DesignError
        #         sym_definition_files[name] = SDUtils.make_sdf(oligomer[name])
        #         break

    logger.debug('%d Matching oligomers found in %s' % (len(oligomer), des_dir.path))

    # TODO insert mechanism to Decorate and then gather my own fragment decoration statistics
    # TODO supplement with names info and pull out by names
    with open(os.path.join(des_dir.path, PUtils.frag_file), 'r') as f:
        frag_match_info_file = f.readlines()
        residue_cluster_dict = {}
        for line in frag_match_info_file:
            if line[:12] == 'Cluster ID: ':
                cluster = line[12:].split()[0].strip().replace('i', '').replace('j', '').replace('k', '')
                if cluster not in residue_cluster_dict:
                    residue_cluster_dict[cluster] = []
            if line[:43] == 'Surface Fragment Oligomer1 Residue Number: ':
                # Always contains I fragment? #JOSH
                res_chain1 = int(line[43:].strip())
            if line[:43] == 'Surface Fragment Oligomer2 Residue Number: ':
                # Always contains J fragment and Guide Atoms? #JOSH
                res_chain2 = int(line[43:].strip())
                residue_cluster_dict[cluster].append((res_chain1, res_chain2))
                # residue_cluster_dict[cluster] = [res_chain1, res_chain2]
            if line[:15] == 'CRYST1 RECORD: ' and sym in [2, 3]:  # TODO josh providing in PDB now
                cryst = line[15:].strip()

    # Set up protocol symmetry
    symm = SDUtils.sdf_lookup(sym)
    # symmetry_variables = [symm, ['-symmetry_definition', 'CRYST1']]
    protocol = PUtils.protocol[sym]
    sym_def_file = symm
    if sym > 1:
        symmetric_cmd = prefix_cmd + ['-symmetry_definition', 'CRYST1']
        # sym_def_file = symmetry_variables[0]
        # symmetric_cmd = prefix_cmd + symmetry_variables[1]
    else:
        logger.error('Not possible to input point groups just yet...')
        sys.exit()
        # sym_def_file = symm
        # symmetric_cmd = prefix_cmd + symmetry_variables[0]
        # sym_def_file = symmetry_variables[0]
    logger.info('Symmetry Option: %s' % protocol)
    logger.info('Symmetry Information: %s' % cryst)
    logger.info('Input PDBs: %s' % ', '.join(name for name in names))
    logger.info('Pulling fragment info from clusters: %s' % ', '.join(residue_cluster_dict))
    for h, pdb_id in enumerate(names):
        logger.info('Fragments identified: %s' % 'Oligomer ' + pdb_id + ', residues: ' +
                        ', '.join(str(residue_cluster_dict[cluster][pair][h]) for cluster in residue_cluster_dict
                                  for pair in range(len(residue_cluster_dict[cluster]))))

    # Fetch IJK Cluster Dictionaries and Setup Interface Residues for Residue Number Conversion. MUST BE PRE-RENUMBER
    frag_residue_object_dict = SDUtils.make_residues_pdb_object(template_pdb, residue_cluster_dict)
    logger.debug('Fragment Residue Object Dict: %s' % str(frag_residue_object_dict))
    # TODO Make chain number independent. Low priority
    int_residues = SDUtils.find_interface_residues(oligomer[pdb_codes[0]], oligomer[pdb_codes[1]])
    # int_residues1, int_residues2
    # int_residue_objects = SDUtils.find_interface_residues(oligomer[pdb_codes[0]], oligomer[pdb_codes[1]])
    # int_res_atoms_pair = [None for j in range(math.factorial(len(template_pdb.chain_id_list) - 1))]
    # int_res_atoms1 = [None for j in range(math.factorial(len(template_pdb.chain_id_list) - 1))]
    # int_res_atoms2 = copy.deepcopy(int_res_atoms1)
    # i = 0
    # for j in range(len(oligomer) - 1):
    #     for k in range(j + 1, len(oligomer)):
    #         int_res_atoms_pair[i] = SDUtils.find_interface_residues(oligomer[j], oligomer[k])
    #         i += 1

    # Get residue numbers as Residue objects to map across chain renumbering
    int_residue_objects = {}
    for k, name in enumerate(names):
        int_residue_objects[name] = []
        for residue in int_residues[k]:
            try:
                int_residue_objects[name].append(template_pdb.get_residue(names[name](k), residue))
            except IndexError:
                raise SDUtils.DesignError('%s: Oligomeric and ASU chains do not match. Interface likely involves '
                                          'missing density at oligomer \'%s\', chain \'%s\', residue \'%d\'. Resolve '
                                          'this error and make sure that all input oligomers are symmetrized for '
                                          'optimal script performance.'
                                          % (des_dir.path, name, names[name](k), residue))

    # Renumber PDB to Rosetta Numbering
    logger.info('Converting to standard Rosetta numbering. '
                    '1st residue of chain A is 1, 1st residue of chain B is last residue in chain A + 1, etc')
    template_pdb.reorder_chains()
    template_pdb.renumber_residues()
    jump = template_pdb.getTermCAAtom('C', template_pdb.chain_id_list[0]).residue_number
    template_residues = template_pdb.get_all_residues()
    logger.info('Last residue of first oligomer %s, chain %s is %d' %
                    (list(names.keys())[0], names[list(names.keys())[0]](0), jump))
    logger.info('Total number of residues is %d' % len(template_residues))
    template_pdb.write(cleaned_pdb, cryst1=cryst)

    # Mutate all design positions to Ala
    mutated_pdb = copy.deepcopy(template_pdb)
    logger.debug('Cleaned PDB: \'%s\'' % cleaned_pdb)

    # Set Up Interface Residues after renumber, remove Side Chain Atoms to Ala NECESSARY for all chains to ASU chain map
    int_res_numbers = {}  # [[] for j in range(num_chains)]
    for c, name in enumerate(int_residue_objects):  # for c, chain in enumerate(int_res_atoms):
        int_res_numbers[name] = []
        for residue in int_residue_objects[name]:  # for res_atoms in int_res_atoms[chain]:
            int_res_numbers[name].append(residue.ca.residue_number)  # must call .ca.residue_number as .number is static
            mutated_pdb.mutate_to_ala(names[name](c), residue.ca.residue_number)

    # # Set Up Interface Residues after renumber, remove Side Chain Atoms to Ala
    # int_res_numbers = {}
    # for c, name in enumerate(int_residue_objects):
    #     int_res_numbers[name] = []
    #     for residue in int_residue_objects[name]:
    #         int_res_numbers[name].append(residue.residue_number)
    #         template_pdb.mutate_to_ala(names[name](c), residue)

    logger.info('Interface Residues: %s' % ', '.join(str(n) for name in int_res_numbers
                                                         for n in int_res_numbers[name]))
    mutated_pdb.write(ala_mut_pdb, cryst1=cryst)
    logger.debug('Cleaned PDB for Refine: \'%s\'' % ala_mut_pdb)

    # Get ASU distance parameters TODO FOR POINT GROUPS
    asu_oligomer_com_dist = []
    for d, name in enumerate(names):
        asu_oligomer_com_dist.append(np.linalg.norm(np.array(template_pdb.center_of_mass())
                                                    - np.array(oligomer[name].center_of_mass())))
    max_com_dist = 0
    for com_dist in asu_oligomer_com_dist:
        if com_dist > max_com_dist:
            max_com_dist = com_dist
    dist = round(math.sqrt(math.ceil(max_com_dist)), 0)  # OPTIMIZE reasonable amount for first tests...
    logger.info('Expanding ASU by %f Angstroms' % dist)

    # REFINE: Prepare Command, and Flags File
    refine_variables = [('pdb_reference', cleaned_pdb), ('scripts', PUtils.rosetta_scripts),
                        ('sym_score_patch', PUtils.sym_weights), ('symmetry', protocol), ('sdf', sym_def_file),
                        ('dist', dist), ('cst_value', cst_value), ('cst_value_sym', (cst_value / 2))]
    for k, name in enumerate(int_res_numbers, 1):
        refine_variables.append(('interface' + str(k), ','.join(str(j) for j in int_res_numbers[name])))

    flagrefine = SDUtils.prepare_rosetta_flags(refine_variables, PUtils.stage[1], outpath=des_dir.path)
    input_refine_cmds = ['-in:file:s', ala_mut_pdb, '@' + os.path.join(des_dir.path, flagrefine), '-parser:protocol',
                         os.path.join(PUtils.rosetta_scripts, PUtils.stage[1] + '.xml'),
                         '-scorefile', os.path.join(des_dir.scores, PUtils.scores_file)]
    refine_cmd = symmetric_cmd + input_refine_cmds

    # Run/Create RosettaScripts executable on Cleaned ASU
    if script:
        SDUtils.write_shell_script(subprocess.list2cmdline(refine_cmd), stage=1, outpath=des_dir.path)
    else:
        if not suspend:
            logger.info('Refine Command: %s' % subprocess.list2cmdline(refine_cmd))
            refine_process = subprocess.Popen(refine_cmd)

    # Check to see if other poses have collected design sequence info and grab PSSM
    temp_file = os.path.join(des_dir.building_blocks, PUtils.temp)
    rerun = False
    if PUtils.clean not in os.listdir(des_dir.building_blocks):
        shutil.copy(cleaned_pdb, des_dir.building_blocks)
        with open(temp_file, 'w') as f:
            f.write('Still fetching data. Process will resume once data is gathered\n')

        pssm_files, pdb_seq, errors, pdb_seq_file, pssm_process = {}, {}, {}, {}, {}
        logger.debug('Fetching PSSM Files')

        # Check if other design combinations have already collected sequence info about design candidates
        for name in names:
            for seq_file in glob.iglob(os.path.join(des_dir.sequences, name + '.*')):  # matching files, .ext ambivalent:
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
                # pdb_seq[name], errors[name] = SDUtils.extract_aa_seq(oligomer[name], chain=names[name](n))
                logger.debug('%s Sequence=%s' % (name, pdb_seq[name]))
                if errors[name]:
                    logger.warning('Sequence generation ran into the following residue errors: %s'
                                       % ', '.join(errors[name]))
                pdb_seq_file[name] = SDUtils.write_fasta_file(pdb_seq[name], name, outpath=des_dir.sequences)
                if not pdb_seq_file[name]:
                    logger.error('Unable to parse sequence. Check if PDB \'%s\' is valid.' % name)
                    # logger.critical('Unable to parse sequence. Check if PDB \'%s\' is valid.' % name)
                    raise SDUtils.DesignError('Unable to parse sequence in %s' % des_dir.path)
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
        logger.info('Waiting for profile generation...')
        while True:
            if os.path.exists(temp_file):
                time.sleep(20)
                continue
            break

        pssm_file = os.path.join(des_dir.building_blocks, PUtils.msa_pssm)
        full_pssm = SDUtils.parse_pssm(pssm_file)

    # Check length for equality before proceeding
    if len(template_residues) != len(full_pssm):
        logger.warning('%s: Profile and Pose sequences are different lengths!\nProfile=%d, Pose=%d. Generating new '
                       'profile' % (des_dir.path, len(template_residues), len(full_pssm)))
        # logging, logger
        rerun = True

    if not rerun:
        # Check sequence from Pose and PSSM to compare identity before proceeding
        pssm_res, pose_res = {}, {}
        for res in range(len(template_residues)):
            pssm_res[res] = full_pssm[res + 1]['type']
            pose_res[res] = IUPACData.protein_letters_3to1[template_residues[res].type.title()]
            if pssm_res[res] != pose_res[res]:
                logger.warning('%s: Profile and Pose sequences are different!\nResidue %d: Profile=%s, Pose=%s. '
                               'Generating new profile' % (des_dir.path, res + SDUtils.index_offset, pssm_res[res],
                                                            pose_res[res]))
                # logging, logger.warning()
                # raise SDUtils.DesignError('%s: Pose length is the same, but residues different!' % des_dir.path)
                rerun = True
                break
    # raise SDUtils.DesignError('%s: Messed up pose')
    if rerun:
        pssm_file, full_pssm = SDUtils.gather_profile_info(template_pdb, des_dir, names, logger)
    logger.debug('Position Specific Scoring Matrix: %s' % str(full_pssm))

    # Parse Fragment Clusters into usable Dictionaries and Flatten for Sequence Design
    fragment_range = SDUtils.parameterize_frag_length(frag_size)
    full_design_dict = SDUtils.populate_design_dict(len(full_pssm), [j for j in range(*fragment_range)])
    residue_cluster_map = SDUtils.convert_to_residue_cluster_map(frag_residue_object_dict, fragment_range)

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

    cluster_dicts = SDUtils.get_cluster_dicts(info_db=frag_db, id_list=[j for j in residue_cluster_dict])
    full_cluster_dict = SDUtils.deconvolve_clusters(cluster_dicts, full_design_dict, residue_cluster_map)
    final_issm = SDUtils.flatten_for_issm(full_cluster_dict)
    logger.debug('Fragment Specific Scoring Matrix: %s' % str(final_issm))

    # Make DSSM, a PSSM with fragment and evolutionary profile combined
    dssm, residue_alphas = SDUtils.combine_ssm(full_pssm, final_issm, residue_cluster_map, db=frag_db, boltzmann=True)
    logger.debug('Design Specific Scoring Matrix: %s' % dssm)
    for residue, data in residue_alphas:
        logger.info('Residue %d Fragment data: %s' % (residue, data))
    dssm_file = SDUtils.make_pssm_file(dssm, PUtils.dssm, outpath=des_dir.path)

    # Wait for Rosetta Refine command to complete
    refine_process.communicate()

    # DESIGN: Prepare Command, and Flags file
    design_variables = copy.deepcopy(refine_variables)
    design_variables.append(('pssm_file', dssm_file))
    flagdesign = SDUtils.prepare_rosetta_flags(design_variables, PUtils.stage[2], outpath=des_dir.path)
    input_design_cmds = ['-in:file:s', refined_pdb, '-in:file:native', cleaned_pdb, '-nstruct', str(CUtils.nstruct),
                         '@' + os.path.join(des_dir.path, flagdesign), '-in:file:pssm', pssm_file, '-parser:protocol',
                         os.path.join(PUtils.rosetta_scripts, PUtils.stage[2] + '.xml'),
                         '-scorefile', os.path.join(des_dir.scores, PUtils.scores_file)]
    design_cmd = symmetric_cmd + input_design_cmds

    # METRICS: Can remove if SimpleMetrics gets pose metric caching and restoration
    pdb_list_file = SDUtils.pdb_list_file(refined_pdb, total_pdbs=CUtils.nstruct, suffix='_' + PUtils.stage[2],
                                          loc=des_dir.design_pdbs)
    flagmetric = SDUtils.prepare_rosetta_flags(design_variables, PUtils.stage[3], outpath=des_dir.path)
    input_metric_cmds = ['-in:file:l', pdb_list_file, '-in:file:native', refined_pdb,
                         '@' + os.path.join(des_dir.path, flagmetric),
                         '-out:file:score_only', os.path.join(des_dir.scores, PUtils.scores_file),
                         '-parser:protocol', os.path.join(PUtils.rosetta_scripts, PUtils.stage[3] + '.xml')]
    metrics_cmd = symmetric_cmd + input_metric_cmds

    # Run/Create RosettaScripts executable on Refined ASU
    if script:
        SDUtils.write_shell_script(subprocess.list2cmdline(design_cmd), stage=2, outpath=des_dir.path,
                                   additional=subprocess.list2cmdline(metrics_cmd))
        SDUtils.write_shell_script(subprocess.list2cmdline(metrics_cmd), stage=3, outpath=des_dir.path)
    else:
        if not suspend:
            logger.info('Design Command: %s' % subprocess.list2cmdline(design_cmd))
            design_process = subprocess.Popen(design_cmd)
            # Wait for Rosetta Design command to complete
            design_process.communicate()
            logger.info('Metrics Command: %s' % subprocess.list2cmdline(metrics_cmd))
            metrics_process = subprocess.Popen(metrics_cmd)
            metrics_process.communicate()

    # Filter each output from the Design process based on score, Analyze Sequence Variation
    if script:
        filter_command = 'python %s -d %s' % (PUtils.filter_designs, des_dir.path)
        SDUtils.write_shell_script(filter_command, stage=4, outpath=des_dir.path)
    else:
        if not suspend:
            AOut.analyze_output(des_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=PUtils.program_name +
                                                 '\nGathers output from SymDock and formats it for input into Rosetta '
                                                 'for interface remodelling. Remodelling is constrained by '
                                                 'fragment profiles extracted from the PDB and evolutionary profiles of'
                                                 'homologous sequences')
    parser.add_argument('-d', '--directory', type=str, help='Directory where SymDock output is located. Default = CWD',
                        default=os.getcwd())
    # TODO, function for selecting the appropriate interface library given a viable input
    parser.add_argument('-f', '--fragment_database', type=str, help='Database to match fragments for interface specific'
                                                                    ' scoring matrices. Default = biological_interfaces'
                        , default=PUtils.biological_fragmentDB)
    parser.add_argument('symmetry_group', type=int, help='What type of symmetry group does your design belong too? '
                                                         'One of 0-Point Group, 2-Plane Group, or 3-Space Group')
    parser.add_argument('-c', '--command_only', action='store_true', help='Should commands be written but not executed?'
                                                                          ' Default = False')
    parser.add_argument('-m', '--multi_processing', action='store_true', help='Should job be run with multiprocessing? '
                                                                              'Default = False')
    # parser.add_argument('-p', '--prioritize_frags', action='store_true', help='Prioritize fragments in Rosetta? '
    #                                                                           'Default = False')
    parser.add_argument('-x', '--suspend', action='store_true', help='Should Rosetta design trajectory be suspended? '
                                                                     'Default = False')
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out? Default = False')
    args = parser.parse_args()

    # Start logging output
    if args.debug:
        logger = SDUtils.start_log(__name__ + PUtils.program_name, level=1)
        logger.debug('Debug mode. Verbose output')
    else:
        # logger = SDUtils.start_log(PUtils.program_name, level=2)
        logger = SDUtils.start_log(__name__ + PUtils.program_name, level=2)
        # warn_logger = SDUtils.start_log(__name__, level=3)
    logger.info('Starting design with options:\n%s' %
                ('\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))
    if args.symmetry_group not in PUtils.protocol:
        logger.critical('Symmetry group \'%s\' is not available. Please choose from %s' %
                        (args.symmetry_group, ', '.join(sym for sym in PUtils.protocol)))
        sys.exit(1)
    if args.command_only:
        args.suspend = True
        logger.info('Writing modelling commands out to file only, no modelling will occur until commands are executed.')

    # Collect all designs to be processed
    all_designs = SDUtils.get_design_directories(args.directory)
    if all_designs == list():
        logger.critical('No SymDock directories found in \'%s\'. Please ensure correct location' % args.directory)
        sys.exit(1)

    logger.info('%d total Poses found in \'%s\'.' % (len(all_designs), args.directory))
    # Start Pose processing and preparation for Rosetta
    logger.info('All pose specific logs are located in their corresponding directories. For example \'%s\'' %
                os.path.join(all_designs[0].path, os.path.basename(all_designs[0].path) + '.log'))
    if args.multi_processing:
        # calculate the number of threads to use depending on computer resources
        mp_threads = SDUtils.calculate_mp_threads(args.suspend)
        logger.info('Beginning Pose specific multiprocessing with %s multiprocessing threads' % str(mp_threads))
        zipped_args = zip(all_designs, repeat(args.fragment_database), repeat(args.symmetry_group),
                          repeat(args.command_only), repeat(args.suspend),
                          repeat(args.debug))  # repeat(args.prioritize_frags)
        results, exceptions = SDUtils.mp_starmap(design, zipped_args, mp_threads)
        if exceptions:
            logger.warning('The following exceptions were thrown. Design for these directories is inaccurate.')
            for exception in exceptions:
                logger.warning(exception)
    else:
        logger.info('Beginning Pose specific processing. If single thread is taking a while, use -m during submission')
        for des_directory in all_designs:
            design(des_directory, args.fragment_database, args.symmetry_group, script=args.command_only,
                   suspend=args.suspend, debug=args.debug)  # fsp=args.prioritize_frags,

    if args.command_only:
        all_commands = [[] for s in PUtils.stage]
        command_files = [[] for s in PUtils.stage]
        for des_directory in all_designs:
            for i, stage in enumerate(PUtils.stage):
                all_commands[i].append(os.path.join(des_directory.path, PUtils.stage[stage] + '.sh'))
        for i, stage in enumerate(PUtils.stage):
            command_files[i] = SDUtils.write_commands(all_commands[i], name=PUtils.stage[stage], loc=args.directory)
            logger.info('List of all \'%s\' commands were written to \'%s\'' %
                        (PUtils.stage[stage].title(), command_files[i]))
        logger.info('To process all commands in correct order, run \'%s\' in %s.\n' %
                    (PUtils.process_commands, args.directory))
