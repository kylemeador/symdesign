'''
Before full run need to alleviate the following tags and incompleteness
-TODO - Make this resource
-OPTIMIZE - make the best protocol
-JOSH - input style from Josh's output

Design Decisions:
-We need to have the structures relaxed before input into the design program. Should relaxation be done on individual
protein building blocks? This again gets into the SDF issues. But for Rosetta scoring, we need a baseline to be relaxed.
        -currently relaxed in unit cell
-Should the sidechain info for each of the interface positions be stripped or made to ALA upon rosetta start up?
        -currently YES

Code Formatting
-PSSM format is dictating the output file format as it works best with Rosetta pose number
'''
import os
import sys
import subprocess
import time
import argparse
import logging
import shutil
import copy
import fnmatch
from itertools import repeat
import math
import numpy as np
import SymDesignUtils as SDUtils
import PathUtils as PUtils
import CmdUtils as CUtils
import AnalyzeOutput as AOut


def design(des_dir, frag_db, sym, script=False, suspend=False, debug=False):  # fsp=False,
    # Variable initialization
    cst_value = 0.2 * PUtils.reference_average_residue_weight
    frag_size = 5
    cmd = ['run="go"']
    done_process = subprocess.Popen(cmd, shell=True)
    refine_process = done_process

    # Log output
    if debug:
        des_logger = logger
    else:
        log_name = __name__ + '.' + str(des_dir)
        des_logger = SDUtils.start_log(log_name, stream=2, level=2,
                                       location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))
    des_logger.info('Starting Design of Directory \'%s\'' % des_dir.path)

    # Set up files
    cleaned_pdb = os.path.join(des_dir.path, PUtils.clean)
    ala_mut_pdb = os.path.splitext(cleaned_pdb)[0] + '_for_refine.pdb'
    refined_pdb = os.path.join(des_dir.design_pdbs, PUtils.stage[1] + '.pdb')
    # os.path.splitext(os.path.basename(ala_mut_pdb))[0] + '_' + PUtils.stage[1] + '.pdb')
    symm = SDUtils.sdf_lookup(sym)
    symmetry_variables = [['sdf', symm], ['-symmetry_definition', 'CRYST1']]

    # Set up Rosetta commands
    if script:
        # can forgo any special program calls as production is handled by SLURM
        prefix_cmd = copy.deepcopy(CUtils.script_cmd)
    else:
        prefix_cmd = CUtils.run_cmds[PUtils.rosetta_extras] + copy.deepcopy(CUtils.script_cmd)

    # Set up protocol symmetry
    if sym in PUtils.protocol:
        protocol = PUtils.protocol[sym]
        if sym > 1:
            symmetric_cmd = prefix_cmd + symmetry_variables[1]
            sym_def_file = SDUtils.sdf_lookup('dummy')
        else:
            des_logger.error('Not possible to input point groups just yet...')
            sys.exit()
            # symmetric_cmd = prefix_cmd + symmetry_variables[0]
            sym_def_file = SDUtils.sdf_lookup(sym)
    des_logger.info('Symmetry Option: %s' % protocol)

    # Extract information from SymDock Output
    template_pdb = SDUtils.read_pdb(os.path.join(des_dir.path, PUtils.asu))
    num_chains = len(template_pdb.chain_id_list)
    pdb_codes = os.path.basename(des_dir.building_blocks).split('_')
    names = {}
    if len(pdb_codes) != num_chains:
        des_logger.critical('Number of chains \'%d\' in ASU doesn\'t match number of building blocks \'%d\''
                            % (num_chains, len(pdb_codes)))
        sys.exit()
    # TODO JOSH Get rid of same chain ID problem....
    if num_chains != 2:
        des_logger.warning('Incorrect chain count: %d' % num_chains)
        for file in os.listdir(des_dir.path):
            if fnmatch.fnmatch(file, names[str(pdb_codes[0])] + '*'):
                first_oligomer = SDUtils.read_pdb(os.path.join(des_dir.path, file))
                break
        for atom_idx in range(len(first_oligomer.chain(template_pdb.chain_id_list[0]))):
            template_pdb.all_atoms[atom_idx].chain = 'x'
        template_pdb.chain_id_list = ['x', template_pdb.chain_id_list[0]]
        des_logger.warning(
            'Chains probably have the same name! Changing IDs temporarily to %s' % template_pdb.chain_id_list)

    # Set up names object containing pdb id and chain info
    for c, chain in enumerate(template_pdb.chain_id_list):
        names[str(pdb_codes[c])] = template_pdb.get_chain_index
    des_logger.debug('Chain, Name Pairs: %s' % ', '.join(key + ', ' + str(value(c)) for c, (key, value) in
                                                         enumerate(names.items())))
    # Fetch PDB object of each chain individually from the design directory
    oligomer = {}
    for name in names:
        for file in os.listdir(des_dir.path):
            if fnmatch.fnmatch(file, name + '*'):
                oligomer[name] = SDUtils.read_pdb(os.path.join(des_dir.path, file))
                break
    # oligomer = [SDUtils.read_pdb(os.path.join(des_dir.path, file)) for name in names
    #             for file in os.listdir(des_dir.path) if fnmatch.fnmatch(file, name + '*')]
    des_logger.debug('%d Matching oligomers found in %s' % (len(oligomer), des_dir.path))

    # TODO insert mechanism to Decorate and then gather my own fragment decoration statistics

    # TODO supplement with names info and pull out by names
    with open(os.path.join(des_dir.path, PUtils.frag_file), 'r') as f:
        frag_match_info_file = f.readlines()
        residue_cluster_dict = {}
        for line in frag_match_info_file:
            if line[:12] == 'Cluster ID: ':
                cluster = line[12:].split()[0].strip().replace('i', '').replace('j', '').replace('k', '')
            if line[:43] == 'Surface Fragment Oligomer1 Residue Number: ':
                # Always contains I fragment? #JOSH
                res_chain1 = int(line[43:].strip())
            if line[:43] == 'Surface Fragment Oligomer2 Residue Number: ':
                # Always contains J fragment and Guide Atoms? #JOSH
                res_chain2 = int(line[43:].strip())
                # handles cluster duplications naturally, unless same cluster is at two separate locations... TODO high priority
                residue_cluster_dict[cluster] = [res_chain1, res_chain2]
            if line[:15] == 'CRYST1 RECORD: ' and sym in [2, 3]:
                cryst = line[15:].strip()

    des_logger.info('Symmetry Information: %s' % cryst)
    des_logger.info('Input PDBs: %s' % ', '.join(name for name in names))
    des_logger.info('Pulling fragment info from clusters: %s' % ', '.join(residue_cluster_dict))
    for h, key in enumerate(names):
        des_logger.info('Fragments identified: %s' % 'Oligomer ' + key + ', residues: ' + ', '.join([str(residue_cluster_dict[cluster][h]) for cluster in residue_cluster_dict]))
    jump = template_pdb.getTermCAAtom('C', template_pdb.chain_id_list[0]).residue_number
    des_logger.info('Last residue of first oligomer %s, chain %s is %d' %
                    (list(names.keys())[0], names[list(names.keys())[0]](0), jump))

    # Fetch IJK Cluster Dictionaries and Setup Interface Residues for Residue Number Conversion. MUST BE PRE-RENUMBER
    frag_residue_list = SDUtils.make_residues_pdb_object(template_pdb, residue_cluster_dict)
    # TODO Make chain number independent low priority
    int_residues = SDUtils.find_interface_residues(oligomer[pdb_codes[0]], oligomer[pdb_codes[1]])  # int_residues1, int_residues2
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
            int_residue_objects[name].append(template_pdb.get_residue(names[name](k), residue))

    # Renumber PDB to Rosetta Numbering
    des_logger.info('Converting to standard Rosetta numbering. 1st residue of chain A is 1, 1st residue of chain B is last residue in chain A + 1, etc')
    template_pdb.reorder_chains()
    template_pdb.renumber_residues()
    template_pdb.write(cleaned_pdb, cryst1=cryst)
    des_logger.debug('Cleaned PDB: \'%s\'' % cleaned_pdb)

    # Set Up Interface Residues after renumber, remove Side Chain Atoms to Ala
    int_res_numbers = {}  # [[] for j in range(num_chains)]
    for c, name in enumerate(int_residue_objects):  # for c, chain in enumerate(int_res_atoms):
        int_res_numbers[name] = []
        for residue in int_residue_objects[name]:  # for res_atoms in int_res_atoms[chain]:
            int_res_numbers[name].append(residue.ca.residue_number)
            template_pdb.mutate_to_ala(residue)

    des_logger.info('Interface Residues: %s' % ', '.join(str(n) for name in int_res_numbers for n in int_res_numbers[name]))
    template_pdb.write(ala_mut_pdb, cryst1=cryst)
    des_logger.debug('Cleaned PDB for Refine: \'%s\'' % ala_mut_pdb)

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
    des_logger.info('Expanding ASU by %f Angstroms' % dist)

    # REFINE: Prepare Command, and Flags File
    refine_variables = [('pdb_reference', cleaned_pdb), ('cst_value', cst_value), ('scripts', PUtils.rosetta_scripts),
                        ('sym_score_patch', PUtils.sym_weights), ('symmetry', protocol), ('sdf', sym_def_file),
                        ('dist', dist)]
    for k, name in enumerate(int_res_numbers, 1):
        refine_variables.append(('interface' + str(k), ','.join(str(j) for j in int_res_numbers[name])))

    flagrefine = SDUtils.prepare_rosetta_flags(refine_variables, PUtils.stage[1], outpath=des_dir.path)
    input_refine_cmds = ['-in:file:s', ala_mut_pdb, '@' + os.path.join(des_dir.path, flagrefine), '-parser:protocol',
                         PUtils.stage[1] + '.xml']
    refine_cmd = symmetric_cmd + input_refine_cmds

    # Run/Create RosettaScripts executable on Cleaned ASU
    if script:
        SDUtils.write_shell_script(subprocess.list2cmdline(refine_cmd), stage=1, outpath=des_dir.path)
    else:
        if not suspend:
            # des_logger.info('Refine Command: %s' % subprocess.list2cmdline(refine_cmd))
            # refine_process = subprocess.Popen(refine_cmd)
            place_holder = None

    # Check to see if other poses have already collected sequence info about design and grab PSSM
    pssm_files = {}
    if PUtils.clean not in os.listdir(des_dir.building_blocks):
        shutil.copy(cleaned_pdb, des_dir.building_blocks)
        with open(os.path.join(des_dir.building_blocks, PUtils.temp), 'w') as f:
            f.write('Still fetching data. Process will resume once data is gathered\n')

        pdb_seq, errors, pdb_seq_file, pssm_process = {}, {}, {}, {}
        des_logger.debug('Fetching PSSM Files')

        # Check if other combinations have already collected sequence info about both design candidates
        for name in names:
            for file in os.listdir(des_dir.sequences):
                if fnmatch.fnmatch(file, name + '.hmm'):
                    pssm_files[name] = os.path.join(des_dir.sequences, file)
                    des_logger.debug('%s PSSM Files=%s' % (name, pssm_files[name]))
            if name not in pssm_files:
                pssm_files[name] = {}
                des_logger.debug('%s PSSM File not found' % name)

        # Extract/Format Sequence Information
        for n, name in enumerate(names):
            des_logger.debug('%s is chain %s in ASU' % (name, names[name](n)))
            pdb_seq[name], errors[name] = SDUtils.extract_aa_seq(oligomer[name], chain=names[name](n))
            des_logger.debug('%s Sequence=%s' % (name, pdb_seq[name]))
            if errors[name]:
                des_logger.error('Sequence generation ran into the following residue errors: %s'
                                 % ', '.join(errors[chain]))

        # Make PSSM of PDB sequence POST-SEQUENCE EXTRACTION
        for name in names:
            if pssm_files[name] == dict():
                pdb_seq_file[name] = SDUtils.write_fasta_file(pdb_seq[name], name, outpath=des_dir.sequences)
                if not pdb_seq_file[name]:
                    des_logger.critical('Unable to parse sequence. Check if PDB \'%s\' is valid.' % name)
                    sys.exit()
                des_logger.info('Generating PSSM file for %s' % name)
                pssm_files[name], pssm_process[name] = SDUtils.hhblits(pdb_seq_file[name], outpath=des_dir.sequences)
            else:
                des_logger.info('Copying PSSM file for %s' % name)
                pssm_process[chain] = done_process
            des_logger.debug('%s seq file: %s' % (name, pdb_seq_file[name]))

        # Wait for PSSM command to complete
        for name in names:
            pssm_process[name].communicate()

        # Extract PSSM for each protein and combine into single PSSM
        pssm_dict = {}
        for name in names:
            pssm_dict[name] = SDUtils.parse_hhblits_pssm(pssm_files[name])
        full_pssm = SDUtils.combine_pssm([pssm_dict[name] for name in pssm_dict])
        pssm_file = SDUtils.make_pssm_file(full_pssm, PUtils.msa_pssm, outpath=des_dir.building_blocks)

        os.remove(os.path.join(des_dir.building_blocks, PUtils.temp))
    else:
        while True:
            time.sleep(2)
            if not os.path.exists(os.path.join(des_dir.building_blocks, PUtils.temp)):
                break
            des_logger.info('Waiting for profile generation...')
            time.sleep(30)
        # for name in names:
        #     pssm_files[name] = os.path.join(des_dir.sequences, name + '.hmm')
        pssm_file = os.path.join(des_dir.building_blocks, PUtils.clean_pssm)
        full_pssm = SDUtils.parse_pssm(pssm_file, full_pssm)

    des_logger.debug('Position Specific Scoring Matrix: %s' % str(full_pssm))

    # Parse Fragment Clusters into usable Dictionaries and Flatten for Sequence Design
    fragment_range = SDUtils.parameterize_frag_length(frag_size)
    full_design_dict = SDUtils.populate_design_dict(len(full_pssm), [j for j in range(*fragment_range)])
    cluster_dicts = SDUtils.get_cluster_dicts(info_db=frag_db, id_list=[j for j in residue_cluster_dict])
    residue_cluster_map = SDUtils.convert_to_residue_cluster_map(frag_residue_list, fragment_range)
    sys.exit()
    full_cluster_dict = SDUtils.deconvolve_clusters(cluster_dicts, full_design_dict, residue_cluster_map)
    final_issm = SDUtils.flatten_for_issm(full_cluster_dict)
    des_logger.debug('Residue Cluster Map: %s' % str(residue_cluster_map))
    des_logger.debug('Fragment Specific Scoring Matrix: %s' % str(final_issm))

    # Make DSSM, a PSSM with fragment and evolutionary profile combined
    dssm, residue_alphas = SDUtils.combine_ssm(full_pssm, final_issm, residue_cluster_map, db=frag_db, boltzmann=True)
    des_logger.debug('Design Specific Scoring Matrix: %s' % dssm)
    for residue, data in residue_alphas:
        des_logger.info('Residue %d Fragment data: %s' % (residue, data))
    dssm_file = SDUtils.make_pssm_file(dssm, PUtils.dssm, outpath=des_dir.path)

    # Wait for Rosetta Refine command to complete
    refine_process.communicate()

    # DESIGN: Prepare Command, and Flags file
    design_variables = refine_variables.append(('msa_pssm', pssm_file))
    # design_variables = refine_variables.append(('msa_pssm', dssm_file))
    # design_variables = [('pdb_reference', cleaned_pdb), ('pssm_file', dssm_file), ('scripts', PUtils.rosetta_scripts),
    #                     ('sym_score_patch', PUtils.sym_weights), ('cst_value', cst_value),
    #                     ('symmetry', protocol), ('sdf', sym_def_file), ('dist', dist),
    #                     ('interface1', ','.join(str(j) for j in int_res_numbers[0])),
    #                     ('interface2', ','.join(str(j) for j in int_res_numbers[1]))]
    # ('interface', ','.join(str(k) for j in int_res_numbers for k in j)), ('interface_separator', jump),
    flagdesign = SDUtils.prepare_rosetta_flags(design_variables, PUtils.stage[2], outpath=des_dir.path)
    input_design_cmds = ['-in:file:s', refined_pdb, '-in:file:native', cleaned_pdb, '-nstruct', str(CUtils.nstruct),
                         '@' + os.path.join(des_dir.path, flagdesign), '-in:file:pssm', dssm_file, '-parser:protocol',
                         protocol + PUtils.stage[2] + '.xml']  # + protocol_extras
    design_cmd = symmetric_cmd + input_design_cmds
    # DESIGN2. Can remove when SimpleMetrics gets pose metric cacheing and restoration
    designed_pdb_list_file = SDUtils.pdb_list_file(refined_pdb, total_pdbs=CUtils.nstruct, suffix='_' + PUtils.stage[2],
                                                   loc=des_dir.design_pdbs)
    flagdesign2 = SDUtils.prepare_rosetta_flags(design_variables, PUtils.stage[3], outpath=des_dir.path)
    input_design2_cmds = ['-in:file:l', designed_pdb_list_file, '-in:file:native', cleaned_pdb,
                          '@' + os.path.join(des_dir.path, flagdesign2), '-parser:protocol',
                          protocol + PUtils.stage[3] + '.xml']  # + protocol_extras
    design2_cmd = symmetric_cmd + input_design2_cmds

    # Run/Create RosettaScripts executable on Refined ASU
    if script:
        SDUtils.write_shell_script(subprocess.list2cmdline(design_cmd), stage=2, outpath=des_dir.path,
                                   additional=subprocess.list2cmdline(design2_cmd))
        # SDUtils.write_shell_script(subprocess.list2cmdline(design2_cmd), stage=3, outpath=des_dir.path)
    else:
        if not suspend:
            des_logger.info('Design Command: %s' % subprocess.list2cmdline(design_cmd))
            design_process = subprocess.Popen(design_cmd)
            # Wait for Rosetta Design command to complete
            design_process.communicate()
            des_logger.info('Design2 Command: %s' % subprocess.list2cmdline(design2_cmd))
            design_process2 = subprocess.Popen(design2_cmd)
            design_process2.communicate()

    # Filter each output from the Design process based on score, Analyze Sequence Variation
    if script:
        filter_command = 'python %s -d %s' % (PUtils.filter_designs, des_dir.path)
        SDUtils.write_shell_script(filter_command, stage=4, outpath=des_dir.path)
    else:
        if not suspend:
            AOut.analyze_output(des_dir)


logging.getLogger().setLevel(logging.DEBUG)


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
        logger = SDUtils.start_log(PUtils.program_name, level=1)
        logger.warning('Debug mode. Verbose output')
    else:
        logger = SDUtils.start_log(PUtils.program_name, level=2)
    logger.info('Starting design with options:\n%s' %
                ('\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))
    if args.command_only:
        args.suspend = True
        logger.info('Writing out commands only, no modelling will occur.')

    # Collect all designs to be processed
    all_design_directories = []
    for design_root, design_dirs, design_files in os.walk(args.directory):
        if os.path.basename(design_root).startswith('tx_'):
            all_design_directories.append(design_root)
        for directory in design_dirs:
            if directory.startswith('tx_'):
                all_design_directories.append(os.path.join(design_root, directory))
    all_design_directories = set(all_design_directories)
    all_designs = []
    for directory in all_design_directories:
        all_designs.append(SDUtils.DesignDirectory(directory))
    if all_designs == list():
        logger.critical('No SymDock directories found in \'%s\'. Please ensure correct location' % args.directory)
        sys.exit()

    # Start Pose processing and preparation for Rosetta
    logger.info('All pose specific logs are located in their corresponding directories. For example \'%s\'' %
                os.path.join(all_designs[0].path, os.path.basename(all_designs[0].path) + '.log'))
    if args.multi_processing:
        logger.info('Beginning multiprocessing')
        mp_threads = SDUtils.calculate_mp_threads(args.suspend)
        logger.info('Number of multiprocessing threads: %s' % str(mp_threads))
        zipped_args = zip(all_designs, repeat(args.fragment_database), repeat(args.symmetry_group),
                          repeat(args.command_only), repeat(args.suspend),
                          repeat(args.debug))  # repeat(args.prioritize_frags)
        results = SDUtils.mp_starmap(design, zipped_args, mp_threads)
    else:
        logger.info('Beginning processing (For faster performance, use -m during submission)')
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
            logger.info('List of all %s commands were written to \'%s\'' %
                        (PUtils.stage[stage].upper(), command_files[i]))
