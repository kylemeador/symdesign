from copy import copy

from PathUtils import program_command, program_name, interface_design
from Query.utils import input_string, confirmation_string, bool_d, invalid_string, header_string, format_string
from SymDesignUtils import pretty_format_table, DesignError, handle_errors, clean_comma_separated_string, \
    format_index_string
from SequenceProfile import read_fasta_file

terminal_formatter = '\n\t\t\t\t\t\t     '
# Todo separate into types of options, aka fragments, residue selection, symmetry
global_flags = {
    # 'symmetry': {'type': str, 'default': None,
    #             'description': 'The symmetry to use for the Design. Symmetry won\'t be assigned%sif not provided '
    #                            'unless Design targets are %s.py outputs' % (terminal_formatter, nano.title())},
    #             'nanohedra_output': {'type': bool, 'default': False,
    #                                  'description': 'Whether the design targets are the result of %s output.%sUse this'
    #                                                 ' flag to set up a %s project from docked poses%sor to analyze the '
    #                                                 'output of %s.%sAfter a %s directory is initialized with '
    #                                                 '--nanohedra_output True,%syou may submit future jobs containing '
    #                                                 'these designs without this flag'
    #                                                 % (terminal_formatter, nano.title(), terminal_formatter,
    #                                                    program_name, terminal_formatter, nano.title(),
    #                                                    terminal_formatter, program_name)},
                'skip_logging': {'type': bool, 'default': False, 'description': 'Whether logging should be suspended'},
                # 'design_range': {'type': str, 'default': None,
                #                  'description': 'Whether to subset selected designs by a range of percentage values'},
                'mpi': {'type': int, 'default': None,
                        'description': 'If commands should be run as MPI parallel processes, how many processes should '
                                       'be invoked for each job?'},
                }
design_flags = {
    # 'design_with_evolution': {'type': bool, 'default': True,
    #                           'description': 'Whether to design with evolutionary amino acid frequency info'},
    # 'no_term_constraint': {'type': bool, 'default': True,
    #                           'description': 'Whether to design with fragment amino acid frequency info'},
    # 'fragments_exist': {'type': bool, 'default': True,
    #                     'description': 'If fragment data has been generated for the design,%s'
    #                                    'If nanohedra_output is True, this is also True'
    #                                    % terminal_formatter},
    # generate_fragments: {'type': bool, 'default': False,
    #                        'description': 'Whether fragments should be generated fresh for each Pose'},
    # 'write_fragments': {'type': bool, 'default': True,
    #                     'description': 'Whether fragments should be written to file for each Pose'},
    # 'output_assembly': {'type': bool, 'default': False,
    #                     'description': 'If symmetric, whether the expanded assembly should be output.%s'
    #                                    '2- and 3-D materials will be output with a single unit cell.'
    #                                    % terminal_formatter},
    # 'number_of_trajectories': {'type': int, 'default': nstruct,
    #                            'description': 'The number of individual design trajectories to be run for each design'
    #                                           '%sThis determines how many sequence sampling runs are used.'
    #                                           % terminal_formatter},
    'require_design_at_residues':
        {'type': str, 'default': None,
         'description': 'Regardless of participation in an interface,%sif certain residues should be included in'
                        'design, specify the%sresidue POSE numbers as a comma separated string.%s'
                        'Ex: \'23,24,35,41,100-110,267,289-293\' Ranges are allowed'
                        % (terminal_formatter, terminal_formatter, terminal_formatter)},
    'select_designable_residues_by_sequence':
        {'type': str, 'default': None,
         'description': 'If design should occur ONLY at certain residues,%sspecify the location of a .fasta file '
                        'containing the design selection.%sRun \'%s --single my_pdb_file.pdb design_selector\' '
                        'to set this up.'
                        % (terminal_formatter, terminal_formatter, program_command)},
    'select_designable_residues_by_pdb_number':
        {'type': str, 'default': None,
         'description': 'If design should occur ONLY at certain residues,%sspecify the residue PDB number(s) '
                        'as a comma separated string.%sRanges are allowed '
                        'Ex: \'40-45,170-180,227,231\'' % (terminal_formatter, terminal_formatter)},
    'select_designable_residues_by_pose_number':
        {'type': str, 'default': None,
         'description': 'If design should occur ONLY at certain residues,%sspecify the residue POSE number(s) '
                        'as a comma separated string.%sRanges are allowed '
                        'Ex: \'23,24,35,41,100-110,267,289-293\'' % (terminal_formatter, terminal_formatter)},
    'select_designable_chains':
        {'type': str, 'default': None,
         'description': 'If a design should occur ONLY at certain chains,%sprovide the chain ID\'s as a comma '
                        'separated string.%sEx: \'A,C,D\'' % (terminal_formatter, terminal_formatter)},
    'mask_designable_residues_by_sequence':
        {'type': str, 'default': None,
         'description': 'If design should NOT occur at certain residues,%sspecify the location of a .fasta file '
                        'containing the design mask.%sRun \'%s --single my_pdb_file.pdb design_selector\' '
                        'to set this up.'
                        % (terminal_formatter, terminal_formatter, program_command)},
    'mask_designable_residues_by_pdb_number':
        {'type': str, 'default': None,
         'description': 'If design should NOT occur at certain residues,%sspecify the residue PDB number(s) '
                        'as a comma separated string.%sEx: \'27-35,118,281\' Ranges are allowed'
                        % (terminal_formatter, terminal_formatter)},
    'mask_designable_residues_by_pose_number':
        {'type': str, 'default': None,
         'description': 'If design should NOT occur at certain residues,%sspecify the residue POSE number(s) '
                        'as a comma separated string.%sEx: \'27-35,118,281\' Ranges are allowed'
                        % (terminal_formatter, terminal_formatter)},
    'mask_designable_chains':
        {'type': str, 'default': None,
         'description': 'If a design should NOT occur at certain chains,%sprovide the chain ID\'s as a comma '
                        'separated string.%sEx: \'C\'' % (terminal_formatter, terminal_formatter)}
    # 'input_location': '(str) Specify a file with a list of input files or a directory where input files are '
    #                   'located. If the input is a %s.py output, specifying the master output directory is '
    #                   'sufficient' % nano
    }
filter_flags = {}

design = copy(global_flags)
design.update(design_flags)
filters = copy(global_flags)
filters.update(filter_flags)
all_flags = copy(design)
all_flags.update(filters)
flags = {interface_design: design, None: all_flags}
#        'analysis': global_flags, 'filter': filters, 'sequence_selection': global_flags,


def process_residue_selector_flags(design_flags):
    # Pull nanohedra_output and mask_design_using_sequence out of flags
    # Todo move to a verify design_selectors function inside of Pose? Own flags module?
    entity_req, chain_req, residues_req, residues_pdb_req = None, None, set(), set()
    if 'require_design_at_pdb_residues' in design_flags and design_flags['require_design_at_pdb_residues']:
        residues_pdb_req = residues_pdb_req.union(
            format_index_string(design_flags['require_design_at_pdb_residues']))
    if 'require_design_at_residues' in design_flags and design_flags['require_design_at_residues']:
        residues_req = residues_req.union(
            format_index_string(design_flags['require_design_at_residues']))
    # -------------------
    pdb_select, entity_select, chain_select, residue_select, residue_pdb_select = None, None, None, set(), set()
    if 'select_designable_residues_by_sequence' in design_flags \
            and design_flags['select_designable_residues_by_sequence']:
        residue_select = residue_select.union(
            generate_sequence_mask(design_flags['select_designable_residues_by_sequence']))
    if 'select_designable_residues_by_pdb_number' in design_flags \
            and design_flags['select_designable_residues_by_pdb_number']:
        residue_pdb_select = residue_pdb_select.union(
            format_index_string(design_flags['select_designable_residues_by_pdb_number']))
    if 'select_designable_residues_by_pose_number' in design_flags \
            and design_flags['select_designable_residues_by_pose_number']:
        residue_select = residue_select.union(
            format_index_string(design_flags['select_designable_residues_by_pose_number']))
    if 'select_designable_chains' in design_flags and design_flags['select_designable_chains']:
        chain_select = generate_chain_mask(design_flags['select_designable_chains'])
    # -------------------
    pdb_mask, entity_mask, chain_mask, residue_mask, residue_pdb_mask = None, None, None, set(), set()
    if 'mask_designable_residues_by_sequence' in design_flags \
            and design_flags['mask_designable_residues_by_sequence']:
        residue_mask = residue_mask.union(
            generate_sequence_mask(design_flags['mask_designable_residues_by_sequence']))
    if 'mask_designable_residues_by_pdb_number' in design_flags \
            and design_flags['mask_designable_residues_by_pdb_number']:
        residue_pdb_mask = residue_pdb_mask.union(
            format_index_string(design_flags['mask_designable_residues_by_pdb_number']))
    if 'mask_designable_residues_by_pose_number' in design_flags \
            and design_flags['mask_designable_residues_by_pose_number']:
        residue_mask = residue_mask.union(
            format_index_string(design_flags['mask_designable_residues_by_pose_number']))
    if 'mask_designable_chains' in design_flags and design_flags['mask_designable_chains']:
        chain_mask = generate_chain_mask(design_flags['mask_designable_chains'])
    # -------------------
    return {'design_selector':
            {'selection': {'pdbs': pdb_select, 'entities': entity_select,
                           'chains': chain_select, 'residues': residue_select,
                           'pdb_residues': residue_pdb_select},
             'mask': {'pdbs': pdb_mask, 'entities': entity_mask, 'chains': chain_mask,
                      'residues': residue_mask, 'pdb_residues': residue_pdb_mask},
             'required': {'entities': entity_req, 'chains': chain_req,
                          'residues': residues_req, 'pdb_residues': residues_pdb_req}}}


def return_default_flags(mode):
    if mode in flags:
        return dict(zip(flags[mode].keys(), [value_format['default'] for value_format in flags[mode].values()]))
    else:
        return dict(zip(all_flags.keys(), [value_format['default'] for value_format in all_flags.values()]))


@handle_errors(errors=KeyboardInterrupt)
def query_user_for_flags(mode=interface_design, template=False):
    flags_file = '%s.flags' % mode
    flag_output = return_default_flags(mode)
    write_file = False
    print('\n%s' % header_string % 'Generate %s Flags' % program_name)
    if template:
        write_file = True
        print('Writing template to %s' % flags_file)

    flags_header = [('Flag', 'Default', 'Description')]
    flags_description = list((flag, values['default'], values['description']) for flag, values in flags[mode].items())
    while not write_file:
        flags_table = pretty_format_table(flags_header + flags_description)
        # logger.info('Starting with options:\n\t%s' % '\n\t'.join(options_table))
        print('For a %s run, the following flags can be used to customize the design. You can include these in'
              ' a flags file\nsuch as \'my_design.flags\' specified on the command line like \'@my_design.flags\' or '
              'manually by typing them in.\nTo automatically generate a flags file template, run \'%s flags --template'
              '\' then modify the defaults.\nAlternatively, input the number(s) corresponding to the flag(s) of '
              'interest from the table below to automatically\ngenerate this file. PDB numbering is defined here as an'
              ' input file with residue numbering reset at each chain. Pose\nnumbering is defined here as input file '
              'with residue numbering incrementing from 1 to N without resetting at chain breaks\n%s'
              % (mode, program_command, '\n'.join('%s %s' % (str(idx).rjust(2, ' ') if idx > 0 else '  ', item)
                                                  for idx, item in enumerate(flags_table))))
        flags_input = input('\nEnter the numbers corresponding to the flags your design requires. Ex: \'1 3 6\'%s'
                            % input_string)
        flag_numbers = flags_input.split()
        chosen_flags = [list(flags[mode].keys())[flag - 1] for flag in map(int, flag_numbers)
                        if len(flags[mode].keys()) >= flag > 0]
        value_array = []
        for idx, flag in enumerate(chosen_flags):
            valid = False
            while not valid:
                arg_value = input('\tFor \'%s\' what %s value should be used? Default is \'%s\'%s'
                                  % (flag, flags[mode][flag]['type'], flags[mode][flag]['default'], input_string))
                if flags[mode][flag]['type'] == bool:
                    if arg_value == '':
                        arg_value = 'None'
                    if isinstance(eval(arg_value.title()), flags[mode][flag]['type']):
                        value_array.append(arg_value.title())
                        valid = True
                    else:
                        print('%s %s is not a valid choice of type %s!'
                              % (invalid_string, arg_value, flags[mode][flag]['type']))
                elif flags[mode][flag]['type'] == str:
                    if arg_value == '':
                        arg_value = None
                    value_array.append(arg_value)
                    valid = True
                elif flags[mode][flag]['type'] == int:
                    if arg_value.isdigit():
                        value_array.append(arg_value)
                        valid = True
                    else:
                        print('%s %s is not a valid choice of type %s!'
                              % (invalid_string, arg_value, flags[mode][flag]['type']))

        flag_input = zip(chosen_flags, value_array)  # flag value (key), user input (str)
        while True:
            validation = input('You selected:\n%s\n\nOther flags will take their default values. %s'
                               % ('\n'.join(format_string % flag for flag in flag_input), confirmation_string))
            if validation.lower() in bool_d:
                break
            else:
                print('%s %s is not a valid choice!' % invalid_string, validation)
        if bool_d[validation]:
            write_file = True
            flag_output.update(dict(zip(chosen_flags, value_array)))  # flag value (key), user input (str)

    with open(flags_file, 'w') as f:
        f.write('%s\n' % '\n'.join('-%s %s' % (flag, variable) for flag, variable in flag_output.items()))


def load_flags(file):
    with open(file, 'r') as f:
        flags = {dict(tuple(flag.lstrip('-').split())) for flag in f.readlines()}

    return flags


def generate_sequence_mask(fasta_file):
    """From a sequence with a design_selector, grab the residue indices that should be designed in the target
    structural calculation

    Returns:
        (list): The residue numbers (in pose format) that should be ignored in design
    """
    sequence_and_mask = read_fasta_file(fasta_file)
    sequences = list(sequence_and_mask)
    sequence = sequences[0]
    mask = sequences[1]
    if not len(sequence) == len(mask):
        raise DesignError('The sequence and design_selector are different lengths! Please correct the alignment and '
                          'lengths before proceeding.')

    return [idx for idx, aa in enumerate(mask, 1) if aa != '-']


def generate_chain_mask(chain_string):
    """From a string with a design_selection, format the chains provided

    Returns:
        (list): chain ids in pose format
    """
    return clean_comma_separated_string(chain_string)
