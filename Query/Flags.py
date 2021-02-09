from PathUtils import program_command, nano, program_name
from Query.PDB import user_input_format, input_string, format_string, numbered_format_string, confirmation_string, \
    bool_d, invalid_string, header_string
from SymDesignUtils import pretty_format_table


terminal_formatter = '\n\t\t\t\t\t\t     '
# Todo separate into types of options, aka fragments, residue selection, symmetry
flags = \
    {'design':
     {'symmetry': {'type': str, 'default': None,
                   'description': 'The symmetry to use for the Design. Symmetry won\'t be assigned%sif not provided '
                                  'unless Design targets are %s.py outputs' % (terminal_formatter, nano)},
      'nanohedra_output': {'type': bool, 'default': True,
                           'description': 'Whether the design targets are a %s output' % nano},
      'fragments_exist': {'type': bool, 'default': True,
                          'description': 'If fragment data has been generated for the design,%swhere is it located?'
                                         % terminal_formatter},
      'generate_fragments': {'type': bool, 'default': False,
                             'description': 'Whether fragments should be generated fresh for each Pose'},
      'design_with_fragments': {'type': bool, 'default': True,
                                'description': 'Whether to design with fragment amino acid frequency info'},
      'design_with_evolution': {'type': bool, 'default': True,
                                'description': 'Whether to design with evolutionary amino acid frequency info'},
      'output_assembly': {'type': bool, 'default': False,
                          'description': 'If symmetric, whether the expanded assembly should be output.%s'
                                         '2- and 3-D materials will be output with a single unit cell.'
                                         % terminal_formatter},
      'require_design_at_residues':
          {'type': str, 'default': None,
           'description': 'Regardless of participation in an interface,%sif certain residues should be included in'
                          'design, specify the%sresidue POSE numbers as a comma separated string.%s'
                          'Ex: \'23,24,35,41,100-110,267,289-293\' Ranges are allowed'
                          % (terminal_formatter, terminal_formatter, terminal_formatter)},
      'select_designable_residues_by_sequence':
          {'type': str, 'default': None,
           'description': 'If design should occur ONLY at certain residues,%sspecify the location of a .fasta file '
                          'containing the design selection.%sRun \'%s design_selector path/to/your.pdb\' '
                          'to set this up.'
                          % (terminal_formatter, terminal_formatter, program_command)},
      'select_designable_residues_by_pose_number':
          {'type': str, 'default': None,
           'description': 'If design should occur ONLY at certain residues,%sspecify the residue POSE numbers '
                          'as a comma separated string.%sRanges are allowed '
                          'Ex: \'23,24,35,41,100-110,267,289-293\'' % (terminal_formatter, terminal_formatter)},
      'select_designable_chains':
          {'type': str, 'default': None,
           'description': 'If a design should occur ONLY at certain chains,%sprovide the chain ID\'s as a comma '
                          'separated string.%sEx: \'A,C,D\'' % (terminal_formatter, terminal_formatter)},
      'mask_designable_residues_by_sequence':
          {'type': str, 'default': None,
           'description': 'If design should NOT occur at certain residues,%sspecify the location of a .fasta file '
                          'containing the design mask.%sRun \'%s design_selector path/to/your.pdb\' '
                          'to set this up.'
                          % (terminal_formatter, terminal_formatter, program_command)},
      'mask_designable_residues_by_pose_number':
          {'type': str, 'default': None,
           'description': 'If design should NOT occur at certain residues,%sspecify the POSE number of residue(s) '
                          'as a comma separated string.%sEx: \'27-35,118,281\' Ranges are allowed'
                          % (terminal_formatter, terminal_formatter)},
      'mask_designable_chains':
          {'type': str, 'default': None,
           'description': 'If a design should NOT occur at certain chains,%sprovide the chain ID\'s as a comma '
                          'separated string.%sEx: \'C\'' % (terminal_formatter, terminal_formatter)}
      # 'input_location': '(str) Specify a file with a list of input files or a directory where input files are '
      #                   'located. If the input is a %s.py output, specifying the master output directory is '
      #                   'sufficient' % nano
      },
     'filter':
         {  # TODO

         }}


def return_default_flags(mode):
    if mode in flags:
        return dict(zip(flags[mode].keys(), [value_format['default'] for value_format in flags[mode].values()]))
    else:
        return dict()


def query_user_for_flags(mode='design', template=False):
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
              'interest from the table below to automatically\ngenerate this file\n\n%s'
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
