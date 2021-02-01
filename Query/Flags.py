from PathUtils import program_command, nano, program_name
from Query.PDB import user_input_format, input_string, format_string, numbered_format_string, confirmation_string, \
    bool_d, invalid_string, header_string


def query_user_for_flags(mode='design', template=False):
    flags_file = '%s.flags' % mode
    # Todo separate into types of options, aka fragments, residue selection, symmetry
    flags = \
        {'design':
         {'symmetry': {'type': str, 'default': None,
                       'description': 'The symmetry to use for the Design. Symmetry won\'t be assigned if not provided '
                                      'unless Design targets are %s.py outputs' % nano},
          'nanohedra_output': {'type': bool, 'default': True,
                               'description': 'Whether the design targets are a %s output' % nano},
          'fragments_exist': {'type': bool, 'default': True,
                              'description': 'If fragment data has been generated for the design, where is it located?'},
          'generate_fragments': {'type': bool, 'default': False,
                                 'description': 'Whether fragments should be generated fresh for each Pose'},
          'design_with_fragments': {'type': bool, 'default': True,
                                    'description': 'Whether to design with fragment amino acid frequency info'},
          'design_with_evolution': {'type': bool, 'default': True,
                                    'description': 'Whether to design with evolutionary amino acid frequency info'},
          'output_assembly': {'type': bool, 'default': False,
                              'description': 'If symmetric, whether the expanded assembly should be output. '
                                             '2- and 3-D materials will be output with a single unit cell.'},
          'select_designable_residues_by_sequence':
              {'type': str, 'default': None,
               'description': 'If design should only occur at certain residues, specify the location of a .fasta file '
                              'containing the design selection. Run \'%s design_selection path/to/your.pdb\' '
                              'to set this up.'
                              % program_command},
          'select_designable_residues_by_pose_number':
              {'type': str, 'default': None,
               'description': 'If design should only occur at certain residues, specify the residue POSE numbers '
                              '(starting with 1) as a comma separated string. Ranges are allowed '
                              'Ex: \'23,24,35,41,100-110,267,289-293\''},
          'select_designable_chains':
              {'type': str, 'default': None,
               'description': 'If a design should be masked at certain chains, provide the chain ID\'s as a comma '
                              'separated string. Ex: \'A,C,D\''}

          # 'input_location': '(str) Specify a file with a list of input files or a directory where input files are '
          #                   'located. If the input is a %s.py output, specifying the master output directory is '
          #                   'sufficient' % nano
          },
         'filter':
             {  # TODO

             }}

    write_file = False
    flag_output = dict(zip(flags[mode].keys(), [value_format['default'] for value_format in flags[mode].values()]))
    print('\n%s' % header_string % 'Generate %s Flags' % program_name)
    if template:
        write_file = True
        print('Writing template to %s' % flags_file)

    flags_description = list(zip(flags[mode].keys(), [value_format['description']
                                                      for value_format in flags[mode].values()]))
    while not write_file:
        print('For a %s run, the following flags can be used to customize the design. You should include these in'
              ' a \'@flags\' file (run \'%s flags --template\' for a template) or select from the following list to '
              'generate a flags file automatically.\n%s'
              % (mode, program_command, user_input_format
                 % '\n'.join(numbered_format_string % (idx, *item) for idx, item in enumerate(flags_description, 1))))
        flags_input = input('\nEnter the numbers corresponding to the flags your design requires. Ex: \'1 3 6\'\n%s'
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
