from __future__ import annotations

import argparse
import os
import sys
from typing import AnyStr

from psutil import cpu_count

from metrics import metric_weight_functions
from utils.path import submodule_guide, submodule_help, force_flags, fragment_dbs, biological_interfaces, \
    sym_entry, program_output, nano_entity_flag1, nano_entity_flag2, data, \
    clustered_poses, interface_design, no_evolution_constraint, no_hbnet, no_term_constraint, number_of_trajectories, \
    structure_background, scout, design_profile, evolutionary_profile, \
    fragment_profile, all_scores, analysis_file, select_sequences, program_name, nano, \
    program_command, analysis, select_poses, output_fragments, output_oligomers, protocol, current_energy_function, \
    ignore_clashes, ignore_pose_clashes, ignore_symmetric_clashes, select_designs
from utils.ProteinExpression import expression_tags
from resources.query.utils import input_string, confirmation_string, bool_d, invalid_string, header_string, \
    format_string
from structure.sequence import read_fasta_file
from utils import handle_errors, pretty_format_table, clean_comma_separated_string, format_index_string, DesignError, \
    ex_path


# terminal_formatter = '\n\t\t\t\t\t\t     '


def process_design_selector_flags(flags: dict[str]) -> dict[str, dict[str, set | set[int] | set[str]]]:
    # Todo move to a verify design_selectors function inside of Pose? Own flags module?
    #  Pull mask_design_using_sequence out of flags
    # -------------------
    entity_select, chain_select, residue_select, residue_pdb_select = set(), set(), set(), set()
    select_residues_by_sequence = flags.get('select_designable_residues_by_sequence')
    if select_residues_by_sequence:
        residue_select = residue_select.union(generate_sequence_mask(select_residues_by_sequence))

    select_residues_by_pdb_number = flags.get('select_designable_residues_by_pdb_number')
    if select_residues_by_pdb_number:
        residue_pdb_select = residue_pdb_select.union(format_index_string(select_residues_by_pdb_number))

    select_residues_by_pose_number = flags.get('select_designable_residues_by_pose_number')
    if select_residues_by_pose_number:
        residue_select = residue_select.union(format_index_string(select_residues_by_pose_number))

    select_chains = flags.get('select_designable_chains')
    if select_chains:
        chain_select = chain_select.union(generate_chain_mask(select_chains))
    # -------------------
    entity_mask, chain_mask, residue_mask, residue_pdb_mask = set(), set(), set(), set()
    mask_residues_by_sequence = flags.get('mask_designable_residues_by_sequence')
    if mask_residues_by_sequence:
        residue_mask = residue_mask.union(generate_sequence_mask(mask_residues_by_sequence))

    mask_residues_by_pdb_number = flags.get('mask_designable_residues_by_pdb_number')
    if mask_residues_by_pdb_number:
        residue_pdb_mask = residue_pdb_mask.union(format_index_string(mask_residues_by_pdb_number))

    mask_residues_by_pose_number = flags.get('mask_designable_residues_by_pose_number')
    if mask_residues_by_pose_number:
        residue_mask = residue_mask.union(format_index_string(mask_residues_by_pose_number))

    mask_chains = flags.get('mask_designable_chains')
    if mask_chains:
        chain_mask = chain_mask.union(generate_chain_mask(mask_chains))
    # -------------------
    entity_req, chain_req, residues_req, residues_pdb_req = set(), set(), set(), set()
    require_residues_by_pdb_number = flags.get('require_design_at_pdb_residues')
    if require_residues_by_pdb_number:
        residues_pdb_req = residues_pdb_req.union(format_index_string(require_residues_by_pdb_number))

    require_residues_by_pose_number = flags.get('require_design_at_residues')
    if require_residues_by_pose_number:
        residues_req = residues_req.union(format_index_string(require_residues_by_pose_number))

    require_residues_by_chain = flags.get('require_design_at_chains')
    if require_residues_by_chain:
        chain_req = chain_req.union(generate_chain_mask(require_residues_by_chain))

    return dict(selection=dict(entities=entity_select, chains=chain_select, residues=residue_select,
                               pdb_residues=residue_pdb_select),
                mask=dict(entities=entity_mask, chains=chain_mask, residues=residue_mask,
                          pdb_residues=residue_pdb_mask),
                required=dict(entities=entity_req, chains=chain_req, residues=residues_req,
                              pdb_residues=residues_pdb_req))


# def return_default_flags():
#     # mode_flags = flags.get(mode, design_flags)
#     # if mode_flags:
#     return dict(zip(design_flags.keys(), [value_format['default'] for value_format in design_flags.values()]))
#     # else:
#     #     return dict(zip(all_flags.keys(), [value_format['default'] for value_format in all_flags.values()]))


@handle_errors(errors=(KeyboardInterrupt,))
def query_user_for_flags(mode=interface_design, template=False):
    flags_file = f'{mode}.flags'
    raise NotImplementedError('This function is not working')
    flag_output = return_default_flags()
    write_file = False
    print('\n%s' % header_string % f'Generate {program_name} Flags')
    if template:
        write_file = True
        print(f'Writing template to {flags_file}')

    flags_header = [('Flag', 'Default', 'Description')]
    flags_description = list((flag, values['default'], values['description']) for flag, values in design_flags.items())
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
        chosen_flags = [list(design_flags.keys())[flag - 1] for flag in map(int, flag_numbers)
                        if len(design_flags.keys()) >= flag > 0]
        value_array = []
        for idx, flag in enumerate(chosen_flags):
            valid = False
            while not valid:
                arg_value = input('\tFor \'%s\' what %s value should be used? Default is \'%s\'%s'
                                  % (flag, design_flags[flag]['type'], design_flags[flag]['default'], input_string))
                if design_flags[flag]['type'] == bool:
                    if arg_value == '':
                        arg_value = 'None'
                    if isinstance(eval(arg_value.title()), design_flags[flag]['type']):
                        value_array.append(arg_value.title())
                        valid = True
                    else:
                        print('%s %s is not a valid choice of type %s!'
                              % (invalid_string, arg_value, design_flags[flag]['type']))
                elif design_flags[flag]['type'] == str:
                    if arg_value == '':
                        arg_value = None
                    value_array.append(arg_value)
                    valid = True
                elif design_flags[flag]['type'] == int:
                    if arg_value.isdigit():
                        value_array.append(arg_value)
                        valid = True
                    else:
                        print('%s %s is not a valid choice of type %s!'
                              % (invalid_string, arg_value, design_flags[flag]['type']))

        flag_input = zip(chosen_flags, value_array)  # flag value (key), user input (str)
        while True:
            validation = input('You selected:\n%s\n\nOther flags will take their default values. %s'
                               % ('\n'.join(format_string.format(*flag) for flag in flag_input), confirmation_string))
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
        return {dict(tuple(flag.lstrip('-').split())) for flag in f.readlines()}


def generate_sequence_mask(fasta_file: AnyStr) -> list[int]:
    """From a sequence with a design_selector, grab the residue indices that should be designed in the target
    structural calculation

    Args:
        fasta_file: The path to a file with fasta information
    Returns:
        The residue numbers (in pose format) that should be ignored in design
    """
    sequence_and_mask = read_fasta_file(fasta_file)
    sequences = list(sequence_and_mask)
    sequence = sequences[0]
    mask = sequences[1]
    if not len(sequence) == len(mask):
        raise DesignError('The sequence and design_selector are different lengths! Please correct the alignment and '
                          'lengths before proceeding.')

    return [idx for idx, aa in enumerate(mask, 1) if aa != '-']


def generate_chain_mask(chains: str) -> set[str]:
    """From a string with a design_selection, format the chains provided

    Args:
        chains: The specified chains separated by commas to split
    Returns:
        The provided chain ids in pose format
    """
    return set(clean_comma_separated_string(chains))


# ---------------------------------------------------
class Formatter(argparse.RawTextHelpFormatter, argparse.RawDescriptionHelpFormatter, argparse.HelpFormatter):

    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            # change to
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append(f'{option_string}')
                parts[-1] += f' {args_string.upper()}'
            return ', '.join(parts)

# The help strings can include various format specifiers to avoid repetition of things like the program name or the
# argument default. The available specifiers include the program name, %(prog)s and most keyword arguments to
# add_argument(), e.g. %(default)s, %(type)s, etc.:

# Todo Found the following for formatting the prog use case in subparsers
#  {'refine': ArgumentParser(prog='python SymDesign.py module [module_arguments] [input_arguments]'
#                                 '[optional_arguments] refine'
# TODO add action=argparse.BooleanOptionalAction to all action='store_true'/'store_false'
optional_title = 'Optional Arguments'
design_selector_title = 'Design Selector Arguments'
input_title = 'Input Arguments'
output_title = 'Output Arguments'
module_title = 'Module Arguments'
usage_string = f'\n      python %s.py %%s [{module_title.lower()}] [{input_title.lower()}] [{output_title.lower()}] ' \
               f'[{design_selector_title.lower()}] [{optional_title.lower()}]' % program_name
guide_args = ('--guide',)
guide_kwargs = dict(action='store_true', help=f'Display the {program_name} or {program_name} Module specific guide\nEx:'
                                              f' "{program_command} --guide"\nor "{submodule_guide}"')
# Todo allow setup.py main
setup_args = ('--set_up',)
setup_kwargs = dict(action='store_true', help='Show the %(prog)s set up instructions\nDefault=%(default)s')

parser_options_group = dict(title=f'{"_" * len(optional_title)}\n{optional_title}',
                            description=f'\nAdditional options control symmetry, the extent of file output, various '
                                        f'{program_name} runtime considerations, and programmatic options for '
                                        f'determining design outcomes')
options_arguments = {
    ('-C', '--cores'): dict(type=int, default=cpu_count(logical=False) - 1,
                            help='Number of cores to use during --multi_processing\nIf run on a cluster, the number of '
                                 'cores will reflect the cluster allocation,\notherwise, will use #physical_cores-1'
                                 '\nDefault=%(default)s'),
    ('--debug',): dict(action='store_true',
                       help='Whether to log debugging messages to stdout\nDefault=%(default)s'),
    ('-F', f'--{force_flags}'): dict(action='store_true',
                                     help='Force generation of a new flags file to update script parameters'
                                          '\nDefault=%(default)s'),
    # ('-gf', f'--{generate_fragments}'): dict(action='store_true',
    #                                          help='Generate interface fragment observations for poses of interest'
    #                                               '\nDefault=%(default)s'),
    guide_args: guide_kwargs,
    # ('--guide',): dict(action='store_true',
    #                    help='Access the %s guide! Display the program or module specific guide\nEx: "%s --guide" '
    #                         'or "%s"' % (program_name, program_command, submodule_guide)),
    ('-i', '--fragment_database'): dict(type=str, choices=fragment_dbs, default=biological_interfaces,
                                        help='Database to match fragments for interface specific scoring matrices\n'
                                             'Default=%(default)s'),
    ('-ic', f'--{ignore_clashes}'): dict(action='store_true',
                                         help='Whether ANY identified backbone/Cb clash should be ignored and '
                                              'allowed to process\nDefault=%(default)s'),
    ('-ipc', f'--{ignore_pose_clashes}'): dict(action='store_true',
                                               help='Whether asu/pose clashes should be '
                                                    'ignored and allowed to process\nDefault=%(default)s'),
    ('-isc', f'--{ignore_symmetric_clashes}'): dict(action='store_true',
                                                    help='Whether symmetric clashes should be ignored and allowed'
                                                         ' to process\nDefault=%(default)s'),
    ('--mpi',): dict(type=int, default=0, help='If commands should be run as MPI parallel processes, how many '
                                               'processes\nshould be invoked for each job?\nDefault=%(default)s'),
    ('-M', '--multi_processing'): dict(action='store_true',
                                       help='Should job be run with multiple processors?\nDefault=%(default)s'),
    ('--overwrite',): dict(action='store_true',
                           help='Whether to overwrite existing structures upon job fulfillment\nDefault=%(default)s'),
    ('-P', '--preprocessed'): dict(action='store_true',
                                   help=f'Whether the designs of interest have been preprocessed for the '
                                        f'{current_energy_function}\nenergy function and/or missing loops\n'
                                        f'Default=%(default)s'),
    ('-R', '--run_in_shell'): dict(action='store_true',
                                   help='Should commands be executed at %(prog)s runtime?\nIn most cases, it won\'t '
                                        'maximize cassini\'s computational resources.\nAll computation may'
                                        'fail on a single trajectory mistake.\nDefault=%(default)s'),
    setup_args: setup_kwargs,
    # ('--set_up',): dict(action='store_true',
    #                     help='Show the %(prog)s set up instructions\nDefault=%(default)s'),  # Todo allow setup.py main
    ('--skip_logging',): dict(action='store_true',
                              help='Skip logging output to files and direct all logging to stream?'
                                   '\nDefault=%(default)s'),
    ('-E', '--entry', f'--{sym_entry}'): dict(type=int, default=None, dest=sym_entry,
                                              help=f'The entry number of {nano.title()} docking combinations to use'),
    ('-S', '--symmetry'): dict(type=str, default=None,
                               help='The specific symmetry of the poses of interest.\nPreferably in a composition '
                                    'formula such as T:{C3}{C3}...\nCan also provide the keyword "cryst" to use crystal'
                                    ' symmetry'),
    ('-U', '--update_database'): dict(action='store_true',
                                      help='Whether to update resources for each Structure in the database'
                                           '\nDefault=%(default)s'),
}
parser_residue_selector_group = \
    dict(title=f'{"_" * len(design_selector_title)}\n{design_selector_title}',
         description='\nResidue selectors control which parts of the Pose are included in calculations')
residue_selector_arguments = {
    ('--require_design_at_residues',):
        dict(type=str, default=None,
             help='Regardless of participation in an interface, if certain\n residues should be included in'
                  'design, specify the\nresidue POSE numbers as a comma separated string.\n'
                  'Ex: "23,24,35,41,100-110,267,289-293" Ranges are allowed'),
    ('--require_design_at_chains',):
        dict(type=str, default=None,
             help='Regardless of participation in an interface, if certain\n chains should be included in'
                  'design, specify the\nchain ID\'s as a comma separated string.\n'
                  'Ex: "A,D"'),
    # ('--select_designable_residues_by_sequence',):
    #     dict(type=str, default=None,
    #          help='If design should occur ONLY at certain residues, specify\nthe location of a .fasta file '
    #               f'containing the design selection\nRun "{program_command} --single my_pdb_file.pdb design_selector" '
    #               'to set this up'),
    ('--select_designable_residues_by_pdb_number',):
        dict(type=str, default=None, metavar=None,
             help='If design should occur ONLY at certain residues, specify\nthe residue PDB number(s) '
                  'as a comma separated string\nRanges are allowed Ex: "40-45,170-180,227,231"'),
    ('--select_designable_residues_by_pose_number',):
        dict(type=str, default=None,
             help='If design should occur ONLY at certain residues, specify\nthe residue POSE number(s) '
                  'as a comma separated string\nRanges are allowed Ex: "23,24,35,41,100-110,267,289-293"'),
    ('--select_designable_chains',):
        dict(type=str, default=None,
             help='If a design should occur ONLY at certain chains, specify\nthe chain ID\'s as a comma '
                  'separated string\nEx: "A,C,D"'),
    # ('--mask_designable_residues_by_sequence',):
    #     dict(type=str, default=None,
    #          help='If design should NOT occur at certain residues, specify\nthe location of a .fasta file '
    #               f'containing the design mask\nRun "{program_command} --single my_pdb_file.pdb design_selector" '
    #               'to set this up'),
    ('--mask_designable_residues_by_pdb_number',):
        dict(type=str, default=None,
             help='If design should NOT occur at certain residues, specify\nthe residue PDB number(s) '
                  'as a comma separated string\nEx: "27-35,118,281" Ranges are allowed'),
    ('--mask_designable_residues_by_pose_number',):
        dict(type=str, default=None,
             help='If design should NOT occur at certain residues, specify\nthe residue POSE number(s) '
                  'as a comma separated string\nEx: "27-35,118,281" Ranges are allowed'),
    ('--mask_designable_chains',):
        dict(type=str, default=None,
             help='If a design should NOT occur at certain chains, provide\nthe chain ID\'s as a comma '
                  'separated string\nEx: "C"')
}
# ---------------------------------------------------
# Set Up SubModule Parsers
# ---------------------------------------------------
# module_parser = argparse.ArgumentParser(add_help=False)  # usage=usage_string,
# subparsers = module_parser.add_subparsers(title='module arguments', dest='module', required=True, description='These are the different modes that designs can be processed. They are\npresented in an order which might be utilized along a design workflow,\nwith utility modules listed at the bottom starting with check_clashes.\nTo see example commands or algorithmic specifics of each Module enter:\n%s\n\nTo get help with Module arguments enter:\n%s' % (submodule_guide, submodule_help))  # metavar='', # description='',
# ---------------------------------------------------
parser_orient = dict(orient=dict(help='Orient a symmetric assembly in a canonical orientation at the origin'))
# parser_orient = subparsers.add_parser('orient', help='Orient a symmetric assembly in a canonical orientation at the origin')
# ---------------------------------------------------
parser_refine = dict(refine=dict(help='Process Structures into an energy function'))
# parser_refine = subparsers.add_parser(refine, help='Process Structures into an energy function')
refine_arguments = {
    ('-ala', '--interface_to_alanine'): dict(action='store_true',
                                             help='Whether to mutate all interface residues to alanine before '
                                                  'refinement'),
    ('-met', '--gather_metrics'): dict(action='store_true',
                                       help='Whether to gather interface metrics for contained interfaces after '
                                            'refinement')
}
# ---------------------------------------------------
parser_nanohedra = dict(nanohedra=dict(help=f'Run or submit jobs to {nano.title()}.py'))
# parser_dock = subparsers.add_parser(nano, help='Run or submit jobs to %s.py.\nUse the Module arguments -c1/-c2, -o1/-o2, or -q to specify PDB Entity codes, building block directories, or query the PDB for building blocks to dock' % nano.title())
nanohedra_arguments = {
    # ('-e', '--entry', f'--{sym_entry}'): dict(type=int, default=None, dest=sym_entry,  # required=True,
    #                                           help=f'The entry number of {nano.title()} docking combinations to use'),
    ('-mv', '--match_value'): dict(type=float, default=0.5, dest='high_quality_match_value',
                                   help='What is the minimum match score required for a high quality fragment?'),
    ('-iz', '--initial_z_value'): dict(type=float, default=1.,
                                       help='The acceptable standard deviation z score for initial fragment overlap '
                                            'identification.\nSmaller values lead to more stringent matching criteria'
                                            '\nDefault=%(default)s'),
    ('-m', '--min_matched'): dict(type=int, default=3,
                                  help='How many high quality fragment pairs should be present before a pose is '
                                       'identified?\nDefault=%(default)s'),
    ('-Od', '--outdir', '--output_directory'): dict(type=str, dest='output_directory', default=None,
                                                    help='Where should the output from commands be written?\nDefault=%s'
                                                         % ex_path(program_output, data.title(),
                                                                   'NanohedraEntry[ENTRYNUMBER]DockedPoses')),
    ('-r1', '--rotation_step1'): dict(type=float, default=3.,
                                      help='The number of degrees to increment the rotational degrees of freedom '
                                           'search\nDefault=%(default)s'),
    ('-r2', '--rotation_step2'): dict(type=float, default=3.,
                                      help='The number of degrees to increment the rotational degrees of freedom '
                                           'search\nDefault=%(default)s'),
    ('-Os', '--output_surrounding_uc'): dict(action='store_true',
                                             help='Whether the surrounding unit cells should be output? Only for '
                                                  'infinite materials\nDefault=%(default)s')
}
parser_nanohedra_run_type_mutual_group = dict()  # required=True <- adding below to different parsers depending on need
nanohedra_run_type_mutual_arguments = {
    ('-e', '-entry', '--entry', f'--{sym_entry}'):
        dict(type=int, default=None, dest=sym_entry, help='The symmetry to use. See --query for possible symmetries'),
    ('-query', '--query',): dict(action='store_true', help='Run in query mode\nDefault=%(default)s'),
    # Todo alias analysis -metric
    ('-postprocess', '--postprocess',): dict(action='store_true',
                                             help='Run in post processing mode\nDefault=%(default)s')
}
# parser_dock_mutual1 = parser_dock.add_mutually_exclusive_group(required=True)
parser_nanohedra_mutual1_group = dict()  # required=True <- adding kwarg below to different parsers depending on need
nanohedra_mutual1_arguments = {
    ('-c1', '--pdb_codes1'): dict(type=os.path.abspath, default=None,
                                  help=f'File with list of PDB_entity codes for {nano} component 1'),
    ('-o1', f'-{nano_entity_flag1}'): dict(type=os.path.abspath, default=None,
                                           help=f'Disk location where {nano} component 1 file(s) are located'),
    # ('--query',): dict(action='store_true', help='Run Nanohedra in query mode\nDefault=%(default)s'),
    ('-Q', '--query_codes'): dict(action='store_true', help='Query the PDB API for corresponding codes')
}
# parser_dock_mutual2 = parser_dock.add_mutually_exclusive_group()
parser_nanohedra_mutual2_group = dict()  # required=False
nanohedra_mutual2_arguments = {
    ('-c2', '--pdb_codes2'): dict(type=os.path.abspath, default=None,
                                  help=f'File with list of PDB_entity codes for {nano} component 2'),
    ('-o2', f'-{nano_entity_flag2}'): dict(type=os.path.abspath, default=None,
                                           help=f'Disk location where {nano} component 2 file(s) are located'),
}
# ---------------------------------------------------
parser_cluster = dict(cluster_poses=dict(help='Cluster all poses by their spatial or interfacial similarity.\nThis is'
                                              ' useful to identify conformationally flexible docked configurations'))
# parser_cluster = subparsers.add_parser(cluster_poses, help='Cluster all poses by their spatial similarity. This can remove redundancy or be useful in identifying conformationally flexible docked configurations')
cluster_poses_arguments = {
    ('-m', '--mode'): dict(type=str, choices=['transform', 'ialign'], default='transform'),
    ('--as_objects',): dict(action='store_true', help='Whether to store the resulting pose cluster file as '
                                                      'PoseDirectory objects\nDefault stores as pose IDs'),
    ('-Of', '--output_file'): dict(type=str, default=clustered_poses,
                                   help='Name of the output .pkl file containing pose clusters. Will be saved to the '
                                        '%s folder of the output.\nDefault=%s'
                                        % (data.title(), clustered_poses % ('LOCATION', 'TIMESTAMP')))
}
# ---------------------------------------------------
parser_design = dict(interface_design=dict(help='Gather poses of interest and format for design using sequence '
                                                'constraints in Rosetta.\nConstrain using evolutionary profiles of '
                                                'homologous sequences and/or fragment profiles\nextracted from the PDB '
                                                'or neither'))
# parser_design = subparsers.add_parser(interface_design, help='Gather poses of interest and format for design using sequence constraints in Rosetta. Constrain using evolutionary profiles of homologous sequences and/or fragment profiles extracted from the PDB or neither.')
nstruct = 20
interface_design_arguments = {
    ('-nec', f'--{no_evolution_constraint}'): dict(action='store_true',
                                                   help='Whether to skip evolutionary constraints during design'
                                                        '\nDefault=%(default)s'),
    ('-nhb', f'--{no_hbnet}'): dict(action='store_true', help='Whether to skip hydrogen bond networks in the design'
                                                              '\nDefault=%(default)s'),
    ('-ntc', f'--{no_term_constraint}'): dict(action='store_true',
                                              help='Whether to skip tertiary motif constraints during design'
                                                   '\nDefault=%(default)s'),
    ('-n', f'--{number_of_trajectories}'): dict(type=int, default=nstruct,
                                                help='How many unique sequences should be generated for each input?'
                                                     '\nDefault=%(default)s'),
    ('-sb', f'--{structure_background}'): dict(action='store_true',
                                               help='Whether to skip all constraints and measure the structure in an '
                                                    'optimal context\nDefault=%(default)s'),
    ('-sc', f'--{scout}'): dict(action='store_true',
                                help='Whether to set up a low resolution scouting protocol to survey designability'
                                     '\nDefault=%(default)s')
}
# ---------------------------------------------------
parser_metrics = dict(interface_metrics=dict(help='Set up RosettaScript to analyze interface metrics from a pose'))
# parser_metrics = subparsers.add_parser(interface_metrics, help='Set up RosettaScript to analyze interface metrics from a pose')
interface_metrics_arguments = {
    ('-sp', '--specific_protocol'): dict(type=str, help='The specific protocol type to perform metrics on')
}
# ---------------------------------------------------
parser_optimize_designs = \
    dict(optimize_designs=dict(help='Optimize and touch up designs after running %s. Useful for '
                                    'reverting\nunnecessary mutations to wild-type, directing exploration of '
                                    'troublesome areas,\nstabilizing an entire design based on evolution, increasing '
                                    'solubility, or modifying\nsurface charge. Optimization is based on amino acid '
                                    'frequency profiles' % interface_design))
# parser_optimize_designs = subparsers.add_parser(optimize_designs, help='Optimize and touch up designs after running interface design. Useful for reverting excess mutations to wild-type, or directing targeted exploration of specific troublesome areas')
optimize_designs_arguments = {
    ('-bg', '--background_profile'): dict(type=str, default=design_profile,
                                          choices=[design_profile, evolutionary_profile, fragment_profile],
                                          help='Which profile should be used as the background profile during '
                                               'optimization\nDefault=%(default)s')
}
# ---------------------------------------------------
# parser_custom = dict(custom_script=dict(help='Set up a custom RosettaScripts.xml for poses.\nThe custom script will '
#                                              'be run in every pose specified using specified options'))
# # parser_custom = subparsers.add_parser('custom_script', help='Set up a custom RosettaScripts.xml for poses. The custom script will be run in every pose specified using specified options')
# parser_custom_script_arguments = {
#     ('-l', '--file_list'): dict(action='store_true',
#                                 help='Whether to use already produced designs in the "designs/" directory'
#                                      '\nDefault=%(default)s'),
#     ('-n', '--native'): dict(type=str, choices=['source', 'asu_path', 'assembly_path', 'refine_pdb', 'refined_pdb',
#                                                 'consensus_pdb', 'consensus_design_pdb'], default='refined_pdb',
#                              help='What structure to use as a "native" structure for Rosetta reference calculations\n'
#                                   'Default=%(default)s'),
#     ('--score_only',): dict(action='store_true', help='Whether to only score the design(s)\nDefault=%(default)s'),
#     ('script',): dict(type=os.path.abspath, help='The location of the custom script'),
#     ('--suffix',): dict(type=str, metavar='SUFFIX',
#                         help='Append to each output file (decoy in .sc and .pdb) the script name (i.e. "decoy_SUFFIX") '
#                              'to identify this protocol. No extension will be included'),
#     ('-v', '--variables'): dict(type=str, nargs='*',
#                                 help='Additional variables that should be populated in the script.\nProvide a list of '
#                                      'such variables with the format "variable1=value variable2=value". Where variable1'
#                                      ' is a RosettaScripts %%%%variable1%%%% and value is a known value. For variables '
#                                      'that must be calculated on the fly for each design, please modify structure.model.py '
#                                      'class to produce a method that can generate an attribute with the specified name')
#     # Todo ' either a known value or an attribute available to the Pose object'
# }
# ---------------------------------------------------
parser_analysis = dict(analysis=dict(help='Analyze all poses specified generating a suite of metrics'))
# parser_analysis = subparsers.add_parser(analysis, help='Analyze all poses specified. %s --guide %s will inform you about the various metrics available to analyze.' % (program_command, analysis))
analysis_arguments = {
    ('--output',): dict(action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to output the --output_file? Use --no-output to prevent'),
    #                          '\nDefault=%(default)s'),
    ('-Of', '--output_file'): dict(type=str, default=analysis_file,
                                   help='Name of the output .csv file containing pose metrics.\nWill be saved to the '
                                        '%s folder of the output\nDefault=%s'
                                        % (all_scores, analysis_file % ('TIMESTAMP', 'LOCATION'))),
    ('--save',): dict(action=argparse.BooleanOptionalAction, default=True,
                      help='Save trajectory information? Use --no-save if False'),  # \nDefault=%(default)s'),
    ('--figures',): dict(action=argparse.BooleanOptionalAction, default=False,
                         help='Create figures for all poses?'),  # \nDefault=%(default)s'),
    ('-j', '--join'): dict(action='store_true', help='Join Trajectory and Residue Dataframes?\nDefault=%(default)s')
}
# ---------------------------------------------------
parser_select_poses = \
    dict(select_poses=dict(help='Select poses based on specific metrics',
                           description='Selection will be the result of a handful of metrics combined using --filter '
                                       'and/or --weights.\nFor metric options see %s --guide. If a pose input option '
                                       'from -d, -f, -p, or -s form isn\'t\nprovided, the flags -sf or -df are possible'
                                       ' where -sf takes priority' % analysis))
# parser_select_poses = subparsers.add_parser(select_poses, help='Select poses based on specific metrics. Selection will be the result of a handful of metrics combined using --filter and/or --weights. For metric options see %s --guide. If a pose specification in the typical d, -f, -p, or -s form isn\'t provided, the arguments -df or -pf are required with -pf taking priority if both provided' % analysis)
select_poses_arguments = {
    ('--filter',): dict(action='store_true', help='Whether to filter pose selection using metrics'),
    (f'--{protocol}',): dict(type=str, default=None, nargs='*', help='Use specific protocol(s) to filter metrics?'),
    ('-n', '--select_number'): dict(type=int, default=sys.maxsize, metavar='int',
                                    help='Number of poses to return\nDefault=No Limit'),
    # ('--prefix',): dict(type=str, metavar='string', help='String to prepend to selection output name'),
    ('--save_total',): dict(action='store_false',
                            help='If --total is used, should the total dataframe be saved?\nDefault=%(default)s'),
    ('--total',): dict(action='store_true',
                       help='Should poses be selected based on their ranking in the total pose pool?\nThis will select '
                            'the top poses based on the average of all designs in\nthat pose for the metrics specified '
                            'unless --protocol is invoked,\nthen the protocol average will be used instead'),
    ('--weight',): dict(action='store_true', help='Whether to weight pose selection results using metrics'
                                                  '\nDefault=%(default)s'),
    ('-wf', '--weight_function'): dict(choices=metric_weight_functions, default='normalize',
                                       help='How to standardize metrics during selection weighting'
                                            '\nDefault=%(default)s'),
# }
# # parser_filter_mutual = parser_select_poses.add_mutually_exclusive_group(required=True)
# parser_select_poses_mutual_group = dict(required=True)
# parser_select_poses_mutual_arguments = {
    ('-m', '--metric'): dict(type=str, choices=['score', 'fragments_matched'],
                             help='If a single metric is sufficient, which metric to sort by?'),
    # ('-pf', '--pose_design_file'): dict(type=str, metavar=ex_path('pose_design.csv'),
    #                                     help='Name of .csv file with (pose, design pairs to serve as sequence selector')
}
# ---------------------------------------------------
parser_select_sequences = dict(select_sequences=dict(help='From the provided poses, generate nucleotide/protein '
                                                          'sequences based on specified selection\ncriteria and '
                                                          'prioritized metrics. Generation of output sequences can take'
                                                          ' multiple forms\ndepending on downstream needs. By default, '
                                                          'disordered region insertion,\ntagging for expression, and '
                                                          'codon optimization (if --nucleotide) are performed'))
# parser_select_sequences = subparsers.add_parser(select_sequences, help='From the provided poses, generate nucleotide/protein sequences based on specified selection criteria and prioritized metrics. Generation of output sequences can take multiple forms depending on downstream needs. By default, disordered region insertion, tagging for expression, and codon optimization (--nucleotide) are performed')
select_sequences_arguments = {
    ('-amp', '--allow_multiple_poses'): dict(action='store_true',
                                             help='Allow multiple sequences to be selected from the same Pose when '
                                                  'using --total\nBy default, --total filters the'
                                                  ' selected sequences by a single Pose\nDefault=%(default)s'),
    ('-ath', '--avoid_tagging_helices'): dict(action='store_true',
                                              help='Should tags be avoided at termini with helices?'
                                                   '\nDefault=%(default)s'),
    ('--csv',): dict(action='store_true', help='Write the sequences file as a .csv instead of the default .fasta'),
    ('--filter',): dict(action='store_true', help='Whether to filter sequence selection using metrics'
                                                  '\nDefault=%(default)s'),
    ('-m', '--multicistronic'): dict(action='store_true',
                                     help='Should nucleotide sequences by output in multicistronic format?\nBy default,'
                                          ' uses the pET-Duet intergeneic sequence containing a T7 promoter, LacO, and '
                                          'RBS\nDefault=%(default)s'),
    ('-ms', '--multicistronic_intergenic_sequence'): dict(type=str,
                                                          help='The sequence to use in the intergenic region of a '
                                                               'multicistronic expression output'),
    ('--nucleotide',): dict(action='store_true', help='Whether to output codon optimized nucleotide sequences'
                                                      '\nDefault=%(default)s'),
    ('-n', '--select_number'): dict(type=int, default=sys.maxsize, metavar='int',
                                    help='Number of sequences to return\nIf total is True, returns the '
                                         'specified number of sequences (Where Default=No Limit).\nOtherwise the '
                                         'specified number will be selected from each pose (Where Default=1/pose)'),
    ('-opt', '--optimize_species'): dict(type=str, default='e_coli',
                                         help='The organism where expression will occur and nucleotide usage should be '
                                              'optimized\nDefault=%(default)s'),
    ('-t', '--preferred_tag'): dict(type=str, choices=expression_tags.keys(), default='his_tag',
                                    help='The name of your preferred expression tag\nDefault=%(default)s'),
    (f'--{protocol}',): dict(type=str, help='Use specific protocol(s) to filter designs?', default=None, nargs='*'),
    ('-ssg', '--skip_sequence_generation'): dict(action='store_true',
                                                 help='Should sequence generation be skipped? Only structures will be '
                                                      'selected\nDefault=%(default)s'),
    # ('--prefix',): dict(type=str, metavar='string', help='String to prepend to selection output name'),
    ('--sequences_per_pose',): dict(type=int, default=1, dest='designs_per_pose',
                                    help='What is the maximum number of sequences that should be selected from '
                                         'each pose?\nDefault=%(default)s'),
    # Todo make work with list... choices=['single', 'all', 'none']
    ('--tag_entities',): dict(type=str, default='none',
                              help='If there are specific entities in the designs you want to tag,\nindicate how '
                                   'tagging should occur. Viable options include:\n\t"single" - a single entity\n\t'
                                   '"all" - all entities\n\t"none" - no entities\n\tcomma separated list such as '
                                   '"1,0,1"\n\t\twhere "1" indicates a tag is required\n\t\tand "0" indicates no tag is'
                                   ' required'),
    ('--save_total',): dict(action='store_false',
                            help='If --total is used, should the total dataframe be saved?\nDefault=%(default)s'),
    ('--total',): dict(action='store_true',
                       help='Should sequences be selected based on their ranking in the total\ndesign pool? Searches '
                            'for the top sequences from all poses,\nthen chooses one sequence/pose unless '
                            '--allow_multiple_poses is invoked\nDefault=%(default)s'),
    ('--weight',): dict(action='store_true', help='Whether to weight sequence selection results using metrics'
                                                  '\nDefault=%(default)s'),
    ('-wf', '--weight_function'): dict(choices=metric_weight_functions, default='normalize',
                                       help='How to standardize metrics during selection weighting'
                                            '\nDefault=%(default)s')
}
# ---------------------------------------------------
parser_select_designs = dict(select_designs=dict(help=f'From the provided poses, select designs based on specified '
                                                      f'selection criteria\nusing metrics. Alias for '
                                                      f'{select_sequences} with --skip_sequence_generation'))
select_designs_arguments = {
    ('-amp', '--allow_multiple_poses'): dict(action='store_true',
                                             help='Allow multiple sequences to be selected from the same Pose when '
                                                  'using --total\nBy default, --total filters the'
                                                  ' selected sequences by a single Pose\nDefault=%(default)s'),
    ('--designs_per_pose',): dict(type=int, default=1,
                                  help='What is the maximum number of sequences that should be selected from '
                                       'each pose?\nDefault=%(default)s'),
    ('--filter',): dict(action='store_true', help='Whether to filter sequence selection using metrics'
                                                  '\nDefault=%(default)s'),
    ('-s', '--select_number'): dict(type=int, default=sys.maxsize, metavar='int',
                                    help='Number of sequences to return\nIf total is True, returns the '
                                         'specified number of sequences (Where Default=No Limit).\nOtherwise the '
                                         'specified number will be selected from each pose (Where Default=1/pose)'),
    (f'--{protocol}',): dict(type=str, help='Use specific protocol(s) to filter designs?', default=None, nargs='*'),
    ('--save_total',): dict(action='store_false',
                            help='If --total is used, should the total dataframe be saved?\nDefault=%(default)s'),
    ('--total',): dict(action='store_true',
                       help='Should sequences be selected based on their ranking in the total\ndesign pool? Searches '
                            'for the top sequences from all poses,\nthen chooses one sequence/pose unless '
                            '--allow_multiple_poses is invoked\nDefault=%(default)s'),
    ('--weight',): dict(action='store_true', help='Whether to weight sequence selection results using metrics'
                                                  '\nDefault=%(default)s'),
    ('-wf', '--weight_function'): dict(choices=metric_weight_functions, default='normalize',
                                       help='How to standardize metrics during selection weighting'
                                            '\nDefault=%(default)s')
}
# ---------------------------------------------------
parser_multicistronic = dict(multicistronic=dict(help='Generate nucleotide sequences for selected designs by codon '
                                                      'optimizing protein\nsequences, then concatenating nucleotide '
                                                      'sequences. REQUIRES an input .fasta file\nspecified with the '
                                                      '-f/--file argument'))
# parser_multicistronic = subparsers.add_parser('multicistronic', help='Generate nucleotide sequences\n for selected designs by codon optimizing protein sequences, then concatenating nucleotide sequences. REQUIRES an input .fasta file specified as -f/--file')
multicistronic_arguments = {
    ('--csv',): dict(action='store_true',
                     help='Write the sequences file as a .csv instead of the default .fasta\nDefault=%(default)s'),
    ('-ms', '--multicistronic_intergenic_sequence'): dict(type=str,
                                                          help='The sequence to use in the intergenic region of a '
                                                               'multicistronic expression output'),
    ('-n', '--number_of_genes'): dict(type=int, help='The number of protein sequences to concatenate into a '
                                                     'multicistronic expression output'),
    ('-opt', '--optimize_species'): dict(type=str, default='e_coli',
                                         help='The organism where expression will occur and nucleotide usage should be '
                                              'optimized\nDefault=%(default)s'),
}
# ---------------------------------------------------
# parser_asu = subparsers.add_parser('find_asu', help='From a symmetric assembly, locate an ASU and save the result.')
# ---------------------------------------------------
parser_check_clashes = dict(check_clashes=dict(help='Check for any clashes in the input poses.\nThis is performed '
                                                    'by default at Pose load and will raise an error if clashes are '
                                                    'found'))
# parser_check_clashes = subparsers.add_parser('check_clashes', help='Check for any clashes in the input poses. This is performed standard in all modules and will return an error if clashes are found')
# ---------------------------------------------------
# parser_check_unmodelled_clashes = subparsers.add_parser('check_unmodelled_clashes', help='Check for clashes between full models. Useful for understanding if loops are missing, whether their modelled density is compatible with the pose')
# ---------------------------------------------------
parser_expand_asu = dict(expand_asu=dict(help='For given poses, expand the asymmetric unit to a symmetric assembly and '
                                              'write the result'))
# parser_expand_asu = subparsers.add_parser('expand_asu', help='For given poses, expand the asymmetric unit to a symmetric assembly and write the result to the design directory.')
# ---------------------------------------------------
parser_generate_fragments = dict(generate_fragments=dict(help='Generate fragment overlap for poses of interest'))
# parser_generate_fragments = subparsers.add_parser(generate_fragments, help='Generate fragment overlap for poses of interest')
# ---------------------------------------------------
parser_rename_chains = dict(rename_chains=dict(help='For given poses, rename the chains in the source PDB to the '
                                                    'alphabetic order.\nUseful for writing a multi-model as distinct '
                                                    'chains or fixing PDB formatting errors'))
# parser_rename_chains = subparsers.add_parser('rename_chains', help='For given poses, rename the chains in the source PDB to the alphabetic order. Useful for writing a multi-model as distinct chains or fixing common PDB formatting errors as well. Writes to design directory')
# # ---------------------------------------------------
# parser_flags = dict(flags=dict(help='Generate a flags file for %s' % program_name))
# # parser_flags = subparsers.add_parser('flags', help='Generate a flags file for %s' % program_name)
# parser_flags_arguments = {
#     ('-t', '--template'): dict(action='store_true', help='Generate a flags template to edit on your own.'),
#     ('-m', '--module'): dict(dest='flags_module', action='store_true',
#                              help='Generate a flags template to edit on your own.')
# }
# # ---------------------------------------------------
# parser_residue_selector = dict(residue_selector=dict(help='Generate a residue selection for %s' % program_name))
# # parser_residue_selector = subparsers.add_parser('residue_selector', help='Generate a residue selection for %s' % program_name)
# ---------------------------------------------------
directory_needed = f'Provide your working {program_output} with -d/--directory to locate poses\nfrom a file utilizing '\
                   f'pose IDs (-df, -pf, and -sf)'
parser_input_group = dict(title=f'{"_" * len(input_title)}\n{input_title}',
                          description=f'\nSpecify where/which poses should be included in processing\n'
                                      f'{directory_needed}')
input_arguments = {
    ('-c', '--cluster_map'): dict(type=os.path.abspath,
                                  help='The location of a serialized file containing spatially\nor interfacial '
                                       'clustered poses'),
    ('-df', '--dataframe'): dict(type=os.path.abspath, metavar=ex_path('Metrics.csv'),
                                 help=f'A DataFrame created by {program_name} analysis containing\npose info. File is '
                                      f'output in .csv format'),
    ('-N', f'--{nano}_output'): dict(action='store_true',
                                     help='Is the input a Nanohedra docking output?\nDefault=%(default)s'),
    ('-pf', '--pose_file'): dict(type=str, dest='specification_file', metavar=ex_path('pose_design_specifications.csv'),
                                 help=f'If pose IDs are specified in a file, say as the result of\n{select_poses} or '
                                      f'{select_designs}'),
    ('-r', '--range'): dict(type=float, default=None, metavar='int-int',
                            help='The range of poses to process from a larger specification.\n'
                                 'Specify a %% between 0 and 100, separating the range by "-"\n'
                                 'Ex: 0-25'),
    ('-sf', '--specification_file'): dict(type=str, metavar=ex_path('pose_design_specifications.csv'),
                                          help='Name of comma separated file with each line formatted:\nposeID, '
                                               '[designID], [residue_number:directive residue_number2-'
                                               'residue_number9:directive ...]')
}
# parser_input_mutual = parser_input.add_mutually_exclusive_group()
parser_input_mutual_group = dict()  # required=True <- adding kwarg below to different parsers depending on need
input_mutual_arguments = {
    ('-d', '--directory'): dict(type=os.path.abspath, metavar=ex_path('your_pdb_files'),
                                help='Master directory where poses to be designed are located. This may be\nthe'
                                     ' output directory from %s.py, a random directory\nwith poses requiring design, or'
                                     ' the output from %s.\nIf the directory of interest resides in a %s directory,\nit'
                                     ' is recommended to use -f, -p, or -s for finer control'
                                     % (nano, program_name, program_output)),
    ('-f', '--file'): dict(type=os.path.abspath, default=None, nargs='*',
                           metavar=ex_path('file_with_pose.paths'),
                           help='File(s) with the location of poses listed. For each run of %s,\na file will be created'
                                'specifying the specific directories to use\nin subsequent commands of the same designs'
                                % program_name),
    ('--fuse_chains',): dict(type=str, nargs='*', default=[],
                             help='The name of a pair of chains to fuse during design.\nPairs should be separated'
                                  ' by a colon, with the n-terminal\npreceding the c-terminal chain new instances by a'
                                  ' space\nEx --fuse_chains A:B C:D'),
    ('-p', '--project'): dict(type=os.path.abspath, nargs='*',
                              metavar=ex_path('SymDesignOutput', 'Projects', 'yourProject'),
                              help='Operate on designs specified within a project(s)'),
    ('-s', '--single'): dict(type=os.path.abspath, nargs='*',
                             metavar=ex_path('SymDesignOutput', 'Projects', 'yourProject', 'single_design[.pdb]'),
                             help='Operate on single pose(s) in a project'),
}
parser_output_group = dict(title=f'{"_" * len(output_title)}\n{output_title}',
                           description='\nSpecify where output should be written')
output_arguments = {
    ('-Oa', '--output_assembly'): dict(action='store_true',
                                       help='Whether the assembly should be output? Infinite materials are output in a '
                                            'unit cell\nDefault=%(default)s'),
    ('-Od', '--outdir', '--output_directory'): dict(type=os.path.abspath, dest='output_directory', default=None,
                                                    help='If provided, the name of the directory to output all created '
                                                         'files.\nOtherwise, one will be generated based on the time, '
                                                         'input, and module'),
    ('-Of', '--output_file'): dict(type=str,  # default=default_path_file,
                                   help='If provided, the name of the output pose file.\nOtherwise, one will be '
                                        'generated based on the time, input, and module'),
    ('-OF', f'--{output_fragments}'): dict(action='store_true',
                                           help='For any fragments generated, write them along with the Pose'),
    ('-Oo', f'--{output_oligomers}'): dict(action='store_true',
                                           help='For any oligomers generated, write them along with the Pose'),
    ('--prefix',): dict(type=str, metavar='string', help='String to prepend to output name'),
    ('--suffix',): dict(type=str, metavar='string', help='String to append to output name'),
}
# If using mutual groups, for the dict "key" (parser name), you must add "_mutual" immediately after the submodule
# string that own the group. i.e nanohedra"_mutual*" indicates nanohedra owns, or interface_design"_mutual*", etc
module_parsers = dict(orient=parser_orient,
                      refine=parser_refine,
                      nanohedra=parser_nanohedra,
                      nanohedra_mutual1=parser_nanohedra_mutual1_group,  # _mutual1,
                      nanohedra_mutual2=parser_nanohedra_mutual2_group,  # _mutual2,
                      nanohedra_mutual_run_type=parser_nanohedra_run_type_mutual_group,  # _mutual,
                      cluster_poses=parser_cluster,
                      interface_design=parser_design,
                      interface_metrics=parser_metrics,
                      optimize_designs=parser_optimize_designs,
                      # custom_script=parser_custom,
                      analysis=parser_analysis,
                      select_poses=parser_select_poses,
                      # select_poses_mutual=parser_select_poses_mutual_group,  # _mutual,
                      select_designs=parser_select_designs,
                      select_sequences=parser_select_sequences,
                      multicistronic=parser_multicistronic,
                      # flags=parser_flags,
                      check_clashes=parser_check_clashes,
                      expand_asu=parser_expand_asu,
                      generate_fragments=parser_generate_fragments,
                      rename_chains=parser_rename_chains,
                      # residue_selector=parser_residue_selector
                      )
input_parsers = dict(input=parser_input_group,
                     input_mutual=parser_input_mutual_group)  # _mutual
output_parsers = dict(output=parser_output_group)
option_parsers = dict(options=parser_options_group)
residue_selector_parsers = dict(residue_selector=parser_residue_selector_group)
parser_arguments = dict(options=options_arguments,
                        residue_selector=residue_selector_arguments,
                        refine=refine_arguments,
                        nanohedra=nanohedra_arguments,
                        nanohedra_mutual1=nanohedra_mutual1_arguments,  # mutually_exclusive_group
                        nanohedra_mutual2=nanohedra_mutual2_arguments,  # mutually_exclusive_group
                        nanohedra_mutual_run_type=nanohedra_run_type_mutual_arguments,  # mutually_exclusive
                        cluster_poses=cluster_poses_arguments,
                        interface_design=interface_design_arguments,
                        interface_metrics=interface_metrics_arguments,
                        optimize_designs=optimize_designs_arguments,
                        analysis=analysis_arguments,
                        select_poses=select_poses_arguments,
                        select_designs=select_designs_arguments,
                        select_sequences=select_sequences_arguments,
                        multicistronic=multicistronic_arguments,
                        input=input_arguments,
                        input_mutual=input_mutual_arguments,  # add_mutually_exclusive_group
                        output=output_arguments,
                        # custom_script_arguments=parser_custom_script_arguments,
                        # select_poses_mutual_arguments=parser_select_poses_mutual_arguments, # mutually_exclusive_group
                        # flags_arguments=parser_flags_arguments,
                        )
parser_options = 'parser_options'
parser_residue_selector = 'parser_residue_selector'
parser_input = 'parser_input'
parser_output = 'parser_output'
parser_module = 'parser_module'
parser_guide = 'parser_guide'
parser_entire = 'parser_entire'
options_argparser = dict(add_help=False, allow_abbrev=False, formatter_class=Formatter)  # Todo? , usage=usage_string)
residue_selector_argparser = \
    dict(add_help=False, allow_abbrev=False, formatter_class=Formatter, usage=usage_string % 'module')
input_argparser = dict(add_help=False, allow_abbrev=False, formatter_class=Formatter, usage=usage_string % 'module')
module_argparser = dict(add_help=False, allow_abbrev=False, formatter_class=Formatter, usage=usage_string % 'module')
guide_argparser = dict(add_help=False, allow_abbrev=False, formatter_class=Formatter, usage=usage_string % 'module')
output_argparser = dict(add_help=False, allow_abbrev=False, formatter_class=Formatter, usage=usage_string % 'module')
argparser_kwargs = dict(parser_options=options_argparser,
                        parser_residue_selector=residue_selector_argparser,
                        parser_input=input_argparser,
                        parser_module=module_argparser,
                        parser_guide=guide_argparser,
                        parser_output=output_argparser,
                        )
# Initialize various independent ArgumentParsers
argparsers: dict[str, argparse.ArgumentParser] = {}
for argparser_name, argparser_args in argparser_kwargs.items():
    argparsers[argparser_name] = argparse.ArgumentParser(**argparser_args)

# Set up module ArgumentParser with module arguments
module_subargparser = dict(title=f'{"_" * len(module_title)}\n{module_title}', dest='module',  # metavar='',
                           description='\nThese are the different modes that designs can be processed. They are'
                                       '\npresented in an order which might be utilized along a design workflow,\nwith '
                                       'utility modules listed at the bottom starting with check_clashes.\nTo see '
                                       'example commands or algorithmic specifics of each Module enter:\n%s\n\nTo get '
                                       'help with Module arguments enter:\n%s' % (submodule_guide, submodule_help))
module_required = ['nanohedra_mutual1']
# for all parsing of module arguments
subparsers = argparsers[parser_module].add_subparsers(**module_subargparser)  # required=True,
# for parsing of guide info
argparsers[parser_guide].add_argument(*guide_args, **guide_kwargs)
argparsers[parser_guide].add_argument(*setup_args, **setup_kwargs)
guide_subparsers = argparsers[parser_guide].add_subparsers(**module_subargparser)
module_suparsers: dict[str, argparse.ArgumentParser] = {}
for parser_name, parser_kwargs in module_parsers.items():
    arguments = parser_arguments.get(parser_name, {})
    # has args as dictionary key (flag names) and keyword args as dictionary values (flag params)
    if 'mutual' in parser_name:  # we must create a mutually_exclusive_group from already formed subparser
        # Remove indication to "mutual" of the argparse group by removing any string after "_mutual"
        exclusive_parser = module_suparsers[parser_name[:parser_name.find('_mutual')]].\
            add_mutually_exclusive_group(**parser_kwargs, **(dict(required=True) if parser_name in module_required
                                                             else {}))
        # add the key word argument "required" to mutual parsers that use it ^
        for args, kwargs in arguments.items():
            exclusive_parser.add_argument(*args, **kwargs)
    else:  # save the subparser in a dictionary to access with mutual groups
        module_suparsers[parser_name] = subparsers.add_parser(prog=usage_string % parser_name,
                                                              # prog=f'python SymDesign.py %(name) '
                                                              #      f'[input_arguments] [optional_arguments]',
                                                              formatter_class=Formatter, allow_abbrev=False,
                                                              name=parser_name, **parser_kwargs[parser_name])
        for args, kwargs in arguments.items():
            module_suparsers[parser_name].add_argument(*args, **kwargs)
        # add each subparser to a guide_subparser as well
        guide_subparser = guide_subparsers.add_parser(name=parser_name, add_help=False, **parser_kwargs[parser_name])
        guide_subparser.add_argument(*guide_args, **guide_kwargs)

# print(module_suparsers['nanohedra'])
# print('after_adding_Args')
# for group in argparsers[parser_module]._action_groups:
#     for arg in group._group_actions:
#         print(arg)

# Set up option ArgumentParser with options arguments
parser = argparsers[parser_options]
option_group = None
for parser_name, parser_kwargs in option_parsers.items():
    arguments = parser_arguments.get(parser_name, {})
    if arguments:
        # has args as dictionary key (flag names) and keyword args as dictionary values (flag params)
        # There are no 'mutual' right now
        if 'mutual' in parser_name:  # only has a dictionary as parser_arguments
            exclusive_parser = option_group.add_mutually_exclusive_group(**parser_kwargs)
            for args, kwargs in arguments.items():
                exclusive_parser.add_argument(*args, **kwargs)
        option_group = parser.add_argument_group(**parser_kwargs)
        for args, kwargs in arguments.items():
            option_group.add_argument(*args, **kwargs)

# Set up residue selector ArgumentParser with residue selector arguments
parser = argparsers[parser_residue_selector]
residue_selector_group = None
for parser_name, parser_kwargs in residue_selector_parsers.items():
    arguments = parser_arguments.get(parser_name, {})
    if arguments:
        # has args as dictionary key (flag names) and keyword args as dictionary values (flag params)
        # There are no 'mutual' right now
        if 'mutual' in parser_name:  # only has a dictionary as parser_arguments
            exclusive_parser = residue_selector_group.add_mutually_exclusive_group(**parser_kwargs)
            for args, kwargs in arguments.items():
                exclusive_parser.add_argument(*args, **kwargs)
        residue_selector_group = parser.add_argument_group(**parser_kwargs)
        for args, kwargs in arguments.items():
            residue_selector_group.add_argument(*args, **kwargs)

# Set up input ArgumentParser with input arguments
parser = argparsers[parser_input]
input_group = None  # must get added before mutual groups can be added
for parser_name, parser_kwargs in input_parsers.items():
    arguments = parser_arguments.get(parser_name, {})
    if 'mutual' in parser_name:  # only has a dictionary as parser_arguments
        exclusive_parser = input_group.add_mutually_exclusive_group(required=True, **parser_kwargs)
        for args, kwargs in arguments.items():
            exclusive_parser.add_argument(*args, **kwargs)
    else:  # has args as dictionary key (flag names) and keyword args as dictionary values (flag params)
        input_group = parser.add_argument_group(**parser_kwargs)
        for args, kwargs in arguments.items():
            input_group.add_argument(*args, **kwargs)

# Set up output ArgumentParser with output arguments
parser = argparsers[parser_output]
output_group = None  # must get added before mutual groups can be added
for parser_name, parser_kwargs in output_parsers.items():
    arguments = parser_arguments.get(parser_name, {})
    if 'mutual' in parser_name:  # only has a dictionary as parser_arguments
        exclusive_parser = output_group.add_mutually_exclusive_group(**parser_kwargs)
        for args, kwargs in arguments.items():
            exclusive_parser.add_argument(*args, **kwargs)
    else:  # has args as dictionary key (flag names) and keyword args as dictionary values (flag params)
        output_group = parser.add_argument_group(**parser_kwargs)
        for args, kwargs in arguments.items():
            output_group.add_argument(*args, **kwargs)

# Set up entire ArgumentParser with all ArgumentParsers
entire_argparser = dict(fromfile_prefix_chars='@', allow_abbrev=False,  # exit_on_error=False, # prog=program_name,
                        description=f'Control all input/output of various {program_name} operations including:'
                                    f'\n\t1. {nano.title()} docking '
                                    '\n\t2. Pose set up, sampling, assembly generation, fragment decoration'
                                    '\n\t3. Interface design using constrained residue profiles and Rosetta'
                                    '\n\t4. Analysis of all designs using metrics'
                                    '\n\t5. Design selection and sequence formatting by combinatorial linear weighting '
                                    'of interface metrics.\n\n'
                                    f'If you\'re a first time user, try "{program_command} --guide"'
                                    '\nAll jobs have built in features for command monitoring & distribution to '
                                    'computational clusters for parallel processing',
                        formatter_class=Formatter, usage=usage_string % 'module',
                        parents=[argparsers.get(parser)
                                 for parser in [parser_module, parser_options, parser_residue_selector, parser_output]])
argparsers[parser_entire] = argparse.ArgumentParser(**entire_argparser)
parser = argparsers[parser_entire]
# Can't set up parser_input via a parent due to mutually_exclusive groups formatting messed up in help.
# Therefore, we repeat the above set-up here...

# Set up entire ArgumentParser with input arguments
input_group = None  # must get added before mutual groups can be added
for parser_name, parser_kwargs in input_parsers.items():
    arguments = parser_arguments.get(parser_name, {})
    if 'mutual' in parser_name:  # only has a dictionary as parser_arguments
        exclusive_parser = input_group.add_mutually_exclusive_group(**parser_kwargs)
        for args, kwargs in arguments.items():
            exclusive_parser.add_argument(*args, **kwargs)
    else:  # has args as dictionary key (flag names) and keyword args as dictionary values (flag params)
        input_group = parser.add_argument_group(**parser_kwargs)
        for args, kwargs in arguments.items():
            input_group.add_argument(*args, **kwargs)

# # can't set up parser_module via a parent due to mutually_exclusive groups formatting messed up in help, repeat above
# # Set up entire ArgumentParser with module arguments
# subparsers = parser.add_subparsers(**module_subargparser)
# entire_module_suparsers: dict[str, argparse.ArgumentParser] = {}
# for parser_name, parser_kwargs in module_parsers.items():
#     arguments = parser_arguments.get(parser_name, {})
#     # has args as dictionary key (flag names) and keyword args as dictionary values (flag params)
#     if 'mutual' in parser_name:  # we must create a mutually_exclusive_group from already formed subparser
#         # remove any indication to "mutual" of the argparse group v
#         exclusive_parser = \
#             entire_module_suparsers['_'.join(parser_name.split('_')[:-1])].add_mutually_exclusive_group(**parser_kwargs)
#         for args, kwargs in arguments.items():
#             exclusive_parser.add_argument(*args, **kwargs)
#     else:  # save the subparser in a dictionary to access with mutual groups
#         entire_module_suparsers[parser_name] = subparsers.add_parser(prog=usage_string % parser_name,
#                                                                      # prog=f'python SymDesign.py %(name) '
#                                                                      #      f'[input_arguments] [optional_arguments]',
#                                                                      formatter_class=Formatter, allow_abbrev=False,
#                                                                      name=parser_name, **parser_kwargs[parser_name])
#         for args, kwargs in arguments.items():
#             entire_module_suparsers[parser_name].add_argument(*args, **kwargs)
