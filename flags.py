from __future__ import annotations

import argparse
import os
import sys
from argparse import _SubParsersAction
from typing import AnyStr

from psutil import cpu_count

from metrics import metric_weight_functions
from utils.path import submodule_guide, submodule_help, force_flags, fragment_dbs, biological_interfaces, \
    sym_entry, program_output, nano_entity_flag1, nano_entity_flag2, data, \
    clustered_poses, interface_design, evolution_constraint, hbnet, term_constraint, number_of_trajectories, \
    structure_background, scout, design_profile, evolutionary_profile, \
    fragment_profile, all_scores, analysis_file, select_sequences, program_name, nano, \
    program_command, analysis, select_poses, output_fragments, output_oligomers, protocol, current_energy_function, \
    ignore_clashes, ignore_pose_clashes, ignore_symmetric_clashes, select_designs, output_structures, rosetta_str, \
    proteinmpnn, output_trajectory, development, consensus, ca_only, sequences, structures, temperatures
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

boolean_positional_prevent_msg = ' Use --no-{} to prevent'.format
"""Use this message in all help keyword arguments using argparse.BooleanOptionalAction with default=True to specify the
 --no- prefix when the argument should be False
"""
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
setup_args = ('--set_up',)
setup_kwargs = dict(action='store_true', help='Show the %(prog)s set up instructions')
default_logging_level = 3
# ---------------------------------------------------
options_description = 'Additional options control symmetry, the extent of file output, various ' \
                      f'{program_name} runtime considerations, and programmatic options for ' \
                      'determining design outcomes'
parser_options = dict(options=dict(description=options_description))
parser_options_group = dict(title=f'{"_" * len(optional_title)}\n{optional_title}',
                            description=f'\n{options_description}')
options_arguments = {
    ('-C', '--cores'): dict(type=int, default=cpu_count(logical=False) - 1,
                            help='Number of cores to use during --multi_processing\nIf run on a cluster, the number of '
                                 'cores will reflect the cluster allocation,\notherwise, will use #physical_cores-1'
                                 '\nDefault=%(default)s'),
    (f'--{ca_only}',): dict(action='store_true',
                            help='Whether a minimal CA variant of the protein should be used for design calculations'),
    ('-D', f'--{development}'): dict(action='store_true',
                                     help='Run in development mode. This should only be used for active development'),
    ('-F', f'--{force_flags}'): dict(action='store_true',
                                     help='Force generation of a new flags file to update script parameters'),
    # ('-gf', f'--{generate_fragments}'): dict(action='store_true',
    #                                          help='Generate interface fragment observations for poses of interest'
    #                                               '\nDefault=%(default)s'),
    guide_args: guide_kwargs,
    ('-i', '--fragment_database'): dict(type=str.lower, choices=fragment_dbs, default=biological_interfaces, metavar='',
                                        help='Database to match fragments for interface specific scoring matrices'
                                             '\nChoices=%(choices)s\nDefault=%(default)s'),
    # ('-ic', f'--{ignore_clashes}'): dict(action=argparse.BooleanOptionalAction, default=False,
    ('-ic', f'--{ignore_clashes}'): dict(action='store_true', help='Whether ANY identified backbone/Cb clash should be '
                                                                   'ignored and allowed to process'),
    ('-ipc', f'--{ignore_pose_clashes}'): dict(action='store_true', help='Whether asu/pose clashes should be '
                                                                         'ignored and allowed to process'),
    ('-isc', f'--{ignore_symmetric_clashes}'): dict(action='store_true', help='Whether symmetric clashes should be '
                                                                              'ignored and allowed to process'),
    ('--log_level',): dict(type=int, default=default_logging_level, choices=set(range(1, 6)),
                           help='What level of log messages should be displayed to stdout?'
                                '\n1-debug, 2-info, 3-warning, 4-error, 5-critical\nDefault=%(default)s'),
    ('--mpi',): dict(type=int, default=0, help='If commands should be run as MPI parallel processes, how many '
                                               'processes\nshould be invoked for each job?\nDefault=%(default)s'),
    ('-M', '--multi_processing'): dict(action='store_true',
                                       help='Should job be run with multiple processors?'),
    ('--overwrite',): dict(action='store_true',
                           help='Whether to overwrite existing structures upon job fulfillment'),
    ('-P', '--preprocessed'): dict(action='store_true',
                                   help=f'Whether the designs of interest have been preprocessed for the '
                                        f'{current_energy_function}\nenergy function and/or missing loops\n'),
    ('-R', '--run_in_shell'): dict(action='store_true',
                                   help="'Should commands be executed at %(prog)s runtime?\nIn most cases, it won't "
                                        "maximize cassini's computational resources.\nAll computation may"
                                        'fail on a single trajectory mistake.\nDefault=%(default)s'),
    setup_args: setup_kwargs,
    (f'--{sequences}',): dict(action='store_true',
                              help='For the protocol, create new sequences for each pose?'),
    (f'--{structures}',): dict(action='store_true',
                               help='Whether the structure of each new sequence should be calculated'),
    ('--skip_logging',): dict(action='store_true',
                              help='Skip logging output to files and direct all logging to stream?'),
    ('-E', f'--{sym_entry}', '--entry'): dict(type=int, default=None, dest=sym_entry,
                                              help=f'The entry number of {nano.title()} docking combinations to use'),
    ('-S', '--symmetry'): dict(type=str, default=None,
                               help='The specific symmetry of the poses of interest.\nPreferably in a composition '
                                    'formula such as T:{C3}{C3}...\nCan also provide the keyword "cryst" to use crystal'
                                    ' symmetry'),
    ('-K', f'--{temperatures}'): dict(type=float, nargs='*', default=(0.1,),
                                      help='Different sampling "temperature(s)", i.e. values greater'
                                           '\nthan 0, to use when performing design. In the form:'
                                           '\nexp(G/T), where G = energy and T = temperature'
                                           '\nHigher temperatures result in more diversity'),
    ('-U', '--update_database'): dict(action='store_true',
                                      help='Whether to update resources for each Structure in the database'),
}
# ---------------------------------------------------
residue_selector_description = 'Residue selectors control which parts of the Pose are included during protocols'
parser_residue_selector = dict(residue_selector=dict(description=residue_selector_description))
parser_residue_selector_group = \
    dict(title=f'{"_" * len(design_selector_title)}\n{design_selector_title}',
         description=f'\n{residue_selector_description}')
residue_selector_arguments = {
    ('--require_design_at_residues',):
        dict(type=str, default=None,
             help='Regardless of participation in an interface, if certain\n residues should be included in'
                  ' design, specify the\nresidue POSE numbers as a comma separated string.\n'
                  'Ex: "23,24,35,41,100-110,267,289-293" Ranges are allowed'),
    ('--require_design_at_chains',):
        dict(type=str, default=None,
             help='Regardless of participation in an interface, if certain\n chains should be included in'
                  " design, specify the\nchain ID's as a comma separated string.\n"
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
             help="If a design should occur ONLY at certain chains, specify\nthe chain ID's as a comma "
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
             help="If a design should NOT occur at certain chains, provide\nthe chain ID's as a comma "
                  'separated string\nEx: "C"')
}
# ---------------------------------------------------
# Set Up SubModule Parsers
# ---------------------------------------------------
# module_parser = argparse.ArgumentParser(add_help=False)  # usage=usage_string,
# subparsers = module_parser.add_subparsers(title='module arguments', dest='module', required=True, description='These are the different modes that designs can be processed. They are\npresented in an order which might be utilized along a design workflow,\nwith utility modules listed at the bottom starting with check_clashes.\nTo see example commands or algorithmic specifics of each Module enter:\n%s\n\nTo get help with Module arguments enter:\n%s' % (submodule_guide, submodule_help))  # metavar='', # description='',
# ---------------------------------------------------
parser_orient = dict(orient=dict(description='Orient a symmetric assembly in a canonical orientation at the origin'))
# parser_orient = subparsers.add_parser('orient', description='Orient a symmetric assembly in a canonical orientation at the origin')
# ---------------------------------------------------
parser_refine = dict(refine=dict(description='Process Structures into an energy function'))
# parser_refine = subparsers.add_parser(refine, description='Process Structures into an energy function')
refine_arguments = {
    ('-ala', '--interface_to_alanine'): dict(action=argparse.BooleanOptionalAction, default=False,
                                             help='Whether to mutate all interface residues to alanine before '
                                                  'refinement'),
    ('-met', '--gather_metrics'): dict(action=argparse.BooleanOptionalAction, default=False,
                                       help='Whether to gather interface metrics for contained interfaces after '
                                            'refinement')
}
# ---------------------------------------------------
parser_nanohedra = dict(nanohedra=dict(description=f'Run or submit jobs to {nano.title()}.py'))
# parser_dock = subparsers.add_parser(nano, description='Run or submit jobs to %s.py.\nUse the Module arguments -c1/-c2, -o1/-o2, or -q to specify PDB Entity codes, building block directories, or query the PDB for building blocks to dock' % nano.title())
nanohedra_arguments = {
    # ('-e', '--entry', f'--{sym_entry}'): dict(type=int, default=None, dest=sym_entry,  # required=True,
    #                                           help=f'The entry number of {nano.title()} docking combinations to use'),
    ('--dock_only',): dict(action=argparse.BooleanOptionalAction, default=False,
                           help='Whether docking should be performed without sequence design'),
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
    ('-Os', '--output_surrounding_uc'): dict(action=argparse.BooleanOptionalAction, default=False,
                                             help='Whether the surrounding unit cells should be output? Only for '
                                                  'infinite materials')
}
parser_nanohedra_run_type_mutual_group = dict()  # required=True <- adding below to different parsers depending on need
nanohedra_run_type_mutual_arguments = {
    ('-e', f'--{sym_entry}', '-entry', '--entry'):
        dict(type=int, default=None, dest=sym_entry, help='The symmetry to use. See --query for possible symmetries'),
    ('-query', '--query',): dict(action='store_true', help='Run in query mode'),
    # Todo alias analysis -metric
    ('-postprocess', '--postprocess',): dict(action='store_true', help='Run in post processing mode')
}
# parser_dock_mutual1 = parser_dock.add_mutually_exclusive_group(required=True)
parser_nanohedra_mutual1_group = dict()  # required=True <- adding kwarg below to different parsers depending on need
nanohedra_mutual1_arguments = {
    ('-c1', '--pdb_codes1'): dict(type=os.path.abspath, default=None,
                                  help=f'File with list of PDB_entity codes for {nano} component 1'),
    ('-o1', f'-{nano_entity_flag1}', f'--{nano_entity_flag1}'):
        dict(type=os.path.abspath, default=None, help=f'Disk location where {nano} component 1 file(s) are located'),
    ('-Q', '--query_codes'): dict(action='store_true', help='Query the PDB API for corresponding codes')
}
# parser_dock_mutual2 = parser_dock.add_mutually_exclusive_group()
parser_nanohedra_mutual2_group = dict()  # required=False
nanohedra_mutual2_arguments = {
    ('-c2', '--pdb_codes2'): dict(type=os.path.abspath, default=None,
                                  help=f'File with list of PDB_entity codes for {nano} component 2'),
    ('-o2', f'-{nano_entity_flag2}', f'--{nano_entity_flag2}'):
        dict(type=os.path.abspath, default=None, help=f'Disk location where {nano} component 2 file(s) are located'),
}
# ---------------------------------------------------
parser_cluster = dict(cluster_poses=dict(description='Cluster all poses by their spatial or interfacial similarity. '
                                                     'This is\nuseful to identify conformationally flexible docked '
                                                     'configurations'))
# parser_cluster = subparsers.add_parser(cluster_poses, description='Cluster all poses by their spatial similarity. This can remove redundancy or be useful in identifying conformationally flexible docked configurations')
cluster_poses_arguments = {
    ('-m', '--mode'): dict(type=str.lower, choices=['transform', 'ialign'], metavar='', default='transform',
                           help='Which type of clustering should be performed?'
                                '\nChoices=%(choices)s\nDefault=%(default)s'),
    ('--as_objects',): dict(action='store_true', help='Whether to store the resulting pose cluster file as '
                                                      'PoseDirectory objects\nDefault stores as pose IDs'),
    ('-Of', '--output_file'): dict(type=str, default=clustered_poses,
                                   help='Name of the output .pkl file containing pose clusters. Will be saved to the '
                                        f'{data.title()} folder of the output.'
                                        f'\nDefault={clustered_poses % ("TIMESTAMP", "LOCATION")}')
}
# ---------------------------------------------------
parser_design = dict(interface_design=dict(description='Gather poses of interest and format for design using sequence '
                                                       'constraints in Rosetta.\nConstrain using evolutionary profiles '
                                                       'of homologous sequences and/or fragment profiles\nextracted '
                                                       'from the PDB or neither'))
# parser_design = subparsers.add_parser(interface_design, description='Gather poses of interest and format for design using sequence constraints in Rosetta. Constrain using evolutionary profiles of homologous sequences and/or fragment profiles extracted from the PDB or neither.')
nstruct = 20
method = 'method'
interface_design_arguments = {
    ('-ec', f'--{evolution_constraint}'): dict(action=argparse.BooleanOptionalAction, default=True,
                                               help='Whether to include evolutionary constraints during design.'
                                                    f'{boolean_positional_prevent_msg(evolution_constraint)}'),
    ('-hb', f'--{hbnet}'): dict(action=argparse.BooleanOptionalAction, default=True,
                                help='Whether to include hydrogen bond networks in the design.'
                                     f'{boolean_positional_prevent_msg(hbnet)}'),
    ('-m', f'--{method}'): dict(type=str.lower, default=proteinmpnn, choices={proteinmpnn, rosetta_str}, metavar='',
                                help='Which design method should be used?\nChoices=%(choices)s\nDefault=%(default)s'),
    ('-n', f'--{number_of_trajectories}'): dict(type=int, default=nstruct,
                                                help='How many unique sequences should be generated for each input?'
                                                     '\nDefault=%(default)s'),
    ('-sb', f'--{structure_background}'): dict(action=argparse.BooleanOptionalAction, default=False,
                                               help='Whether to skip all constraints and measure the structure using '
                                                    'only the selected energy function'),
    ('-sc', f'--{scout}'): dict(action=argparse.BooleanOptionalAction, default=False,
                                help='Whether to set up a low resolution scouting protocol to survey designability'),
    ('-tc', f'--{term_constraint}'): dict(action=argparse.BooleanOptionalAction, default=True,
                                          help='Whether to include tertiary motif constraints during design.'
                                               f'{boolean_positional_prevent_msg(term_constraint)}'),
}
# ---------------------------------------------------
parser_metrics = dict(interface_metrics=
                      dict(description='Set up RosettaScript to analyze interface metrics from a pose'))
# parser_metrics = subparsers.add_parser(interface_metrics, description='Set up RosettaScript to analyze interface metrics from a pose')
interface_metrics_arguments = {
    ('-sp', '--specific_protocol'): dict(type=str, help='The specific protocol type to perform metrics on')
}
# ---------------------------------------------------
parser_optimize_designs = \
    dict(optimize_designs=
         dict(description=f'Optimize and touch up designs after running {interface_design}. Useful for '
                          'reverting\nunnecessary mutations to wild-type, directing exploration of '
                          'troublesome areas,\nstabilizing an entire design based on evolution, increasing '
                          'solubility, or modifying\nsurface charge. Optimization is based on amino acid '
                          'frequency profiles. Use with a --specification_file is suggested'))
# parser_optimize_designs = subparsers.add_parser(optimize_designs, help='Optimize and touch up designs after running interface design. Useful for reverting excess mutations to wild-type, or directing targeted exploration of specific troublesome areas')
optimize_designs_arguments = {
    ('-bg', '--background_profile'): dict(type=str.lower, default=design_profile, metavar='',
                                          choices={design_profile, evolutionary_profile, fragment_profile},
                                          help='Which profile should be used as the background profile during '
                                               'optimization\nChoices=%(choices)s\nDefault=%(default)s')
}
# ---------------------------------------------------
# parser_custom = dict(custom_script=
#                      dict(description='Set up a custom RosettaScripts.xml for poses.\nThe custom script will '
#                                       'be run in every pose specified using specified options'))
# # parser_custom = subparsers.add_parser('custom_script', description='Set up a custom RosettaScripts.xml for poses. The custom script will be run in every pose specified using specified options')
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
parser_analysis = dict(analysis=dict(description='Analyze all poses specified generating a suite of metrics'))
# parser_analysis = subparsers.add_parser(analysis, description='Analyze all poses specified. %s --guide %s will inform you about the various metrics available to analyze.' % (program_command, analysis))
analysis_arguments = {
    ('--output',): dict(action=argparse.BooleanOptionalAction, default=True,
                        help=f'Whether to output the --output_file?{boolean_positional_prevent_msg("output")}'),
    #                          '\nDefault=%(default)s'),
    ('-Of', '--output_file'): dict(type=str, default=analysis_file,
                                   help='Name of the output .csv file containing pose metrics.\nWill be saved to the '
                                        f'{all_scores} folder of the output'
                                        f'\nDefault={analysis_file % ("TIMESTAMP", "LOCATION")}'),
    ('--save',): dict(action=argparse.BooleanOptionalAction, default=True,
                      help=f'Save trajectory information?{boolean_positional_prevent_msg("save")}'),
    ('--figures',): dict(action=argparse.BooleanOptionalAction, default=False,
                         help='Create figures for all poses?'),
    ('-j', '--join'): dict(action='store_true', help='Join Trajectory and Residue Dataframes?')
}
# ---------------------------------------------------
parser_select_poses = \
    dict(select_poses=dict(help='Select poses based on specific metrics',
                           description='Selection will be the result of a handful of metrics combined using --filter '
                                       f'and/or --weights.\nFor metric options see {analysis} --guide. If a pose input '
                                       "option from -d, -f, -p, or -s form isn't\nprovided, the flags -sf or -df are "
                                       'possible where -sf takes priority'))
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
    ('--weight',): dict(action='store_true', help='Whether to weight pose selection results using metrics'),
    ('-wf', '--weight_function'): dict(type=str.lower, choices=metric_weight_functions, default='normalize', metavar='',
                                       help='How to standardize metrics during selection weighting'
                                            '\nChoices=%(choices)s\nDefault=%(default)s'),
# }
# # parser_filter_mutual = parser_select_poses.add_mutually_exclusive_group(required=True)
# parser_select_poses_mutual_group = dict(required=True)
# parser_select_poses_mutual_arguments = {
    ('-m', '--metric'): dict(type=str.lower, choices=['score', 'fragments_matched'], metavar='', default='score',
                             help='If a single metric is sufficient, which metric to sort by?'
                                  '\nChoices=%(choices)s\nDefault=%(default)s'),
    # ('-pf', '--pose_design_file'): dict(type=str, metavar=ex_path('pose_design.csv'),
    #                                     help='Name of .csv file with (pose, design pairs to serve as sequence selector')
}
# ---------------------------------------------------
# Common selection arguments
allow_multiple_poses_args = ('-amp', '--allow_multiple_poses')
allow_multiple_poses_kwargs = dict(action='store_true',
                                   help='Allow multiple sequences to be selected from the same Pose when using --total'
                                        '\nBy default, --total filters the selected sequences by a single Pose')
csv_args = ('--csv',)
csv_kwargs = dict(action='store_true', help='Write the sequences file as a .csv instead of the default .fasta')

filter_args = ('--filter',)
filter_kwargs = dict(action='store_true', help='Whether to filter sequence selection using metrics')
optimize_species_args = ('-opt', '--optimize_species')
# Todo choices=DNAChisel. optimization species options
optimize_species_kwargs = dict(type=str, default='e_coli',
                               help='The organism where expression will occur and nucleotide usage should be '
                                    'optimized\nDefault=%(default)s')
protocol_args = (f'--{protocol}',)
protocol_kwargs = dict(type=str, help='Use specific protocol(s) to filter designs?', default=None, nargs='*')

save_total_args = ('--save_total',)
save_total_kwargs = dict(action='store_false', help='If --total is used, should the total dataframe be saved?')
select_number_args = ('-s', '--select_number')
select_number_kwargs = dict(type=int, default=sys.maxsize, metavar='int',
                            help='Number of sequences to return\nIf total is True, returns the '
                                 'specified number of sequences (Where Default=No Limit).\nOtherwise the '
                                 'specified number will be selected from each pose (Where Default=1/pose)')
total_args = ('--total',)
total_kwargs = dict(action='store_true',
                    help='Should sequences be selected based on their ranking in the total\ndesign pool? Searches '
                         'for the top sequences from all poses,\nthen chooses one sequence/pose unless '
                         '--allow_multiple_poses is invoked')
weight_args = ('--weight',)
weight_kwargs = dict(action='store_true', help='Whether to weight sequence selection results using metrics')
weight_function_args = ('-wf', '--weight_function')
weight_function_kwargs = dict(type=str.lower, choices=metric_weight_functions, default='normalize', metavar='',
                              help='How to standardize metrics during selection weighting'
                                   '\nChoices=%(choices)s\nDefault=%(default)s')
parser_select_sequences = dict(select_sequences=
                               dict(description='From the provided poses, generate nucleotide/protein '
                                                'sequences based on specified selection\ncriteria and '
                                                'prioritized metrics. Generation of output sequences can take'
                                                ' multiple forms\ndepending on downstream needs. By default, '
                                                'disordered region insertion,\ntagging for expression, and '
                                                'codon optimization (if --nucleotide) are performed'))
# parser_select_sequences = subparsers.add_parser(select_sequences, help='From the provided poses, generate nucleotide/protein sequences based on specified selection criteria and prioritized metrics. Generation of output sequences can take multiple forms depending on downstream needs. By default, disordered region insertion, tagging for expression, and codon optimization (--nucleotide) are performed')
select_sequences_arguments = {
    allow_multiple_poses_args: allow_multiple_poses_kwargs,
    ('-ath', '--avoid_tagging_helices'): dict(action='store_true',
                                              help='Should tags be avoided at termini with helices?'),
    csv_args: csv_kwargs,
    filter_args: filter_kwargs,
    ('-m', '--multicistronic'): dict(action='store_true',
                                     help='Should nucleotide sequences by output in multicistronic format?\nBy default,'
                                          ' uses the pET-Duet intergeneic sequence containing\na T7 promoter, LacO, and'
                                          ' RBS'),
    ('-ms', '--multicistronic_intergenic_sequence'): dict(type=str,
                                                          help='The sequence to use in the intergenic region of a '
                                                               'multicistronic expression output'),
    ('--nucleotide',): dict(action='store_true', help='Whether to output codon optimized nucleotide sequences'),
    select_number_args: select_number_kwargs,
    optimize_species_args: optimize_species_kwargs,
    ('-t', '--preferred_tag'): dict(type=str.lower, choices=expression_tags.keys(), default='his_tag', metavar='',
                                    help='The name of your preferred expression tag'
                                         '\nChoices=%(choices)s\nDefault=%(default)s'),
    protocol_args: protocol_kwargs,
    ('-ssg', '--skip_sequence_generation'): dict(action='store_true',
                                                 help='Should sequence generation be skipped? Only structures will be '
                                                      'selected'),
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
    save_total_args: save_total_kwargs,
    total_args: total_kwargs,
    weight_args: weight_kwargs,
    weight_function_args: weight_function_kwargs
}
# ---------------------------------------------------
parser_select_designs = dict(select_designs=
                             dict(description=f'From the provided poses, select designs based on specified '
                                              f'selection criteria\nusing metrics. Alias for '
                                              f'{select_sequences} with --skip_sequence_generation'))
select_designs_arguments = {
    allow_multiple_poses_args: allow_multiple_poses_kwargs,
    ('--designs_per_pose',): dict(type=int, default=1,
                                  help='What is the maximum number of sequences that should be selected from '
                                       'each pose?\nDefault=%(default)s'),
    csv_args: csv_kwargs,
    filter_args: filter_kwargs,
    select_number_args: select_number_kwargs,
    protocol_args: protocol_kwargs,
    save_total_args: save_total_kwargs,
    total_args: total_kwargs,
    weight_args: weight_kwargs,
    weight_function_args: weight_function_kwargs
}
# ---------------------------------------------------
parser_multicistronic = dict(multicistronic=
                             dict(description='Generate nucleotide sequences for selected designs by codon '
                                              'optimizing protein\nsequences, then concatenating nucleotide '
                                              'sequences. REQUIRES an input .fasta file\nspecified with the '
                                              '-f/--file argument'))
# parser_multicistronic = subparsers.add_parser('multicistronic', description='Generate nucleotide sequences\n for selected designs by codon optimizing protein sequences, then concatenating nucleotide sequences. REQUIRES an input .fasta file specified as -f/--file')
multicistronic_arguments = {
    csv_args: csv_kwargs,
    ('-ms', '--multicistronic_intergenic_sequence'): dict(type=str,
                                                          help='The sequence to use in the intergenic region of a '
                                                               'multicistronic expression output'),
    ('-n', '--number_of_genes'): dict(type=int, help='The number of protein sequences to concatenate into a '
                                                     'multicistronic expression output'),
    optimize_species_args: optimize_species_kwargs,
}
# ---------------------------------------------------
# parser_asu = subparsers.add_parser('find_asu', description='From a symmetric assembly, locate an ASU and save the result.')
# ---------------------------------------------------
parser_check_clashes = dict(check_clashes=dict(help='Check for any clashes in the input poses.\nThis is performed '
                                                    'by default at Pose load and will raise an error if clashes are '
                                                    'found'))
# parser_check_clashes = subparsers.add_parser('check_clashes', description='Check for any clashes in the input poses. This is performed standard in all modules and will return an error if clashes are found')
# ---------------------------------------------------
# parser_check_unmodelled_clashes = subparsers.add_parser('check_unmodelled_clashes', description='Check for clashes between full models. Useful for understanding if loops are missing, whether their modelled density is compatible with the pose')
# ---------------------------------------------------
parser_expand_asu = dict(expand_asu=
                         dict(description='For given poses, expand the asymmetric unit to a symmetric assembly and '
                                          'write the result'))
# parser_expand_asu = subparsers.add_parser('expand_asu', description='For given poses, expand the asymmetric unit to a symmetric assembly and write the result to the design directory.')
# ---------------------------------------------------
parser_generate_fragments = dict(generate_fragments=dict(description='Generate fragment overlap for poses of interest'))
# parser_generate_fragments = subparsers.add_parser(generate_fragments, description='Generate fragment overlap for poses of interest')
# ---------------------------------------------------
parser_rename_chains = dict(rename_chains=
                            dict(description='For given poses, rename the chains in the source PDB to the '
                                             'alphabetic order.\nUseful for writing a multi-model as distinct '
                                             'chains or fixing PDB formatting errors'))
# parser_rename_chains = subparsers.add_parser('rename_chains', description='For given poses, rename the chains in the source PDB to the alphabetic order. Useful for writing a multi-model as distinct chains or fixing common PDB formatting errors as well. Writes to design directory')
# # ---------------------------------------------------
# parser_flags = dict(flags=dict(description='Generate a flags file for %s' % program_name))
# # parser_flags = subparsers.add_parser('flags', description='Generate a flags file for %s' % program_name)
# parser_flags_arguments = {
#     ('-t', '--template'): dict(action='store_true', help='Generate a flags template to edit on your own.'),
#     ('-m', '--module'): dict(dest='flags_module', action='store_true',
#                              help='Generate a flags template to edit on your own.')
# }
# # ---------------------------------------------------
# parser_residue_selector = dict(residue_selector=dict(description='Generate a residue selection for %s' % program_name))
# # parser_residue_selector = subparsers.add_parser('residue_selector', description='Generate a residue selection for %s' % program_name)
# ---------------------------------------------------
directory_needed = f'Provide your working {program_output} with -d/--directory to locate poses\nfrom a file utilizing '\
                   f'pose IDs (-df, -pf, and -sf)'
parser_input = dict(input=dict(description='Specify where/which poses should be included in processing\n'
                                           f'{directory_needed}'))
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
    ('-N', f'--{nano}_output'): dict(action='store_true', help='Is the input a Nanohedra docking output?'),
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
                                     f' output directory from {nano}.py, a random directory\nwith poses requiring '
                                     f'design, or the output from {program_name}.\nIf the directory of interest resides'
                                     f' in a {program_output} directory,\nit is recommended to use -f, -p, or -s for '
                                     f'finer control'),
    ('-f', '--file'): dict(type=os.path.abspath, default=None, nargs='*',
                           metavar=ex_path('file_with_pose.paths'),
                           help=f'File(s) with the location of poses listed. For each run of {program_name},\na file '
                                f'will be created specifying the specific directories to use\nin subsequent commands of'
                                f' the same designs'),
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
parser_output = dict(output=dict(description='Specify where output should be written'))
parser_output_group = dict(title=f'{"_" * len(output_title)}\n{output_title}',
                           description='\nSpecify where output should be written')
output_arguments = {
    ('-Oa', '--output_assembly'): dict(action=argparse.BooleanOptionalAction, default=False,
                                       help='Whether the assembly should be output? Infinite materials are output in a '
                                            'unit cell\nDefault=%(default)s'),
    ('-Od', '--outdir', '--output_directory'): dict(type=os.path.abspath, dest='output_directory', default=None,
                                                    help='If provided, the name of the directory to output all created '
                                                         'files.\nOtherwise, one will be generated based on the time, '
                                                         'input, and module'),
    ('-Of', '--output_file'): dict(type=str,  # default=default_path_file,
                                   help='If provided, the name of the output pose file.\nOtherwise, one will be '
                                        'generated based on the time, input, and module'),
    ('-OF', f'--{output_fragments}'): dict(action=argparse.BooleanOptionalAction, default=False,
                                           help='For any fragments generated, write them along with the Pose'),
    ('-Oo', f'--{output_oligomers}'): dict(action=argparse.BooleanOptionalAction, default=False,
                                           help='For any oligomers generated, write them along with the Pose'),
    ('-Os', f'--{output_structures}'): dict(action=argparse.BooleanOptionalAction, default=True,
                                            help=f'For any structures generated, write them'
                                                 f'{boolean_positional_prevent_msg(output_structures)}'),
    ('-Ot', f'--{output_trajectory}'): dict(action=argparse.BooleanOptionalAction, default=False,
                                            help=f'For all structures generated, write them as a single multimodel '
                                                 f'file'),
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
                      input=parser_input,
                      input_mutual=parser_input_mutual_group,
                      output=parser_output,
                      options=parser_options,
                      residue_selector=parser_residue_selector,
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
                                       'example commands or algorithmic specifics of each Module enter:'
                                       f'\n{submodule_guide}\n\nTo get help with Module arguments enter:'
                                       f'\n{submodule_help}')
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

# Separate the provided arguments for modules or overall program arguments to into flags namespaces
design_arguments = {
    ignore_clashes, ignore_pose_clashes, ignore_symmetric_clashes, method, evolution_constraint, hbnet,
    number_of_trajectories, structure_background, scout, term_constraint, consensus, ca_only, temperatures,
    sequences, structures
}
design = {}
"""Contains all the arguments used in design and their default parameters"""
for group in parser._action_groups:
    for arg in group._group_actions:
        if isinstance(arg, _SubParsersAction):  # we have a sup parser, recurse
            for name, sub_parser in arg.choices.items():
                for sub_group in sub_parser._action_groups:
                    for arg in sub_group._group_actions:
                        if arg.dest in design_arguments:
                            design[arg.dest] = arg.default

        elif arg.dest in design_arguments:
            design[arg.dest] = arg.default
