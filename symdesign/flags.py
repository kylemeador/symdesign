from __future__ import annotations

import argparse
import os
import operator
import sys
from typing import Any, AnyStr, Callable, Container, Literal, get_args

import pandas as pd
from psutil import cpu_count

from symdesign.sequence import constants
from symdesign.resources import config
from symdesign.resources.query.utils import input_string, confirmation_string, bool_d, invalid_string, header_string, \
    format_string
from symdesign.utils import ex_path, handle_errors, pretty_format_table, read_json, InputError, remove_digit_table, \
    path as putils
from symdesign.utils.path import fragment_dbs, biological_interfaces, default_logging_level
# These attributes ^ shouldn't be moved here. Below should be with proper handling of '-' vs. '_'
from symdesign.utils.path import submodule_guide, submodule_help, force, sym_entry, program_output, projects, \
    interface_metrics, nano_entity_flag1, nano_entity_flag2, data, multi_processing, residue_selector, options, \
    cluster_poses, orient, default_clustered_pose_file, interface_design, evolution_constraint, hbnet, term_constraint,\
    number_of_designs, refine, structure_background, scout, design_profile, evolutionary_profile, \
    fragment_profile, all_scores, default_analysis_file, select_sequences, program_name, nanohedra, predict_structure, \
    program_command, analysis, select_poses, output_fragments, output_oligomers, protocol, current_energy_function, \
    ignore_clashes, ignore_pose_clashes, ignore_symmetric_clashes, select_designs, output_structures, proteinmpnn, \
    output_trajectory, development, profile, consensus, ca_only, sequences, structures, temperatures, optimize_species,\
    distribute_work, output_directory, output_surrounding_uc, skip_logging, output_file, avoid_tagging_helices, \
    multicistronic, multicistronic_intergenic_sequence, generate_fragments, input_, output, output_assembly, \
    preferred_tag, expand_asu, check_clashes, rename_chains, optimize_designs, perturb_dof, tag_entities, design, \
    default_path_file

design_programs_literal = Literal['consensus', 'proteinmpnn', 'rosetta']
design_programs: tuple[str, ...] = get_args(design_programs_literal)
nstruct = 20
query_codes1 = 'query_codes1'
query_codes2 = 'query_codes2'
nanohedra_output = 'nanohedra_output'
modules = 'modules'
score = 'score'
module = 'module'
method = 'method'
neighbors = 'neighbors'
# dock_only = 'dock_only'
rotation_step1 = 'rotation_step1'
rotation_step2 = 'rotation_step2'
min_matched = 'min_matched'
match_value = 'match_value'
initial_z_value = 'initial_z_value'
only_write_frag_info = 'only_write_frag_info'
proteinmpnn_score = 'proteinmpnn_score'
contiguous_ghosts = 'contiguous_ghosts'
perturb_dof_steps = 'perturb_dof_steps'
perturb_dof_rot = 'perturb_dof_rot'
perturb_dof_tx = 'perturb_dof_tx'
perturb_dof_steps_rot = 'perturb_dof_steps_rot'
perturb_dof_steps_tx = 'perturb_dof_steps_tx'
cluster_map = 'cluster_map'
poses = 'poses'
specific_protocol = 'specific_protocol'
directory = 'directory'
dataframe = 'dataframe'
fragment_database = 'fragment_database'
interface_to_alanine = 'interface_to_alanine'
_metrics = 'metrics'
increment_chains = 'increment_chains'
number = 'number'
nucleotide = 'nucleotide'
as_objects = 'as_objects'
allow_multiple_poses = 'allow_multiple_poses'
project_name = 'project_name'
profile_memory = 'profile_memory'
preprocessed = 'preprocessed'

# Set up JobResources namespaces for different categories of flags
design_namespace = {
    ignore_clashes, ignore_pose_clashes, ignore_symmetric_clashes, method, evolution_constraint, hbnet,
    number_of_designs, structure_background, scout, term_constraint, consensus, ca_only, temperatures,
    sequences, structures, neighbors
}
dock_namespace = {
    proteinmpnn_score, contiguous_ghosts, perturb_dof, perturb_dof_rot, perturb_dof_tx,
    perturb_dof_steps, perturb_dof_steps_rot, perturb_dof_steps_tx, initial_z_value, match_value, min_matched,
    rotation_step1, rotation_step2, score
}
predict_namespace = {
    method
}
cluster_namespace = {
    as_objects, 'map', 'mode', number
}


def format_for_cmdline(flag: str):
    return flag.replace('_', '-')


def format_from_cmdline(flag: str):
    return flag.replace('-', '_')


as_objects = format_for_cmdline(as_objects)
query_codes1 = format_for_cmdline(query_codes1)
query_codes2 = format_for_cmdline(query_codes2)
predict_structure = format_for_cmdline(predict_structure)
cluster_poses = format_for_cmdline(cluster_poses)
generate_fragments = format_for_cmdline(generate_fragments)
fragment_database = format_for_cmdline(fragment_database)
interface_metrics = format_for_cmdline(interface_metrics)
optimize_designs = format_for_cmdline(optimize_designs)
interface_design = format_for_cmdline(interface_design)
residue_selector = format_for_cmdline(residue_selector)
select_designs = format_for_cmdline(select_designs)
select_poses = format_for_cmdline(select_poses)
select_sequences = format_for_cmdline(select_sequences)
check_clashes = format_for_cmdline(check_clashes)
expand_asu = format_for_cmdline(expand_asu)
rename_chains = format_for_cmdline(rename_chains)
evolution_constraint = format_for_cmdline(evolution_constraint)
term_constraint = format_for_cmdline(term_constraint)
number_of_designs = format_for_cmdline(number_of_designs)
structure_background = format_for_cmdline(structure_background)
# design_profile = format_for_cmdline(design_profile)
# evolutionary_profile = format_for_cmdline(evolutionary_profile)
# fragment_profile = format_for_cmdline(fragment_profile)
cluster_map = format_for_cmdline(cluster_map)
specification_file = format_for_cmdline(putils.specification_file)
# poses = format_for_cmdline(poses)
specific_protocol = format_for_cmdline(specific_protocol)
sym_entry = format_for_cmdline(sym_entry)
# dock_only = format_for_cmdline(dock_only)
rotation_step1 = format_for_cmdline(rotation_step1)
rotation_step2 = format_for_cmdline(rotation_step2)
min_matched = format_for_cmdline(min_matched)
match_value = format_for_cmdline(match_value)
initial_z_value = format_for_cmdline(initial_z_value)
only_write_frag_info = format_for_cmdline(only_write_frag_info)
proteinmpnn_score = format_for_cmdline(proteinmpnn_score)
contiguous_ghosts = format_for_cmdline(contiguous_ghosts)
perturb_dof_steps = format_for_cmdline(perturb_dof_steps)
perturb_dof_rot = format_for_cmdline(perturb_dof_rot)
perturb_dof_tx = format_for_cmdline(perturb_dof_tx)
perturb_dof_steps_rot = format_for_cmdline(perturb_dof_steps_rot)
perturb_dof_steps_tx = format_for_cmdline(perturb_dof_steps_tx)
ca_only = format_for_cmdline(ca_only)
perturb_dof = format_for_cmdline(perturb_dof)
distribute_work = format_for_cmdline(distribute_work)
multi_processing = format_for_cmdline(multi_processing)
output_assembly = format_for_cmdline(output_assembly)
output_fragments = format_for_cmdline(output_fragments)
output_oligomers = format_for_cmdline(output_oligomers)
output_structures = format_for_cmdline(output_structures)
output_trajectory = format_for_cmdline(output_trajectory)
output_directory = format_for_cmdline(output_directory)
output_file = format_for_cmdline(output_file)
output_surrounding_uc = format_for_cmdline(output_surrounding_uc)
ignore_clashes = format_for_cmdline(ignore_clashes)
ignore_pose_clashes = format_for_cmdline(ignore_pose_clashes)
ignore_symmetric_clashes = format_for_cmdline(ignore_symmetric_clashes)
nano_entity_flag1 = format_for_cmdline(nano_entity_flag1)
nano_entity_flag2 = format_for_cmdline(nano_entity_flag2)
skip_logging = format_for_cmdline(skip_logging)
interface_to_alanine = format_for_cmdline(interface_to_alanine)
increment_chains = format_for_cmdline(increment_chains)
tag_entities = format_for_cmdline(tag_entities)
optimize_species = format_for_cmdline(optimize_species)
avoid_tagging_helices = format_for_cmdline(avoid_tagging_helices)
preferred_tag = format_for_cmdline(preferred_tag)
multicistronic_intergenic_sequence = format_for_cmdline(multicistronic_intergenic_sequence)
allow_multiple_poses = format_for_cmdline(allow_multiple_poses)
project_name = format_for_cmdline(project_name)
profile_memory = format_for_cmdline(profile_memory)


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
        print('For a %s run, the following flags can be used to customize the design. You can include these in'
              ' a flags file\nsuch as "my_design.flags" specified on the command line like "@my_design.flags" or '
              'manually by typing them in.\nTo automatically generate a flags file template, run "%s flags --template"'
              ' then modify the defaults.\nAlternatively, input the number(s) corresponding to the flag(s) of '
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
                arg_value = input('\tFor "%s" what %s value should be used? Default is "%s"%s'
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


# Put longer operations first so that they are scanned first, then shorter ones i.e. '>=' then '>'
def not_contains(__a: Container[object], __b: object) -> bool:
    return operator.not_(operator.contains(__a, __b))


viable_operations = {
    # '+': operator.add, '-': operator.sub,
    # '*': operator.mul, '@': operator.matmul,
    # '/': operator.truediv, '//': operator.floordiv,
    '!->': not_contains,
    '->': operator.contains,  # value in metric
    '>=': operator.ge,  # '=>': operator.ge,
    '<=': operator.le,  # '=<': operator.le,
    '!=': operator.ne,  # '=!': operator.ne,
    '=': operator.eq,
    '>': operator.gt,
    '<': operator.lt,
}
inverse_operations = {
    # '+': operator.add, '-': operator.sub,
    # '*': operator.mul, '@': operator.matmul,
    # '/': operator.truediv, '//': operator.floordiv,
    # '->': operator.contains,  # value in metric
    '>=': operator.lt,  # '=>': operator.ge,
    '<=': operator.gt,  # '=<': operator.le,
    # '!=': operator.eq,  # '=!': operator.ne,
    # '=': operator.ne,
    '>': operator.le,
    '<': operator.ge,
    # '!': operator.not_,
}
operator_strings = {
    not_contains: '!->',
    operator.contains: '->',
    operator.ge: '>=',
    operator.le: '<=',
    operator.ne: '!=',
    operator.eq: '=',
    operator.gt: '>',
    operator.lt: '<',
}


def parse_weights(weights: list[str] = None, file: AnyStr = None) \
        -> dict[str, list[tuple[Callable, Callable, dict, Any]]]:
    """

    Args:
        weights:
        file:

    Returns:

    """
    parsed_weights = parse_filters(weights, file=file)
    # Ensure proper formatting of weight specific parameters
    for idx, (metric_name, weight) in enumerate(parsed_weights.items()):
        if len(weight) != 1:
            raise InputError(f"Can't assign more than one weight for every provided metric. "
                             f"'{weights[idx]}' is invalid")
        operation, pre_operation, pre_kwargs, value = weight[0]
        if operation != operator.eq:
            raise InputError(f"Can't assign a selection weight with the operator '{operator_strings[operation]}'. "
                             f"'{weights[idx]}' is invalid")
        if isinstance(value, str):
            raise InputError(f"Can't assign a numerical weight to the provided weight '{weights[idx]}'")

    return parsed_weights


def parse_filters(filters: list[str] = None, file: AnyStr = None) \
        -> dict[str, list[tuple[Callable, Callable, dict, Any]]]:
    """Given a command line specified set of metrics and values, parse into filters to sort Pose DataFrames accordingly

    Args:
        filters: The command line collected filter arguments as specified in the filters --help
        file: The path to a file specifying filters in JSON as specified in the filters --help
    Returns:
        The parsed filter mapping linking each metric to a specified filtering operation
    """
    def null(df, **_kwargs):
        """Do nothing and return a passed DataFrame"""
        return df

    if file is not None:
        parsed_filters = read_json(file)
    else:  # Parse input filters as individual filtering directives
        parsed_filters = {}
        for filter_str in filters:
            # Make an additional variable to substitute operations as they are found
            _filter_str = filter_str
            # Find the indices of each operation and save to use as slices
            indices = []
            operations_syntax = []
            for syntax in viable_operations:
                # if syntax in filter_str:
                while syntax in _filter_str:
                    f_index = _filter_str.find(syntax)
                    # It is important that shorter operations do not pull out longer ones i.e. '>' and '>='
                    # # Check if equals is the next character and set operation_length
                    # if filter_str[f_index + 1] == '=':
                    #     operation_length = 2
                    # else:
                    #     operation_length = 1
                    operation_length = len(syntax)
                    r_index = f_index + operation_length
                    # r_index = filter_str.rfind(syntax)
                    indices.append([f_index, r_index])
                    operations_syntax.append(syntax)
                    # Substitute out the found operation for '`'
                    _filter_str = _filter_str[:f_index] + '`' * operation_length + _filter_str[r_index:]

            # Sort the indices according to the first index number
            full_indices = []
            for idx_pair in sorted(indices, key=lambda pair: pair[0]):
                full_indices.extend(idx_pair)

            # Separate the filter components along the operations
            unique_components = [filter_str[:full_indices[0]]] \
                + [filter_str[f_index:r_index] for f_index, r_index in zip(full_indices[:-1], full_indices[1:])] \
                + [filter_str[full_indices[-1]:]]

            # Set up function to parse values properly
            def extract_format_value(_value: Any) -> Any:
                """Values can be of type int, float, or string. Further, strings could be a list of int/float"""
                try:
                    formatted_value = int(_value)
                except ValueError:  # Not simply an integer
                    try:
                        formatted_value = float(_value)
                    except ValueError:  # Not numeric
                        # This is either a list of numerics or likely a string value
                        _values = [_component.strip() for _component in _value.split(',')]
                        if len(_values) > 1:
                            # We have a list of values
                            formatted_value = [extract_format_value(_value_) for _value_ in _values]
                        else:
                            # This is some type of string value
                            formatted_value = _values[0]
                            # if _value[-1] == '%':
                            #     # This should be treated as a percentage
                            #     formatted_value = extract_format_value(_value)

                return formatted_value

            # Find the values that are metrics and those that are values, then ensure they are parsed correct
            metric = metric_specs = metric_idx = None
            invert_ops = True
            operations_syntax_iter = iter(operations_syntax)
            parsed_values = []
            parsed_operations = []
            parsed_percents = []
            for idx, component in enumerate(unique_components):
                # Value, operation, value, operation, value
                if idx % 2 == 0:  # Zero or even index which must contain metric/values
                    # Substitute any numerical characters to test if the provided component is a metric
                    substituted_component = component.translate(remove_digit_table)
                    _metric_specs = config.metrics.get(substituted_component)
                    if _metric_specs:  # We found in the list of available program metrics
                        if metric_idx is None:
                            metric_specs = _metric_specs
                            metric = component
                            metric_idx = idx
                            # if idx != 0:
                            # We must negate operations until the metric is found, i.e. metric > 1 is expected, but
                            # 1 < metric < 10 is allowed. This becomes metric > 1 and metric < 10
                            invert_ops = False
                            continue
                        else:  # We found two metrics...
                            raise ValueError(f"Can't accept more than one value per filter as of now")
                    else:  # Either a value or bad metric name
                        if component[-1] == '%':
                            # This should be treated as a percentage. Remove and parse
                            component = component[:-1]
                            parsed_percents.append(True)
                        else:
                            parsed_percents.append(False)
                        component = extract_format_value(component)

                        parsed_values.append(component)
                    # # This may be required if section is reworked to allow more than one value per filter
                    # parsed_values.append(component)
                else:  # This is an operation
                    syntax = next(operations_syntax_iter)
                    if invert_ops:
                        operation = inverse_operations.get(syntax, viable_operations[syntax])
                    else:
                        operation = viable_operations[syntax]
                    parsed_operations.append(operation)

            if metric_idx is None:
                possible_metrics = []
                for value in parsed_values:
                    if isinstance(value, str):
                        possible_metrics.append(value)
                # Todo find the closest and report to the user!
                raise InputError(f"Couldn't coerce {'any of' if len(possible_metrics) > 1 else ''} '"
                                 f"{', '.join(possible_metrics)}' to a viable metric")

            # Format the metric for use by select-* protocols
            filter_specification = []
            for value, percent, operation in zip(parsed_values, parsed_percents, parsed_operations):
                if percent:
                    # pre_operation = pd.DataFrame.sort_values
                    pre_operation = pd.DataFrame.rank
                    pre_kwargs = dict(
                        # Whether to sort dataframe with ascending, bigger values (higher rank) on top. Default True
                        # ascending=metric_specs['direction'],
                        # Will determine how ties are sorted which depends on the directional preference of the filter
                        # 'min' makes equal values assume lower percent, 'max' higher percent
                        method=metric_specs['direction'],
                        # Treat the rank as a percent so that a percentage can be used to filter
                        pct=True
                    )
                else:
                    pre_operation = null
                    pre_kwargs = {}

                filter_specification.append((operation, pre_operation, pre_kwargs, value))

            parsed_filters[metric] = filter_specification

    return parsed_filters


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
#  {'refine': ArgumentParser(prog='python {program_exe} module [module_arguments] [input_arguments]'
#                                 '[optional_arguments] refine'

boolean_positional_prevent_msg = 'Use --no-{} to prevent'.format
"""Use this message in all help keyword arguments using argparse.BooleanOptionalAction with default=True to specify the
 --no- prefix when the argument should be False
"""
optional_title = 'Optional Arguments'
design_selector_title = 'Design Selector Arguments'
input_title = 'Input Arguments'
output_title = 'Output Arguments'
module_title = 'Module Arguments'
usage_str = f'\n      python {program_name}.py module [{module_title.lower()}][{input_title.lower()}]' \
               f'[{output_title.lower()}][{design_selector_title.lower()}][{optional_title.lower()}]'
module_usage_str = f'\n      python {program_name}.py %s [{input_title.lower()}]' \
                   f'[{output_title.lower()}][{design_selector_title.lower()}][{optional_title.lower()}]'

# Reused arguments
distribute_args = ('-D', f'--{distribute_work}')
evolution_constraint_args = ('-ec', f'--{evolution_constraint}')
evolution_constraint_kwargs = dict(action=argparse.BooleanOptionalAction, default=True,
                                   help='Whether to include evolutionary constraints during design.\n'
                                        f'{boolean_positional_prevent_msg(evolution_constraint)}')
term_constraint_args = ('-tc', f'--{term_constraint}')
term_constraint_kwargs = dict(action=argparse.BooleanOptionalAction, default=True,
                              help='Whether to include tertiary motif constraints during design.\n'
                                   f'{boolean_positional_prevent_msg(term_constraint)}')
guide_args = ('--guide',)
guide_kwargs = dict(action='store_true', help=f'Display the {program_name}/module specific guide\nEx:'
                                              f' "{program_command} --guide"\nor "{submodule_guide}"')
output_directory_args = ('-Od', f'--{output_directory}', '--outdir')
output_file_args = ('-Of', f'--{output_file}')
setup_args = ('--setup',)
setup_kwargs = dict(action='store_true', help='Show the %(prog)s set up instructions')
sym_entry_args = ('-E', f'--{sym_entry}', '--entry', '-entry')
sym_entry_kwargs = dict(type=int, default=None, metavar='INT',
                        help=f'The entry number of {nanohedra.title()} docking combinations to use.\n'
                             f'See {nanohedra} --query for possible symmetries')
# ---------------------------------------------------
options_help = f'Additional options control symmetry, the extent of file output,\nvarious runtime ' \
               'considerations, and miscellaneous programmatic options'
parser_options = {options: dict(description=options_help, help=options_help)}
parser_options_group = dict(title=f'{"_" * len(optional_title)}\n{optional_title}',
                            description=f'\n{options_help}')
options_arguments = {
    ('-C', '--cores'): dict(type=int, default=cpu_count(logical=False) - 1, metavar='INT',
                            help=f'Number of cores to use with flag --{multi_processing}\n'
                                 'If run on a cluster, cores will reflect the cluster allocation,\n'
                                 'otherwise, will use #physical_cores-1\nDefault=%(default)s'),
    ('--database',): dict(action=argparse.BooleanOptionalAction, default=True,
                          help=f'Whether to utilize the SQL database for intermediate result processing\n'
                               f'{boolean_positional_prevent_msg("database")}'),
    (f'--{development}',): dict(action='store_true',
                                help='Run in development mode. This should only be used for active development'),
    distribute_args: dict(action='store_true',
                          help="Should commands be distributed to a cluster?\nIn most cases, this will "
                               'maximize computational resources\nDefault=%(default)s'),
    ('-F', f'--{force}'): dict(action='store_true', help='Force generation of new files for existing projects'),
    # ('-gf', f'--{generate_fragments}'): dict(action='store_true',
    #                                          help='Generate interface fragment observations for poses of interest'
    #                                               '\nDefault=%(default)s'),
    guide_args: guide_kwargs,
    ('-i', f'--{fragment_database}'): dict(type=str.lower, choices=fragment_dbs, default=biological_interfaces,
                                           metavar='STR',
                                           help='Database to match fragments for interface specific scoring matrices'
                                                '\nChoices=%(choices)s\nDefault=%(default)s'),
    # ('-ic', f'--{ignore_clashes}'): dict(action=argparse.BooleanOptionalAction, default=False,
    ('-ic', f'--{ignore_clashes}'):
        dict(action='store_true',
             help='Whether ANY identified backbone/Cb clashes should be ignored and allowed to process'),
    ('-ipc', f'--{ignore_pose_clashes}'):
        dict(action='store_true', help='Whether asu/pose clashes should be ignored and allowed to process'),
    ('-isc', f'--{ignore_symmetric_clashes}'):
        dict(action='store_true', help='Whether symmetric clashes should be ignored and allowed to process'),
    ('--log-level',): dict(type=int, default=default_logging_level, choices=set(range(1, 6)),
                           help='What level of log messages should be displayed to stdout?'
                                '\n1-debug, 2-info, 3-warning, 4-error, 5-critical\nDefault=%(default)s'),
    ('--mpi',): dict(type=int, metavar='INT',
                     help='If commands should be run as MPI parallel processes, how many '
                          'processes\nshould be invoked for each job?\nDefault=%(default)s'),
    (f'--{project_name}',): dict(type=str, metavar='STR',
                                 help='If desired, the name of the initialized project\nDefault is inferred from file'),
    ('-M', f'--{multi_processing}'): dict(action='store_true', help='Should job be run with multiple processors?'),
    (f'--{profile_memory}',): dict(action='store_true',
                                   help='memory_profiler.profile() a module. Must be run with --development'),
    setup_args: setup_kwargs,
    (f'--{skip_logging}',): dict(action='store_true',
                                 help='Skip logging output to files and direct all logging to stream?'),
    sym_entry_args: sym_entry_kwargs,
    ('-S', '--symmetry'): dict(type=str, default=None, metavar='RESULT:{GROUP1}{GROUP2}...',
                               help='The specific symmetry of the poses of interest.\nPreferably in a composition '
                                    'formula such as T:{C3}{C3}...\nCan also provide the keyword "cryst" to use crystal'
                                    ' symmetry'),
    ('-K', f'--{temperatures}'): dict(type=float, nargs='*', default=(0.1,), metavar='FLOAT',
                                      help='Different sampling "temperature(s)", i.e. values greater'
                                           '\nthan 0, to use when performing design. In the form:'
                                           '\nexp(G/T), where G = energy and T = temperature'
                                           '\nHigher temperatures result in more diversity'),
    ('-U', '--update-database'): dict(action='store_true',
                                      help='Whether to update resources for each Structure in the database')
}
# ---------------------------------------------------
residue_selector_help = 'Residue selectors control which parts of the Pose are included during protocols'
parser_residue_selector = {residue_selector: dict(description=residue_selector_help, help=residue_selector_help)}
parser_residue_selector_group = dict(title=f'{"_" * len(design_selector_title)}\n{design_selector_title}',
                                     description=f'\n{residue_selector_help}')
residue_selector_arguments = {
    ('--require-design-at-residues',):
        dict(type=str, default=None,
             help='Regardless of participation in an interface, if certain\nresidues should be included in'
                  ' design, specify the\nresidue POSE numbers as a comma separated string.\n'
                  'Ex: "23,24,35,41,100-110,267,289-293" Ranges are allowed'),
    ('--require-design-at-chains',):
        dict(type=str, default=None,
             help='Regardless of participation in an interface, if certain\nchains should be included in'
                  " design, specify the\nchain ID's as a comma separated string.\n"
                  'Ex: "A,D"'),
    # ('--select-designable-residues-by-sequence',):
    #     dict(type=str, default=None,
    #          help='If design should occur ONLY at certain residues, specify\nthe location of a .fasta file '
    #               f'containing the design selection\nRun "{program_command} --single my_pdb_file.pdb design_selector"'
    #               ' to set this up'),
    ('--select-designable-residues-by-pdb-number',):
        dict(type=str, default=None, metavar=None,
             help='If design should occur ONLY at certain residues, specify\nthe residue PDB number(s) '
                  'as a comma separated string\nRanges are allowed Ex: "40-45,170-180,227,231"'),
    ('--select-designable-residues-by-pose-number',):
        dict(type=str, default=None,
             help='If design should occur ONLY at certain residues, specify\nthe residue POSE number(s) '
                  'as a comma separated string\nRanges are allowed Ex: "23,24,35,41,100-110,267,289-293"'),
    ('--select-designable-chains',):
        dict(type=str, default=None,
             help="If a design should occur ONLY at certain chains, specify\nthe chain ID's as a comma "
                  'separated string\nEx: "A,C,D"'),
    # ('--mask-designable-residues-by-sequence',):
    #     dict(type=str, default=None,
    #          help='If design should NOT occur at certain residues, specify\nthe location of a .fasta file '
    #               f'containing the design mask\nRun "{program_command} --single my_pdb_file.pdb design_selector" '
    #               'to set this up'),
    ('--mask-designable-residues-by-pdb-number',):
        dict(type=str, default=None,
             help='If design should NOT occur at certain residues, specify\nthe residue PDB number(s) '
                  'as a comma separated string\nEx: "27-35,118,281" Ranges are allowed'),
    ('--mask-designable-residues-by-pose-number',):
        dict(type=str, default=None,
             help='If design should NOT occur at certain residues, specify\nthe residue POSE number(s) '
                  'as a comma separated string\nEx: "27-35,118,281" Ranges are allowed'),
    ('--mask-designable-chains',):
        dict(type=str, default=None,
             help="If a design should NOT occur at certain chains, provide\nthe chain ID's as a comma "
                  'separated string\nEx: "C"')
}
# ---------------------------------------------------
# Set Up SubModule Parsers
# ---------------------------------------------------
# module_parser = argparse.ArgumentParser(add_help=False)  # usage=usage_str,
# ---------------------------------------------------
protocol_help = 'Perform a series of modules in a specified order'
parser_protocol = {protocol: dict(description=protocol_help, help=protocol_help)}
protocol_arguments = {
    ('-m', f'--{modules}'): dict(nargs='*', help='The modules to run in order'),
}
# ---------------------------------------------------
predict_structure_help = 'Predict the 3D structure from specified sequence(s)'
parser_predict_structure = \
    {predict_structure: dict(description=predict_structure_help, help=predict_structure_help)}
predict_structure_arguments = {
    ('-m', f'--{method}'f'--predict-{method}'):
        dict(choices={'thread'}, default='thread',
             help=f'The method utilized to {predict_structure}\nChoices=%(choices)s\nDefault=%(default)s'),
}
# ---------------------------------------------------
orient_help = 'Orient a symmetric assembly in a canonical orientation at the origin'
parser_orient = {orient: dict(description=orient_help, help=orient_help)}
# ---------------------------------------------------
refine_help = 'Process Structures into an energy function'
parser_refine = {refine: dict(description=refine_help, help=refine_help)}
refine_arguments = {
    ('-ala', f'--{interface_to_alanine}'): dict(action=argparse.BooleanOptionalAction, default=False,
                                                help='Whether to mutate all interface residues to alanine before '
                                                     'refinement'),
    ('-met', f'--{_metrics}'): dict(action=argparse.BooleanOptionalAction, default=True,
                                    help='Whether to calculate metrics for contained interfaces after refinement\n'
                                         f'{boolean_positional_prevent_msg(_metrics)}')
}
# ---------------------------------------------------
nanohedra_help = f'Run {nanohedra.title()}.py'
parser_nanohedra = {nanohedra: dict(description=nanohedra_help, help=nanohedra_help)}
default_perturbation_steps = 3
nanohedra_arguments = {
    (f'--{contiguous_ghosts}',): dict(action=argparse.BooleanOptionalAction, default=False,
                                      help='Whether to prioritize docking with ghost fragments that form continuous'
                                           '\nsegments on a single component'),
    # (f'--{dock_only}',): dict(action=argparse.BooleanOptionalAction, default=False,
    #                           help='Whether docking should be performed without sequence design'),
    evolution_constraint_args: evolution_constraint_kwargs,
    ('-iz', f'--{initial_z_value}'): dict(type=float, default=1.,
                                          help='The acceptable standard deviation z score for initial fragment overlap '
                                               'identification.\nSmaller values lead to more stringent matching '
                                               'criteria\nDefault=%(default)s'),
    ('-mv', f'--{match_value}'):
        dict(type=float, default=0.5, dest='match_value',
             help='What is the minimum match score required for a high quality fragment?'),
    (f'--{min_matched}',): dict(type=int, default=3,
                                help='How many high quality fragment pairs should be present before a pose is '
                                     'identified?\nDefault=%(default)s'),
    (f'--{score}',): dict(type=str, choices={'nanohedra', 'proteinmpnn'},  # default='nanohedra'
                          help='Which metric should be used to rank output poses?\nDefault=%(default)s'),
    (f'--{only_write_frag_info}',): dict(action=argparse.BooleanOptionalAction, default=False,
                                         help='Used to write fragment information to a directory for C1 based docking'),
    output_directory_args:
        dict(type=os.path.abspath, default=None,
             help='Where should the output be written?\nDefault='
                  f'{ex_path(program_output, projects, "NanohedraEntry[ENTRYNUMBER]_[BUILDING-BLOCKS]_Poses")}'),
    (f'--{perturb_dof}',): dict(action=argparse.BooleanOptionalAction, default=False,
                                help='Whether the degrees of freedom should be finely sampled during\n by perturbing '
                                     'found transformations and repeating docking iterations'),
    (f'--{perturb_dof_rot}',): dict(action=argparse.BooleanOptionalAction, default=False,
                                    help='Whether the rotational degrees of freedom should be finely sampled in\n'
                                         'subsequent docking iterations'),
    (f'--{perturb_dof_tx}',): dict(action=argparse.BooleanOptionalAction, default=False,
                                   help='Whether the translational degrees of freedom should be finely sampled in\n'
                                        'subsequent docking iterations'),
    (f'--{perturb_dof_steps}',): dict(type=int, metavar='INT',
                                      help='How many dof steps should be used during subsequent docking iterations.\n'
                                           f'For each DOF, a total of --{perturb_dof_steps} will be sampled during '
                                           f'perturbation\nDefault={default_perturbation_steps}'),
    (f'--{perturb_dof_steps_rot}',): dict(type=int, metavar='INT',
                                          help='How many rotational dof steps should be used during perturbations\n'
                                               f'Default={default_perturbation_steps}'),
    (f'--{perturb_dof_steps_tx}',): dict(type=int, metavar='INT',
                                         help='How many translational dof steps should be used during perturbations\n'
                                              f'Default={default_perturbation_steps}'),
    (f'--{proteinmpnn_score}',): dict(action=argparse.BooleanOptionalAction, default=False,
                                      help='Whether docking fit should be measured using ProteinMPNN'),
    ('-r1', f'--{rotation_step1}'): dict(type=float, default=3.,
                                         help='The number of degrees to increment the rotational degrees of freedom '
                                              'search\nDefault=%(default)s'),
    ('-r2', f'--{rotation_step2}'): dict(type=float, default=3.,
                                         help='The number of degrees to increment the rotational degrees of freedom '
                                              'search\nDefault=%(default)s'),
}
query_codes_kwargs = dict(action='store_true', help='Query the PDB API for corresponding codes')
parser_nanohedra_run_type_mutual_group = dict()  # required=True <- adding below to different parsers depending on need
nanohedra_run_type_mutual_arguments = {
    sym_entry_args: sym_entry_kwargs,
    ('-query', '--query',): dict(action='store_true', help='Run in query mode'),
    # # Todo alias analysis -metric
    # ('-postprocess', '--postprocess',): dict(action='store_true', help='Run in post processing mode')
}
# parser_dock_mutual1 = parser_dock.add_mutually_exclusive_group(required=True)
parser_nanohedra_mutual1_group = dict()  # required=True <- adding kwarg below to different parsers depending on need
nanohedra_mutual1_arguments = {
    ('-c1', '--pdb-codes1'): dict(type=os.path.abspath, default=None,
                                  help=f'File with list of PDB_entity codes for {nanohedra} component 1'),
    ('-o1', f'-{nano_entity_flag1}', f'--{nano_entity_flag1}'):
        dict(type=os.path.abspath, default=None,
             help=f'Disk location where {nanohedra} component 1 file(s) are located'),
    ('-Q1', f'--{query_codes1}'): query_codes_kwargs
}
# parser_dock_mutual2 = parser_dock.add_mutually_exclusive_group()
parser_nanohedra_mutual2_group = dict()  # required=False
nanohedra_mutual2_arguments = {
    ('-c2', '--pdb-codes2'): dict(type=os.path.abspath, default=None,
                                  help=f'File with list of PDB_entity codes for {nanohedra} component 2'),
    ('-o2', f'-{nano_entity_flag2}', f'--{nano_entity_flag2}'):
        dict(type=os.path.abspath, default=None,
             help=f'Disk location where {nanohedra} component 2 file(s) are located'),
    ('-Q2', f'--{query_codes2}'): query_codes_kwargs
}
# ---------------------------------------------------
cluster_map_args = ('-c', f'--{cluster_map}')
cluster_map_kwargs = dict(type=os.path.abspath, dest='map',
                          metavar=ex_path(default_clustered_pose_file.format('TIMESTAMP', 'LOCATION')),
                          help='The location of a serialized file containing spatially\nor interfacial '
                               'clustered poses')
number_args = ('-n', f'--{number}')  # '--select-number')
cluster_poses_help = 'Cluster all poses by their spatial or interfacial similarity. This is\nuseful to identify ' \
                     'conformationally flexible docked configurations'
parser_cluster = {cluster_poses: dict(description=cluster_poses_help, help=cluster_poses_help)}
cluster_poses_arguments = {
    (f'--{as_objects}',): dict(action='store_true', help='Whether to store the resulting pose cluster file as '
                                                         'PoseJob objects\nDefault stores as pose IDs'),
    ('--mode',): dict(type=str.lower, choices={'ialign', 'rmsd', 'transform'}, default='transform', metavar='',
                      help='Which type of clustering should be performed?\nChoices=%(choices)s\nDefault=%(default)s'),
    number_args + (f'--c-{number}',):
        dict(type=int, default=sys.maxsize, metavar='int', help='The number of cluster members to return'),
    output_file_args: dict(type=str,
                           help='Name of the output .pkl file containing pose clusters. Will be saved to the'
                                f' {data.title()} folder of the output.'
                                f'\nDefault={default_clustered_pose_file.format("TIMESTAMP", "LOCATION")}')
}
# ---------------------------------------------------
# Todo add to design protocols...
sequences_args = (f'--{sequences}',)
sequences_kwargs = dict(action=argparse.BooleanOptionalAction, default=True,  # action='store_true',
                        help='For the protocol, create new sequences for each pose?\n'
                             f'{boolean_positional_prevent_msg(sequences)}'),
ca_only_args = (f'--{ca_only}',)
ca_only_kwargs = dict(action='store_true',
                      help='Whether a minimal CA variant of the protein should be used for design calculations')
structures_args = (f'--{structures}',)
structures_kwargs = dict(action='store_true',
                         help='Whether the structure of each new sequence should be calculated'),
neighbors_args = (f'--{neighbors}',)
neighbors_kwargs = \
    dict(action='store_true', help='Whether the neighboring residues should be considered during sequence design')
design_method_args = ('-m', f'--{method}', f'--design-{method}')
design_method_kwargs = dict(type=str.lower, default=proteinmpnn, choices=design_programs, metavar='',
                            help='Which design method should be used?\nChoices=%(choices)s\nDefault=%(default)s')
hbnet_args = ('-hb', f'--{hbnet}')
hbnet_kwargs = dict(action=argparse.BooleanOptionalAction, default=True,
                    help=f'Whether to include hydrogen bond networks in the design.'
                         f'\n{boolean_positional_prevent_msg(hbnet)}')

structure_background_args = ('-sb', f'--{structure_background}')
structure_background_kwargs = dict(action=argparse.BooleanOptionalAction, default=False,
                                   help='Whether to skip all constraints and measure the structure using only the '
                                        'selected energy function')
design_number_args = ('-n', f'--{number_of_designs}')
design_number_kwargs = dict(type=int, default=nstruct, metavar='INT',
                            help='How many unique sequences should be generated for each input?\nDefault=%(default)s')
scout_args = ('-sc', f'--{scout}')
scout_kwargs = dict(action=argparse.BooleanOptionalAction, default=False,
                    help='Whether to set up a low resolution scouting protocol to survey designability')
design_help = 'Gather poses of interest and format for sequence design using Rosetta/ProteinMPNN.' \
              '\nConstrain using evolutionary profiles of homologous sequences' \
              '\nand/or fragment profiles extracted from the PDB or neither'
parser_design = {design: dict(description=design_help, help=design_help)}
design_arguments = {
    ca_only_args: ca_only_kwargs,
    design_method_args: design_method_kwargs,
    evolution_constraint_args: evolution_constraint_kwargs,
    hbnet_args: hbnet_kwargs,
    design_number_args: design_number_kwargs,
    structure_background_args: structure_background_kwargs,
    scout_args: scout_kwargs,
    term_constraint_args: term_constraint_kwargs
}
interface_design_help = 'Gather poses of interest and format for interface specific sequence design using' \
                        '\nRosetta/ProteinMPNN. Constrain using evolutionary profiles of homologous' \
                        '\nsequences and/or fragment profiles extracted from the PDB, or neither'
parser_interface_design = {interface_design: dict(description=interface_design_help, help=interface_design_help)}
interface_design_arguments = {
    ca_only_args: ca_only_kwargs,
    design_method_args: design_method_kwargs,
    evolution_constraint_args: evolution_constraint_kwargs,
    hbnet_args: hbnet_kwargs,
    neighbors_args: neighbors_kwargs,
    design_number_args: design_number_kwargs,
    structure_background_args: structure_background_kwargs,
    scout_args: scout_kwargs,
    term_constraint_args: term_constraint_kwargs
}
# ---------------------------------------------------
interface_metrics_help = 'Analyze interface metrics from a pose'
parser_metrics = {interface_metrics: dict(description=interface_metrics_help, help=interface_metrics_help)}
interface_metrics_arguments = {
    ('-sp', f'--{specific_protocol}'): dict(type=str, metavar='PROTOCOL', default=None,
                                            help='A specific type of design protocol to perform metrics on. If not '
                                                 'provided, capture all design protocols')
}
# ---------------------------------------------------
optimize_designs_help = f'Optimize and touch up designs after running {interface_design}. Useful for reverting\n' \
                        'unnecessary mutations to wild-type, directing exploration of troublesome areas,\nstabilizing '\
                        'an entire design based on evolution, increasing solubility, or modifying\nsurface charge. ' \
                        'Optimization is based on amino acid frequency profiles.\nUse with a --specification-file is ' \
                        'suggested'
parser_optimize_designs = {optimize_designs: dict(description=optimize_designs_help, help=optimize_designs_help)}
optimize_designs_arguments = {
    ('-bg', '--background-profile'): dict(type=str.lower, default=design_profile, metavar='',
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
#     ('-l', '--file-list'): dict(action='store_true',
#                                 help='Whether to use already produced designs in the "designs/" directory'
#                                      '\nDefault=%(default)s'),
#     ('-n', '--native'): dict(type=str, choices=['source', 'asu_path', 'assembly_path', 'refine_pdb', 'refined_pdb',
#                                                 'consensus_pdb', 'consensus_design_pdb'], default='refined_pdb',
#                              help='What structure to use as a "native" structure for Rosetta reference calculations\n'
#                                   'Default=%(default)s'),
#     ('--score-only',): dict(action='store_true', help='Whether to only score the design(s)\nDefault=%(default)s'),
#     ('script',): dict(type=os.path.abspath, help='The location of the custom script'),
#     ('--suffix',): dict(type=str, metavar='SUFFIX',
#                         help='Append to each output file (decoy in .sc and .pdb) the script name (i.e. '
#                              '"decoy_SUFFIX") to identify this protocol. No extension will be included'),
#     ('-v', '--variables'): dict(type=str, nargs='*',
#                                 help='Additional variables that should be populated in the script.\nProvide a list of'
#                                      ' such variables with the format "variable1=value variable2=value". Where '
#                                      'variable1 is a RosettaScripts %%%%variable1%%%% and value is a known value. For'
#                                      'variables that must be calculated on the fly for each design, please modify '
#                                      'structure.model.py class to produce a method that can generate an attribute '
#                                      'with the specified name')
#     # Todo ' either a known value or an attribute available to the Pose object'
# }
# ---------------------------------------------------
analysis_help = 'Analyze all poses specified generating a suite of metrics'
parser_analysis = {analysis: dict(description=analysis_help, help=analysis_help)}
analysis_arguments = {
    ('--figures',): dict(action=argparse.BooleanOptionalAction, default=False,
                         help='Create figures for all poses?'),
    ('--merge',): dict(action='store_true', help='Whether to merge Trajectory and Residue Dataframes'),
    ('--output',): dict(action=argparse.BooleanOptionalAction, default=True,
                        help=f'Whether to output the --{output_file}?\n{boolean_positional_prevent_msg("output")}'),
    #                          '\nDefault=%(default)s'),
    output_file_args: dict(type=str,
                           help='Name of the output .csv file containing pose metrics.\nWill be saved to the '
                                f'{all_scores} folder of the output'
                                f'\nDefault={default_analysis_file.format("TIMESTAMP", "LOCATION")}'),
    ('--save', ): dict(action=argparse.BooleanOptionalAction, default=True,
                       help=f'Save Trajectory and Residues dataframes?\n{boolean_positional_prevent_msg("save")}')
}
# ---------------------------------------------------
# Common selection arguments
allow_multiple_poses_args = ('-amp', f'--{allow_multiple_poses}')
allow_multiple_poses_kwargs = dict(action='store_true',
                                   help='Allow multiple sequences to be selected from the same Pose when using --total'
                                        '\nBy default, --total filters the selected sequences by a single Pose')
csv_args = ('--csv',)
csv_kwargs = dict(action='store_true', help='Write the sequences file as a .csv instead of the default .fasta')
designs_per_pose_args = ('--designs-per-pose',)
designs_per_pose_kwargs = dict(type=int, default=1, help='What is the maximum number of designs that should be selected'
                                                         ' from each pose?\nDefault=%(default)s')

filter_file_args = ('--filter-file',)
filter_file_kwargs = dict(type=os.path.abspath, help='Whether to filter selection using metrics provided in a file')
filter_args = ('--filter',)
filter_kwargs = dict(nargs='*', default=None, help='Whether to filter selection using metrics')
# filter_kwargs = dict(action='store_true', help='Whether to filter selection using metrics')
optimize_species_args = ('-opt', f'--{optimize_species}')
# Todo choices=DNAChisel. optimization species options
optimize_species_kwargs = dict(type=str, default='e_coli',
                               help='The organism where expression will occur and nucleotide usage should be '
                                    'optimized\nDefault=%(default)s')
protocol_args = (f'--{protocol}',)
protocol_kwargs = dict(type=str, help='Use specific protocol(s) to filter designs?', default=None, nargs='*')

save_total_args = ('--save-total',)
save_total_kwargs = dict(action='store_false', help='If --total is used, should the total dataframe be saved?')
pose_select_number_kwargs = \
    dict(type=int, default=sys.maxsize, metavar='int', help='Number to return\nDefault=No Limit')
select_number_kwargs = dict(type=int, default=sys.maxsize, metavar='int',
                            help='Number to return\nIf --total, returns the '
                                 'specified number of sequences (Where Default=No Limit).\nOtherwise the '
                                 'specified number will be selected from each pose (Where Default=1/pose)')
total_args = ('--total',)
total_kwargs = dict(action='store_true',
                    help='Should sequences be selected based on their ranking in the total\ndesign pool? Searches '
                         'for the top sequences from all poses,\nthen chooses one sequence/pose unless '
                         f'--{allow_multiple_poses} is invoked')
weight_file_args = ('--weight-file',)
weight_file_kwargs = dict(type=os.path.abspath,
                          help='Whether to weight selection results using metrics provided in a file')
weight_args = ('--weight',)
weight_kwargs = dict(nargs='*', default=None, help='Whether to weight selection results using metrics')
# weight_kwargs = dict(action='store_true', help='Whether to weight selection results using metrics')
weight_function_args = ('-wf', '--weight-function')
weight_function_kwargs = dict(type=str.lower, choices=config.metric_weight_functions, default='normalize', metavar='',
                              help='How to standardize metrics during selection weighting'
                                   '\nChoices=%(choices)s\nDefault=%(default)s')
# ---------------------------------------------------
select_arguments = {
    cluster_map_args: cluster_map_kwargs,
    filter_args: filter_kwargs,
    number_args + (f'--s-{number}',): select_number_kwargs,
    protocol_args: protocol_kwargs,
    save_total_args: save_total_kwargs,
    total_args: total_kwargs,
    weight_args: weight_kwargs,
    weight_function_args: weight_function_kwargs,
}
# ---------------------------------------------------
select_poses_help = 'Select poses based on specific metrics.\n' \
                    'Selection will be the result of a handful of metrics combined using --filter and/or\n' \
                    f'--weights. For metric options, see the {analysis} --guide. The pose input flag\n' \
                    f'--{specification_file} can be provided to restrict selection criteria'
parser_select_poses = {select_poses: dict(description=select_poses_help, help=select_poses_help)}
select_poses_arguments = {
    **select_arguments,
    number_args + (f'--s-{number}',): pose_select_number_kwargs,
    total_args: dict(action='store_true',
                     help='Should poses be selected based on their ranking in the total\npose pool? This will select '
                          'the top poses based on the\naverage of all designs in that pose for the metrics specified\n'
                          'unless --protocol is invoked, then the protocol average\nwill be used instead'),
    # }
    # # parser_filter_mutual = parser_select_poses.add_mutually_exclusive_group(required=True)
    # parser_select_poses_mutual_group = dict(required=True)
    # parser_select_poses_mutual_arguments = {
    #     ('-m', '--metric'): dict(type=str.lower, choices=['score', 'fragments_matched'], metavar='', default='score',
    #                              help='If a single metric is sufficient, which metric to sort by?'
    #                                   '\nChoices=%(choices)s\nDefault=%(default)s'),
}
# ---------------------------------------------------
intergenic_sequence_args = ('-ms', f'--{multicistronic_intergenic_sequence}')
intergenic_sequence_kwargs = dict(type=str, default=constants.ncoI_multicistronic_sequence,
                                  help='The sequence to use in the intergenic region of a multicistronic expression '
                                       'output')
select_sequences_help = 'From the provided poses, generate sequences (nucleotide/protein) based on specified\n' \
                        'selection criteria and prioritized metrics. Generation of output sequences can take\n' \
                        'multiple forms depending on downstream needs. By default, disordered region insertion,\n' \
                        f'tagging for expression, and codon optimization (--{nucleotide}) are performed. The\n' \
                        f'pose input flag --{specification_file} can be provided to restrict selection criteria'
multicistronic_args = {
    csv_args: csv_kwargs,
    intergenic_sequence_args: intergenic_sequence_kwargs,
    optimize_species_args: optimize_species_kwargs,
}
_select_designs_arguments = {
    allow_multiple_poses_args: allow_multiple_poses_kwargs,
    designs_per_pose_args: designs_per_pose_kwargs,
    output_directory_args:
        dict(type=os.path.abspath, default=None,
             help=f'Where should the output be written?\nDefault={ex_path(os.getcwd(), "SelectedDesigns")}'),
}
parser_select_sequences = {select_sequences: dict(description=select_sequences_help, help=select_sequences_help)}
select_sequences_arguments = {
    **select_arguments,
    **_select_designs_arguments,
    ('-ath', f'--{avoid_tagging_helices}'):
        dict(action='store_true', help='Should tags be avoided at termini with helices?'),
    ('-m', f'--{multicistronic}'):
        dict(action='store_true',
             help='Should nucleotide sequences by output in multicistronic format?\nBy default, uses the pET-Duet '
                  'intergeneic sequence containing\na T7 promoter, LacO, and RBS'),
    (f'--{nucleotide}',): dict(action=argparse.BooleanOptionalAction, default=True,
                               help=f'Whether to output codon optimized nucleotide sequences'
                                    f'\n{boolean_positional_prevent_msg(nucleotide)}'),
    ('-t', f'--{preferred_tag}'): dict(type=str.lower, choices=config.expression_tags.keys(), default='his_tag', metavar='',
                                       help='The name of your preferred expression tag'
                                            '\nChoices=%(choices)s\nDefault=%(default)s'),
    (f'--{tag_entities}',): dict(type=str,
                                 help='If there are specific entities in the designs you want to tag,\nindicate how '
                                      'tagging should occur. Viable options include:\n\t"single" - a single entity\n\t'
                                      '"all" - all entities\n\t"none" - no entities\n\tcomma separated list such as '
                                      '"1,0,1"\n\t\twhere "1" indicates a tag is required\n\t\tand "0" indicates no tag'
                                      ' is required'),
    **multicistronic_args,
}
# ---------------------------------------------------
select_designs_help = 'From the provided poses, select designs based on specified selection criteria using\n' \
                      f'metrics. Alias for {select_sequences} with --skip-sequence-generation. The pose\n' \
                      f'input flag --{specification_file} can be provided to restrict selection criteria'
parser_select_designs = {select_designs: dict(description=select_designs_help, help=select_designs_help)}
select_designs_arguments = {
    **select_arguments,
    **_select_designs_arguments,
    csv_args: csv_kwargs,
}
# ---------------------------------------------------
multicistronic_help = 'Generate nucleotide sequences for selected designs by codon optimizing protein\n' \
                      'sequences, then concatenating nucleotide sequences. REQUIRES an input .fasta file\n' \
                      'specified with the -f/--file argument'
parser_multicistronic = {multicistronic: dict(description=multicistronic_help, help=multicistronic_help)}
multicistronic_arguments = {
    **multicistronic_args,
    number_args: dict(type=int, help='The number of protein sequences to concatenate into a '
                                     'multicistronic expression output'),
}
# ---------------------------------------------------
# parser_asu = subparsers.add_parser('find_asu', description='From a symmetric assembly, locate an ASU and save the result.')
# ---------------------------------------------------
check_clashes_help = 'Check for any clashes in the input poses. This is performed by default at Pose\n' \
                     f'load and will raise a ClashError (caught and reported) if clashes are found'
parser_check_clashes = {check_clashes: dict(description=check_clashes_help, help=check_clashes_help)}
# ---------------------------------------------------
# parser_check_unmodelled_clashes = subparsers.add_parser('check_unmodelled_clashes', description='Check for clashes between full models. Useful for understanding if loops are missing, whether their modelled density is compatible with the pose')
# ---------------------------------------------------
expand_asu_help = 'For given poses, expand the asymmetric unit to a symmetric assembly and write the result'
parser_expand_asu = {expand_asu: dict(description=expand_asu_help, help=expand_asu_help)}
# ---------------------------------------------------
generate_fragments_help = 'Generate fragment overlap for poses of interest and write fragments'
parser_generate_fragments = \
    {generate_fragments: dict(description=generate_fragments_help, help=generate_fragments_help)}
# ---------------------------------------------------
rename_chains_help = 'For given poses, rename the chains in the source PDB to the alphabetic order.\n' \
                     'Useful for writing a multi-model as distinct chains or fixing PDB formatting errors'
parser_rename_chains = {rename_chains:
                        dict(description=rename_chains_help, help=rename_chains_help)}
# # ---------------------------------------------------
# parser_flags = {'flags': dict(description='Generate a flags file for %s' % program_name)}
# # parser_flags = subparsers.add_parser('flags', description='Generate a flags file for %s' % program_name)
# parser_flags_arguments = {
#     ('-t', '--template'): dict(action='store_true', help='Generate a flags template to edit on your own.'),
#     ('-m', '--module'): dict(dest='flags_module', action='store_true',
#                              help='Generate a flags template to edit on your own.')
# }
# # ---------------------------------------------------
# parser_residue_selector = {'residue_selector': dict(description='Generate a residue selection for %s' % program_name)}
# ---------------------------------------------------
directory_needed = f'To locate poses from a file utilizing pose identifiers (--{poses}, -sf)\n' \
                   f'provide your working {program_output} directory with -d/--directory.\n' \
                   f'If you run {program_name} in the context of an existing {program_output}\n' \
                   f'the directory will automatically be inferred'
input_help = f'Specify where/which poses should be included in processing\n{directory_needed}'
parser_input = {input_: dict(description=input_help)}  # , help=input_help
parser_input_group = dict(title=f'{"_" * len(input_title)}\n{input_title}',
                          description=f'\nSpecify where/which poses should be included in processing\n'
                                      f'{directory_needed}')
input_arguments = {
    cluster_map_args: cluster_map_kwargs,
    ('-df', f'--{dataframe}'): dict(type=os.path.abspath, metavar=ex_path('Metrics.csv'),
                                    help=f'A DataFrame created by {program_name} analysis containing\n'
                                         f'pose metrics. File is output in .csv format'),
    ('--fuse-chains',): dict(type=str, nargs='*', default=[], metavar='A:B C:D',
                             help='The name of a pair of chains to fuse during design. Paired\n'
                                  'chains should be separated by a colon, with the n-terminal\n'
                                  'preceding the c-terminal chain. Fusion instances should be\n'
                                  'separated by a space\n'
                                  'Ex --fuse-chains A:B C:D'),
    ('-N', f'--{nanohedra}V1-output'): dict(action='store_true', dest=nanohedra_output,
                                            help='Is the input a Nanohedra wersion 1 docking output?'),
    (f'--{poses}',): dict(type=os.path.abspath, nargs='*',  # dest=putils.specification_file,
                          metavar=ex_path(default_path_file.format('TIMESTAMP', 'MODULE', 'LOCATION')),
                          help=f'For each run of {program_name}, a file will be created that\n'
                               f'specifies the specific poses used during that module. Use\n'
                               f'these files to interact with those poses in subsequent commands'),
    #                            'If pose identifiers are specified in a file, say as the result of\n'
    #                            f'{select_poses} or {select_designs}'),
    ('-P', f'--{preprocessed}'): dict(action='store_true',
                                      help='Whether the designs of interest have been preprocessed for the '
                                           f'{current_energy_function}\nenergy function and/or missing loops'),
    ('-r', '--range'): dict(type=float, default=None, metavar='int-int',
                            help='The range of poses to process from a larger specification.\n'
                                 'Specify a %% between 0 and 100, separating the range by "-"\n'
                                 # Required ^ for formatting
                                 'Ex: 0-25'),
    ('-sf', f'--{specification_file}'):
        dict(type=os.path.abspath, nargs='*', metavar=ex_path('pose_design_specifications.csv'),
             help='Name of comma separated file with each line formatted:\n'
                  # 'poseID, [designID], [1:directive 2-9:directive ...]\n'
                  '"pose_identifier, [design_name], [1:directive 2-9:directive ...]"\n'
                  'where [] indicate optional arguments. Both individual residue\n'
                  'numbers and ranges (specified with "-") are possible indicators')
}
# parser_input_mutual = parser_input.add_mutually_exclusive_group()
parser_input_mutual_group = dict()  # required=True <- adding kwarg below to different parsers depending on need
input_mutual_arguments = {
    ('-d', f'--{directory}'): dict(type=os.path.abspath, metavar=ex_path('your_pdb_files'),
                                   help='Master directory where poses to be designed are located. This may be\n'
                                        f'the output directory from {nanohedra}.py, a random directory with\n'
                                        f'poses requiring design, or the output from {program_name}. If the\n'
                                        f'directory of interest resides in a {program_output} directory, it is\n'
                                        'recommended to use --file, --project, or --single for finer control'),
    ('-f', '--file'): dict(type=os.path.abspath, default=None, nargs='*',
                           metavar=ex_path('file_with_pose.paths'),
                           help='File(s) with structure files to be input listed'),
    ('-p', '--project'): dict(type=os.path.abspath, nargs='*',
                              metavar=ex_path(program_output, projects, 'yourProject'),
                              help='Operate on designs specified within a project(s)'),
    ('-s', '--single'): dict(type=os.path.abspath, nargs='*',
                             metavar=ex_path(program_output, projects, 'yourProject', 'single_pose[.pdb]'),
                             help='Operate on single pose(s) in a project'),
}
output_help = 'Specify where output should be written'
parser_output = {output: dict(description=output_help)}  # , help=output_help
parser_output_group = dict(title=f'{"_" * len(output_title)}\n{output_title}',
                           description='\nSpecify where output should be written')
output_arguments = {
    (f'--{increment_chains}',): dict(action='store_true',
                                     help='Whether resulting assembly files should output with chain IDs alphabetically'
                                          '\nincremented or in "Multimodel" format. Multimodel format is useful for\n'
                                          'PyMol visualization with the command "set all_states, on"'),
    ('-Oa', f'--{output_assembly}'):
        dict(action=argparse.BooleanOptionalAction, default=False,
             help='Whether the assembly should be output? Infinite materials are output in a unit cell'),
    output_directory_args:
        dict(type=os.path.abspath, default=None,
             help='If provided, the name of the directory to output all created files.\nOtherwise, one will be '
                  'generated based on the time, input, and module'),
    output_file_args: dict(type=str, help='If provided, the name of the output pose file.\nOtherwise, one will be '
                                          'generated based on the time, input, and module'),
    ('-OF', f'--{output_fragments}'):
        dict(action=argparse.BooleanOptionalAction, default=False,
             help='For any fragments generated, write them along with the Pose'),
    ('-Oo', f'--{output_oligomers}'):
        dict(action=argparse.BooleanOptionalAction, default=False,
             help='For any oligomers generated, write them along with the Pose'),
    ('-Os', f'--{output_structures}'):
        dict(action=argparse.BooleanOptionalAction, default=True,
             help=f'For any structures generated, write them.\n{boolean_positional_prevent_msg(output_structures)}'),
    ('-Ou', f'--{output_surrounding_uc}'):
        dict(action=argparse.BooleanOptionalAction, default=False,
             help='Whether the surrounding unit cells should be output?\nOnly for infinite materials'),
    ('-Ot', f'--{output_trajectory}'):
        dict(action=argparse.BooleanOptionalAction, default=False,
             help=f'For all structures generated, write them as a single multimodel file'),
    ('--overwrite',): dict(action='store_true', help='Whether to overwrite existing structures upon job fulfillment'),
    ('--prefix',): dict(type=str, metavar='string', help='String to prepend to output name'),
    ('--suffix',): dict(type=str, metavar='string', help='String to append to output name'),
}
# If using mutual groups, for the dict "key" (parser name), you must add "_mutual" immediately after the submodule
# string that own the group. i.e nanohedra"_mutual*" indicates nanohedra owns, or interface_design"_mutual*", etc
module_parsers = {
    orient: parser_orient,
    refine: parser_refine,
    nanohedra: parser_nanohedra,
    'nanohedra_mutual1': parser_nanohedra_mutual1_group,  # _mutual1,
    'nanohedra_mutual2': parser_nanohedra_mutual2_group,  # _mutual2,
    'nanohedra_mutual_run_type': parser_nanohedra_run_type_mutual_group,  # _mutual,
    cluster_poses: parser_cluster,
    design: parser_design,
    interface_design: parser_interface_design,
    interface_metrics: parser_metrics,
    optimize_designs: parser_optimize_designs,
    # custom_script: parser_custom,
    analysis: parser_analysis,
    select_poses: parser_select_poses,
    # select_poses_mutual: parser_select_poses_mutual_group,  # _mutual,
    select_designs: parser_select_designs,
    select_sequences: parser_select_sequences,
    multicistronic: parser_multicistronic,
    # flags: parser_flags,
    check_clashes: parser_check_clashes,
    expand_asu: parser_expand_asu,
    generate_fragments: parser_generate_fragments,
    rename_chains: parser_rename_chains,
    input_: parser_input,
    'input_mutual': parser_input_mutual_group,
    output: parser_output,
    options: parser_options,
    predict_structure: parser_predict_structure,
    protocol: parser_protocol,
    residue_selector: parser_residue_selector,
    # residue_selector: parser_residue_selector
}
input_parsers = dict(input=parser_input_group,
                     input_mutual=parser_input_mutual_group)  # _mutual
output_parsers = dict(output=parser_output_group)
option_parsers = dict(options=parser_options_group)
residue_selector_parsers = dict(residue_selector=parser_residue_selector_group)
parser_arguments = {
    options: options_arguments,
    residue_selector: residue_selector_arguments,
    refine: refine_arguments,
    nanohedra: nanohedra_arguments,
    'nanohedra_mutual1': nanohedra_mutual1_arguments,  # mutually_exclusive_group
    'nanohedra_mutual2': nanohedra_mutual2_arguments,  # mutually_exclusive_group
    'nanohedra_mutual_run_type': nanohedra_run_type_mutual_arguments,  # mutually_exclusive
    cluster_poses: cluster_poses_arguments,
    design: design_arguments,
    interface_design: interface_design_arguments,
    interface_metrics: interface_metrics_arguments,
    optimize_designs: optimize_designs_arguments,
    analysis: analysis_arguments,
    select_poses: select_poses_arguments,
    select_designs: select_designs_arguments,
    select_sequences: select_sequences_arguments,
    multicistronic: multicistronic_arguments,
    input_: input_arguments,
    'input_mutual': input_mutual_arguments,  # add_mutually_exclusive_group
    output: output_arguments,
    predict_structure: predict_structure_arguments,
    protocol: protocol_arguments,
    # custom_script_arguments: parser_custom_script_arguments,
    # select_poses_mutual_arguments: parser_select_poses_mutual_arguments, # mutually_exclusive_group
    # flags_arguments: parser_flags_arguments,
}
parser_options = 'parser_options'
parser_residue_selector = 'parser_residue_selector'
parser_input = 'parser_input'
parser_output = 'parser_output'
parser_module = 'parser_module'
parser_guide = 'parser_guide'
parser_entire = 'parser_entire'
options_argparser = dict(add_help=False, allow_abbrev=False, formatter_class=Formatter)  # Todo? , usage=usage_str)
residue_selector_argparser = \
    dict(add_help=False, allow_abbrev=False, formatter_class=Formatter, usage=usage_str)
input_argparser = dict(add_help=False, allow_abbrev=False, formatter_class=Formatter, usage=usage_str)
module_argparser = dict(add_help=False, allow_abbrev=False, formatter_class=Formatter, usage=usage_str)
guide_argparser = dict(add_help=False, allow_abbrev=False, formatter_class=Formatter, usage=usage_str)
output_argparser = dict(add_help=False, allow_abbrev=False, formatter_class=Formatter, usage=usage_str)
argparsers_kwargs = dict(parser_options=options_argparser,
                         parser_residue_selector=residue_selector_argparser,
                         parser_input=input_argparser,
                         parser_module=module_argparser,
                         parser_guide=guide_argparser,
                         parser_output=output_argparser,
                         )
# Initialize various independent ArgumentParsers
argparsers: dict[str, argparse.ArgumentParser] = {}
for argparser_name, argparser_kwargs in argparsers_kwargs.items():
    # Todo https://gist.github.com/fonic/fe6cade2e1b9eaf3401cc732f48aeebd
    #  argparsers[argparser_name] = ArgumentParser(**argparser_args)
    argparsers[argparser_name] = argparse.ArgumentParser(**argparser_kwargs)

# Set up module ArgumentParser with module arguments
module_subargparser = dict(title=f'{"_" * len(module_title)}\n{module_title}', dest='module',  # metavar='',
                           # allow_abbrev=False,  # Not allowed here, but some modules try to parse -s as -sc...
                           description='\nThese are the different modes that designs can be processed. They are'
                                       '\npresented in an order which might be utilized along a design workflow,\nwith '
                                       'utility modules listed at the bottom starting with check_clashes.\nTo see '
                                       'example commands or algorithmic specifics of each Module enter:'
                                       f'\n{submodule_guide}\n\nTo get help with Module arguments enter:'
                                       f'\n{submodule_help}')
module_required = ['nanohedra_mutual1']
# For all parsing of module arguments
subparsers = argparsers[parser_module].add_subparsers(**module_subargparser)  # required=True,
# For parsing of guide info
argparsers[parser_guide].add_argument(*guide_args, **guide_kwargs)
argparsers[parser_guide].add_argument(*setup_args, **setup_kwargs)
guide_subparsers = argparsers[parser_guide].add_subparsers(**module_subargparser)
module_suparsers: dict[str, argparse.ArgumentParser] = {}
for parser_name, parser_kwargs in module_parsers.items():
    arguments = parser_arguments.get(parser_name, {})
    # Has args as dictionary key (flag names) and keyword args as dictionary values (flag params)
    if 'mutual' in parser_name:  # We must create a mutually_exclusive_group from already formed subparser
        # Remove indication to "mutual" of the argparse group by removing any string after "_mutual"
        exclusive_parser = module_suparsers[parser_name[:parser_name.find('_mutual')]].\
            add_mutually_exclusive_group(**parser_kwargs, **(dict(required=True) if parser_name in module_required
                                                             else {}))
        # Add the key word argument "required" to mutual parsers that use it ^
        for args, kwargs in arguments.items():
            exclusive_parser.add_argument(*args, **kwargs)
    else:  # Save the subparser in a dictionary to access with mutual groups
        module_suparsers[parser_name] = subparsers.add_parser(prog=module_usage_str % parser_name,
                                                              # prog=f'python SymDesign.py %(name) '
                                                              #      f'[input_arguments] [optional_arguments]',
                                                              formatter_class=Formatter, allow_abbrev=False,
                                                              name=parser_name, **parser_kwargs[parser_name])
        for args, kwargs in arguments.items():
            module_suparsers[parser_name].add_argument(*args, **kwargs)
        # Add each subparser to a guide_subparser as well
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
        # Has args as dictionary key (flag names) and keyword args as dictionary values (flag params)
        # There are no 'mutual' right now
        if 'mutual' in parser_name:  # Only has a dictionary as parser_arguments
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
        # Has args as dictionary key (flag names) and keyword args as dictionary values (flag params)
        # There are no 'mutual' right now
        if 'mutual' in parser_name:  # Only has a dictionary as parser_arguments
            exclusive_parser = residue_selector_group.add_mutually_exclusive_group(**parser_kwargs)
            for args, kwargs in arguments.items():
                exclusive_parser.add_argument(*args, **kwargs)
        residue_selector_group = parser.add_argument_group(**parser_kwargs)
        for args, kwargs in arguments.items():
            residue_selector_group.add_argument(*args, **kwargs)

# Set up input ArgumentParser with input arguments
parser = argparsers[parser_input]
input_group = None  # Must get added before mutual groups can be added
for parser_name, parser_kwargs in input_parsers.items():
    arguments = parser_arguments.get(parser_name, {})
    if 'mutual' in parser_name:  # Only has a dictionary as parser_arguments
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
                        description=f'{"_" * len(program_name)}\n{program_name}\n\n'
                                    f'Control all input/output of various {program_name} operations including:'
                                    f'\n  1. {nanohedra.title()} docking '
                                    '\n  2. Pose set up, sampling, assembly generation, fragment decoration'
                                    '\n  3. Interface design using constrained residue profiles and various design '
                                    'algorithms'
                                    '\n  4. Analysis of all design metrics'
                                    '\n  5. Selection of designs by prioritization of calculated metrics'
                                    '\n  6. Sequence formatting for biochemical characterization\n\n'
                                    f"If you're a first time user, try:\n{program_command} --guide"
                                    '\nMost modules have features for command monitoring, parallel processing, and '
                                    'distribution to computational clusters',
                        formatter_class=Formatter, usage=usage_str,
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
#         entire_module_suparsers[parser_name] = subparsers.add_parser(prog=module_usage_str % parser_name,
#                                                                      # prog=f'python SymDesign.py %(name) '
#                                                                      #      f'[input_arguments] [optional_arguments]',
#                                                                      formatter_class=Formatter, allow_abbrev=False,
#                                                                      name=parser_name, **parser_kwargs[parser_name])
#         for args, kwargs in arguments.items():
#             entire_module_suparsers[parser_name].add_argument(*args, **kwargs)

# Separate the provided arguments for modules or overall program arguments to into flags namespaces
design_defaults = {}
"""Contains all the arguments and their default parameters used in design"""
dock_defaults = {}
"""Contains all the arguments and their default parameters used in docking"""
predict_defaults = {}
"""Contains all the arguments and their default parameters used in structure prediction"""
cluster_defaults = {}
"""Contains all the arguments and their default parameters used in clustering Poses"""
for group in parser._action_groups:
    for arg in group._group_actions:
        if isinstance(arg, argparse._SubParsersAction):  # We have a sup parser, recurse
            for name, sub_parser in arg.choices.items():
                for sub_group in sub_parser._action_groups:
                    for arg in sub_group._group_actions:
                        if arg.dest in design_namespace:
                            design_defaults[arg.dest] = arg.default
                        elif arg.dest in dock_namespace:
                            dock_defaults[arg.dest] = arg.default
                        elif arg.dest in predict_namespace:
                            predict_defaults[arg.dest] = arg.default
                        elif arg.dest in cluster_namespace:
                            cluster_defaults[arg.dest] = arg.default

        elif arg.dest in design_namespace:
            design_defaults[arg.dest] = arg.default
        elif arg.dest in dock_namespace:
            dock_defaults[arg.dest] = arg.default
        elif arg.dest in predict_namespace:
            predict_defaults[arg.dest] = arg.default
        elif arg.dest in cluster_namespace:
            cluster_defaults[arg.dest] = arg.default

predict_defaults[method] = design_defaults[method]  # Also in design...
