from __future__ import annotations

import argparse
import ast
import logging
import os
import operator
import sys
from collections.abc import Callable, Sequence
from typing import Any, AnyStr, Literal, get_args

import pandas as pd
from psutil import cpu_count

from symdesign.sequence import constants, optimization_species_literal
from symdesign.resources import config
from symdesign.structure.utils import coords_types, default_clash_criteria, default_clash_distance, \
    design_programs_literal, termini_literal
from symdesign.utils import handle_errors, InputError, log_levels, remove_digit_table, path as putils, \
    pretty_format_table, to_iterable, logging_levels
from symdesign.utils.path import biological_interfaces, default_clustered_pose_file, default_logging_level, \
    default_path_file, ex_path, fragment_dbs, program_output, program_command, program_name, projects, \
    submodule_guide, submodule_help
from symdesign.utils.path import design_profile, evolutionary_profile, fragment_profile, consensus
from symdesign.utils.query import input_string, confirmation_string, bool_d, invalid_string, header_string, \
    format_string
from symdesign.utils.rosetta import current_energy_function
from symdesign.utils.SymEntry import query_mode_args

# Gloabls
logger = logging.getLogger(__name__)
# Flag names
force = 'force'
sym_entry = 'sym_entry'
interface_metrics = 'interface_metrics'
component1 = 'component1'
component2 = 'component2'
data = 'data'
multi_processing = 'multi_processing'
residue_selector = 'residue_selector'
options = 'options'
cluster_poses = 'cluster_poses'
interface_design = 'interface_design'
evolution_constraint = 'evolution_constraint'
use_evolution = 'use_evolution'
hbnet = 'hbnet'
term_constraint = 'term_constraint'
design_number = 'design_number'
refine = 'refine'
structure_background = 'structure_background'
scout = 'scout'
profile = 'profile'
# design_profile = 'design_profile'
# evolutionary_profile = 'evolutionary_profile'
# fragment_profile = 'fragment_profile'
select_sequences = 'select_sequences'
nanohedra = 'nanohedra'
predict_structure = 'predict_structure'
output_interface = 'output_interface'
analysis = 'analysis'
select_poses = 'select_poses'
output_fragments = 'output_fragments'
output_oligomers = 'output_oligomers'
output_entities = 'output_entities'
protocol = 'protocol'
ignore_clashes = 'ignore_clashes'
ignore_pose_clashes = 'ignore_pose_clashes'
ignore_symmetric_clashes = 'ignore_symmetric_clashes'
select_designs = 'select_designs'
output_structures = 'output_structures'
proteinmpnn = 'proteinmpnn'
output_trajectory = 'output_trajectory'
development = 'development'
ca_only = 'ca_only'
sequences = 'sequences'
structures = 'structures'
temperatures = 'temperatures'
optimize_species = 'optimize_species'
distribute_work = 'distribute_work'
output_directory = 'output_directory'
output_surrounding_uc = 'output_surrounding_uc'
skip_logging = 'skip_logging'
output_file = 'output_file'
avoid_tagging_helices = 'avoid_tagging_helices'
multicistronic = 'multicistronic'
multicistronic_intergenic_sequence = 'multicistronic_intergenic_sequence'
generate_fragments = 'generate_fragments'
input_ = 'input'
output = 'output'
output_assembly = 'output_assembly'
preferred_tag = 'preferred_tag'
expand_asu = 'expand_asu'
check_clashes = 'check_clashes'
rename_chains = 'rename_chains'
optimize_designs = 'optimize_designs'
perturb_dof = 'perturb_dof'
tag_entities = 'tag_entities'
design = 'design'
process_rosetta_metrics = 'process_rosetta_metrics'
query_codes = 'query_codes'
nanohedra_output = 'nanohedra_output'
modules = 'modules'
score = 'score'
module = 'module'
method = 'method'
interface = 'interface'
interface_only = 'interface_only'
oligomeric_interfaces = 'oligomeric_interfaces'
neighbors = 'neighbors'
# dock_only = 'dock_only'
rotation_step1 = 'rotation_step1'
rotation_step2 = 'rotation_step2'
min_matched = 'min_matched'
minimum_matched = 'minimum_matched'
match_value = 'match_value'
initial_z_value = 'initial_z_value'
interface_distance = 'interface_distance'
only_write_frag_info = 'only_write_frag_info'
proteinmpnn_score = 'proteinmpnn_score'
contiguous_ghosts = 'contiguous_ghosts'
perturb_dof_steps = 'perturb_dof_steps'
perturb_dof_rot = 'perturb_dof_rot'
perturb_dof_tx = 'perturb_dof_tx'
perturb_dof_steps_rot = 'perturb_dof_steps_rot'
perturb_dof_steps_tx = 'perturb_dof_steps_tx'
dock_filter = 'dock_filter'
dock_filter_file = 'dock_filter_file'
dock_weight = 'dock_weight'
dock_weight_file = 'dock_weight_file'
clash_distance = 'clash_distance'
clash_criteria = 'clash_criteria'
cluster_map = 'cluster_map'
cluster_mode = 'cluster_mode'
cluster_number = 'cluster_number'
specification_file = 'specification_file'
poses = 'poses'
specific_protocol = 'specific_protocol'
directory = 'directory'
dataframe = 'dataframe'
fragment_source = 'fragment_source'
database_url = 'database_url'
interface_to_alanine = 'interface_to_alanine'
metrics = 'metrics'
increment_chains = 'increment_chains'
number = 'number'
bend = 'bend'
select_number = 'select_number'
nucleotide = 'nucleotide'
as_objects = 'as_objects'
# allow_multiple_poses = 'allow_multiple_poses'
designs_per_pose = 'designs_per_pose'
project_name = 'project_name'
profile_memory = 'profile_memory'
preprocessed = 'preprocessed'
quick = 'quick'
pose_format = 'pose_format'
use_gpu_relax = 'use_gpu_relax'
design_method = 'design_method'
predict_method = 'predict_method'
predict_assembly = 'predict_assembly'
predict_designs = 'predict_designs'
predict_pose = 'predict_pose'
num_predictions_per_model = 'num_predictions_per_model'
predict_entities = 'predict_entities'
models_to_relax = 'models_to_relax'
debug_db = 'debug_db'
reset_db = 'reset_db'
load_to_db = 'load_to_db'
all_flags = 'all_flags'
loop_model_input = 'loop_model_input'
refine_input = 'refine_input'
pre_loop_modeled = 'pre_loop_modeled'
pre_refined = 'pre_refined'
initialize_building_blocks = 'initialize_building_blocks'
background_profile = 'background_profile'
pdb_code = 'pdb_code'
update_metadata = 'update_metadata'
proteinmpnn_model = 'proteinmpnn_model'
tag_linker = 'tag_linker'
update_db = 'update_db'
measure_pose = 'measure_pose'
cluster_selection = 'cluster_selection'
number_of_commands = 'number_of_commands'
specify_entities = 'specify_entities'
helix_bending = 'helix_bending'
direction = 'direction'
joint_chain = 'joint_chain'
joint_residue = 'joint_residue'
sample_number = 'sample_number'
align_helices = 'align_helices'
target_start = 'target_start'
target_end = 'target_end'
target_chain = 'target_chain'
target_termini = 'target_termini'
trim_termini = 'trim_termini'
aligned_start = 'aligned_start'
aligned_end = 'aligned_end'
aligned_chain = 'aligned_chain'
extend = 'extend'
alignment_length = 'alignment_length'
length = 'length'
design_chains = 'design_chains'
design_residues = 'design_residues'
mask_residues = 'mask_residues'
mask_chains = 'mask_chains'
require_residues = 'require_residues'
require_chains = 'require_chains'
use_proteinmpnn = 'use_proteinmpnn'
setup = 'setup'
mpi = 'mpi'
cores = 'cores'
range_ = 'range'
project = 'project'
single = 'single'
fuse_chains = 'fuse_chains'
save_total = 'save_total'
weight_file = 'weight_file'
weight = 'weight'
weight_function = 'weight_function'
file = 'file'
filter_ = 'filter'
filter_file = 'filter_file'
csv = 'csv'
symmetry = 'symmetry'
query = 'query'
guide = 'guide'
overwrite = 'overwrite'
prefix = 'prefix'
suffix = 'suffix'
log_level = 'log_level'
# Set up JobResources namespaces for different categories of flags
cluster_namespace = {
    as_objects, cluster_map, cluster_mode, cluster_number
}
design_namespace = {
    consensus, ca_only, clash_distance, clash_criteria, design_method, design_number, evolution_constraint, hbnet,
    ignore_clashes, ignore_pose_clashes, ignore_symmetric_clashes, interface, neighbors, proteinmpnn_model, scout,
    sequences, structure_background, structures, term_constraint, temperatures
}
dock_namespace = {
    contiguous_ghosts, dock_filter, dock_filter_file, dock_weight, dock_weight_file, initial_z_value, match_value,
    minimum_matched, perturb_dof, perturb_dof_rot, perturb_dof_tx, perturb_dof_steps, perturb_dof_steps_rot,
    perturb_dof_steps_tx, proteinmpnn_score, quick, rotation_step1, rotation_step2
}  # score,
init_namespace = {
    loop_model_input, refine_input, pre_loop_modeled, pre_refined
}
predict_namespace = {
    models_to_relax, num_predictions_per_model, predict_assembly, predict_designs, predict_entities, predict_method,
    predict_pose, use_gpu_relax
}
namespaces = dict(
    cluster=cluster_namespace,
    design=design_namespace,
    dock=dock_namespace,
    init=init_namespace,
    predict=predict_namespace
)
# Modify specific flags from their prefix to their suffix
modify_options = dict(
    cluster=[cluster_map, cluster_mode, cluster_number],
    design=[design_method, design_number],
    dock=[dock_filter, dock_filter_file, dock_weight, dock_weight_file],
    init=[],  # loop_model_input, refine_input],
    predict=[predict_assembly, predict_designs, predict_entities, predict_method, predict_pose],
)

argparse_flag_delimiter = '-'


def format_for_cmdline(string) -> str:
    """Format a flag for the command line

    Args:
        string: The string to format as a commandline flag

    Returns:
        The string formatted by replacing any underscores '_' with a dash '-'
    """
    return string.replace('_', argparse_flag_delimiter)


def format_from_cmdline(string) -> str:
    """Format a string from the command line format to a program acceptable string

    Args:
        string: The string to format as a python string

    Returns:
        The flag formatted by replacing any dash '-' with an underscore '_'
    """
    return string.replace(argparse_flag_delimiter, '_')


class FlagStr(str):
    """Flag instances are strings which represent possible input parameters to the program with additional formatting
    for program runtime to properly reflect underscore and dashed versions
    """
    flag_character = argparse_flag_delimiter

    def __new__(cls, string: str):
        return super().__new__(cls, format_for_cmdline(string))

    @property
    def _(self) -> str:
        """Format a string from the command line format to a program acceptable string

        Returns:
            The flag formatted by replacing any dash '-' with an underscore '_'
        """
        return format_from_cmdline(self)

    @property
    def long(self) -> str:
        """Format a flag for the command line

        Returns:
            The flag formatted by replacing any underscore '_' with a dash '-'
        """
        return f'{self.flag_character}{self.flag_character}{self}'


def format_args(flag_args: Sequence[str]) -> str:
    """Create a string to format different flags for their various acceptance options on the command line

    Args:
        flag_args: Typically a tuple of allowed flag "keywords" specified using "-" or "--"

    Returns:
        The flag arguments formatted with a "/" between each allowed version
    """
    return '/'.join(flag_args)


project = FlagStr(project)
single = FlagStr(single)
fuse_chains = FlagStr(fuse_chains)
setup = FlagStr(setup)
mpi = FlagStr(mpi)
cores = FlagStr(cores)
range_ = FlagStr(range_)
save_total = FlagStr(save_total)
weight_file = FlagStr(weight_file)
weight = FlagStr(weight)
weight_function = FlagStr(weight_function)
file = FlagStr(file)
filter_ = FlagStr(filter_)
filter_file = FlagStr(filter_file)
csv = FlagStr(csv)
symmetry = FlagStr(symmetry)
query = FlagStr(query)
guide = FlagStr(guide)
overwrite = FlagStr(overwrite)
prefix = FlagStr(prefix)
suffix = FlagStr(suffix)
log_level = FlagStr(log_level)
development = FlagStr(development)
force = FlagStr(force)
quick = FlagStr(quick)
nanohedra = FlagStr(nanohedra)
protocol = FlagStr(protocol)
nucleotide = FlagStr(nucleotide)
multicistronic = FlagStr(multicistronic)
number = FlagStr(number)
interface = FlagStr(interface)
directory = FlagStr(directory)
dataframe = FlagStr(dataframe)
neighbors = FlagStr(neighbors)
temperatures = FlagStr(temperatures)
poses = FlagStr(poses)
modules = FlagStr(modules)
scout = FlagStr(scout)
bend = FlagStr(bend)
hbnet = FlagStr(hbnet)
as_objects = FlagStr(as_objects)
query_codes = FlagStr(query_codes)
predict_structure = FlagStr(predict_structure)
predict_method = FlagStr(predict_method)
num_predictions_per_model = FlagStr(num_predictions_per_model)
predict_pose = FlagStr(predict_pose)
predict_assembly = FlagStr(predict_assembly)
predict_designs = FlagStr(predict_designs)
predict_entities = FlagStr(predict_entities)
models_to_relax = FlagStr(models_to_relax)
cluster_poses = FlagStr(cluster_poses)
generate_fragments = FlagStr(generate_fragments)
fragment_source = FlagStr(fragment_source)
database_url = FlagStr(database_url)
interface_metrics = FlagStr(interface_metrics)
optimize_designs = FlagStr(optimize_designs)
interface_design = FlagStr(interface_design)
interface_only = FlagStr(interface_only)
oligomeric_interfaces = FlagStr(oligomeric_interfaces)
residue_selector = FlagStr(residue_selector)
select_designs = FlagStr(select_designs)
select_poses = FlagStr(select_poses)
select_sequences = FlagStr(select_sequences)
check_clashes = FlagStr(check_clashes)
expand_asu = FlagStr(expand_asu)
rename_chains = FlagStr(rename_chains)
evolution_constraint = FlagStr(evolution_constraint)
use_evolution = FlagStr(use_evolution)
use_proteinmpnn = FlagStr(use_proteinmpnn)
term_constraint = FlagStr(term_constraint)
design_number = FlagStr(design_number)
design_method = FlagStr(design_method)
select_number = FlagStr(select_number)
structure_background = FlagStr(structure_background)
# design_profile = FlagStr(design_profile)
# evolutionary_profile = FlagStr(evolutionary_profile)
# fragment_profile = FlagStr(fragment_profile)
cluster_map = FlagStr(cluster_map)
cluster_mode = FlagStr(cluster_mode)
cluster_number = FlagStr(cluster_number)
specification_file = FlagStr(specification_file)
# poses = FlagStr(poses)
specific_protocol = FlagStr(specific_protocol)
sym_entry = FlagStr(sym_entry)
# dock_only = FlagStr(dock_only)
rotation_step1 = FlagStr(rotation_step1)
rotation_step2 = FlagStr(rotation_step2)
min_matched = FlagStr(min_matched)
minimum_matched = FlagStr(minimum_matched)
match_value = FlagStr(match_value)
initial_z_value = FlagStr(initial_z_value)
interface_distance = FlagStr(interface_distance)
only_write_frag_info = FlagStr(only_write_frag_info)
proteinmpnn_score = FlagStr(proteinmpnn_score)
contiguous_ghosts = FlagStr(contiguous_ghosts)
perturb_dof_steps = FlagStr(perturb_dof_steps)
perturb_dof_rot = FlagStr(perturb_dof_rot)
perturb_dof_tx = FlagStr(perturb_dof_tx)
perturb_dof_steps_rot = FlagStr(perturb_dof_steps_rot)
perturb_dof_steps_tx = FlagStr(perturb_dof_steps_tx)
dock_filter = FlagStr(dock_filter)
dock_filter_file = FlagStr(dock_filter_file)
dock_weight = FlagStr(dock_weight)
dock_weight_file = FlagStr(dock_weight_file)
ca_only = FlagStr(ca_only)
perturb_dof = FlagStr(perturb_dof)
distribute_work = FlagStr(distribute_work)
multi_processing = FlagStr(multi_processing)
output_assembly = FlagStr(output_assembly)
output_fragments = FlagStr(output_fragments)
output_oligomers = FlagStr(output_oligomers)
output_entities = FlagStr(output_entities)
output_structures = FlagStr(output_structures)
output_trajectory = FlagStr(output_trajectory)
output_directory = FlagStr(output_directory)
output_file = FlagStr(output_file)
output_surrounding_uc = FlagStr(output_surrounding_uc)
output_interface = FlagStr(output_interface)
clash_distance = FlagStr(clash_distance)
clash_criteria = FlagStr(clash_criteria)
ignore_clashes = FlagStr(ignore_clashes)
ignore_pose_clashes = FlagStr(ignore_pose_clashes)
ignore_symmetric_clashes = FlagStr(ignore_symmetric_clashes)
component1 = FlagStr(component1)
component2 = FlagStr(component2)
skip_logging = FlagStr(skip_logging)
interface_to_alanine = FlagStr(interface_to_alanine)
metrics = FlagStr(metrics)
increment_chains = FlagStr(increment_chains)
tag_entities = FlagStr(tag_entities)
optimize_species = FlagStr(optimize_species)
avoid_tagging_helices = FlagStr(avoid_tagging_helices)
preferred_tag = FlagStr(preferred_tag)
multicistronic_intergenic_sequence = FlagStr(multicistronic_intergenic_sequence)
# allow_multiple_poses = FlagStr(allow_multiple_poses)
designs_per_pose = FlagStr(designs_per_pose)
project_name = FlagStr(project_name)
profile_memory = FlagStr(profile_memory)
process_rosetta_metrics = FlagStr(process_rosetta_metrics)
pose_format = FlagStr(pose_format)
use_gpu_relax = FlagStr(use_gpu_relax)
debug_db = FlagStr(debug_db)
reset_db = FlagStr(reset_db)
load_to_db = FlagStr(load_to_db)
all_flags = FlagStr(all_flags)
loop_model_input = FlagStr(loop_model_input)
refine_input = FlagStr(refine_input)
pre_loop_modeled = FlagStr(pre_loop_modeled)
pre_refined = FlagStr(pre_refined)
initialize_building_blocks = FlagStr(initialize_building_blocks)
background_profile = FlagStr(background_profile)
pdb_code = FlagStr(pdb_code)
update_metadata = FlagStr(update_metadata)
proteinmpnn_model = FlagStr(proteinmpnn_model)
tag_linker = FlagStr(tag_linker)
update_db = FlagStr(update_db)
measure_pose = FlagStr(measure_pose)
cluster_selection = FlagStr(cluster_selection)
number_of_commands = FlagStr(number_of_commands)
specify_entities = FlagStr(specify_entities)
helix_bending = FlagStr(helix_bending)
direction = FlagStr(direction)
joint_chain = FlagStr(joint_chain)
joint_residue = FlagStr(joint_residue)
sample_number = FlagStr(sample_number)
align_helices = FlagStr(align_helices)
target_start = FlagStr(target_start)
target_end = FlagStr(target_end)
target_chain = FlagStr(target_chain)
target_termini = FlagStr(target_termini)
trim_termini = FlagStr(trim_termini)
aligned_start = FlagStr(aligned_start)
aligned_end = FlagStr(aligned_end)
aligned_chain = FlagStr(aligned_chain)
extend = FlagStr(extend)
alignment_length = FlagStr(alignment_length)
design_chains = FlagStr(design_chains)
require_residues = FlagStr(require_residues)
require_chains = FlagStr(require_chains)
design_residues = FlagStr(design_residues)
mask_residues = FlagStr(mask_residues)
mask_chains = FlagStr(mask_chains)

select_modules = (
    select_poses,
    select_designs,
    select_sequences,
)


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
              " a flags file\nsuch as 'my_design.flags' specified on the command line like '@my_design.flags' or "
              "manually by typing them in.\nTo automatically generate a flags file template, run '%s flags --template'"
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
        return {dict(tuple(flag.lstrip(argparse_flag_delimiter).split())) for flag in f.readlines()}

# def not_contains(__a: Container[object], __b: object) -> bool:
#     return operator.not_(operator.contains(__a, __b))


def not_contains(__a: pd.Series, __b: object) -> pd.Series:
    return operator.invert(pd.Series.isin(__a, __b))


# Put longer operations first so that they are scanned first, then shorter ones i.e. '>=' then '>'
viable_operations = {
    # '+': operator.add, '-': operator.sub,
    # '*': operator.mul, '@': operator.matmul,
    # '/': operator.truediv, '//': operator.floordiv,
    '!->': not_contains,
    '->': pd.Series.isin,  # operator.contains,  # value in metric
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
    # operator.contains: '->',
    pd.Series.isin: '->',
    operator.ge: '>=',
    operator.le: '<=',
    operator.ne: '!=',
    operator.eq: '=',
    operator.gt: '>',
    operator.lt: '<',
}


def parse_weights(weights: list[str] = None, file: AnyStr = None) \
        -> dict[str, list[tuple[Callable, Callable, dict, Any]]]:
    """Given a command line specified set of metrics and values, parse into weights to select DataFrames accordingly

    Args:
        weights: The command line collected weight arguments as specified in the weights --help
        file: The path to a file specifying weights in JSON as specified in the weights --help

    Returns:
        The parsed metric mapping linking each metric to a specified operation
    """
    parsed_weights = parse_filters(weights, file=file)
    # Ensure proper formatting of weight specific parameters
    for idx, (metric_name, weight) in enumerate(parsed_weights.items()):
        if len(weight) != 1:
            raise InputError(
                f"Can't assign more than one weight for every provided metric. '{weights[idx]}' is invalid")
        operation, pre_operation, pre_kwargs, value = weight[0]
        if operation != operator.eq:
            raise InputError(
                f"Can't assign a selection weight with the operator '{operator_strings[operation]}'. "
                f"'{weights[idx]}' is invalid")
        if isinstance(value, str):
            raise InputError(
                f"Can't assign a numerical weight to the provided weight '{weights[idx]}'")

    return parsed_weights


def parse_filters(filters: list[str] = None, file: AnyStr = None) \
        -> dict[str, list[tuple[Callable, Callable, dict, Any]]]:
    """Given a command line specified set of metrics and values, parse into filters to select DataFrames accordingly

    Args:
        filters: The command line collected filter arguments as specified in the filters --help
        file: The path to a file specifying filters in JSON as specified in the filters --help

    Returns:
        The parsed metric mapping linking each metric to a specified operation
    """
    def null(df, **_kwargs):
        """Do nothing and return a passed DataFrame"""
        return df

    if file is not None:
        # parsed_filters = read_json(file)
        filters = to_iterable(file)

    # Parse input filters as individual filtering directives
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

        if not indices:
            raise InputError(
                f"Couldn't create a filter from '{filter_str}'. Ensure that your input contains an operation specifying"
                " the relationship between the filter and the expected value")

        # Sort the indices according to the first index number
        full_indices = []
        for idx_pair in sorted(indices, key=lambda pair: pair[0]):
            full_indices.extend(idx_pair)

        # Separate the filter components along the operations
        unique_components = [filter_str[:full_indices[0]]] \
            + [filter_str[f_index:r_index] for f_index, r_index in zip(full_indices[:-1], full_indices[1:])] \
            + [filter_str[full_indices[-1]:]]

        # Set up function to parse values properly
        def extract_format_value(_value: str) -> Any:
            """Values can be of type int, float, or string. Further, strings could be a list of int/float
            comma separated

            Args:
                _value: The string to format

            Returns:
                The value formatted from a string input to the correct python type for filter evaluation
            """
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
                logger.debug(f'substituted_component |{substituted_component}|')
                _metric_specs = config.metrics.get(substituted_component.strip())
                if _metric_specs:  # We found in the list of available program metrics
                    logger.debug(f'metric specifications {",".join(f"{k}={v}" for k, v in _metric_specs.items())}')
                    if metric_idx is None:
                        metric_specs = _metric_specs
                        metric = component.strip()
                        metric_idx = idx
                        # if idx != 0:
                        # We must negate operations until the metric is found, i.e. metric > 1 is expected, but
                        # 1 < metric < 10 is allowed. This becomes metric > 1 and metric < 10
                        invert_ops = False
                        continue
                    else:  # We found two metrics...
                        raise ValueError(
                            f"Can't accept more than one metric name per filter")
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

                logger.debug(f'operation is {operation}')
                parsed_operations.append(operation)

        if metric_idx is None:
            possible_metrics = []
            for value in parsed_values:
                if isinstance(value, str):
                    possible_metrics.append(value)
            # Todo find the closest and report to the user!
            raise InputError(
                f"Couldn't coerce{' any of' if len(possible_metrics) > 1 else ''} '{', '.join(possible_metrics)}' to a"
                f" viable metric. Ensure you used the correct spelling")

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


# Retrieved from 'https://stackoverflow.com/questions/29986185/python-argparse-dict-arg'
class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser_, namespace, values, option_string=None):
        # print(f'{StoreDictKeyPair.__name__} received values: {values}')
        # # May have to use this if used as a subparser
        # my_dict = getattr(namespace, self.dest, {})
        # input(f'existing my_dict: {my_dict}')
        my_dict = {}
        for kv in values:
            k, v = kv.split('=')
            my_dict[k] = ast.literal_eval(v)

        # print(f'Final parsed: {my_dict}')
        setattr(namespace, self.dest, my_dict)


# The help strings can include various format specifiers to avoid repetition of things like the program name or the
# argument default. The available specifiers include the program name, %(prog)s and most keyword arguments to
# add_argument(), e.g. %(default)s, %(type)s, etc.:
# Todo Found the following for formatting the prog use case in subparsers
#  {'refine': ArgumentParser(prog='{putils.program_command} module [module_arguments] [input_arguments]'
#                                 '[optional_arguments] refine'
# make_no_argument = '--no-{}'.format
boolean_positional_prevent_msg = 'Use --no-{} to prevent'.format
"""Use this message in all help keyword arguments using argparse.BooleanOptionalAction with default=True to specify the
 --no- prefix when the argument should be False
"""
symmetry_title = 'Symmetry Arguments'
options_title = 'Options Arguments'
design_selector_title = 'Design Selector Arguments'
input_title = 'Input Arguments'
output_title = 'Output Arguments'
module_title = 'Module Arguments'


def arg_cat_usage(title):
    return f'{argparse_flag_delimiter}{argparse_flag_delimiter}{argparse_flag_delimiter.join(title.lower().split())}'


module_usage_str = \
    f'\n      {program_command} ' \
    '{} ' \
    f'[{arg_cat_usage(input_title)}][{arg_cat_usage(symmetry_title)}]' \
    f'[{arg_cat_usage(output_title)}][{arg_cat_usage(options_title)}]' \
    f'[{arg_cat_usage(design_selector_title)}]'
usage_str = module_usage_str.format(f'module [{arg_cat_usage(module_title)}]')

# Reused arguments
distribute_args = ('-D', distribute_work.long)
use_proteinmpnn_args = (use_proteinmpnn.long,)
use_proteinmpnn_kwargs = dict(action=argparse.BooleanOptionalAction, default=True,
                              help='Whether to perform calculations with ProteinMPNN\n'
                                   'sequences/profile during the job\n'
                                   f'{boolean_positional_prevent_msg(use_evolution)}')
use_evolution_args = (use_evolution.long,)
use_evolution_kwargs = dict(action=argparse.BooleanOptionalAction, default=True,
                            help='Whether to perform calculations with an evolution profile\n'
                                 'during the job. Will create one if not made.\n'
                                 f'{boolean_positional_prevent_msg(use_evolution)}')
evolution_constraint_args = ('-ec', evolution_constraint.long)
evolution_constraint_kwargs = dict(action=argparse.BooleanOptionalAction, default=True,
                                   help='Whether to include evolutionary constraints during design.\n'
                                        f'{boolean_positional_prevent_msg(evolution_constraint)}')
term_constraint_args = ('-tc', term_constraint.long)
term_constraint_kwargs = dict(action=argparse.BooleanOptionalAction, default=True,
                              help='Whether to include tertiary motif constraints during design.\n'
                                   f'{boolean_positional_prevent_msg(term_constraint)}')
guide_args = (guide.long,)
guide_kwargs = dict(action='store_true', help=f'Display guides for the full {program_name} and specific modules\n'
                                              f"Ex: '{program_command} {guide.long}'\nor: '{submodule_guide}'")
help_args = ('-h', '-help', '--help')
help_kwargs = dict(action='store_true', help=f"Display argument help\nEx: '{program_command} --help'")
clash_distance_args = (clash_distance.long,)
clash_criteria_args = (clash_criteria.long,)
ignore_clashes_args = ('-ic', ignore_clashes.long)
ignore_pose_clashes_args = ('-ipc', ignore_pose_clashes.long)
ignore_symmetric_clashes_args = ('-isc', ignore_symmetric_clashes.long)
output_directory_args = ('-Od', '--outdir', output_directory.long)
output_directory_kwargs = dict(type=os.path.abspath, dest='output_directory',
                               help='If provided, the name of the directory to output all created files.\n'
                                    'Otherwise, one will be generated based on the time, input, and module')
output_file_args = ('-Of', output_file.long)
quick_args = (quick.long,)
# ---------------------------------------------------
all_flags_help = f'Display all {program_name} flags'
parser_all_flags = dict(description=all_flags_help, add_help=False)  # help=all_flags_help,
parser_all_flags_group = dict(description=f'\n{all_flags_help}')
# ---------------------------------------------------
symmetry_args = ('-S', symmetry.long)
symmetry_kwargs = dict(metavar='RESULT:{GROUP1}{GROUP2}...',
                       help='The specific symmetry of the poses of interest. Preferably\n'
                            'in a composition formula such as T:{C3}{C3}... Can also\n'
                            "provide the keyword 'cryst' to use crystal symmetry")
sym_entry_args = ('-E', sym_entry.long, '--entry')
sym_entry_kwargs = dict(type=int, metavar='INT',
                        help=f'The entry number of {nanohedra.title()} docking combinations to use.\n'
                             f'See {symmetry} {query.long} for possible symmetries')
# ---------------------------------------------------
symmetry_help = 'Specify a symmetric system in which to model inputs. Default module use performs a symmetry query'
parser_symmetry = dict(description=symmetry_help, help=symmetry_help)
parser_symmetry_group = dict(title=f'{"_" * len(symmetry_title)}\n{symmetry_title}',
                             description=f'\n{symmetry_help}')
sym_query_args = (query.long,)
symmetry_arguments = {
    sym_entry_args: sym_entry_kwargs,
    symmetry_args: symmetry_kwargs,
    sym_query_args: dict(choices=query_mode_args,  # required=True
                         help='Query the symmetries available for modeling\n'
                              'Choices=%(choices)s'),
    (nanohedra.long,): dict(action='store_true',
                            help=f'True if only {nanohedra.title()} docking symmetries should be queried')
}
# ---------------------------------------------------
options_help = 'Control runtime considerations and miscellaneous\n' \
               'program execution options'
parser_options = dict(description=options_help, help=options_help)
parser_options_group = dict(title=f'{"_" * len(options_title)}\n{options_title}',
                            description=f'\n{options_help}')
cores_args = (cores.long,)
multiprocessing_args = ('-M', multi_processing.long)
multiprocessing_kwargs = dict(action='store_true', help='Should job be run with multiple processors?')
project_name_args = (project_name.long,)
proteinmpnn_models = ['v_48_002', 'v_48_010', 'v_48_020', 'v_48_030']
proteinmpnn_model_args = (proteinmpnn_model.long,)
proteinmpnn_model_kwargs = dict(choices=proteinmpnn_models, default='v_48_020', metavar='',
                                help='The name of the model to use for ProteinMPNN design/scoring\n'
                                     'where the model name takes the form v_X_Y, with X indicating\n'
                                     'The number of neighbors, and Y indicating the training noise\n'
                                     'Choices=%(choices)s\nDefault=%(default)s')
setup_args = (setup.long,)
setup_kwargs = dict(action='store_true', help=f'Show set up instructions')
options_arguments = {
    cores_args: dict(type=int, default=cpu_count(logical=False) - 1, metavar='INT',
                     help=f'Number of cores to use with {format_args(multiprocessing_args)}\n'
                          'If run on a cluster, cores will reflect the cluster allocation,\n'
                          'otherwise, will use #physical_cores-1\nDefault=%(default)s (this computer)'),
    # ('--database',): dict(action=argparse.BooleanOptionalAction, default=True,
    #                       help=f'Whether to utilize the SQL database for result processing\n'
    #                            f'{boolean_positional_prevent_msg("database")}'),
    (database_url.long,): dict(help='The location/server used to connect to a SQL database.\nOnly used '
                                    f'during initial {program_output} directory set up.\nSubsequent jobs '
                                    'will use the same url'),
    (development.long,): dict(action='store_true',
                              help="Run in development mode. Only use if you're actively\n"
                                   'developing and understand the side effects'),
    use_evolution_args: use_evolution_kwargs,
    use_proteinmpnn_args: use_proteinmpnn_kwargs,
    distribute_args: dict(action='store_true',
                          help='Should individual jobs be formatted for distribution across\n'
                               'computational resources? This is useful on a cluster\nDefault=%(default)s'),
    ('-F', force.long): dict(action='store_true', help='Force generation of new files for existing projects'),
    guide_args: guide_kwargs,
    (interface_distance.long,): dict(type=float, default=9.0, metavar='FLOAT',
                                     help='The default value to use for querying Cb-Cb\n'
                                          'residue contacts across and interface'),
    ('-i', fragment_source.long): dict(type=str.lower, choices=fragment_dbs, default=biological_interfaces,
                                       metavar='',
                                       help='Database to match fragments for interface specific scoring matrices'
                                            '\nChoices=%(choices)s\nDefault=%(default)s'),
    clash_distance_args:
        dict(type=float, default=default_clash_distance, metavar='FLOAT',
             help='What distance should be used for clash checking?\nDefault=%(default)s'),
    clash_criteria_args:
        dict(default=default_clash_criteria, choices=coords_types,
             help='What type of atom should be used for clash checking?'),
    ignore_clashes_args:
        dict(action='store_true', help='Ignore ANY backbone/Cb clashes found during clash checks'),
    ignore_pose_clashes_args:
        dict(action='store_true', help='Ignore asu/pose clashes found during clash checks'),
    ignore_symmetric_clashes_args:
        dict(action='store_true', help='Ignore symmetric clashes found during clash checks'),
    (log_level.long,): dict(type=log_levels.get, default=default_logging_level, choices=logging_levels, metavar='',
                            help='What level of log messages should be displayed to stdout?'
                                 '\n1-debug, 2-info, 3-warning, 4-error, 5-critical\nDefault=%(default)s'),
    (mpi.long,): dict(type=int, metavar='INT',
                      help='If commands should be run as MPI parallel processes,\n'
                           'how many processes should be invoked for each job?\nDefault=%(default)s'),
    multiprocessing_args: multiprocessing_kwargs,
    project_name_args: dict(help='If desired, the name of the initialized project\n'
                                 'Default is inferred from input'),
    proteinmpnn_model_args: proteinmpnn_model_kwargs,
    setup_args: setup_kwargs,
    (skip_logging.long,): dict(action='store_true',
                               help='Skip logging to files and direct all logging to stream'),
    (profile_memory.long,): dict(action='store_true',
                                 help='Use memory_profiler.profile() to understand memory usage of a module.\n'
                                      'Must be run with --development'),
    quick_args: dict(action='store_true',
                     help='Run Nanohedra in minimal sampling mode to generate enough hits to\n'
                          'test quickly. This should only be used for active development'),
    (debug_db.long,): dict(action='store_true', help='Whether to log SQLAlchemy output for db development'),
    (reset_db.long,): dict(action='store_true', help='Whether to reset the database for development')
}
# ---------------------------------------------------
residue_selector_help = 'Residue selectors control which parts of the Pose are included during protocols'
parser_residue_selector = dict(description=residue_selector_help, help=residue_selector_help)
parser_residue_selector_group = dict(title=f'{"_" * len(design_selector_title)}\n{design_selector_title}',
                                     description=f'\n{residue_selector_help}')
residue_selector_arguments = {
    (design_chains.long,):
        dict(metavar='',
             help="If design should ONLY occur on certain chains, specify\n"
                  "the chain ID's as a comma separated string\n"
                  "Ex: 'A,C,D'"),
    (mask_chains.long,):
        dict(metavar='',
             help="If a design should NOT occur at certain chains, provide\n"
                  "the chain ID's as a comma separated string\n"
                  "Ex: 'B,C'"),
    (require_chains.long,):
        dict(metavar='',
             help='If design MUST occur on certain chains, specify their\n'
                  "chain ID's as a comma separated string.\n"
                  "Ex: 'A,D'"),
    (design_residues.long,):  # Todo make a ChainResidue parser like A27-35,B118,B281
        dict(metavar='',
             help='If design should ONLY occur at certain residues, specify\n'
                  'their numbers as a comma separated, range string\n'
                  "Ex: '23,24,35,41,100-110,267,289-293'"),
    (mask_residues.long,):  # Todo make a ChainResidue parser like A27-35,B118,B281
        dict(metavar='',
             help='If design should NOT occur at certain residues, specify\n'
                  'their numbers as a comma separated, range string\n'
                  "Ex: '27-35,118,281'"),
    (require_residues.long,):  # Todo make a ChainResidue parser like A27-35,B118,B281
        dict(metavar='',
             help='If design MUST occur at certain residues, specify their\n'
                  'numbers as a comma separated, range string\n'
                  "Ex: '23,24,35,41,100-110,267,289-293'"),
}
# ('--design-by-sequence',):
#     dict(help='If design should occur ONLY at certain residues, specify\nthe location of a .fasta file '
#               f'containing the design selection\nRun "{program_command} --single my_pdb_file.pdb design_selector"'
#               ' to set this up'),
# ('--mask-by-sequence',):
#     dict(help='If design should NOT occur at certain residues, specify\nthe location of a .fasta file '
#               f'containing the design mask\nRun "{program_command} --single my_pdb_file.pdb design_selector" '
#               'to set this up'),
# ---------------------------------------------------
# Set Up SubModule Parsers
# ---------------------------------------------------
# module_parser = argparse.ArgumentParser(add_help=False)  # usage=usage_str,
# ---------------------------------------------------
protocol_help = 'Perform a series of modules in a specified order'
parser_protocol = dict(description=protocol_help, help=protocol_help)
protocol_arguments = {
    ('-m', modules.long): dict(nargs='*', default=tuple(), required=True, help='The modules to run in order'),
}
# ---------------------------------------------------
predict_pose_args = (predict_pose.long,)
predict_structure_help = 'Predict 3D structures from specified sequences'
parser_predict_structure = dict(description=f'{predict_structure_help}\nPrediction occurs on designed sequences by '
                                            f'default.\nIf prediction should be performed on the Pose, use '
                                            f'{format_args(predict_pose_args)}', help=predict_structure_help)
predict_structure_arguments = {
    ('-m', predict_method.long):
        dict(choices={'alphafold', 'thread'}, default='alphafold', metavar='',
             help=f'The method utilized to {predict_structure}\nChoices=%(choices)s\nDefault=%(default)s'),
    (num_predictions_per_model.long, '--number-predictions-per-model'):  # '-n',
        dict(type=int, metavar='INT',  # default=5,
             help=f'How many iterations of prediction should be used\nfor each individual Alphafold model.\n'
                  'Default=5(multimer mode),1(monomer mode)'),
    ('-A', predict_assembly.long):
        dict(action='store_true', help='Whether the assembly state should be predicted\ninstead of the ASU'),
    (predict_designs.long,):
        dict(action=argparse.BooleanOptionalAction, default=True,
             help='Whether the full design state should be predicted\nincluding all entities\n'
                  f'{boolean_positional_prevent_msg(predict_designs)}'),
    ('-E', predict_entities.long):
        dict(action='store_true', help='Whether individual entities should be predicted\ninstead of the entire Pose'),
    predict_pose_args:
        dict(action='store_true', help='Whether individual entities should be predicted\ninstead of the entire Pose'),
    (models_to_relax.long,):
        dict(type=str.lower, default='best', metavar='',
             choices=config.relax_options, help='Specify which predictions should be relaxed'
                                                '\nChoices=%(choices)s\nDefault=%(default)s'),
    (use_gpu_relax.long,):
        dict(action='store_true', help='Whether predictions should be relaxed using a GPU (if one is available)'),
}
# ---------------------------------------------------
# orient_help = 'Orient a symmetric assembly in a canonical orientation at the origin'
# parser_orient = dict(description=orient_help, help=orient_help)
# orient_arguments = {}
# ---------------------------------------------------
helix_bending_help = 'Bend helices along known modes of helical flexibility'
parser_helix_bending = dict(description=helix_bending_help, help=helix_bending_help)
joint_residue_args = (joint_residue.long,)
sample_number_args = (sample_number.long,)
sample_number_kwargs = dict(type=int, default=10, metavar='INT',
                            help='How many times should the bending be performed?\nDefault=%(default)s')
possible_termini = get_args(termini_literal)
helix_bending_arguments = {
    (direction.long,): dict(type=str.lower, metavar='', choices=possible_termini,  # required=True,
                            help='Which direction should the bending be applied?\n'
                                 f"Choices=%(choices)s. Where 'c' would imply residues c-terminal\n"
                                 f"to {format_args(joint_residue_args)} will be bent"),
    joint_residue_args: dict(type=int, metavar='INT', required=True,
                             help='The residue number to perform the bending at'),
    (joint_chain.long,): dict(required=True, help='The chain where the bending is desired at'),
    sample_number_args: sample_number_kwargs
}
# ---------------------------------------------------
align_helices_help = 'Align, then fuse, the helices of two protein systems. The aligned\n' \
                     'molecule is transformed to the reference frame of the target\n' \
                     'molecule. All symmetry reflects the target while the aligned\n' \
                     'component must be asymmetric'
# To align helices where both components are symmetric, run Nanohedra to perform helical alignment in the available
# Nanohedra symmetry combination materials (SCM)
parser_align_helices = dict(description=align_helices_help, help=align_helices_help)
target_start_args = (target_start.long,)
target_start_kwargs = dict(type=int, metavar='INT', help='First residue of the target molecule to align on')
target_end_args = (target_end.long,)
target_end_kwargs = dict(type=int, metavar='INT', help='Last residue of the target molecule to align on')
target_chain_args = (target_chain.long,)
target_chain_kwargs = dict(help='A desired chainID of the target molecule')
target_termini_args = (target_termini.long,)
target_termini_kwargs = dict(type=str.lower, nargs='*', choices=possible_termini,
                             help="If particular termini of the target are desired,\n"
                                  "specify with 'n' and/or 'c'")
trim_termini_args = (trim_termini.long,)
trim_termini_kwargs = dict(action=argparse.BooleanOptionalAction, default=True,
                           help='Whether the termini should be trimmed of irregularly\nstructured residues\n'
                                f'{boolean_positional_prevent_msg(trim_termini)}')
aligned_start_args = (aligned_start.long,)
aligned_start_kwargs = dict(type=int, metavar='INT', help='First residue of the aligned molecule to align on')
aligned_end_args = (aligned_end.long,)
aligned_end_kwargs = dict(type=int, metavar='INT', help='Last residue of the aligned molecule to align on')
aligned_chain_args = (aligned_chain.long,)
aligned_chain_kwargs = dict(help='A desired chainID of the aligned molecule')
alignment_length_args = (alignment_length.long,)  # length.long,
alignment_length_kwargs = dict(type=int, metavar='INT', help='The number of residues used to measure overlap')
extend_args = (extend.long,)
# extend_kwargs = dict(action='store_true',
#                      help='Whether to extend alignment termini with a ten residue ideal\n'
#                           'alpha helix. All specified residues are modified accordingly\n')
extend_kwargs = dict(type=int, metavar='INT',  # action='store_true',
                     help='Whether to extend target termini with an ideal alpha helix\n'
                          'Argument should specify how many residues to extend')
align_helices_arguments = {
    aligned_chain_args: aligned_chain_kwargs,
    aligned_end_args: aligned_end_kwargs,
    aligned_start_args: aligned_start_kwargs,
    alignment_length_args: alignment_length_kwargs,
    (bend.long,): dict(type=int, metavar='INT',  # action='store_true',
                       help=helix_bending_help
                       + '\nArgument should specify how many bent positions should be sampled'),
    extend_args: extend_kwargs,
    # sample_number_args: sample_number_kwargs,
    target_chain_args: target_chain_kwargs,
    target_end_args: target_end_kwargs,
    target_start_args: target_start_kwargs,
    target_termini_args: target_termini_kwargs,
    trim_termini_args: trim_termini_kwargs,
}
# ---------------------------------------------------
query_codes_kwargs = dict(action='store_true', help='Query the PDB API for corresponding codes')
# parser_component_mutual1 = parser_dock.add_mutually_exclusive_group(required=True)
component1_args = ('-c1', component1.long)
component2_args = ('-c2', component2.long)
align_component1_args = (*component1_args, f'--target')
align_component2_args = (*component2_args, f'--aligned')
# Todo make multiple files?
component_kwargs = dict(type=os.path.abspath, metavar=ex_path('[file.ext,directory]'),
                        help='Path to component file, either directory or single file')
pdb_codes_args = ('-C1', pdb_code.long, f'--{pdb_code}s', f'--{pdb_code}1', f'--{pdb_code}s1')
pdb_codes2_args = ('-C2', f'--{pdb_code}2', f'--{pdb_code}s2')
query_pdb_codes_args = ('-Q', query_codes.long)
query_pdb_codes2_args = ('-Q2', f'--{query_codes}2')
align_pdb_codes1_args = (*pdb_codes_args, f'--target-{pdb_code}', f'--target-{pdb_code}s')
align_pdb_codes2_args = (*pdb_codes2_args, f'--aligned-{pdb_code}', f'--aligned-{pdb_code}s')
pdb_codes_kwargs = dict(nargs='*',  default=tuple(),
                        help='Accession code(s) OR the path to file(s) containing codes\n'
                             'where each code is a PDB EntryID|EntityID|AssemblyID')
parser_component_mutual1_group = dict()  # required=True <- adding kwarg below to different parsers depending on need
component_mutual1_arguments = {
    component1_args: component_kwargs,
    pdb_codes_args: pdb_codes_kwargs,
    query_pdb_codes_args: query_codes_kwargs
}
# parser_component_mutual2 = parser_dock.add_mutually_exclusive_group()
parser_component_mutual2_group = dict()  # required=False
component_mutual2_arguments = {
    component2_args: component_kwargs,
    pdb_codes2_args: pdb_codes_kwargs,
    query_pdb_codes2_args: query_codes_kwargs
}
align_component_mutual1_arguments = component_mutual1_arguments.copy()
align_component_mutual1_arguments[align_component1_args] = align_component_mutual1_arguments.pop(component1_args)
align_component_mutual1_arguments[align_pdb_codes1_args] = align_component_mutual1_arguments.pop(pdb_codes_args)
align_component_mutual2_arguments = component_mutual2_arguments.copy()
align_component_mutual2_arguments[align_component2_args] = align_component_mutual2_arguments.pop(component2_args)
align_component_mutual2_arguments[align_pdb_codes2_args] = align_component_mutual2_arguments.pop(pdb_codes2_args)
# ---------------------------------------------------
measure_pose_args = (measure_pose.long,)
measure_pose_kwargs = dict(action='store_true', help=f'Whether the pose should be included in measurements')
refine_help = 'Process structures into an energy function'
parser_refine = dict(description=refine_help, help=refine_help)
refine_arguments = {
    ('-ala', interface_to_alanine.long): dict(action=argparse.BooleanOptionalAction, default=False,
                                              help='Whether to mutate interface residues to alanine before '
                                                   'refinement\n'),
    measure_pose_args: measure_pose_kwargs,
    ('-met', metrics.long): dict(action=argparse.BooleanOptionalAction, default=True,
                                 help='Whether to calculate interface metrics after refinement\n'
                                      f'{boolean_positional_prevent_msg(metrics)}')
}
# ---------------------------------------------------
nanohedra_help = f'Run {nanohedra.title()}.py'
parser_nanohedra = dict(description=nanohedra_help, help=nanohedra_help)
default_perturbation_steps = 3
dock_filter_args = (dock_filter.long,)
dock_filter_kwargs = dict(nargs='*', default=tuple(), help='Whether to filter dock trajectory according to metrics')
dock_filter_file_args = (dock_filter_file.long,)
dock_filter_file_kwargs = dict(type=os.path.abspath,
                               help='Whether to filter dock trajectory according to metrics provided in a file')
dock_weight_args = (dock_weight.long,)
dock_weight_kwargs = dict(nargs='*', default=tuple(), help='Whether to filter dock trajectory according to metrics')
dock_weight_file_args = (dock_weight_file.long,)
dock_weight_file_kwargs = dict(type=os.path.abspath,
                               help='Whether to filter dock trajectory according to metrics provided in a file')
nanohedra_arguments = {
    (contiguous_ghosts.long,): dict(action='store_true',  # argparse.BooleanOptionalAction, default=False,
                                    help='Whether to prioritize docking with ghost fragments that form continuous'
                                         '\nsegments on a single component\nDefault=%(default)s'),
    dock_filter_args: dock_filter_kwargs,
    dock_filter_file_args: dock_filter_file_kwargs,
    dock_weight_args: dock_weight_kwargs,
    dock_weight_file_args: dock_weight_file_kwargs,
    ('-iz', initial_z_value.long): dict(type=float, default=1.,
                                        help='The standard deviation z-score threshold for initial fragment overlap\n'
                                             'Smaller values lead to more stringent matching overlaps\n'
                                             'Default=%(default)s'),
    ('-mv', match_value.long):
        dict(type=float, metavar='FLOAT', default=0.5,
             help='What is the minimum match score required for a high quality fragment?\n'
                  'Lower values more poorly overlap with a perfect overlap being 1'),
    (minimum_matched.long, min_matched.long):
        dict(type=int, metavar='INT', default=3,
             help='How many high quality fragment pairs are required for a Pose to pass?\nDefault=%(default)s'),
    (only_write_frag_info.long,): dict(action=argparse.BooleanOptionalAction, default=False,
                                       help='Used to write fragment information to a directory for C1 based docking'),
    output_directory_args:
        dict(type=os.path.abspath, default=None,
             help='Where should the output be written?\nDefault='
                  f'{ex_path(program_output, projects, "NanohedraEntry[ENTRYNUMBER]_[BUILDING-BLOCKS]_Poses")}'),
    (perturb_dof.long,): dict(action=argparse.BooleanOptionalAction, default=False,
                              help='Whether the degrees of freedom should be finely sampled during\n by perturbing '
                                   'found transformations and repeating docking iterations'),
    (perturb_dof_rot.long,): dict(action=argparse.BooleanOptionalAction, default=False,
                                  help='Whether the rotational degrees of freedom should be finely sampled in\n'
                                       'subsequent docking iterations'),
    (perturb_dof_tx.long,): dict(action=argparse.BooleanOptionalAction, default=False,
                                 help='Whether the translational degrees of freedom should be finely sampled in\n'
                                      'subsequent docking iterations'),
    # These have no default since None is used to signify whether they were explicitly requested
    (perturb_dof_steps.long,): dict(type=int, metavar='INT',
                                    help='How many dof steps should be used during subsequent docking iterations.\n'
                                         f'For each DOF, a total of --{perturb_dof_steps} will be sampled during '
                                         f'perturbation\nDefault={default_perturbation_steps}'),
    (perturb_dof_steps_rot.long,): dict(type=int, metavar='INT',
                                        help='How many rotational dof steps should be used during perturbations\n'
                                             f'Default={default_perturbation_steps}'),
    (perturb_dof_steps_tx.long,): dict(type=int, metavar='INT',
                                       help='How many translational dof steps should be used during perturbations\n'
                                            f'Default={default_perturbation_steps}'),
    (proteinmpnn_score.long,): dict(action=argparse.BooleanOptionalAction, default=False,
                                    help='Whether docking fit should be measured using ProteinMPNN'),
    ('-r1', rotation_step1.long): dict(type=float, metavar='FLOAT', default=3.,
                                       help='The size of degree increments to search during initial rotational\n'
                                            'degrees of freedom search\nDefault=%(default)s'),
    ('-r2', rotation_step2.long): dict(type=float, metavar='FLOAT', default=3.,
                                       help='The size of degree increments to search during initial rotational\n'
                                            'degrees of freedom search\nDefault=%(default)s'),
    trim_termini_args: trim_termini_kwargs,
}
# parser_nanohedra_run_type_mutual_group = dict()  # required=True <- adding to parsers depending on need below
# nanohedra_run_type_mutual_arguments = {
#     sym_entry_args: sym_entry_kwargs,
# }
# ---------------------------------------------------
initialize_building_blocks_help = 'Initialize building blocks for downstream Pose creation'
parser_initialize_building_blocks = dict(description=initialize_building_blocks_help,
                                         help=initialize_building_blocks_help)
initialize_building_blocks_arguments = {
    **component_mutual1_arguments,
    (update_metadata.long,): dict(nargs='*', action=StoreDictKeyPair,
                                  help='Whether ProteinMetadata should be update with some\n'
                                       'particular value in the database'),
}
# ---------------------------------------------------
cluster_map_args = ('-c', cluster_map.long)
cluster_map_kwargs = dict(type=os.path.abspath,
                          metavar=ex_path(default_clustered_pose_file.format('TIMESTAMP', 'LOCATION')),
                          help='The location of a serialized file containing spatially\nor interfacial '
                               'clustered poses')
cluster_selection_args = ('-Cs', cluster_selection.long)
cluster_selection_kwargs = dict(action='store_true',
                                help='Whether clustering should be performed using select-* results')
cluster_poses_help = 'Cluster all poses by their spatial or interfacial similarity. This is\n' \
                     'used to identify conformationally flexible docked configurations'
parser_cluster = dict(description=cluster_poses_help, help=cluster_poses_help)
cluster_poses_arguments = {
    (as_objects.long,): dict(action='store_true', help='Whether to store the resulting pose cluster file as '
                                                       'PoseJob objects\nDefault stores as pose IDs'),
    (cluster_mode.long,):
        dict(type=str.lower, choices={'ialign', 'rmsd', 'transform'}, default='transform', metavar='',
             help='Which type of clustering should be performed?\nChoices=%(choices)s\nDefault=%(default)s'),
    (cluster_number.long,):
        dict(type=int, default=1, metavar='INT', help='The number of cluster members to return'),
    output_file_args: dict(type=os.path.abspath,
                           help='Name of the output .pkl file containing pose clusters.\n'
                                f'Will be saved to the {data.title()} folder of the output.\n'
                                f'Default={default_clustered_pose_file.format("TIMESTAMP", "LOCATION")}')
}
# ---------------------------------------------------
ca_only_args = (ca_only.long,)
ca_only_kwargs = dict(action='store_true',
                      help='Whether a minimal CA variant of the protein should be used for design calculations')
neighbors_args = (neighbors.long,)
neighbors_kwargs = \
    dict(action='store_true', help='Whether the neighboring residues should be considered during sequence design')
design_method_args = ('-m', design_method.long)
design_programs: tuple[str, ...] = get_args(design_programs_literal)
design_method_kwargs = dict(type=str.lower, default=proteinmpnn, choices=design_programs, metavar='',
                            help='Which design method should be used?\nChoices=%(choices)s\nDefault=%(default)s')
hbnet_args = ('-hb', hbnet.long)
hbnet_kwargs = dict(action=argparse.BooleanOptionalAction, default=True,
                    help=f'Whether to include hydrogen bond networks in the design.'
                         f'\n{boolean_positional_prevent_msg(hbnet)}')
structure_background_args = ('-sb', structure_background.long)
structure_background_kwargs = dict(action='store_true',  # action=argparse.BooleanOptionalAction, default=False,
                                   help='Whether to skip all constraints and measure the structure\nusing only the '
                                        'selected energy function\nDefault=%(default)s')
design_number_args = ('-n', design_number.long)
nstruct = 20
design_number_kwargs = dict(type=int, default=nstruct, metavar='INT',
                            help='How many unique sequences should be generated for each input?\nDefault=%(default)s')
scout_args = ('-sc', scout.long)
scout_kwargs = dict(action='store_true',  # action=argparse.BooleanOptionalAction, default=False,
                    help='Whether to set up a low resolution scouting protocol to\n'
                         'survey designability\nDefault=%(default)s')


def temp_gt0(temp: str) -> float:
    """Convert temperatures flags to float ensuring no 0 value"""
    temp = float(temp)
    return temp if temp > 0 else 0.0001


temperature_args = ('-K', temperatures.long)
temperature_kwargs = dict(type=temp_gt0, nargs='*', default=(0.1,), metavar='FLOAT',
                          help="'Temperature', i.e. the value(s) to use as the denominator in\n"
                               'the equation: exp(G/T), where G=energy and T=temperature, when\n'
                               'performing design. Higher temperatures result in more diversity\n'
                               'each temperature must be > 0\nDefault=%(default)s')
design_help = 'Gather poses of interest and format for sequence design using Rosetta/ProteinMPNN.\n' \
              'Constrain using evolutionary profiles of homologous sequences\n' \
              'and/or fragment profiles extracted from the PDB, or neither'
parser_design = dict(description=design_help, help=design_help)
design_arguments = {
    ca_only_args: ca_only_kwargs,
    design_method_args: design_method_kwargs,
    design_number_args: design_number_kwargs,
    evolution_constraint_args: evolution_constraint_kwargs,
    hbnet_args: hbnet_kwargs,
    proteinmpnn_model_args: proteinmpnn_model_kwargs,
    structure_background_args: structure_background_kwargs,
    scout_args: scout_kwargs,
    temperature_args: temperature_kwargs,
    term_constraint_args: term_constraint_kwargs
}
interface_design_help = 'Gather poses of interest and format for interface specific sequence design using\n' \
                        'ProteinMPNN/Rosetta. Constrain using evolutionary profiles of homologous\n' \
                        'sequences and/or fragment profiles extracted from the PDB, or neither'
parser_interface_design = dict(description=interface_design_help, help=interface_design_help)
interface_design_arguments = {
    **design_arguments,
    neighbors_args: neighbors_kwargs,
}
# ---------------------------------------------------
interface_metrics_help = f'Analyze {interface_metrics} for each pose'
parser_metrics = dict(description=interface_metrics_help, help=interface_metrics_help)
interface_metrics_arguments = {
    measure_pose_args: measure_pose_kwargs,
    ('-sp', specific_protocol.long): dict(metavar='PROTOCOL',
                                          help='A specific type of design protocol to perform metrics on.\n'
                                               'If not provided, captures all design protocols')
}
# ---------------------------------------------------
poses_args = (poses.long,)
specification_file_args = ('-sf', specification_file.long)
use_specification_file_str = f'The input flag {format_args(specification_file_args)} can be provided\n' \
                             'to restrict selection to specific designs from each pose'
optimize_designs_help = f'Subtly and explicitly modify pose/designs. Useful for reverting\n' \
                        'mutations to wild-type, directing exploration of troublesome areas,\n' \
                        'stabilizing an entire design based on evolution, increasing solubility,\n' \
                        f'or modifying surface charge. {optimize_designs} is based on amino acid\n' \
                        f'frequency profiles. Use with {format_args(specification_file_args)} is suggested'
parser_optimize_designs = dict(description=optimize_designs_help, help=optimize_designs_help)
background_profile_args = ('-bg', background_profile.long)
optimize_designs_arguments = {
    background_profile_args: dict(type=str.lower, default=design_profile, metavar='',
                                  choices={design_profile, evolutionary_profile, fragment_profile},
                                  help='Which profile should be used as the background profile?\n'
                                       'Choices=%(choices)s\nDefault=%(default)s')
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
#     ('-n', '--native'): dict(choices=['source', 'asu_path', 'assembly_path', 'refine_pdb', 'refined_pdb',
#                                       'consensus_pdb', 'consensus_design_pdb'], default='refined_pdb',
#                              help='What structure to use as a "native" structure for Rosetta reference calculations\n'
#                                   'Default=%(default)s'),
#     ('--score-only',): dict(action='store_true', help='Whether to only score the design(s)\nDefault=%(default)s'),
#     ('script',): dict(type=os.path.abspath, help='The location of the custom script'),
#     ('-v', '--variables'): dict(nargs='*',
#                                 help='Additional variables that should be populated in the script.\nProvide a list of'
#                                      ' such variables with the format "variable1=value variable2=value". Where '
#                                      'variable1 is a RosettaScripts %%%%variable1%%%% and value is a known value. For'
#                                      'variables that must be calculated on the fly for each design, please modify '
#                                      'structure.model.py class to produce a method that can generate an attribute '
#                                      'with the specified name')
#     # Todo either a known value or an attribute available to the Pose object
# }
# ---------------------------------------------------
analysis_help = 'Analyze specified Pose/Designs to generate a suite of metrics'
parser_analysis = dict(description=analysis_help, help=analysis_help)
analysis_arguments = {
    # ('--figures',): dict(action=argparse.BooleanOptionalAction, default=False,
    #                      help='Create figures for all poses?'),
    # ('--merge',): dict(action='store_true', help='Whether to merge Trajectory and Residue Dataframes'),
    # ('--output',): dict(action=argparse.BooleanOptionalAction, default=True,
    #                     help=f'Whether to output the --{output_file}?\n{boolean_positional_prevent_msg("output")}'),
    # #                          '\nDefault=%(default)s'),
    # output_file_args: dict(help='Name of the output .csv file containing pose metrics.\nWill be saved to the '
    #                             f'{all_scores} folder of the output'
    #                             f'\nDefault={default_analysis_file.format("TIMESTAMP", "LOCATION")}'),
    # ('--save', ): dict(action=argparse.BooleanOptionalAction, default=True,
    #                    help=f'Save Trajectory and Residues dataframes?\n{boolean_positional_prevent_msg("save")}')
}
# ---------------------------------------------------
process_rosetta_metrics_help = 'Analyze all poses output from Rosetta metrics'
parser_process_rosetta_metrics = dict(description=process_rosetta_metrics_help, help=process_rosetta_metrics_help)
process_rosetta_metrics_arguments = {}
# ---------------------------------------------------
# Common selection arguments
# allow_multiple_poses_args = ('-amp', allow_multiple_poses.long)
# allow_multiple_poses_kwargs = dict(action='store_true',
#                                    help='Allow multiple designs to be selected from the same Pose when using --total'
#                                         '\nBy default, --total filters the selected designs by a single Pose')
csv_args = (csv.long,)
csv_kwargs = dict(action='store_true', help='Write the sequences file as a .csv instead of the default .fasta')
designs_per_pose_args = (designs_per_pose.long,)
designs_per_pose_kwargs = dict(type=int, metavar='INT', default=1,
                               help='What is the maximum number of designs to select from each pose?\n'
                                    'Default=%(default)s')
filter_file_args = (filter_file.long,)
filter_file_kwargs = dict(type=os.path.abspath, help='Whether to filter selection using metrics provided in a file')
filter_args = (filter_.long,)
all_filter_args = filter_args + filter_file_args
filter_kwargs = dict(nargs='*', default=tuple(), help='Whether to filter selection using metrics')  # default=None,
# filter_kwargs = dict(action='store_true', help='Whether to filter selection using metrics')
optimize_species_args = ('-opt', optimize_species.long)
optimize_species_kwargs = dict(default='e_coli', choices=get_args(optimization_species_literal), metavar='',
                               help='The organism where nucleotide usage should be optimized\n'
                                    'Choices=%(choices)s\nDefault=%(default)s')
output_structures_args = ('-Os', output_structures.long)
protocol_args = (protocol.long,)
protocol_kwargs = dict(nargs='*', default=tuple(), help='Use specific protocol(s) to filter designs?')
pose_select_number_kwargs = \
    dict(type=int, default=sys.maxsize, metavar='INT', help='Number to return\nDefault=No Limit')
save_total_args = (save_total.long,)
save_total_kwargs = dict(action='store_true', help='Should the total dataframe accessed by selection be saved?')
select_number_args = (select_number.long,)
select_number_kwargs = dict(type=int, default=sys.maxsize, metavar='INT',
                            help='Limit selection to a certain number. If not specified, returns all')
# total_args = ('--total',)
# total_kwargs = dict(action='store_true',
#                     help='Should selection be based on the total design pool?\n'
#                          'Searches for the top sequences from all poses, then\n'
#                          f'chooses one sequence/pose unless --{allow_multiple_poses} is invoked')
weight_file_args = (weight_file.long,)
weight_file_kwargs = dict(type=os.path.abspath,
                          help='Whether to weight selection results using metrics provided in a file')
weight_args = (weight.long,)
all_weight_args = weight_args + weight_file_args
weight_kwargs = dict(nargs='*', default=tuple(), help='Whether to weight selection results using metrics')  # default=None,
# weight_kwargs = dict(action='store_true', help='Whether to weight selection results using metrics')
weight_function_args = ('-wf', weight_function.long)
weight_function_kwargs = dict(type=str.lower, choices=config.metric_weight_functions, default='normalize', metavar='',
                              help='How to standardize metrics during selection weighting'
                                   '\nChoices=%(choices)s\nDefault=%(default)s')
# ---------------------------------------------------
select_arguments = {
    cluster_selection_args: cluster_selection_kwargs,
    filter_args: filter_kwargs,
    filter_file_args: filter_file_kwargs,
    select_number_args: select_number_kwargs,
    output_structures_args:
        dict(action='store_true', help='For the selection, should the corresponding structures\n'
                                       'be written to the output directory?'),
    protocol_args: protocol_kwargs,
    save_total_args: save_total_kwargs,
    # total_args: total_kwargs,
    weight_args: weight_kwargs,
    weight_file_args: weight_file_kwargs,
    weight_function_args: weight_function_kwargs,
}
# ---------------------------------------------------
select_poses_help = f'Select poses based on specific metrics. Use {format_args(all_filter_args)}\n' \
                    f'and/or {format_args(all_weight_args)}. ' \
                    f'For metric options, see {analysis} {format_args(guide_args)}.\n{use_specification_file_str}'
parser_select_poses = dict(description=select_poses_help, help=select_poses_help)
select_poses_arguments = {
    **select_arguments,
    select_number_args: pose_select_number_kwargs,
    # # parser_filter_mutual = parser_select_poses.add_mutually_exclusive_group(required=True)
    # parser_select_poses_mutual_group = dict(required=True)
    # parser_select_poses_mutual_arguments = {
    #     ('-m', '--metric'): dict(type=str.lower, choices=['score', 'fragments_matched'], metavar='', default='score',
    #                              help='If a single metric is sufficient, which metric to sort by?'
    #                                   '\nChoices=%(choices)s\nDefault=%(default)s'),
}
# ---------------------------------------------------
intergenic_sequence_args = ('-ms', multicistronic_intergenic_sequence.long)
intergenic_sequence_kwargs = dict(default=constants.ncoI_multicistronic_sequence,
                                  help='The sequence to use in the intergenic region of a multicistronic expression '
                                       'output')
nucleotide_args = (nucleotide.long,)
select_sequences_help = 'Select designs and output their sequences (nucleotide/protein) based on\n' \
                        'selection criteria and metrics. Generation of output sequences can take\n' \
                        'multiple forms depending on downstream needs. By default, disordered\n' \
                        'region sequence insertion, expression tagging, and codon optimization\n' \
                        f'({format_args(nucleotide_args)}) are performed.\n{use_specification_file_str}'
multicistronic_args = {
    csv_args: csv_kwargs,
    intergenic_sequence_args: intergenic_sequence_kwargs,
    optimize_species_args: optimize_species_kwargs,
}
_select_designs_arguments = {
    # allow_multiple_poses_args: allow_multiple_poses_kwargs,
    designs_per_pose_args: designs_per_pose_kwargs,
    output_directory_args:
        dict(type=os.path.abspath,
             help=f'Where should the output be written?\nDefault={ex_path(os.getcwd(), "SelectedDesigns")}'),
}
tagging_literal = Literal['all', 'none', 'single']
tagging_args: tuple[str, ...] = get_args(tagging_literal)
parser_select_sequences = dict(description=select_sequences_help, help=select_sequences_help)
select_sequences_arguments = {
    **select_arguments,
    **_select_designs_arguments,
    ('-ath', avoid_tagging_helices.long):
        dict(action='store_true', help='Should tags be avoided at termini with helices?'),
    ('-m', multicistronic.long):
        dict(action='store_true',
             help='Should nucleotide sequences by output in multicistronic format?\nBy default, uses the pET-Duet '
                  'intergeneic sequence containing\na T7 promoter, LacO, and RBS'),
    nucleotide_args: dict(action=argparse.BooleanOptionalAction, default=True,
                          help='Should codon optimized nucleotide sequences be output?'
                               f'\n{boolean_positional_prevent_msg(nucleotide)}'),
    ('-t', preferred_tag.long): dict(type=str.lower, choices=constants.expression_tags.keys(), default='his_tag',
                                     metavar='', help='The name of your preferred expression tag\n'
                                                      'Choices=%(choices)s\nDefault=%(default)s'),
    (tag_entities.long,): dict(type=str.lower,  # choices=tagging_args,
                               help='If there are specific entities in the designs you want to tag,\n'
                                    'indicate how tagging should occur. Choices:\n\t'
                                    '"single" - a single entity\n\t'
                                    '"all" - all entities\n\t'
                                    '"none" - no entities\n\t'
                                    'comma separated list such as '
                                    '"1,0,1" where\n'
                                    '\t    - "1" indicates a tag is required\n'
                                    '\t    - "0" indicates no tag is required'),
    (tag_linker.long,): dict(type=str.upper,  # metavar='', choices=constants.expression_tags.keys(),
                             help='The amino acid sequence of the linker region between each\n'
                                  f'expression tag and the protein\nDefault={constants.default_tag_linker}'),
    **multicistronic_args,
}
# ---------------------------------------------------
select_designs_help = f'Select designs based on specified criteria using metrics.\n{use_specification_file_str}'
parser_select_designs = dict(description=select_designs_help, help=select_designs_help)
select_designs_arguments = {
    **select_arguments,
    **_select_designs_arguments,
    csv_args: csv_kwargs,
}
# ---------------------------------------------------
file_args = ('-f', file.long)
multicistronic_help = 'Generate nucleotide sequences for selected designs by codon\n' \
                      'optimizing protein sequences, then concatenating nucleotide\n' \
                      f'sequences. Either .csv or .fasta file accepted with {format_args(file_args)}'
parser_multicistronic = dict(description=multicistronic_help, help=multicistronic_help)
number_args = ('-n', number.long)
multicistronic_arguments = {
    **multicistronic_args,
    number_args: dict(type=int, metavar='INT', help='The number of protein sequences to concatenate into a '
                                                    'multicistronic expression output'),
    output_directory_args: dict(type=os.path.abspath, required=True, help='Where should the output be written?'),
}
parser_update_db = dict()
update_db_arguments = {}
# distribute = 'distribute'
# parser_distribute = {distribute: dict()}
# distribute_arguments = {
#     (f'--command',): dict(help='The command to distribute over the input specification'),  # required=True,
#     (number_of_commands.long,): dict(type=int, required=True, help='The number of commands to spawn'),
#     output_file_args: dict(required=True, help='The location to write the commands')
# }
# ---------------------------------------------------
check_clashes_help = 'Check for any clashes in the input poses.\n' \
                     'This is always performed by default at Pose load and will \n' \
                     'raise a ClashError if clashes are found'
parser_check_clashes = dict(description=check_clashes_help, help=check_clashes_help)
# ---------------------------------------------------
# parser_check_unmodeled_clashes = subparsers.add_parser('check_unmodeled_clashes', description='Check for clashes between full models. Useful for understanding if loops are missing, whether their modeled density is compatible with the pose')
# ---------------------------------------------------
expand_asu_help = 'For given poses, expand the asymmetric unit to the full symmetric\n' \
                  'assembly and write the result'
parser_expand_asu = dict(description=expand_asu_help, help=expand_asu_help)
# ---------------------------------------------------
generate_fragments_help = 'Generate fragment potentials for secondary structure groups\n' \
                          'divisions of interest and write found fragment representative\n' \
                          'out. Default potential search is for intra-chain motifs'
parser_generate_fragments = dict(description=generate_fragments_help, help=generate_fragments_help)
generate_fragments_arguments = {
    (interface.long,): dict(action='store_true', help=f'Whether to {generate_fragments} at interface residues'),
    (interface_only.long,): dict(action='store_true', help=f'Whether to limit to interface residues'),
    (oligomeric_interfaces.long,):
        dict(action='store_true', help=f'Whether to {generate_fragments} at oligomeric interfaces in\naddition to '
                                       'hetrotypic interfaces')
}
# ---------------------------------------------------
rename_chains_help = 'For given poses, rename the chains in the source file in alphabetic order'
parser_rename_chains = dict(description=rename_chains_help, help=rename_chains_help)
# # ---------------------------------------------------
# parser_flags = {'flags': dict(description='Generate a flags file for %s' % program_name)}
# # parser_flags = subparsers.add_parser('flags', description='Generate a flags file for %s' % program_name)
# parser_flags_arguments = {
#     ('-t', '--template'): dict(action='store_true', help='Generate a flags template to edit on your own.'),
#     ('-m', '--module'): dict(dest='flags_module', action='store_true',
#                              help='Generate a flags template to edit on your own.')
# }
# # ---------------------------------------------------
fuse_chains_args = (fuse_chains.long,)
load_to_db_args = (load_to_db.long,)
load_to_db_kwargs = dict(action='store_true',
                         help=f'Use this input flag to load files existing in a {putils.program_output} to the DB')
range_args = ('-r', range_.long)
# ---------------------------------------------------
project_args = ('-p', project.long)
single_args = ('-s', single.long)
directory_args = ('-d', directory.long)
directory_needed = f'To locate poses from a file utilizing pose identifiers (--{poses}, -sf)\n' \
                   f'provide your working {program_output} directory with {format_args(directory_args)}.\n' \
                   f'If you run {program_name} in the context of an existing {program_output},\n' \
                   'the directory will automatically be inferred.'
input_help = f'Specify where/which poses should be included in processing.\n{directory_needed}'
parser_input = dict(description=input_help)  # , help=input_help
parser_input_group = dict(title=f'{"_" * len(input_title)}\n{input_title}',
                          description=f'\n{input_help}')
pose_inputs = {
    poses_args: dict(type=os.path.abspath, nargs='*', default=tuple(),
                     metavar=ex_path(default_path_file.format('TIMESTAMP', 'MODULE', 'LOCATION')),
                     help=f'For each run of {program_name}, a file will be created that\n'
                          f'specifies the specific poses used during that module. Use\n'
                          f'these files to interact with those poses in subsequent commands'),
    #                       'If pose identifiers are specified in a file, say as the result of\n'
    #                       f'{select_poses} or {select_designs}'),
    specification_file_args:
        dict(type=os.path.abspath, nargs='*', default=tuple(), metavar=ex_path('pose_design_specifications.csv'),
             help='Name of comma separated file with each line formatted:\n'
             #      'poseID, [designID], [1:directive 2-9:directive ...]\n'
                  '"pose_identifier, [design_name], [1:directive 2-9:directive ...]"\n'
                  'where [] indicate optional arguments. Both individual residue\n'
                  'numbers and ranges (specified with "-") are possible indicators'),
    }
input_arguments = {
    **pose_inputs,
    cluster_map_args: cluster_map_kwargs,
    ('-df', dataframe.long): dict(type=os.path.abspath, metavar=ex_path('Metrics.csv'),
                                  help=f'A DataFrame created by {program_name} analysis containing\n'
                                       'pose metrics. File is output in .csv format'),
    fuse_chains_args: dict(nargs='*', default=tuple(), metavar='A:B C:D',
                           help='The name of a pair of chains to fuse during design. Paired\n'
                                'chains should be separated by a colon, with the n-terminal\n'
                                'preceding the c-terminal chain. Fusion instances should be\n'
                                'separated by a space\n'
                                f'Ex {format_args(fuse_chains_args)} A:B C:D'),
    load_to_db_args: load_to_db_kwargs,
    # ('-N', f'--{nanohedra}V1-output'): dict(action='store_true', dest=nanohedra_output,
    #                                         help='Is the input a Nanohedra wersion 1 docking output?'),
    # ('-P', preprocessed.long): dict(action='store_true',
    #                                   help='Whether the designs of interest have been preprocessed for the '
    #                                        f'{current_energy_function}\nenergy function and/or missing loops'),
    (loop_model_input.long,):
        dict(action=argparse.BooleanOptionalAction, default=None,
             help='Whether the input building blocks should have missing regions modeled'),
    (refine_input.long,):
        dict(action=argparse.BooleanOptionalAction, default=None,
             help=f'Whether the input building blocks should be refined into {current_energy_function}'),
    (pre_loop_modeled.long,):
        dict(action='store_true', help='Whether input building blocks have been preprocessed for\nmissing density'),
    (pre_refined.long,):
        dict(action='store_true', help='Whether input building blocks have been preprocessed by\nrefinement into the'
                                       f' {current_energy_function}'),  # Todo change this
    range_args: dict(metavar='INT-INT',
                     help='The range of poses to process from a larger specification.\n'
                          'Specify a %% between 0 and 100, separating the range by "-"\n'
                          # Required ^ for formatting
                          'Ex: 0-25'),
    (specify_entities.long,): dict(action='store_true', help='Whether to initialize input Poses with user specified\n'
                                                               'identities of each constituent Entity')
}
directory_kwargs = dict(type=os.path.abspath, metavar=ex_path('your_pdb_files'),
                        help='Master directory where files to be designed are located. This may be\n'
                             'a random directory with poses requiring design, or the output from\n'
                             f'{program_name}. If the directory of interest resides in a {program_output}\n'
                             f'directory, it is recommended to use {format_args(file_args)}, '
                             f'{format_args(project_args)}, or {format_args(single_args)}')
file_kwargs = dict(type=os.path.abspath, default=tuple(), nargs='*', metavar=ex_path('file_with_pose.paths'),
                   help='File(s) to be input or containing list of files to be input to the program')
parser_input_mutual_group = dict()  # required=True <- adding kwarg below to different parsers depending on need
input_mutual_arguments = {
    directory_args: directory_kwargs,
    file_args: file_kwargs,
    project_args: dict(type=os.path.abspath, nargs='*', default=tuple(),
                       metavar=ex_path(program_output, projects, 'yourProject'),
                       help='Operate on designs specified within a project(s)'),
    single_args: dict(type=os.path.abspath, nargs='*', default=tuple(),
                      metavar=ex_path(program_output, projects, 'yourProject', 'single_pose[.pdb]'),
                      help='Operate on single pose(s) in a project'),
    pdb_codes_args: pdb_codes_kwargs,
    query_pdb_codes_args: query_codes_kwargs
}
output_help = 'Specify where output should be written'
parser_output = dict(description=output_help)  # , help=output_help
parser_output_group = dict(title=f'{"_" * len(output_title)}\n{output_title}',
                           description='\nSpecify where output should be written')
output_arguments = {
    (increment_chains.long,): dict(action='store_true',
                                   help='Whether assembly files should output with chain IDs incremented\n'
                                        "or in 'Multimodel' format. Multimodel format is useful for PyMol\n"
                                        "visualization with the command 'set all_states, on'. Chimera can\n"
                                        'utilize either format as the BIOMT record is respected'),
    ('-Oa', output_assembly.long):
        dict(action='store_true',
             help='Whether the symmetric assembly should be output.\nInfinite assemblies are output as a unit cell'),
    output_directory_args: output_directory_kwargs,
    output_file_args: dict(type=os.path.abspath,
                           help='If provided, the name of the output pose file. Otherwise, one\n'
                                'will be generated based on the time, input, and module'),
    ('-OF', output_fragments.long):
        dict(action='store_true', help='Write any fragments generated for each Pose'),
    ('-Oi', output_interface.long):
        dict(action='store_true', help='Write the residues that comprise the interface for each Pose'),
    ('-Oo', output_oligomers.long):
        dict(action=argparse.BooleanOptionalAction, default=False, help='Write any oligomers generated for each Pose'),
    ('-Oe', output_entities.long):
        dict(action=argparse.BooleanOptionalAction, default=False, help='Write the entities located for each Pose'),
    # output_structures_args:
    #     dict(action=argparse.BooleanOptionalAction, default=True,
    #          help=f'For any structures generated, write them.\n{boolean_positional_prevent_msg(output_structures)}'),
    ('-Ou', output_surrounding_uc.long):
        dict(action='store_true', help='For infinite materials, whether surrounding unit cells are output'),
    ('-Ot', output_trajectory.long):
        dict(action='store_true', help=f'For all structures generated, write them as a single multimodel file'),
    (overwrite.long,): dict(action='store_true', help='Whether to overwrite existing structural info'),
    ('-Pf', pose_format.long): dict(action='store_true',
                                    help='Whether outputs should be converted to pose number formatting,\n'
                                         'where residue numbers start at one and increase sequentially\n'
                                         'instead of using the original numbering'),
    (prefix.long,): dict(metavar='STRING', help='String to prepend to output name'),
    (suffix.long,): dict(metavar='STRING', help='String to append to output name'),
}
# all_flags_parsers = dict(all_flags=parser_all_flags_group)
all_flags_arguments = {}
# all_flags_arguments = {
#     **options_arguments, **output_arguments, **input_arguments, **input_mutual_arguments
#    # **residue_selector_arguments,
# }

# If using mutual groups, for the dict "key" (parser name), you must add "_mutual" immediately after the submodule
# string that own the group. i.e nanohedra"_mutual*" indicates nanohedra owns, or interface_design"_mutual*", etc
input_parser_groups = dict(input=parser_input_group, input_mutual=parser_input_mutual_group)  # _mutual
output_parser_groups = dict(output=parser_output_group)
option_parser_groups = dict(options=parser_options_group)
symmetry_parser_groups = dict(symmetry=parser_symmetry_group)
residue_selector_parser_groups = {residue_selector: parser_residue_selector_group}
mutual_keyword = '_mutual'
module_parser_groups = {
    # orient: parser_orient,
    f'{align_helices}{mutual_keyword}1': parser_component_mutual1_group,
    f'{align_helices}{mutual_keyword}2': parser_component_mutual2_group,
    align_helices: parser_align_helices,
    helix_bending: parser_helix_bending,
    refine: parser_refine,
    f'{nanohedra}{mutual_keyword}1': parser_component_mutual1_group,
    f'{nanohedra}{mutual_keyword}2': parser_component_mutual2_group,
    nanohedra: parser_nanohedra,
    # f'{nanohedra}_mutual_run_type': parser_nanohedra_run_type_mutual_group,
    cluster_poses: parser_cluster,
    design: parser_design,
    interface_design: parser_interface_design,
    interface_metrics: parser_metrics,
    optimize_designs: parser_optimize_designs,
    analysis: parser_analysis,
    process_rosetta_metrics: parser_process_rosetta_metrics,
    select_poses: parser_select_poses,
    select_designs: parser_select_designs,
    select_sequences: parser_select_sequences,
    check_clashes: parser_check_clashes,
    expand_asu: parser_expand_asu,
    generate_fragments: parser_generate_fragments,
    rename_chains: parser_rename_chains,
    predict_structure: parser_predict_structure,
    initialize_building_blocks: parser_initialize_building_blocks,
    protocol: parser_protocol,
    # These are "tools"
    multicistronic: parser_multicistronic,
    update_db: parser_update_db,
    # distribute: parser_distribute,
    # These are decoy modules to help with argument parsing
    all_flags: parser_all_flags,
    f'input{mutual_keyword}': parser_input_mutual_group,
    input_: parser_input,
    options: parser_options,
    output: parser_output,
    residue_selector: parser_residue_selector,
    symmetry: parser_symmetry,
}
# custom_script: parser_custom,
# select_poses_mutual: parser_select_poses_mutual_group,  # _mutual,
# flags: parser_flags,

# Todo?
#  {module: dict(flags=getattr(globals(), f'{module}_arguments'),
#                groups=getattr(globals(), f'parser_{module}_group'))
#  }
module_required = [f'{nanohedra}{mutual_keyword}1']
parser_arguments = {
    # orient: orient_arguments,
    f'{align_helices}{mutual_keyword}1': align_component_mutual1_arguments,  # mutually_exclusive_group
    f'{align_helices}{mutual_keyword}2': align_component_mutual2_arguments,  # mutually_exclusive_group
    align_helices: align_helices_arguments,
    helix_bending: helix_bending_arguments,
    refine: refine_arguments,
    f'{nanohedra}{mutual_keyword}1': component_mutual1_arguments,  # mutually_exclusive_group
    f'{nanohedra}{mutual_keyword}2': component_mutual2_arguments,  # mutually_exclusive_group
    nanohedra: nanohedra_arguments,
    # f'{nanohedra}{mutual_keyword}_run_type': nanohedra_run_type_mutual_arguments,  # mutually_exclusive
    cluster_poses: cluster_poses_arguments,
    design: design_arguments,
    interface_design: interface_design_arguments,
    interface_metrics: interface_metrics_arguments,
    optimize_designs: optimize_designs_arguments,
    analysis: analysis_arguments,
    process_rosetta_metrics: process_rosetta_metrics_arguments,
    select_poses: select_poses_arguments,
    select_designs: select_designs_arguments,
    select_sequences: select_sequences_arguments,
    generate_fragments: generate_fragments_arguments,
    predict_structure: predict_structure_arguments,
    initialize_building_blocks: initialize_building_blocks_arguments,
    protocol: protocol_arguments,
    # These are "tools"
    multicistronic: multicistronic_arguments,
    update_db: update_db_arguments,
    # distribute: distribute_arguments,
    # These are decoy modules to help with argument parsing
    all_flags: all_flags_arguments,
    f'input{mutual_keyword}': input_mutual_arguments,  # add_mutually_exclusive_group
    input_: input_arguments,
    options: options_arguments,
    output: output_arguments,
    residue_selector: residue_selector_arguments,
    symmetry: symmetry_arguments
}
# custom_script_arguments: parser_custom_script_arguments,
# select_poses_mutual_arguments: parser_select_poses_mutual_arguments, # mutually_exclusive_group
# flags_arguments: parser_flags_arguments,
# Todo? , usage=usage_str)
# Initialize various independent ArgumentParsers
# Todo https://gist.github.com/fonic/fe6cade2e1b9eaf3401cc732f48aeebd
#  argparsers[name] = ArgumentParser(**argparser_args)
standard_argparser_kwargs = dict(add_help=False, allow_abbrev=False, formatter_class=Formatter, usage=usage_str)
symmetry_parser = argparse.ArgumentParser(**standard_argparser_kwargs.copy())
options_parser = argparse.ArgumentParser(
    **dict(add_help=False, allow_abbrev=False, formatter_class=Formatter))
residue_selector_parser = argparse.ArgumentParser(**standard_argparser_kwargs.copy())
input_parser = argparse.ArgumentParser(**standard_argparser_kwargs.copy())
output_parser = argparse.ArgumentParser(**standard_argparser_kwargs.copy())
module_parser = argparse.ArgumentParser(**standard_argparser_kwargs.copy())
guide_parser = argparse.ArgumentParser(**standard_argparser_kwargs.copy())
entire_parser = argparse.ArgumentParser(**standard_argparser_kwargs.copy())

# Set up module ArgumentParser with module arguments
module_subargparser = dict(title=f'{"_" * len(module_title)}\n{module_title}', dest='module',  # metavar='',
                           # allow_abbrev=False,  # Not allowed here, but some modules try to parse -s as -sc...
                           description='\nThese are the different modes that designs can be processed. They are'
                                       '\npresented in an order which might be utilized along a design workflow,\nwith '
                                       'utility modules listed at the bottom starting with check_clashes.\nTo see '
                                       'example commands or algorithmic specifics of each Module enter:'
                                       f'\n{submodule_guide}\n\nTo get help with Module arguments enter:'
                                       f'\n{submodule_help}')
# For parsing of guide info
guide_parser.add_argument(*guide_args, **guide_kwargs)
guide_parser.add_argument(*help_args, **help_kwargs)
guide_parser.add_argument(*setup_args, **setup_kwargs)
# Add all modules to the guide_subparsers
guide_subparsers = guide_parser.add_subparsers(**module_subargparser)
# For all parsing of module arguments
subparsers = module_parser.add_subparsers(**module_subargparser)  # required=True,
module_suparsers: dict[str, argparse.ArgumentParser] = {}
for parser_name, parser_kwargs in module_parser_groups.items():
    flags_kwargs = parser_arguments.get(parser_name, {})
    """flags_kwargs has args (flag names) as key and keyword args (flag params) as values"""
    mutual_parser_name = mutual_parser_kwargs = None
    if mutual_keyword in parser_name:  # Create a mutually_exclusive_group from already formed subparser
        # Remove indication to "mutual" of the argparse group by removing any characters after mutual_keyword
        mutual_parser_name = parser_name
        mutual_parser_kwargs = parser_kwargs
        parser_name = mutual_parser_name[:mutual_parser_name.find(mutual_keyword)]
        parser_kwargs = module_parser_groups[parser_name]

    subparser = module_suparsers.get(parser_name)
    if not subparser:
        subparser = subparsers.add_parser(formatter_class=Formatter, allow_abbrev=False,
                                          prog=module_usage_str.format(parser_name),
                                          name=parser_name, **parser_kwargs)
        # Save the subparser in a dictionary to access later by mutual groups
        module_suparsers[parser_name] = subparser
        # Add each subparser to a guide_subparser as well
        guide_subparser = guide_subparsers.add_parser(name=parser_name, add_help=False)  #, **parser_kwargs[parser_name])
        guide_subparser.add_argument(*guide_args, **guide_kwargs)

    if mutual_parser_name:
        add_to_parser = subparser. \
            add_mutually_exclusive_group(**mutual_parser_kwargs,  # {mutual_parser_name: }
                                         **(dict(required=True) if parser_name in module_required else {}))
        # Add the key word argument "required" ^ to mutual parsers that use it
    else:
        add_to_parser = subparser

    for flags_, kwargs in flags_kwargs.items():
        add_to_parser.add_argument(*flags_, **kwargs)


def set_up_parser_with_groups(parser: argparse.ArgumentParser, parser_groups: dict[str, dict], required: bool = False):
    """Add the input arguments to the passed ArgumentParser

    Args:
        parser: The ArgumentParser to add_argument_group() to
        parser_groups: The groups to add to the ArgumentParser
        required: Whether the mutually exclusive group input is required
    """
    group = None  # Must get added before mutual groups can be added
    for parser_name, parser_kwargs in parser_groups.items():
        flags_kwargs = parser_arguments.get(parser_name, {})
        """flags_kwargs has args (flag names) as key and keyword args (flag params) as values"""
        if mutual_keyword in parser_name:  # Only has a dictionary as parser_arguments
            if group is None:
                # Remove indication to "mutual" of the argparse group by removing any characters after mutual_keyword
                group_parser_kwargs = parser_groups[parser_name[:parser_name.find(mutual_keyword)]]
                group = parser.add_argument_group(**group_parser_kwargs)
            exclusive_parser = group.add_mutually_exclusive_group(required=required, **parser_kwargs)
            for flags_, kwargs in flags_kwargs.items():
                exclusive_parser.add_argument(*flags_, **kwargs)
        else:
            if group is None:
                group = parser.add_argument_group(**parser_kwargs)
            for flags_, kwargs in flags_kwargs.items():
                group.add_argument(*flags_, **kwargs)


# Set up option ArgumentParser with options arguments
set_up_parser_with_groups(options_parser, option_parser_groups)
# Set up residue selector ArgumentParser with residue selector arguments
set_up_parser_with_groups(residue_selector_parser, residue_selector_parser_groups)
# Set up output ArgumentParser with output arguments
set_up_parser_with_groups(output_parser, output_parser_groups)
# Set up symmetry ArgumentParser with symmetry arguments
set_up_parser_with_groups(symmetry_parser, symmetry_parser_groups)
# Set up input ArgumentParser with input arguments
set_up_parser_with_groups(input_parser, input_parser_groups, required=True)
additional_parsers = [options_parser, residue_selector_parser, output_parser, symmetry_parser, input_parser]
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
                        parents=[module_parser, options_parser, residue_selector_parser, output_parser,
                                 symmetry_parser])

entire_parser = argparse.ArgumentParser(**entire_argparser)
# Can't set up input_parser via a parent due to mutually_exclusive groups formatting messed up in help.
# Therefore, repeat the input_parser_groups set up here with entire ArgumentParser
set_up_parser_with_groups(entire_parser, input_parser_groups)

# # can't set up module_parser via a parent due to mutually_exclusive groups formatting messed up in help, repeat above
# Separate the provided arguments for modules or overall program arguments to into flags namespaces
cluster_defaults = dict(namespace='cluster')
"""Contains all the arguments and their default parameters used in clustering Poses"""
design_defaults = dict(namespace='design')
"""Contains all the arguments and their default parameters used in design"""
dock_defaults = dict(namespace='dock')
"""Contains all the arguments and their default parameters used in docking"""
init_defaults = dict(namespace='init')
"""Contains all the arguments and their default parameters used in structure/sequence initialization"""
predict_defaults = dict(namespace='predict')
"""Contains all the arguments and their default parameters used in structure prediction"""
available_modules = []
"""Contains all the registered modules"""


def parse_flags_to_namespaces(parser: argparse.ArgumentParser):
    for group in parser._action_groups:
        # input(group._group_actions)
        for arg in group._group_actions:
            if isinstance(arg, argparse._SubParsersAction):
                global available_modules
                available_modules = sorted(arg.choices.keys())
                # This is a SubParser, recurse
                for name, sub_parser in arg.choices.items():
                    parse_flags_to_namespaces(sub_parser)
            elif arg.dest in cluster_namespace:
                cluster_defaults[arg.dest] = arg.default
            elif arg.dest in design_namespace:
                design_defaults[arg.dest] = arg.default
            elif arg.dest in dock_namespace:
                dock_defaults[arg.dest] = arg.default
            elif arg.dest in init_namespace:
                init_defaults[arg.dest] = arg.default
            elif arg.dest in predict_namespace:
                predict_defaults[arg.dest] = arg.default


parse_flags_to_namespaces(entire_parser)
registered_tools = [multicistronic, update_db]  # , distribute]
# Todo register these tools!
#  ['concatenate-files', 'list-overlap', 'retrieve-oligomers', 'retrieve-pdb-codes']
decoy_modules = [all_flags, initialize_building_blocks, input_, output, options, residue_selector]
options_modules = [input_, output, options, residue_selector]
available_tools = registered_tools + decoy_modules
# logger.debug(f'Found the tools: {", ".join(available_tools)}')
available_modules = sorted(set(available_modules).difference(available_tools))
# logger.debug(f'Found the modules: {", ".join(available_modules)}')
# Add orphaned 'interface', i.e. interface-design module alias for design with interface=True to design_defaults
design_defaults[interface] = False
defaults = dict(
    cluster=cluster_defaults,
    design=design_defaults,
    dock=dock_defaults,
    init=init_defaults,
    predict=predict_defaults
)


def format_defaults_for_namespace(defaults_: dict[str, dict[str, Any]]):
    for default_group, modify_flags in modify_options.items():
        default_group_flags = defaults_[default_group]  # defaults[default_group]
        for flag in modify_flags:
            # Replace the flag destination with the plain flag, no namespace prefix
            default_group_flags[flag.replace(f'{default_group}_', '')] = default_group_flags.pop(flag)


format_defaults_for_namespace(defaults)
