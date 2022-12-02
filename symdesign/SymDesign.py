"""
Module for distribution of SymDesign commands. Includes pose initialization, distribution of Rosetta commands to
SLURM computational clusters, analysis of designed poses, and sequence selection of completed structures.

"""
from __future__ import annotations

# import logging
import logging.config
import os
import shutil
import sys
from glob import glob
from itertools import repeat, product, combinations
from subprocess import list2cmdline
from typing import Any, AnyStr

import pandas as pd

import symdesign.utils.path as putils
# logging.config.fileConfig(putils.logging_cfg_file)
# print(putils.logging_cfg['loggers'])
logging.config.dictConfig(putils.logging_cfg)
logger = logging.getLogger(putils.program_name.lower())  # __name__)
# print(__name__)
# print(logger.__dict__)
# logger.info('Starting logger')
# logger.warning('Starting logger')
# input('WHY LOGGING')
from symdesign import flags, guide, metrics, protocols, utils
from symdesign.protocols.protocols import PoseDirectory
from symdesign.resources.job import job_resources_factory
from symdesign.resources.query.pdb import retrieve_pdb_entries_by_advanced_query
from symdesign.resources.query.utils import input_string, boolean_choice, validate_input_return_response_value
from symdesign.structure.fragment.db import fragment_factory, euler_factory
from symdesign.structure.model import Model
from symdesign.structure.sequence import generate_mutations, find_orf_offset, write_sequences
from symdesign.third_party.DnaChisel.dnachisel.DnaOptimizationProblem.NoSolutionError import NoSolutionError
from symdesign.utils import ProteinExpression, nanohedra


# def format_additional_flags(flags):
#     """Takes non-argparse specified flags and returns them into a dictionary compatible with argparse style.
#     This is useful for handling general program flags that apply to many modules, would be tedious to hard code and
#     request from the user
#
#     Returns:
#         (dict)
#     """
#     # combines ['-symmetry', 'O', '-nanohedra_output', True', ...]
#     combined_extra_flags = []
#     for idx, flag in enumerate(flags):
#         if flag[0] == '-' and flag[1] == '-':  # format flags by removing extra '-'. Issue with PyMol command in future?
#             flags[idx] = flag[1:]
#
#         if flag.startswith('-'):  # this is a real flag
#             extra_arguments = ''
#             # iterate over arguments after the flag until a flag with "-" is reached. This is a new flag
#             increment = 1
#             while (idx + increment) != len(flags) and not flags[idx + increment].startswith('-'):  # we have an argument
#                 extra_arguments += ' %s' % flags[idx + increment]
#                 increment += 1
#             # remove - from the front and add all arguments to single flag argument list item
#             combined_extra_flags.append('%s%s' % (flag.lstrip('-'), extra_arguments))  # extra_flags[idx + 1]))
#     # logger.debug('Combined flags: %s' % combined_extra_flags)
#
#     # parse the combined flags ['-nanohedra_output True', ...]
#     final_flags = {}
#     for flag_arg in combined_extra_flags:
#         if ' ' in flag_arg:
#             flag, *args = flag_arg.split()  #[0]
#             # flag = flag.lstrip('-')
#             # final_flags[flag] = flag_arg.split()[1]
#             if len(args) > 1:  # we have multiple arguments, set all to the flag
#                 final_flags[flag] = args
#
#             # check for specific strings and set to corresponding python values
#             elif args[0].lower() == 'true':
#                 # final_flags[flag] = eval(final_flags[flag].title())
#                 final_flags[flag] = True
#             elif args[0].lower() == 'false':
#                 final_flags[flag] = False
#             elif args[0].lower() == 'none':
#                 final_flags[flag] = None
#             else:
#                 final_flags[flag] = args[0]
#         else:  # add to the dictionary with default argument of True
#             final_flags[flag_arg] = True
#
#     return final_flags


sbatch_warning = 'Ensure the SBATCH script(s) below are correct. Specifically, check that the job array and any '\
                 'node specifications are accurate. You can look at the SBATCH manual (man sbatch or sbatch --help) to'\
                 ' understand the variables or ask for help if you are still unsure.'


def main():
    """Run the SymDesign program"""
    # -----------------------------------------------------------------------------------------------------------------
    #  Initialize local functions
    # -----------------------------------------------------------------------------------------------------------------
    def parse_protocol_results(jobs: list[Any] = None, _results: list[Any] | dict = None, **kwargs):
        """For a multimodule protocol, filter out any exceptions before proceeding to the next module

        Args:
            jobs: Separate items of work to undertake
            _results: The returned values from the jobs
        Returns:
            Tuple of passing PoseDirectories and Exceptions
        """
        if _results is None:
            _results = []

        _exceptions = [(jobs.pop(idx), _results[idx]) for idx, exception in enumerate(_results)
                       if isinstance(exception, BaseException)]
        return jobs, _exceptions

    def terminate(results: list[Any] | dict = None, output: bool = True, **kwargs):
        """Format designs passing output parameters and report program exceptions

        Args:
            results: The returned results from the module run. By convention contains results and exceptions
            output: Whether the module used requires a file to be output
        """
        output_analysis = True
        # Save any information found during the command to it's serialized state
        try:
            for design in pose_directories:
                design.pickle_info()
        except AttributeError:  # This isn't a PoseDirectory. Likely is a nanohedra job
            pass

        exceptions = kwargs.get('exceptions', [])
        if results:
            if job.module == flags.nanohedra:  # pose_directories is empty list when nanohedra
                success = results
                if job.distribute_work:
                    output_analysis = False
            else:
                success = \
                    [pose_directories[idx] for idx, result in enumerate(results) if
                     not isinstance(result, BaseException)]
                exceptions += \
                    [(pose_directories[idx], exception) for idx, exception in enumerate(results)
                     if isinstance(exception, BaseException)]
        else:
            success = []

        exit_code = 0
        if exceptions:
            print('\n')
            logger.warning(f'Exceptions were thrown for {len(exceptions)} designs. Check their logs for further details'
                           '\n\t%s' % '\n\t'.join(f'{design.path}: {_error}' for design, _error in exceptions))
            print('\n')

        if success and output:
            nonlocal design_source
            job_paths = job.job_paths

            if low and high:
                design_source = f'{design_source}-{low:.2f}-{high:.2f}'

            # Format the output file depending on specified name and module type
            default_output_tuple = (utils.starttime, job.module, design_source)
            designs_file = None
            if job.output_file:
                # if job.module not in [flags.analysis, flags.cluster_poses]:
                #     designs_file = job.output_file
                if job.module == flags.analysis:
                    if len(job.output_file.split(os.sep)) <= 1:
                        # The path isn't an absolute or relative path, so prepend the job.all_scores location
                        job.output_file = os.path.join(job.all_scores, job.output_file)
                    if not job.output_file.endswith('.csv'):
                        job.output_file = f'{job.output_file}.csv'
                    if not args.output:  # No output is specified
                        output_analysis = False
                else:
                    # Set the designs_file to the provided job.output_file
                    designs_file = job.output_file
            else:
                # For certain modules, use the default file type
                if job.module == flags.analysis:
                    job.output_file = putils.default_analysis_file.format(utils.starttime, design_source)
                else:  # We don't have a default output specified
                    pass

            # Make single file with names of each directory where all_docked_poses can be found
            if output_analysis:
                if designs_file is None:  # Make a default file name
                    putils.make_path(job_paths)
                    # Remove possible multiple instances of _pose from location in default_output_tuple
                    scratch_designs = \
                        os.path.join(job_paths, putils.default_path_file.format(*default_output_tuple)).split('_pose')
                    designs_file = f'{scratch_designs[0]}_pose{scratch_designs[-1]}'

                with open(designs_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(design.path for design in success))
                logger.critical(f'The file "{designs_file}" contains the locations of every pose that passed checks/'
                                f'filters for this job. Utilize this file to input these poses in future '
                                f'{putils.program_name} commands such as:'
                                f'\n\t{putils.program_command} MODULE --file {designs_file} ...')

            # Output any additional files for the module
            if job.module == flags.analysis:
                # Save Design DataFrame
                design_df = pd.DataFrame([result for result in results if not isinstance(result, BaseException)])
                if args.output:  # Create a new file
                    design_df.to_csv(args.output_file)
                else:
                    # This is the mechanism set up to append to an existing file. Check if we should add a header
                    header = False if os.path.exists(args.output_file) else True
                    design_df.to_csv(args.output_file, mode='a', header=header)

                logger.info(f'Analysis of all poses written to {args.output_file}')
                if args.save:
                    logger.info(f'Analysis of all Trajectories and Residues written to {job.all_scores}')

            # Set up sbatch scripts for processed Poses
            design_stage = putils.scout if job.design.scout \
                else (putils.hbnet_design_profile if job.design.hbnet
                      else (putils.structure_background if job.design.structure_background
                            else putils.interface_design))
            module_files = {flags.interface_design: design_stage,
                            flags.nanohedra: flags.nanohedra,
                            flags.refine: flags.refine,
                            flags.interface_metrics: putils.interface_metrics,
                            flags.optimize_designs: putils.optimize_designs
                            # custom_script: os.path.splitext(os.path.basename(getattr(args, 'script', 'c/custom')))[0],
                            }
            stage = module_files.get(job.module)
            if stage and job.distribute_work:
                if len(success) == 0:
                    exit_code = 1
                    exit(exit_code)

                putils.make_path(job_paths)
                putils.make_path(job.sbatch_scripts)
                if job.module == flags.nanohedra:
                    command_file = utils.write_commands([list2cmdline(cmd) for cmd in success], out_path=job_paths,
                                                        name='_'.join(default_output_tuple))
                    sbatch_file = utils.CommandDistributer.distribute(file=command_file, out_path=job.sbatch_scripts,
                                                                      scale=job.module, number_of_commands=len(success))
                else:
                    command_file = utils.write_commands([os.path.join(des.scripts, f'{stage}.sh') for des in success],
                                                        out_path=job_paths, name='_'.join(default_output_tuple))
                    sbatch_file = utils.CommandDistributer.distribute(file=command_file, out_path=job.sbatch_scripts,
                                                                      scale=job.module)
                    #                                                                        ^ for sbatch template
                logger.critical(sbatch_warning)

                if job.module == flags.interface_design and job.initial_refinement:  # True, should refine before design
                    refine_file = utils.write_commands([os.path.join(design.scripts, f'{flags.refine}.sh')
                                                        for design in success], out_path=job_paths,
                                                       name='_'.join((utils.starttime, flags.refine, design_source)))
                    sbatch_refine_file = utils.CommandDistributer.distribute(file=refine_file, scale=flags.refine,
                                                                             out_path=job.sbatch_scripts)
                    logger.info(f'Once you are satisfied, enter the following to distribute:\n\tsbatch '
                                f'{sbatch_refine_file}\nTHEN:\n\tsbatch {sbatch_file}')
                else:
                    logger.info(f'Once you are satisfied, enter the following to distribute:\n\tsbatch {sbatch_file}')

        # # test for the size of each of the PoseDirectory instances
        # if pose_directories:
        #     print('Average_design_directory_size equals %f' %
        #           (float(psutil.virtual_memory().used) / len(pose_directories)))

        # print('\n')
        exit(exit_code)

    def get_sym_entry_from_nanohedra_directory(nanohedra_dir: AnyStr) -> utils.SymEntry.SymEntry:
        """Handles extraction of Symmetry info from Nanohedra outputs.

        Args:
            nanohedra_dir: The path to a Nanohedra master output directory
        Raises:
            FileNotFoundError: If no nanohedra master log file is found
            SymmetryError: If no symmetry is found
        Returns:
            The SymEntry specified by the Nanohedra docking run
        """
        log_path = os.path.join(nanohedra_dir, putils.master_log)
        try:
            with open(log_path, 'r') as f:
                for line in f.readlines():
                    if 'Nanohedra Entry Number: ' in line:  # "Symmetry Entry Number: " or
                        return utils.SymEntry.symmetry_factory.get(int(line.split(':')[-1]))  # sym_map inclusion?
        except FileNotFoundError:
            raise FileNotFoundError('Nanohedra master directory is malformed. '
                                    f'Missing required docking file {log_path}')
        raise utils.InputError(f'Nanohedra master docking file {log_path} is malformed. Missing required info'
                               ' "Nanohedra Entry Number:"')

    resubmit_command_message = f'After completion of sbatch script(s), re-submit your {putils.program_name} ' \
                               f'command:\n\tpython {" ".join(sys.argv)}'
    # -----------------------------------------------------------------------------------------------------------------
    #  Process optional program flags
    # -----------------------------------------------------------------------------------------------------------------
    # Ensure module specific arguments are collected and argument help is printed in full
    args, additional_args = flags.argparsers[flags.parser_guide].parse_known_args()
    # -----------------------------------------------------------------------------------------------------------------
    #  Display the program guide if requested
    # -----------------------------------------------------------------------------------------------------------------
    if args.guide:
        if args.module == flags.analysis:
            print(guide.analysis)
        elif args.module == flags.cluster_poses:
            print(guide.cluster_poses)
        elif args.module == flags.interface_design:
            print(guide.interface_design)
        elif args.module == flags.interface_metrics:
            print(guide.interface_metrics)
        elif args.module == flags.optimize_designs:
            print(guide.optimize_designs)
        elif args.module == flags.refine:
            print(guide.refine)
        elif args.module == flags.nanohedra:
            print(guide.nanohedra)
        elif args.module == flags.select_poses:
            print(guide.select_poses)
        elif args.module == flags.select_designs:
            print(guide.select_designs)
        elif args.module == flags.select_sequences:
            print(guide.select_sequences)
        elif args.module == flags.expand_asu:
            print(guide.expand_asu)
        elif args.module == flags.orient:
            print(guide.orient)
        # elif args.module == 'custom_script':
        #     print()
        # elif args.module == 'check_clashes':
        #     print()
        # elif args.module == 'residue_selector':
        #     print()
        # elif args.module == 'visualize':
        #     print('Usage: %s -r %s -- [-d %s, -df %s, -f %s] visualize --range 0-10'
        #           % (SDUtils.ex_path('pymol'), putils.program_command.replace('python ', ''),
        #              SDUtils.ex_path('pose_directory'), SDUtils.ex_path('DataFrame.csv'),
        #              SDUtils.ex_path('design.paths')))
        else:  # Print the full program readme and exit
            guide.print_guide()
            exit()
    elif args.setup:
        guide.setup_instructions()
        exit()
    # ---------------------------------------------------
    # elif args.flags:  # Todo
    #     if args.template:
    #         flags.query_user_for_flags(template=True)
    #     else:
    #         flags.query_user_for_flags(mode=args.flags_module)
    # ---------------------------------------------------
    # elif args.module == 'distribute':  # -s stage, -y success_file, -n failure_file, -m max_jobs
    #     utils.CommandDistributer.distribute(**vars(args))
    # ---------------------------------------------------
    # elif args.residue_selector:  # Todo
    #     def generate_sequence_template(pdb_file):
    #         pdb = Model.from_file(pdb_file, entities=False)
    #         sequence = SeqRecord(Seq(''.join(chain.sequence for chain in pdb.chains), 'Protein'), id=pdb.file_path)
    #         sequence_mask = copy.copy(sequence)
    #         sequence_mask.id = 'residue_selector'
    #         sequences = [sequence, sequence_mask]
    #         raise NotImplementedError('This write_fasta call needs to have its keyword arguments refactored')
    #         return write_fasta(sequences, file_name=f'{os.path.splitext(pdb.file_path)[0]}_residue_selector_sequence')
    #
    #     if not args.single:
    #         raise SDUtils.DesignError('You must pass a single pdb file to %s. Ex:\n\t%s --single my_pdb_file.pdb '
    #                                   'residue_selector' % (putils.program_name, putils.program_command))
    #     fasta_file = generate_sequence_template(args.single)
    #     logger.info('The residue_selector template was written to %s. Please edit this file so that the '
    #                 'residue_selector can be generated for protein design. Selection should be formatted as a "*" '
    #                 'replaces all sequence of interest to be considered in design, while a Mask should be formatted as '
    #                 'a "-". Ex:\n>pdb_template_sequence\nMAGHALKMLV...\n>residue_selector\nMAGH**KMLV\n\nor'
    #                 '\n>pdb_template_sequence\nMAGHALKMLV...\n>design_mask\nMAGH----LV\n'
    #                 % fasta_file)
    # -----------------------------------------------------------------------------------------------------------------
    #  Initialize program with provided flags and arguments
    # -----------------------------------------------------------------------------------------------------------------
    # Parse arguments for the actual runtime which accounts for differential argument ordering from standard argparse
    module_parser = flags.argparsers[flags.parser_module]
    args, additional_args = module_parser.parse_known_args()
    remove_dummy = False
    if args.module == flags.nanohedra:
        if args.query:  # We need to submit before we check for additional_args as query comes with additional args
            nanohedra.cmdline.query_mode([__file__, '-query'] + additional_args)
            exit()
        else:  # We need to add a dummy input for argparse to happily continue with required args
            additional_args.extend(['--file', 'dummy'])
            remove_dummy = True
    elif args.module == flags.protocol:
        # We need to add a dummy input for argparse to happily continue with required args
        if flags.nanohedra in args.modules:
            additional_args.extend(['--file', 'dummy'])
            remove_dummy = True
        # Parse all options for every module provided
        for module in args.modules:
            additional_args = [module] + additional_args
            args, additional_args = module_parser.parse_known_args(args=additional_args, namespace=args)
        # Set the module to flags.protocol again after parsing
        args.module = flags.protocol

    # Check the provided flags against the full SymDesign entire_parser to print any help
    entire_parser = flags.argparsers[flags.parser_entire]
    _args, _additional_args = entire_parser.parse_known_args()
    # Parse the provided flags
    for argparser in [flags.parser_options, flags.parser_residue_selector, flags.parser_output, flags.parser_input]:
        args, additional_args = flags.argparsers[argparser].parse_known_args(args=additional_args, namespace=args)

    if additional_args:
        exit(f'\nSuspending run. Found flag(s) that are not recognized program wide: {", ".join(additional_args)}\n'
             'Please correct (try adding --help if unsure), and resubmit your command\n')

    if remove_dummy:  # Remove the dummy input
        del args.file
    # -----------------------------------------------------------------------------------------------------------------
    #  Find base symdesign_directory and check for proper set up of program i/o
    # -----------------------------------------------------------------------------------------------------------------
    symdesign_directory = None
    # Check if output already exists or --overwrite is provided
    if args.output_file and os.path.exists(args.output_file) and args.module not in flags.analysis \
            and not args.overwrite:
        exit(f'The specified output file "{args.output_file}" already exists, this will overwrite your old '
             f'data! Please specify a new name with with -Of/--{flags.output_file}, '
             'or append --overwrite to your command')
    elif args.output_directory:
        if os.path.exists(args.output_directory) and not args.overwrite:
            exit(f'The specified output directory "{args.output_directory}" already exists, this will overwrite '
                 f'your old data! Please specify a new name with with -Od/--{flags.output_directory}, '
                 '--prefix or --suffix, or append --overwrite to your command')
        else:
            symdesign_directory = args.output_directory
            putils.make_path(symdesign_directory)
    else:
        symdesign_directory = utils.get_base_symdesign_dir(
            (args.directory or (args.project or args.single or [None])[0] or os.getcwd()))

    root = None
    if symdesign_directory is None:  # Check if there is a file and see if we can solve there
        # By default, assume new input and make in the current directory
        symdesign_directory = os.path.join(os.getcwd(), putils.program_output)
        if args.file:
            # See if the file contains SymDesign specified paths
            # Must index the first file with [0]...
            with open(args.file[0], 'r') as f:
                line = f.readline()
                if os.path.splitext(line)[1] == '':  # No extension. Provided as directory/poseid from SymDesign output
                    file_directory = utils.get_base_symdesign_dir(line)
                    if file_directory is not None:
                        symdesign_directory = file_directory
                else:
                    # Set file basename as "root". Designs are being integrated for the first time
                    # In this case, design names might be the same, so we have to take the full file path as the name
                    # to discriminate between separate files
                    root = os.path.splitext(os.path.basename(args.file))[0]

        putils.make_path(symdesign_directory)
    # -----------------------------------------------------------------------------------------------------------------
    #  Process JobResources which holds shared program objects and command-line arguments
    # -----------------------------------------------------------------------------------------------------------------
    job = job_resources_factory.get(program_root=symdesign_directory, arguments=args)
    if args.module == flags.protocol:
        # Set the first module as the program module to allow initialization
        job.module = args.modules[0]
    # -----------------------------------------------------------------------------------------------------------------
    #  Start Logging
    # -----------------------------------------------------------------------------------------------------------------
    if args.log_level == 1:  # Debugging
        # Root logs to stream with level debug
        # logger = utils.start_log(level=job.log_level)
        # utils.start_log(level=job.log_level)
        # utils.set_loggers_to_propagate()
        utils.set_logging_to_level(job.log_level)
        logger.warning('Debug mode. Generates verbose output. No writing to *.log files will occur')
    else:
        # # Root logger logs to stream with level 'warning'
        # utils.start_log(handler_level=args.log_level)
        # # Stream above still emits at 'warning'
        # Set all modules to propagate logs to write to master log file
        utils.set_loggers_to_propagate()
        utils.set_logging_to_level(level=job.log_level)
        # utils.set_logging_to_level(handler_level=job.log_level)
        # # Root logger logs to a single file with level 'info'
        # utils.start_log(handler=2, location=os.path.join(symdesign_directory, putils.program_name))
        # SymDesign main logs to stream with level info and propagates to main log
        # logger = utils.start_log(name=putils.program_name, propagate=True)
        # All Designs will log to specific file with level info unless skip_logging is passed
    # -----------------------------------------------------------------------------------------------------------------
    #  Process job information which is necessary for processing and i/o
    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------
    #  Check for tool request
    # ---------------------------------------------------
    decoy_modules = ['input', 'output', 'options', 'residue_selector']
    symdesign_tools = [flags.multicistronic] + decoy_modules
    #     flags.input, flags.output, flags.options, flags.residue_selector]
    if job.module in symdesign_tools:
        if job.module == flags.multicistronic:
            ProteinExpression.create_mulitcistronic_sequences(args)
        else:  # if job.module in decoy_modules:
            pass  # exit()
        exit()

    # Set up module specific arguments
    # Todo we should run this check before every module used as in the case of universal protocols
    #  See if it can be detached here and made into function in main() scope
    initialize = True
    if job.module in [flags.interface_design, flags.generate_fragments, flags.orient, flags.expand_asu,
                      flags.interface_metrics, flags.refine, flags.optimize_designs, flags.rename_chains,
                      flags.check_clashes]:  # , 'custom_script', 'find_asu', 'status', 'visualize'
        # Set up design directories
        if job.module == flags.generate_fragments:  # Ensure we write fragments out
            job.write_fragments = True
    elif job.module in [flags.analysis, flags.cluster_poses,
                        flags.select_poses, flags.select_designs, flags.select_sequences]:
        # Analysis types can be run from nanohedra_output, so ensure that we don't construct new
        job.construct_pose = False
        if job.module == flags.analysis:
            # Ensure analysis write directory exists
            putils.make_path(job.all_scores)
        # Analysis types can be run from nanohedra_output, so ensure that we don't construct new
        job.construct_pose = False
        if job.module == flags.select_designs:  # Alias to module select_sequences with --skip-sequence-generation
            job.module = flags.select_sequences
            job.skip_sequence_generation = True
        if job.module == flags.select_poses:
            # When selecting by dataframe or metric, don't initialize, input is handled in module protocol
            if args.dataframe or args.metric:
                initialize = False
        if job.module == flags.select_sequences and args.select_number == sys.maxsize and not args.total:
            # Change default number to a single sequence/pose when not doing a total selection
            args.select_number = 1
    elif job.module == flags.nanohedra:
        initialize = False
        if not job.sym_entry:
            raise RuntimeError(f'When running {flags.nanohedra}, the argument -e/--entry/--{flags.sym_entry} is '
                               'required')
    else:  # We have no module passed. Print the guide and exit
        guide.print_guide()
        exit()
    # -----------------------------------------------------------------------------------------------------------------
    #  Report options and Set up Databases
    # -----------------------------------------------------------------------------------------------------------------
    reported_args = job.report_specified_arguments(args)
    logger.info('Starting with options:\n\t%s' % '\n\t'.join(utils.pretty_format_table(reported_args.items())))
    if job.debug:
        reported_args = job.report_unspecified_arguments(args)

    logger.info(f'Using resources in Database located at "{job.data}"')
    if job.module in [flags.nanohedra, flags.generate_fragments, flags.interface_design, flags.analysis]:
        if job.design.term_constraint:
            job.fragment_db = fragment_factory(source=args.fragment_database)
            # Initialize EulerLookup class
            euler_factory()
    # -----------------------------------------------------------------------------------------------------------------
    #  Grab all Poses (PoseDirectory instance) from either database, directory, project, single, or file
    # -----------------------------------------------------------------------------------------------------------------
    all_poses: list[AnyStr] | None = None
    # pose_directories hold jobs with specific poses
    # list[PoseDirectory] for an establishes pose
    # list[tuple[Structure, Structure]] for a nanohedra docking job
    pose_directories: list[PoseDirectory] | list[tuple[Any, Any]] = []
    low = high = low_range = high_range = None
    # Start with the assumption that we aren't loading resources
    load_resources = False
    if initialize:
        if args.range:
            try:
                low, high = map(float, args.range.split('-'))
            except ValueError:  # we didn't unpack correctly
                raise ValueError('The input flag -r/--range argument must take the form "LOWER-UPPER"')
            low_range, high_range = int((low / 100) * len(all_poses)), int((high / 100) * len(all_poses))
            if low_range < 0 or high_range > len(all_poses):
                raise ValueError('The input flag -r/--range argument is outside of the acceptable bounds [0-100]')
            logger.info(f'Selecting poses within range: {low_range if low_range else 1}-{high_range}')

        logger.info(f'Setting up input files for {job.module}')
        if args.nanohedra_output:  # Nanohedra directory
            all_poses, job.location = utils.collect_nanohedra_designs(files=args.file, directory=args.directory)
            if all_poses:
                first_pose_path = all_poses[0]
                if first_pose_path.count(os.sep) == 0:
                    job.nanohedra_root = args.directory
                else:
                    job.nanohedra_root = f'{os.sep}{os.path.join(*first_pose_path.split(os.sep)[:-4])}'
                if not job.sym_entry:  # Get from the Nanohedra output
                    job.sym_entry = get_sym_entry_from_nanohedra_directory(job.nanohedra_root)
                pose_directories = [PoseDirectory.from_file(pose, root=root)
                                    for pose in all_poses[low_range:high_range]]
                # copy the master nanohedra log
                project_designs = \
                    os.path.join(job.projects, f'{os.path.basename(job.nanohedra_root)}_{putils.pose_directory}')
                if not os.path.exists(os.path.join(project_designs, putils.master_log)):
                    putils.make_path(project_designs)
                    shutil.copy(os.path.join(job.nanohedra_root, putils.master_log), project_designs)
        elif args.specification_file:
            if not args.directory:
                raise utils.InputError(f'A --{flags.directory} must be provided when using '
                                       f'--{flags.specification_file}')
            # Todo, combine this with collect_designs
            #  this works for file locations as well! should I have a separate mechanism for each?
            design_specification = utils.PoseSpecification(args.specification_file)
            pose_directories = [PoseDirectory.from_pose_id(pose, root=args.directory, specific_design=design,
                                                           directives=directives)
                                for pose, design, directives in design_specification.get_directives()]
            job.location = args.specification_file
        else:
            all_poses, job.location = utils.collect_designs(files=args.file, directory=args.directory,
                                                            projects=args.project, singles=args.single)
            if all_poses:
                if all_poses[0].count(os.sep) == 0:  # check to ensure -f wasn't used when -pf was meant
                    # assume that we have received pose-IDs and process accordingly
                    if not args.directory:
                        raise utils.InputError('Your input specification appears to be pose IDs, however no '
                                               f'--{flags.directory} was passed. Please resubmit with '
                                               f'--{flags.directory} and use --{flags.pose_file}/'
                                               f'--{flags.specification_file} with pose IDs')
                    pose_directories = [PoseDirectory.from_pose_id(pose, root=args.directory)
                                        for pose in all_poses[low_range:high_range]]
                else:
                    pose_directories = [PoseDirectory.from_file(pose, root=root)
                                        for pose in all_poses[low_range:high_range]]
        if not pose_directories:
            raise utils.InputError(f'No {putils.program_name} directories found within "{job.location}"! Please ensure '
                                   f'correct location')
        representative_pose_directory = next(iter(pose_directories))
        design_source = os.path.splitext(os.path.basename(job.location))[0]

        # Todo logic error when initialization occurs with module that doesn't call this, subsequent runs are missing
        #  directories/resources that haven't been made
        # Check to see that proper files have been created including orient, refinement, loop modeling, hhblits, bmdca?
        initialized = representative_pose_directory.initialized
        initialize_modules = [flags.interface_design, flags.interface_metrics, flags.optimize_designs]
        #      flags.analysis,  # maybe hhblits, bmDCA. Only refine if Rosetta were used, no loop_modelling
        #      flags.refine]  # pre_refine not necessary. maybe hhblits, bmDCA, loop_modelling
        # Todo fix below sloppy logic
        if (not initialized and job.module in initialize_modules) or args.nanohedra_output or args.update_database:
            all_structures = []
            if not initialized and args.preprocessed:
                # args.orient, args.refine = True, True  # Todo make part of argparse? Could be variables in NanohedraDB
                # SDUtils.make_path(job.refine_dir)
                putils.make_path(job.full_model_dir)
                putils.make_path(job.stride_dir)
                all_entities, found_entity_names = [], set()
                for entity in [entity for pose in pose_directories for entity in pose.initial_model.entities]:
                    if entity.name not in found_entity_names:
                        all_entities.append(entity)
                        found_entity_names.add(entity.name)
                # Todo save all the Entities to the StructureDatabase
                #  How to know if Entity is needed or a combo? Need sym map to tell if they are the same length?
            elif initialized and args.update_database:
                for pose in pose_directories:
                    pose.setup()

                all_entities, found_entity_names = [], set()
                for pose in pose_directories:
                    for name in pose.entity_names:
                        if name not in found_entity_names:
                            found_entity_names.add(name)
                            pose.load_pose()
                            all_entities.append(pose.pose.entity(name))

                all_entities = [entity for entity in all_entities if entity]
            else:
                logger.critical('The requested poses require structural preprocessing before design modules should be '
                                'used')
                # Collect all entities required for processing the given commands
                required_entities = list(map(set, list(zip(*[pose.entity_names for pose in pose_directories]))))
                # Select entities, orient them, then load each entity to all_structures for further database processing
                symmetry_map = job.sym_entry.groups if job.sym_entry else repeat(None)
                for symmetry, entities in zip(symmetry_map, required_entities):
                    if not entities:  # useful in a case where symmetry groups are the same or group is None
                        continue
                    # elif not symmetry:
                    #     logger.info(f'Files are being processed without consideration for symmetry: '
                    #                 f'{", ".join(entities)}')
                    #     # all_structures.extend(job.structure_db.orient_structures(entities))
                    #     # continue
                    # else:
                    #     logger.info(f'Files are being processed with {symmetry} symmetry: {", ".join(entities)}')
                    all_structures.extend(job.structure_db.orient_structures(entities, symmetry=symmetry))
                # Create entities iterator to set up sequence dependent resources
                all_entities = [entity for structure in all_structures for entity in structure.entities]

            # Set up common Structure/Entity resources
            info_messages = []
            if job.design.evolution_constraint:
                evolution_instructions = job.setup_evolution_constraint(all_entities)
                load_resources = True if evolution_instructions else False
                info_messages.extend(evolution_instructions)

            if args.preprocessed:
                # Ensure we report to PoseDirectory the results after skipping set up
                job.initial_refinement = job.initial_loop_model = False
            else:
                preprocess_instructions, initial_refinement, initial_loop_model = \
                    job.structure_db.preprocess_structures_for_design(all_structures,
                                                                      script_out_path=job.sbatch_scripts)
                #                                                       batch_commands=args.distribute_work)
                if info_messages:
                    info_messages += ['The following can be run at any time regardless of evolutionary script progress']
                info_messages += preprocess_instructions
                job.initial_refinement = initial_refinement
                job.initial_loop_model = initial_loop_model

            # Entity processing commands are needed
            if load_resources or job.initial_refinement or job.initial_loop_model:
                if info_messages:
                    logger.critical(sbatch_warning)
                    for message in info_messages:
                        logger.info(message)
                    print('\n')
                    logger.info(resubmit_command_message)
                    terminate(output=False)
                    # After completion of sbatch, the next time initialized, there will be no refine files left allowing
                    # initialization to proceed
                else:
                    # We always prepare info_messages when jobs should be run
                    raise utils.InputError("This shouldn't have happened. info_messages can't be False here...")

        if args.multi_processing:  # and not args.skip_master_db:
            logger.debug('Loading Database for multiprocessing fork')
            # Todo set up a job based data acquisition as it takes some time and isn't always necessary!
            job.structure_db.load_all_data()
            job.api_db.load_all_data()
            # Todo tweak behavior of these two parameters. Need Queue based PoseDirectory
            # utils.mp_map(PoseDirectory.setup, pose_directories, processes=job.cores)
            # utils.mp_map(PoseDirectory.link_master_database, pose_directories, processes=job.cores)
        # Set up in series
        for pose in pose_directories:
            pose.setup(pre_refine=not job.initial_refinement, pre_loop_model=not job.initial_loop_model)

        logger.info(f'{len(pose_directories)} unique poses found in "{job.location}"')
        if not job.debug and not job.skip_logging:
            if representative_pose_directory.log_path:
                logger.info(f'All design specific logs are located in their corresponding directories\n\tEx: '
                            f'{representative_pose_directory.log_path}')

    elif job.module == flags.nanohedra:
        logger.critical(f'Setting up inputs for {job.module.title()} docking')
        # Todo make current with sql ambitions
        # Make master output directory. sym_entry is required, so this won't fail v
        if args.output_directory is None:
            # job.output_directory = os.path.join(job.projects, f'NanohedraEntry{sym_entry.entry_number}_BUILDING-BLOCKS_Poses')
            job.output_directory = job.projects
            os.makedirs(job.output_directory, exist_ok=True)
        # Transform input entities to canonical orientation and return their ASU
        all_structures = []
        # Set up variables for the correct parsing of provided file paths
        by_file1 = by_file2 = False
        eventual_structure_names1 = eventual_structure_names2 = None
        if args.oligomer1:
            by_file1 = True
            logger.critical(f'Ensuring provided file(s) at {args.oligomer1} are oriented for Nanohedra Docking')
            if '.pdb' in args.oligomer1:
                pdb1_filepaths = [args.oligomer1]
            else:
                pdb1_filepaths = utils.get_directory_file_paths(args.oligomer1, extension='.pdb*')
                if not pdb1_filepaths:
                    logger.warning(f'Found no .pdb files at {args.oligomer1}')
            # Set filepaths to structure_names, reformat the file paths to the file_name for structure_names
            structure_names1 = pdb1_filepaths
            eventual_structure_names1 = \
                list(map(os.path.basename, [os.path.splitext(file)[0] for file in pdb1_filepaths]))
        elif args.pdb_codes1:
            # Collect all provided codes required for component 1 processing
            structure_names1 = set(utils.to_iterable(args.pdb_codes1, ensure_file=True))
            # Make all names lowercase
            structure_names1 = list(map(str.lower, structure_names1))
        elif args.query_codes1:
            args.save_query = validate_input_return_response_value(
                'Do you want to save your PDB query to a local file?', {'y': True, 'n': False})

            print('\nStarting PDB query for component 1\n')
            structure_names1 = retrieve_pdb_entries_by_advanced_query(save=args.save_query, entity=True)
            # Make all names lowercase
            structure_names1 = list(map(str.lower, structure_names1))
        else:
            raise RuntimeError('This should be impossible with mutually exclusive argparser group')

        all_structures.extend(job.structure_db.orient_structures(structure_names1,
                                                                 symmetry=job.sym_entry.group1,
                                                                 by_file=by_file1))
        single_component_design = False
        structure_names2 = []
        if args.oligomer2:
            if args.oligomer1 != args.oligomer2:  # See if they are the same input
                by_file2 = True
                logger.critical(f'Ensuring provided file(s) at {args.oligomer2} are oriented for Nanohedra Docking')
                if '.pdb' in args.oligomer2:
                    pdb2_filepaths = [args.oligomer2]
                else:
                    pdb2_filepaths = utils.get_directory_file_paths(args.oligomer2, extension='.pdb*')
                    if not pdb2_filepaths:
                        logger.warning(f'Found no .pdb files at {args.oligomer2}')

                # Set filepaths to structure_names, reformat the file paths to the file_name for structure_names
                structure_names2 = pdb2_filepaths
                eventual_structure_names2 = \
                    list(map(os.path.basename, [os.path.splitext(file)[0] for file in pdb2_filepaths]))
            else:  # The entities are the same symmetry, or we have single component and bad input
                single_component_design = True
        elif args.pdb_codes2:
            # Collect all provided codes required for component 2 processing
            structure_names2 = set(utils.to_iterable(args.pdb_codes2, ensure_file=True))
            # Make all names lowercase
            structure_names2 = list(map(str.lower, structure_names2))
        elif args.query_codes2:
            args.save_query = validate_input_return_response_value(
                'Do you want to save your PDB query to a local file?', {'y': True, 'n': False})
            print('\nStarting PDB query for component 2\n')
            structure_names2 = retrieve_pdb_entries_by_advanced_query(save=args.save_query, entity=True)
            # Make all names lowercase
            structure_names2 = list(map(str.lower, structure_names2))
        else:
            single_component_design = True
        # Select entities, orient them, then load each Structure to all_structures for further database processing
        all_structures.extend(job.structure_db.orient_structures(structure_names2,
                                                                 symmetry=job.sym_entry.group2,
                                                                 by_file=by_file2))
        # Create entities iterator to set up sequence dependent resources
        all_entities = [entity for structure in all_structures for entity in structure.entities]

        # Set up common Structure/Entity resources
        info_messages = []
        if job.design.evolution_constraint:
            evolution_instructions = job.setup_evolution_constraint(all_entities)
            load_resources = True if evolution_instructions else False
            info_messages.extend(evolution_instructions)

        if args.preprocessed:
            # Ensure we report to PoseDirectory the results after skipping set up
            job.initial_refinement = job.initial_loop_model = False
        else:
            preprocess_instructions, initial_refinement, initial_loop_model = \
                job.structure_db.preprocess_structures_for_design(all_structures, script_out_path=job.sbatch_scripts)
            #                                                       batch_commands=args.distribute_work)
            if info_messages:
                info_messages += ['The following can be run at any time regardless of evolutionary script progress']
            info_messages += preprocess_instructions
            job.initial_refinement = initial_refinement
            job.initial_loop_model = initial_loop_model

        # Entity processing commands are needed
        if load_resources or job.initial_refinement or job.initial_loop_model:
            if info_messages:
                logger.critical(sbatch_warning)
                for message in info_messages:
                    logger.info(message)
                print('\n')
                logger.info(resubmit_command_message)
                terminate(output=False)
                # After completion of sbatch, the next time command is entered docking will proceed
            else:
                # We always prepare info_messages when jobs should be run
                raise utils.InputError("This shouldn't have happened. info_messages can't be False here...")

        # # Ensure all_entities are symmetric. As of now, all orient_structures returns are the symmetrized structure
        # for entity in [entity for structure in all_structures for entity in structure.entities]:
        #     entity.make_oligomer(symmetry=entity.symmetry)

        if by_file1:
            structure_names1 = eventual_structure_names1
        if by_file2:
            structure_names2 = eventual_structure_names2
        # Make all possible structure pairs given input entities by finding entities from entity_names
        structures1 = []
        for structure_name in structure_names1:
            for structure in all_structures:
                if structure_name in structure.name:
                    structures1.append(structure)

        # Using combinations of directories with .pdb files
        if single_component_design:
            logger.info('No additional entities requested for docking, treating as single component')
            # structures1 = [entity for entity in all_entities if entity.name in structures1]
            # ^ doesn't work as entity_id is set in orient_structures, but structure name is entry_id
            pose_directories = list(combinations(structures1, 2))
        else:
            structures2 = []
            for structure_name in structure_names2:
                for structure in all_structures:
                    if structure_name in structure.name:
                        structures2.append(structure)
                        break
            # v doesn't work as entity_id is set in orient_structures, but structure name is entry_id
            # structures1 = [entity for entity in all_entities if entity.name in structures1]
            # structures2 = [entity for entity in all_entities if entity.name in structures2]
            pose_directories = list(product(structures1, structures2))

        if not pose_directories:  # No pairs were located
            exit('No docking pairs were located from your input. Please ensure that your input flags are as intended! '
                 f'{putils.issue_submit_warning}')
        elif job.distribute_work:
            # Write all commands to a file and use sbatch
            design_source = f'Entry{job.sym_entry.entry_number}'  # used for terminate()
            # script_out_dir = os.path.join(job.output_directory, putils.scripts)
            # os.makedirs(script_out_dir, exist_ok=True)
            possible_input_args = [arg for args in flags.nanohedra_mutual1_arguments.keys() for arg in args] \
                + [arg for args in flags.nanohedra_mutual2_arguments.keys() for arg in args]
            #     + list(flags.distribute_args)
            submitted_args = sys.argv[1:]
            for input_arg in possible_input_args:
                try:
                    pop_index = submitted_args.index(input_arg)
                except ValueError:  # Not in list
                    continue
                submitted_args.pop(pop_index)
                if input_arg in ('-Q1', f'--{flags.query_codes1}', '-Q2', f'--{flags.query_codes2}'):
                    continue
                else:  # Pop the index twice if the argument requires an input
                    submitted_args.pop(pop_index)

            for arg in flags.distribute_args:
                try:
                    pop_index = submitted_args.index(arg)
                except ValueError:  # Not in list
                    continue
                submitted_args.pop(pop_index)

            # Format commands
            cmd = ['python', putils.program_exe] + submitted_args
            commands = [cmd.copy() + [f'--{putils.nano_entity_flag1}', model1.file_path,
                                      f'--{putils.nano_entity_flag2}', model2.file_path]
                        for idx, (model1, model2) in enumerate(pose_directories)]
            # logger.debug([list2cmdline(cmd) for cmd in commands])
            # utils.write_shell_script(list2cmdline(commands), name=flags.nanohedra, out_path=job.job_paths)
            terminate(results=commands)

        job.location = args.oligomer1
        design_source = os.path.splitext(os.path.basename(job.location))[0]
        job.sym_entry.log_parameters()
    else:
        # This logic is possible with job.module as select_poses with --metric or --dataframe
        # job.structure_db = None
        # job.api_db = None
        # design_source = os.path.basename(representative_pose_directory.project_designs)
        pass

    # -----------------------------------------------------------------------------------------------------------------
    # Set up Job specific details and resources
    # -----------------------------------------------------------------------------------------------------------------
    # Format computational requirements
    distribute_modules = [
        flags.nanohedra, flags.refine, flags.interface_design, flags.interface_metrics, flags.optimize_designs
    ]
    if job.module in distribute_modules:
        if job.distribute_work:
            logger.info('Writing modeling commands out to file, no modeling will occur until commands are executed')
        else:
            logger.info("Modeling will occur in this process, ensure you don't lose connection to the shell!")

    if job.multi_processing:
        logger.info(f'Starting multiprocessing using {job.cores} cores')
    else:
        logger.info(f'Starting processing. To increase processing speed, '
                    f'use --{flags.multi_processing} during submission')
    # if args.mpi:  # Todo implement
    #     # extras = ' mpi %d' % CommmandDistributer.mpi
    #     logger.info(
    #         'Setting job up for submission to MPI capable computer. Pose trajectories run in parallel,'
    #         ' %s at a time. This will speed up pose processing ~%f-fold.' %
    #         (CommmandDistributer.mpi - 1, flags.nstruct / (CommmandDistributer.mpi - 1)))
    #     queried_flags.update({'mpi': True, 'script': True})

    job.calculate_memory_requirements(len(pose_directories))
    # -----------------------------------------------------------------------------------------------------------------
    # Parse SubModule specific commands and performs the protocol specified.
    # Finally, run terminate(). This formats output parameters and reports on exceptions
    # -----------------------------------------------------------------------------------------------------------------
    results, success = [], []
    # exceptions = []
    # ---------------------------------------------------
    if args.module == flags.protocol:
        run_on_pose_directory = (
            putils.orient,
            putils.expand_asu,
            putils.rename_chains,
            putils.check_clashes,
            putils.generate_fragments,
            putils.interface_metrics,
            putils.optimize_designs,
            putils.refine,
            putils.interface_design,
            putils.analysis,
            putils.nanohedra
        )
        returns_pose_directories = (
            putils.nanohedra,
            putils.select_poses,
            putils.select_designs,
            putils.select_sequences
        )
        # terminate_options = dict(
        #     # analysis=dict(output_analysis=args.output),  # Replaced with args.output in terminate()
        # )
        # terminate_kwargs = {}
        # Universal protocol runner
        exceptions = []
        for protocol_name in job.modules:
            protocol = getattr(protocols, protocol_name)

            # Figure out how the job should be set up
            if protocol_name in run_on_pose_directory:  # Single poses
                if job.multi_processing:
                    results = utils.mp_map(protocol, pose_directories, processes=job.cores)
                else:
                    for pose_dir in pose_directories:
                        results.append(protocol(pose_dir))
            else:  # Collection of poses
                results = protocol(pose_directories)

            # Handle any returns that require particular treatment
            if protocol_name in returns_pose_directories:
                _results = []
                for result_list in results:
                    _results.extend(result_list)
                pose_directories = _results
            # elif putils.cluster_poses:  # Returns None
            #    pass

            # Update the current state of protocols and exceptions
            pose_directories, additional_exceptions = parse_protocol_results(pose_directories, results)
            exceptions.extend(additional_exceptions)
        #     # Retrieve any program flags necessary for termination
        #     terminate_kwargs.update(**terminate_options.get(protocol_name, {}))
        #
        # terminate(results=results, **terminate_kwargs, exceptions=exceptions)
        terminate(results=results, exceptions=exceptions)
    # ---------------------------------------------------
    if job.module == flags.orient:
        if args.multi_processing:
            results = utils.mp_map(PoseDirectory.orient, pose_directories, processes=job.cores)
        else:
            for design_dir in pose_directories:
                results.append(design_dir.orient())  # to_pose_directory=args.to_pose_directory))

        terminate(results=results)
    # ---------------------------------------------------
    # elif job.module == 'find_asu':
    #     if args.multi_processing:
    #         results = utils.mp_map(PoseDirectory.find_asu, pose_directories, processes=job.cores)
    #     else:
    #         for design_dir in pose_directories:
    #             results.append(design_dir.find_asu())
    #
    #     terminate(results=results)
    # ---------------------------------------------------
    elif job.module == 'find_transforms':
        # if args.multi_processing:
        #     results = SDUtils.mp_map(PoseDirectory.find_transforms, pose_directories, processes=job.cores)
        # else:
        stacked_transforms = [pose_directory.pose.entity_transformations for pose_directory in pose_directories]
        trans1_rot1, trans1_tx1, trans1_rot2, trans1_tx2 = zip(*[transform[0].values()
                                                                 for transform in stacked_transforms])
        trans2_rot1, trans2_tx1, trans2_rot2, trans2_tx2 = zip(*[transform[1].values()
                                                                 for transform in stacked_transforms])
        # Create the full dictionaries
        transformation1 = dict(rotation=trans1_rot1, translation=trans1_tx1,
                               rotation2=trans1_rot2, translation2=trans1_tx2)
        transformation2 = dict(rotation=trans2_rot1, translation=trans2_tx1,
                               rotation2=trans2_rot2, translation2=trans2_tx2)
        file1 = utils.pickle_object(transformation1, name='transformations1', out_path=os.getcwd())
        file2 = utils.pickle_object(transformation2, name='transformations2', out_path=os.getcwd())
        logger.info(f'Wrote transformation1 parameters to {file1}')
        logger.info(f'Wrote transformation2 parameters to {file2}')

        terminate(results=results)
    # ---------------------------------------------------
    elif job.module == flags.expand_asu:
        if args.multi_processing:
            results = utils.mp_map(PoseDirectory.expand_asu, pose_directories, processes=job.cores)
        else:
            for design_dir in pose_directories:
                results.append(design_dir.expand_asu())

        terminate(results=results)
    # ---------------------------------------------------
    elif job.module == flags.rename_chains:
        if args.multi_processing:
            results = utils.mp_map(PoseDirectory.rename_chains, pose_directories, processes=job.cores)
        else:
            for design_dir in pose_directories:
                results.append(design_dir.rename_chains())

        terminate(results=results)
    # ---------------------------------------------------
    # elif job.module == 'check_unmodelled_clashes':  # Todo
    #     if args.multi_processing:
    #         results = SDUtils.mp_map(PoseDirectory.check_unmodelled_clashes, pose_directories,
    #                                  processes=job.cores)
    #     else:
    #         for design_dir in pose_directories:
    #             results.append(design_dir.check_unmodelled_clashes())
    #
    #     terminate(results=results)
    # ---------------------------------------------------
    elif job.module == flags.check_clashes:
        if args.multi_processing:
            results = utils.mp_map(PoseDirectory.check_clashes, pose_directories, processes=job.cores)
        else:
            for design_dir in pose_directories:
                results.append(design_dir.check_clashes())

        terminate(results=results)
    # ---------------------------------------------------
    elif job.module == flags.generate_fragments:
        # job.write_fragments = True
        if args.multi_processing:
            results = utils.mp_map(PoseDirectory.generate_interface_fragments, pose_directories,
                                   processes=job.cores)
        else:
            for design in pose_directories:
                results.append(design.generate_interface_fragments())

        terminate(results=results)
    # ---------------------------------------------------
    elif job.module == flags.nanohedra:
        if args.multi_processing:
            results = utils.mp_map(protocols.fragdock.fragment_dock, pose_directories, processes=job.cores)
        else:
            for pair in pose_directories:
                results.append(protocols.fragdock.fragment_dock(pair))
        terminate(results=results, output=False)
    # ---------------------------------------------------
    elif job.module == flags.interface_metrics:
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            results = utils.mp_map(PoseDirectory.interface_metrics, pose_directories, processes=job.cores)
        else:
            for design in pose_directories:
                results.append(design.interface_metrics())

        terminate(results=results)
    # ---------------------------------------------------
    elif job.module == flags.optimize_designs:
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            results = utils.mp_map(PoseDirectory.optimize_designs, pose_directories, processes=job.cores)
        else:
            for design in pose_directories:
                results.append(design.optimize_designs())

        terminate(results=results)
    # ---------------------------------------------------
    # elif job.module == 'custom_script':
    #     # Start pose processing and preparation for Rosetta
    #     if args.multi_processing:
    #         zipped_args = zip(pose_directories, repeat(args.script), repeat(args.force), repeat(args.file_list),
    #                           repeat(args.native), repeat(job.suffix), repeat(args.score_only),
    #                           repeat(args.variables))
    #         results = utils.mp_starmap(PoseDirectory.custom_rosetta_script, zipped_args, processes=job.cores)
    #     else:
    #         for design in pose_directories:
    #             results.append(design.custom_rosetta_script(args.script, force=args.force,
    #                                                         file_list=args.file_list, native=args.native,
    #                                                         suffix=job.suffix, score_only=args.score_only,
    #                                                         variables=args.variables))
    #
    #     terminate(results=results)
    # ---------------------------------------------------
    elif job.module == flags.refine:
        if args.multi_processing:
            results = utils.mp_map(PoseDirectory.refine, pose_directories, processes=job.cores)
        else:
            for design in pose_directories:
                results.append(design.refine())

        terminate(results=results)
    # ---------------------------------------------------
    elif job.module == flags.interface_design:
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            results = utils.mp_map(PoseDirectory.interface_design, pose_directories, processes=job.cores)
        else:
            for design in pose_directories:
                results.append(design.interface_design())

        terminate(results=results)
    # ---------------------------------------------------
    elif job.module == flags.analysis:
        # Start pose analysis of all designed files
        if args.multi_processing:
            # zipped_args = zip(pose_directories, repeat(args.save), repeat(args.figures))
            # results = utils.mp_starmap(PoseDirectory.interface_design_analysis, zipped_args, processes=job.cores)
            results = utils.mp_map(PoseDirectory.interface_design_analysis, pose_directories, processes=job.cores)
        else:
            # @profile  # memory_profiler
            # def run_single_analysis():
            for design in pose_directories:
                results.append(design.interface_design_analysis())
            # run_single_analysis()
        terminate(results=results)  # , output_analysis=args.output)
    # ---------------------------------------------------
    elif job.module == flags.cluster_poses:
        protocols.cluster.cluster_poses(pose_directories)
        terminate(output=False)
    # ---------------------------------------------------
    elif job.module == flags.select_poses:
        # Need to initialize pose_directories, design_source to terminate()
        pose_directories = protocols.select.poses()
        design_source = job.program_root
        # Write out the chosen poses to a pose.paths file
        terminate(results=pose_directories)
    # ---------------------------------------------------
    elif job.module == flags.select_sequences:
        program_root = job.program_root
        if job.specification_file:
            loc_result = [(pose_directory, design) for pose_directory in pose_directories
                          for design in pose_directory.specific_designs]
            total_df = protocols.load_total_dataframe(pose_directories)
            selected_poses_df = \
                metrics.prioritize_design_indices(total_df.loc[loc_result, :], filter=job.filter, weight=job.weight,
                                                  protocol=job.protocol, function=job.weight_function)
            # Specify the result order according to any filtering, weighting, and select_number
            results = {}
            for pose_directory, design in selected_poses_df.index.to_list()[:job.select_number]:
                if pose_directory in results:
                    results[pose_directory].add(design)
                else:
                    results[pose_directory] = {design}

            save_poses_df = selected_poses_df.droplevel(0)  # .droplevel(0, axis=1).droplevel(0, axis=1)
            # convert to PoseDirectory objects
            # results = {pose_directory: results[str(pose_directory)] for pose_directory in pose_directories
            #            if str(pose_directory) in results}
        elif job.total:
            total_df = protocols.load_total_dataframe(pose_directories)
            if job.protocol:
                group_df = total_df.groupby('protocol')
                df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
                               keys=list(zip(group_df.groups, repeat('mean'))))
            else:
                df = pd.concat([total_df], axis=1, keys=['pose', 'metric'])
            # Figure out designs from dataframe, filters, and weights
            selected_poses_df = metrics.prioritize_design_indices(df, filter=job.filter, weight=job.weight,
                                                                  protocol=job.protocol, function=job.weight_function)
            selected_designs = selected_poses_df.index.to_list()
            job.select_number = \
                len(selected_designs) if len(selected_designs) < job.select_number else job.select_number
            if job.allow_multiple_poses:
                logger.info(f'Choosing {job.select_number} designs, from the top ranked designs regardless of pose')
                loc_result = selected_designs[:job.select_number]
                results = {pose_dir: design for pose_dir, design in loc_result}
            else:  # elif job.designs_per_pose:
                logger.info(f'Choosing up to {job.select_number} designs, with {job.designs_per_pose} designs per pose')
                number_chosen = 0
                selected_poses = {}
                for pose_directory, design in selected_designs:
                    designs = selected_poses.get(pose_directory, None)
                    if designs:
                        if len(designs) >= job.designs_per_pose:
                            # We already have too many, continue with search. No need to check as no addition
                            continue
                        selected_poses[pose_directory].add(design)
                    else:
                        selected_poses[pose_directory] = {design}
                    number_chosen += 1
                    if number_chosen == job.select_number:
                        break

                results = selected_poses
                loc_result = [(pose_dir, design) for pose_dir, designs in selected_poses.items() for design in designs]

            # Include only the found index names to the saved dataframe
            save_poses_df = selected_poses_df.loc[loc_result, :]  # .droplevel(0).droplevel(0, axis=1).droplevel(0, axis=1)
            # convert to PoseDirectory objects
            # results = {pose_directory: results[str(pose_directory)] for pose_directory in pose_directories
            #            if str(pose_directory) in results}
        else:  # Select designed sequences from each pose provided (PoseDirectory)
            trajectory_df = sequence_metrics = None  # Used to get the column headers
            try:
                example_trajectory = representative_pose_directory.trajectories
            except AttributeError:
                raise RuntimeError('Missing the representative_pose_directory. It must be initialized to continue')
                example_trajectory = None

            if job.filter:
                trajectory_df = pd.read_csv(example_trajectory, index_col=0, header=[0])
                sequence_metrics = set(trajectory_df.columns.get_level_values(-1).to_list())
                sequence_filters = metrics.query_user_for_metrics(sequence_metrics, mode='filter', level='sequence')
            else:
                sequence_filters = None

            if job.weight:
                if not trajectory_df:
                    trajectory_df = pd.read_csv(example_trajectory, index_col=0, header=[0])
                    sequence_metrics = set(trajectory_df.columns.get_level_values(-1).to_list())
                sequence_weights = metrics.query_user_for_metrics(sequence_metrics, mode='weight', level='sequence')
            else:
                sequence_weights = None

            if job.multi_processing:
                # sequence_weights = {'buns_per_ang': 0.2, 'observed_evolution': 0.3, 'shape_complementarity': 0.25,
                #                     'int_energy_res_summary_delta': 0.25}
                zipped_args = zip(pose_directories, repeat(sequence_filters), repeat(sequence_weights),
                                  repeat(job.designs_per_pose), repeat(job.protocol))
                # result_mp = zip(*SDUtils.mp_starmap(Ams.select_sequences, zipped_args, processes=job.cores))
                result_mp = utils.mp_starmap(PoseDirectory.select_sequences, zipped_args, processes=job.cores)
                # results - contains tuple of (PoseDirectory, design index) for each sequence
                # could simply return the design index then zip with the directory
                results = {pose_dir: designs for pose_dir, designs in zip(pose_directories, result_mp)}
            else:
                results = {pose_dir: pose_dir.select_sequences(filters=sequence_filters, weights=sequence_weights,
                                                               number=job.designs_per_pose, protocols=job.protocol)
                           for pose_dir in pose_directories}
            # Todo there is no sort here so the select_number isn't really doing anything
            results = {pose_dir: designs for pose_dir, designs in list(results.items())[:job.select_number]}
            loc_result = [(pose_dir, design) for pose_dir, designs in results.items() for design in designs]
            total_df = protocols.load_total_dataframe(pose_directories)
            save_poses_df = total_df.loc[loc_result, :].droplevel(0).droplevel(0, axis=1).droplevel(0, axis=1)

        # Format selected sequences for output
        if job.prefix == '':
            job.prefix = f'{os.path.basename(os.path.splitext(job.location)[0])}_'

        outdir = os.path.join(os.path.dirname(program_root), f'{job.prefix}SelectedDesigns{job.suffix}')
        putils.make_path(outdir)

        if job.total and job.save_total:
            total_df_filename = os.path.join(outdir, 'TotalPosesTrajectoryMetrics.csv')
            total_df.to_csv(total_df_filename)
            logger.info(f'Total Pose/Designs DataFrame was written to: {total_df}')

        logger.info(f'{len(save_poses_df)} poses were selected')
        # if save_poses_df is not None:  # Todo make work if DataFrame is empty...
        if job.filter or job.weight:
            new_dataframe = os.path.join(outdir, f'{utils.starttime}-{"Filtered" if job.filter else ""}'
                                                 f'{"Weighted" if job.weight else ""}DesignMetrics.csv')
        else:
            new_dataframe = os.path.join(outdir, f'{utils.starttime}-DesignMetrics.csv')
        save_poses_df.to_csv(new_dataframe)
        logger.info(f'New DataFrame with selected designs was written to: {new_dataframe}')

        logger.info(f'Relevant design files are being copied to the new directory: {outdir}')
        # Create new output of designed PDB's  # TODO attach the state to these files somehow for further SymDesign use
        exceptions = []
        for pose_dir, designs in results.items():
            for design in designs:
                file_path = os.path.join(pose_dir.designs, f'*{design}*')
                file = sorted(glob(file_path))
                if not file:  # add to exceptions
                    exceptions.append((pose_dir, f'No file found for "{file_path}"'))
                    continue
                out_path = os.path.join(outdir, f'{pose_dir}_design_{design}.pdb')
                if not os.path.exists(out_path):
                    shutil.copy(file[0], out_path)  # [i])))
                    # shutil.copy(des_dir.trajectories, os.path.join(outdir_traj, os.path.basename(des_dir.trajectories)))
                    # shutil.copy(des_dir.residues, os.path.join(outdir_res, os.path.basename(des_dir.residues)))
            # try:
            #     # Create symbolic links to the output PDB's
            #     os.symlink(file[0], os.path.join(outdir, '%s_design_%s.pdb' % (str(des_dir), design)))  # [i])))
            #     os.symlink(des_dir.trajectories, os.path.join(outdir_traj, os.path.basename(des_dir.trajectories)))
            #     os.symlink(des_dir.residues, os.path.join(outdir_res, os.path.basename(des_dir.residues)))
            # except FileExistsError:
            #     pass

        # Check if sequences should be generated
        if job.skip_sequence_generation:
            terminate(exceptions=exceptions, output=False)
        else:
            # Format sequences for expression
            job.output_file = os.path.join(outdir, f'{job.prefix}SelectedDesigns{job.suffix}.paths')
            # pose_directories = list(results.keys())
            with open(job.output_file, 'w') as f:
                f.write('%s\n' % '\n'.join(pose_dir.path for pose_dir in list(results.keys())))

        # Use one directory as indication of entity specification for them all. Todo modify for different length inputs
        representative_pose_directory.load_pose()
        if args.tag_entities:
            if args.tag_entities == 'all':
                tag_index = [True for _ in representative_pose_directory.pose.entities]
                number_of_tags = len(representative_pose_directory.pose.entities)
            elif args.tag_entities == 'single':
                tag_index = [True for _ in representative_pose_directory.pose.entities]
                number_of_tags = 1
            elif args.tag_entities == 'none':
                tag_index = [False for _ in representative_pose_directory.pose.entities]
                number_of_tags = None
            else:
                tag_specified_list = list(map(str.translate, set(args.entity_specification.split(',')).difference(['']),
                                              repeat(utils.digit_translate_table)))
                for idx, item in enumerate(tag_specified_list):
                    try:
                        tag_specified_list[idx] = int(item)
                    except ValueError:
                        continue

                for _ in range(len(representative_pose_directory.pose.entities) - len(tag_specified_list)):
                    tag_specified_list.append(0)
                tag_index = [True if is_tag else False for is_tag in tag_specified_list]
                number_of_tags = sum(tag_specified_list)
        else:
            tag_index = [False for _ in representative_pose_directory.pose.entities]
            number_of_tags = None

        if args.multicistronic or args.multicistronic_intergenic_sequence:
            args.multicistronic = True
            if args.multicistronic_intergenic_sequence:
                intergenic_sequence = args.multicistronic_intergenic_sequence
            else:
                intergenic_sequence = ProteinExpression.default_multicistronic_sequence
        else:
            intergenic_sequence = ''

        missing_tags = {}  # result: [True, True] for result in results
        tag_sequences, final_sequences, inserted_sequences, nucleotide_sequences = {}, {}, {}, {}
        codon_optimization_errors = {}
        # for des_dir, design in results:
        for des_dir, designs in results.items():
            des_dir.load_pose()  # source=des_dir.asu_path)
            des_dir.pose.rename_chains()  # Do I need to modify chains?
            for design in designs:
                file_glob = f'{des_dir.designs}{os.sep}*{design}*'
                file = sorted(glob(file_glob))
                if not file:
                    logger.error(f'No file found for {file_glob}')
                    continue
                design_pose = Model.from_file(file[0], log=des_dir.log, entity_names=des_dir.entity_names)
                designed_atom_sequences = [entity.sequence for entity in design_pose.entities]

                missing_tags[(des_dir, design)] = [1 for _ in des_dir.pose.entities]
                prior_offset = 0
                # all_missing_residues = {}
                # mutations = []
                # referenced_design_sequences = {}
                sequences_and_tags = {}
                entity_termini_availability, entity_helical_termini = {}, {}
                for idx, (source_entity, design_entity) in enumerate(zip(des_dir.pose.entities, design_pose.entities)):
                    # source_entity.retrieve_info_from_api()
                    # source_entity.reference_sequence
                    sequence_id = f'{des_dir}_{source_entity.name}'
                    # design_string = '%s_design_%s_%s' % (des_dir, design, source_entity.name)  # [i])), pdb_code)
                    design_string = f'{design}_{source_entity.name}'
                    termini_availability = des_dir.pose.get_termini_accessibility(source_entity)
                    logger.debug(f'Design {sequence_id} has the following termini accessible for tags: '
                                 f'{termini_availability}')
                    if args.avoid_tagging_helices:
                        termini_helix_availability = \
                            des_dir.pose.get_termini_accessibility(source_entity, report_if_helix=True)
                        logger.debug(f'Design {sequence_id} has the following helical termini available: '
                                     f'{termini_helix_availability}')
                        termini_availability = {'n': termini_availability['n'] and not termini_helix_availability['n'],
                                                'c': termini_availability['c'] and not termini_helix_availability['c']}
                        entity_helical_termini[design_string] = termini_helix_availability
                    logger.debug(f'The termini {termini_availability} are available for tagging')
                    entity_termini_availability[design_string] = termini_availability
                    true_termini = [term for term, is_true in termini_availability.items() if is_true]

                    # Find sequence specified attributes required for expression formatting
                    # disorder = generate_mutations(source_entity.sequence, source_entity.reference_sequence,
                    #                               only_gaps=True)
                    # disorder = source_entity.disorder
                    source_offset = source_entity.offset_index
                    indexed_disordered_residues = {res_number + source_offset + prior_offset: mutation
                                                   for res_number, mutation in source_entity.disorder.items()}
                    # Todo, moved below indexed_disordered_residues on 7/26, ensure correct!
                    prior_offset += len(indexed_disordered_residues)
                    # generate the source TO design mutations before any disorder handling
                    mutations = generate_mutations(source_entity.sequence, design_entity.sequence, offset=False)
                    # Insert the disordered residues into the design pose
                    for residue_number, mutation in indexed_disordered_residues.items():
                        logger.debug(f'Inserting {mutation["from"]} into position {residue_number} on chain '
                                     f'{source_entity.chain_id}')
                        design_pose.insert_residue_type(mutation['from'], at=residue_number,
                                                        chain=source_entity.chain_id)
                        # adjust mutations to account for insertion
                        for mutation_index in sorted(mutations.keys(), reverse=True):
                            if mutation_index < residue_number:
                                break
                            else:  # mutation should be incremented by one
                                mutations[mutation_index + 1] = mutations.pop(mutation_index)

                    # Check for expression tag addition to the designed sequences after disorder addition
                    inserted_design_sequence = design_entity.sequence
                    selected_tag = {}
                    available_tags = ProteinExpression.find_expression_tags(inserted_design_sequence)
                    if available_tags:  # look for existing tag to remove from sequence and save identity
                        tag_names, tag_termini, existing_tag_sequences = \
                            zip(*[(tag['name'], tag['termini'], tag['sequence']) for tag in available_tags])
                        try:
                            preferred_tag_index = tag_names.index(args.preferred_tag)
                            if tag_termini[preferred_tag_index] in true_termini:
                                selected_tag = available_tags[preferred_tag_index]
                        except ValueError:
                            pass
                        pretag_sequence = ProteinExpression.remove_expression_tags(inserted_design_sequence,
                                                                                   existing_tag_sequences)
                    else:
                        pretag_sequence = inserted_design_sequence
                    logger.debug(f'The pretag sequence is:\n{pretag_sequence}')

                    # Find the open reading frame offset using the structure sequence after insertion
                    offset = find_orf_offset(pretag_sequence, mutations)
                    formatted_design_sequence = pretag_sequence[offset:]
                    logger.debug(f'The open reading frame offset is {offset}')
                    logger.debug(f'The formatted_design sequence is:\n{formatted_design_sequence}')

                    if number_of_tags is None:  # don't solve tags
                        sequences_and_tags[design_string] = {'sequence': formatted_design_sequence, 'tag': {}}
                        continue

                    if not selected_tag:  # find compatible tags from matching PDB observations
                        uniprot_id = source_entity.uniprot_id
                        uniprot_id_matching_tags = tag_sequences.get(uniprot_id, None)
                        if not uniprot_id_matching_tags:
                            uniprot_id_matching_tags = \
                                ProteinExpression.find_matching_expression_tags(uniprot_id=uniprot_id)
                            tag_sequences[uniprot_id] = uniprot_id_matching_tags

                        if uniprot_id_matching_tags:
                            tag_names, tag_termini, _ = \
                                zip(*[(tag['name'], tag['termini'], tag['sequence'])
                                      for tag in uniprot_id_matching_tags])
                        else:
                            tag_names, tag_termini, _ = [], [], []

                        iteration = 0
                        while iteration < len(tag_names):
                            try:
                                preferred_tag_index_2 = tag_names[iteration:].index(args.preferred_tag)
                                if tag_termini[preferred_tag_index_2] in true_termini:
                                    selected_tag = uniprot_id_matching_tags[preferred_tag_index_2]
                                    break
                            except ValueError:
                                selected_tag = \
                                    ProteinExpression.select_tags_for_sequence(sequence_id,
                                                                               uniprot_id_matching_tags,
                                                                               preferred=args.preferred_tag,
                                                                               **termini_availability)
                                break
                            iteration += 1

                    if selected_tag.get('name'):
                        missing_tags[(des_dir, design)][idx] = 0
                        logger.debug(f'The pre-existing, identified tag is:\n{selected_tag}')
                    sequences_and_tags[design_string] = {'sequence': formatted_design_sequence, 'tag': selected_tag}

                # after selecting all tags, consider tagging the design as a whole
                if number_of_tags is not None:
                    number_of_found_tags = len(des_dir.pose.entities) - sum(missing_tags[(des_dir, design)])
                    if number_of_tags > number_of_found_tags:
                        print(f'There were {number_of_tags} requested tags for design {des_dir} and '
                              f'{number_of_found_tags} were found')
                        current_tag_options = \
                            '\n\t'.join([f'{i} - {entity_name}\n'
                                         f'\tAvailable Termini: {entity_termini_availability[entity_name]}'
                                         f'\n\t\t   TAGS: {tag_options["tag"]}'
                                         for i, (entity_name, tag_options) in enumerate(sequences_and_tags.items(), 1)])
                        print(f'Current Tag Options:\n\t{current_tag_options}')
                        if args.avoid_tagging_helices:
                            print('Helical Termini:\n\t%s'
                                  % '\n\t'.join(f'{entity_name}\t{availability}'
                                                for entity_name, availability in entity_helical_termini.items()))
                        satisfied = input('If this is acceptable, enter "continue", otherwise, '
                                          f'you can modify the tagging options with any other input.{input_string}')
                        if satisfied == 'continue':
                            number_of_found_tags = number_of_tags

                        iteration_idx = 0
                        while number_of_tags != number_of_found_tags:
                            if iteration_idx == len(missing_tags[(des_dir, design)]):
                                print(f'You have seen all options, but the number of requested tags ({number_of_tags}) '
                                      f"doesn't equal the number selected ({number_of_found_tags})")
                                satisfied = input('If you are satisfied with this, enter "continue", otherwise enter '
                                                  'anything and you can view all remaining options starting from the '
                                                  f'first entity{input_string}')
                                if satisfied == 'continue':
                                    break
                                else:
                                    iteration_idx = 0
                            for idx, entity_missing_tag in enumerate(missing_tags[(des_dir, design)][iteration_idx:]):
                                sequence_id = f'{des_dir}_{des_dir.pose.entities[idx].name}'
                                if entity_missing_tag and tag_index[idx]:  # isn't tagged but could be
                                    print(f'Entity {sequence_id} is missing a tag. Would you like to tag this entity?')
                                    if not boolean_choice():
                                        continue
                                else:
                                    continue
                                if args.preferred_tag:
                                    tag = args.preferred_tag
                                    while True:
                                        termini = input('Your preferred tag will be added to one of the termini. Which '
                                                        f'termini would you prefer? [n/c]{input_string}')
                                        if termini.lower() in ['n', 'c']:
                                            break
                                        else:
                                            print(f'"{termini}" is an invalid input. One of "n" or "c" is required')
                                else:
                                    while True:
                                        tag_input = input('What tag would you like to use? Enter the number of the '
                                                          f'below options.\n\t%s\n{input_string}' %
                                                          '\n\t'.join([f'{i} - {tag}'
                                                                       for i, tag in enumerate(
                                                                        ProteinExpression.expression_tags, 1)]))
                                        if tag_input.isdigit():
                                            tag_input = int(tag_input)
                                            if tag_input <= len(ProteinExpression.expression_tags):
                                                tag = list(ProteinExpression.expression_tags.keys())[tag_input - 1]
                                                break
                                        print("Input doesn't match available options. Please try again")
                                    while True:
                                        termini = input('Your tag will be added to one of the termini. Which termini '
                                                        f'would you prefer? [n/c]{input_string}')
                                        if termini.lower() in ['n', 'c']:
                                            break
                                        else:
                                            print(f'"{termini}" is an invalid input. One of "n" or "c" is required')

                                selected_entity = list(sequences_and_tags.keys())[idx]
                                if termini == 'n':
                                    new_tag_sequence = \
                                        ProteinExpression.expression_tags[tag] + 'SG' \
                                        + sequences_and_tags[selected_entity]['sequence'][:12]
                                else:  # termini == 'c'
                                    new_tag_sequence = \
                                        sequences_and_tags[selected_entity]['sequence'][-12:] \
                                        + 'GS' + ProteinExpression.expression_tags[tag]
                                sequences_and_tags[selected_entity]['tag'] = {'name': tag, 'sequence': new_tag_sequence}
                                missing_tags[(des_dir, design)][idx] = 0
                                break

                            iteration_idx += 1
                            number_of_found_tags = len(des_dir.pose.entities) - sum(missing_tags[(des_dir, design)])

                    elif number_of_tags < number_of_found_tags:  # when more than the requested number of tags were id'd
                        print(f'There were only {number_of_tags} requested tags for design {des_dir} and '
                              f'{number_of_found_tags} were found')
                        while number_of_tags != number_of_found_tags:
                            tag_input = input(f'Which tag would you like to remove? Enter the number of the currently '
                                              f'configured tag option that you would like to remove. If you would like '
                                              f'to keep all, specify "keep"\n\t%s\n{input_string}'
                                              % '\n\t'.join([f'{i} - {entity_name}\n\t\t{tag_options["tag"]}'
                                                             for i, (entity_name, tag_options)
                                                             in enumerate(sequences_and_tags.items(), 1)]))
                            if tag_input == 'keep':
                                break
                            elif tag_input.isdigit():
                                tag_input = int(tag_input)
                                if tag_input <= len(sequences_and_tags):
                                    missing_tags[(des_dir, design)][tag_input - 1] = 1
                                    selected_entity = list(sequences_and_tags.keys())[tag_input - 1]
                                    sequences_and_tags[selected_entity]['tag'] = \
                                        {'name': None, 'termini': None, 'sequence': None}
                                    # tag = list(ProteinExpression.expression_tags.keys())[tag_input - 1]
                                    break
                                else:
                                    print("Input doesn't match an integer from the available options. Please try again")
                            else:
                                print(f'"{tag_input}" is an invalid input. Try again')
                            number_of_found_tags = len(des_dir.pose.entities) - sum(missing_tags[(des_dir, design)])

                # Apply all tags to the sequences
                from symdesign.structure.utils import protein_letters_alph1
                # Todo indicate the linkers that will be used!
                #  Request a new one if not ideal!
                cistronic_sequence = ''
                for idx, (design_string, sequence_tag) in enumerate(sequences_and_tags.items()):
                    tag, sequence = sequence_tag['tag'], sequence_tag['sequence']
                    # print('TAG:\n', tag.get('sequence'), '\nSEQUENCE:\n', sequence)
                    design_sequence = ProteinExpression.add_expression_tag(tag.get('sequence'), sequence)
                    if tag.get('sequence') and design_sequence == sequence:  # tag exists and no tag added
                        tag_sequence = ProteinExpression.expression_tags[tag.get('name')]
                        if tag.get('termini') == 'n':
                            if design_sequence[0] == 'M':  # remove existing Met to append tag to n-term
                                design_sequence = design_sequence[1:]
                            design_sequence = tag_sequence + 'SG' + design_sequence
                        else:  # termini == 'c'
                            design_sequence = design_sequence + 'GS' + tag_sequence

                    # If no MET start site, include one
                    if design_sequence[0] != 'M':
                        design_sequence = f'M{design_sequence}'

                    # If there is an unrecognized amino acid, modify
                    if 'X' in design_sequence:
                        logger.critical(f'An unrecognized amino acid was specified in the sequence {design_string}. '
                                        'This requires manual intervention!')
                        # idx = 0
                        seq_length = len(design_sequence)
                        while True:
                            idx = design_sequence.find('X')
                            if idx == -1:  # Todo clean
                                break
                            idx_range = (idx - 6 if idx - 6 > 0 else 0,
                                         idx + 6 if idx + 6 < seq_length else seq_length)
                            while True:
                                new_amino_acid = input('What amino acid should be swapped for "X" in this sequence '
                                                       f'context?\n\t{idx_range[0] + 1}'
                                                       f'{" " * (len(range(*idx_range))-(len(str(idx_range[0]))+1))}'
                                                       f'{idx_range[1] + 1}'
                                                       f'\n\t{design_sequence[idx_range[0]:idx_range[1]]}'
                                                       f'{input_string}').upper()
                                if new_amino_acid in protein_letters_alph1:
                                    design_sequence = design_sequence[:idx] + new_amino_acid + design_sequence[idx + 1:]
                                    break
                                else:
                                    print(f"{new_amino_acid} doesn't match a single letter canonical amino acid. "
                                          "Please try again")

                    # For a final manual check of sequence generation, find sequence additions compared to the design
                    # model and save to view where additions lie on sequence. Cross these additions with design
                    # structure to check if insertions are compatible
                    all_insertions = {residue: {'to': aa} for residue, aa in enumerate(design_sequence, 1)}
                    all_insertions.update(generate_mutations(design_sequence, designed_atom_sequences[idx],
                                                             blanks=True))
                    # Reduce to sequence only
                    inserted_sequences[design_string] = \
                        f'{"".join([res["to"] for res in all_insertions.values()])}\n{design_sequence}'
                    logger.info(f'Formatted sequence comparison:\n{inserted_sequences[design_string]}')
                    final_sequences[design_string] = design_sequence
                    if args.nucleotide:
                        try:
                            nucleotide_sequence = \
                                ProteinExpression.optimize_protein_sequence(design_sequence,
                                                                            species=args.optimize_species)
                        except NoSolutionError:  # add the protein sequence?
                            logger.warning(f'Optimization of {design_string} was not successful!')
                            codon_optimization_errors[design_string] = design_sequence
                            break

                        if args.multicistronic:
                            if idx > 0:
                                cistronic_sequence += intergenic_sequence
                            cistronic_sequence += nucleotide_sequence
                        else:
                            nucleotide_sequences[design_string] = nucleotide_sequence
                if args.multicistronic:
                    nucleotide_sequences[str(des_dir)] = cistronic_sequence

        # Report Errors
        if codon_optimization_errors:
            # Todo utilize errors
            error_file = \
                write_sequences(codon_optimization_errors, csv=args.csv,
                                file_name=os.path.join(outdir, f'{job.prefix}OptimizationErrorProteinSequences'
                                                               f'{job.suffix}'))
        # Write output sequences to fasta file
        seq_file = write_sequences(final_sequences, csv=args.csv,
                                   file_name=os.path.join(outdir, f'{job.prefix}SelectedSequences{job.suffix}'))
        logger.info(f'Final Design protein sequences written to: {seq_file}')
        seq_comparison_file = \
            write_sequences(inserted_sequences, csv=args.csv,
                            file_name=os.path.join(outdir,
                                                   f'{job.prefix}SelectedSequencesExpressionAdditions'
                                                   f'{job.suffix}'))
        logger.info(f'Final Expression sequence comparison to Design sequence written to: {seq_comparison_file}')
        # check for protein or nucleotide output
        if args.nucleotide:
            nucleotide_sequence_file = \
                write_sequences(nucleotide_sequences, csv=args.csv,
                                file_name=os.path.join(outdir, f'{job.prefix}SelectedSequencesNucleotide'
                                                               f'{job.suffix}'))
            logger.info(f'Final Design nucleotide sequences written to: {nucleotide_sequence_file}')
    # # ---------------------------------------------------
    # elif job.module == 'status':  # -n number, -s stage, -u update
    #     if args.update:
    #         for design in pose_directories:
    #             update_status(design.serialized_info, args.stage, mode=args.update)
    #     else:
    #         if args.number_of_trajectories:
    #             logger.info('Checking for %d files based on --number_of_trajectories flag' % args.number_of_trajectories)
    #         if args.stage:
    #             status(pose_directories, args.stage, number=args.number_of_trajectories)
    #         else:
    #             for stage in putils.stage_f:
    #                 s = status(pose_directories, stage, number=args.number_of_trajectories)
    #                 if s:
    #                     logger.info('For "%s" stage, default settings should generate %d files'
    #                                 % (stage, putils.stage_f[stage]['len']))
    # ---------------------------------------------------
    elif job.module == 'visualize':
        import visualization.VisualizeUtils as VSUtils
        from pymol import cmd

        # if 'escher' in sys.argv[1]:
        if not args.directory:
            exit(f'A directory with the desired designs must be specified using -d/--{flags.directory}')

        if ':' in args.directory:  # args.file  Todo job.location
            print('Starting the data transfer from remote source now...')
            os.system(f'scp -r {args.directory} .')
            file_dir = os.path.basename(args.directory)
        else:  # assume the files are local
            file_dir = args.directory
        # files = VSUtils.get_all_file_paths(file_dir, extension='.pdb', sort=not args.order)

        if args.order == 'alphabetical':
            files = VSUtils.get_all_file_paths(file_dir, extension='.pdb')  # sort=True)
        else:  # if args.order == 'none':
            files = VSUtils.get_all_file_paths(file_dir, extension='.pdb', sort=False)

        print(f'FILES:\n {files[:4]}')
        if args.order == 'paths':  # TODO FIX janky paths handling below
            # for design in pose_directories:
            with open(args.file[0], 'r') as f:
                paths = \
                    map(str.replace, map(str.strip, f.readlines()),
                        repeat('/yeates1/kmeador/Nanohedra_T33/SymDesignOutput/Projects/'
                               'NanohedraEntry54DockedPoses_Designs/'), repeat(''))
                paths = list(paths)
            ordered_files = []
            for path in paths:
                for file in files:
                    if path in file:
                        ordered_files.append(file)
                        break
            files = ordered_files
            # raise NotImplementedError('--order choice "paths" hasn\'t been set up quite yet... Use another method')
            # ordered_files = []
            # for index in df.index:
            #     for file in files:
            #         if index in file:
            #             ordered_files.append(file)
            #             break
            # files = ordered_files
        elif args.order == 'dataframe':
            if not args.dataframe:
                df_glob = sorted(glob(os.path.join(file_dir, 'TrajectoryMetrics.csv')))
                try:
                    args.dataframe = df_glob[0]
                except IndexError:
                    raise IndexError(f"There was no --{flags.dataframe} specified and one couldn't be located in "
                                     f'{job.location}. Initialize again with the path to the relevant dataframe')

            df = pd.read_csv(args.dataframe, index_col=0, header=[0])
            print('INDICES:\n %s' % df.index.to_list()[:4])
            ordered_files = []
            for index in df.index:
                for file in files:
                    # if index in file:
                    if os.path.splitext(os.path.basename(file))[0] in index:
                        ordered_files.append(file)
                        break
            # print('ORDERED FILES (%d):\n %s' % (len(ordered_files), ordered_files))
            files = ordered_files

        if not files:
            exit(f'No .pdb files found at "{job.location}". Are you sure this is correct?')

        # if len(sys.argv) > 2:
        #     low, high = map(float, sys.argv[2].split('-'))
        #     low_range, high_range = int((low / 100) * len(files)), int((high / 100) * len(files))
        #     if low_range < 0 or high_range > len(files):
        #         raise ValueError('The input range is outside of the acceptable bounds [0-100]')
        #     print('Selecting Designs within range: %d-%d' % (low_range if low_range else 1, high_range))
        # else:
        print(low_range, high_range)
        print(all_poses)
        for idx, file in enumerate(files[low_range:high_range], low_range + 1):
            if args.name == 'original':
                cmd.load(file)
            else:  # if args.name == 'numerical':
                cmd.load(file, object=idx)

        print('\nTo expand all designs to the proper symmetry, issue:\nPyMOL> expand name=all, symmetry=T'
              '\nYou should replace "T" with whatever symmetry your design is in\n')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nRun Ended By KeyboardInterrupt\n')
        exit(2)
