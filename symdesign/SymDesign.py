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
try:
    from memory_profiler import profile
    profile_error = None
except ImportError as profile_error:
    profile = None

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
from symdesign import flags, protocols, utils
from symdesign.protocols import PoseDirectory
from symdesign.resources.job import job_resources_factory
from symdesign.resources.query.pdb import retrieve_pdb_entries_by_advanced_query
from symdesign.resources.query.utils import validate_input_return_response_value
from symdesign.structure.fragment.db import fragment_factory, euler_factory
from symdesign.utils import guide, nanohedra, ProteinExpression

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
            if job.module == flags.interface_design:
                design_stage = putils.scout if job.design.scout \
                    else (putils.hbnet_design_profile if job.design.hbnet
                          else (putils.structure_background if job.design.structure_background
                                else putils.interface_design))
            else:
                design_stage = flags.design

            module_files = {
                flags.design: design_stage,
                flags.interface_design: design_stage,
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

                if job.module in [flags.interface_design, flags.design] and job.initial_refinement:
                    # We should refine before design
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
    if args.module:
        if args.guide:
            try:
                module_guide = getattr(guide, args.module.replace('-', '_'))
                exit(module_guide)
            except AttributeError:
                exit(f'There is no guide created for {args.module} yet. Try --help instead of --guide')
            # Tod0 below are known to be missing
            # custom_script
            # check_clashes
            # residue_selector
            # visualize
            #     print('Usage: %s -r %s -- [-d %s, -df %s, -f %s] visualize --range 0-10'
            #           % (SDUtils.ex_path('pymol'), putils.program_command.replace('python ', ''),
            #              SDUtils.ex_path('pose_directory'), SDUtils.ex_path('DataFrame.csv'),
            #              SDUtils.ex_path('design.paths')))
        # else:  # Print the full program readme and exit
        #     guide.print_guide()
        #     exit()
    elif args.setup:
        guide.setup_instructions()
        exit()
    else:  # Print the full program readme and exit
        guide.print_guide()
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
    if job.module in [flags.interface_design, flags.design, flags.generate_fragments, flags.orient, flags.expand_asu,
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
        # if job.module == flags.select_designs:  # Alias to module select_sequences with --skip-sequence-generation
        #     job.module = flags.select_sequences
        #     job.skip_sequence_generation = True
        elif job.module == flags.select_poses:
            # When selecting by dataframe or metric, don't initialize, input is handled in module protocol
            if job.dataframe or job.metric:
                initialize = False
        elif job.module in [flags.select_designs, flags.select_sequences] \
                and job.number == sys.maxsize and not args.total:
            # Change default number to a single sequence/pose when not doing a total selection
            job.number = 1
        elif job.module in flags.cluster_poses and job.number == sys.maxsize:
            # Change default number to a single pose
            job.number = 1
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

    logger.info(f'Using resources in Database located at "{job.data}"')
    if job.module in [flags.nanohedra, flags.generate_fragments, flags.interface_design, flags.design, flags.analysis]:
        if job.design.term_constraint:
            job.fragment_db = fragment_factory(source=args.fragment_database)
            # Initialize EulerLookup class
            euler_factory()
            if job.module == flags.generate_fragments and job.fragment_db.source == putils.biological_interfaces:
                logger.warning(f'The FragmentDatabase {job.fragment_db.source} has only been created with '
                               'biological homo-oligomers. Use fragment information with caution')
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
            pose_directories = [PoseDirectory.from_pose_id(pose, root=args.directory, specific_designs=designs,
                                                           directives=directives)
                                for pose, designs, directives in design_specification.get_directives()]
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
        initialize_modules = [flags.interface_design, flags.design, flags.interface_metrics, flags.optimize_designs]
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
                # for pose in pose_directories:
                #     pose.initialize_structure_attributes()

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
                job.initial_refinement = job.initial_loop_model = True
            else:
                preprocess_instructions, initial_refinement, initial_loop_model = \
                    job.structure_db.preprocess_structures_for_design(all_structures,
                                                                      script_out_path=job.sbatch_scripts)
                #                                                       batch_commands=args.distribute_work)
                if info_messages and preprocess_instructions:
                    info_messages += ['The following can be run at any time regardless of evolutionary script progress']
                info_messages += preprocess_instructions
                job.initial_refinement = initial_refinement
                job.initial_loop_model = initial_loop_model

            # Entity processing commands are needed
            if load_resources or not job.initial_refinement or not job.initial_loop_model:
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
            # Todo set up a job based data acquisition as this takes some time and loading everythin isn't necessary!
            job.structure_db.load_all_data()
            job.api_db.load_all_data()
        # Set up in series
        for pose in pose_directories:
            pose.initialize_structure_attributes(pre_refine=job.initial_refinement,
                                                 pre_loop_model=job.initial_loop_model)

        logger.info(f'{len(pose_directories)} unique poses found in "{job.location}"')
        if not job.debug and not job.skip_logging:
            if representative_pose_directory.log_path:
                logger.info(f'All design specific logs are located in their corresponding directories\n\tEx: '
                            f'{representative_pose_directory.log_path}')

    elif job.module == flags.nanohedra:
        logger.info(f'Setting up inputs for {job.module.title()} docking')
        job.sym_entry.log_parameters()
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
            job.initial_refinement = job.initial_loop_model = True
        else:
            preprocess_instructions, initial_refinement, initial_loop_model = \
                job.structure_db.preprocess_structures_for_design(all_structures, script_out_path=job.sbatch_scripts)
            #                                                       batch_commands=args.distribute_work)
            if info_messages and preprocess_instructions:
                info_messages += ['The following can be run at any time regardless of evolutionary script progress']
            info_messages += preprocess_instructions
            job.initial_refinement = initial_refinement
            job.initial_loop_model = initial_loop_model

        # Entity processing commands are needed
        if load_resources or not job.initial_refinement or not job.initial_loop_model:
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
    else:
        # This logic is possible with job.module as select_poses with --metric or --dataframe
        # job.structure_db = None
        # job.api_db = None
        # design_source = os.path.basename(representative_pose_directory.project_designs)
        pass

    # -----------------------------------------------------------------------------------------------------------------
    #  Set up Job specific details and resources
    # -----------------------------------------------------------------------------------------------------------------
    # Format computational requirements
    distribute_modules = [
        flags.nanohedra, flags.refine, flags.interface_design, flags.design, flags.interface_metrics,
        flags.optimize_designs
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
    #  Parse SubModule specific commands and performs the protocol specified.
    # -----------------------------------------------------------------------------------------------------------------
    results = []
    exceptions = []
    # ---------------------------------------------------
    if args.module == flags.protocol:  # Use args.module as job.module is set as first in protocol
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
            putils.design,
            putils.analysis,
            putils.nanohedra
        )
        returns_pose_directories = (
            putils.nanohedra,
            putils.select_poses,
            putils.select_designs,
            # putils.select_sequences
        )
        # terminate_options = dict(
        #     # analysis=dict(output_analysis=args.output),  # Replaced with args.output in terminate()
        # )
        # terminate_kwargs = {}
        # Universal protocol runner
        for idx, protocol_name in enumerate(job.modules, 1):
            logger.info(f'Starting protocol {idx}: {protocol_name}')
            # Update this mechanism with each module
            job.module = protocol_name

            # Fetch the specified protocol
            protocol = getattr(protocols, protocol_name)
            # Figure out how the job should be set up
            if protocol_name in run_on_pose_directory:  # Single poses
                if job.multi_processing:
                    _results = utils.mp_map(protocol, pose_directories, processes=job.cores)
                else:
                    _results = []
                    for pose_dir in pose_directories:
                        _results.append(protocol(pose_dir))
            else:  # Collection of poses
                _results = protocol(pose_directories)

            # Handle any returns that require particular treatment
            if protocol_name in returns_pose_directories:
                results = []
                if _results:  # Not an empty list
                    if isinstance(_results[0], list):  # In the case of nanohedra
                        for result in _results:
                            results.extend(result)
                    else:
                        results.extend(_results)  # append(result)
                pose_directories = results
            else:
                results = _results
            # elif putils.cluster_poses:  # Returns None
            #    pass

            # Update the current state of protocols and exceptions
            pose_directories, additional_exceptions = parse_protocol_results(pose_directories, results)
            exceptions.extend(additional_exceptions)
        #     # Retrieve any program flags necessary for termination
        #     terminate_kwargs.update(**terminate_options.get(protocol_name, {}))
        #
        # terminate(results=results, **terminate_kwargs, exceptions=exceptions)
        # terminate(results=results, exceptions=exceptions)
    # -----------------------------------------------------------------------------------------------------------------
    #  Run a single submodule
    # -----------------------------------------------------------------------------------------------------------------
    else:
        if job.module == 'find_transforms':
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
        # Todo
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
        # # ---------------------------------------------------
        # Todo
        # elif job.module == 'find_asu':
        #     # Fetch the specified protocol
        #     protocol = getattr(protocols, job.module)
        #     if args.multi_processing:
        #         results = utils.mp_map(protocol, pose_directories, processes=job.cores)
        #     else:
        #         for pose_dir in pose_directories:
        #             results.append(protocol(pose_dir))
        #
        #     terminate(results=results)
        # # ---------------------------------------------------
        # Todo
        # elif job.module == 'check_unmodelled_clashes':
        #     # Fetch the specified protocol
        #     protocol = getattr(protocols, job.module)
        #     if args.multi_processing:
        #         results = utils.mp_map(protocol, pose_directories, processes=job.cores)
        #     else:
        #         for pose_dir in pose_directories:
        #             results.append(protocol(pose_dir))
        #
        #     terminate(results=results)
        # # ---------------------------------------------------
        # Todo
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
        elif job.module == flags.cluster_poses:
            protocols.cluster.cluster_poses(pose_directories)
            terminate(output=False)
        # ---------------------------------------------------
        elif job.module == flags.select_poses:
            # Need to initialize pose_directories, design_source to terminate()
            pose_directories = results = protocols.select.poses(pose_directories)
            design_source = job.program_root
            # Write out the chosen poses to a pose.paths file
            terminate(results=results)
        # ---------------------------------------------------
        elif job.module == flags.select_designs:
            # Need to initialize pose_directories, design_source to terminate()
            pose_directories = results = protocols.select.designs(pose_directories)
            design_source = job.program_root
            # Write out the chosen poses to a pose.paths file
            terminate(results=results)
        # ---------------------------------------------------
        elif job.module == flags.select_sequences:
            # Need to initialize pose_directories, design_source to terminate()
            pose_directories = results = protocols.select.sequences(pose_directories)
            design_source = job.program_root
            # Write out the chosen poses to a pose.paths file
            terminate(results=results)
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
                if not job.dataframe:
                    df_glob = sorted(glob(os.path.join(file_dir, 'TrajectoryMetrics.csv')))
                    try:
                        job.dataframe = df_glob[0]
                    except IndexError:
                        raise IndexError(f"There was no --{flags.dataframe} specified and one couldn't be located in "
                                         f'{job.location}. Initialize again with the path to the relevant dataframe')

                df = pd.read_csv(job.dataframe, index_col=0, header=[0])
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
        else:  # Fetch the specified protocol
            protocol = getattr(protocols, flags.format_from_cmdline(job.module))
            if job.development:
                if job.profile:
                    if profile:

                        # Run the profile decorator from memory_profiler
                        # Todo insert into the bottom most decorator slot
                        profile(protocol)(pose_directories[0])
                    else:
                        logger.critical(f"The module memory_profiler isn't installed {profile_error}")
                    exit('Done profiling')

            if args.multi_processing:
                results = utils.mp_map(protocol, pose_directories, processes=job.cores)
            else:
                for pose_dir in pose_directories:
                    results.append(protocol(pose_dir))
    # -----------------------------------------------------------------------------------------------------------------
    #  Finally, run terminate(). This formats output parameters and reports on exceptions
    # -----------------------------------------------------------------------------------------------------------------
    terminate(results=results, exceptions=exceptions)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nRun Ended By KeyboardInterrupt\n')
        exit(2)
