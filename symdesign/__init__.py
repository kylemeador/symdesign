"""Model and design proteins with inherent consideration for local and global symmetry

Modules exported by this package:

- `metrics`: Perform calculations on a protein pose
- `protocols`: Implement a defined set of instructions for a protein pose
- `resources`: Common methods and variable to connect job instructions to protocol implementation during job runtime
- `sequence`: Handle biological sequences as python objects
- `structure`: Handle biological structures as python objects
- `third_party`: External dependencies that are installed by this package
- `utils`: Miscellaneous functions, methods, and tools for all modules
"""
from __future__ import annotations

# import logging
import csv
import logging.config
import os
import random
import shutil
import sys
import traceback
from argparse import Namespace
from collections import defaultdict
from glob import glob
from itertools import count, repeat, product, combinations
from subprocess import list2cmdline
from typing import Any, AnyStr, Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import lazyload

try:
    from memory_profiler import profile
    profile_error = None
except ImportError as profile_error:
    profile = None

import symdesign.utils.path as putils
logging.config.dictConfig(putils.logging_cfg)
logger = logging.getLogger(putils.program_name.lower())

from symdesign import flags, protocols, utils
from symdesign.protocols.pose import PoseJob
from symdesign.resources.job import JobResources, job_resources_factory
from symdesign.resources.query.pdb import retrieve_pdb_entries_by_advanced_query
from symdesign.resources import distribute, ml, sql, structure_db, wrapapi
from symdesign.structure import Entity, Pose
from symdesign.structure.utils import StructureException
import symdesign.utils.guide


def initialize_entities(job: JobResources, uniprot_entities: Iterable[wrapapi.UniProtEntity],
                        metadata: Iterable[sql.ProteinMetadata], batch_commands: bool = False):
    """Handle evolutionary and structural data creation

    Args:
        job: The active JobResources singleton
        uniprot_entities: All UniProtEntity instances which should be checked for evolutionary info
        metadata: The ProteinMetadata instances that are being imported for the first time
        batch_commands: Whether commands should be made for batch submission
    Returns:
        The processed structures
    """

    def check_if_script_and_exit(messages: list[str]):
        if messages:
            # Entity processing commands are needed
            if distribute.is_sbatch_available():
                logger.info(distribute.sbatch_warning)
            else:
                logger.info(distribute.script_warning)

            for message in messages:
                logger.info(message)
            print('\n')
            logger.info(f'After completion of script(s), re-submit your {putils.program_name} command:\n'
                        f'\tpython {" ".join(sys.argv)}')
            sys.exit(1)

    # Set up common Entity resources
    if job.use_evolution:
        profile_search_instructions = \
            job.process_evolutionary_info(uniprot_entities=uniprot_entities, batch_commands=batch_commands)
    else:
        profile_search_instructions = []

    check_if_script_and_exit(profile_search_instructions)

    # Ensure files exist
    for data in metadata:
        if data.model_source is None:
            raise ValueError(
                f"Couldn't find {data}.model_source")
    # Check whether loop modeling or refinement should be performed
    # If so, move the oriented Entity.model_source (should be the ASU) to the respective directory
    if job.init.pre_refined:  # Indicate refine stuff is done
        refine_dir = job.structure_db.refined.location
        putils.make_path(refine_dir)
        for data in metadata:
            dirname, basename = os.path.split(data.model_source)
            if not os.path.exists(os.path.join(refine_dir, basename)):
                try:  # To copy the file to this location
                    shutil.copy(data.model_source, refine_dir)
                except shutil.SameFileError:
                    pass
            # Set True regardless
            data.refined = True
    if job.init.pre_loop_modeled:  # Indicate loop model stuff is done
        full_model_dir = job.structure_db.full_models.location
        putils.make_path(full_model_dir)
        for data in metadata:
            dirname, basename = os.path.split(data.model_source)
            if not os.path.exists(os.path.join(full_model_dir, basename)):
                try:  # To copy the file to this location
                    shutil.copy(data.model_source, full_model_dir)
                except shutil.SameFileError:
                    pass
            # Set True regardless
            data.loop_modeled = True

    # preprocess_instructions, initial_refinement, initial_loop_model = \
    #     job.structure_db.preprocess_structures_for_design(structures, script_out_path=job.sbatch_scripts)
    preprocess_instructions = \
        job.structure_db.preprocess_metadata_for_design(
            metadata, script_out_path=job.sbatch_scripts, batch_commands=batch_commands)

    check_if_script_and_exit(preprocess_instructions)
    # After completion of indicated scripts, the next time command is entered
    # these checks will not raise and the program will proceed

    for data in metadata:
        if data.refined:
            data.model_source = job.structure_db.refined.retrieve_file(data.entity_id)
        elif data.loop_modeled:
            data.model_source = job.structure_db.full_models.retrieve_file(data.entity_id)
        else:  # oriented asu:
            data.model_source = job.structure_db.oriented_asu.retrieve_file(data.entity_id)
        # Ensure the file exists again
        if data.model_source is None:
            raise ValueError(
                f"Couldn't find {data}.model_source")


def initialize_structures(job: JobResources, sym_entry: utils.SymEntry.SymEntry = None, paths: Iterable[AnyStr] = False,
                          pdb_codes: list[str] = None, query_codes: bool = False) \
        -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, ...], list[sql.ProteinMetadata]]]:
    """From provided codes, files, or query directive, load structures into the runtime and orient them in the database

    Args:
        job: The active JobResources singleton
        sym_entry: The SymEntry used to perform the orient protocol on the specified structure identifiers
        paths: The locations on disk to search for structural files
        pdb_codes: The PDB API EntryID, EntityID, or AssemblyID codes to fetch structure information
        query_codes: Whether a PDB API query should be initiated
    Returns:
        The tuple consisting of (
            A map of the Model name to each Entity name in the Model,
            A mapping of the UniprotID's to their ProteinMetadata instance for every Entity loaded
        )
    """
    # Set up variables for the correct parsing of provided file paths
    by_file = False
    if paths:
        by_file = True
        if '.pdb' in paths:
            pdb_filepaths = [paths]
        else:
            extension = '.pdb*'
            pdb_filepaths = utils.get_directory_file_paths(paths, extension=extension)
            if not pdb_filepaths:
                logger.warning(f'Found no {extension} files at {paths}')
        # Set filepaths to structure_names, reformat the file paths to the file_name for structure_names
        structure_names = pdb_filepaths
    elif pdb_codes:
        # Make all names lowercase
        structure_names = list(map(str.lower, pdb_codes))
    elif query_codes:
        save_query = utils.query.validate_input_return_response_value(
            'Do you want to save your PDB query to a local file?', {'y': True, 'n': False})
        print(f'\nStarting PDB query\n')
        structure_names = retrieve_pdb_entries_by_advanced_query(save=save_query, entity=True)
        # Make all names lowercase
        structure_names = list(map(str.lower, structure_names))
    else:
        structure_names = []

    # Select entities, orient them, then load ProteinMetadata for further processing
    return job.structure_db.orient_structures(structure_names, sym_entry=sym_entry, by_file=by_file)


def load_poses_from_structure_and_entity_pairs(job: JobResources,
                                               structure_id_to_entity_ids: dict[str, Iterable[str]] | list[Pose],
                                               all_protein_metadata: list[sql.ProteinMetadata]) -> list[Pose]:
    """From specified identifiers, load the corresponding Pose and return all instances

    Args:
        job:
        structure_id_to_entity_ids:
        all_protein_metadata:
    Returns:
        The loaded Pose instances. No symmetry properties are attached
    """
    structures = []
    if not structure_id_to_entity_ids:
        pass
    elif isinstance(structure_id_to_entity_ids, dict):
        structures = []
        for structure_id, entity_ids in structure_id_to_entity_ids.items():
            entities = []
            if entity_ids:
                for entity_id in entity_ids:
                    # # This data may not be the one that is initialized, grab the correct one
                    # data = all_uniprot_id_to_prot_data[uniprot_ids]
                    for data in all_protein_metadata:
                        if data.entity_id == entity_id:
                            break
                    else:
                        print([data.entity_id for data in all_protein_metadata])
                        raise utils.SymDesignException(
                            f"Indexing the correct entity_id for {entity_id} has failed")

                    if not data.model_source:
                        raise RuntimeError(
                            f"There was an issue with locating the .model_source during processing the "
                            f"Entity {data.entity_id} for input. {putils.report_issue}"
                        )
                    entity = Entity.from_file(data.model_source, name=data.entity_id, metadata=data)
                    entities.append(entity)

                # For Pose constructor, don't include symmetry.
                # Symmetry is initialized by protocols
                pose = Pose.from_entities(entities, name=structure_id)
            else:  # Just load the whole model based off of the name
                # This is useful in the case when CRYST record is used in crystalline symmetries
                whole_model_file = os.path.join(
                    os.path.dirname(job.structure_db.oriented.location), f'{structure_id}.pdb*')
                matching_files = glob(whole_model_file)
                if matching_files:
                    if len(matching_files) > 1:
                        logger.warning(f"{len(matching_files)} matching files for {structure_id} at "
                                       f"'{whole_model_file}'. Choosing the first")
                    pose = Pose.from_file(matching_files[0], name=structure_id)
                else:
                    logger.warning(f"No matching files for {structure_id} at '{whole_model_file}'")
                    continue

            # Set .metadata attribute to carry through protocol
            for entity in pose.entities:
                for data in all_protein_metadata:
                    if data.entity_id == entity.name:
                        break
                else:
                    print([data.entity_id for data in all_protein_metadata])
                    raise utils.SymDesignException(
                        f"Indexing the correct entity_id for {entity.name} has failed")

                entity.metadata = data
            structures.append(pose)
    else:  # These are already processed structure instances
        structures = structure_id_to_entity_ids

    return structures


def parse_results_for_exceptions(pose_jobs: list[PoseJob], results: list[Any], **kwargs) \
        -> list[tuple[PoseJob, Exception]] | list:
    """Filter out any exceptions from results

    Args:
        pose_jobs: The PoseJob instances that attempted work
        results: The returned values from a job
    Returns:
        Tuple of passing PoseDirectories and Exceptions
    """
    if results:
        exception_indices = [idx for idx, result_ in enumerate(results) if isinstance(result_, BaseException)]
        if pose_jobs:
            if len(pose_jobs) == len(results):
                return [(pose_jobs.pop(idx), results.pop(idx)) for idx in reversed(exception_indices)]
            else:
                raise ValueError(
                    f"The number of PoseJob instances {len(pose_jobs)} != {len(results)}, the number of job results."
                )
        else:
            return [(None, results.pop(idx)) for idx in reversed(exception_indices)]
            # return list(zip(repeat(None), results))
    else:
        return []


def destruct_factories():
    """Remove data from existing singletons to destruct this session"""
    # factories = [JobResourcesFactory, ProteinMPNNFactory, APIDatabaseFactory, SymEntryFactory,
    #              FragmentDatabaseFactory, EulerLookupFactory, StructureDatabaseFactory]
    from symdesign.structure.fragment.db import fragment_factory, euler_factory
    factories = [job_resources_factory, ml.proteinmpnn_factory, wrapapi.api_database_factory, utils.SymEntry.symmetry_factory,
                 fragment_factory, euler_factory, structure_db.structure_database_factory]
    for factory in factories:
        factory.destruct()


def main():
    """Run the program"""
    # -----------------------------------------------------------------------------------------------------------------
    #  Initialize local functions
    # -----------------------------------------------------------------------------------------------------------------
    def fetch_pose_jobs_from_database(args: Namespace, in_root: bool = False) -> list[PoseJob]:
        # Use sqlalchemy database selection to find the requested work
        pose_identifiers = []
        designs = []
        directives = []
        if args.poses or args.specification_file:
            # These poses are already included in the "program state"
            if not in_root or not args.directory:
                directory_required = f'A {flags.format_args(flags.directory_args)} must be provided when using '
                not_in_current = f" if the target {putils.program_name} isn't in, or the current working directory"
                if args.poses:
                    raise utils.InputError(
                        f'{directory_required}{flags.format_args(flags.poses_args)}{not_in_current}')
                elif args.specification_file:
                    raise utils.InputError(
                        f'{directory_required}{flags.format_args(flags.specification_file_args)}{not_in_current}')
            if args.poses:
                # Extend pose_identifiers with each parsed pose_identifiers set
                job.location = args.poses
                for pose_file in args.poses:
                    pose_identifiers.extend(utils.PoseSpecification(pose_file).pose_identifiers)
            else:
                job.location = args.specification_file
                for specification_file in args.specification_file:
                    # Returns list of _designs and _directives in addition to pose_identifiers
                    _pose_identifiers, _designs, _directives = \
                        zip(*utils.PoseSpecification(specification_file).get_directives())
                    pose_identifiers.extend(_pose_identifiers)
                    designs.extend(_designs)
                    directives.extend(_directives)

        else:  # args.project or args.single:
            paths, job.location = utils.collect_designs(projects=args.project, singles=args.single)
            #                                             directory = symdesign_directory,
            if paths:  # There are files present
                for path in paths:
                    name, project, *_ = reversed(path.split(os.sep))
                    pose_identifiers.append(f'{project}{os.sep}{name}')
            # elif args.project:
            #     job.location = args.project
            #     projects = [os.path.basename(project) for project in args.project]
            #     fetch_jobs_stmt = select(PoseJob).where(PoseJob.project.in_(projects))
            # else:  # args.single:
            #     job.location = args.project
            #     singles = [os.path.basename(single) for single in args.project]
            #     for single in singles:
            #         name, project, *_ = reversed(single.split(os.sep))
            #         pose_identifiers.append(f'{project}{os.sep}{name}')
        if not pose_identifiers:
            raise utils.InputError(
                f"No pose identifiers found from input location '{job.location}'")
        else:
            pose_identifiers = job.get_range_slice(pose_identifiers)
            if not pose_identifiers:
                raise utils.InputError(
                    f"No pose identifiers found with {flags.format_args(flags.range_args)}={job.range}'")

        # Fetch identified. No writes
        with job.db.session(expire_on_commit=False) as session:
            if job.module in flags.select_modules:
                pose_job_stmt = select(PoseJob).options(
                    lazyload(PoseJob.entity_data),
                    lazyload(PoseJob.metrics))
            else:  # Load all attributes
                pose_job_stmt = select(PoseJob)
            try:  # To convert the identifier to an integer
                int(pose_identifiers[0])
            except ValueError:  # Can't convert to integer, identifiers_are_database_id = False
                fetch_jobs_stmt = pose_job_stmt.where(PoseJob.pose_identifier.in_(pose_identifiers))
            else:
                fetch_jobs_stmt = pose_job_stmt.where(PoseJob.id.in_(pose_identifiers))

            pose_jobs = session.scalars(fetch_jobs_stmt).all()

            # Check all jobs that were checked out against those that were requested
            pose_identifier_to_pose_job_map = {pose_job.pose_identifier: pose_job for pose_job in pose_jobs}
            missing_pose_identifiers = set(pose_identifier_to_pose_job_map.keys()).difference(pose_identifiers)
            if missing_pose_identifiers:
                logger.warning(
                    "Couldn't find the following identifiers:\n\t%s\nRemoving them from the Job"
                    % '\n'.join(missing_pose_identifiers))
                remove_missing_identifiers = [pose_identifiers.index(identifier)
                                              for identifier in missing_pose_identifiers]
                for index in sorted(remove_missing_identifiers, reverse=True):
                    pose_identifiers.pop(index)
            # else:
            #     remove_missing_identifiers = []

            if args.specification_file:
                # designs and directives should always be the same length
                # if remove_missing_identifiers:
                #     raise utils.InputError(
                #         f"Can't set up job from {flags.format_args(flags.specification_file_args)} with missing "
                #         f"pose identifier(s)")
                # if designs:
                designs = job.get_range_slice(designs)
                # if directives:
                directives = job.get_range_slice(directives)
                # Set up PoseJob with the specific designs and any directives
                for pose_identifier, _designs, _directives in zip(pose_identifiers, designs, directives):
                    # Since the PoseJob were loaded from the database, the order of inputs needs to be used
                    pose_identifier_to_pose_job_map[pose_identifier].use_specific_designs(_designs, _directives)

        return pose_jobs

    def terminate(results: list[Any] | dict = None, output: bool = True, **kwargs):
        """Format designs passing output parameters and report program exceptions

        Args:
            results: The returned results from the module run. By convention contains results and exceptions
            output: Whether the module used requires a file to be output
        """
        nonlocal exceptions
        output_analysis = True
        if results:
            exceptions += parse_results_for_exceptions(pose_jobs, results)
            successful_pose_jobs = pose_jobs
        else:
            successful_pose_jobs = []

        # Format the output file(s) depending on specified name and module type
        job_paths = job.job_paths
        # Only using exit_code 0 if the program completed and any exceptions raised by program were caught and handled
        exit_code = 0
        if exceptions:
            print('\n')
            logger.warning(f'Exceptions were thrown for {len(exceptions)} jobs. '
                           f'Check their individual .log files for more details\n\t%s'
                           % '\n\t'.join(f'{pose_job}: {error_}' for pose_job, error_ in exceptions))
            print('\n')
            exceptions_file = os.path.join(job_paths, putils.default_execption_file.format(*job.default_output_tuple))
            with open(exceptions_file, 'w') as f_out:
                f_out.write('%s\n' % '\n'.join(str(pj) for pj, error_ in exceptions))
            logger.info(f"The file '{exceptions_file}' contains the pose identifier of every pose that failed "
                        'checks/filters for this job')

        if output and successful_pose_jobs:
            poses_file = None
            if job.output_file:
                # if job.module not in [flags.analysis, flags.cluster_poses]:
                #     poses_file = job.output_file
                if job.module == flags.analysis:
                    if len(job.output_file.split(os.sep)) <= 1:
                        # The path isn't an absolute or relative path, so prepend the job.all_scores location
                        job.output_file = os.path.join(job.all_scores, job.output_file)
                    if not job.output_file.endswith('.csv'):
                        job.output_file = f'{job.output_file}.csv'
                    if not job.output:  # No output is specified
                        output_analysis = False
                else:
                    # Set the poses_file to the provided job.output_file
                    poses_file = job.output_file
            else:
                # For certain modules, use the default file type
                if job.module == flags.analysis:
                    job.output_file = putils.default_analysis_file.format(utils.starttime, job.input_source)
                else:  # We don't have a default output specified
                    pass

            # Make single file with names of each directory where poses can be found
            if output_analysis:
                if poses_file is None:  # Make a default file name
                    putils.make_path(job_paths)
                    poses_file = \
                        os.path.join(job_paths, putils.default_path_file.format(*job.default_output_tuple))

                with open(poses_file, 'w') as f_out:
                    f_out.write('%s\n' % '\n'.join(str(pj) for pj in successful_pose_jobs))
                logger.info(f'The file "{poses_file}" contains the pose identifier of every pose that passed checks'
                            f'/filters for this job. Utilize this file to input these poses in future '
                            f'{putils.program_name} commands such as:'
                            f'\n\t{putils.program_command} MODULE --{flags.poses} {poses_file} ...')

            # Output any additional files for the module
            if job.module in [flags.select_designs, flags.select_sequences]:
                designs_file = \
                    os.path.join(job_paths, putils.default_specification_file.format(*job.default_output_tuple))
                try:
                    with open(designs_file, 'w') as f_out:
                        f_out.write('%s\n' % '\n'.join(f'{pj}, {design.name}' for pj in successful_pose_jobs
                                                       for design in pj.current_designs))
                    logger.info(f'The file "{designs_file}" contains the pose identifier and design identifier, of '
                                f'every design selected by this job. Utilize this file to input these designs in '
                                f'future'
                                f' {putils.program_name} commands such as:\n\t{putils.program_command} MODULE '
                                f'{flags.format_args(flags.specification_file_args)} {designs_file} ...')
                except AttributeError:  # The pose_job variable is a str from select-designs
                    pass

            # if job.module == flags.analysis:
            #     # Save Design DataFrame
            #     design_df = pd.DataFrame([result_ for result_ in results if not isinstance(result_, BaseException)])
            #     if job.output:  # Create a new file
            #         design_df.to_csv(args.output_file)
            #     else:
            #         # This is the mechanism set up to append to an existing file. Check if we should add a header
            #         header = False if os.path.exists(args.output_file) else True
            #         design_df.to_csv(args.output_file, mode='a', header=header)
            #
            #     logger.info(f'Analysis of all poses written to {args.output_file}')
            #     if args.save:
            #         logger.info(f'Analysis of all Trajectories and Residues written to {job.all_scores}')

            # Set up sbatch scripts for processed Poses
            def _get_module_specific_protocol(module: str) -> str:
                if module == flags.design:
                    if job.design.interface:
                        if job.design.method == putils.consensus:
                            scale = flags.refine
                        elif job.design.method == putils.proteinmpnn:
                            scale = putils.proteinmpnn
                        else:  # if job.design.method == putils.rosetta_str:
                            scale = putils.scout if job.design.scout \
                                else (putils.hbnet_design_profile if job.design.hbnet
                                      else (putils.structure_background if job.design.structure_background
                                            else flags.interface_design))
                    else:
                        scale = flags.design
                        # custom_script: os.path.splitext(os.path.basename(getattr(args, 'script', 'c/custom')))[0]
                else:
                    scale = module

                return scale

            if job.distribute_work:
                scripts_to_distribute = [pose_job.current_script for pose_job in successful_pose_jobs]
                distribute.check_scripts_exist(directives=scripts_to_distribute)
                distribute.commands(scripts_to_distribute, name='_'.join(job.default_output_tuple),
                                    protocol=_get_module_specific_protocol(job.module),
                                    out_path=job.sbatch_scripts, commands_out_path=job.job_paths)

                # if job.module == flags.design and job.initial_refinement:
                #     # We should refine before design
                #     refine_file = utils.write_commands([os.path.join(pose_job.scripts_path, f'{flags.refine}.sh')
                #                                         for pose_job in successful_pose_jobs], out_path=job_paths,
                #                                        name='_'.join((utils.starttime, flags.refine, design_source)))
                #     script_refine_file = distribute.distribute(refine_file, flags.refine, out_path=job.sbatch_scripts)
                #     logger.info(f'Once you are satisfied, enter the following to distribute:\n\t{shell} '
                #                 f'{script_refine_file}\nTHEN:\n\t{shell} {script_file}')
                # else:
        elif len(successful_pose_jobs) == 0:
            exit_code = 0

        # # Test for the size of each of the PoseJob instances
        # if pose_jobs:
        #     print('Average_design_directory_size equals %f' %
        #           (float(psutil.virtual_memory().used) / len(pose_jobs)))

        sys.exit(exit_code)

    # -----------------------------------------------------------------------------------------------------------------
    #  Process optional program flags
    # -----------------------------------------------------------------------------------------------------------------
    # Ensure module specific arguments are collected and argument help is printed in full
    args, additional_args = flags.guide_parser.parse_known_args()
    # -----------------------------------------------------------------------------------------------------------------
    #  Display the program guide if requested
    # -----------------------------------------------------------------------------------------------------------------
    if args.module:
        if args.guide:
            try:
                print(getattr(utils.guide, args.module.replace('-', '_')))
            except AttributeError:
                print(f'There is no guide created for {args.module} yet. Try --help instead of --guide')
            sys.exit()
            # Known to be missing
            # [custom_script, check_clashes, residue_selector, visualize]
            #     print('Usage: %s -r %s -- [-d %s, -df %s, -f %s] visualize --range 0-10'
            #           % (putils.ex_path('pymol'), putils.program_command.replace('python ', ''),
            #              putils.ex_path('pose_directory'), SDUtils.ex_path('DataFrame.csv'),
            #              putils.ex_path('design.paths')))
    elif args.setup:
        utils.guide.setup_instructions()
        sys.exit()
    elif args.help:
        pass  # Let the entire_parser handle their formatting
    else:  # Print the full program README.md and exit
        utils.guide.print_guide()
        sys.exit()

    #  ---------------------------------------------------
    #  elif args.flags:  # Todo
    #      if args.template:
    #          flags.query_user_for_flags(template=True)
    #      else:
    #          flags.query_user_for_flags(mode=args.flags_module)
    #  ---------------------------------------------------
    #  elif args.residue_selector:  # Todo
    #      def generate_sequence_template(file):
    #          model = Pose.from_file(file)
    #          sequence = SeqRecord(Seq(model.sequence), 'Protein'), id=model.name)
    #          sequence_mask = copy.copy(sequence)
    #          sequence_mask.id = 'residue_selector'
    #          sequences = [sequence, sequence_mask]
    #          raise NotImplementedError('This write_fasta call needs to have its keyword arguments refactored')
    #          return write_fasta(sequences, file_name=f'{model.name}_residue_selector_sequence')
    #
    #      if not args.single:
    #          raise utils.DesignError('You must pass a single pdb file to %s. Ex:\n\t%s --single my_pdb_file.pdb '
    #                                  'residue_selector' % (putils.program_name, putils.program_command))
    #      fasta_file = generate_sequence_template(args.single)
    #      logger.info('The residue_selector template was written to %s. Please edit this file so that the '
    #                  'residue_selector can be generated for protein design. Selection should be formatted as a "*" '
    #                  'replaces all sequence of interest to be considered in design, while a Mask should be formatted as '
    #                  'a "-". Ex:\n>pdb_template_sequence\nMAGHALKMLV...\n>residue_selector\nMAGH**KMLV\n\nor'
    #                  '\n>pdb_template_sequence\nMAGHALKMLV...\n>design_mask\nMAGH----LV\n'
    #                  % fasta_file)
    # -----------------------------------------------------------------------------------------------------------------
    #  Initialize program with provided flags and arguments
    # -----------------------------------------------------------------------------------------------------------------
    # Parse arguments for the actual runtime which accounts for differential argument ordering from standard argparse
    args, additional_args = flags.module_parser.parse_known_args()

    # Checking for query before additional_args error as query has additional_args
    if args.module == flags.symmetry:
        # if not args.query:
        #     args.query = 'all-entries'
        if args.query not in utils.SymEntry.query_mode_args:
            print(f"Error: Please specify the query mode after '{flags.query.long}' to proceed")
            sys.exit(1)
        utils.SymEntry.query(args.query, *additional_args, nanohedra=args.nanohedra)
        sys.exit()

    def handle_atypical_inputs(modules: Iterable[str]) -> bool:
        """Add a dummy input for argparse to happily continue with required args

        Args:
            modules: The iterable of moduls to search
        Returns:
            True if the module has a fake input
        """
        atypical_input_modules = [flags.align_helices, flags.initialize_building_blocks, flags.nanohedra]
        for module in modules:
            if module in atypical_input_modules:
                additional_args.extend(['--file', 'dummy'])
                return True
        return False

    remove_dummy: bool = handle_atypical_inputs([args.module])
    if args.module == flags.all_flags:
        # Overwrite the specified arguments to just print the program help
        sys.argv = [putils.program_exe, '--help']
    elif args.module == flags.protocol:
        remove_dummy = handle_atypical_inputs(args.modules)

        # Parse all options for every module provided
        all_args = [args]
        for idx, module in enumerate(args.modules):
            # additional_args = [module] + additional_args
            args, additional_args = flags.module_parser.parse_known_args(args=[module] + additional_args)
            all_args.append(args)

        # Invert all the arguments to ensure those that were specified first are set and not overwritten by default
        for _args in reversed(all_args):
            _args_ = vars(args)
            _args_.update(**vars(_args))
            args = Namespace(**_args_)
            # args = Namespace(**vars(args), **vars(_args))

        # Set the module to flags.protocol again after parsing
        args.module = flags.protocol

    # Check the provided flags against the entire_parser to print any help
    _args, _additional_args = flags.entire_parser.parse_known_args()
    # Parse the provided flags
    for argparser in flags.additional_parsers:
        args, additional_args = argparser.parse_known_args(args=additional_args, namespace=args)

    if additional_args:
        print(f"\nFound flag(s) that aren't recognized with the requested job/module(s): {', '.join(additional_args)}\n"
              'Please correct/remove them and resubmit your command. Try adding -h/--help for available formatting\n'
              f"If you want to view all {putils.program_name} flags, "
              f"replace the MODULE '{args.module}' with '{flags.all_flags}'\n")
        sys.exit(1)

    if remove_dummy:  # Remove the dummy input
        del args.file
    # -----------------------------------------------------------------------------------------------------------------
    #  Find base output directory and check for proper set up of program i/o
    # -----------------------------------------------------------------------------------------------------------------
    root_directory = utils.get_program_root_directory(
        args.directory or (args.project or args.single or [None])[0])  # Get the first in the sequence
    root_in_cwd = utils.get_program_root_directory(os.getcwd())
    if root_in_cwd:
        root_in_current = True
        if root_directory is None:  # No directory from the input. Is it in, or the current working directory?
            root_directory = root_in_cwd
    else:
        root_in_current = False

    project_name = None
    """Used to set a leading string on all new PoseJob paths"""
    if root_directory is None:  # No program output directory from the input
        # By default, assume new input and make in the current directory
        new_program_output = True
        root_directory = os.path.abspath(os.path.join(os.getcwd(), putils.program_output))
        # Check if there is a file and see if we can solve there
        file_sources = ['file', 'poses', 'specification_file']  # 'poses' 'pose_file'
        for file_source in file_sources:
            file = getattr(args, file_source, None)
            if file:  # See if the specified file source contains compatible paths
                # Must index the first file with [0] as it can be a list...
                file = file[0]
                with open(file, 'r') as f:
                    line = f.readline()
                    basename, extension = os.path.splitext(line)
                    if extension == '':  # Provided as directory/pose_name (pose_identifier) from putils.program_output
                        file_directory = utils.get_program_root_directory(line)
                        if file_directory is not None:
                            root_directory = file_directory
                            new_program_output = False
                    else:
                        # Set file basename as "project_name" in case poses are being integrated for the first time
                        # In this case, pose names might be the same, so we take the basename of the first file path as
                        # the name to discriminate between separate files
                        project_name = os.path.splitext(os.path.basename(file))[0]
                break
        # Ensure the base directory is made if it is indeed new and in os.getcwd()
        putils.make_path(root_directory)
    else:
        new_program_output = False
    # -----------------------------------------------------------------------------------------------------------------
    #  Process JobResources which holds shared program objects and command-line arguments
    # -----------------------------------------------------------------------------------------------------------------
    job = job_resources_factory.get(program_root=root_directory, arguments=args, initial=new_program_output)
    if job.project_name:  # One was provided using flags (arguments=args) ^
        project_name = job.project_name
    # else:
    #     project_name = fallback_project_name
    if args.module == flags.protocol:
        # Set the first module as the program module to allow initialization
        job.module = args.modules[0]
    # -----------------------------------------------------------------------------------------------------------------
    #  Start Logging
    # -----------------------------------------------------------------------------------------------------------------
    if args.log_level == logging.DEBUG:  # Debugging
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
        # __main__ logs to stream with level info and propagates to main log
        # logger = utils.start_log(name=putils.program_name, propagate=True)
        # All Designs will log to specific file with level info unless skip_logging is passed
    # -----------------------------------------------------------------------------------------------------------------
    #  Process job information which is necessary for processing and i/o
    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------
    #  Check for tool request
    # ---------------------------------------------------
    if job.module in flags.available_tools:
        if job.module == flags.multicistronic:
            protocols.create_mulitcistronic_sequences(args)
        # elif job.module == flags.distribute:
        #     # formatted_command = job.command.replace('`', '-')
        #     formatted_command = sys.argv[1:]
        #     print(formatted_command)
        #     n_cmds_index = formatted_command.index(f'--{flags.number_of_commands}')
        #     formatted_command.pop(n_cmds_index)
        #     formatted_command.pop(n_cmds_index + 1)
        #     of_index = formatted_command.index(f'--{flags.output_file}')
        #     formatted_command.pop(of_index)
        #     formatted_command.pop(of_index + 1)
        #     print(formatted_command)
        #     with open(job.output_file, 'w') as f:
        #         for i in range(job.number_of_commands):
        #             f.write(f'{formatted_command} --range {100*i / job.number_of_commands:.4f}'
        #                     f'-{100 * (1+i) / job.number_of_commands:.4f}\n')
        #
        #     logger.info(f"Distribution file written to '{job.output_file}'")
        elif job.module == flags.update_db:
            with job.db.session() as session:
                design_stmt = select(sql.DesignData).where(sql.DesignProtocol.file.is_not(None)) \
                    .where(sql.DesignProtocol.design_id == sql.DesignData.id)

                design_data = session.scalars(design_stmt).unique().all()
                for data in design_data:
                    for protocol in data.protocols:
                        if protocol.protocol == 'thread':
                            data.structure_path = protocol.file
                        # else:
                        #     print(protocol.protocol)

                counter = count()
                for data in design_data:
                    if data.structure_path is None:
                        logger.error(f'The design {data} has no .structure_path')
                    else:
                        next(counter)
                logger.info(f'Found {next(counter)} updated designs')
                session.commit()
        else:  # if job.module in decoy_modules:
            pass
        # Shut down, this is just a tool
        sys.exit()

    # Set up module specific arguments
    # Todo we should run this check before every module used as in the case of universal protocols
    #  See if it can be detached here and made into function in main() scope
    select_from_directory = False
    # if job.module in (flags.cluster_poses,) + flags.select_modules:
    #     # # Analysis types can be run from nanohedra_output, so ensure that we don't construct new
    #     # job.construct_pose = False
    #     # if job.module == flags.analysis:
    #     #     # Ensure analysis write directory exists
    #     #     putils.make_path(job.all_scores)
    #     # # if job.module == flags.select_designs:  # Alias to module select_sequences with --skip-sequence-generation
    #     # #     job.module = flags.select_sequences
    #     # #     job.skip_sequence_generation = True
    if job.module in flags.select_modules:
        # Set selection based on full database when no PoseJob are specified through other flags
        # if utils.is_program_base(symdesign_directory): <- hypothetical test for correct program_root
        base_symdesign_dir = utils.get_program_root_directory(args.directory)
        if base_symdesign_dir == args.directory:
            select_from_directory = True
    # elif job.module in [flags.select_designs, flags.select_sequences] \
    #         and job.select_number == sys.maxsize and not args.total:
    #     # Change default number to a single sequence/pose when not doing a total selection
    #     job.select_number = 1
    # elif job.module in flags.cluster_poses and job.number == sys.maxsize:
    #     # Change default number to a single pose
    #     job.number = 1
    elif job.module == flags.nanohedra:
        if not job.sym_entry:
            raise utils.InputError(
                f'When running {flags.nanohedra}, the argument {flags.format_args(flags.sym_entry_args)} is required')
    elif job.module == flags.generate_fragments:  # Ensure we write fragments out
        job.output_fragments = True
    else:
        pass
    # -----------------------------------------------------------------------------------------------------------------
    #  Report options and Set up Databases
    # -----------------------------------------------------------------------------------------------------------------
    reported_args = job.report_specified_arguments(args)
    logger.info('Starting with options:\n\t%s' % '\n\t'.join(utils.pretty_format_table(reported_args.items())))

    logger.info(f"Using resources in Database located at '{job.data}'")
    uses_fragments = [flags.nanohedra, flags.generate_fragments, flags.design, flags.analysis]
    if job.module in uses_fragments:
        if job.module == flags.generate_fragments:
            from symdesign.structure.fragment.db import fragment_factory
            job.fragment_db = fragment_factory(source=job.fragment_source)
            if job.fragment_db.source == putils.biological_interfaces:
                logger.info(f'The FragmentDatabase {job.fragment_db.source} has only been created with '
                            'biological homo-oligomers. Understand the caveats of using fragment information at '
                            'non-interface positions')
    # -----------------------------------------------------------------------------------------------------------------
    #  Initialize the db.session, set into job namespace
    # -----------------------------------------------------------------------------------------------------------------
    results = []
    exceptions = []
    # -----------------------------------------------------------------------------------------------------------------
    #  Grab all Poses (PoseJob instance) from either database, directory, project, single, or file
    # -----------------------------------------------------------------------------------------------------------------
    """pose_jobs hold jobs with specific Structure that constitue a pose
    list[PoseJob] for an establishes pose
    list[tuple[StructureBase, StructureBase]] for a nanohedra docking job
    """
    pose_jobs: list[PoseJob] | list[tuple[Any, Any]] = []
    logger.info(f'Setting up input for {job.module}')
    if job.module == flags.initialize_building_blocks:
        logger.info(f'Ensuring provided building blocks are oriented')
        # Create a SymEntry for just group1
        sym_entry = utils.SymEntry.parse_symmetry_to_sym_entry(symmetry=job.sym_entry.group1)
        _, possibly_new_uniprot_to_prot_metadata = \
            initialize_structures(job, sym_entry=sym_entry, paths=job.component1,
                                  pdb_codes=job.pdb_codes, query_codes=job.query_codes)

        # Make a copy of the new ProteinMetadata if they were already loaded without a .model_source attribute
        possibly_new_uniprot_to_prot_metadata_copy = possibly_new_uniprot_to_prot_metadata.copy()

        # Write new data to the database
        with job.db.session(expire_on_commit=False) as session:
            all_uniprot_id_to_prot_data = sql.initialize_metadata(session, possibly_new_uniprot_to_prot_metadata)
            # Finalize additions to the database
            session.commit()

            # Get all uniprot_entities, and fix ProteinMetadata that is already loaded
            uniprot_entities = []
            all_protein_metadata = []
            for protein_metadata in all_uniprot_id_to_prot_data.values():
                all_protein_metadata.extend(protein_metadata)
                for data in protein_metadata:
                    if data.model_source is None:
                        logger.info(f'{data}.model_source is None')
                        for uniprot_ids, load_protein_metadata in possibly_new_uniprot_to_prot_metadata_copy.items():
                            for load_data in load_protein_metadata:
                                if data.entity_id == load_data.entity_id:
                                    data.model_source = load_data.model_source
                                    logger.info(
                                        f'Set existing {data}.model_source to new {load_data}.model_source')
                    uniprot_entities.extend(data.uniprot_entities)

            if job.update_metadata:
                for attribute, value in job.update_metadata.items():
                    try:
                        getattr(sql.ProteinMetadata, attribute)
                    except AttributeError:
                        raise utils.InputError(
                            f"Couldn't set the {sql.ProteinMetadata.__name__} attribute '.{attribute}' as it doesn't "
                            f"exist")
                    for uniprot_id, protein_metadata in all_uniprot_id_to_prot_data.items():
                        for data in protein_metadata:
                            setattr(data, attribute, value)
                session.commit()
            else:
                # Set up evolution and structures. All attributes will be reflected in ProteinMetadata
                session.add_all(uniprot_entities)
                initialize_entities(job, uniprot_entities, all_protein_metadata, batch_commands=job.distribute_work)
                session.commit()

        terminate(output=False)
    elif job.module in [flags.align_helices, flags.nanohedra]:

        # Transform input entities to canonical orientation, i.e. "orient" and return their metadata
        if job.module == flags.nanohedra:
            job.sym_entry.log_parameters()
            # Create a SymEntry for just group1
            sym_entry1 = utils.SymEntry.parse_symmetry_to_sym_entry(symmetry=job.sym_entry.group1)
        elif job.module == flags.align_helices:
            sym_entry1 = job.sym_entry
        else:
            raise NotImplementedError(
                f"Can't set up component building blocks for {job.module}")

        grouped_structures_entity_ids: list[list[Pose] | dict[str, tuple[str, ...]]] = []
        possibly_new_uniprot_to_prot_metadata: dict[tuple[str, ...], list[sql.ProteinMetadata]] = defaultdict(list)
        if args.poses or args.specification_file or args.project or args.single:
            pose_jobs = fetch_pose_jobs_from_database(args, in_root=root_in_current)
            poses = []
            for pose_job in pose_jobs:
                pose_job.load_pose()
                poses.append(pose_job.pose)
            grouped_structures_entity_ids.append(poses)
        else:
            logger.info(f'Ensuring provided building blocks are oriented for {job.module}')
            structure_id_to_entity_ids, possibly_new_uni_to_prot_metadata = \
                initialize_structures(job, paths=job.component1, pdb_codes=job.pdb_codes,
                                      query_codes=job.query_codes, sym_entry=sym_entry1)
            possibly_new_uniprot_to_prot_metadata.update(possibly_new_uni_to_prot_metadata)
            grouped_structures_entity_ids.append(structure_id_to_entity_ids)

        if job.module == flags.nanohedra:
            # Create a SymEntry for just group2
            sym_entry2 = utils.SymEntry.parse_symmetry_to_sym_entry(symmetry=job.sym_entry.group2)
        elif job.module == flags.align_helices:
            sym_entry2 = None
        else:
            raise NotImplementedError(
                f"Can't set up component building blocks for {job.module}")

        # See if they are the same input
        if job.component1 != job.component2 or job.pdb_codes != job.pdb_codes2 or job.query_codes2:
            structure_id_to_entity_ids, possibly_new_uni_to_prot_metadata = \
                initialize_structures(job, paths=job.component2, pdb_codes=job.pdb_codes2,
                                      query_codes=job.query_codes2, sym_entry=sym_entry2)
            # Update the dictionary without overwriting prior entries
            for uniprot_ids, protein_metadatas in possibly_new_uni_to_prot_metadata.items():
                possibly_new_uniprot_to_prot_metadata[uniprot_ids].extend(protein_metadatas)

            if structure_id_to_entity_ids:
                single_component_design = False
            else:
                single_component_design = True
        else:  # The entities are the same symmetry, or there is a single component and bad input
            single_component_design = True
            structure_id_to_entity_ids = {}

        grouped_structures_entity_ids.append(structure_id_to_entity_ids)

        # Make a copy of the new ProteinMetadata if they were already loaded without a .model_source attribute
        possibly_new_uniprot_to_prot_metadata_copy = possibly_new_uniprot_to_prot_metadata.copy()

        # Write new data to the database
        with job.db.session(expire_on_commit=False) as session:
            all_uniprot_id_to_prot_data = sql.initialize_metadata(session, possibly_new_uniprot_to_prot_metadata)
            # Finalize additions to the database
            session.commit()

            # Get all uniprot_entities to set up sequence dependent resources and fix already loaded ProteinMetadata
            uniprot_entities = []
            all_protein_metadata = []
            for protein_metadata in all_uniprot_id_to_prot_data.values():
                all_protein_metadata.extend(protein_metadata)
                for data in protein_metadata:
                    if data.model_source is None:
                        logger.info(f'{data}.model_source is None')
                        for uniprot_ids, load_protein_metadata in possibly_new_uniprot_to_prot_metadata_copy.items():
                            for load_data in load_protein_metadata:
                                if data.entity_id == load_data.entity_id:
                                    data.model_source = load_data.model_source
                                    logger.info(
                                        f'Set existing {data}.model_source to new {load_data}.model_source')
                    uniprot_entities.extend(data.uniprot_entities)

            # Set up evolution and structures. All attributes will be reflected in ProteinMetadata
            session.add_all(uniprot_entities)
            initialize_entities(job, uniprot_entities, all_protein_metadata, batch_commands=job.distribute_work)

            #  then usage for docking pairs below...

            # Correct existing ProteinMetadata, now that Entity instances are processed
            structures_grouped_by_component = []
            for structure_id_to_entity_ids in grouped_structures_entity_ids:
                structures = load_poses_from_structure_and_entity_pairs(
                    job, structure_id_to_entity_ids, all_protein_metadata)
                structures_grouped_by_component.append(structures)
            session.commit()

        # Make all possible structure pairs given input entities by finding entities from entity_names
        # Using combinations of directories with .pdb files
        if single_component_design:
            logger.info(f'Treating as single component {job.module}, no additional entities requested')
            all_structures = []
            for structures in structures_grouped_by_component:
                all_structures.extend(structures)
            pose_jobs.extend(combinations(all_structures, 2))
        else:
            pose_jobs = []
            for structures1, structures2 in combinations(structures_grouped_by_component, 2):
                pose_jobs.extend(product(structures1, structures2))

        if job.module == flags.nanohedra:
            job.location = f'NanohedraEntry{job.sym_entry.number}'  # Used for terminate()
            if not pose_jobs:  # No pairs were located
                print('\nNo docking pairs were located from your input. Please ensure that your flags are as intended.'
                      f'{putils.issue_submit_warning}\n')
                sys.exit(1)
            if job.distribute_work:
                # Write all commands to a file and distribute a batched job
                # The new command will be the same, but contain a pair of inputs

                # Format all commands given model pair
                base_cmd = list(putils.program_command_tuple) + job.get_parsed_arguments()
                commands = [base_cmd + [flags.pdb_codes_args[-1], model1.name,
                                        flags.pdb_codes2_args[-1], model2.name]
                            for idx, (model1, model2) in enumerate(pose_jobs)]
                # Write commands
                distribute.commands([list2cmdline(cmd) for cmd in commands], name='_'.join(job.default_output_tuple),
                                    protocol=job.module, out_path=job.sbatch_scripts, commands_out_path=job.job_paths)
                terminate(output=False)
        elif job.module == flags.align_helices:
            job.location = f'AlignHelices'  # Used for terminate()
        else:
            raise NotImplementedError()
    else:  # Load from existing files, usually Structural files in a directory or in the program already
        if job.load_to_db:
            # These are not part of the db, but exist in a program_output
            if args.project or args.single:
                paths, job.location = utils.collect_designs(projects=args.project, singles=args.single)
                #                                             directory=symdesign_directory,
            # elif select_from_directory:
            #     paths, job.location = utils.collect_designs(directory=args.directory)
            else:
                raise utils.InputError(
                    f"Can't --load-to-db without passing {flags.format_args(flags.project_args)} or "
                    f"{flags.format_args(flags.single_args)}")

            pose_jobs = [PoseJob.from_directory(path) for path in job.get_range_slice(paths)]
        elif args.poses or args.specification_file or args.project or args.single:
            pose_jobs = fetch_pose_jobs_from_database(args, in_root=root_in_current)
        elif select_from_directory:
            # Can make an empty pose_jobs when the program_root is args.directory
            job.location = args.directory
        elif args.file or args.directory:
            file_paths, job.location = utils.collect_designs(files=args.file, directory=args.directory)
            pose_jobs = [PoseJob.from_path(path, project=project_name)
                         for path in job.get_range_slice(file_paths)]
        else:  # job.pdb_codes job.query_codes
            # Load PoseJob from pdb codes or a pdb query...
            structure_id_to_entity_ids, possibly_new_uniprot_to_prot_metadata = \
                initialize_structures(job, sym_entry=job.sym_entry,
                                      pdb_codes=job.pdb_codes, query_codes=job.query_codes)
            job.location = list(structure_id_to_entity_ids.keys())

            # Write new data to the database and fetch existing data
            with job.db.session(expire_on_commit=False) as session:
                all_uniprot_id_to_prot_data = sql.initialize_metadata(session, possibly_new_uniprot_to_prot_metadata)
                # Finalize additions to the database
                session.commit()

            all_protein_metadata = []
            for protein_metadata in all_uniprot_id_to_prot_data.values():
                all_protein_metadata.extend(protein_metadata)

            range_structure_id_to_entity_ids = {
                struct_id: structure_id_to_entity_ids[struct_id]
                for struct_id in job.get_range_slice(list(structure_id_to_entity_ids.keys()))}
            poses = load_poses_from_structure_and_entity_pairs(
                job, range_structure_id_to_entity_ids, all_protein_metadata)
            pose_jobs = [PoseJob.from_pose(pose, project=project_name) for pose in poses]

        if job.module not in flags.select_modules:  # select_from_directory:
            if not pose_jobs and not select_from_directory:
                #  For instance, a --project provided with the directory of type --single would die without much
                #  knowledge of why
                raise utils.InputError(
                    f"No {PoseJob.__name__}'s found from input location '{job.location}'")
            """Check to see that proper data/files have been created 
            Data includes:
            - UniProtEntity
            - ProteinMetadata
            Files include:
            - Structure in orient, refined, loop modeled
            - Profile from hhblits
            """

            if job.sym_entry:
                symmetry_map = job.sym_entry.groups
                preprocess_entities_by_symmetry: dict[str, list[Entity]] = \
                    {symmetry: [] for symmetry in job.sym_entry.groups}
            else:
                symmetry_map = repeat('C1')
                preprocess_entities_by_symmetry = {'C1': []}

            # Get api wrapper
            retrieve_stride_info = job.structure_db.stride.retrieve_data
            possibly_new_uniprot_to_prot_metadata: dict[tuple[str, ...], list[sql.ProteinMetadata]] = defaultdict(list)
            # possibly_new_uniprot_to_prot_metadata: dict[tuple[str, ...], sql.ProteinMetadata] = {}
            # Todo
            # existing_uniprot_ids = set()
            # existing_protein_properties_ids = set()
            existing_uniprot_entities = set()
            existing_protein_metadata = set()
            existing_random_ids = []
            remove_pose_jobs = []
            pose_jobs_to_commit = []
            if job.specify_entities:
                print("You can use a map to relate the entity by its position in the pose with it's name. "
                      "Would you like to use a map?")
                use_map = utils.query.boolean_choice()
                if use_map:
                    print('Please provide the name of a .csv file with a map in the form of:'
                          '\n\tPoseName,EntityName1,EntityName2,...')
                    file = input(utils.query.input_string)
                    with open(file) as f:
                        pose_entity_mapping = {row[0]: row[1:] for row in csv.reader(f)}

            pose_job: PoseJob
            warn = True
            for idx, pose_job in enumerate(pose_jobs):
                if pose_job.id is not None:  # PoseJob is initialized
                    # Add each UniProtEntity to existing_uniprot_entities to limit work
                    for data in pose_job.entity_data:
                        existing_protein_metadata.add(data.meta)
                        for uniprot_entity in data.meta.uniprot_entities:
                            existing_uniprot_entities.add(uniprot_entity)
                else:  # Not loaded previously
                    # Todo
                    #  Expand the definition of SymEntry/Entity to include specification of T:{T:{C3}{C3}}{C1} where
                    #  an Entity is composed of multiple Entity (Chain) instances
                    # Need to initialize the local database. Load this model to get required info
                    try:
                        pose_job.load_initial_pose()
                    except utils.InputError as error:
                        logger.error(error)
                        remove_pose_jobs.append(idx)
                        continue

                    if job.specify_entities:
                        # Give the input new EntityID's
                        logger.info(f"Modifying identifiers for the input '{pose_job.name}'")
                        if use_map:
                            this_pose_entities = pose_entity_mapping[pose_job.name]
                            modify_map = False

                        while True:
                            using_names = []
                            for entity_idx, entity in enumerate(pose_job.initial_pose.entities):
                                if use_map and not modify_map:
                                    specified_name = this_pose_entities[entity_idx].lower()
                                    if len(specified_name) == 4:
                                        # Add an entity identifier underscore and assume it is the first
                                        specified_name = f'{specified_name}_1'
                                else:
                                    proceed = False
                                    while not proceed:
                                        specified_name = utils.query.format_input(
                                            f"Which name should be used for {entity.__class__.__name__} with name "
                                            f"'{entity.name}' and chainID '{entity.chain_id}'")
                                        if specified_name == entity.name:
                                            break
                                        # If different, ensure that it is desired
                                        if len(specified_name) != 6:  # 6 is the length for pdb entities i.e. 1abc_1
                                            logger.warning(
                                                f"'{specified_name}' isn't the expected number of characters (6)")
                                        proceed = utils.query.confirm_input_action(
                                            f"The name '{specified_name}' will be used instead of '{entity.name}'")

                                using_names.append(specified_name)

                            if use_map:
                                logger.info(f"Using identifiers '{pose_job.name}':{{{'}{'.join(using_names)}}}")
                                print("If this isn't correct, you can repeat with 'n'."
                                      " Otherwise, press enter, or 'y'")
                                if utils.query.boolean_choice():
                                    break
                                else:
                                    modify_map = True

                        for name, entity in zip(using_names, pose_job.initial_pose.entities):
                            if name != entity.name:
                                entity.name = name
                                # Explicitly clear old metadata
                                entity.clear_api_data()
                                entity.retrieve_api_metadata()
                                if entity._api_data is None:  # Information wasn't found
                                    logger.warning(f"There wasn't any information found from the PDB API for the "
                                                   f"name '{name}")

                    for entity, symmetry in zip(pose_job.initial_pose.entities, symmetry_map):
                        try:
                            ''.join(entity.uniprot_ids)
                        except (TypeError, AttributeError):  # Uniprot_ids is (None,), Unable to retrieve .uniprot_ids
                            if len(entity.name) < wrapapi.uniprot_accession_length:
                                entity.uniprot_ids = (entity.name,)
                            else:  # Make up an accession
                                random_accession = f'Rid{random.randint(0,99999):5d}'
                                while random_accession in existing_random_ids:
                                    random_accession = f'Rid{random.randint(0,99999):5d}'
                                else:
                                    existing_random_ids.append(random_accession)
                                entity.uniprot_ids = (random_accession,)
                        # else:  # .uniprot_ids work. Use as parsed
                        uniprot_ids = entity.uniprot_ids

                        # Check if the tuple of UniProtIDs has already been observed
                        # protein_metadata = possibly_new_uniprot_to_prot_metadata.get(uniprot_ids, None)
                        # if protein_metadata is None:
                        # Process for persistent state
                        protein_metadata = sql.ProteinMetadata(
                            entity_id=entity.name,
                            reference_sequence=entity.reference_sequence,
                            thermophilicity=entity.thermophilicity,
                            # There could be no sym_entry, so fall back on the entity.symmetry
                            symmetry_group=symmetry if symmetry else entity.symmetry
                        )
                        # Try to get the already parsed secondary structure information
                        parsed_secondary_structure = retrieve_stride_info(name=entity.name)
                        if parsed_secondary_structure:
                            # Already have this SS information
                            entity.secondary_structure = parsed_secondary_structure
                        else:
                            # entity = Entity.from_file(data.model_source, name=data.entity_id, metadata=data)
                            entity.calculate_secondary_structure(
                                to_file=job.structure_db.stride.path_to(name=entity.name))
                        protein_metadata.n_terminal_helix = entity.is_termini_helical()
                        protein_metadata.c_terminal_helix = entity.is_termini_helical('c')

                        possibly_new_uniprot_to_prot_metadata[uniprot_ids].append(protein_metadata)
                        preprocess_entities_by_symmetry[symmetry].append(entity)
                        # else:
                        #     # This Entity already found for processing, and we shouldn't have duplicates
                        #     found_metadata = sql.ProteinMetadata(
                        #         entity_id=entity.name,
                        #         reference_sequence=entity.reference_sequence,
                        #         thermophilicity=entity.thermophilicity,
                        #         # There could be no sym_entry, so fall back on the entity.symmetry
                        #         symmetry_group=symmetry if symmetry else entity.symmetry
                        #     )
                        #     logger.debug(f"Found duplicate UniProtID identifier, {uniprot_ids}, for {found_metadata}")
                        #     attrs_of_interest = \
                        #         ['entity_id', 'reference_sequence', 'thermophilicity', 'symmetry_group', 'model_source']
                        #     exist = '\n\t.'.join([f'{attr}={getattr(protein_metadata, attr)}'
                        #                           for attr in attrs_of_interest])
                        #     found = '\n\t.'.join([f'{attr}={getattr(found_metadata, attr)}'
                        #                           for attr in attrs_of_interest])
                        #     logger.debug(f'Existing ProteinMetadata:\n{exist}\nNew ProteinMetadata:{found}\n')

                        # # Create EntityData
                        # # entity_data.append(sql.EntityData(pose=pose_job,
                        # sql.EntityData(pose=pose_job,
                        #                meta=protein_metadata
                        #                )

                    if not pose_job.pose:
                        # The pose was never set, first try to orient it
                        try:
                            pose_job.orient()
                        except (utils.SymDesignException, StructureException):
                            # Some aspect of orient() failed. This is fine if the molecules are already oriented
                            if warn:
                                logger.warning(
                                    f"Couldn't {pose_job.orient.__name__}() {repr(pose_job)}. If the input is passed as"
                                    " an asymmetric unit, ensure that it is oriented in a canonical direction."
                                    f"{putils.see_symmetry_documentation}")
                                warn = False

                    pose_jobs_to_commit.append(pose_job)

            for idx in reversed(remove_pose_jobs):
                pose_jobs.pop(idx)

            if not pose_jobs and not select_from_directory:
                raise utils.InputError(
                    f"No viable {PoseJob.__name__}'s found at location '{job.location}'. See the log for details")

            with job.db.session(expire_on_commit=False) as session:
                # Deal with new data compared to existing entries
                all_uniprot_id_to_prot_data = \
                    sql.initialize_metadata(session, possibly_new_uniprot_to_prot_metadata,
                                            existing_uniprot_entities=existing_uniprot_entities,
                                            existing_protein_metadata=existing_protein_metadata)
                # Finalize additions to the database
                session.commit()

                # # Deal with new data compared to existing entries
                # all_uniprot_id_to_prot_data = \
                #     sql.initialize_metadata(session, possibly_new_uniprot_to_prot_metadata,
                #                             # existing_uniprot_ids=existing_uniprot_ids,
                #                             existing_protein_metadata_ids=existing_protein_properties_ids)

                # Get all uniprot_entities, and fix ProteinMetadata that is already loaded
                uniprot_entities = []
                for protein_metadata in all_uniprot_id_to_prot_data.values():
                    for data in protein_metadata:
                        uniprot_entities.extend(data.uniprot_entities)

                # Set up evolution and structures. All attributes will be reflected in ProteinMetadata
                session.add_all(uniprot_entities)
                initialize_entities(job, uniprot_entities, [],  # all_uniprot_id_to_prot_data.values())
                                    batch_commands=job.distribute_work)

                if pose_jobs_to_commit:
                    # Write new data to the database with correct ProteinMetadata and UniProtEntity entries
                    for pose_job in pose_jobs_to_commit:
                        entity_data = []
                        for entity in pose_job.initial_pose.entities:
                            for protein_metadata in all_uniprot_id_to_prot_data[entity.uniprot_ids]:
                                if protein_metadata.entity_id == entity.entity_id:
                                    entity_data.append(sql.EntityData(meta=protein_metadata))
                                    break
                            else:
                                # There could be an issue here where the same entity.entity_id could be used for
                                # different entity.uniprot_ids and sql.initialize_metadata() would overwrite the first
                                # entity_id upon seeing the second.
                                error_str = f"Couldn't set up the {repr(pose_job)} {repr(entity)} with a " \
                                            f"corresponding {sql.ProteinMetadata.__name__} entry. Found " \
                                            f"{repr(entity)} features: .entity_id={entity.entity_id} " \
                                            f".uniprot_ids={entity.uniprot_ids}"
                                available_entity_ids = \
                                    [data.entity_id for data in all_uniprot_id_to_prot_data[entity.uniprot_ids]]
                                if available_entity_ids:
                                    raise utils.InputError(
                                        f"{error_str} and corresponding {sql.ProteinMetadata.__name__} entries:"
                                        f"{', '.join(available_entity_ids)}")

                                # Otherwise, this is likely a random_accession like format "Rid" that doesn't match an
                                # entity_id already imported
                                _found = False
                                for protein_metadatas in all_uniprot_id_to_prot_data.values():
                                    for protein_metadata in protein_metadatas:
                                        if protein_metadata.entity_id == entity.entity_id:
                                            entity_data.append(sql.EntityData(meta=protein_metadata))
                                            _found = True
                                            break
                                    if _found:
                                        break
                                else:
                                    raise utils.InputError(error_str)
                        pose_job.entity_data = entity_data

                        # Todo this could be useful if optimizing database access with concurrency
                        # # Ensure that the pose has entity_transform information saved to the db
                        # pose_job.load_pose()
                        # for data, transformation in zip(self.entity_data, self.pose.entity_transformations):
                        #     # Make an empty EntityTransform and add transformation data
                        #     data.transform = sql.EntityTransform()
                        #     data.transform.transformation = transformation
                    session.add_all(pose_jobs_to_commit)
                    # When pose_jobs_to_commit already exist, deal with it by getting those already
                    # OR raise a useful error for the user about input
                    try:
                        session.commit()
                    except SQLAlchemyError:
                        # Remove pose_jobs_to_commit from session
                        session.rollback()
                        # Find the actual pose_jobs_to_commit and place in session
                        possible_pose_identifiers = [pose_job.new_pose_identifier for pose_job in pose_jobs_to_commit]
                        fetch_jobs_stmt = select(PoseJob).where(
                            PoseJob.pose_identifier.in_(possible_pose_identifiers))
                        existing_pose_jobs = session.scalars(fetch_jobs_stmt).all()

                        existing_pose_identifiers = [pose_job.pose_identifier for pose_job in existing_pose_jobs]
                        pose_jobs = []
                        for pose_job in pose_jobs_to_commit:
                            if pose_job.new_pose_identifier not in existing_pose_identifiers:
                                pose_jobs.append(pose_job)
                                # session.add(pose_job)

                        session.add_all(pose_jobs)
                        session.commit()
                        pose_jobs += existing_pose_jobs
            # End session
        else:  # This is a select_module
            pass

        if args.multi_processing:  # and not args.skip_master_db:
            logger.debug('Loading Database for multiprocessing fork')
            job.structure_db.load_all_data()
            job.api_db.load_all_data()

        # Todo
        #  f'Found {len(pose_jobs)}...' not accurate with select_from_directory
        logger.info(f"Found {len(pose_jobs)} unique poses from provided input location '{job.location}'")
        if not job.debug and not job.skip_logging and not select_from_directory:
            representative_pose_job = next(iter(pose_jobs))
            if representative_pose_job.log_path:
                logger.info(f'All design specific logs are located in their corresponding directories\n\tEx: '
                            f'{representative_pose_job.log_path}')
    # -----------------------------------------------------------------------------------------------------------------
    #  Set up Job specific details and resources
    # -----------------------------------------------------------------------------------------------------------------
    # Format computational requirements
    if job.distribute_work:
        logger.info('Writing module commands out to file, no modeling will occur until commands are executed')
        distribute_modules = [
            flags.nanohedra, flags.refine, flags.design, flags.interface_metrics, flags.optimize_designs, flags.analysis
        ]  # flags.interface_design,
        if job.module not in distribute_modules:
            logger.warning(f"The module '{job.module}' hasn't been tested for distribution methods")

    if job.multi_processing:
        logger.info(f'Starting multiprocessing using {job.cores} cores')
    # else:
    #     logger.info('Starting processing. To increase processing speed, '
    #                 f'use --{flags.multi_processing} during submission')

    job.calculate_memory_requirements(len(pose_jobs))
    # -----------------------------------------------------------------------------------------------------------------
    #  Perform the specified protocol
    # -----------------------------------------------------------------------------------------------------------------
    if args.module == flags.protocol:  # Use args.module as job.module is set as first in protocol
        # Universal protocol runner
        for idx, protocol_name in enumerate(job.modules, 1):
            logger.info(f'Starting protocol {idx}: {protocol_name}')
            # Update this mechanism with each module
            job.module = protocol_name
            job.load_job_protocol()

            # Fetch the specified protocol with python acceptable naming
            protocol = getattr(protocols, flags.format_from_cmdline(job.module))
            # Figure out how the job should be set up
            if job.module in protocols.config.run_on_pose_job:  # Single poses
                if job.multi_processing:
                    results_ = utils.mp_map(protocol, pose_jobs, processes=job.cores)
                else:
                    results_ = [protocol(pose_job) for pose_job in pose_jobs]
            else:  # Collection of pose_jobs
                results_ = [protocol(pose_job) for pose_job in pose_jobs]

            # Handle any returns that require particular treatment
            if job.module in protocols.config.returns_pose_jobs:
                results = []
                if results_:  # Not an empty list
                    if isinstance(results_[0], list):  # In the case returning collection of pose_jobs, i.e. nanohedra
                        for result in results_:
                            results.extend(result)
                    else:
                        results.extend(results_)
                pose_jobs = results.copy()
            # elif job.module == flags.cluster_poses:  # Returns None
            #    pass
            else:
                results = results_

            # Update the current state of protocols and exceptions
            exceptions.extend(parse_results_for_exceptions(pose_jobs, results))

            # # Retrieve any program flags necessary for termination
            # terminate_kwargs.update(**terminate_options.get(protocol_name, {}))
    # -----------------------------------------------------------------------------------------------------------------
    #  Run a single submodule
    # -----------------------------------------------------------------------------------------------------------------
    else:
        job.load_job_protocol()
        if job.module == 'find_transforms':
            # if args.multi_processing:
            #     results = SDUtils.mp_map(PoseJob.find_transforms, pose_jobs, processes=job.cores)
            # else:
            stacked_transforms = [pose_job.pose_transformation for pose_job in pose_jobs]
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
        #     if job.design.number:
        #         logger.info('Checking for %d files based on --{flags.design_number}' % args.design_number)
        #     if args.stage:
        #         status(pose_jobs, args.stage, number=job.design.number)
        #     else:
        #         for stage in putils.stage_f:
        #             s = status(pose_jobs, stage, number=job.design.number)
        #             if s:
        #                 logger.info('For "%s" stage, default settings should generate %d files'
        #                             % (stage, putils.stage_f[stage]['len']))
        # # ---------------------------------------------------
        # Todo
        # elif job.module == 'check_unmodeled_clashes':
        #     # Fetch the specified protocol
        #     protocol = getattr(protocols, job.module)
        #     if args.multi_processing:
        #         results = utils.mp_map(protocol, pose_jobs, processes=job.cores)
        #     else:
        #         for pose_job in pose_jobs:
        #             results.append(protocol(pose_job))
        #
        #     terminate(results=results)
        # # ---------------------------------------------------
        # Todo
        # elif job.module == 'custom_script':
        #     # Start pose processing and preparation for Rosetta
        #     if args.multi_processing:
        #         zipped_args = zip(pose_jobs, repeat(args.script), repeat(args.file_list),
        #                           repeat(args.native), repeat(job.suffix), repeat(args.score_only),
        #                           repeat(args.variables))
        #         results = utils.mp_starmap(PoseJob.custom_rosetta_script, zipped_args, processes=job.cores)
        #     else:
        #         for pose_job in pose_jobs:
        #             results.append(pose_job.custom_rosetta_script(args.script,
        #                                                           file_list=args.file_list, native=args.native,
        #                                                           suffix=job.suffix, score_only=args.score_only,
        #                                                           variables=args.variables))
        #
        #     terminate(results=results)
        # ---------------------------------------------------
        elif job.module == flags.cluster_poses:
            pose_jobs = results = protocols.cluster.cluster_poses(pose_jobs)
            terminate(output=False)
        # ---------------------------------------------------
        elif job.module == flags.select_poses:
            # Need to initialize pose_jobs to terminate()
            pose_jobs = results = protocols.select.sql_poses(pose_jobs)
            # Write out the chosen poses to a pose.paths file
            terminate(results=results)
        # ---------------------------------------------------
        elif job.module == flags.select_designs:
            # Need to initialize pose_jobs to terminate()
            pose_jobs = results = protocols.select.sql_designs(pose_jobs)
            # Write out the chosen poses to a pose.paths file
            terminate(results=results)
        # ---------------------------------------------------
        elif job.module == flags.select_sequences:
            # Need to initialize pose_jobs to terminate()
            pose_jobs = results = protocols.select.sql_sequences(pose_jobs)
            # Write out the chosen poses to a pose.paths file
            terminate(results=results)
        # ---------------------------------------------------
        elif job.module == 'visualize':
            import visualization.VisualizeUtils as VSUtils
            from pymol import cmd

            # if 'escher' in sys.argv[1]:
            if not args.directory:
                print('A directory with the desired designs must be specified using '
                      f'{flags.format_args(flags.directory_args)}')
                sys.exit(1)

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
                # for pose_job in pose_jobs:
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
                        raise IndexError(
                            f"There was no {flags.dataframe.long} specified and one couldn't be located at the location"
                            f" '{job.location}'. Initialize again with the path to the relevant dataframe")

                df = pd.read_csv(job.dataframe, index_col=0, header=[0])
                print('INDICES:\n %s' % df.index.tolist()[:4])
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
                print(f'No .pdb files found at location "{job.location}"')
                sys.exit(1)

            if job.range:
                start_idx = job.low
            else:
                start_idx = 0

            for idx, file in enumerate(job.get_range_slice(files), start_idx):
                if args.name == 'original':
                    cmd.load(file)
                else:  # if args.name == 'numerical':
                    cmd.load(file, object=idx)

            print('\nTo expand all designs to the proper symmetry, issue:\nPyMOL> expand name=all, symmetry=T'
                  '\nYou should replace "T" with whatever symmetry your design is in\n')
        else:  # Fetch the specified protocol
            protocol = getattr(protocols, flags.format_from_cmdline(job.module))
            if job.development:
                if job.profile_memory:
                    if profile:
                        # Run the profile decorator from memory_profiler
                        # Todo insert into the bottom most decorator slot
                        profile(protocol)(pose_jobs[0])
                        print('Done profiling')
                    else:
                        logger.critical(f"The module 'memory_profiler' isn't installed {profile_error}")
                    sys.exit()

            if args.multi_processing:
                results = utils.mp_map(protocol, pose_jobs, processes=job.cores)
            else:
                for pose_job in pose_jobs:
                    results.append(protocol(pose_job))

            # Handle the particulars of multiple PoseJob returns
            if job.module in [flags.align_helices, flags.nanohedra]:
                results_ = []
                for result in results:
                    results_.extend(result)
                results = results_
                pose_jobs = results.copy()
    # -----------------------------------------------------------------------------------------------------------------
    #  Finally, run terminate(). This formats output parameters and reports on exceptions
    # -----------------------------------------------------------------------------------------------------------------
    terminate(results=results)


def app(*args):
    exit_code = None
    try:
        main()
    except KeyboardInterrupt:
        print('\nJob Ended By KeyboardInterrupt\n')
        exit_code = 1
    except (utils.SymDesignException, StructureException):
        exit_code = 1
        print(''.join(traceback.format_exc()))
    except Exception:
        exit_code = 1
        error = 'ERROR'
        print(
            f"\n{''.join(traceback.format_exc())}\n"
            f"\033[1;31;40m{error}\033[0;0m: If your issue persists, please open an issue at: {putils.git_issue_url}\n"
        )
    finally:
        destruct_factories()
        if exit_code is not None:
            sys.exit(exit_code)


if __name__ == '__main__':
    app()
