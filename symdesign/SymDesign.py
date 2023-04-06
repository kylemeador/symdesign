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
from argparse import Namespace
from glob import glob
from itertools import count, repeat, product, combinations
from subprocess import list2cmdline
from typing import Any, AnyStr, Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, lazyload

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
from symdesign.protocols.pose import PoseJob
from symdesign.resources.job import JobResources, job_resources_factory
from symdesign.resources.query.pdb import retrieve_pdb_entries_by_advanced_query
from symdesign.resources.query.utils import validate_input_return_response_value
from symdesign.resources import distribute, sql, wrapapi
from symdesign.structure.fragment.db import fragment_factory, euler_factory
from symdesign.structure.model import Entity, Model, Pose
# from symdesign.structure import utils as stutils
from symdesign.sequence import create_mulitcistronic_sequences


resubmit_command_message = f'After completion of sbatch script(s), re-submit your {putils.program_name} ' \
                           f'command:\n\tpython {" ".join(sys.argv)}'


def initialize_entities(job: JobResources, uniprot_entities: Iterable[wrapapi.UniProtEntity],
                        metadata: Iterable[sql.ProteinMetadata], batch_commands: bool = False):
    """Handle evolutionary and structural data creation

    Args:
        job: The active JobResources singleton
        uniprot_entities: All UniProtEntity instances which should be checked for evolutionary info
        metadata: The ProteinMetadata instances that are being imported for the first time
        batch_commands: Whether commands should be made for batch submission
        # structures: The Structure instances that are being imported for the first time
    Returns:
        The processed structures
    """

    def check_if_script_and_exit(messages: list[str]):
        if messages:
            # Entity processing commands are needed
            if distribute.is_sbatch_available():
                logger.critical(distribute.sbatch_warning)
            else:
                logger.critical(distribute.script_warning)

            for message in messages:
                logger.info(message)
            print('\n')
            logger.info(resubmit_command_message)
            sys.exit(1)

    # Set up common Structure/Entity resources
    if job.design.evolution_constraint:
        profile_search_instructions = \
            job.process_evolutionary_info(uniprot_entities=uniprot_entities, batch_commands=batch_commands)
    else:
        profile_search_instructions = []

    check_if_script_and_exit(profile_search_instructions)

    # Ensure files exist
    for data in metadata:
        if data.model_source is None:
            raise ValueError(f"Couldn't find {data}.model_source")
    # Check whether loop modeling or refinement should be performed
    # If so, move the oriented Entity.file_path (should be the ASU) to the respective directory
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
            raise ValueError(f"Couldn't find {data}.model_source")


def initialize_metadata(session: Session,
                        possibly_new_uniprot_to_prot_data: dict[tuple[str, ...], sql.ProteinMetadata] = None,
                        existing_uniprot_entities: Iterable[wrapapi.UniProtEntity] = None,
                        existing_protein_metadata: Iterable[sql.ProteinMetadata] = None) -> \
        dict[tuple[str, ...], sql.ProteinMetadata] | dict:
    """Compare newly described work to the existing database and set up metadata for all described entities

    Args:
        session: A currently open transaction within sqlalchemy
        possibly_new_uniprot_to_prot_data: A mapping of the possibly required UniProtID entries and their associated
            ProteinMetadata. These could already exist in database, but were indicated they are needed
        existing_uniprot_entities: If any UniProtEntity instances are already loaded, pass them to expedite setup
        existing_protein_metadata: If any ProteinMetadata instances are already loaded, pass them to expedite setup
    """
    if not possibly_new_uniprot_to_prot_data:
        if existing_protein_metadata:
            pass
            # uniprot_id_to_metadata = {protein_data.uniprot_ids: protein_data for protein_data in existing_protein_metadata}
        elif existing_uniprot_entities:
            existing_protein_metadata = {unp_entity.protein_metadata for unp_entity in existing_uniprot_entities}
        else:
            existing_protein_metadata = {}

        return {protein_data.uniprot_ids: protein_data for protein_data in existing_protein_metadata}

    # Todo
    #  If I ever adopt the UniqueObjectValidatedOnPending recipe, that could perform the work of getting the
    #  correct objects attached to the database

    # Get the set of all UniProtIDs
    possibly_new_uniprot_ids = set()
    for uniprot_ids in possibly_new_uniprot_to_prot_data.keys():
        possibly_new_uniprot_ids.update(uniprot_ids)
    # Find existing UniProtEntity instances from database
    if existing_uniprot_entities is None:
        existing_uniprot_entities = []
    else:
        existing_uniprot_entities = list(existing_uniprot_entities)

    existing_uniprot_ids = {unp_ent.id for unp_ent in existing_uniprot_entities}
    # Remove the certainly existing from possibly new and query for any new that already exist
    query_additional_existing_uniprot_entities_stmt = \
        select(wrapapi.UniProtEntity).where(wrapapi.UniProtEntity.id.in_(
            possibly_new_uniprot_ids.difference(existing_uniprot_ids)))
    # Add all requested to those known about
    existing_uniprot_entities += session.scalars(query_additional_existing_uniprot_entities_stmt).all()

    # Todo Maybe needed?
    #  Emit this select when there is a stronger association between the multiple
    #  UniProtEntity.uniprot_ids and referencing a unique ProteinMetadata
    #  The below were never tested
    # existing_uniprot_entities_stmt = \
    #     select(sql.UniProtProteinAssociation.protein)\
    #     .where(sql.UniProtProteinAssociation.uniprot_id.in_(possibly_new_uniprot_ids))
    #     # NEED TO GROUP THESE BY ProteinMetadata.uniprot_entities
    # OR
    # existing_uniprot_entities_stmt = \
    #     select(wrapapi.UniProtEntity).join(sql.ProteinMetadata)\
    #     .where(wrapapi.UniProtEntity.uniprot_id.in_(possibly_new_uniprot_ids))
    #     # NEED TO GROUP THESE BY ProteinMetadata.uniprot_entities

    # Map the existing uniprot_id to UniProtEntity
    uniprot_id_to_unp_entity = {unp_entity.id: unp_entity for unp_entity in existing_uniprot_entities}
    insert_uniprot_ids = possibly_new_uniprot_ids.difference(uniprot_id_to_unp_entity.keys())

    # Get the remaining UniProtIDs as UniProtEntity entries
    new_uniprot_id_to_unp_entity = {uniprot_id: wrapapi.UniProtEntity(id=uniprot_id)
                                    for uniprot_id in insert_uniprot_ids}
    # Update entire dictionary for ProteinMetadata ops below
    uniprot_id_to_unp_entity.update(new_uniprot_id_to_unp_entity)
    # Insert new
    new_uniprot_entities = list(new_uniprot_id_to_unp_entity.values())
    session.add_all(new_uniprot_entities)

    # Repeat the process for ProteinMetadata
    # Map entity_id to uniprot_id for later cleaning of UniProtEntity
    possibly_new_entity_id_to_uniprot_ids = \
        {protein_data.entity_id: uniprot_ids
         for uniprot_ids, protein_data in possibly_new_uniprot_to_prot_data.items()}
    # Map entity_id to ProteinMetadata
    possibly_new_entity_id_to_protein_data = \
        {protein_data.entity_id: protein_data
         for protein_data in possibly_new_uniprot_to_prot_data.values()}
    possibly_new_entity_ids = set(possibly_new_entity_id_to_protein_data.keys())

    if existing_protein_metadata is None:
        existing_protein_metadata = []
    else:
        existing_protein_metadata = list(existing_protein_metadata)

    existing_entity_ids = {data.entity_id for data in existing_protein_metadata}
    # Remove the certainly existing from possibly new and query the new
    existing_protein_metadata_stmt = \
        select(sql.ProteinMetadata) \
        .where(sql.ProteinMetadata.entity_id.in_(possibly_new_entity_ids.difference(existing_entity_ids)))
    # Add all requested to those known about
    existing_protein_metadata += session.scalars(existing_protein_metadata_stmt).all()

    # Get the existing ProteinMetadata.entity_ids
    existing_entity_ids = {protein_data.entity_id for protein_data in existing_protein_metadata}
    # The remaining entity_ids are new and must be added
    new_entity_ids = possibly_new_entity_ids.difference(existing_entity_ids)
    uniprot_ids_to_new_metadata = {
        possibly_new_entity_id_to_uniprot_ids[entity_id]: possibly_new_entity_id_to_protein_data[entity_id]
        for entity_id in new_entity_ids}

    # Attach UniProtEntity to new ProteinMetadata by UniProtID
    for uniprot_ids, protein_metadata in uniprot_ids_to_new_metadata.items():
        # Create the ordered_list of UniProtIDs (UniProtEntity) on ProteinMetadata entry
        try:
            protein_metadata.uniprot_entities.extend(
                uniprot_id_to_unp_entity[uniprot_id] for uniprot_id in uniprot_ids)
        except KeyError:  # uniprot_id_to_unp_entity is missing a key, but I can't see a way it would be here...
            raise utils.SymDesignException(putils.report_issue)

    # Insert the remaining ProteinMetadata
    session.add_all(uniprot_ids_to_new_metadata.values())
    # Finalize additions to the database
    session.commit()

    # Now that new are found, map all UniProtIDs to all ProteinMetadata
    uniprot_id_to_metadata = {protein_data.uniprot_ids: protein_data
                              for protein_data in existing_protein_metadata}
    uniprot_id_to_metadata.update(uniprot_ids_to_new_metadata)

    return uniprot_id_to_metadata


def initialize_structures(job: JobResources, symmetry: str = None, paths: Iterable[AnyStr] = False,
                          pdb_codes: AnyStr | list = False, query_codes: bool = False) -> list[Model | Entity]:
    """From provided codes, files, or query directive, load structures into the runtime and orient them in the database

    Args:
        job: The active JobResources singleton
        symmetry: The symmetry to orient the structure identifiers
        paths: The locations on disk to search for Structure instances
        pdb_codes: The PDB API EntryID, EntityID, or AssemblyID codes to fetch Structure instances
        query_codes: Whether a PDB API query should be initiated
    Returns:
        The oriented Structure instances
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
        # eventual_structure_names1 = \
        #     list(map(os.path.basename, [os.path.splitext(file)[0] for file in pdb1_filepaths]))
    elif pdb_codes:
        # Make all names lowercase
        structure_names = list(map(str.lower, pdb_codes))
    elif query_codes:
        save_query = validate_input_return_response_value(
            'Do you want to save your PDB query to a local file?', {'y': True, 'n': False})
        print(f'\nStarting PDB query\n')
        structure_names = retrieve_pdb_entries_by_advanced_query(save=save_query, entity=True)
        # Make all names lowercase
        structure_names = list(map(str.lower, structure_names))
    else:
        structure_names = []

    # # Ensure all_entities are symmetric. As of now, all orient_structures returns are the symmetrized structure
    # for entity in [entity for structure in all_structures for entity in structure.entities]:
    #     entity.make_oligomer(symmetry=entity.symmetry)

    # Select entities, orient them, then load each Structure for further database processing
    return job.structure_db.orient_structures(structure_names, symmetry=symmetry, by_file=by_file)


def create_protein_metadata(structures: list[Model | Entity], symmetry: str = None) \
        -> tuple[list[list[tuple[str, ...]]], dict[tuple[str, ...], sql.ProteinMetadata]]:
    """From a Structure instance, extract the unique metadata to identify the entities involved

    Args:
        structures: The Structure instances to initialize to ProteinMetadata
        symmetry: The symmetry to use in initialization of ProteinMetadata
    Returns:
        A tuple comprising the pair(
            A mapping of the Structure name to the UniProtIDs of each Structure Entity,
            A mapping of the UniProtIDs to the ProteinMetadata for each unique Entity instance
        )
    """
    structures_uniprot_ids: list[list[tuple[str, ...]]] = []
    uniprot_ids_to_prot_metadata = {}
    for structure in structures:
        structure_uniprot_ids = []
        for entity in structure.entities:
            protein_metadata = sql.ProteinMetadata(
                entity_id=entity.name,
                reference_sequence=entity.reference_sequence,
                thermophilicity=entity.thermophilicity,
                symmetry_group=symmetry,
                model_source=entity.file_path
            )

            try:
                ''.join(entity.uniprot_ids)
            except TypeError:  # Uniprot_ids is (None,)
                entity.uniprot_ids = (entity.name,)
            except AttributeError:  # Unable to retrieve .uniprot_ids
                entity.uniprot_ids = (entity.name,)
            # else:  # .uniprot_ids work. Use as parsed
            uniprot_ids = entity.uniprot_ids

            if uniprot_ids in uniprot_ids_to_prot_metadata:
                # This Entity already found for processing, and we shouldn't have duplicates
                logger.error(f"Found duplicate UniProtID identifier, {uniprot_ids}, for {protein_metadata}. "
                             f"This error wasn't expected to occur.{putils.report_issue}")
                attrs_of_interest = \
                    ['entity_id', 'reference_sequence', 'thermophilicity', 'symmetry_group', 'model_source']
                exist = '\n\t.'.join([f'{attr}={getattr(protein_metadata, attr)}' for attr in attrs_of_interest])
                found_metadata = uniprot_ids_to_prot_metadata[uniprot_ids]
                found = '\n\t.'.join([f'{attr}={getattr(found_metadata, attr)}' for attr in attrs_of_interest])
                logger.critical(f'Existing ProteinMetadata:\n{exist}\nNew ProteinMetadata:{found}\n')
            else:  # Process for persistent state
                uniprot_ids_to_prot_metadata[uniprot_ids] = protein_metadata
            structure_uniprot_ids.append(uniprot_ids)  # protein_metadata)
        structures_uniprot_ids.append(structure_uniprot_ids)

    return structures_uniprot_ids, uniprot_ids_to_prot_metadata


def parse_results_for_exceptions(pose_jobs: list[PoseJob], results: Iterable[Any], **kwargs) \
        -> list[tuple[PoseJob, Exception]] | list:
    """Filter out any exceptions from results

    Args:
        pose_jobs: The PoseJob instances that attempted work
        results: The returned values from a job
    Returns:
        Tuple of passing PoseDirectories and Exceptions
    """
    if results is None:
        return []
    else:
        exception_indices = [idx for idx, result_ in enumerate(results) if isinstance(result_, BaseException)]
        return [(pose_jobs.pop(idx), results.pop(idx)) for idx in reversed(exception_indices)]


def main():
    """Run the SymDesign program"""
    # -----------------------------------------------------------------------------------------------------------------
    #  Initialize local functions
    # -----------------------------------------------------------------------------------------------------------------
    def terminate(results: list[Any] | dict = None, output: bool = True, **kwargs):
        """Format designs passing output parameters and report program exceptions

        Args:
            results: The returned results from the module run. By convention contains results and exceptions
            output: Whether the module used requires a file to be output
        """
        output_analysis = True
        # # Save any information found during the command to it's serialized state
        # try:
        #     for pose_job in pose_jobs:
        #         pose_job.pickle_info()
        # except AttributeError:  # This isn't a PoseJob. Likely is a nanohedra job
        #     pass
        nonlocal exceptions
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
                f_out.write('%s\n' % '\n'.join(str(pose_job) for pose_job, error_ in exceptions))
            logger.critical(f'The file "{exceptions_file}" contains the pose identifier of every pose that failed '
                            f'checks/filters for this job')

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
                    f_out.write('%s\n' % '\n'.join(str(pose_job) for pose_job in successful_pose_jobs))
                logger.critical(f'The file "{poses_file}" contains the pose identifier of every pose that passed checks'
                                f'/filters for this job. Utilize this file to input these poses in future '
                                f'{putils.program_name} commands such as:'
                                f'\n\t{putils.program_command} MODULE --{flags.poses} {poses_file} ...')

            # Output any additional files for the module
            if job.module in [flags.select_designs, flags.select_sequences]:
                designs_file = \
                    os.path.join(job_paths, putils.default_specification_file.format(*job.default_output_tuple))
                try:
                    with open(designs_file, 'w') as f_out:
                        f_out.write('%s\n' % '\n'.join(f'{pose_job}, {design.name}' for pose_job in successful_pose_jobs
                                                       for design in pose_job.current_designs))
                        # Todo ensure .current_designs present ^
                    logger.critical(f'The file "{designs_file}" contains the pose identifier and design identifier, of '
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
            if job.module == flags.design:
                if job.design.interface:
                    if job.design.method == putils.consensus:
                        # Todo ensure consensus sbatch generator working
                        design_stage = flags.refine
                    elif job.design.method == putils.proteinmpnn:
                        design_stage = putils.proteinmpnn
                    else:  # if job.design.method == putils.rosetta_str:
                        design_stage = putils.scout if job.design.scout \
                            else (putils.hbnet_design_profile if job.design.hbnet
                                  else (putils.structure_background if job.design.structure_background
                                        else putils.interface_design))
                else:
                    # Todo make viable rosettascripts
                    design_stage = flags.design
            else:
                design_stage = None

            module_files = {
                flags.design: design_stage,
                # flags.interface_design: design_stage,
                flags.nanohedra: flags.nanohedra,
                flags.refine: flags.refine,
                flags.interface_metrics: putils.interface_metrics,
                flags.optimize_designs: putils.optimize_designs
                # custom_script: os.path.splitext(os.path.basename(getattr(args, 'script', 'c/custom')))[0],
            }
            stage = module_files.get(job.module)
            if stage and job.distribute_work:
                scripts_to_distribute = [os.path.join(pose_job.scripts_path, f'{utils.starttime}{stage}.sh')
                                         for pose_job in successful_pose_jobs]
                distribute.check_scripts_exist(commands=scripts_to_distribute)
                distribute.commands(scripts_to_distribute, name='_'.join(job.default_output_tuple), scale=stage,
                                    out_path=job.sbatch_scripts, commands_out_path=job.job_paths)

                # Todo this mechanism fell out of favor, but really should be suggested to the user
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
    args, additional_args = flags.argparsers[flags.parser_guide].parse_known_args()
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
        # else:  # Print the full program readme and exit
        #     utils.guide.print_guide()
        #     sys.exit()
    elif args.setup:
        utils.guide.setup_instructions()
        sys.exit()
    elif args.help:
        pass  # Let the entire_parser handle their formatting
    else:  # Print the full program readme and exit
        utils.guide.print_guide()
        sys.exit()

    # ---------------------------------------------------
    # elif args.flags:  # Todo
    #     if args.template:
    #         flags.query_user_for_flags(template=True)
    #     else:
    #         flags.query_user_for_flags(mode=args.flags_module)
    # ---------------------------------------------------
    # elif args.module == 'distribute':  # -s stage, -y success_file, -n failure_file, -m max_jobs
    #     distribute.distribute(**vars(args))
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
    #         raise utils.DesignError('You must pass a single pdb file to %s. Ex:\n\t%s --single my_pdb_file.pdb '
    #                                 'residue_selector' % (putils.program_name, putils.program_command))
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
    if args.module == flags.all_flags:
        sys.argv = ['symdesign', '--help']
    elif args.module == flags.nanohedra:
        if args.query:  # Submit before we check for additional_args as query comes with additional args
            utils.nanohedra.cmdline.query_mode([__file__, '-query'] + additional_args)
            sys.exit()
        else:  # Add a dummy input for argparse to happily continue with required args
            additional_args.extend(['--file', 'dummy'])
            remove_dummy = True
    elif args.module == flags.initialize_building_blocks:
        # Add a dummy input for argparse to happily continue with required args
        additional_args.extend(['--file', 'dummy'])
        remove_dummy = True
    elif args.module == flags.protocol:
        # Add a dummy input for argparse to happily continue with required args
        if flags.nanohedra in args.modules:
            additional_args.extend(['--file', 'dummy'])
            remove_dummy = True
        # Parse all options for every module provided
        # input(f'{args}\n\n{additional_args}')
        all_args = [args]
        for idx, module in enumerate(args.modules):
            # additional_args = [module] + additional_args
            args, additional_args = \
                module_parser.parse_known_args(args=[module] + additional_args)
                                               # , namespace=args)
            all_args.append(args)
            # input(f'{args}\n\n{additional_args}')

        # Invert all the arguments to ensure those that were specified first are set and not overwritten by default
        for _args in reversed(all_args):
            _args_ = vars(args)
            _args_.update(**vars(_args))
            # print(_args_)
            args = Namespace(**_args_)
            # args = Namespace(**vars(args), **vars(_args))
            # input(args)
        # Set the module to flags.protocol again after parsing
        args.module = flags.protocol

    # Check the provided flags against the full SymDesign entire_parser to print any help
    entire_parser = flags.argparsers[flags.parser_entire]
    _args, _additional_args = entire_parser.parse_known_args()
    # Parse the provided flags
    for argparser in [flags.parser_options, flags.parser_residue_selector, flags.parser_output, flags.parser_input]:
        args, additional_args = flags.argparsers[argparser].parse_known_args(args=additional_args, namespace=args)

    if additional_args:
        print(f"\nFound flag(s) that aren't recognized with the requested job/module(s): {', '.join(additional_args)}\n"
              'Please correct/remove them and resubmit your command. Try adding -h/--help for available formatting\n'
              f"If you want to view all {putils.program_name} flags, "
              f"replace the MODULE '{args.module}' with '{flags.all_flags}'")
        sys.exit(1)

    if remove_dummy:  # Remove the dummy input
        del args.file
    # -----------------------------------------------------------------------------------------------------------------
    #  Find base output directory and check for proper set up of program i/o
    # -----------------------------------------------------------------------------------------------------------------
    symdesign_directory = utils.get_program_root_directory(
        (args.directory or (args.project or args.single or [None])[0] or os.getcwd()))

    project_name = None
    """Used to set a leading string on all new PoseJob paths"""
    if symdesign_directory is None:  # We found no directory from the input
        # By default, assume new input and make in the current directory
        new_program_output = True
        symdesign_directory = os.path.abspath(os.path.join(os.getcwd(), putils.program_output))
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
                    if extension == '':  # Provided as directory/pose_name (pose_identifier) from SymDesignOutput
                        file_directory = utils.get_program_root_directory(line)
                        if file_directory is not None:
                            symdesign_directory = file_directory
                            new_program_output = False
                    else:
                        # Set file basename as "project_name" in case poses are being integrated for the first time
                        # In this case, pose names might be the same, so we take the basename of the first file path as
                        # the name to discriminate between separate files
                        project_name = os.path.splitext(os.path.basename(file))[0]
                break
        # Ensure the base directory is made if it is indeed new and in os.getcwd()
        putils.make_path(symdesign_directory)
    else:
        new_program_output = False
    # -----------------------------------------------------------------------------------------------------------------
    #  Process JobResources which holds shared program objects and command-line arguments
    # -----------------------------------------------------------------------------------------------------------------
    job = job_resources_factory.get(program_root=symdesign_directory, arguments=args, initial=new_program_output)
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
    decoy_modules = ['input', 'output', 'options', 'residue_selector']
    symdesign_tools = [flags.multicistronic] + decoy_modules
    #     flags.input, flags.output, flags.options, flags.residue_selector]
    if job.module in symdesign_tools:
        if job.module == flags.multicistronic:
            create_mulitcistronic_sequences(args)
        elif job.module == flags.update_db:
            with job.db.session() as session:
                design_stmt = select(sql.DesignData).where(sql.DesignProtocol.file.is_not(None)) \
                    .where(sql.DesignProtocol.design_id == sql.DesignData.id)

                designs = session.scalars(design_stmt).unique().all()
                for design in designs:
                    for protocol in design.protocols:
                        if protocol.protocol == 'thread':
                            design.structure_path = protocol.file
                        # else:
                        #     print(protocol.protocol)

                counter = count()
                for design in designs:
                    if design.structure_path is None:
                        print(design)
                    else:
                        next(counter)
                print(f'Found {next(counter)} updated designs')
                session.rollback()
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
        # # When selecting by dataframe or metric, don't initialize, input is handled in module protocol
        # if job.dataframe:  # or job.metric:
        #     initialize = False
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
    # elif job.module == flags.interface_design:
    else:
        pass
        # if job.module not in [flags.interface_design, flags.design, flags.refine, flags.optimize_designs,
        #                       flags.interface_metrics, flags.analysis, flags.process_rosetta_metrics,
        #                       flags.generate_fragments, flags.orient, flags.expand_asu,
        #                       flags.rename_chains, flags.check_clashes]:
        #     # , 'custom_script', 'find_asu', 'status', 'visualize'
        #     # We have no module passed. Print the guide and exit
        #     utils.guide.print_guide()
        #     sys.exit()
        # else:
        #     # Set up design directories
        #     pass
    # -----------------------------------------------------------------------------------------------------------------
    #  Report options and Set up Databases
    # -----------------------------------------------------------------------------------------------------------------
    reported_args = job.report_specified_arguments(args)
    logger.info('Starting with options:\n\t%s' % '\n\t'.join(utils.pretty_format_table(reported_args.items())))

    logger.info(f'Using resources in Database located at "{job.data}"')
    if job.module in [flags.nanohedra, flags.generate_fragments, flags.design, flags.analysis]:  # interface_design
        if job.design.term_constraint:
            job.fragment_db = fragment_factory(source=args.fragment_database)
            # Initialize EulerLookup class
            euler_factory()
            if job.module == flags.generate_fragments and job.fragment_db.source == putils.biological_interfaces:
                logger.warning(f'The FragmentDatabase {job.fragment_db.source} has only been created with '
                               'biological homo-oligomers. Use fragment information with caution')
    # -----------------------------------------------------------------------------------------------------------------
    #  Initialize the db.session, set into job namespace
    # -----------------------------------------------------------------------------------------------------------------
    results = []
    exceptions = []
    # -----------------------------------------------------------------------------------------------------------------
    #  Grab all Poses (PoseJob instance) from either database, directory, project, single, or file
    # -----------------------------------------------------------------------------------------------------------------
    # pose_jobs hold jobs with specific poses
    # list[PoseJob] for an establishes pose
    # list[tuple[Structure, Structure]] for a nanohedra docking job
    pose_jobs: list[PoseJob] | list[tuple[Any, Any]] = []
    logger.info(f'Setting up input for {job.module}')
    if job.module == flags.initialize_building_blocks:
        logger.critical(f'Ensuring provided building blocks are oriented')
        symmetry = job.sym_entry.group1
        orient_structures = initialize_structures(job, symmetry=symmetry, paths=args.oligomer1,
                                                  pdb_codes=job.pdb_codes1, query_codes=args.query_codes1)
        structures_uniprot_ids, possibly_new_uniprot_to_prot_metadata = \
            create_protein_metadata(orient_structures, symmetry=symmetry)

        # Make a copy of the new ProteinMetadata if they were already loaded without a .model_source attribute
        possibly_new_uniprot_to_prot_metadata_copy = possibly_new_uniprot_to_prot_metadata.copy()

        # Write new data to the database
        with job.db.session(expire_on_commit=False) as session:
            all_uniprot_id_to_prot_data = initialize_metadata(session, possibly_new_uniprot_to_prot_metadata)

            # Get all uniprot_entities, and fix ProteinMetadata that is already loaded
            uniprot_entities = []
            for data in all_uniprot_id_to_prot_data.values():
                if data.model_source is None:
                    logger.info(f'{data}.model_source is None')
                    for uniprot_ids, protein_metadata in possibly_new_uniprot_to_prot_metadata_copy.items():
                        if data.entity_id == protein_metadata.entity_id:
                            data.model_source = protein_metadata.model_source
                            logger.info(f'Set existing {data}.model_source to new {protein_metadata}.model_source')
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
                        setattr(protein_metadata, attribute, value)
                session.commit()
            else:
                # Set up evolution and structures. All attributes will be reflected in ProteinMetadata
                initialize_entities(job, uniprot_entities, all_uniprot_id_to_prot_data.values(),
                                    batch_commands=job.distribute_work)
                session.commit()

        terminate(output=False)
    elif job.module == flags.nanohedra:
        # logger.info(f'Setting up inputs for {job.module.title()} docking')
        job.sym_entry.log_parameters()
        # # Make master output directory. sym_entry is required, so this won't fail v
        # if args.output_directory is None:
        #     # job.output_directory = os.path.join(job.projects, f'NanohedraEntry{sym_entry.number}_BUILDING-BLOCKS_Poses')
        #     job.output_directory = job.projects
        #     putils.make_path(job.output_directory)
        # Transform input entities to canonical orientation and return their ASU
        grouped_orient_structures: list[list[Model | Entity]] = []
        logger.critical(f'Ensuring provided building blocks are oriented for docking')
        structures1 = initialize_structures(job, symmetry=job.sym_entry.group1, paths=args.oligomer1,
                                            pdb_codes=job.pdb_codes1, query_codes=args.query_codes1)
        grouped_orient_structures.append(structures1)
        structures2 = []
        # See if they are the same input
        if args.oligomer1 != args.oligomer2 or job.pdb_codes1 != job.pdb_codes2 or args.query_codes2:
            structures2 = initialize_structures(job, symmetry=job.sym_entry.group2, paths=args.oligomer2,
                                                pdb_codes=job.pdb_codes2, query_codes=args.query_codes2)
            if structures2:
                single_component_design = False
            else:
                single_component_design = True
        else:  # The entities are the same symmetry, or there is a single component and bad input
            single_component_design = True
        grouped_orient_structures.append(structures2)

        # Initialize the local database
        # Populate all_entities to set up sequence dependent resources
        # grouped_structures_ids = defaultdict(list)
        # Todo 2 expand the definition of SymEntry/Entity to include
        #  specification of T:{T:{C3}{C3}}{C1}
        #  where an Entity is composed of multiple Entity (Chain) instances
        #  This helps with the grouping by input model... not entities such as in Nanohedra pose.output_pose()
        #  # Write Model1, Model2
        #  if job.output_oligomers:
        #      for entity in pose.entities:
        #          entity.write(oligomer=True, out_path=os.path.join(out_dir, f'{entity.name}_{pose_name}.pdb'))
        #  Which should be
        #      for model in pose.entities:
        #          model.write(assembly=True, out_path=os.path.join(out_dir, f'{entity.name}_{pose_name}.pdb'))
        #  Essentially this would make oligomer/assembly keywords the same
        #  and allow a multi-entity Model/Pose as an Entity in a Pose... Recursion baby
        #
        grouped_structures_uniprot_ids: list[list[list[tuple[str, ...]]]] = []
        possibly_new_uniprot_to_prot_metadata = {}
        # symmetry_map = job.sym_entry.groups if job.sym_entry else repeat(None)
        for orient_structures, symmetry in zip(grouped_orient_structures, job.sym_entry.groups):
            if not orient_structures:  # Useful in a case where symmetry groups are the same or group is None
                continue
            structures_uniprot_ids, possibly_new_uni_to_prot_metadata = \
                create_protein_metadata(orient_structures, symmetry=symmetry)
            grouped_structures_uniprot_ids.append(structures_uniprot_ids)
            possibly_new_uniprot_to_prot_metadata.update(possibly_new_uni_to_prot_metadata)

        # Make a copy of the new ProteinMetadata if they were already loaded without a .model_source attribute
        possibly_new_uniprot_to_prot_metadata_copy = possibly_new_uniprot_to_prot_metadata.copy()

        # Write new data to the database
        with job.db.session(expire_on_commit=False) as session:
            all_uniprot_id_to_prot_data = initialize_metadata(session, possibly_new_uniprot_to_prot_metadata)

            # Get all uniprot_entities, and fix ProteinMetadata that is already loaded
            uniprot_entities = []
            for data in all_uniprot_id_to_prot_data.values():
                if data.model_source is None:
                    logger.info(f'{data}.model_source is None')
                    for uniprot_ids, protein_metadata in possibly_new_uniprot_to_prot_metadata_copy.items():
                        if data.entity_id == protein_metadata.entity_id:
                            data.model_source = protein_metadata.model_source
                            logger.info(f'Set existing {data}.model_source to new {protein_metadata}.model_source')
                uniprot_entities.extend(data.uniprot_entities)

            # Set up evolution and structures. All attributes will be reflected in ProteinMetadata
            initialize_entities(job, uniprot_entities, all_uniprot_id_to_prot_data.values(),
                                batch_commands=job.distribute_work)

            # Todo need to take the version of all_structures from refine/loop modeling and insert entity.metadata
            #  then usage for docking pairs below...

            # Correct existing ProteinMetadata, now that Entity instances are processed
            grouped_docking_structures = []
            # for symmetry, structures_ids in grouped_structures_ids:
            # for group_idx, (symmetry, structures_uniprot_ids) in enumerate(
            #         zip(job.sym_entry.groups, grouped_structures_uniprot_ids)):
            for orient_structures, structures_uniprot_ids in zip(grouped_orient_structures, grouped_structures_uniprot_ids):
                structures = []
                for orient_structure, structure_uniprot_ids in zip(orient_structures, structures_uniprot_ids):
                    entities = []
                    for uniprot_ids in structure_uniprot_ids:
                        # This data may not be the one that is initialized, grab the correct one
                        data = all_uniprot_id_to_prot_data[uniprot_ids]
                        entity = Entity.from_file(data.model_source, name=data.entity_id, metadata=data)
                        entity.stride(to_file=job.api_db.stride.path_to(name=data.entity_id))
                        data.n_terminal_helix = entity.is_termini_helical()
                        data.c_terminal_helix = entity.is_termini_helical('c')
                        # Set this attribute to carry through Nanohedra
                        entity.metadata = data
                        # entity.properties_id = data.id
                        entities.append(entity)
                    # Don't include symmetry as this will be initialized by fragdock.fragment_dock()
                    structures.append(Pose.from_entities(entities, name=orient_structure.name))  # symmetry=symmetry))
                grouped_docking_structures.append(structures)
            session.commit()

        # Make all possible structure pairs given input entities by finding entities from entity_names
        # Using combinations of directories with .pdb files
        if single_component_design:
            logger.info('Treating as single component docking, no additional entities requested')
            # structures1 = [entity for entity in all_entities if entity.name in structures1]
            # ^ doesn't work as entity_id is set in orient_structures, but structure name is entry_id
            all_structures = []
            for structures in grouped_docking_structures:
                all_structures.extend(structures)
            pose_jobs.extend(combinations(all_structures, 2))
        else:
            # # v doesn't work as entity_id is set in orient_structures, but structure name is entry_id
            # # structures1 = [entity for entity in all_entities if entity.name in structures1]
            # # structures2 = [entity for entity in all_entities if entity.name in structures2]
            pose_jobs = []
            for structures1, structures2 in combinations(grouped_docking_structures, 2):
                pose_jobs.extend(product(structures1, structures2))

        job.location = f'NanohedraEntry{job.sym_entry.number}'  # Used for terminate()
        if not pose_jobs:  # No pairs were located
            print('No docking pairs were located from your input. Please ensure that your flags are as intended.'
                  f'{putils.issue_submit_warning}')
            sys.exit(1)
        # Todo
        #  This could be moved above Entity/Pose load as it's not necessary in that situation
        elif job.distribute_work:
            # Write all commands to a file and distribute a batched job
            # The theory of the new command is to keep everything that was submitted however modify the input
            # arguments so that a single command contains a pair of input models

            possible_input_args = [arg for args in flags.nanohedra_mutual1_arguments.keys() for arg in args] \
                + [arg for args in flags.nanohedra_mutual2_arguments.keys() for arg in args]
            #     + list(flags.distribute_args)
            submitted_args = sys.argv[1:]
            for input_arg in possible_input_args:
                try:
                    pop_index = submitted_args.index(input_arg)
                except ValueError:  # Not in list
                    continue
                else:
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
                else:
                    submitted_args.pop(pop_index)

            # Format all commands given model pair
            cmd = ['python', putils.program_exe] + submitted_args
            commands = [cmd.copy() + [f'--{flags.pdb_codes1}', model1.name,
                                      f'--{flags.pdb_codes2}', model2.name]
                        for idx, (model1, model2) in enumerate(pose_jobs)]
            # Write commands
            distribute.commands([list2cmdline(cmd) for cmd in commands], name='_'.join(job.default_output_tuple),
                                protocol=job.module, out_path=job.sbatch_scripts, commands_out_path=job.job_paths)
            terminate(output=False)
    else:  # Load from existing files, usually Structural files in a directory or in the program already
        # if args.nanohedra_output:  # Nanohedra directory
        #     file_paths, job.location = utils.collect_nanohedra_designs(files=args.file, directory=args.directory)
        #     if file_paths:
        #         first_pose_path = file_paths[0]
        #         if first_pose_path.count(os.sep) == 0:
        #             job.nanohedra_root = args.directory
        #         else:
        #             job.nanohedra_root = f'{os.sep}{os.path.join(*first_pose_path.split(os.sep)[:-4])}'
        #         if not job.sym_entry:  # Get from the Nanohedra output
        #             job.sym_entry = get_sym_entry_from_nanohedra_directory(job.nanohedra_root)
        #         pose_jobs = [PoseJob.from_file(pose, project=project_name)
        #                             for pose in job.get_range_slice(file_paths)]
        #         # Copy the master nanohedra log
        #         project_designs = \
        #             os.path.join(job.projects, f'{os.path.basename(job.nanohedra_root)}')  # _{putils.pose_directory}')
        #         if not os.path.exists(os.path.join(project_designs, putils.master_log)):
        #             putils.make_path(project_designs)
        #             shutil.copy(os.path.join(job.nanohedra_root, putils.master_log), project_designs)
        if job.load_to_db:
            # These are not part of the db, but exist in a program_output
            if args.project or args.single:
                paths, job.location = utils.collect_designs(projects=args.project, singles=args.single)
                #                                             directory = symdesign_directory,
            # elif select_from_directory:
            #     paths, job.location = utils.collect_designs(directory=args.directory)
            else:
                raise utils.InputError(
                    f"Can't --load-to-db without passing {flags.format_args(flags.project_args)} or "
                    f"{flags.format_args(flags.single_args)}")
            if paths:  # There are files present
                pose_jobs = [PoseJob.from_directory(path) for path in job.get_range_slice(paths)]
        elif args.poses or args.specification_file or args.project or args.single:
            # Use sqlalchemy database selection to find the requested work
            pose_identifiers = []
            designs = []
            directives = []
            if args.poses or args.specification_file:
                # Todo no --file and --specification-file at the same time
                # These poses are already included in the "program state"
                if not args.directory:  # Todo react .directory to program operation inside or in dir with SymDesignOutput
                    raise utils.InputError(
                        f'A {flags.format_args(flags.directory_args)} must be provided when using '
                        f'{flags.format_args(flags.specification_file_args)}')
                # Todo, combine this with collect_designs
                #  this works for file locations as well! should I have a separate mechanism for each?
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
        elif select_from_directory:
            # Can make an empty pose_jobs when the program_root is args.directory
            job.location = args.directory
        else:  # args.file or args.directory
            file_paths, job.location = utils.collect_designs(files=args.file, directory=args.directory)
            if file_paths:
                pose_jobs = [PoseJob.from_path(path, project=project_name)
                             for path in job.get_range_slice(file_paths)]
        if job.module not in flags.select_modules:  # select_from_directory:
            if not pose_jobs and not select_from_directory:
                # Todo this needs a more informative error. Is the location of the correct format?
                #  For instance, a --project provided with the directory of type --single would die without much
                #  knowledge of why
                raise utils.InputError(
                    f"No {PoseJob.__name__}'s found from input location '{job.location}'")
            """Check to see that proper data/files have been created 
            Data includes:
            - UniProtEntity
            - ProteinMetadata
            Files include:
            - Structure in orient, refined, loop modelled
            - Profile from hhblits, bmdca?
            """

            if job.sym_entry:
                symmetry_map = job.sym_entry.groups
                preprocess_entities_by_symmetry: dict[str, list[Entity]] = \
                    {symmetry: [] for symmetry in job.sym_entry.groups}
            else:
                symmetry_map = repeat('C1')
                preprocess_entities_by_symmetry = {'C1': []}

            # Get api wrapper
            retrieve_stride_info = job.api_db.stride.retrieve_data
            possibly_new_uniprot_to_prot_metadata: dict[tuple[str, ...], sql.ProteinMetadata] = {}
            # Todo
            # existing_uniprot_ids = set()
            # existing_protein_properties_ids = set()
            existing_uniprot_entities = set()
            existing_protein_metadata = set()
            remove_pose_jobs = []
            pose_jobs_to_commit = []
            for idx, pose_job in enumerate(pose_jobs):
                if pose_job.id is None:
                    # Todo expand the definition of SymEntry/Entity to include
                    #  specification of T:{T:{C3}{C3}}{C1}
                    #  where an Entity is composed of multiple Entity (Chain) instances
                    # Need to initialize the local database. Load this model to get required info
                    try:
                        pose_job.load_initial_model()
                    except utils.InputError as error:
                        logger.error(error)
                        remove_pose_jobs.append(idx)
                        continue

                    for entity, symmetry in zip(pose_job.initial_model.entities, symmetry_map):
                        try:
                            ''.join(entity.uniprot_ids)
                        except TypeError:  # Uniprot_ids is (None,)
                            entity.uniprot_ids = (entity.name,)
                        except AttributeError:  # Unable to retrieve .uniprot_ids
                            entity.uniprot_ids = (entity.name,)
                        # else:  # .uniprot_ids work. Use as parsed
                        uniprot_ids = entity.uniprot_ids

                        # Check if the tuple of UniProtIDs has already been observed
                        protein_metadata = possibly_new_uniprot_to_prot_metadata.get(uniprot_ids, None)
                        if protein_metadata is None:
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
                                # We already have this SS information
                                entity.secondary_structure = parsed_secondary_structure
                            else:
                                # entity = Entity.from_file(data.model_source, name=data.entity_id, metadata=data)
                                entity.stride(to_file=job.api_db.stride.path_to(name=entity.name))
                            protein_metadata.n_terminal_helix = entity.is_termini_helical()
                            protein_metadata.c_terminal_helix = entity.is_termini_helical('c')
                            # for uniprot_id in entity.uniprot_ids:
                            #     if uniprot_id in possibly_new_uniprot_to_prot_metadata:
                            #         # This Entity already found for processing
                            #         pass
                            #     else:  # Process for persistent state
                            #         possibly_new_uniprot_to_prot_metadata[uniprot_id] = protein_metadata

                            possibly_new_uniprot_to_prot_metadata[uniprot_ids] = protein_metadata
                            preprocess_entities_by_symmetry[symmetry].append(entity)
                        else:
                            # This Entity already found for processing, and we shouldn't have duplicates
                            found_metadata = sql.ProteinMetadata(
                                entity_id=entity.name,
                                reference_sequence=entity.reference_sequence,
                                thermophilicity=entity.thermophilicity,
                                # There could be no sym_entry, so fall back on the entity.symmetry
                                symmetry_group=symmetry if symmetry else entity.symmetry
                            )
                            logger.error(f"Found duplicate UniProtID identifier, {uniprot_ids}, for {protein_metadata}. "
                                         f"This error wasn't expected to occur.{putils.report_issue}")
                            attrs_of_interest = \
                                ['entity_id', 'reference_sequence', 'thermophilicity', 'symmetry_group', 'model_source']
                            exist = '\n\t.'.join(
                                [f'{attr}={getattr(protein_metadata, attr)}' for attr in attrs_of_interest])

                            found = '\n\t.'.join([f'{attr}={getattr(found_metadata, attr)}' for attr in attrs_of_interest])
                            logger.critical(f'Existing ProteinMetadata:\n{exist}\nNew ProteinMetadata:{found}\n')

                        # # Create EntityData
                        # # entity_data.append(sql.EntityData(pose=pose_job,
                        # sql.EntityData(pose=pose_job,
                        #                meta=protein_metadata
                        #                )
                    # # Update PoseJob
                    # pose_job.entity_data = entity_data
                    pose_jobs_to_commit.append(pose_job)
                else:  # PoseJob is initialized
                    # # Add each UniProtEntity to existing_uniprot_entities to limit work
                    # Todo
                    # for data in pose_job.entity_data:
                    #     existing_protein_properties_ids.add(data.properties_id)
                    #     # Now loading these in initialize_metadata()
                    #     # for uniprot_id in data.meta.uniprot_ids:
                    #     #     existing_uniprot_ids.add(uniprot_id)
                    # Add each UniProtEntity to existing_uniprot_entities to limit work
                    for data in pose_job.entity_data:
                        existing_protein_metadata.add(data.meta)
                        for uniprot_entity in data.meta.uniprot_entities:
                            existing_uniprot_entities.add(uniprot_entity)

            for idx in reversed(remove_pose_jobs):
                pose_jobs.pop(idx)

            if not pose_jobs and not select_from_directory:
                raise utils.InputError(
                    f"No viable {PoseJob.__name__}'s found at location '{job.location}'. See the log for details")

            with job.db.session(expire_on_commit=False) as session:
                # Deal with new data compared to existing entries
                all_uniprot_id_to_prot_data = \
                    initialize_metadata(session, possibly_new_uniprot_to_prot_metadata,
                                        existing_uniprot_entities=existing_uniprot_entities,
                                        existing_protein_metadata=existing_protein_metadata)
                # # Deal with new data compared to existing entries
                # # Todo
                # all_uniprot_id_to_prot_data = \
                #     initialize_metadata(session, possibly_new_uniprot_to_prot_metadata,
                #                         # existing_uniprot_ids=existing_uniprot_ids,
                #                         existing_protein_metadata_ids=existing_protein_properties_ids)

                # Get all uniprot_entities, and fix ProteinMetadata that is already loaded
                uniprot_entities = []
                for data in all_uniprot_id_to_prot_data.values():
                    uniprot_entities.extend(data.uniprot_entities)

                # # Populate all_structures to set up structure dependent resources
                # all_structures = []
                # # Orient entities, then load each entity to all_structures for further database processing
                # for symmetry, entities in preprocess_entities_by_symmetry.items():
                #     if not entities:  # Useful in a case where symmetry groups are the same or group is None
                #         continue
                #     # all_entities.extend(entities)
                #     # job.structure_db.orient_structures(
                #     #     [entity.name for entity in entities], symmetry=symmetry)
                #     # Can't do this ^ as structure_db.orient_structures sets .name, .symmetry, and .file_path on each Entity
                #     all_structures.extend(job.structure_db.orient_structures(
                #         [entity.name for entity in entities], symmetry=symmetry))
                #     # Todo orient Entity individually, which requires symmetric oligomer be made
                #     #  This could be found from Pose._assign_pose_transformation() or new mechanism
                #     #  Where oligomer is deduced from available surface fragment overlap with the specified symmetry...
                #     #  job.structure_db.orient_entities(entities, symmetry=symmetry)
                #
                # # Indicate for the ProteinMetadata the characteristics of the Structure in the database
                # for structure in all_structures:
                #     for entity in structure.entities:
                #         protein_metadata = all_uniprot_id_to_prot_data[entity.uniprot_ids]
                #         # Importantly, we add oriented attribute to aid in any future processing
                #         protein_metadata.model_source = entity.file_path
                #
                # Set up evolution and structures. All attributes will be reflected in ProteinMetadata
                initialize_entities(job, uniprot_entities, [])  # all_uniprot_id_to_prot_data.values())
                #
                # # Todo replace the passed files with the processed versions?
                # #  See PoseJob.load_pose()
                # # for pose_job in pose_jobs:
                # #     for idx, entity in enumerate(pose_job.initial_model.entities):
                # #         pose_job.initial_model.entities[idx]
                #
                # for uniprot_ids, data in all_uniprot_id_to_prot_data.items():
                #     # Try to get the already parsed secondary structure information
                #     parsed_secondary_structure = retrieve_stride_info(name=entity.name)
                #     if parsed_secondary_structure:
                #         continue  # We already have this SS information
                #         # entity.secondary_structure = parsed_secondary_structure
                #     else:
                #         entity = Entity.from_file(data.model_source, name=data.entity_id, metadata=data)
                #         entity.stride(to_file=job.api_db.stride.path_to(name=data.entity_id))
                #         data.n_terminal_helix = entity.is_termini_helical()
                #         data.c_terminal_helix = entity.is_termini_helical('c')

                if pose_jobs_to_commit:
                    # Write new data to the database with correct unique entries
                    for pose_job in pose_jobs_to_commit:
                        pose_job.entity_data.extend(
                            sql.EntityData(meta=all_uniprot_id_to_prot_data[entity.uniprot_ids])
                            for entity in pose_job.initial_model.entities)
                        # Todo this could be useful if optimizing database access with concurrency
                        # # Ensure that the pose has entity_transform information saved to the db
                        # pose_job.load_pose()
                    session.add_all(pose_jobs_to_commit)

                    # When pose_jobs_to_commit already exist, deal with it by getting those already
                    # OR raise a useful error for the user about input
                    try:
                        session.commit()
                    except SQLAlchemyError:
                        # Remove pose_jobs_to_commit from session
                        session.rollback()
                        # Find the actual pose_jobs_to_commit and place in session
                        pose_identifiers = [pose_job.new_pose_identifier for pose_job in pose_jobs_to_commit]
                        fetch_jobs_stmt = select(PoseJob).where(PoseJob.pose_identifier.in_(pose_identifiers))
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
            # Todo set up a job based data acquisition as this takes some time and loading everythin isn't necessary!
            job.structure_db.load_all_data()
            job.api_db.load_all_data()

        # Todo
        #  f'Found {len(pose_jobs)}...' not accurate with select_from_directory
        logger.info(f'Found {len(pose_jobs)} unique poses from provided input location "{job.location}"')
        if not job.debug and not job.skip_logging and not select_from_directory:
            representative_pose_job = next(iter(pose_jobs))
            if representative_pose_job.log_path:
                logger.info(f'All design specific logs are located in their corresponding directories\n\tEx: '
                            f'{representative_pose_job.log_path}')
    # -----------------------------------------------------------------------------------------------------------------
    #  Set up Job specific details and resources
    # -----------------------------------------------------------------------------------------------------------------
    # Format computational requirements
    distribute_modules = [
        flags.nanohedra, flags.refine, flags.design, flags.interface_metrics, flags.optimize_designs
    ]  # flags.interface_design,
    if job.module in distribute_modules:
        if job.distribute_work:
            logger.info('Writing module commands out to file, no modeling will occur until commands are executed')

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
            protocol = getattr(protocols, protocol_name.replace('-', '_'))
            # Figure out how the job should be set up
            if job.module in protocols.config.run_on_pose_job:  # Single poses
                if job.multi_processing:
                    results_ = utils.mp_map(protocol, pose_jobs, processes=job.cores)
                else:
                    results_ = [protocol(pose_job) for pose_job in pose_jobs]
            else:  # Collection of pose_jobs
                results_ = protocol(pose_jobs)

            # Handle any returns that require particular treatment
            if job.module in protocols.config.returns_pose_jobs:
                results = []
                if results_:  # Not an empty list
                    if isinstance(results_[0], list):  # In the case returning collection of pose_jobs (nanohedra)
                        for result in results_:
                            results.extend(result)
                    else:
                        results.extend(results_)
                pose_jobs = results
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
        #     if args.update:
        #         for pose_job in pose_jobs:
        #             update_status(pose_job.serialized_info, args.stage, mode=args.update)
        #     else:
        #         if job.design.number:
        #             logger.info('Checking for %d files based on --{flags.design_number}' % args.design_number)
        #         if args.stage:
        #             status(pose_jobs, args.stage, number=job.design.number)
        #         else:
        #             for stage in putils.stage_f:
        #                 s = status(pose_jobs, stage, number=job.design.number)
        #                 if s:
        #                     logger.info('For "%s" stage, default settings should generate %d files'
        #                                 % (stage, putils.stage_f[stage]['len']))
        # # ---------------------------------------------------
        # Todo
        # elif job.module == 'find_asu':
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
        # elif job.module == 'check_unmodelled_clashes':
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
        #         zipped_args = zip(pose_jobs, repeat(args.script), repeat(args.force), repeat(args.file_list),
        #                           repeat(args.native), repeat(job.suffix), repeat(args.score_only),
        #                           repeat(args.variables))
        #         results = utils.mp_starmap(PoseJob.custom_rosetta_script, zipped_args, processes=job.cores)
        #     else:
        #         for pose_job in pose_jobs:
        #             results.append(pose_job.custom_rosetta_script(args.script, force=args.force,
        #                                                           file_list=args.file_list, native=args.native,
        #                                                           suffix=job.suffix, score_only=args.score_only,
        #                                                           variables=args.variables))
        #
        #     terminate(results=results)
        # ---------------------------------------------------
        elif job.module == flags.cluster_poses:
            protocols.cluster.cluster_poses(pose_jobs)
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
                        raise IndexError(f"There was no --{flags.dataframe} specified and one couldn't be located at "
                                         f'the location "{job.location}". Initialize again with the path to the '
                                         'relevant dataframe')

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
                    else:
                        logger.critical(f"The module 'memory_profiler' isn't installed {profile_error}")
                    print('Done profiling')
                    sys.exit()

            if args.multi_processing:
                results = utils.mp_map(protocol, pose_jobs, processes=job.cores)
            else:
                for pose_job in pose_jobs:
                    results.append(protocol(pose_job))

            # Handle the particulars of multiple PoseJob returns
            if job.module == flags.nanohedra:
                results_ = []
                for result in results:
                    results_.extend(result)
                results = results_
                pose_jobs = results.copy()
    # -----------------------------------------------------------------------------------------------------------------
    #  Finally, run terminate(). This formats output parameters and reports on exceptions
    # -----------------------------------------------------------------------------------------------------------------
    terminate(results=results)


def app():
    try:
        main()
    except KeyboardInterrupt:
        print('\nRun Ended By KeyboardInterrupt\n')
        sys.exit(1)


if __name__ == '__main__':
    app()
