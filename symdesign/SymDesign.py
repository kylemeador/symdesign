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
from itertools import repeat, product, combinations
from subprocess import list2cmdline
from typing import Any, AnyStr, Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

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
from symdesign.resources.job import job_resources_factory
from symdesign.resources.query.pdb import retrieve_pdb_entries_by_advanced_query
from symdesign.resources.query.utils import validate_input_return_response_value
from symdesign.resources import sql, wrapapi
from symdesign.structure.fragment.db import fragment_factory, euler_factory
from symdesign.structure.model import Entity, Model
# from symdesign.structure import utils as stutils
from symdesign.sequence import create_mulitcistronic_sequences
from symdesign.utils import guide, nanohedra

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
    def parse_results_for_exceptions(results_: list[Any] | dict = None, **kwargs) \
            -> list[tuple[PoseJob, Exception]] | list:
        """For a multimodule protocol, filter out any exceptions before proceeding to the next module

        Args:
            results_: The returned values from the jobs
        Returns:
            Tuple of passing PoseDirectories and Exceptions
        """
        if results_ is None:
            return []
        else:
            exception_indices = [idx for idx, result_ in enumerate(results_) if isinstance(result_, BaseException)]
            return [(pose_jobs.pop(idx), results_.pop(idx)) for idx in reversed(exception_indices)]

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
            if job.module == flags.nanohedra:
                success = results
                if job.distribute_work:
                    output_analysis = False
            else:
                # success_indices = [idx for idx, result in enumerate(results) if not isinstance(result, BaseException)]
                # success = [pose_jobs[idx] for idx in success_indices]
                # _exceptions = [(pose_job, exception) for pose_job, exception in zip(pose_jobs, results)
                #                if isinstance(exception, BaseException)]
                # exception_indices = [idx for idx, result in enumerate(results) if isinstance(result, BaseException)]
                # _exceptions = [(pose_jobs.pop(idx), results.pop(idx)) for idx in reversed(exception_indices)]
                # exceptions += _exceptions
                exceptions += parse_results_for_exceptions(results)
                success = pose_jobs
        else:
            success = []

        exit_code = 0
        if exceptions:
            print('\n')
            logger.warning(f'Exceptions were thrown for {len(exceptions)} designs. '
                           f'Check their individual .log files for more details\n\t%s'
                           % '\n\t'.join(f'{pose_job}: {error_}' for pose_job, error_ in exceptions))
            print('\n')

        if success and output:
            design_source = job.input_source
            job_paths = job.job_paths

            if low and high:
                design_source = f'{design_source}-{low:.2f}-{high:.2f}'

            # Format the output file depending on specified name and module type
            default_output_tuple = (utils.starttime, job.module, design_source)
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
                    if not args.output:  # No output is specified
                        output_analysis = False
                else:
                    # Set the poses_file to the provided job.output_file
                    poses_file = job.output_file
            else:
                # For certain modules, use the default file type
                if job.module == flags.analysis:
                    job.output_file = putils.default_analysis_file.format(utils.starttime, design_source)
                else:  # We don't have a default output specified
                    pass

            # Make single file with names of each directory where poses can be found
            if output_analysis:
                if poses_file is None:  # Make a default file name
                    putils.make_path(job_paths)
                    # # Remove possible multiple instances of _pose from location in default_output_tuple
                    # scratch_designs = \
                    #     os.path.join(job_paths, putils.default_path_file.format(*default_output_tuple)).split('_pose')
                    # poses_file = f'{scratch_designs[0]}_pose{scratch_designs[-1]}'
                    poses_file = \
                        os.path.join(job_paths, putils.default_path_file.format(*default_output_tuple))

                with open(poses_file, 'w') as f_out:
                    f_out.write('%s\n' % '\n'.join(str(pose_job) for pose_job in success))
                logger.critical(f'The file "{poses_file}" contains the locations of every pose that passed checks/'
                                f'filters for this job. Utilize this file to input these poses in future '
                                f'{putils.program_name} commands such as:'
                                f'\n\t{putils.program_command} MODULE --{flags.poses} {poses_file} ...')

            # Output any additional files for the module
            if job.module in [flags.select_designs, flags.select_sequences]:
                designs_file = \
                    os.path.join(job_paths, putils.default_specification_file.format(*default_output_tuple))
                with open(designs_file, 'w') as f_out:
                    f_out.write('%s\n' % '\n'.join(f'{pose_job}, {design.name}' for pose_job in success
                                                   for design in pose_job.current_designs))
                logger.critical(f'The file "{designs_file}" contains the locations of every design selected by this job'
                                f'. Utilize this file to input these designs in future {putils.program_name} commands '
                                f'such as:\n\t{putils.program_command} MODULE --{flags.specification_file} '
                                f'{designs_file} ...')

            # if job.module == flags.analysis:
            #     # Save Design DataFrame
            #     design_df = pd.DataFrame([result_ for result_ in results if not isinstance(result_, BaseException)])
            #     if args.output:  # Create a new file
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
            if job.module == flags.interface_design:
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
            elif job.module == flags.design:
                # Todo make viable rosettascripts
                design_stage = flags.design
            else:
                design_stage = None

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
                    sbatch_file = \
                        utils.CommandDistributer.distribute(command_file, job.module, out_path=job.sbatch_scripts,
                                                            number_of_commands=len(success))
                else:
                    command_file = utils.write_commands([os.path.join(des.scripts_path, f'{stage}.sh') for des in success],
                                                        out_path=job_paths, name='_'.join(default_output_tuple))
                    sbatch_file = utils.CommandDistributer.distribute(command_file, job.module,
                                                                      out_path=job.sbatch_scripts)
                logger.critical(sbatch_warning)

                if job.module in [flags.interface_design, flags.design] and job.initial_refinement:
                    # We should refine before design
                    refine_file = utils.write_commands([os.path.join(design.scripts_path, f'{flags.refine}.sh')
                                                        for design in success], out_path=job_paths,
                                                       name='_'.join((utils.starttime, flags.refine, design_source)))
                    sbatch_refine_file = utils.CommandDistributer.distribute(refine_file, flags.refine,
                                                                             out_path=job.sbatch_scripts)
                    logger.info(f'Once you are satisfied, enter the following to distribute:\n\tsbatch '
                                f'{sbatch_refine_file}\nTHEN:\n\tsbatch {sbatch_file}')
                else:
                    logger.info(f'Once you are satisfied, enter the following to distribute:\n\tsbatch {sbatch_file}')

        # # test for the size of each of the PoseJob instances
        # if pose_jobs:
        #     print('Average_design_directory_size equals %f' %
        #           (float(psutil.virtual_memory().used) / len(pose_jobs)))

        # print('\n')
        exit(exit_code)

    # def get_sym_entry_from_nanohedra_directory(nanohedra_dir: AnyStr) -> utils.SymEntry.SymEntry:
    #     """Handles extraction of Symmetry info from Nanohedra outputs.
    #
    #     Args:
    #         nanohedra_dir: The path to a Nanohedra master output directory
    #     Raises:
    #         FileNotFoundError: If no nanohedra master log file is found
    #         SymmetryError: If no symmetry is found
    #     Returns:
    #         The SymEntry specified by the Nanohedra docking run
    #     """
    #     log_path = os.path.join(nanohedra_dir, putils.master_log)
    #     try:
    #         with open(log_path, 'r') as f:
    #             for line in f.readlines():
    #                 if 'Nanohedra Entry Number: ' in line:  # "Symmetry Entry Number: " or
    #                     return utils.SymEntry.symmetry_factory.get(int(line.split(':')[-1]))  # sym_map inclusion?
    #     except FileNotFoundError:
    #         raise FileNotFoundError('Nanohedra master directory is malformed. '
    #                                 f'Missing required docking file {log_path}')
    #     raise utils.InputError(f'Nanohedra master docking file {log_path} is malformed. Missing required info'
    #                            ' "Nanohedra Entry Number:"')

    def initialize_entities(structures: Iterable[structure.base.Structure],
                            possibly_new_uniprot_to_prot_data: dict[tuple[str, ...], sql.ProteinMetadata] = None,
                            existing_uniprot_entities: Iterable[wrapapi.UniProtEntity] = None,
                            existing_protein_metadata: Iterable[wrapapi.UniProtEntity] = None) -> \
            dict[tuple[str, ...], sql.ProteinMetadata]:
        """Perform the routine of comparing newly described work to the existing database and handling set up of
        structural and evolutionary data creation

        Args:
            structures: The Structure instances that are being imported for the first time
            possibly_new_uniprot_to_prot_data: A mapping of the possibly required UniProtID entries and their associated
                ProteinProperty. These could already exist but have indicated they should be added
            existing_uniprot_entities: If any UniProtEntity instances are already available, pass them to expedite setup
        """
        # Find existing UniProtEntity table entry instances
        # Get the set of all UniProtIDs
        possibly_new_uniprot_ids = set()
        for uniprot_ids in possibly_new_uniprot_to_prot_data.keys():
            possibly_new_uniprot_ids.update(uniprot_ids)

        # DONE Tod0 move session outside?
        # with job.db.session() as session:  # current_session
        if not existing_uniprot_entities:  # is None:
            # itertools.chain.from_iterable(possibly_new_uniprot_to_prot_data.keys())
            existing_uniprot_entities_stmt = \
                select(wrapapi.UniProtEntity) \
                .where(wrapapi.UniProtEntity.id.in_(possibly_new_uniprot_ids))
            existing_uniprot_entities = session.scalars(existing_uniprot_entities_stmt).all()
            not_new_uniprot_entities = set()
        elif possibly_new_uniprot_ids:
            existing_uniprot_ids = {unp_ent.id for unp_ent in existing_uniprot_entities}
            # Remove the certainly existing from possibly new and query the new
            existing_uniprot_entities_stmt = \
                select(wrapapi.UniProtEntity) \
                .where(wrapapi.UniProtEntity.id.in_(possibly_new_uniprot_ids.difference(existing_uniprot_ids)))
            not_new_uniprot_entities = session.scalars(existing_uniprot_entities_stmt).all()
        else:  # We have existing_uniprot_entities and no possibly_new_uniprot_ids
            # This case is when we are accessing already initialized data
            not_new_uniprot_entities = set()
            # raise NotImplementedError("This wasn't expected to happen:\n"
            #                           f"existing_uniprot_entities={existing_uniprot_entities}\n"
            #                           f"possibly_new_uniprot_ids={possibly_new_uniprot_ids}\n")
            # Todo emit this select when there is a stronger association between the multiple
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
        # Create a container of all UniProtEntity instances to see about evolutionary profile creation
        uniprot_entities = set(not_new_uniprot_entities).union(existing_uniprot_entities)

        # Remove those from the possible that already exist
        # for entity in existing_uniprot_entities:
        #     possibly_new_uniprot_to_prot_data.pop(entity.id, None)
        # Map the existing uniprot_id to UniProtEntity
        existing_uniprot_id_to_unp_entity = {unp_entity.id: unp_entity
                                             for unp_entity in existing_uniprot_entities}
        insert_uniprot_ids = possibly_new_uniprot_ids.difference(existing_uniprot_id_to_unp_entity.keys())
        # logger.debug(f'possibly_new_uniprot_ids {possibly_new_uniprot_ids}')
        # logger.debug(f'insert_uniprot_ids {insert_uniprot_ids}')

        # Insert the remaining UniProtIDs as UniProtEntity entries
        new_uniprot_id_to_unp_entity = {uniprot_id: wrapapi.UniProtEntity(id=uniprot_id)
                                        for uniprot_id in insert_uniprot_ids}

        # Insert new entries
        new_uniprot_entities = new_uniprot_id_to_unp_entity.values()
        session.add_all(new_uniprot_entities)
        uniprot_entities = uniprot_entities.union(new_uniprot_entities)
        # Make a cumulative dictionary for ProteinMetadata ops below
        all_uniprot_id_to_entity = {
            **existing_uniprot_id_to_unp_entity,
            **new_uniprot_id_to_unp_entity,
        }

        # # Attach ProteinMetadata to UniProtEntity entries by UniProtID
        # uniprot_entities = []
        # for uniprot_id, protein_metadata in possibly_new_uniprot_to_prot_data.items():
        #     # Create new entry
        #     new_entity = wrapapi.UniProtEntity(id=uniprot_id)
        #     # Add ProteinProperty to new entry
        #     new_entity.protein_metadata = protein_metadata
        #     uniprot_entities.append(new_entity)

        # Repeat the process for ProteinMetadata
        # This time we insert all new ProteinMetadata and let the UniqueObjectValidatedOnPending recipe
        # associated with PoseJob finish the work of getting the correct objects attached
        possible_entity_id_to_protein_data = \
            {protein_data.entity_id: protein_data
             for protein_data in possibly_new_uniprot_to_prot_data.values()}
        possibly_new_entity_ids = set(possible_entity_id_to_protein_data.keys())
        # all_protein_metadata = possibly_new_uniprot_to_prot_data.values()
        if not existing_protein_metadata:
            existing_protein_metadata_stmt = \
                select(sql.ProteinMetadata) \
                .where(sql.ProteinMetadata.entity_id.in_(possibly_new_entity_ids))
            existing_protein_metadata = session.scalars(existing_protein_metadata_stmt).all()
        elif possibly_new_entity_ids:
            existing_entity_ids = {data.entity_id for data in existing_protein_metadata}
            # Remove the certainly existing from possibly new and query the new
            existing_uniprot_entities_stmt = \
                select(sql.ProteinMetadata) \
                .where(sql.ProteinMetadata.entity_id.in_(possibly_new_entity_ids.difference(existing_entity_ids)))
            not_new_protein_metadata = session.scalars(existing_uniprot_entities_stmt).all()
            existing_protein_metadata = set(not_new_protein_metadata).union(existing_protein_metadata)
        else:
            existing_protein_metadata = set()
            # raise NotImplementedError("This wasn't expected to happen:\n"
            #                           f"existing_protein_metadata={existing_protein_metadata}\n"
            #                           f"possibly_new_entity_ids={possibly_new_entity_ids}\n")
        # # Create a container of all UniProtEntity instances to see about evolutionary profile creation
        # protein_metadata = set(existing_protein_metadata)

        # Map the existing entity_id to ProteinMetadata
        existing_entity_id_to_protein_data = {protein_data.entity_id: protein_data
                                              for protein_data in existing_protein_metadata}
        insert_entity_ids = possibly_new_entity_ids.difference(existing_entity_id_to_protein_data.keys())

        # Insert the remaining UniProtIDs as UniProtEntity entries
        session.add_all(tuple(possible_entity_id_to_protein_data[entity_id]
                              for entity_id in insert_entity_ids))

        # Fix the "possible" protein_property with true set of all (now) existing
        possible_entity_id_to_uniprot_ids = \
            {protein_data.entity_id: uniprot_ids
             for uniprot_ids, protein_data in possibly_new_uniprot_to_prot_data.items()}

        # Remove any pre-existing ProteinMetadata for the correct addition to UniProtEntity
        if possibly_new_uniprot_to_prot_data:
            for entity_id, protein_data in existing_entity_id_to_protein_data.items():
                possibly_new_uniprot_to_prot_data.pop(possible_entity_id_to_uniprot_ids[entity_id])

        # Attach UniProtEntity to new ProteinMetadata by UniProtID
        for uniprot_ids, protein_metadata in possibly_new_uniprot_to_prot_data.items():
            # Create the ordered_list of UniProtIDs (UniProtEntity) on ProteinMetadata entry
            protein_metadata.uniprot_entities.extend(all_uniprot_id_to_entity[uniprot_id]
                                                     for uniprot_id in uniprot_ids)

        # Seal all gaps in data by mapping all UniProtIDs to all ProteinMetadata
        all_uniprot_to_prot_data = {
            possible_entity_id_to_uniprot_ids[entity_id]: protein_data
            for entity_id, protein_data in existing_entity_id_to_protein_data.items()}
        all_uniprot_to_prot_data.update(possibly_new_uniprot_to_prot_data)

        # Set up common Structure/Entity resources
        info_messages = []
        if job.design.evolution_constraint:
            evolution_instructions = \
                job.setup_evolution_constraint(uniprot_entities=uniprot_entities)
            #                                    entities=all_entities)
            # load_resources = True if evolution_instructions else False
            info_messages.extend(evolution_instructions)

        if job.preprocessed:
            # Don't perform refinement or loop modeling, this has already been done or isn't desired
            # args.orient, args.refine = True, True  # Todo make part of argparse?
            # Move the oriented Entity.file_path (should be the ASU) to the respective directory
            putils.make_path(job.refine_dir)
            putils.make_path(job.full_model_dir)
            for entity in structures:  # entities:
                shutil.copy(entity.file_path, job.refine_dir)
                shutil.copy(entity.file_path, job.full_model_dir)

            # Report to JobResources the results after skipping set up
            job.initial_refinement = job.initial_loop_model = True
        else:
            preprocess_instructions, job.initial_refinement, job.initial_loop_model = \
                job.structure_db.preprocess_structures_for_design(structures, script_out_path=job.sbatch_scripts)
            if info_messages and preprocess_instructions:
                info_messages += ['The following can be run at any time regardless of evolutionary script progress']
            info_messages += preprocess_instructions

        if info_messages:
            # Entity processing commands are needed
            logger.critical(sbatch_warning)
            for message in info_messages:
                logger.info(message)
            print('\n')
            logger.info(resubmit_command_message)
            terminate(output=False)

        for uniprot_ids, protein_metadata in possibly_new_uniprot_to_prot_data.items():
            protein_metadata.pre_refine = job.initial_refinement
            protein_metadata.pre_loop_model = job.initial_loop_model

            # After completion of sbatch, the next time command is entered program will proceed
        #     else:
        #         # We always prepare info_messages when jobs should be run
        #         raise utils.InputError("This shouldn't have happened. 'info_messages' can't be False here...")

        # Finalize additions to the database
        session.commit()
        # # Ensure all_entities are symmetric. As of now, all orient_structures returns are the symmetrized structure
        # for entity in [entity for structure in all_structures for entity in structure.entities]:
        #     entity.make_oligomer(symmetry=entity.symmetry)
        return all_uniprot_to_prot_data

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
            #           % (putils.ex_path('pymol'), putils.program_command.replace('python ', ''),
            #              putils.ex_path('pose_directory'), SDUtils.ex_path('DataFrame.csv'),
            #              putils.ex_path('design.paths')))
        # else:  # Print the full program readme and exit
        #     guide.print_guide()
        #     exit()
    elif args.setup:
        guide.setup_instructions()
        exit()
    elif args.help:
        pass  # Let the entire_parser handle their formatting
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
        exit(f'\nSuspending run. Found flag(s) that are not recognized program wide: {", ".join(additional_args)}\n'
             'Please correct (try adding --help if unsure), and resubmit your command\n')

    if remove_dummy:  # Remove the dummy input
        del args.file
    # -----------------------------------------------------------------------------------------------------------------
    #  Find base symdesign_directory and check for proper set up of program i/o
    # -----------------------------------------------------------------------------------------------------------------
    # Check if output already exists or --overwrite is provided
    if args.output_file and os.path.exists(args.output_file) and args.module not in flags.analysis \
            and not args.overwrite:
        exit(f'The specified output file "{args.output_file}" already exists, this will overwrite your old '
             f'data! Please specify a new name with with -Of/--{flags.output_file}, '
             'or append --overwrite to your command')
    symdesign_directory = None
    # If output exists, get the base SymDesign dir -> symdesign_directory
    # Todo make output_directory separate from existing symdesign_directory storage location
    if args.output_directory:
        if os.path.exists(args.output_directory) and not args.overwrite:
            exit(f'The specified output directory "{args.output_directory}" already exists, this will overwrite '
                 f'your old data! Please specify a new name with with -Od/--{flags.output_directory}, '
                 '--prefix or --suffix, or append --overwrite to your command')
        else:
            symdesign_directory = args.output_directory
            putils.make_path(symdesign_directory)
    else:
        symdesign_directory = utils.get_program_root_directory(
            (args.directory or (args.project or args.single or [None])[0] or os.getcwd()))

    project_name = None
    """Used to set a leading string on all new PoseJob paths"""
    if symdesign_directory is None:  # We found no directory from the input
        # By default, assume new input and make in the current directory
        new_symdesign_output = True
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
                            new_symdesign_output = False
                    else:
                        # Set file basename as "project_name" in case poses are being integrated for the first time
                        # In this case, pose names might be the same, so we take the basename of the first file path as
                        # the name to discriminate between separate files
                        project_name = os.path.splitext(os.path.basename(file))[0]
                break
        # Ensure the base directory is made if it is indeed new and in os.getcwd()
        putils.make_path(symdesign_directory)
    else:
        new_symdesign_output = False
    # -----------------------------------------------------------------------------------------------------------------
    #  Process JobResources which holds shared program objects and command-line arguments
    # -----------------------------------------------------------------------------------------------------------------
    job = job_resources_factory.get(program_root=symdesign_directory, arguments=args)
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
        else:  # if job.module in decoy_modules:
            pass  # exit()
        exit()

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
    #         and job.number == sys.maxsize and not args.total:
    #     # Change default number to a single sequence/pose when not doing a total selection
    #     job.number = 1
    elif job.module in flags.cluster_poses and job.number == sys.maxsize:
        # Change default number to a single pose
        job.number = 1
    elif job.module == flags.nanohedra:
        if not job.sym_entry:
            raise utils.InputError(f'When running {flags.nanohedra}, the argument -e/--entry/--{flags.sym_entry} is '
                                   'required')
    elif job.module == flags.generate_fragments:  # Ensure we write fragments out
        job.write_fragments = True
    else:
        if job.module not in [flags.interface_design, flags.design, flags.refine, flags.optimize_designs,
                              flags.interface_metrics, flags.analysis, flags.process_rosetta_metrics,
                              flags.generate_fragments, flags.orient, flags.expand_asu,
                              flags.rename_chains, flags.check_clashes]:
            # , 'custom_script', 'find_asu', 'status', 'visualize'
            # We have no module passed. Print the guide and exit
            guide.print_guide()
            exit()
        # else:
        #     # Set up design directories
        #     pass
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
    #  Initialize the db.session, set into job namespace
    # -----------------------------------------------------------------------------------------------------------------
    results = []
    exceptions = []
    with job.db.session(expire_on_commit=False) as session:
        job.current_session = session
    # -----------------------------------------------------------------------------------------------------------------
    #  Grab all Poses (PoseJob instance) from either database, directory, project, single, or file
    # -----------------------------------------------------------------------------------------------------------------
        file_paths: list[AnyStr] | None = None
        # pose_jobs hold jobs with specific poses
        # list[PoseJob] for an establishes pose
        # list[tuple[Structure, Structure]] for a nanohedra docking job
        pose_jobs: list[PoseJob] | list[tuple[Any, Any]] = []
        low = high = low_range = high_range = None
        logger.info(f'Setting up input for {job.module}')
        if job.module == flags.nanohedra:
            # logger.info(f'Setting up inputs for {job.module.title()} docking')
            job.sym_entry.log_parameters()
            # # Make master output directory. sym_entry is required, so this won't fail v
            # if args.output_directory is None:
            #     # job.output_directory = os.path.join(job.projects, f'NanohedraEntry{sym_entry.number}_BUILDING-BLOCKS_Poses')
            #     job.output_directory = job.projects
            #     putils.make_path(job.output_directory)
            # Transform input entities to canonical orientation and return their ASU
            all_structures = []
            grouped_structures: list[list[Model | Entity]] = []
            # Set up variables for the correct parsing of provided file paths
            by_file1 = by_file2 = False
            eventual_structure_names1 = eventual_structure_names2 = None
            idx = 1
            if args.oligomer1:
                by_file1 = True
                logger.critical(f'Ensuring provided file(s) at {args.oligomer1} are oriented for Nanohedra Docking')
                if '.pdb' in args.oligomer1:
                    pdb1_filepaths = [args.oligomer1]
                else:
                    extension = '.pdb*'
                    pdb1_filepaths = utils.get_directory_file_paths(args.oligomer1, extension=extension)
                    if not pdb1_filepaths:
                        logger.warning(f'Found no {extension} files at {args.oligomer1}')
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
                save_query = validate_input_return_response_value(
                    'Do you want to save your PDB query to a local file?', {'y': True, 'n': False})
                print(f'\nStarting PDB query for component {idx}\n')
                structure_names1 = retrieve_pdb_entries_by_advanced_query(save=save_query, entity=True)
                # Make all names lowercase
                structure_names1 = list(map(str.lower, structure_names1))
            else:
                raise RuntimeError('This should be impossible with mutually exclusive argparser group')

            # Select entities, orient them, then load each Structure for further database processing
            grouped_structures.append(job.structure_db.orient_structures(structure_names1,
                                                                         symmetry=job.sym_entry.group1,
                                                                         by_file=by_file1))
            single_component_design = False
            structure_names2 = []
            idx = 2
            if args.oligomer2:
                if args.oligomer1 != args.oligomer2:  # See if they are the same input
                    by_file2 = True
                    logger.critical(f'Ensuring provided file(s) at {args.oligomer2} are oriented for Nanohedra Docking')
                    if '.pdb' in args.oligomer2:
                        pdb2_filepaths = [args.oligomer2]
                    else:
                        extension = '.pdb*'
                        pdb2_filepaths = utils.get_directory_file_paths(args.oligomer2, extension=extension)
                        if not pdb2_filepaths:
                            logger.warning(f'Found no {extension} files at {args.oligomer2}')

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
                save_query = validate_input_return_response_value(
                    'Do you want to save your PDB query to a local file?', {'y': True, 'n': False})
                print(f'\nStarting PDB query for component {idx}\n')
                structure_names2 = retrieve_pdb_entries_by_advanced_query(save=save_query, entity=True)
                # Make all names lowercase
                structure_names2 = list(map(str.lower, structure_names2))
            else:
                single_component_design = True

            # Select entities, orient them, then load each Structure for further database processing
            grouped_structures.append(job.structure_db.orient_structures(structure_names2,
                                                                         symmetry=job.sym_entry.group2,
                                                                         by_file=by_file2))
            # Get api wrapper
            retrieve_stride_info = wrapapi.api_database_factory().stride.retrieve_data
            # Initialize the local database
            # # Populate all_entities to set up sequence dependent resources
            # all_entities = []
            possibly_new_uniprot_to_prot_metadata = {}
            # Todo expand the definition of SymEntry/Entity to include
            #  specification of T:{T:{C3}{C3}}{C1}
            #  where an Entity is composed of multiple Entity (Chain) instances
            # symmetry_map = job.sym_entry.groups if job.sym_entry else repeat(None)
            for structures, symmetry in zip(grouped_structures, job.sym_entry.groups):  # symmetry_map):
                if not structures:  # Useful in a case where symmetry groups are the same or group is None
                    continue
                for structure in structures:
                    # all_entities.extend(structure.entities)
                    for entity in structure.entities:
                        # Try to get the already parsed secondary structure informatino
                        parsed_secondary_structure = retrieve_stride_info(name=entity.name)
                        if parsed_secondary_structure:
                            entity.secondary_structure = parsed_secondary_structure
                        else:
                            entity.stride(to_file=job.api_db.stride.path_to(entity.name))
                            # entity.calculate_secondary_structure()
                        protein_metadata = sql.ProteinMetadata(
                            entity_id=entity.name,
                            reference_sequence=entity.reference_sequence,
                            n_terminal_helix=entity.is_termini_helical(),
                            c_terminal_helix=entity.is_termini_helical('c'),
                            thermophilicity=entity.thermophilicity,
                            symmetry_group=symmetry)
                        # # Set the Entity with .metadata attribute to fetch in fragdock()
                        # entity.metadata = protein_metadata
                        # for uniprot_id in entity.uniprot_ids:
                        try:
                            ''.join(entity.uniprot_ids)
                        except TypeError:  # Uniprot_ids is (None,)
                            uniprot_ids = (entity.name,)
                        except AttributeError:  # Unable to retrieve
                            uniprot_ids = (entity.name,)
                        else:
                            uniprot_ids = entity.uniprot_ids

                        if uniprot_ids in possibly_new_uniprot_to_prot_metadata:
                            # This Entity already found for processing and we shouldn't have duplicates
                            raise RuntimeError(f"This error wasn't expected to occur.{putils.report_issue}")
                        else:  # Process for persistent state
                            possibly_new_uniprot_to_prot_metadata[uniprot_ids] = protein_metadata

                all_structures.extend(structures)

            # Write new data to the database
            # with job.db.session(expire_on_commit=False) as session:
            all_uniprot_to_prot_data = \
                initialize_entities(all_structures, possibly_new_uniprot_to_prot_metadata)
            # Correct existing ProteinMetadata
            for structures in grouped_structures:
                for structure in structures:
                    for entity in structure.entities:
                        # Set the Entity with .metadata attribute to fetch in fragdock()
                        entity.metadata = all_uniprot_to_prot_data[entity.uniprot_ids]

            # Make all possible structure pairs given input entities by finding entities from entity_names
            # Using combinations of directories with .pdb files
            if single_component_design:
                logger.info('No additional entities requested for docking, treating as single component')
                # structures1 = [entity for entity in all_entities if entity.name in structures1]
                # ^ doesn't work as entity_id is set in orient_structures, but structure name is entry_id
                pose_jobs.extend(combinations(all_structures, 2))
            else:
                # # v doesn't work as entity_id is set in orient_structures, but structure name is entry_id
                # # structures1 = [entity for entity in all_entities if entity.name in structures1]
                # # structures2 = [entity for entity in all_entities if entity.name in structures2]
                pose_jobs = []
                for structures1, structures2 in combinations(grouped_structures, 2):
                    pose_jobs.extend(product(structures1, structures2))

            job.location = f'NanohedraEntry{job.sym_entry.number}'  # Used for terminate()
            if not pose_jobs:  # No pairs were located
                exit('No docking pairs were located from your input. Please ensure that your flags are as intended.'
                     f'{putils.issue_submit_warning}')
            elif job.distribute_work:
                # Write all commands to a file and use sbatch
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
                            for idx, (model1, model2) in enumerate(pose_jobs)]
                # logger.debug([list2cmdline(cmd) for cmd in commands])
                # utils.write_shell_script(list2cmdline(commands), name=flags.nanohedra, out_path=job.job_paths)
                terminate(results=commands)
        else:
            if args.range:
                try:
                    low, high = map(float, args.range.split('-'))
                except ValueError:  # We didn't unpack correctly
                    raise ValueError('The input flag -r/--range argument must take the form "LOWER-UPPER"')
                n_files = len(file_paths)
                low_range, high_range = int((low / 100) * n_files), int((high / 100) * n_files)
                if low_range < 0 or high_range > n_files:
                    raise ValueError('The input flag --range argument is outside of the acceptable bounds [0-100]')
                logger.info(f'Selecting poses within range: {low_range if low_range else 1}-{high_range}')
                range_slice = slice(low_range, high_range)
            else:
                range_slice = slice(None)

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
            #                             for pose in file_paths[range_slice]]
            #         # Copy the master nanohedra log
            #         project_designs = \
            #             os.path.join(job.projects, f'{os.path.basename(job.nanohedra_root)}')  # _{putils.pose_directory}')
            #         if not os.path.exists(os.path.join(project_designs, putils.master_log)):
            #             putils.make_path(project_designs)
            #             shutil.copy(os.path.join(job.nanohedra_root, putils.master_log), project_designs)
            if args.poses or args.specification_file or args.project or args.single:
                # Use sqlalchemy database selection to find the requested work
                pose_identifiers = []
                if args.specification_file or args.poses:
                    # Todo no --file and --specification-file at the same time
                    # These poses are already included in the "program state"
                    if not args.directory:  # Todo react .directory to program operation inside or in dir with SymDesignOutput
                        raise utils.InputError(f'A --{flags.directory} must be provided when using '
                                               f'--{flags.specification_file}')
                    # Todo, combine this with collect_designs
                    #  this works for file locations as well! should I have a separate mechanism for each?
                    if args.poses:
                        job.location = args.poses
                        for specification_file in args.poses:
                            # Returns list of _designs and _directives
                            pose_identifiers.extend(utils.PoseSpecification(specification_file).pose_identifiers)
                    else:
                        job.location = args.specification_file
                        designs = []
                        directives = []
                        for specification_file in args.specification_file:
                            # Returns list of _designs and _directives
                            _pose_identifiers, _designs, _directives = \
                                zip(*utils.PoseSpecification(specification_file).get_directives())
                            pose_identifiers.extend(_pose_identifiers)
                            designs.extend(_designs)
                            directives.extend(_directives)

                    # with job.db.session(expire_on_commit=False) as session:
                    fetch_jobs_stmt = select(PoseJob).where(PoseJob.pose_identifier.in_(pose_identifiers))
                    pose_jobs = list(session.scalars(fetch_jobs_stmt))
                    # pose_jobs = list(job.current_session.scalars(fetch_jobs_stmt))
                    # for specification_file in args.specification_file:
                    #     pose_jobs.extend(
                    #         [PoseJob.from_directory(pose_identifier, root=job.projects,
                    #                                 specific_designs=designs,
                    #                                 directives=directives)
                    #          for pose_identifier, designs, directives in
                    #          utils.PoseSpecification(specification_file).get_directives()])

                    # Check all jobs that were checked out against those that were requested
                    checked_out_identifiers = {pose_job.pose_identifier for pose_job in pose_jobs}
                    missing_input_identifiers = checked_out_identifiers.difference(pose_identifiers)
                    if missing_input_identifiers:
                        logger.warning(
                            "Couldn't find the following identifiers:\n%s" % '\n'.join(missing_input_identifiers))

                    if args.specification_file:
                        for pose_job, _designs, _directives in zip(pose_jobs, designs, directives):
                            pose_job.use_specific_designs(_designs, _directives)
                else:
                    paths, job.location = utils.collect_designs(projects=args.project, singles=args.single)
                    #                                                  directory = symdesign_directory,
                    if paths:  # There are files present
                        # pose_jobs = [PoseJob.from_directory(path) for path in file_paths[range_slice]]
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
                    fetch_jobs_stmt = select(PoseJob).where(PoseJob.pose_identifier.in_(pose_identifiers))
                    pose_jobs = list(session.scalars(fetch_jobs_stmt))
            elif select_from_directory:
                # Can make an empty pose_jobs when the program_root is args.directory
                job.location = args.directory
                pass
            else:  # args.file or args.directory
                file_paths, job.location = utils.collect_designs(files=args.file, directory=args.directory)
                if file_paths:
                    pose_jobs = [PoseJob.from_path(path, project=project_name)
                                 for path in file_paths[range_slice]]
            if not pose_jobs:
                raise utils.InputError(f"No {PoseJob.__name__}'s found at location '{job.location}'")
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
            retrieve_stride_info = wrapapi.api_database_factory().stride.retrieve_data
            possibly_new_uniprot_to_prot_metadata: dict[tuple[str, ...], sql.ProteinMetadata] = {}
            existing_uniprot_ids = set()
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
                        if protein_metadata is None:  # uniprot_ids in possibly_new_uniprot_to_prot_metadata:
                            # Try to get the already parsed secondary structure informatino
                            parsed_secondary_structure = retrieve_stride_info(name=entity.name)
                            if parsed_secondary_structure:
                                entity.secondary_structure = parsed_secondary_structure
                            else:
                                entity.stride(to_file=job.api_db.stride.path_to(entity.name))
                                # entity.calculate_secondary_structure()
                            # Process for persistent state
                            protein_metadata = sql.ProteinMetadata(
                                entity_id=entity.name,
                                reference_sequence=entity.reference_sequence,
                                n_terminal_helix=entity.is_termini_helical(),
                                c_terminal_helix=entity.is_termini_helical('c'),
                                thermophilicity=entity.thermophilicity,
                                symmetry_group=symmetry
                                # # Todo there could be no sym_entry, the use the entity.symmetry
                                # symmetry=entity.symmetry
                            )
                            # for uniprot_id in entity.uniprot_ids:
                            #     if uniprot_id in possibly_new_uniprot_to_prot_metadata:
                            #         # This Entity already found for processing
                            #         pass
                            #     else:  # Process for persistent state
                            #         possibly_new_uniprot_to_prot_metadata[uniprot_id] = protein_metadata

                            possibly_new_uniprot_to_prot_metadata[uniprot_ids] = protein_metadata
                            preprocess_entities_by_symmetry[symmetry].append(entity)
                        # else:  # This Entity already found for processing
                        #     pass  # protein_metadata = protein_metadata
                        # # Create EntityData
                        # # entity_data.append(sql.EntityData(pose=pose_job,
                        # sql.EntityData(pose=pose_job,
                        #                meta=protein_metadata
                        #                )
                    # # Update PoseJob
                    # pose_job.entity_data = entity_data
                    pose_jobs_to_commit.append(pose_job)
                else:  # PoseJob is initialized
                    # Add each UniProtEntity to existing_uniprot_entities to limit work
                    for data in pose_job.entity_data:
                        existing_protein_metadata.add(data.meta)
                        for uniprot_entity in data.meta.uniprot_entities:
                            if uniprot_entity.id not in existing_uniprot_ids:
                                existing_uniprot_entities.add(uniprot_entity)
                        # meta = data.meta
                        # if meta.uniprot_id in possibly_new_uniprot_to_prot_metadata:
                        #     existing_uniprot_entities.add(meta.uniprot_entity)
            for idx in reversed(remove_pose_jobs):
                pose_jobs.pop(idx)

            if not pose_jobs:
                raise utils.InputError(f"No viable {PoseJob.__name__}'s found at location '{job.location}'")
            # all_structures = []
            # Populate all_entities to set up sequence dependent resources
            all_entities = []
            # Orient entities, then load each entity to all_structures for further database processing
            for symmetry, entities in preprocess_entities_by_symmetry.items():
                if not entities:  # useful in a case where symmetry groups are the same or group is None
                    continue
                # all_entities.extend(entities)
                # job.structure_db.orient_structures(
                #     [entity.name for entity in entities], symmetry=symmetry)
                # Can't do this ^ as structure_db.orient_structures sets .name, .symmetry, and .file_path on each Entity
                all_entities.extend(job.structure_db.orient_structures(
                    [entity.name for entity in entities], symmetry=symmetry))
                # Todo orient Entity individually, which requires symmetric oligomer be made
                #  This could be found from Pose._assign_pose_transformation() or new meachanism
                #  Where oligomer is deduced from available surface fragment overlap with the specified symmetry...
                #  job.structure_db.orient_entities(entities, symmetry=symmetry)

            # Deal with new data compared to existing entries
            all_uniprot_to_prot_data = \
                initialize_entities(all_entities, possibly_new_uniprot_to_prot_metadata,
                                    existing_uniprot_entities=existing_uniprot_entities,
                                    existing_protein_metadata=existing_protein_metadata)
            # Todo does all_uniprot_to_prot_data need to be returned or can I update in place?
            # Write new data to the database with correct unique entries
            # with job.db.session(expire_on_commit=False) as session:
            for pose_job in pose_jobs_to_commit:
                pose_job.entity_data.extend(
                    sql.EntityData(meta=all_uniprot_to_prot_data[entity.uniprot_ids])
                    for entity in pose_job.initial_model.entities)
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
                existing_pose_jobs = list(session.scalars(fetch_jobs_stmt))

                existing_pose_identifiers = [pose_job.pose_identifier for pose_job in existing_pose_jobs]
                pose_jobs = []
                for pose_job in pose_jobs_to_commit:
                    if pose_job.new_pose_identifier not in existing_pose_identifiers:
                        pose_jobs.append(pose_job)
                        # session.add(pose_job)

                session.add_all(pose_jobs)
                session.commit()
                pose_jobs += existing_pose_jobs

            if args.multi_processing:  # and not args.skip_master_db:
                logger.debug('Loading Database for multiprocessing fork')
                # Todo set up a job based data acquisition as this takes some time and loading everythin isn't necessary!
                job.structure_db.load_all_data()
                job.api_db.load_all_data()
            # # Set up in series
            # for pose_job in pose_jobs:
            #     # pose.initialize_structure_attributes(pre_refine=job.initial_refinement,
            #     #                                      pre_loop_model=job.initial_loop_model)
            #     pose_job.pre_refine = job.initial_refinement
            #     pose_job.pre_loop_model = job.initial_loop_model

            logger.info(f'Found {len(pose_jobs)} unique poses from provided input location "{job.location}"')
            if not job.debug and not job.skip_logging:
                representative_pose_job = next(iter(pose_jobs))
                if representative_pose_job.log_path:
                    logger.info(f'All design specific logs are located in their corresponding directories\n\tEx: '
                                f'{representative_pose_job.log_path}')
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
                logger.info("Modeling will occur in this process, ensure you don't lose connection to the shell")

        if job.multi_processing:
            logger.info(f'Starting multiprocessing using {job.cores} cores')
        else:
            logger.info(f'Starting processing. To increase processing speed, '
                        f'use --{flags.multi_processing} during submission')

        job.calculate_memory_requirements(len(pose_jobs))
    # -----------------------------------------------------------------------------------------------------------------
    #  Perform the specified protocol
    # -----------------------------------------------------------------------------------------------------------------
        if args.module == flags.protocol:  # Use args.module as job.module is set as first in protocol
            run_on_pose_job = (
                flags.orient,
                flags.expand_asu,
                flags.rename_chains,
                flags.check_clashes,
                flags.generate_fragments,
                flags.interface_metrics,
                flags.optimize_designs,
                flags.refine,
                flags.interface_design,
                flags.design,
                flags.analysis,
                flags.process_rosetta_metrics,
                flags.nanohedra
            )
            returns_pose_jobs = (
                flags.nanohedra,
                flags.select_poses,
                flags.select_designs,
                flags.select_sequences
            )

            # Universal protocol runner
            for idx, protocol_name in enumerate(job.modules, 1):
                logger.info(f'Starting protocol {idx}: {protocol_name}')
                # Update this mechanism with each module
                job.module = protocol_name

                # Fetch the specified protocol with python acceptable naming
                protocol = getattr(protocols, protocol_name.replace('-', '_'))
                # Figure out how the job should be set up
                if protocol_name in run_on_pose_job:  # Single poses
                    if job.multi_processing:
                        results_ = utils.mp_map(protocol, pose_jobs, processes=job.cores)
                    else:
                        results_ = []
                        for pose_job in pose_jobs:
                            results_.append(protocol(pose_job))
                else:  # Collection of poses
                    results_ = protocol(pose_jobs)

                # Handle any returns that require particular treatment
                if protocol_name in returns_pose_jobs:
                    results = []
                    if results_:  # Not an empty list
                        if isinstance(results_[0], list):  # In the case of nanohedra
                            for result in results_:
                                results.extend(result)
                        else:
                            results.extend(results_)
                    pose_jobs = results
                else:
                    results = results_
                # elif putils.cluster_poses:  # Returns None
                #    pass

                # Update the current state of protocols and exceptions
                exceptions.extend(parse_results_for_exceptions(results))
                # # Retrieve any program flags necessary for termination
                # terminate_kwargs.update(**terminate_options.get(protocol_name, {}))
        # -----------------------------------------------------------------------------------------------------------------
        #  Run a single submodule
        # -----------------------------------------------------------------------------------------------------------------
        else:
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
            #             logger.info('Checking for %d files based on --number_of_designs flag' % args.number_of_designs)
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
                    exit(f'No .pdb files found at location "{job.location}"')

                # if len(sys.argv) > 2:
                #     low, high = map(float, sys.argv[2].split('-'))
                #     low_range, high_range = int((low / 100) * len(files)), int((high / 100) * len(files))
                #     if low_range < 0 or high_range > len(files):
                #         raise ValueError('The input range is outside of the acceptable bounds [0-100]')
                #     print('Selecting Designs within range: %d-%d' % (low_range if low_range else 1, high_range))
                # else:
                print(low_range, high_range)
                print(file_paths)
                for idx, file in enumerate(files[range_slice], low_range + 1):
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
                            logger.critical(f"The module memory_profiler isn't installed {profile_error}")
                        exit('Done profiling')

                if args.multi_processing:
                    results = utils.mp_map(protocol, pose_jobs, processes=job.cores)
                else:
                    for pose_job in pose_jobs:
                        results.append(protocol(pose_job))
    # -----------------------------------------------------------------------------------------------------------------
    #  Finally, run terminate(). This formats output parameters and reports on exceptions
    # -----------------------------------------------------------------------------------------------------------------
        # # Reset the current_session
        # job.current_session = None
        terminate(results=results)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nRun Ended By KeyboardInterrupt\n')
        exit(2)
