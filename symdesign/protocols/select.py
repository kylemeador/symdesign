import logging
import os
import shutil
from glob import glob
from itertools import repeat, count
from typing import Any, Iterable, Sequence

import pandas as pd

from symdesign import flags, metrics, protocols, resources, utils
from symdesign.resources.job import job_resources_factory
from symdesign.resources.query.utils import input_string, boolean_choice
import symdesign.utils.path as putils
from symdesign.structure.model import Model, Pose
from symdesign.structure.sequence import generate_mutations, find_orf_offset, write_sequences
from symdesign.third_party.DnaChisel.dnachisel.DnaOptimizationProblem.NoSolutionError import NoSolutionError

logger = logging.getLogger(__name__)


def poses(pose_directories: Iterable[protocols.protocols.PoseDirectory]) -> list[protocols.protocols.PoseDirectory]:
    job = job_resources_factory.get()

    if job.specification_file:  # Figure out poses from a specification file, filters, and weights
        loc_result = [(pose_directory, design) for pose_directory in pose_directories
                      for design in pose_directory.specific_designs]
        total_df = protocols.load_total_dataframe(pose_directories, pose=True)
        selected_poses_df = \
            metrics.prioritize_design_indices(total_df.loc[loc_result, :], filter=job.filter, weight=job.weight,
                                              protocol=job.protocol, function=job.weight_function)
        # Remove excess pose instances
        number_chosen = 0
        selected_indices, selected_poses = [], set()
        for pose_directory, design in selected_poses_df.index.to_list():
            if pose_directory not in selected_poses:
                selected_poses.add(pose_directory)
                selected_indices.append((pose_directory, design))
                number_chosen += 1
                if number_chosen == job.number:
                    break

        # Specify the result order according to any filtering and weighting
        # Drop the specific design for the dataframe. If they want the design, they should run select_sequences
        save_poses_df = \
            selected_poses_df.loc[selected_indices, :].droplevel(-1)  # .droplevel(0, axis=1).droplevel(0, axis=1)
        # # convert selected_poses to PoseDirectory objects
        # selected_poses = [pose_directory for pose_directory in pose_directories if pose_dir_name in selected_poses]
    elif job.total:  # Figure out poses from file/directory input, filters, and weights
        total_df = protocols.load_total_dataframe(pose_directories, pose=True)
        if job.protocol:  # Todo adapt to protocol column not in Trajectories right now...
            group_df = total_df.groupby(putils.protocol)
            df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
                           keys=list(zip(group_df.groups, repeat('mean'))))
        else:
            df = pd.concat([total_df], axis=1, keys=['pose', 'metric'])
        # Figure out designs from dataframe, filters, and weights
        selected_poses_df = metrics.prioritize_design_indices(df, filter=job.filter, weight=job.weight,
                                                              protocol=job.protocol, function=job.weight_function)
        # Remove excess pose instances
        number_chosen = 0
        selected_indices, selected_poses = [], set()
        for pose_directory, design in selected_poses_df.index.to_list():
            if pose_directory not in selected_poses:
                selected_poses.add(pose_directory)
                selected_indices.append((pose_directory, design))
                number_chosen += 1
                if number_chosen == job.number:
                    break

        # Specify the result order according to any filtering and weighting
        # Drop the specific design for the dataframe. If they want the design, they should run select_sequences
        save_poses_df = \
            selected_poses_df.loc[selected_indices, :].droplevel(-1)  # .droplevel(0, axis=1).droplevel(0, axis=1)
        # # convert selected_poses to PoseDirectory objects
        # selected_poses = [pose_directory for pose_directory in pose_directories if pose_dir_name in selected_poses]
    elif job.dataframe:  # Figure out poses from a pose dataframe, filters, and weights
        if not pose_directories:  # not job.directory:
            logger.critical(f'If using a --{flags.dataframe} for selection, you must include the directory where '
                            f'the designs are located in order to properly select designs. Please specify '
                            f'-d/--{flags.directory} with your command')
            exit(1)

        total_df = pd.read_csv(job.dataframe, index_col=0, header=[0, 1, 2])
        total_df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)

        selected_poses_df = metrics.prioritize_design_indices(total_df, filter=job.filter, weight=job.weight,
                                                              protocol=job.protocol, function=job.weight_function)
        # Only drop excess columns as there is no MultiIndex, so no design in the index
        save_poses_df = selected_poses_df.droplevel(0, axis=1).droplevel(0, axis=1)
        program_root = job.program_root
        selected_poses = [protocols.PoseDirectory.from_pose_id(pose, root=program_root)
                          for pose in save_poses_df.index.to_list()]
    else:  # Generate design metrics on the spot
        raise NotImplementedError('This functionality is currently broken')
        selected_poses, selected_poses_df, total_df = [], pd.DataFrame(), pd.DataFrame()
        logger.debug('Collecting designs to sort')
        if job.metric == 'score':
            metric_design_dir_pairs = [(des_dir.score, des_dir.path) for des_dir in pose_directories]
        elif job.metric == 'fragments_matched':
            metric_design_dir_pairs = [(des_dir.number_of_fragments, des_dir.path)
                                       for des_dir in pose_directories]
        else:  # This is impossible with the argparse options
            raise NotImplementedError(f'The metric "{job.metric}" is not supported!')
            metric_design_dir_pairs = []

        logger.debug(f'Sorting designs according to "{job.metric}"')
        metric_design_dir_pairs = [(score, path) for score, path in metric_design_dir_pairs if score]
        sorted_metric_design_dir_pairs = sorted(metric_design_dir_pairs, key=lambda pair: (pair[0] or 0),
                                                reverse=True)
        top_designs_string = \
            f'Top ranked Designs according to {job.metric}:\n\t{job.metric.title()}\tDesign\n\t%s'
        results_strings = ['%.2f\t%s' % tup for tup in sorted_metric_design_dir_pairs]
        logger.info(top_designs_string % '\n\t'.join(results_strings[:500]))
        if len(pose_directories) > 500:
            design_source = f'top_{job.metric}'
            # default_output_tuple = (utils.starttime, job.module, design_source)
            putils.make_path(job.all_scores)
            designs_file = \
                os.path.join(job.all_scores, f'{utils.starttime}_{job.module}_{design_source}_pose.scores')
            with open(designs_file, 'w') as f:
                f.write(top_designs_string % '\n\t'.join(results_strings))
            logger.info(f'Stdout performed a cutoff of ranked Designs at ranking 500. See the output design file '
                        f'"{designs_file}" for the remainder')

        terminate(output=False)
    # else:
    #     logger.critical('Missing a required method to provide or find metrics from %s. If you meant to gather '
    #                     'metrics from every pose in your input specification, ensure you include the --global '
    #                     'argument' % putils.program_output)
    #     exit()

    if job.total and job.save_total:
        total_df_filename = os.path.join(program_root, 'TotalPosesTrajectoryMetrics.csv')
        total_df.to_csv(total_df_filename)
        logger.info(f'Total Pose/Designs DataFrame was written to: {total_df_filename}')

    logger.info(f'{len(save_poses_df)} poses were selected')
    if len(save_poses_df) != len(total_df):
        if job.filter or job.weight:
            new_dataframe = os.path.join(program_root, f'{utils.starttime}-{"Filtered" if job.filter else ""}'
                                                       f'{"Weighted" if job.weight else ""}PoseMetrics.csv')
        else:
            new_dataframe = os.path.join(program_root, f'{utils.starttime}-PoseMetrics.csv')
        save_poses_df.to_csv(new_dataframe)
        logger.info(f'New DataFrame with selected poses was written to: {new_dataframe}')

    # # Select by clustering analysis
    # if job.cluster:
    # Sort results according to clustered poses if clustering exists
    if job.cluster.map:
        if os.path.exists(job.cluster.map):
            cluster_map = utils.unpickle(job.cluster.map)
        else:
            raise FileNotFoundError(f'No --{flags.cluster_map} "{job.cluster.map}" file was found')

        # Make the selected_poses into strings
        selected_pose_strs = list(map(str, selected_poses))
        # Check if the cluster map is stored as PoseDirectories or strings and convert
        representative_representative = next(iter(cluster_map))
        if not isinstance(representative_representative, protocols.PoseDirectory):
            # Make the cluster map based on strings
            for representative in list(cluster_map.keys()):
                # Remove old entry and convert all arguments to pose_id strings, saving as pose_id strings
                cluster_map[str(representative)] = [str(member) for member in cluster_map.pop(representative)]

        final_pose_indices = select_from_cluster_map(selected_pose_strs, cluster_map, number=job.cluster.number)
        final_poses = [selected_poses[idx] for idx in final_pose_indices]
        logger.info(f'Selected {len(final_poses)} poses after clustering')
    else:  # Try to generate the cluster_map?
        # raise utils.InputError(f'No --{flags.cluster_map} was provided. To cluster poses, specify:'
        logger.info(f'No {flags.cluster_map} was provided. To cluster poses, specify:'
                    f'"{putils.program_command} {flags.cluster_poses}" or '
                    f'"{putils.program_command} {flags.protocol} '
                    f'--{flags.modules} {flags.cluster_poses} {flags.select_poses}')
        logger.info('Grabbing all selected poses')
        final_poses = selected_poses
        # cluster_map: dict[str | protocols.PoseDirectory, list[str | protocols.PoseDirectory]] = {}
        # # {pose_string: [pose_string, ...]} where key is representative, values are matching designs
        # # OLD -> {composition: {pose_string: cluster_representative}, ...}
        # compositions: dict[tuple[str, ...], list[protocols.PoseDirectory]] = \
        #     protocols.cluster.group_compositions(selected_poses)
        # if job.multi_processing:
        #     mp_results = utils.mp_map(protocols.cluster.cluster_pose_by_transformations, compositions.values(),
        #                               processes=job.cores)
        #     for result in mp_results:
        #         cluster_map.update(result.items())
        # else:
        #     for composition_group in compositions.values():
        #         cluster_map.update(protocols.cluster.cluster_pose_by_transformations(composition_group))
        #
        # cluster_map_file = \
        #     os.path.join(job.clustered_poses, putils.default_clustered_pose_file.format(utils.starttime, location))
        # pose_cluster_file = utils.pickle_object(cluster_map, name=cluster_map_file, out_path='')
        # logger.info(f'Found {len(cluster_map)} unique clusters from {len(pose_directories)} pose inputs. '
        #             f'All clusters stored in {pose_cluster_file}')
    # else:
    #     logger.info('Grabbing all selected poses')
    #     final_poses = selected_poses

    if len(final_poses) > job.number:
        final_poses = final_poses[:job.number]
        logger.info(f'Found {len(final_poses)} poses after applying your number selection criteria')

    return final_poses


def select_from_cluster_map(selected_members: Sequence[Any], cluster_map: dict[Any, list[Any]], number: int = 1) \
        -> list[int]:
    """From a mapping of cluster representatives to their members, select members based on their ranking in the
    selected_members sequence

    Args:
        selected_members: A sorted list of members that are members of the cluster_map
        cluster_map: A mapping of cluster representatives to their members
        number: The number of members to select
    Returns:
        The indices of selected_members, trimmed and retrieved according to cluster_map membership
    """
    membership_representative_map = protocols.cluster.invert_cluster_map(cluster_map)
    representative_found: dict[Any, list[Any]] = {}
    not_found = []
    for idx, member in enumerate(selected_members):
        cluster_representative = membership_representative_map.get(member, None)
        if cluster_representative:
            if cluster_representative not in representative_found:
                # Include. This representative hasn't been identified
                representative_found[cluster_representative] = [idx]  # [member]
            else:
                # This cluster has already been found, and it was identified again. Report and only
                # include the highest ranked pose in the output as it provides info on all occurrences
                representative_found[cluster_representative].append(idx)  # member)
        else:
            not_found.append(idx)  # member)

    final_member_indices = []
    for member_indices in representative_found.values():
        final_member_indices.extend(member_indices[:number])

    if not_found:
        logger.warning(f"Couldn't locate the following members:\n\t%s\nAdding all of these to your selection..." %
                       '\n\t'.join(map(str, [selected_members[idx] for idx in not_found])))
        # 'Was {flags.cluster_poses} only run on a subset of the poses that were selected?
        final_member_indices.extend(not_found)

    return final_member_indices


def designs(pose_directories: Iterable[protocols.protocols.PoseDirectory]) \
        -> dict[protocols.protocols.PoseDirectory, list[str]]:
    job = job_resources_factory.get()
    if job.specification_file:
        loc_result = [(pose_directory, design) for pose_directory in pose_directories
                      for design in pose_directory.specific_designs]
        total_df = protocols.load_total_dataframe(pose_directories)
        selected_poses_df = \
            metrics.prioritize_design_indices(total_df.loc[loc_result, :], filter=job.filter, weight=job.weight,
                                              protocol=job.protocol, function=job.weight_function)
        # Specify the result order according to any filtering, weighting, and number
        results = {}
        for pose_directory, design in selected_poses_df.index.to_list()[:job.number]:
            if pose_directory in results:
                results[pose_directory].append(design)
            else:
                results[pose_directory] = [design]

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
        job.number = \
            len(selected_designs) if len(selected_designs) < job.number else job.number
        if job.allow_multiple_poses:
            logger.info(f'Choosing {job.number} designs, from the top ranked designs regardless of pose')
            loc_result = selected_designs[:job.number]
            results = {pose_dir: design for pose_dir, design in loc_result}
        else:  # elif job.designs_per_pose:
            logger.info(f'Choosing up to {job.number} designs, with {job.designs_per_pose} designs per pose')
            number_chosen = count(0)
            selected_poses = {}
            for pose_directory, design in selected_designs:
                designs = selected_poses.get(pose_directory, None)
                if designs:
                    if len(designs) >= job.designs_per_pose:
                        # We already have too many, continue with search. No need to check as no addition
                        continue
                    selected_poses[pose_directory].append(design)
                else:
                    selected_poses[pose_directory] = [design]

                if next(number_chosen) == job.number:
                    break

            results = selected_poses
            loc_result = [(pose_dir, design) for pose_dir, designs in selected_poses.items() for design in designs]

        # Include only the found index names to the saved dataframe
        save_poses_df = selected_poses_df.loc[loc_result, :]  # .droplevel(0).droplevel(0, axis=1).droplevel(0, axis=1)
        # convert to PoseDirectory objects
        # results = {pose_directory: results[str(pose_directory)] for pose_directory in pose_directories
        #            if str(pose_directory) in results}
    else:  # Select designed sequences from each pose provided (PoseDirectory)
        sequence_metrics = []  # Used to get the column headers
        sequence_filters = sequence_weights = None

        if job.filter or job.weight:
            try:
                representative_pose_directory = next(iter(pose_directories))
            except StopIteration:
                raise RuntimeError('Missing the required argument pose_directories. It must be passed to continue')
            example_trajectory = representative_pose_directory.trajectories
            trajectory_df = pd.read_csv(example_trajectory, index_col=0, header=[0])
            sequence_metrics = set(trajectory_df.columns.get_level_values(-1).to_list())

        if job.filter == list():
            sequence_filters = metrics.query_user_for_metrics(sequence_metrics, mode='filter', level='sequence')

        if job.weight == list():
            sequence_weights = metrics.query_user_for_metrics(sequence_metrics, mode='weight', level='sequence')

        if job.multi_processing:
            # sequence_weights = {'buns_per_ang': 0.2, 'observed_evolution': 0.3, 'shape_complementarity': 0.25,
            #                     'int_energy_res_summary_delta': 0.25}
            zipped_args = zip(pose_directories, repeat(sequence_filters), repeat(sequence_weights),
                              repeat(job.designs_per_pose), repeat(job.protocol))
            # result_mp = zip(*SDUtils.mp_starmap(Ams.select_sequences, zipped_args, processes=job.cores))
            result_mp = \
                utils.mp_starmap(protocols.protocols.PoseProtocol.select_sequences, zipped_args, processes=job.cores)
            # results - contains tuple of (PoseDirectory, design index) for each sequence
            # could simply return the design index then zip with the directory
            results = {pose_dir: designs for pose_dir, designs in zip(pose_directories, result_mp)}
        else:
            results = {pose_dir: protocols.protocols.PoseProtocol.select_sequences(
                                 pose_dir, filters=sequence_filters, weights=sequence_weights,
                                 number=job.designs_per_pose, protocols=job.protocol)
                       for pose_dir in pose_directories}

        # Todo there is no sort here so the number isn't really doing anything
        results = dict(list(results.items())[:job.number])
        loc_result = [(pose_dir, design) for pose_dir, designs in results.items() for design in designs]
        total_df = protocols.load_total_dataframe(pose_directories)
        save_poses_df = total_df.loc[loc_result, :].droplevel(0).droplevel(0, axis=1).droplevel(0, axis=1)

    # Format selected sequences for output
    putils.make_path(job.output_directory)
    logger.info(f'Relevant design files are being copied to the new directory: {job.output_directory}')

    if job.total and job.save_total:
        total_df_filename = os.path.join(job.output_directory, 'TotalPosesTrajectoryMetrics.csv')
        total_df.to_csv(total_df_filename)
        logger.info(f'Total Pose/Designs DataFrame was written to: {total_df}')

    logger.info(f'{len(save_poses_df)} poses were selected')
    # if save_poses_df is not None:  # Todo make work if DataFrame is empty...
    if job.filter or job.weight:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-{"Filtered" if job.filter else ""}'
                                                           f'{"Weighted" if job.weight else ""}DesignMetrics.csv')
    else:
        new_dataframe = os.path.join(job.output_directory, f'{utils.starttime}-DesignMetrics.csv')
    save_poses_df.to_csv(new_dataframe)
    logger.info(f'New DataFrame with selected designs was written to: {new_dataframe}')

    # Create new output of designed PDB's  # TODO attach the state to these files somehow for further SymDesign use
    exceptions = []
    for pose_dir, designs in results.items():
        for design in designs:
            file_path = os.path.join(pose_dir.designs, f'*{design}*')
            file = sorted(glob(file_path))
            if not file:  # Add to exceptions
                exceptions.append((pose_dir, f'No file found for "{file_path}"'))
                continue
            out_path = os.path.join(job.output_directory, f'{pose_dir}_design_{design}.pdb')
            if not os.path.exists(out_path):
                shutil.copy(file[0], out_path)  # [i])))
                # shutil.copy(des_dir.trajectories, os.path.join(outdir_traj, os.path.basename(des_dir.trajectories)))
                # shutil.copy(des_dir.residues_file, os.path.join(outdir_res, os.path.basename(des_dir.residues_file)))
        # try:
        #     # Create symbolic links to the output PDB's
        #     os.symlink(file[0], os.path.join(job.output_directory, '%s_design_%s.pdb' % (str(des_dir), design)))  # [i])))
        #     os.symlink(des_dir.trajectories, os.path.join(outdir_traj, os.path.basename(des_dir.trajectories)))
        #     os.symlink(des_dir.residues_file, os.path.join(outdir_res, os.path.basename(des_dir.residues_file)))
        # except FileExistsError:
        #     pass

    return results  # , exceptions


# def sequences(results: dict[protocols.protocols.PoseDirectory, ]):
def sequences(pose_directories: list[protocols.protocols.PoseDirectory]):
    """Perform design selection followed by sequence formatting on those designs

    Args:
        pose_directories:

    Returns:

    """
    job = job_resources_factory.get()
    results = designs(pose_directories)

    job.output_file = os.path.join(job.output_directory, f'{job.prefix}SelectedDesigns{job.suffix}.paths')
    # Todo move this to be in terminate(results=results.keys()) -> success
    # pose_directories = list(results.keys())
    with open(job.output_file, 'w') as f:
        f.write('%s\n' % '\n'.join(pose_dir.path for pose_dir in list(results.keys())))

    # Set up mechanism to solve sequence tagging preferences
    def solve_tags(pose: Pose) -> list[bool]:
        if job.tag_entities is None:
            boolean_tags = [False for _ in pose.entities]
        elif job.tag_entities == 'all':
            boolean_tags = [True for _ in pose.entities]
        elif job.tag_entities == 'single':
            boolean_tags = [True for _ in pose.entities]
        else:
            boolean_tags = []
            for tag_specification in map(str.strip, job.tag_entities.split(',')):
                # Remove non-numeric stuff
                if tag_specification == '':  # Probably a trailing ',' ...
                    continue
                else:
                    tag_specification.translate(utils.keep_digit_table)

                try:  # To convert to an integer
                    boolean_tags.append(True if int(tag_specification) == 1 else False)
                except ValueError:  # Not an integer False
                    boolean_tags.append(False)

            # Add any missing arguments to the tagging scheme
            for _ in range(pose.number_of_entities - len(boolean_tags)):
                boolean_tags.append(False)

        return boolean_tags

    if job.multicistronic:
        intergenic_sequence = job.multicistronic_intergenic_sequence
    else:
        intergenic_sequence = ''

    # Format sequences for expression
    missing_tags = {}  # result: [True, True] for result in results
    tag_sequences, final_sequences, inserted_sequences, nucleotide_sequences = {}, {}, {}, {}
    codon_optimization_errors = {}
    for des_dir, _designs in results.items():
        des_dir.load_pose()  # source=des_dir.asu_path)
        tag_index = solve_tags(des_dir.pose)
        number_of_tags = sum(tag_index)
        des_dir.pose.rename_chains()  # Do I need to modify chains?
        for design in _designs:
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
                if job.avoid_tagging_helices:
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
                available_tags = utils.ProteinExpression.find_expression_tags(inserted_design_sequence)
                if available_tags:  # look for existing tag to remove from sequence and save identity
                    tag_names, tag_termini, existing_tag_sequences = \
                        zip(*[(tag['name'], tag['termini'], tag['sequence']) for tag in available_tags])
                    try:
                        preferred_tag_index = tag_names.index(job.preferred_tag)
                        if tag_termini[preferred_tag_index] in true_termini:
                            selected_tag = available_tags[preferred_tag_index]
                    except ValueError:
                        pass
                    pretag_sequence = utils.ProteinExpression.remove_expression_tags(inserted_design_sequence,
                                                                                     existing_tag_sequences)
                else:
                    pretag_sequence = inserted_design_sequence
                logger.debug(f'The pretag sequence is:\n{pretag_sequence}')

                # Find the open reading frame offset using the structure sequence after insertion
                offset = find_orf_offset(pretag_sequence, mutations)
                formatted_design_sequence = pretag_sequence[offset:]
                logger.debug(f'The open reading frame offset is {offset}')
                logger.debug(f'The formatted_design sequence is:\n{formatted_design_sequence}')

                if number_of_tags == 0:  # Don't solve tags
                    sequences_and_tags[design_string] = {'sequence': formatted_design_sequence, 'tag': {}}
                    continue

                if not selected_tag:  # find compatible tags from matching PDB observations
                    uniprot_id = source_entity.uniprot_id
                    uniprot_id_matching_tags = tag_sequences.get(uniprot_id, None)
                    if not uniprot_id_matching_tags:
                        uniprot_id_matching_tags = \
                            utils.ProteinExpression.find_matching_expression_tags(uniprot_id=uniprot_id)
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
                            preferred_tag_index_2 = tag_names[iteration:].index(job.preferred_tag)
                            if tag_termini[preferred_tag_index_2] in true_termini:
                                selected_tag = uniprot_id_matching_tags[preferred_tag_index_2]
                                break
                        except ValueError:
                            selected_tag = \
                                utils.ProteinExpression.select_tags_for_sequence(sequence_id,
                                                                                 uniprot_id_matching_tags,
                                                                                 preferred=job.preferred_tag,
                                                                                 **termini_availability)
                            break
                        iteration += 1

                if selected_tag.get('name'):
                    missing_tags[(des_dir, design)][idx] = 0
                    logger.debug(f'The pre-existing, identified tag is:\n{selected_tag}')
                sequences_and_tags[design_string] = {'sequence': formatted_design_sequence, 'tag': selected_tag}

            # After selecting all tags, consider tagging the design as a whole
            if number_of_tags > 0:
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
                    if job.avoid_tagging_helices:
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
                            if job.preferred_tag:
                                tag = job.preferred_tag
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
                                                              resources.config.expression_tags, 1)]))
                                    if tag_input.isdigit():
                                        tag_input = int(tag_input)
                                        if tag_input <= len(resources.config.expression_tags):
                                            tag = list(resources.config.expression_tags.keys())[tag_input - 1]
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
                                    resources.config.expression_tags[tag] + 'SG' \
                                    + sequences_and_tags[selected_entity]['sequence'][:12]
                            else:  # termini == 'c'
                                new_tag_sequence = \
                                    sequences_and_tags[selected_entity]['sequence'][-12:] \
                                    + 'GS' + resources.config.expression_tags[tag]
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
                                # tag = list(utils.ProteinExpression.expression_tags.keys())[tag_input - 1]
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
                design_sequence = utils.ProteinExpression.add_expression_tag(tag.get('sequence'), sequence)
                if tag.get('sequence') and design_sequence == sequence:  # tag exists and no tag added
                    tag_sequence = resources.config.expression_tags[tag.get('name')]
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
                                                   f'{" " * (len(range(*idx_range)) - (len(str(idx_range[0])) + 1))}'
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
                if job.nucleotide:
                    try:
                        nucleotide_sequence = \
                            utils.ProteinExpression.optimize_protein_sequence(design_sequence,
                                                                              species=job.optimize_species)
                    except NoSolutionError:  # add the protein sequence?
                        logger.warning(f'Optimization of {design_string} was not successful!')
                        codon_optimization_errors[design_string] = design_sequence
                        break

                    if job.multicistronic:
                        if idx > 0:
                            cistronic_sequence += intergenic_sequence
                        cistronic_sequence += nucleotide_sequence
                    else:
                        nucleotide_sequences[design_string] = nucleotide_sequence
            if job.multicistronic:
                nucleotide_sequences[str(des_dir)] = cistronic_sequence

    # Report Errors
    if codon_optimization_errors:
        # Todo utilize errors
        error_file = \
            write_sequences(codon_optimization_errors, csv=job.csv,
                            file_name=os.path.join(job.output_directory,
                                                   f'{job.prefix}OptimizationErrorProteinSequences{job.suffix}'))
    # Write output sequences to fasta file
    seq_file = write_sequences(final_sequences, csv=job.csv,
                               file_name=os.path.join(job.output_directory, f'{job.prefix}SelectedSequences{job.suffix}'))
    logger.info(f'Final Design protein sequences written to: {seq_file}')
    seq_comparison_file = \
        write_sequences(inserted_sequences, csv=job.csv,
                        file_name=os.path.join(job.output_directory,
                                               f'{job.prefix}SelectedSequencesExpressionAdditions{job.suffix}'))
    logger.info(f'Final Expression sequence comparison to Design sequence written to: {seq_comparison_file}')
    # check for protein or nucleotide output
    if job.nucleotide:
        nucleotide_sequence_file = \
            write_sequences(nucleotide_sequences, csv=job.csv,
                            file_name=os.path.join(job.output_directory,
                                                   f'{job.prefix}SelectedSequencesNucleotide{job.suffix}'))
        logger.info(f'Final Design nucleotide sequences written to: {nucleotide_sequence_file}')
