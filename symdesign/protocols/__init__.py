from __future__ import annotations

import functools
import logging
import os
import traceback
from itertools import repeat, combinations
from subprocess import list2cmdline, Popen
from typing import Iterable, AnyStr, Callable, Type, Any

import numpy as np
import pandas as pd
import sklearn as skl
from scipy.spatial.distance import pdist

from . import cluster, config, fragdock, pose, select
from symdesign import flags, metrics
from symdesign.resources.config import default_pca_variance
from symdesign.sequence import protein_letters_1to3, protein_letters_3to1
from symdesign.structure.model import Models, MultiModel, Model, Pose
from symdesign.structure.sequence import write_pssm_file, sequence_difference
from symdesign.structure.utils import DesignError, SymmetryError
from symdesign.utils import condensed_to_square, get_directory_file_paths, InputError, path as putils, \
    ReportException, rosetta, starttime, sym, write_shell_script
# from ..resources.job import JobResources, job_resources_factory


logger = logging.getLogger(__name__)
# Protocols
nanohedra = fragdock.fragment_dock
cluster_poses = cluster.cluster_poses
select_poses = select.sql_poses  # select.poses
select_designs = select.sql_designs  # select.designs
select_sequences = select.sql_sequences  # select.sequences
warn_missing_symmetry = \
    f'Cannot %s without providing symmetry! Provide symmetry with "--symmetry" or "--{putils.sym_entry}"'


def close_logs(func: Callable):
    """Wrap a function/method to close the functions first arguments .log attribute FileHandlers after use"""
    @functools.wraps(func)
    def wrapped(job, *args, **kwargs):
        func_return = func(job, *args, **kwargs)
        # Adapted from https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile
        for handler in job.log.handlers:
            handler.close()
        return func_return
    return wrapped


def remove_structure_memory(func):
    """Decorator to remove large memory attributes from the instance after processing is complete"""
    @functools.wraps(func)
    def wrapped(job, *args, **kwargs):
        func_return = func(job, *args, **kwargs)
        if job.job.reduce_memory:
            job.pose = None
            # self.entities.clear()
        return func_return
    return wrapped


def handle_design_errors(errors: tuple[Type[Exception], ...] = (DesignError,)) -> Callable:
    """Wrap a function/method with try: except errors: and log exceptions to the functions first argument .log attribute

    This argument is typically self and is in a class with .log attribute

    Args:
        errors: A tuple of exceptions to monitor. Must be a tuple even if single exception
    Returns:
        Function return upon proper execution, else is error if exception raised, else None
    """
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapped(job, *args, **kwargs) -> Any:
            try:
                return func(job, *args, **kwargs)
            except errors as error:
                # Perform exception reporting using self.log
                job.log.error(error)
                job.log.info(''.join(traceback.format_exc()))  # .format_exception(error)))
                return ReportException(str(error))
        return wrapped
    return wrapper


def protocol_decorator(errors: tuple[Type[Exception], ...] = (DesignError,)) -> Callable:
    """Wrap a function/method with try: except errors: and log exceptions to the functions first argument .log attribute

    This argument is typically self and is in a class with .log attribute

    Args:
        errors: A tuple of exceptions to monitor. Must be a tuple even if single exception
    Returns:
        Function return upon proper execution, else is error if exception raised, else None
    """
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapped(job, *args, **kwargs) -> Any:
            # handle_design_errors()
            try:
                func_return = func(job, *args, **kwargs)
            except errors as error:
                # Perform exception reporting using self.log
                job.log.error(error)
                job.log.info(''.join(traceback.format_exc()))  # .format_exception(error)))
                func_return = ReportException(str(error))
            # remove_structure_memory()
            if job.job.reduce_memory:
                job.measure_evolution = job.measure_alignment = \
                    job.pose = job.initial_model = None
                # job.entities.clear()
            job.protocol = None
            # close_logs()
            # Adapted from https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile
            for handler in job.log.handlers:
                handler.close()

            return func_return
        return wrapped
    return wrapper


@protocol_decorator()
def predict_structure(job: pose.PoseJob):
    """From a sequence input, predict the structure using one of various structure prediction pipelines

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    # job.load_pose()
    job.identify_interface()
    # Acquire the pose_metrics if None have been made yet
    job.calculate_pose_metrics()

    job.predict_structure()


@protocol_decorator()
def custom_rosetta_script(job: pose.PoseJob, script, file_list=None, native=None, suffix=None,
                          score_only=None, variables=None, **kwargs):
    """Generate a custom script to dispatch to the design using a variety of parameters

    Args:
        job: The PoseJob for which the protocol should be performed on
        script:
        file_list:
        native:
        suffix:
        score_only:
        variables:
    """
    # Todo reflect modern metrics collection
    raise NotImplementedError('This module is outdated, please update it to use')
    job.identify_interface()

    # Now acquiring in process_rosetta_metrics()
    # # Acquire the pose_metrics if None have been made yet
    # job.calculate_pose_metrics()

    cmd = rosetta.script_cmd.copy()
    script_name = os.path.splitext(os.path.basename(script))[0]

    if not os.path.exists(job.flags) or job.job.force:
        job.prepare_rosetta_flags(out_dir=job.scripts_path)
        job.log.debug(f'Pose flags written to: {job.flags}')

    if job.symmetry_dimension is not None and job.symmetry_dimension > 0:
        cmd += ['-symmetry_definition', 'CRYST1']

    if file_list:
        pdb_input = os.path.join(job.scripts_path, 'design_files.txt')
        generate_files_cmd = ['python', putils.list_pdb_files, '-d', job.designs_path, '-o', pdb_input, '-e', '.pdb']
    else:
        pdb_input = job.refined_pdb
        generate_files_cmd = []  # empty command

    if native:
        native = getattr(job, native, 'refined_pdb')
    else:
        native = job.refined_pdb

    # if isinstance(suffix, str):
    #     suffix = ['-out:suffix', '_%s' % suffix]
    # if isinstance(suffix, bool):
    if suffix:
        suffix = ['-out:suffix', f'_{script_name}']
    else:
        suffix = []

    if score_only:
        score = ['-out:file:score_only', job.scores_file]
    else:
        score = []

    if job.job.design.number:
        trajectories = ['-nstruct', str(job.job.design.number)]
    else:
        trajectories = ['-no_nstruct_label true']

    if variables:
        for idx, var_val in enumerate(variables):
            variable, value = var_val.split('=')
            variables[idx] = '%s=%s' % (variable, getattr(job.pose, value, ''))
        variables = ['-parser:script_vars'] + variables
    else:
        variables = []

    cmd += [f'@{flags_file}', f'-in:file:{"l" if file_list else "s"}', pdb_input, '-in:file:native', native] \
        + score + suffix + trajectories + ['-parser:protocol', script] + variables
    if job.job.mpi > 0:
        cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(job.job.mpi)] + cmd

    # Create executable to gather interface Metrics on all Designs
    if job.job.distribute_work:
        analysis_cmd = job.make_analysis_cmd()
        write_shell_script(list2cmdline(generate_files_cmd), name=script_name, out_path=job.scripts_path,
                           additional=[list2cmdline(cmd)] + [list2cmdline(analysis_cmd)])
        # Todo metrics: [list2cmdline(command) for command in metric_cmds]
    else:
        list_all_files_process = Popen(generate_files_cmd)
        list_all_files_process.communicate()
        # Todo
        # for metric_cmd in metric_cmds:
        #     metrics_process = Popen(metric_cmd)
        #     metrics_process.communicate()  # wait for command to complete

        # Gather metrics for each design produced from this proceedure
        if os.path.exists(job.scores_file):
            job.process_rosetta_metrics()


@protocol_decorator()
def interface_metrics(job: pose.PoseJob):
    """Generate a script capable of running Rosetta interface metrics analysis on the bound and unbound states

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    job.identify_interface()
    # metrics_flags = 'repack=yes'
    job.protocol = putils.interface_metrics
    main_cmd = rosetta.script_cmd.copy()

    # Now acquiring in process_rosetta_metrics()
    # # Acquire the pose_metrics if None have been made yet
    # job.calculate_pose_metrics()

    if not os.path.exists(job.flags) or job.job.force:
        job.prepare_rosetta_flags(out_dir=job.scripts_path)
        job.log.debug(f'Pose flags written to: {job.flags}')

    design_files = \
        os.path.join(job.scripts_path, f'{starttime}_design_files'
                     f'{f"_{job.job.specific_protocol}" if job.job.specific_protocol else ""}.txt')
    # Inclue the pose source in the designs to perform metrics on
    file_paths = [job.pose_path] if os.path.exists(job.pose_path) else []
    file_paths.extend(get_directory_file_paths(job.designs_path,
                                               suffix=job.job.specific_protocol if job.job.specific_protocol else '',
                                               extension='.pdb'))
    with open(design_files, 'w') as f:
        f.write('%s\n' % '\n'.join(file_paths))

    # generate_files_cmd = ['python', putils.list_pdb_files, '-d', job.designs_path, '-o', design_files, '-e', '.pdb'] \
    #     + (['-s', job.job.specific_protocol] if job.job.specific_protocol else [])
    main_cmd += [f'@{job.flags}', '-in:file:l', design_files,
                 # Todo out:file:score_only file is not respected if out:path:score_file given
                 #  -run:score_only true?
                 '-out:file:score_only', job.scores_file, '-no_nstruct_label', 'true', '-parser:protocol']
    #              '-in:file:native', job.refined_pdb,
    if job.job.mpi > 0:
        main_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(job.job.mpi)] + main_cmd

    metric_cmd_bound = main_cmd.copy()
    if job.symmetry_dimension is not None and job.symmetry_dimension > 0:
        metric_cmd_bound += ['-symmetry_definition', 'CRYST1']
    metric_cmd_bound += \
        [os.path.join(putils.rosetta_scripts_dir, f'{job.protocol}{"_DEV" if job.job.development else ""}.xml')]
    job.log.info(f'Metrics command for Pose: {list2cmdline(metric_cmd_bound)}')
    entity_cmd = main_cmd + [os.path.join(putils.rosetta_scripts_dir,
                                          f'metrics_entity{"_DEV" if job.job.development else ""}.xml')]
    # metric_cmds = [metric_cmd_bound]
    # metric_cmds.extend(job.generate_entity_metrics_commands(entity_cmd))
    entity_metric_cmds = job.generate_entity_metrics_commands(entity_cmd)

    # Create executable to gather interface Metrics on all Designs
    if job.job.distribute_work:
        analysis_cmd = job.make_analysis_cmd()
        # write_shell_script(list2cmdline(generate_files_cmd), name=putils.interface_metrics, out_path=job.scripts_path,
        write_shell_script(metric_cmd_bound, name=putils.interface_metrics, out_path=job.scripts_path,
                           additional=[list2cmdline(command) for command in entity_metric_cmds]
                                      + [list2cmdline(analysis_cmd)])
    else:
        # list_all_files_process = Popen(generate_files_cmd)
        # list_all_files_process.communicate()
        for metric_cmd in [metric_cmd_bound] + entity_metric_cmds:
            metrics_process = Popen(metric_cmd)
            metrics_process.communicate()  # wait for command to complete

        # Gather metrics for each design produced from this proceedure
        if os.path.exists(job.scores_file):
            job.process_rosetta_metrics()


@protocol_decorator()
def check_unmodelled_clashes(job: pose.PoseJob, clashing_threshold: float = 0.75):
    """Given a multimodel file, measure the number of clashes is less than a percentage threshold

    Args:
        job: The PoseJob for which the protocol should be performed on
        clashing_threshold: The number of Model instances which have observed clashes
    """
    raise DesignError('This module is not working correctly at the moment')
    models = [Models.from_PDB(job.job.structure_db.full_models.retrieve_data(name=entity), log=job.log)
              for entity in job.entity_names]
    # models = [Models.from_file(job.job.structure_db.full_models.retrieve_data(name=entity))
    #           for entity in job.entity_names]

    # for each model, transform to the correct space
    models = job.transform_structures_to_pose(models)
    multimodel = MultiModel.from_models(models, independent=True, log=job.log)

    clashes = 0
    prior_clashes = 0
    for idx, state in enumerate(multimodel, 1):
        clashes += (1 if state.is_clash() else 0)
        state.write(out_path=os.path.join(job.path, f'state_{idx}.pdb'))
        print(f'State {idx} - Clashes: {"YES" if clashes > prior_clashes else "NO"}')
        prior_clashes = clashes

    if clashes / float(len(multimodel)) > clashing_threshold:
        raise DesignError(f'The frequency of clashes ({clashes / float(len(multimodel))}) exceeds the clashing '
                          f'threshold ({clashing_threshold})')


@protocol_decorator()
def check_clashes(job: pose.PoseJob):
    """Check for clashes in the input and in the symmetric assembly if symmetric

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    job.load_pose()


@protocol_decorator()
def rename_chains(job: pose.PoseJob):
    """Standardize the chain names in incremental order found in the design source file

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    model = Model.from_file(job.source, log=job.log)
    model.rename_chains()
    model.write(out_path=job.pose_path)


@protocol_decorator(errors=(DesignError, RuntimeError))  # Todo remove RuntimeError from .orient()
def orient(job: pose.PoseJob, to_pose_directory: bool = True):
    """Orient the Pose with the prescribed symmetry at the origin and symmetry axes in canonical orientations
    job.symmetry is used to specify the orientation

    Args:
        job: The PoseJob for which the protocol should be performed on
        to_pose_directory: Whether to write the file to the pose_directory or to another source
    """
    if not job.initial_model:
        job.load_initial_model()

    if job.symmetry:
        if to_pose_directory:
            out_path = job.assembly_path
        else:
            putils.make_path(job.job.orient_dir)
            out_path = os.path.join(job.job.orient_dir, f'{job.initial_model.name}.pdb')

        job.initial_model.orient(symmetry=job.symmetry)

        orient_file = job.initial_model.write(out_path=out_path)
        job.log.info(f'The oriented file was saved to {orient_file}')
        for entity in job.initial_model.entities:
            entity.remove_mate_chains()
            # job.entity_names.append(entity.name)

        # Load the pose and save the asu
        job.load_pose()  # entities=model.entities)
    else:
        raise SymmetryError(warn_missing_symmetry % orient.__name__)


@protocol_decorator()
def find_asu(job: pose.PoseJob):
    """From a PDB with multiple Chains from multiple Entities, return the minimal configuration of Entities.
    ASU will only be a true ASU if the starting PDB contains a symmetric system, otherwise all manipulations find
    the minimal unit of Entities that are in contact

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    # Check if the symmetry is known, otherwise this wouldn't work without the old, "symmetry-less", protocol
    if job.is_symmetric():
        if os.path.exists(job.assembly_path):
            job.load_pose(file=job.assembly_path)
        else:
            job.load_pose()
    else:
        raise NotImplementedError('Not sure if asu format matches pose.get_contacting_asu() standard with no symmetry'
                                  '. This might cause issues')
        # Todo ensure asu format matches pose.get_contacting_asu() standard
        # pdb = Model.from_file(job.structure_source, log=job.log)
        # asu = pdb.return_asu()
        job.load_pose()
        # asu.update_attributes_from_pdb(pdb)

    # Save the Pose.asu
    job.output_pose(path=job.pose_path)


@protocol_decorator()
def expand_asu(job: pose.PoseJob):
    """For the design info given by a PoseJob source, initialize the Pose with job.source file,
    job.symmetry, and job.log objects then expand the design given the provided symmetry operators and write to a
    file

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    if job.is_symmetric():
        job.load_pose()
    else:
        raise SymmetryError(warn_missing_symmetry % expand_asu.__name__)
    # job.pickle_info()  # Todo remove once PoseJob state can be returned to the dispatch w/ MP


@protocol_decorator()
def generate_fragments(job: pose.PoseJob):
    """For the design info given by a PoseJob source, initialize the Pose then generate interfacial fragment
    information between Entities. Aware of symmetry and design_selectors in fragment generation file

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    job.load_pose()
    if job.job.interface_only:
        entities = False
        interface = True
    else:
        entities = True
        interface = job.job.interface
    job.generate_fragments(interface=interface, oligomeric_interfaces=job.job.oligomeric_interfaces, entities=entities)


@protocol_decorator()
def refine(job: pose.PoseJob):
    """Refine the source Pose

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    # job.load_pose()
    job.identify_interface()
    job.refine()  # Inherently utilized... gather_metrics=job.job.metrics)


@protocol_decorator()
def interface_design(job: pose.PoseJob):
    """For the design info given by a PoseJob source, initialize the Pose then prepare all parameters for
    interfacial redesign between Pose Entities. Aware of symmetry, design_selectors, fragments, and
    evolutionary information in interface design

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    # Save the prior value then set after the protocol completes
    prior_value = job.job.design.interface
    job.job.design.interface = True
    design_return = design(job)
    job.job.design.interface = prior_value
    return design_return
    raise NotImplementedError('This protocol has been depreciated in favor of design() with job.design.interface=True')
    job.identify_interface()

    putils.make_path(job.data_path)
    # Create all files which store the evolutionary_profile and/or fragment_profile -> design_profile
    if job.job.design.method == putils.rosetta_str:
        # Update the Pose with the number of designs
        raise NotImplementedError('Need to generate job.number_of_designs matching job.proteinmpnn_design()...')
        # Todo update upon completion given results of designs list file...
        job.update_design_data(design_parent=job.pose_source, number=job.job.design.number)
        favor_fragments = evo_fill = True
        # Ensure the Pose is refined into the current_energy_function
        if not job.refined and not os.path.exists(job.refined_pdb):
            job.refine(gather_metrics=False)
    else:
        favor_fragments = evo_fill = False

    if job.job.design.term_constraint:
        # if not job.pose.fragment_queries:
        job.generate_fragments(interface=True)
        job.pose.calculate_fragment_profile(evo_fill=evo_fill)
    # elif isinstance(job.fragment_observations, list):
    #     raise NotImplementedError(f"Can't put fragment observations taken away from the pose onto the pose due to "
    #                               f"entities")
    #     job.pose.fragment_pairs = job.fragment_observations
    #     job.pose.calculate_fragment_profile(evo_fill=evo_fill)
    # elif os.path.exists(job.frag_file):
    #     job.retrieve_fragment_info_from_file()

    job.set_up_evolutionary_profile()

    # job.pose.combine_sequence_profiles()
    # I could also add the combined profile here instead of at each Entity
    # job.pose.calculate_profile(favor_fragments=favor_fragments)
    # Todo this is required to simplify pose.profile from each source
    job.pose.add_profile(evolution=job.job.design.evolution_constraint,
                         fragments=job.job.design.term_constraint, favor_fragments=favor_fragments,
                         out_dir=job.job.api_db.hhblits_profiles.location)

    # -------------------------------------------------------------------------
    # Todo job.solve_consensus()
    # -------------------------------------------------------------------------
    putils.make_path(job.designs_path)
    # Acquire the pose_metrics if None have been made yet
    job.calculate_pose_metrics()

    # match job.job.design.method:  # Todo python 3.10
    #     case [putils.rosetta_str | putils.consensus]:
    #         # Write generated files
    #         job.pose.pssm_file = \
    #             write_pssm_file(job.pose.evolutionary_profile, file_name=job.evolutionary_profile_file)
    #         write_pssm_file(job.pose.profile, file_name=job.design_profile_file)
    #         job.pose.fragment_profile.write(file_name=job.fragment_profile_file)
    #         job.rosetta_interface_design()  # Sets job.protocol
    #     case putils.proteinmpnn:
    #         job.proteinmpnn_design(interface=True, neighbors=job.job.design.neighbors)  # Sets job.protocol
    #     case _:
    #         raise ValueError(f"The method '{job.job.design.method}' isn't available")
    if job.job.design.method in [putils.rosetta_str, putils.consensus]:
        # Write generated files
        job.pose.pssm_file = \
            write_pssm_file(job.pose.evolutionary_profile, file_name=job.evolutionary_profile_file)
        write_pssm_file(job.pose.profile, file_name=job.design_profile_file)
        job.pose.fragment_profile.write(file_name=job.fragment_profile_file)
        job.rosetta_interface_design()  # Sets job.protocol
    elif job.job.design.method == putils.proteinmpnn:
        job.proteinmpnn_design(interface=True, neighbors=job.job.design.neighbors)  # Sets job.protocol
    else:
        raise InputError(
            f"The method '{job.job.design.method}' isn't available")


@protocol_decorator()
def design(job: pose.PoseJob):
    """For the design info given by a PoseJob source, initialize the Pose then prepare all parameters for
    sequence design. Aware of symmetry, design_selectors, fragments, and evolutionary information

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    # job.load_pose()
    job.identify_interface()

    putils.make_path(job.data_path)
    # Create all files which store the evolutionary_profile and/or fragment_profile -> design_profile
    if job.job.design.method == putils.rosetta_str:
        # Update upon completion given results of designs list file...
        # NOT # Update the Pose with the number of designs
        # raise NotImplementedError('Need to generate design_number matching job.proteinmpnn_design()...')
        # job.update_design_data(design_parent=job.pose_source, number=job.job.design.number)
        if job.job.design.interface:
            pass
        else:
            raise NotImplementedError(
                f"Can't perform design using Rosetta just yet. Try {flags.interface_design} instead...")
        favor_fragments = evo_fill = True
        # Ensure the Pose is refined into the current_energy_function
        if not job.refined and not os.path.exists(job.refined_pdb):
            job.refine(gather_metrics=False)
    else:
        favor_fragments = evo_fill = False

    if job.job.design.term_constraint:
        # if not job.pose.fragment_queries:
        # Todo this is working but the information isn't really used...
        #  ALSO Need to get oligomeric type frags
        job.generate_fragments(interface=True)  # job.job.design.interface
        job.pose.calculate_fragment_profile(evo_fill=evo_fill)
    # elif isinstance(job.fragment_observations, list):
    #     raise NotImplementedError(f"Can't put fragment observations taken away from the pose onto the pose due to "
    #                               f"entities")
    #     job.pose.fragment_pairs = job.fragment_observations
    #     job.pose.calculate_fragment_profile(evo_fill=evo_fill)
    # elif os.path.exists(job.frag_file):
    #     job.retrieve_fragment_info_from_file()

    job.set_up_evolutionary_profile()

    # job.pose.combine_sequence_profiles()
    # I could also add the combined profile here instead of at each Entity
    # job.pose.calculate_profile(favor_fragments=favor_fragments)
    # Todo this is required to simplify pose.profile from each source
    job.pose.add_profile(evolution=job.job.design.evolution_constraint,
                         fragments=job.job.design.term_constraint, favor_fragments=favor_fragments,
                         out_dir=job.job.api_db.hhblits_profiles.location)

    # -------------------------------------------------------------------------
    # Todo job.solve_consensus()
    # -------------------------------------------------------------------------
    putils.make_path(job.designs_path)
    # Acquire the pose_metrics if None have been made yet
    job.calculate_pose_metrics()

    # match job.job.design.method:  # Todo python 3.10
    #     case [putils.rosetta_str | putils.consensus]:
    #         # Write generated files
    #         job.pose.pssm_file = \
    #             write_pssm_file(job.pose.evolutionary_profile, file_name=job.evolutionary_profile_file)
    #         write_pssm_file(job.pose.profile, file_name=job.design_profile_file)
    #         job.pose.fragment_profile.write(file_name=job.fragment_profile_file)
    #         if job.job.design.interface:
    #             job.rosetta_interface_design()  # Sets job.protocol
    #         else:
    #             raise NotImplementedError(f'No function for all residue Rosetta design yet')
    #             job.rosetta_design()  # Sets job.protocol
    #     case putils.proteinmpnn:
    #         job.proteinmpnn_design()  # Sets job.protocol
    #     case _:
    #         raise ValueError(f"The method '{job.job.design.method}' isn't available")
    if job.job.design.method in [putils.rosetta_str, putils.consensus]:
        # Write generated files
        job.pose.pssm_file = \
            write_pssm_file(job.pose.evolutionary_profile, file_name=job.evolutionary_profile_file)
        write_pssm_file(job.pose.profile, file_name=job.design_profile_file)
        job.pose.fragment_profile.write(file_name=job.fragment_profile_file)
        if job.job.design.interface:
            job.rosetta_interface_design()  # Sets job.protocol
        else:
            raise NotImplementedError(f'No function for all residue Rosetta design yet')
            job.rosetta_design()  # Sets job.protocol
    elif job.job.design.method == putils.proteinmpnn:
        # Sets job.protocol
        job.proteinmpnn_design()  # interface=job.job.design.interface, neighbors=job.job.design.neighbors
    else:
        raise InputError(
            f"The method '{job.job.design.method}' isn't available")


@protocol_decorator()
def optimize_designs(job: pose.PoseJob, threshold: float = 0.):
    """To touch up and optimize a design, provide a list of optional directives to view mutational landscape around
    certain residues in the design as well as perform wild-type amino acid reversion to mutated residues

    Args:
        job: The PoseJob for which the protocol should be performed on
        # residue_directives=None (dict[Residue | int, str]):
        #     {Residue object: 'mutational_directive', ...}
        # design_file=None (str): The name of a particular design file present in the designs output
        threshold: The threshold above which background amino acid frequencies are allowed for mutation
    """
    job.protocol = protocol_xml1 = putils.optimize_designs
    # job.protocol = putils.pross
    # Todo Notes for PROSS implementation
    #  I need to use a mover like FilterScan to measure all the energies for a particular residue and it's possible
    #  mutational space. Using these measurements, I then need to choose only those ones which make a particular
    #  energetic contribution to the structure and test these out using a FastDesign protocol where each is tried.
    #  This will likely utilize a resfile as in PROSS implementation and here as creating a PSSM could work but is a
    #  bit convoluted. I think finding the energy threshold to use as a filter cut off is going to be a bit
    #  heuristic as the REF2015 scorefunction wasn't used in PROSS publication.

    generate_files_cmd = pose.null_cmd

    # Create file output
    raise NotImplementedError('Must make the infile a in:file:s derivative')
    designed_files_file = os.path.join(job.scripts_path, f'{starttime}_{job.protocol}_files_output.txt')
    if job.current_designs:
        design_files = [design_.structure_file for design_ in job.current_designs]
        design_files_file = os.path.join(job.scripts_path, f'{starttime}_{job.protocol}_files.txt')
        with open(design_files_file, 'w') as f:
            f.write('%s\n' % '\n'.join(design_files))
        # Write the designed_files_file with all "tentatively" designed file paths
        pdb_out_path = job.designs_path
        out_file_string = f'%s{os.sep}{pdb_out_path}{os.sep}%s'
        with open(design_files_file, 'w') as f:
            f.write('%s\n' % '\n'.join(out_file_string % os.path.split(file) for file in design_files))
        # -in:file:native is here to block flag file version, not actually useful for refine
        infile = ['-in:file:l', design_files_file, '-in:file:native', job.source_path]
        metrics_pdb = ['-in:file:l', designed_files_file, '-in:file:native', job.source_path]
    else:
        infile = ['-in:file:s', job.refined_pdb]

    job.identify_interface()  # job.load_pose()
    # for design in job.current_designs:
    #     job.load_pose(structure_source=design.structure_path)
    #     job.identify_interface()

    # Now acquiring in process_rosetta_metrics()
    # # Acquire the pose_metrics if None have been made yet
    # job.calculate_pose_metrics()

    # format all amino acids in job.interface_design_residue_numbers with frequencies above the threshold to a set
    # Todo, make threshold and return set of strings a property of a profile object
    # Locate the desired background profile from the pose
    background_profile = getattr(job.pose, job.job.background_profile)
    raise NotImplementedError("background_profile doesn't account for residue.index versus residue.number")
    background = {residue: {protein_letters_1to3.get(aa) for aa in protein_letters_1to3
                            if background_profile[residue.number].get(aa, -1) > threshold}
                  for residue in job.pose.interface_residues}
    # include the wild-type residue from PoseJob Pose source and the residue identity of the selected design
    wt = {residue: {background_profile[residue.number].get('type'), protein_letters_3to1[residue.type]}
          for residue in background}
    bkgnd_directives = dict(zip(background.keys(), repeat(None)))

    directives = [bkgnd_directives.copy() for _ in job.directives]
    for idx, design_directives in enumerate(self.directives):
        directives[idx].update({residue: design_directives[residue.number]
                                for residue in job.pose.get_residues(design_directives.keys())})

    res_files = [job.pose.make_resfile(directives_, out_path=job.data_path, include=wt, background=background)
                 for directives_ in directives]

    # nstruct_instruct = ['-no_nstruct_label', 'true']
    nstruct_instruct = ['-nstruct', str(job.job.design.number)]
    generate_files_cmd = \
        ['python', putils.list_pdb_files, '-d', job.designs_path, '-o', designed_files_file,  '-e', '.pdb',
         '-s', f'_{job.protocol}']

    main_cmd = rosetta.script_cmd.copy()
    if job.symmetry_dimension is not None and job.symmetry_dimension > 0:
        main_cmd += ['-symmetry_definition', 'CRYST1']
    if not os.path.exists(job.flags) or job.job.force:
        job.prepare_rosetta_flags(out_dir=job.scripts_path)
        job.log.debug(f'Pose flags written to: {job.flags}')

    # DESIGN: Prepare command and flags file
    # Todo - Has this been solved?
    #  must set up a blank -in:file:pssm in case the evolutionary matrix is not used. Design will fail!!
    profile_cmd = ['-in:file:pssm', job.evolutionary_profile_file] \
        if os.path.exists(job.evolutionary_profile_file) else []
    design_cmds = []
    for res_file in res_files:
        design_cmds.append(
            main_cmd + profile_cmd + infile  # Todo this must be in:file:s
            + [f'@{job.flags}', '-out:suffix', f'_{job.protocol}', '-packing:resfile', res_file, '-parser:protocol',
               os.path.join(putils.rosetta_scripts_dir, f'{protocol_xml1}.xml')]
            + nstruct_instruct)

    # metrics_pdb = ['-in:file:l', designed_files_file]  # job.pdb_list]
    # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
    # Assumes all entity chains are renamed from A to Z for entities (1 to n)
    # metric_cmd = main_cmd + ['-in:file:s', job.specific_design if job.specific_design else job.refined_pdb] + \
    entity_cmd = main_cmd + ['-in:file:l', designed_files_file] + \
        [f'@{job.flags}', '-out:file:score_only', job.scores_file, '-no_nstruct_label', 'true',
         '-parser:protocol', os.path.join(putils.rosetta_scripts_dir, 'metrics_entity.xml')]

    if job.job.mpi > 0:
        design_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(job.job.mpi)] + design_cmd
        entity_cmd = rosetta.run_cmds[putils.rosetta_extras] + [str(job.job.mpi)] + entity_cmd

    job.log.info(f'{optimize_designs.__name__} command: {list2cmdline(design_cmd)}')
    metric_cmds = []
    metric_cmds.extend(job.generate_entity_metrics_commands(entity_cmd))

    # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics
    if job.job.distribute_work:
        analysis_cmd = job.make_analysis_cmd()
        write_shell_script(list2cmdline(design_cmd), name=job.protocol, out_path=job.scripts_path,
                           additional=[list2cmdline(generate_files_cmd)] +
                                      [list2cmdline(command) for command in metric_cmds] +
                                      [list2cmdline(analysis_cmd)])
    else:
        design_process = Popen(design_cmd)
        design_process.communicate()  # wait for command to complete
        list_all_files_process = Popen(generate_files_cmd)
        list_all_files_process.communicate()
        for metric_cmd in metric_cmds:
            metrics_process = Popen(metric_cmd)
            metrics_process.communicate()

        # Gather metrics for each design produced from this proceedure
        if os.path.exists(job.scores_file):
            job.process_rosetta_metrics()


@protocol_decorator()
def process_rosetta_metrics(job: pose.PoseJob):
    """From Rosetta based protocols, tally the resulting metrics and integrate with the metrics database

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    # job.load_pose()
    job.identify_interface()
    # Now acquiring in process_rosetta_metrics()
    # # Acquire the pose_metrics if None have been made yet
    # job.calculate_pose_metrics()
    if os.path.exists(job.scores_file):
        job.process_rosetta_metrics()
    else:
        raise DesignError(f'No scores from Rosetta present at "{job.scores_file}"')


@protocol_decorator()
def analysis(job: pose.PoseJob, designs: Iterable[Pose] | Iterable[AnyStr] = None) -> pd.Series:
    """Retrieve all score information from a PoseJob and write results to .csv file

    Args:
        job: The PoseJob for which the protocol should be performed on
        designs: The subsequent designs to perform analysis on
    Returns:
        Series containing summary metrics for all designs in the design directory
    """
    # job.load_pose()
    job.identify_interface()
    # Acquire the pose_metrics if None have been made yet
    job.calculate_pose_metrics()

    return job.analyze_pose_designs(designs=designs)


# @remove_structure_memory  # NO structures used in this protocol
@protocol_decorator()
def select_sequences(job: pose.PoseJob, filters: dict = None, weights: dict = None, number: int = 1,
                     protocols: list[str] = None, **kwargs) -> list[str]:
    """Select sequences for further characterization. If weights, then user can prioritize by metrics, otherwise
    sequence with the most neighbors as calculated by sequence distance will be selected. If there is a tie, the
    sequence with the lowest weight will be selected

    Args:
        job: The PoseJob for which the protocol should be performed on
        filters: The filters to use in sequence selection
        weights: The weights to use in sequence selection
        number: The number of sequences to consider for each design
        protocols: Whether particular design protocol(s) should be chosen
    Keyword Args:
        default_weight: str = 'interface_energy': The metric to sort the dataframe by default if no weights are provided
    Returns:
        The selected designs for the Pose trajectories
    """
    # Load relevant data from the design directory
    designs_df = pd.read_csv(job.designs_metrics_csv, index_col=0, header=[0])
    designs_df.dropna(inplace=True)
    if protocols:
        designs = []
        for protocol in protocols:
            designs.extend(designs_df[designs_df['protocol'] == protocol].index.to_list())

        if not designs:
            raise DesignError(f'No designs found for protocols {protocols}!')
    else:
        designs = designs_df.index.to_list()

    job.log.info(f'Number of starting trajectories = {len(designs_df)}')
    df = designs_df.loc[designs, :]

    if filters:
        job.log.info(f'Using filter parameters: {filters}')
        # Filter the DataFrame to include only those values which are le/ge the specified filter
        filtered_designs = metrics.index_intersection(
            metrics.filter_df_for_index_by_value(df, filters).values())
        df = df.loc[filtered_designs, :]

    if weights:
        # No filtering of protocol/indices to use as poses should have similar protocol scores coming in
        job.log.info(f'Using weighting parameters: {weights}')
        designs = metrics.pareto_optimize_trajectories(df, weights=weights, **kwargs).index.to_list()
    else:
        # sequences_pickle = glob(os.path.join(job.job.all_scores, '%s_Sequences.pkl' % str(job)))
        # assert len(sequences_pickle) == 1, 'Couldn\'t find files for %s' % \
        #                                     os.path.join(job.job.all_scores, '%s_Sequences.pkl' % str(job))
        #
        # chain_sequences = SDUtils.unpickle(sequences_pickle[0])
        # {chain: {name: sequence, ...}, ...}
        # designed_sequences_by_entity: list[dict[str, str]] = unpickle(job.designed_sequences)
        # designed_sequences_by_entity: list[dict[str, str]] = job.designed_sequences
        # entity_sequences = list(zip(*[list(designed_sequences.values())
        #                               for designed_sequences in designed_sequences_by_entity]))
        # concatenated_sequences = [''.join(entity_sequence) for entity_sequence in entity_sequences]
        pose_sequences = job.designed_sequences
        job.log.debug(f'The final concatenated sequences are:\n{pose_sequences}')

        # pairwise_sequence_diff_np = SDUtils.all_vs_all(concatenated_sequences, sequence_difference)
        # Using concatenated sequences makes the values very similar and inflated as most residues are the same
        # doing min/max normalization to see variation
        pairwise_sequence_diff_l = [sequence_difference(*seq_pair)
                                    for seq_pair in combinations(pose_sequences, 2)]
        pairwise_sequence_diff_np = np.array(pairwise_sequence_diff_l)
        _min = min(pairwise_sequence_diff_l)
        # max_ = max(pairwise_sequence_diff_l)
        pairwise_sequence_diff_np = np.subtract(pairwise_sequence_diff_np, _min)
        # job.log.info(pairwise_sequence_diff_l)

        # PCA analysis of distances
        pairwise_sequence_diff_mat = np.zeros((len(designs), len(designs)))
        for k, dist in enumerate(pairwise_sequence_diff_np):
            i, j = condensed_to_square(k, len(designs))
            pairwise_sequence_diff_mat[i, j] = dist
        pairwise_sequence_diff_mat = sym(pairwise_sequence_diff_mat)

        pairwise_sequence_diff_mat = skl.preprocessing.StandardScaler().fit_transform(pairwise_sequence_diff_mat)
        seq_pca = skl.decomposition.PCA(default_pca_variance)
        seq_pc_np = seq_pca.fit_transform(pairwise_sequence_diff_mat)
        seq_pca_distance_vector = pdist(seq_pc_np)
        # epsilon = math.sqrt(seq_pca_distance_vector.mean()) * 0.5
        epsilon = seq_pca_distance_vector.mean() * 0.5
        job.log.info(f'Finding maximum neighbors within distance of {epsilon}')

        # job.log.info(pairwise_sequence_diff_np)
        # epsilon = pairwise_sequence_diff_mat.mean() * 0.5
        # epsilon = math.sqrt(seq_pc_np.myean()) * 0.5
        # epsilon = math.sqrt(pairwise_sequence_diff_np.mean()) * 0.5

        # Find the nearest neighbors for the pairwise-distance matrix using the X*X^T (PCA) matrix, linear transform
        seq_neighbors = skl.neighbors.BallTree(seq_pc_np)
        seq_neighbor_counts = seq_neighbors.query_radius(seq_pc_np, epsilon, count_only=True)  # sort_results=True)
        top_count, top_idx = 0, None
        count_list = seq_neighbor_counts.tolist()
        for count in count_list:  # idx, enumerate()
            if count > top_count:
                top_count = count

        sorted_seqs = sorted(count_list, reverse=True)
        top_neighbor_counts = sorted(set(sorted_seqs[:number]), reverse=True)

        # Find only the designs which match the top x (number) of neighbor counts
        final_designs = {designs[idx]: num_neighbors for num_neighbors in top_neighbor_counts
                         for idx, count in enumerate(count_list) if count == num_neighbors}
        job.log.info('The final sequence(s) and file(s):\nNeighbors\tDesign\n%s'
                     # % '\n'.join('%d %s' % (top_neighbor_counts.index(neighbors) + SDUtils.zero_offset,
                     % '\n'.join(f'\t{neighbors}\t{os.path.join(job.designs_path, _design)}'
                                 for _design, neighbors in final_designs.items()))

        # job.log.info('Corresponding PDB file(s):\n%s' % '\n'.join('%d %s' % (i, os.path.join(job.designs_path, seq))
        #                                                         for i, seq in enumerate(final_designs, 1)))

        # Compute the highest density cluster using DBSCAN algorithm
        # seq_cluster = DBSCAN(eps=epsilon)
        # seq_cluster.fit(pairwise_sequence_diff_np)
        #
        # seq_pc_df = pd.DataFrame(seq_pc, index=designs, columns=['pc' + str(x + SDUtils.zero_offset)
        #                                                          for x in range(len(seq_pca.components_))])
        # seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)

        # If final designs contains more sequences than specified, find the one with the lowest energy
        if len(final_designs) > number:
            energy_s = df.loc[final_designs.keys(), 'interface_energy']
            energy_s.sort_values(inplace=True)
            designs = energy_s.index.to_list()
        else:
            designs = list(final_designs.keys())

    designs = designs[:number]
    job.log.info(f'Final ranking of trajectories:\n{", ".join(_design for _design in designs)}')
    return designs
