from __future__ import annotations

import csv
import logging
import os
from collections.abc import Iterable
from itertools import count, repeat, combinations
from subprocess import list2cmdline, Popen
from typing import AnyStr

import numpy as np
import pandas as pd
import sklearn as skl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.spatial.distance import pdist

from . import align, cluster, config, fragdock, pose, select
from .utils import handle_job_errors, protocol_decorator, warn_missing_symmetry
from symdesign import flags, metrics
from symdesign.resources.config import default_pca_variance
from symdesign.resources import distribute
from symdesign.resources.job import job_resources_factory  # job has namespace overloaded
from symdesign.sequence import optimize_protein_sequence, protein_letters_1to3, protein_letters_3to1, \
    read_fasta_file, write_sequences
from symdesign.structure.model import Pose
from symdesign.structure.sequence import write_pssm_file, sequence_difference
from symdesign.structure.utils import SymmetryError
from symdesign.utils import condensed_to_square, get_directory_file_paths, InputError, path as putils, \
    rosetta, starttime, sym

# Globals
logger = logging.getLogger(__name__)

# Protocols
align_helices = handle_job_errors()(align.align_helices)
nanohedra = handle_job_errors()(fragdock.fragment_dock)
cluster_poses = handle_job_errors()(cluster.cluster_poses)
select_poses = handle_job_errors()(select.sql_poses)  # select.poses
select_designs = handle_job_errors()(select.sql_designs)  # select.designs
select_sequences = handle_job_errors()(select.sql_sequences)  # select.sequences


@protocol_decorator()
def predict_structure(job: pose.PoseJob):
    """From a sequence input, predict the structure using one of various structure prediction pipelines

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    # No need to load the yet as the prediction only uses sequence. Load the design in individual method when needed
    # job.identify_interface()
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
    raise NotImplementedError('This module is outdated, please update it if you would like to use it')
    job.identify_interface()

    cmd = rosetta.script_cmd.copy()
    script_name = os.path.splitext(os.path.basename(script))[0]

    job.prepare_rosetta_flags(out_dir=job.scripts_path)

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
        analysis_cmd = job.get_cmd_process_rosetta_metrics()
        job.current_script = distribute.write_script(
            list2cmdline(generate_files_cmd), name=f'{starttime}_{script_name}.sh', out_path=job.scripts_path,
            additional=[list2cmdline(cmd)] + [list2cmdline(analysis_cmd)])
        # Todo metrics: [list2cmdline(command) for command in metric_cmds]
    else:
        list_all_files_process = Popen(generate_files_cmd)
        list_all_files_process.communicate()
        # Todo
        # for metric_cmd in metric_cmds:
        #     metrics_process = Popen(metric_cmd)
        #     metrics_process.communicate()  # wait for command to complete

        # Gather metrics for each design produced from this procedure
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
    job.protocol = flags.interface_metrics
    main_cmd = rosetta.script_cmd.copy()

    job.prepare_rosetta_flags(out_dir=job.scripts_path)

    if job.current_designs:
        file_paths = [design_.structure_path for design_ in job.current_designs if design_.structure_path]
    else:
        file_paths = get_directory_file_paths(
            job.designs_path, suffix=job.job.specific_protocol if job.job.specific_protocol else '', extension='.pdb')
    # Include the pose source in the designs to perform metrics on
    if job.job.measure_pose and os.path.exists(job.pose_path):
        file_paths.append(job.pose_path)
    # If no designs specified or found and the pose_path exists, add it
    # The user probably wants pose metrics without specifying so
    elif not file_paths:
        with job.job.db.session(expire_on_commit=False) as session:
            session.add(job)
            if not job.designs and os.path.exists(job.pose_path):
                file_paths.append(job.pose_path)

    if not file_paths:
        raise InputError(
            f'No files found for {job.job.module}')

    design_files = \
        os.path.join(job.scripts_path, f'{starttime}_{job.protocol}_files'
                     f'{f"_{job.job.specific_protocol}" if job.job.specific_protocol else ""}.txt')
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
        [os.path.join(putils.rosetta_scripts_dir, f'interface_metrics{"_DEV" if job.job.development else ""}.xml')]
    job.log.info(f'Metrics command for Pose: {list2cmdline(metric_cmd_bound)}')
    entity_cmd = main_cmd + [os.path.join(putils.rosetta_scripts_dir,
                                          f'metrics_entity{"_DEV" if job.job.development else ""}.xml')]
    entity_metric_cmds = job.generate_entity_metrics_commands(entity_cmd)

    # Create executable to gather interface Metrics on all Designs
    if job.job.distribute_work:
        analysis_cmd = job.get_cmd_process_rosetta_metrics()
        job.current_script = distribute.write_script(
            list2cmdline(metric_cmd_bound), name=f'{starttime}_{job.protocol}.sh', out_path=job.scripts_path,
            additional=[list2cmdline(command) for command in entity_metric_cmds]
            + [list2cmdline(analysis_cmd)])
    else:
        # list_all_files_process = Popen(generate_files_cmd)
        # list_all_files_process.communicate()
        for metric_cmd in [metric_cmd_bound] + entity_metric_cmds:
            metrics_process = Popen(metric_cmd)
            metrics_process.communicate()  # wait for command to complete

        # Gather metrics for each design produced from this procedure
        if os.path.exists(job.scores_file):
            job.process_rosetta_metrics()


@protocol_decorator()
def check_unmodeled_clashes(job: pose.PoseJob, clashing_threshold: float = 0.75):
    """Given a multimodel file, measure the number of clashes is less than a percentage threshold

    Args:
        job: The PoseJob for which the protocol should be performed on
        clashing_threshold: The number of Model instances which have observed clashes
    """
    raise NotImplementedError("This module currently isn't working")
    from symdesign.structure.model import MultiModel, Model
    models = [Model.from_file(job.job.structure_db.full_models.retrieve_data(name=entity), log=job.log)
              for entity in job.entity_names]
    # models = [Models.from_file(job.job.structure_db.full_models.retrieve_data(name=entity))
    #           for entity in job.entity_names]

    # For each model, transform to the correct space
    models = job.transform_structures_to_pose(models)
    multimodel = MultiModel.from_models(models, independent=True, log=job.log)

    clashes = prior_clashes = 0
    for idx, state in enumerate(multimodel, 1):
        clashes += (1 if state.is_clash(measure=job.job.design.clash_criteria,
                                        distance=job.job.design.clash_distance) else 0)
        state.write(out_path=os.path.join(job.path, f'state_{idx}.pdb'))
        logger.info(f'State {idx} - Clashes: {"YES" if clashes > prior_clashes else "NO"}')
        prior_clashes = clashes

    if clashes / float(len(multimodel)) > clashing_threshold:
        raise ClashError(
            f'The frequency of clashes ({clashes / float(len(multimodel))}) exceeds the clashing '
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
    job.load_pose()
    job.pose.rename_chains()
    job.output_pose(force=True)


# @protocol_decorator()
# def find_asu(job: pose.PoseJob):
#     """From a PDB with multiple Chains from multiple Entities, return the minimal configuration of Entities.
#     ASU will only be a true ASU if the starting PDB contains a symmetric system, otherwise all manipulations find
#     the minimal unit of Entities that are in contact
#
#     Args:
#         job: The PoseJob for which the protocol should be performed on
#     """
#     # Check if the symmetry is known, otherwise this wouldn't work without the old, "symmetry-less", protocol
#     if job.is_symmetric():
#         if os.path.exists(job.assembly_path):
#             job.load_pose(file=job.assembly_path)
#         else:
#             job.load_pose()
#
#     # Save the Pose.asu
#     job.output_pose()


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
        raise SymmetryError(
            warn_missing_symmetry % expand_asu.__name__)


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
    job.identify_interface()
    job.protocol = job.job.module
    # Todo
    #  Make a common helper. PoseJob.get_active_structure_paths()...
    if job.current_designs:
        file_paths = [design_.structure_path for design_ in job.current_designs if design_.structure_path]
    else:
        file_paths = get_directory_file_paths(
            job.designs_path, suffix=job.job.specific_protocol if job.job.specific_protocol else '', extension='.pdb')
    # Include the pose source in the designs to perform metrics on
    if job.job.measure_pose and os.path.exists(job.pose_path):
        file_paths.append(job.pose_path)
    # If no designs specified or found and the pose_path exists, add it
    # The user probably wants pose metrics without specifying so
    elif not file_paths:
        with job.job.db.session(expire_on_commit=False) as session:
            session.add(job)
            if not job.designs and os.path.exists(job.pose_path):
                file_paths.append(job.pose_path)

    if not file_paths:
        raise InputError(
            f'No files found for {job.job.module}')

    job.refine(design_files=file_paths)  # Inherently utilized... gather_metrics=job.job.metrics)


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
    if job.job.design.method == putils.rosetta:
        # Update upon completion given results of designs list file...
        # NOT # Update the Pose with the number of designs
        # raise NotImplementedError('Need to generate design_number matching job.proteinmpnn_design()...')
        # job.update_design_data(design_parent=job.pose_source, number=job.job.design.number)
        if job.job.design.interface:
            pass
        else:
            raise NotImplementedError(
                f"Can't perform module 'design' using Rosetta. Try '{flags.interface_design}' instead")
        favor_fragments = evo_fill = True
        if job.job.design.term_constraint:
            # Todo
            #  Need to get oligomeric type frags
            job.generate_fragments(interface=True)  # job.job.design.interface
    elif job.job.design.method == putils.consensus:
        raise NotImplementedError('Consensus calculation needs work')
        # -------------------------------------------------------------------------
        # Todo job.solve_consensus()
        # -------------------------------------------------------------------------
    else:
        favor_fragments = evo_fill = False
        # Todo
        #  The information isn't really used. Also need to get oligomeric type frags
        job.generate_fragments(interface=True)  # job.job.design.interface

    job.pose.calculate_fragment_profile(evo_fill=evo_fill)
    # elif isinstance(job.fragment_observations, list):
    #     raise NotImplementedError(f"Can't put fragment observations taken away from the pose onto the pose due to "
    #                               f"entities")
    #     job.pose.calculate_fragment_profile(evo_fill=evo_fill)
    # elif os.path.exists(job.frag_file):
    #     job.retrieve_fragment_info_from_file()

    job.set_up_evolutionary_profile()

    # job.pose.combine_sequence_profiles()
    # I could also add the combined profile here instead of at each Entity
    job.pose.calculate_profile(favor_fragments=favor_fragments)
    putils.make_path(job.designs_path)
    # Acquire the pose_metrics if None have been made yet
    job.calculate_pose_metrics()

    # match job.job.design.method:  # Todo python 3.10
    #     case [putils.rosetta_str | putils.consensus]:
    #         # Write generated files
    #         write_pssm_file(job.pose.evolutionary_profile, file_name=job.evolutionary_profile_file)
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
    if job.job.design.method in [putils.rosetta, putils.consensus]:
        # Write generated files
        write_pssm_file(job.pose.evolutionary_profile, file_name=job.evolutionary_profile_file)
        write_pssm_file(job.pose.profile, file_name=job.design_profile_file)
        job.pose.fragment_profile.write(file_name=job.fragment_profile_file)
        # Ensure the Pose is refined into the current_energy_function
        if not job.refined and not os.path.exists(job.refined_pdb):
            job.refine(gather_metrics=False)
        if job.job.design.interface:
            raise NotImplementedError('Need to generate job.number_of_designs matching job.proteinmpnn_design()...')
            # Todo update upon completion given results of designs list file...
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
    job.protocol = protocol_xml1 = flags.optimize_designs._
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
    raise NotImplementedError('Must make the infile an "in:file:s" derivative')
    designed_files_file = os.path.join(job.scripts_path, f'{starttime}_{job.protocol}_files_output.txt')
    if job.current_designs:
        design_ids = [design_.id for design_ in job.current_designs]
        design_files = [design_.structure_file for design_ in job.current_designs]
        design_poses = [Pose.from_file(file, **job.pose_kwargs) for file in design_files]
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

    # Format all amino acids in design with frequencies above the threshold to a set
    # Todo, make threshold and return set of strings a property of a profile object
    # Locate the desired background profile from the pose
    background_profile = getattr(job.pose, job.job.background_profile)
    raise NotImplementedError("Chain.*_profile all need to be zero-index to account for residue.index")
    background = {residue: {aaa for a, aaa in protein_letters_1to3.items()
                            if background_profile[residue.index].get(a, -1) > threshold}
                  for residue in job.pose.residues}
    # Include the wild-type residue from PoseJob.pose and the residue identity of the selected design
    wt = {residue: {background_profile[residue.index].get('type'), protein_letters_3to1[residue.type]}
          for residue in background}
    bkgnd_directives = dict(zip(background.keys(), repeat(None)))

    design_directives = []
    # design_directives = [bkgnd_directives.copy() for _ in job.directives]
    for design_pose, directive in zip(design_poses, job.directives):
        # Grab those residues from background that are considered designed in the current design
        raise NotImplementedError(
            f"Must only use the background amino acid types for the positions that were marked as "
            f"designable for each design in job.current_designs")
        # Todo
        #  with job.job.db.session(expire_on_commit=False) as session:
        #      designable_residues_stmt = select((sql.DesignResidues.design_id, sql.DesignResidues.index)) \
        #          .where(sql.DesignResidues.design_id.in_(design_ids)) \
        #          .where(sql.DesignResidues.design_residue == True)
        #      designable_residues = session.execute(designable_residues_stmt).all()
        design_directive = {residue: background[residue] for residue in design_designed_residues}
        design_directive.update({residue: directive[residue.index]
                                for residue in design_pose.get_residues(directive.keys())})
        design_directives.append(design_directive)
        # design_directive.update({residue: directive[residue.index]
        #                          for residue in job.pose.get_residues(directive.keys())})

    res_files = [design_pose.make_resfile(directive, out_path=job.data_path, include=wt, background=background)
                 for design_pose, directive in zip(design_poses, design_directives)]

    # nstruct_instruct = ['-no_nstruct_label', 'true']
    nstruct_instruct = ['-nstruct', str(job.job.design.number)]
    generate_files_cmd = \
        ['python', putils.list_pdb_files, '-d', job.designs_path, '-o', designed_files_file,  '-e', '.pdb',
         '-s', f'_{job.protocol}']

    main_cmd = rosetta.script_cmd.copy()
    if job.symmetry_dimension is not None and job.symmetry_dimension > 0:
        main_cmd += ['-symmetry_definition', 'CRYST1']

    job.prepare_rosetta_flags(out_dir=job.scripts_path)

    # DESIGN: Prepare command and flags file
    # Todo - Has this been solved?
    #  must set up a blank -in:file:pssm in case the evolutionary matrix is not used. Design will fail!!
    profile_cmd = ['-in:file:pssm', job.evolutionary_profile_file] \
        if os.path.exists(job.evolutionary_profile_file) else []
    design_cmds = []
    for res_file, design_ in zip(res_files, job.current_designs):
        design_cmds.append(
            main_cmd + profile_cmd + infile  # Todo this must be 'in:file:s'
            + [f'@{job.flags}', '-out:suffix', f'_{job.protocol}', '-packing:resfile', res_file, '-parser:protocol',
               os.path.join(putils.rosetta_scripts_dir, f'{protocol_xml1}.xml')]
            + nstruct_instruct + ['-parser:script_vars', f'{putils.design_parent}={design_.name}'])

    # metrics_pdb = ['-in:file:l', designed_files_file]
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
        analysis_cmd = job.get_cmd_process_rosetta_metrics()
        self.current_script = distribute.write_script(
            list2cmdline(design_cmd), name=f'{starttime}_{job.protocol}.sh', out_path=job.scripts_path,
            additional=[list2cmdline(generate_files_cmd)]
            + [list2cmdline(command) for command in metric_cmds] + [list2cmdline(analysis_cmd)])
    else:
        design_process = Popen(design_cmd)
        design_process.communicate()  # wait for command to complete
        list_all_files_process = Popen(generate_files_cmd)
        list_all_files_process.communicate()
        for metric_cmd in metric_cmds:
            metrics_process = Popen(metric_cmd)
            metrics_process.communicate()

        # Gather metrics for each design produced from this procedure
        if os.path.exists(job.scores_file):
            job.process_rosetta_metrics()


@protocol_decorator()
def process_rosetta_metrics(job: pose.PoseJob) -> None:
    """From Rosetta based protocols, tally the resulting metrics and integrate with the metrics database

    Args:
        job: The PoseJob for which the protocol should be performed on
    """
    # Gather metrics for each design produced from other modules
    if os.path.exists(job.scores_file):
        job.identify_interface()
        job.process_rosetta_metrics()
    else:
        raise InputError(
            f'No scores from Rosetta present at "{job.scores_file}"')


@protocol_decorator()
def analysis(job: pose.PoseJob, designs: Iterable[Pose] | Iterable[AnyStr] = None) -> None:
    """Retrieve all metrics information from a PoseJob

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

    job.analyze_pose_designs(designs=designs)


@protocol_decorator()
def helix_bending(job: pose.PoseJob):
    """Retrieve all metrics information from a PoseJob

    Args:
        job: The PoseJob for which the protocol should be performed on
    Returns:
        Series containing summary metrics for all designs in the design directory
    """
    job.load_pose()
    if job.job.joint_chain:  # A chain designation was provided
        model_to_select = job.pose.chain(job.job.joint_chain)
    else:  # Just use the residue number to select
        model_to_select = job.pose

    if job.job.output_directory:
        putils.make_path(job.job.output_directory)
        out_dir = job.job.output_directory
    else:
        out_dir = job.pose_directory

    joint_residue = model_to_select.residue(job.job.joint_residue)
    bent_coords = align.bend(job.pose, joint_residue, job.job.direction, samples=job.job.sample_number)

    output_number = count(1)
    for bent_idx, coords in enumerate(bent_coords, 1):
        job.pose.coords = coords
        # Check for clashes
        if job.pose.is_clash(measure=job.job.design.clash_criteria,
                             distance=job.job.design.clash_distance, silence_exceptions=True):
            logger.info(f'Bend index {bent_idx} clashes')
            continue
        if job.pose.is_symmetric() and not job.job.design.ignore_symmetric_clashes and \
                job.pose.symmetric_assembly_is_clash(measure=job.job.design.clash_criteria,
                                                     distance=job.job.design.clash_distance):
            logger.info(f'Bend index {bent_idx} has symmetric clashes')
            continue

        trial_path = os.path.join(out_dir, f'{job.name}-bent{next(output_number)}.pdb')
        job.pose.write(out_path=trial_path)


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
            designs.extend(designs_df[designs_df['protocol'] == protocol].index.tolist())

        if not designs:
            raise InputError(
                f'No designs found for protocols {protocols}!')
    else:
        designs = designs_df.index.tolist()

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
        designs = metrics.pareto_optimize_trajectories(df, weights=weights, **kwargs).index.tolist()
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
                     # % '\n'.join('%d %s' % (top_neighbor_counts.index(neighbors) + 1,
                     % '\n'.join(f'\t{neighbors}\t{os.path.join(job.designs_path, _design)}'
                                 for _design, neighbors in final_designs.items()))

        # job.log.info('Corresponding PDB file(s):\n%s' % '\n'.join('%d %s' % (i, os.path.join(job.designs_path, seq))
        #                                                         for i, seq in enumerate(final_designs, 1)))

        # Compute the highest density cluster using DBSCAN algorithm
        # seq_cluster = DBSCAN(eps=epsilon)
        # seq_cluster.fit(pairwise_sequence_diff_np)
        #
        # seq_pc_df = pd.DataFrame(seq_pc, index=designs, columns=['pc' + str(x + 1)
        #                                                          for x in range(len(seq_pca.components_))])
        # seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)

        # If final designs contains more sequences than specified, find the one with the lowest energy
        if len(final_designs) > number:
            energy_s = df.loc[final_designs.keys(), 'interface_energy']
            energy_s.sort_values(inplace=True)
            designs = energy_s.index.tolist()
        else:
            designs = list(final_designs.keys())

    designs = designs[:number]
    job.log.info(f'Final ranking of trajectories:\n{", ".join(_design for _design in designs)}')
    return designs


def create_mulitcistronic_sequences(args):
    # if not args.multicistronic_intergenic_sequence:
    #     args.multicistronic_intergenic_sequence = expression.ncoI_multicistronic_sequence
    # raise NotImplementedError('Please refactor to a protocols/tools module so that JobResources can be used.')
    job = job_resources_factory()
    file = args.file[0]  # Since args.file is collected with nargs='*', select the first
    if file.endswith('.csv'):
        with open(file) as f:
            protein_sequences = [SeqRecord(Seq(name_sequence[1]), annotations={'molecule_type': 'Protein'},
                                           id=name_sequence[0]) for name_sequence in csv.reader(f)]
    elif file.endswith('.fasta'):
        protein_sequences = list(read_fasta_file(file))
    else:
        raise NotImplementedError(f'Sequence file with extension {os.path.splitext(file)[-1]} is not supported!')

    # Convert the SeqRecord to a plain sequence
    # design_sequences = [str(seq_record.seq) for seq_record in design_sequences]
    nucleotide_sequences = {}
    for idx, group_start_idx in enumerate(list(range(len(protein_sequences)))[::args.number], 1):
        # Call attribute .seq to get the sequence
        cistronic_sequence = optimize_protein_sequence(protein_sequences[group_start_idx].seq,
                                                       species=args.optimize_species)
        for protein_sequence in protein_sequences[group_start_idx + 1: group_start_idx + args.number]:
            cistronic_sequence += args.multicistronic_intergenic_sequence
            cistronic_sequence += optimize_protein_sequence(protein_sequence.seq,
                                                            species=args.optimize_species)
        new_name = f'{protein_sequences[group_start_idx].id}_cistronic'
        nucleotide_sequences[new_name] = cistronic_sequence
        logger.info(f'Finished sequence {idx} - {new_name}')

    location = file
    if not args.prefix:
        args.prefix = f'{os.path.basename(os.path.splitext(location)[0])}_'
    else:
        args.prefix = f'{args.prefix}_'

    # Format sequences for output
    putils.make_path(job.output_directory)
    nucleotide_sequence_file = write_sequences(nucleotide_sequences, csv=args.csv,
                                               file_name=os.path.join(job.output_directory,
                                                                      'MulticistronicNucleotideSequences'))
    logger.info(f'Multicistronic nucleotide sequences written to: {nucleotide_sequence_file}')
