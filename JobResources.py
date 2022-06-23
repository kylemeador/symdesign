from __future__ import annotations

import json
import os
from copy import copy
from glob import glob
from logging import Logger
from pathlib import Path
from subprocess import list2cmdline
from typing import List, Tuple, Iterable, Dict, Union, Optional, Any

import numpy as np
from Bio.Data.IUPACData import protein_letters

import SymDesignUtils as SDUtils
from CommandDistributer import rosetta_flags, script_cmd, distribute, relax_flags, rosetta_variables
from PDB import PDB, fetch_pdb_file, query_qs_bio
from PathUtils import monofrag_cluster_rep_dirpath, intfrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath, \
    frag_directory, sym_entry
from PathUtils import orient_log_file, rosetta_scripts, models_to_multimodel_exe, refine, biological_interfaces, \
    biological_fragment_db_pickle, all_scores, projects, sequence_info, data, output_oligomers, output_fragments, \
    structure_background, scout, generate_fragments, number_of_trajectories, nstruct, no_hbnet, \
    ignore_symmetric_clashes, ignore_pose_clashes, ignore_clashes, force_flags, no_evolution_constraint, \
    no_term_constraint, consensus
from Query.utils import boolean_choice
from SequenceProfile import parse_hhblits_pssm, MultipleSequenceAlignment, read_fasta_file  # parse_pssm
from Structure import parse_stride, Entity
from SymDesignUtils import DesignError, unpickle, get_base_root_paths_recursively, start_log, dictionary_lookup, \
    parameterize_frag_length
from classes.EulerLookup import EulerLookup
from classes.SymEntry import sdf_lookup, SymEntry
from utils.MysqlPython import Mysql

# import dependencies.bmdca as bmdca


# Globals
# for checking out the options to read and write Rosetta runs to a relational DB such as MySQL
# https://new.rosettacommons.org/docs/latest/rosetta_basics/options/Database-options
logger = start_log(name=__name__)
index_offset = 1


class Database:  # Todo ensure that the single object is completely loaded before multiprocessing... Queues and whatnot
    def __init__(self, oriented: str | bytes | Path = None, oriented_asu: str | bytes | Path = None,
                 refined: str | bytes | Path = None, full_models: str | bytes | Path = None,
                 stride: str | bytes | Path = None, sequences: str | bytes | Path = None,
                 hhblits_profiles: str | bytes | Path = None, pdb_api: str | bytes | Path = None,
                 uniprot_api: str | bytes | Path = None, sql=None, log: Logger = logger):  # sql: sqlite = None,
        if sql:
            raise DesignError('SQL set up has not been completed!')

        self.log = log
        self.oriented = DataStore(location=oriented, extension='.pdb*', sql=sql, log=log)
        self.oriented_asu = DataStore(location=oriented_asu, extension='.pdb', sql=sql, log=log)
        self.refined = DataStore(location=refined, extension='.pdb', sql=sql, log=log)
        self.full_models = DataStore(location=full_models, extension='_ensemble.pdb', sql=sql, log=log)
        self.stride = DataStore(location=stride, extension='.stride', sql=sql, log=log)
        self.sequences = DataStore(location=sequences, extension='.fasta', sql=sql, log=log)
        self.alignments = DataStore(location=hhblits_profiles, extension='.sto', sql=sql, log=log)
        self.hhblits_profiles = DataStore(location=hhblits_profiles, extension='.hmm', sql=sql, log=log)
        self.pdb_api = DataStore(location=pdb_api, extension='.json', sql=sql, log=log)
        self.uniprot_api = DataStore(location=uniprot_api, extension='.json', sql=sql, log=log)
        # self.bmdca_fields = \
        #     DataStore(location=hhblits_profiles, extension='_bmDCA%sparameters_h_final.bin' % os.sep, sql=sql, log=log)
        # self.bmdca_couplings = \
        #     DataStore(location=hhblits_profiles, extension='_bmDCA%sparameters_J_final.bin' % os.sep, sql=sql, log=log)

    def load_all_data(self):
        """For every resource, acquire all existing data in memory"""
        #              self.oriented_asu, self.sequences,
        for source in [self.stride, self.alignments, self.hhblits_profiles, self.oriented, self.refined]:  # self.full_models
            try:
                source.get_all_data()
            except ValueError:
                raise ValueError('Issue from source %s' % source)
        # self.log.debug('The data in the Database is: %s'
        #                % '\n'.join(str(store.__dict__) for store in self.__dict__.values()))

    def source(self, name):
        """Return on of the various DataStores supported by the Database"""
        try:
            return getattr(self, name)
        except AttributeError:
            raise AttributeError(f'There is no Database source named "{name}" found. '
                                 f'Possible sources are: {", ".join(self.__dict__)}')

    def retrieve_data(self, source: str = None, name: str = None) -> object | None:
        """Return the data requested by name from the specified source. Otherwise, load into the Database from a
        specified location

        Args:
            source: The name of the data source to use
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        object_db = self.source(source)
        data = getattr(object_db, name, None)
        if not data:
            object_db.name = object_db._load_data(name, log=None)
            data = object_db.name  # store the new data as an attribute

        return data

    def retrieve_file(self, from_source=None, name=None):
        """Retrieve the specified file on disk for subsequent parsing"""
        object_db = getattr(self, from_source, None)
        if not object_db:
            raise DesignError(f'There is no source named {from_source} found in the Design Database')

        return object_db.retrieve_file(name)

    def orient_entities(self, entity_ids: Iterable[str], symmetry: str = 'C1') -> List[Entity]:
        """Given entity_ids and their corresponding symmetry, retrieve .pdb files, orient and save Database files then
        return the ASU for each

        Args:
            entity_ids: The names of all entity_ids requiring orientation
            symmetry: The symmetry to treat each passed Entity. Default assumes no symmetry
        Returns:
            The resulting asymmetric Entities that have been oriented
        """
        self.oriented.make_path()
        orient_dir = self.oriented.location
        # os.makedirs(orient_dir, exist_ok=True)
        pdbs_dir = os.path.dirname(orient_dir)  # this is the case because it was specified this way, but not required
        self.oriented_asu.make_path()
        # orient_asu_dir = self.oriented_asu.location
        # os.makedirs(orient_asu_dir, exist_ok=True)
        self.stride.make_path()
        # stride_dir = self.stride.location
        # os.makedirs(stride_dir, exist_ok=True)
        # orient_log = SDUtils.start_log(name='orient', handler=1)
        orient_log = SDUtils.start_log(name='orient', handler=2,
                                       location=os.path.join(orient_dir, orient_log_file), propagate=True)
        orient_names = self.oriented.retrieve_names()
        orient_asu_names = self.oriented_asu.retrieve_names()
        all_entities = []
        logger.info(f'The requested files/IDs are being checked for proper orientation with symmetry {symmetry}: '
                    f'{", ".join(entity_ids)}')
        non_viable_structures = []
        for entry_entity in entity_ids:  # ex: 1ABC_1
            if entry_entity not in orient_names:  # add the proper files
                entry = entry_entity.split('_')
                # in case entry_entity is coming from a new SymDesign Directory the entity name is probably 1ABC_1
                if len(entry) == 2:
                    entry, entity = entry
                    logger.debug(f'Fetching entry {entry}, entity {entity} from PDB')
                else:
                    entry = entry_entity  # entry[0]
                    entity = None  # False
                    logger.debug(f'Fetching entry {entry} from PDB')

                if symmetry == 'C1':  # translate the monomer to the origin for the database
                    assembly = None  # 1 is the default
                    asu = True
                else:
                    asu = False
                    assembly = query_qs_bio(entry)
                # get the specified filepath for the assembly state of interest
                file_path = fetch_pdb_file(entry, assembly=assembly, asu=asu, out_dir=pdbs_dir)

                if not file_path:
                    logger.warning(f'Couldn\'t locate the file "{file_path}", there may have been an issue '
                                   'downloading it from the PDB. Attempting to copy from job data source...')
                    raise SDUtils.DesignError('This functionality hasn\'t been written yet. Use the '
                                              'canonical_pdb1/2 attribute of PoseDirectory to pull the'
                                              ' pdb file source.')
                    # Todo
                # remove any mirror specific naming from fetch_pdb_file
                file_name = os.path.splitext(os.path.basename(file_path))[0].replace('pdb', '')
                pdb = PDB.from_file(file_path, name=file_name)  # , log=None)
                if entity:  # replace PDB Structure from fetched file with the Entity Structure
                    # entity_pdb = pdb.entity(entry_entity).oligomer <- not quite as desired
                    entity = pdb.entity(entry_entity)
                    # print(','.join(entity.name for entity in pdb.entities))
                    entity_out_path = os.path.join(pdbs_dir, f'{entry_entity}.pdb')
                    if symmetry == 'C1':  # write out only entity
                        entity_file_path = entity.write(out_path=entity_out_path)
                    else:  # write out the entity as parsed. since this is assembly we should get the correct state
                        entity_file_path = entity.write_oligomer(out_path=entity_out_path)
                    # Todo make Entity capable of orient() then don't need this ugly mechanism
                    pdb = PDB.from_chains(entity.chains, entity_names=[entry_entity])  # , log=None)
                    pdb.entities = [entity]
                else:  # orient the whole set of chains based on orient() multicomponent solution
                    # entity = pdb.entities[0]  # assume that there is only one entity and grab the first
                    entry_entity = pdb.entities[0].name
                entry_entity_base = f'{entry_entity}.pdb'

                # if entity:  # ensure not none, otherwise, report
                #     if symmetry == 'C1':  # write out only entity
                #         entity_file_path = entity.write(out_path=os.path.join(pdbs_dir, entry_entity_base))
                #     else:  # write out the entity as parsed. since this is assembly we should get the correct state
                #         entity_file_path = entity.write_oligomer(out_path=os.path.join(pdbs_dir, entry_entity_base))
                #     pdb = PDB.from_chains(entity.chains, entity_names=[entry_entity])  # , log=None)
                # else:
                #     raise ValueError('No entity with the name %s found in file %s' % (entry_entity, pdb.filepath))

                # write out file for the orient database
                if symmetry == 'C1':  # translate the monomer to the origin
                    entity = pdb.entities[0]  # may be an issue if we passed a hetero dimer and treated as a C1
                    entity.translate(-entity.center_of_mass)
                    # entity.name = entry_entity
                    # orient_file = entity.write(out_path=os.path.join(orient_dir, entry_entity_base))
                    orient_file = entity.write(out_path=self.oriented.store(name=entity.name))
                    entity.symmetry = symmetry
                    entity.filepath = entity.write(out_path=self.oriented_asu.store(name=entity.name))
                    entity.stride(to_file=self.stride.store(name=entity.name))
                    all_entities.append(entity)  # .entities[0]
                else:
                    try:
                        pdb.orient(symmetry=symmetry, log=orient_log)
                        orient_file = pdb.write(out_path=os.path.join(orient_dir, entry_entity_base))
                        # Todo
                        #  orient_file = pdb.write(out_path=self.oriented.store(name=entity.name))
                        orient_log.info(f'Oriented: {orient_file}')
                    except (ValueError, RuntimeError) as err:
                        orient_log.error(str(err))
                        non_viable_structures.append(entry_entity)
                        continue
                    # extract the asu from the oriented file for symmetric refinement
                    # Todo include multiple entities if they are used...? Maybe these are coming from PDB
                    #  Need to move make_loop_file to Pose/Structure (with SequenceProfile superclass)
                    # all_entities.append(return_orient_asu(orient_file, entry_entity, symmetry))
                    entity = pdb.entities[0]
                    # entity.name = pdb.name  # use oriented_pdb.name (pdbcode_assembly), not API name
                    entity.symmetry = symmetry
                    entity.filepath = entity.write(out_path=self.oriented_asu.store(name=entity.name))
                    # save Stride results
                    entity.stride(to_file=self.stride.store(name=entity.name))
                    all_entities.append(entity)
            # for those below, entry_entity may not be the right format
            elif entry_entity not in orient_asu_names:  # orient file exists, but not asu or stride, create and load asu
                orient_file = self.oriented.retrieve_file(name=entry_entity)
                # all_entities.append(return_orient_asu(orient_file, symmetry))  # entry_entity,
                oriented_pdb = PDB.from_file(orient_file)  # , log=None, entity_names=[entry_entity])
                entity = oriented_pdb.entities[0]
                # entity = oriented_asu.entities[0]
                # entity.name = oriented_pdb.name  # use oriented_pdb.name (pdbcode_assembly), not API name
                entity.symmetry = symmetry
                entity.filepath = entity.write(out_path=self.oriented_asu.store(name=entity.name))
                # save Stride results
                entity.stride(to_file=self.stride.store(name=entity.name))
                all_entities.append(entity)  # entry_entity,
            else:  # orient_asu file exists, stride file should as well. load asu
                orient_asu_file = self.oriented_asu.retrieve_file(name=entry_entity)
                # all_entities.append(return_orient_asu(orient_file, entry_entity, symmetry))
                oriented_asu = PDB.from_file(orient_asu_file)  #, entity_names=[entry_entity])  # , log=None)
                # all_entities[oriented_asu.name] = oriented_asu.entities[0]
                entity = oriented_asu.entities[0]
                # entity.name = entry_entity  # make explicit
                entity.symmetry = symmetry
                entity.filepath = oriented_asu.filepath
                all_entities.append(entity)
        non_str = ', '.join(non_viable_structures)
        orient_log.error(f'The Entit{f"ies {non_str} were" if len(non_viable_structures) > 1 else f"y {non_str} was"} '
                         f'unable to be oriented properly')
        return all_entities

    def preprocess_entities_for_design(self, entities: list[Entity], script_out_path: str | bytes = os.getcwd(),
                                       load_resources: bool = False, batch_commands: bool = True) -> \
            tuple[list, bool, bool]:
        """Assess whether Entity objects require any processing prior to design calculations.
        Processing includes relaxation into the energy function and/or modelling missing loops and segments

        Args:
            entities: A collection of the Entity objects of interest
            script_out_path: Where should Entity processing commands be written?
            load_resources: Whether resources have been specified to be loaded already
            batch_commands: Whether commands should be made for batch submission
        Returns:
            Any instructions, then booleans for whether designs are pre_refined and whether they are pre_loop_modeled
        """
        self.refined.make_path()
        refine_names = self.refined.retrieve_names()
        refine_dir = self.refined.location
        self.full_models.make_path()
        full_model_names = self.full_models.retrieve_names()
        full_model_dir = self.full_models.location
        # Identify the entities to refine and to model loops before proceeding
        entities_to_refine, entities_to_loop_model, sym_def_files = [], [], {}
        for entity in entities:  # if entity is here, the file should've been oriented...
            # for entry_entity in entities:  # ex: 1ABC_1
            # symmetry = master_directory.sym_entry.sym_map[idx]
            # if entity.symmetry == 'C1':
            sym_def_files[entity.symmetry] = sdf_lookup() if entity.symmetry == 'C1' else sdf_lookup(entity.symmetry)
            # for entry_entity in entities:
            #     entry = entry_entity.split('_')
            # for orient_asu_file in oriented_asu_files:  # iterating this way to forgo missing "missed orient"
            #     base_pdb_code = os.path.splitext(orient_asu_file)[0]
            #     if base_pdb_code in entities:
            if entity.name not in refine_names:  # assumes oriented_asu entity name is the same
                entities_to_refine.append(entity)
            if entity.name not in full_model_names:  # assumes oriented_asu entity name is the same
                entities_to_loop_model.append(entity)

        # query user and set up commands to perform refinement on missing entities
        info_messages = []
        pre_refine = False
        if entities_to_refine:  # if files found unrefined, we should proceed
            logger.info('The following oriented entities are not yet refined and are being set up for refinement'
                        ' into the Rosetta ScoreFunction for optimized sequence design: '
                        f'{", ".join(set(entity.name for entity in entities_to_refine))}')
            print('Would you like to refine them now? If you plan on performing sequence design with models '
                  'containing them, it is highly recommended you perform refinement')
            if not boolean_choice():
                print('To confirm, asymmetric units are going to be generated with unrefined coordinates. Confirm '
                      'with "y" to ensure this is what you want')
                if not boolean_choice():
                    pre_refine = True
            else:
                pre_refine = True
            if pre_refine:
                # Generate sbatch refine command
                flags_file = os.path.join(refine_dir, 'refine_flags')
                # if not os.path.exists(flags_file):
                flags = copy(rosetta_flags) + relax_flags
                flags.extend([f'-out:path:pdb {refine_dir}', '-no_scorefile true'])
                flags.remove('-output_only_asymmetric_unit true')  # want full oligomers
                variables = copy(rosetta_variables)
                variables.append(('dist', 0))  # Todo modify if not point groups used
                flags.append('-parser:script_vars %s' % ' '.join(f'{var}={val}' for var, val in variables))

                with open(flags_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(flags))

                refine_cmd = [f'@{flags_file}', '-parser:protocol', os.path.join(rosetta_scripts, f'{refine}.xml')]
                refine_cmds = [script_cmd + refine_cmd + ['-in:file:s', entity.filepath, '-parser:script_vars'] +
                               [f'sdf={sym_def_files[entity.symmetry]}',
                                f'symmetry={"make_point_group" if entity.symmetry != "C1" else "asymmetric"}']
                               for entity in entities_to_refine]
                if batch_commands:
                    commands_file = \
                        SDUtils.write_commands([list2cmdline(cmd) for cmd in refine_cmds], out_path=refine_dir,
                                               name=f'{SDUtils.starttime}-refine_entities')
                    refine_sbatch = distribute(file=commands_file, out_path=script_out_path, scale=refine,
                                               log_file=os.path.join(refine_dir, f'{refine}.log'),
                                               max_jobs=int(len(refine_cmds) / 2 + 0.5),
                                               number_of_commands=len(refine_cmds))
                    refine_sbatch_message = \
                        'Once you are satisfied%snter the following to distribute refine jobs:\n\tsbatch %s' \
                        % (', you can run this script at any time. E' if load_resources else ', e', refine_sbatch)
                    info_messages.append(refine_sbatch_message)
                else:
                    raise DesignError('Entity refine run_in_shell functionality hasn\'t been implemented yet')
                load_resources = True

        # query user and set up commands to perform loop modelling on missing entities
        pre_loop_model = False
        if entities_to_loop_model:
            logger.info('The following structures have not been modelled for disorder. Missing loops will '
                        'be built for optimized sequence design: %s'
                        % ', '.join(set(entity.name for entity in entities_to_loop_model)))
            print('Would you like to model loops for these structures now? If you plan on performing sequence '
                  'design with them, it is highly recommended you perform loop modelling to avoid designed clashes')
            if not boolean_choice():
                print('To confirm, asymmetric units are going to be generated without disordered loops. Confirm '
                      'with "y" to ensure this is what you want')
                if not boolean_choice():
                    pre_loop_model = True
            else:
                pre_loop_model = True
            if pre_loop_model:
                # Generate sbatch refine command
                flags_file = os.path.join(full_model_dir, 'loop_model_flags')
                # if not os.path.exists(flags_file):
                loop_model_flags = ['-remodel::save_top 0', '-run:chain A', '-remodel:num_trajectory 1']
                #                   '-remodel:run_confirmation true', '-remodel:quick_and_dirty',
                flags = copy(rosetta_flags) + loop_model_flags
                # flags.extend(['-out:path:pdb %s' % full_model_dir, '-no_scorefile true'])
                flags.extend(['-no_scorefile true', '-no_nstruct_label true'])
                variables = [('script_nstruct', '100')]
                flags.append('-parser:script_vars %s' % ' '.join(f'{var}={val}' for var, val in variables))
                with open(flags_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(flags))
                loop_model_cmd = [f'@{flags_file}', '-parser:protocol',
                                  os.path.join(rosetta_scripts, 'loop_model_ensemble.xml'), '-parser:script_vars']
                # Make all output paths and files for each loop ensemble
                # logger.info('Preparing blueprint and loop files for entity:')
                # for entity in entities_to_loop_model:
                loop_model_cmds = []
                for idx, entity in enumerate(entities_to_loop_model):
                    entity_out_path = os.path.join(full_model_dir, entity.name)
                    SDUtils.make_path(entity_out_path)  # make a new directory for each entity
                    entity.renumber_residues()
                    entity_loop_file = entity.make_loop_file(out_path=full_model_dir)
                    if not entity_loop_file:  # no loops found, copy input file to the full model
                        copy_cmd = ['scp', self.refined.store(entity.name), self.full_models.store(entity.name)]
                        loop_model_cmds.append(SDUtils.write_shell_script(list2cmdline(copy_cmd), name=entity.name,
                                                                          out_path=full_model_dir))
                        continue
                    entity_blueprint = entity.make_blueprint_file(out_path=full_model_dir)
                    entity_cmd = script_cmd + loop_model_cmd + \
                        [f'blueprint={entity_blueprint}', f'loop_file={entity_loop_file}',
                         '-in:file:s', self.refined.store(entity.name), '-out:path:pdb', entity_out_path] + \
                        (['-symmetry:symmetry_definition', sym_def_files[entity.symmetry]] if entity.symmetry != 'C1'
                         else [])
                    # create a multimodel from all output loop models
                    multimodel_cmd = ['python', models_to_multimodel_exe, '-d', entity_loop_file,
                                      '-o', os.path.join(full_model_dir, f'{entity.name}_ensemble.pdb')]
                    # copy the first model from output loop models to be the full model
                    copy_cmd = ['scp', os.path.join(entity_out_path, f'{entity.name}_0001.pdb'),
                                self.full_models.store(entity.name)]
                    loop_model_cmds.append(
                        SDUtils.write_shell_script(list2cmdline(entity_cmd), name=entity.name, out_path=full_model_dir,
                                                   additional=[list2cmdline(multimodel_cmd), list2cmdline(copy_cmd)]))
                if batch_commands:
                    loop_cmds_file = \
                        SDUtils.write_commands(loop_model_cmds, name=f'{SDUtils.starttime}-loop_model_entities',
                                               out_path=full_model_dir)
                    loop_model_sbatch = distribute(file=loop_cmds_file, out_path=script_out_path, scale=refine,
                                                   log_file=os.path.join(full_model_dir, 'loop_model.log'),
                                                   max_jobs=int(len(loop_model_cmds) / 2 + 0.5),
                                                   number_of_commands=len(loop_model_cmds))
                    loop_model_sbatch_message = \
                        'Once you are satisfied%snter the following to distribute loop_modeling jobs:\n\tsbatch %s' \
                        % (', run this script AFTER completion of the Entity refinement script. E' if load_resources
                           else ', e', loop_model_sbatch)
                    info_messages.append(loop_model_sbatch_message)
                else:
                    raise DesignError('Entity refine run_in_shell functionality hasn\'t been implemented yet')

        return info_messages, pre_refine, pre_loop_model


def write_str_to_file(string, file_name, **kwargs) -> str | bytes:
    """Use standard file IO to write a string to a file

    Args:
        string: The string to write
        file_name: The location of the file to write
    Returns:
        The name of the written file
    """
    with open(file_name, 'w') as f_save:
        f_save.write(f'{string}\n')

    return file_name


def write_list_to_file(_list, file_name, **kwargs) -> str | bytes:
    """Use standard file IO to write a string to a file

    Args:
        _list: The string to write
        file_name: The location of the file to write
    Returns:
        The name of the written file
    """
    with open(file_name, 'w') as f_save:
        lines = '\n'.join(map(str, _list))
        f_save.write(f'{lines}\n')

    return file_name


def write_json(data, file_name, **kwargs) -> str | bytes:
    """Use json.dump to write an object to a file

    Args:
        data: The object to write
        file_name: The location of the file to write
    Returns:
        The name of the written file
    """
    with open(file_name, 'w') as f_save:
        json.dump(data, f_save, **kwargs)

    return file_name


not_implemented = \
    lambda data, file_name: (_ for _ in ()).throw(NotImplemented(f'For save_file with {os.path.splitext(file_name)[-1]}'
                                                                 f'DataStore method not available'))


class DataStore:
    def __init__(self, location: str = None, extension: str = '.txt', sql=None, log: Logger = logger):
        self.location = location
        self.extension = extension
        self.sql = sql
        self.log = log

        # load_file must be a callable which takes as first argument the file_name
        # save_file must be a callable which takes as first argument the object to save and second argument is file_name
        if '.pdb' in extension:
            self.load_file = PDB.from_file
            self.save_file = not_implemented
        elif '.json' in extension:
            self.load_file = json.load
            self.save_file = write_json
        elif extension == '.fasta':
            self.load_file = read_fasta_file
            self.save_file = write_sequence_to_fasta
        elif extension == '.stride':
            self.load_file = parse_stride
            self.save_file = not_implemented
        elif extension == '.hmm':  # in ['.hmm', '.pssm']:
            self.load_file = parse_hhblits_pssm  # parse_pssm
            self.save_file = not_implemented
        # elif extension == '.fasta' and msa:  # Todo if msa is in fasta format
        elif extension == '.sto':
            self.load_file = MultipleSequenceAlignment.from_stockholm  # parse_stockholm_to_msa
            self.save_file = not_implemented
        elif extension == f'_bmDCA{os.sep}parameters_h_final.bin':
            self.load_file = bmdca.load_fields
            self.save_file = not_implemented
        elif extension == f'_bmDCA{os.sep}sparameters_J_final.bin':
            self.load_file = bmdca.load_couplings
            self.save_file = not_implemented
        else:  # '.txt' read the file and return the lines
            self.load_file = self.read_file
            self.save_file = write_list_to_file

    def make_path(self, condition=True):
        """Make all required directories in specified path if it doesn't exist, and optional condition is True

        Keyword Args:
            condition=True (bool): A condition to check before the path production is executed
        """
        if condition:
            os.makedirs(self.location, exist_ok=True)

    def store(self, name: str = '*') -> str | bytes:  # Todo resolve with def store_data() below. This to path() -> Path
        """Return the path of the storage location given an entity name"""
        return os.path.join(self.location, f'{name}{self.extension}')

    def retrieve_file(self, name: str) -> str | bytes:
        """Returns the actual location by combining the requested name with the stored .location"""
        path = self.store(name)
        files = sorted(glob(path))
        if files:
            file = files[0]
            if len(files) > 1:
                self.log.warning(f'Found more than one file at "{path}". Grabbing the first one: {file}')
            return file
        else:
            self.log.info(f'No files found for "{path}"')

    def retrieve_files(self) -> List:
        """Returns the actual location of all files in the stored .location"""
        path = self.store()
        files = sorted(glob(path))
        if not files:
            self.log.info(f'No files found for "{path}"')
        return files

    def retrieve_names(self) -> List[str]:
        """Returns the names of all objects in the stored .location"""
        path = self.store()
        names = list(map(os.path.basename, [os.path.splitext(file)[0] for file in sorted(glob(path))]))
        if not names:
            self.log.warning(f'No files found for "{path}"')
        return names

    def store_data(self, data: Any, name: str, **kwargs):  # Todo resolve with def store() above
        """Return the path of the storage location given an entity name"""
        setattr(self, name, data)
        self._save_data(name, **kwargs)

    def retrieve_data(self, name: str = None) -> object | None:
        """Return the data requested by name. Otherwise, load into the Database from a specified location

        Args:
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        data = getattr(self, name, None)
        if data:
            self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
        else:
            setattr(self, name, self._load_data(name, log=None))  # attempt to store the new data as an attribute
            data = getattr(self, name)
            if data:
                self.log.debug(f'Database file {name}{self.extension} was loaded fresh')

        return data

    def _save_data(self, name: str, **kwargs) -> str | bytes | None:
        """Return the data located in a particular entry specified by name

        Returns:
            The name of the saved data if there was one or the return from the Database insertion
        """
        if self.sql:
            # dummy = True
            return
        else:
            return self.save_file(self.store, self.retrieve_data(name), **kwargs)

    def _load_data(self, name: str, **kwargs) -> Any | None:
        """Return the data located in a particular entry specified by name

        Returns:
            (Union[None, Any])
        """
        if self.sql:
            dummy = True
        else:
            file = self.retrieve_file(name)
            if file:
                return self.load_file(file, **kwargs)
        return

    def get_all_data(self, **kwargs):
        """Return all data located in the particular DataStore storage location"""
        if self.sql:
            dummy = True
        else:
            for file in sorted(glob(os.path.join(self.location, f'*{self.extension}'))):
                # self.log.debug('Fetching %s' % file)
                setattr(self, os.path.splitext(os.path.basename(file))[0], self.load_file(file))

    @staticmethod
    def read_file(file, **kwargs):
        with open(file, 'r') as f:
            lines = f.readlines()

        return lines


class ClusterInfoFile:
    def __init__(self, infofile_path):
        # self.infofile_path = infofile_path
        self.name = os.path.splitext(os.path.basename(infofile_path))[0]
        self.size = None
        self.rmsd = None
        self.representative_filename = None
        self.central_residue_pair_freqs = []
        # self.central_residue_pair_counts = []
        # self.load_info()

    # def load_info(self):
        with open(infofile_path, 'r') as f:
            info_lines = f.readlines()

        is_res_freq_line = False
        for line in info_lines:
            # if line.startswith("CLUSTER NAME:"):
            #     self.name = line.split()[2]
            if line.startswith("CLUSTER SIZE:"):
                self.size = int(line.split()[2])
            elif line.startswith("CLUSTER RMSD:"):
                self.rmsd = float(line.split()[2])
            elif line.startswith("CLUSTER REPRESENTATIVE NAME:"):
                self.representative_filename = line.split()[3]
            elif line.startswith("CENTRAL RESIDUE PAIR COUNT:"):
                is_res_freq_line = False
            elif is_res_freq_line:
                res_pair_type = (line.split()[0][0], line.split()[0][1])
                res_pair_freq = float(line.split()[1])
                self.central_residue_pair_freqs.append((res_pair_type, res_pair_freq))
            elif line.startswith("CENTRAL RESIDUE PAIR FREQUENCY:"):
                is_res_freq_line = True

    def get_name(self):
        return self.name

    def get_size(self):
        return self.size

    def get_rmsd(self):
        return self.rmsd

    def get_representative_filename(self):
        return self.representative_filename

    def get_central_residue_pair_freqs(self):
        return self.central_residue_pair_freqs


class FragmentDB:
    cluster_representatives_path: str
    cluster_info_path: str
    fragment_length: int
    indexed_ghosts: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | dict
    # dict[int, tuple[3x3, 1x3, tuple[int, int, int], float]]
    info: dict[int, dict[int, dict[int, ClusterInfoFile]]] | None
    # monofrag_representatives_path: str
    paired_frags: dict[int, dict[int, dict[int, tuple[PDB, str]]]] | None
    reps: dict[int, np.ndarray]

    def __init__(self, fragment_length: int = 5):
        self.cluster_representatives_path = intfrag_cluster_rep_dirpath
        self.cluster_info_path = intfrag_cluster_info_dirpath
        self.fragment_length = fragment_length
        self.indexed_ghosts = {}
        self.info = None
        # self.monofrag_representatives_path = monofrag_cluster_rep_dirpath
        self.paired_frags = None
        # self.reps = None

    # def get_monofrag_cluster_rep_dict(self):
        self.reps = {int(os.path.splitext(file)[0]):
                     PDB.from_file(os.path.join(root, file), entities=False, log=None).get_ca_coords()
                     for root, dirs, files in os.walk(monofrag_cluster_rep_dirpath) for file in files}

    def get_intfrag_cluster_rep_dict(self):
        ijk_cluster_representatives = {}
        for root, dirs, files in os.walk(self.cluster_representatives_path):
            if not dirs:
                i_cluster_type, j_cluster_type, k_cluster_type = map(int, root.split(os.sep)[-1].split('_'))

                if i_cluster_type not in ijk_cluster_representatives:
                    ijk_cluster_representatives[i_cluster_type] = {}
                if j_cluster_type not in ijk_cluster_representatives[i_cluster_type]:
                    ijk_cluster_representatives[i_cluster_type][j_cluster_type] = {}

                for file in files:
                    ijk_frag_cluster_rep_pdb = PDB.from_file(os.path.join(root, file), entities=False, log=None)
                    # mapped_chain_idx = file.find('mappedchain')
                    # ijk_cluster_rep_mapped_chain = file[mapped_chain_idx + 12:mapped_chain_idx + 13]
                    # must look up the partner coords later by using chain_id stored in file
                    partner_chain_idx = file.find('partnerchain')
                    ijk_cluster_rep_partner_chain = file[partner_chain_idx + 13:partner_chain_idx + 14]
                    ijk_cluster_representatives[i_cluster_type][j_cluster_type][k_cluster_type] = \
                        (ijk_frag_cluster_rep_pdb, ijk_cluster_rep_partner_chain)  # ijk_cluster_rep_mapped_chain,

        self.paired_frags = ijk_cluster_representatives
        if self.info:
            self.index_ghosts()

    def get_intfrag_cluster_info_dict(self):
        intfrag_cluster_info_dict = {}
        for root, dirs, files in os.walk(self.cluster_info_path):
            if not dirs:
                i_cluster_type, j_cluster_type, k_cluster_type = map(int, root.split(os.sep)[-1].split('_'))

                if i_cluster_type not in intfrag_cluster_info_dict:
                    intfrag_cluster_info_dict[i_cluster_type] = {}
                if j_cluster_type not in intfrag_cluster_info_dict[i_cluster_type]:
                    intfrag_cluster_info_dict[i_cluster_type][j_cluster_type] = {}

                for file in files:
                    intfrag_cluster_info_dict[i_cluster_type][j_cluster_type][k_cluster_type] = \
                        ClusterInfoFile(os.path.join(root, file))

        self.info = intfrag_cluster_info_dict
        if self.paired_frags:
            self.index_ghosts()

    def index_ghosts(self):
        """From the fragment database, precompute all required data into arrays to populate Ghost Fragments"""
        for i_type in self.paired_frags:
            # must look up the partner coords by using stored chain_id
            stacked_bb_coords = np.array([frag_pdb.chain(frag_paired_chain).get_backbone_coords()
                                          for j_dict in self.paired_frags[i_type].values()
                                          for frag_pdb, frag_paired_chain in j_dict.values()])
            # guide coords are stored with chain_id "9"
            stacked_guide_coords = np.array([frag_pdb.chain('9').coords for j_dict in self.paired_frags[i_type].values()
                                             for frag_pdb, _, in j_dict.values()])
            ijk_types = \
                np.array([(i_type, j_type, k_type) for j_type, j_dict in self.paired_frags[i_type].items()
                          for k_type in j_dict])
            # rmsd_array = np.array([self.info.cluster(type_set).rmsd for type_set in ijk_types])  # Todo
            rmsd_array = np.array([dictionary_lookup(self.info, type_set).rmsd for type_set in ijk_types])
            self.indexed_ghosts[i_type] = stacked_bb_coords, stacked_guide_coords, ijk_types, rmsd_array


class FragmentDatabase(FragmentDB):
    def __init__(self, source: str = biological_interfaces, fragment_length: int = 5, init_db: bool = True,
                 sql: bool = False, **kwargs):
        super().__init__()  # FragmentDB
        # self.monofrag_representatives_path = monofrag_representatives_path
        # self.cluster_representatives_path
        # self.cluster_info_path = cluster_info_path
        # self.reps = None
        # self.paired_frags = None
        # self.info = None
        self.source: str = source
        # Todo load all statistics files into the pickle!
        # self.location = frag_directory.get(self.source, None)
        self.statistics: Dict = {}
        # {cluster_id: [[mapped, paired, {max_weight_counts}, ...], ..., frequencies: {'A': 0.11, ...}}
        #  ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]
        self.fragment_range: Tuple[int, int]
        self.cluster_info: Dict = {}
        # self.fragdb = None  # Todo

        if sql:
            self.start_mysql_connection()
            self.db = True
        else:  # self.source == 'directory':
            # Todo initialize as local directory
            self.db = False
            if init_db:
                logger.info(f'Initializing {source} FragmentDatabase from disk. This may take awhile...')
                # self.get_monofrag_cluster_rep_dict()
                self.get_intfrag_cluster_rep_dict()
                self.get_intfrag_cluster_info_dict()
                # self.get_cluster_info()

        self.get_db_statistics()
        self.fragment_range = parameterize_frag_length(fragment_length)

    @property
    def location(self) -> Optional[Union[str, bytes]]:
        """Provide the location where fragments are stored"""
        return frag_directory.get(self.source, None)

    def get_db_statistics(self) -> Dict:
        """Retrieve summary statistics for a specific fragment database located on directory

        Returns:
            {cluster_id1: [[mapped_index_average, paired_index_average, {max_weight_counts_mapped}, {paired}],
                           total_fragment_observations], cluster_id2: ...,
             frequencies: {'A': 0.11, ...}}
                ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749], ...}
        """
        if self.db:
            logger.warning('No SQL DB connected yet!')  # Todo
            raise DesignError('Can\'t connect to MySQL database yet')
        else:
            stats_file = sorted(glob(os.path.join(self.location, 'statistics.pkl')))
            if len(stats_file) == 1:
                self.statistics = unpickle(stats_file[0])
            else:
                raise DesignError('There were too many statistics.pkl files found from the fragment database source!')
            # for file in os.listdir(self.location):
            #     if 'statistics.pkl' in file:
            #         self.statistics = unpickle(os.path.join(self.location, file))
            #         return

    def get_db_aa_frequencies(self) -> Dict[protein_letters, float]:
        """Retrieve database specific amino acid representation frequencies

        Returns:
            {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
        """
        return self.statistics.get('frequencies', {})

    def retrieve_cluster_info(self, cluster: str = None, source: str = None, index: str = None) -> \
            Dict[str, Union[int, float, str, Dict[int, Dict[Union[protein_letters, str],
                                                            Union[float, Tuple[int, float]]]]]]:
        # Todo rework this and below func for Database
        """Return information from the fragment information database by cluster_id, information source, and source index

        Args:
            cluster: A cluster_id to get information about
            source: The source of information to gather from: ['size', 'rmsd', 'rep', 'mapped', 'paired']
            index: The index to gather information from. Must be from 'mapped' or 'paired'
        Returns:
            {'size': ..., 'rmsd': ..., 'rep': ..., 'mapped': indexed_frequencies, 'paired': indexed_frequencies}
            Where indexed_frequencies has format {-2: {'A': 0.1, 'C': 0., ..., 'info': (12, 0.41)}, -1: {}, ..., 2: {}}
        """
        if cluster:
            if cluster not in self.cluster_info:
                self.get_cluster_info(ids=[cluster])
            if source:
                if index is not None and source in ['mapped', 'paired']:  # must check for not None. The index can be 0
                    return self.cluster_info[cluster][source][index]
                else:
                    return self.cluster_info[cluster][source]
            else:
                return self.cluster_info[cluster]
        else:
            return self.cluster_info

    def get_cluster_info(self, ids: List[str] = None):
        """Load cluster information from the fragment database source into attribute cluster_info
        # todo change ids to a tuple
        Args:
            ids: ['1_2_123', ...]
        Sets:
            self.cluster_info (dict): {'1_2_123': {'size': , 'rmsd': , 'rep': , 'mapped': , 'paired': }, ...}
        """
        if self.db:
            logger.warning('No SQL DB connected yet!')
            raise DesignError('Can\'t connect to MySQL database yet')
        else:
            if not ids:
                directories = get_base_root_paths_recursively(self.location)
            else:
                directories = []
                for _id in ids:
                    c_id = _id.split('_')
                    _dir = os.path.join(self.location, c_id[0], '%s_%s' % (c_id[0], c_id[1]),
                                        '%s_%s_%s' % (c_id[0], c_id[1], c_id[2]))
                    directories.append(_dir)

            for cluster_directory in directories:
                cluster_id = os.path.basename(cluster_directory)
                self.cluster_info[cluster_id] = unpickle(os.path.join(cluster_directory, '%s.pkl' % cluster_id))

    @staticmethod
    def get_cluster_id(cluster_id, index=3):
        """Returns the cluster identification string according the specified index

        Args:
            cluster_id (str): The id of the fragment cluster. Ex: 1_2_123
        Keyword Args:
            index_number=3 (int): The index on which to return. Ex: index_number=2 gives 1_2
        Returns:
            (str): The cluster_id modified by the requested index_number
        """
        while len(cluster_id) < 3:
            cluster_id += '0'

        if len(cluster_id.split('_')) != 3:  # in case of 12123?
            id_l = [cluster_id[:1], cluster_id[1:2], cluster_id[2:]]
        else:
            id_l = cluster_id.split('_')

        info = [id_l[i] for i in range(index)]

        while len(info) < 3:  # ensure the returned string has at least 3 indices
            info.append('0')

        return '_'.join(info)

    # def parameterize_frag_length(self, length):
    #     """Generate fragment length range parameters for use in fragment functions"""
    #     _range = math.floor(length / 2)  # get the number of residues extending to each side
    #     if length % 2 == 1:  # fragment length is odd
    #         self.fragment_range = (0 - _range, 0 + _range + index_offset)
    #         # return 0 - _range, 0 + _range + index_offset
    #     else:  # length is even
    #         logger.critical(f'{length} is an even integer which is not symmetric about a single residue. '
    #                         'Ensure this is what you want')
    #         self.fragment_range = (0 - _range, 0 + _range)

    def start_mysql_connection(self):
        self.fragdb = Mysql(host='cassini-mysql', database='kmeader', user='kmeader', password='km3@d3r')


class FragmentDatabaseFactory:
    """Return an FragmentDatabase instance by calling the Factory instance with the FragmentDatabase source name

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the named FragmentDatabase
    """

    def __init__(self, **kwargs):
        self._databases = {}

    def __call__(self, source: str = biological_interfaces, **kwargs) -> FragmentDatabase:
        """Return the specified FragmentDatabase object singleton

        Args:
            source: The FragmentDatabase source name
        Returns:
            The instance of the specified FragmentDatabase
        """
        fragment_db = self._databases.get(source)
        if fragment_db:
            return fragment_db
        elif source == biological_interfaces:
            logger.info(f'Initializing {source} {FragmentDatabase.__name__}')
            self._databases[source] = unpickle(biological_fragment_db_pickle)
        else:
            logger.info(f'Initializing {source} {FragmentDatabase.__name__}')
            self._databases[source] = FragmentDatabase(source=source, **kwargs)

        return self._databases[source]

    def get(self, **kwargs) -> FragmentDatabase:
        """Return the specified FragmentDatabase object singleton

        Keyword Args:
            source: The FragmentDatabase source name
        Returns:
            The instance of the specified FragmentDatabase
        """
        return self.__call__(**kwargs)


fragment_factory = FragmentDatabaseFactory()
# fragment_factory.set(biological_interfaces, unpickle(biological_fragment_db_pickle))


class JobResources:
    """The intention of JobResources is to serve as a singular source of design info which is common accross all
    designs. This includes common paths, databases, and design flags which should only be set once in program operation,
    then shared across all member designs"""
    def __init__(self, program_root, **kwargs):
        """For common resources for all SymDesign outputs, ensure paths to these resources are available attributes"""
        if os.path.exists(program_root):
            self.program_root = program_root
        else:
            raise FileNotFoundError(f'Path does not exist!\n\t{program_root}')

        # program_root subdirectories
        self.protein_data = os.path.join(self.program_root, data.title())
        self.projects = os.path.join(self.program_root, projects)
        self.job_paths = os.path.join(self.program_root, 'JobPaths')
        self.sbatch_scripts = os.path.join(self.program_root, 'Scripts')
        # TODO ScoreDatabase integration
        self.all_scores = os.path.join(self.program_root, all_scores)

        # data subdirectories
        self.clustered_poses = os.path.join(self.protein_data, 'ClusteredPoses')
        self.pdbs = os.path.join(self.protein_data, 'PDBs')  # Used to store downloaded PDB's
        self.sequence_info = os.path.join(self.protein_data, sequence_info)
        # pdbs subdirectories
        self.orient_dir = os.path.join(self.pdbs, 'oriented')
        self.orient_asu_dir = os.path.join(self.pdbs, 'oriented_asu')
        self.refine_dir = os.path.join(self.pdbs, 'refined')
        self.full_model_dir = os.path.join(self.pdbs, 'full_models')
        self.stride_dir = os.path.join(self.pdbs, 'stride')
        # sequence_info subdirectories
        self.sequences = os.path.join(self.sequence_info, 'sequences')
        self.profiles = os.path.join(self.sequence_info, 'profiles')
        # try:
        # if not self.projects:  # used for subclasses
        # if not getattr(self, 'projects', None):  # used for subclasses
        #     self.projects = os.path.join(self.program_root, projects)
        # except AttributeError:
        #     self.projects = os.path.join(self.program_root, projects)
        # self.design_db = None
        # self.score_db = None
        # self.make_path(self.protein_data)
        # self.make_path(self.projects)
        # self.make_path(self.job_paths)
        # self.make_path(self.sbatch_scripts)
        # sequence database specific
        # self.make_path(self.sequence_info)
        # self.make_path(self.sequences)
        # self.make_path(self.profiles)
        # structure database specific
        # self.make_path(self.pdbs)
        # self.make_path(self.orient_dir)
        # self.make_path(self.orient_asu_dir)
        # self.make_path(self.refine_dir)
        # self.make_path(self.full_model_dir)
        # self.make_path(self.stride_dir)
        self.reduce_memory = False
        self.resources = Database(self.orient_dir, self.orient_asu_dir, self.refine_dir, self.full_model_dir,
                                  self.stride_dir, self.sequences, self.profiles, sql=None)  # , log=logger)
        # self.symmetry_factory = symmetry_factory
        self.fragment_db: FragmentDatabase | None = None
        self.euler_lookup: EulerLookup | None = None

        # Program flags
        self.consensus: bool = kwargs.get(consensus, False)  # Whether to run consensus
        self.construct_pose: bool = kwargs.get('construct_pose', False)  # whether to construct Nanohedra pose
        self.design_selector: dict[str, dict[str, dict[str, set[int] | set[str]]]] | dict = kwargs.get('design_selector'
                                                                                                       , {})
        self.debug: bool = kwargs.get('debug', False)
        self.force_flags: bool = kwargs.get(force_flags, False)
        self.fuse_chains: list[tuple[str]] = [tuple(pair.split(':')) for pair in kwargs.get('fuse_chains', [])]
        self.ignore_clashes: bool = kwargs.get(ignore_clashes, False)
        if self.ignore_clashes:
            self.ignore_pose_clashes = self.ignore_symmetric_clashes = True
        else:
            self.ignore_pose_clashes: bool = kwargs.get(ignore_pose_clashes, False)
            self.ignore_symmetric_clashes: bool = kwargs.get(ignore_symmetric_clashes, False)
        self.increment_chains: bool = kwargs.get('increment_chains', False)
        self.mpi: int = kwargs.get('mpi', 0)
        self.no_evolution_constraint: bool = kwargs.get(no_evolution_constraint, False)
        self.no_hbnet: bool = kwargs.get(no_hbnet, False)
        self.no_term_constraint: bool = kwargs.get(no_term_constraint, False)
        self.number_of_trajectories: int = kwargs.get(number_of_trajectories, nstruct)
        self.overwrite: bool = kwargs.get('overwrite', False)
        self.output_directory: bool = kwargs.get('output_directory', False)
        self.output_assembly: bool = kwargs.get('output_assembly', False)
        self.run_in_shell: bool = kwargs.get('run_in_shell', False)
        # self.pre_refine: bool = kwargs.get('pre_refine', True)
        # self.pre_loop_model: bool = kwargs.get('pre_loop_model', True)
        self.generate_fragments: bool = kwargs.get(generate_fragments, True)
        self.scout: bool = kwargs.get(scout, False)
        self.specific_protocol: str = kwargs.get('specific_protocol', False)
        self.structure_background: bool = kwargs.get(structure_background, False)
        self.sym_entry: SymEntry | None = kwargs.get(sym_entry, None)
        self.write_frags: bool = kwargs.get(output_fragments, True)  # todo, if generate_fragments module, ensure True
        self.write_oligomers: bool = kwargs.get(output_oligomers, False)
        self.skip_logging: bool = kwargs.get('skip_logging', False)
        self.nanohedra_output: bool = kwargs.get('nanohedra_output', False)
        self.nanohedra_root: str | None = None
        # Development Flags
        self.command_only: bool = kwargs.get('command_only', False)  # Whether to reissue commands, only if run_in_shell=False
        self.development: bool = kwargs.get('development', False)

        if self.nanohedra_output and not self.construct_pose:  # no construction specific flags
            self.write_frags = False
            self.write_oligomers = False

        if self.no_term_constraint:
            self.generate_fragments = False

    @staticmethod
    def make_path(path: Union[str, bytes], condition: bool = True):
        """Make all required directories in specified path if it doesn't exist, and optional condition is True

        Args:
            path: The path to create
            condition: A condition to check before the path production is executed
        """
        if condition:
            os.makedirs(path, exist_ok=True)
