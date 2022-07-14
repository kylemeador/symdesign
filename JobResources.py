from __future__ import annotations

import json
import os
import subprocess
from copy import copy
from glob import glob
from logging import Logger
from pathlib import Path
from typing import Iterable, Any, Annotated

from CommandDistributer import rosetta_flags, script_cmd, distribute, relax_flags, rosetta_variables
from Query.UniProt import query_uniprot
from utils.PDBUtils import orient_structure_file
from PathUtils import sym_entry, program_name, orient_log_file, rosetta_scripts, models_to_multimodel_exe, refine, \
    all_scores, projects, sequence_info, data, output_oligomers, output_fragments, \
    structure_background, scout, generate_fragments, number_of_trajectories, nstruct, no_hbnet, ignore_symmetric_clashes, ignore_pose_clashes, ignore_clashes, force_flags, no_evolution_constraint, \
    no_term_constraint, consensus, qs_bio, pdb_db
import Pose
from Query.PDB import query_entry_id, query_entity_id, query_assembly_id, \
    parse_entry_json, parse_entities_json, parse_assembly_json
from Query.utils import boolean_choice
from SequenceProfile import parse_hhblits_pssm, MultipleSequenceAlignment, read_fasta_file, write_sequence_to_fasta
from Structure import parse_stride, Structure
from SymDesignUtils import DesignError, unpickle, start_log, write_commands, starttime, make_path, write_shell_script, to_iterable
from classes.EulerLookup import EulerLookup
from classes.SymEntry import sdf_lookup, SymEntry, parse_symmetry_to_sym_entry
from fragment import FragmentDatabase
# import dependencies.bmdca as bmdca


# Globals
# for checking out the options to read and write Rosetta runs to a relational DB such as MySQL
# https://new.rosettacommons.org/docs/latest/rosetta_basics/options/Database-options
logger = start_log(name=__name__)
index_offset = 1
qsbio_confirmed = unpickle(qs_bio)


def _fetch_pdb_from_api(pdb_codes: str | list, assembly: int = 1, asu: bool = False, out_dir: str | bytes = os.getcwd(),
                        **kwargs) -> list[str | bytes]:  # Todo mmcif
    """Download PDB files from pdb_codes provided in a file, a supplied list, or a single entry
    Can download a specific biological assembly if asu=False.
    Ex: _fetch_pdb_from_api('1bkh', assembly=2) fetches 1bkh biological assembly 2 "1bkh.pdb2"

    Args:
        pdb_codes: PDB IDs of interest.
        assembly: The integer of the assembly to fetch
        asu: Whether to download the asymmetric unit file
        out_dir: The location to save downloaded files to
    Returns:
        Filenames of the retrieved files
    """
    file_names = []
    for pdb_code in to_iterable(pdb_codes):
        clean_pdb = pdb_code[:4].lower()
        if asu:
            clean_pdb = f'{clean_pdb}.pdb'
        else:
            # assembly = pdb[-3:]
            # try:
            #     assembly = assembly.split('_')[1]
            # except IndexError:
            #     assembly = '1'
            clean_pdb = f'{clean_pdb}.pdb{assembly}'

        # clean_pdb = '%s.pdb%d' % (clean_pdb, assembly)
        file_name = os.path.join(out_dir, clean_pdb)
        current_file = sorted(glob(file_name))
        # print('Found the files %s' % current_file)
        # current_files = os.listdir(location)
        # if clean_pdb not in current_files:
        if not current_file:  # glob will return an empty list if the file is missing and therefore should be downloaded
            # Always returns files in lowercase
            # status = os.system(f'wget -q -O {file_name} https://files.rcsb.org/download/{clean_pdb}')
            cmd = ['wget', '-q', '-O', file_name, f'https://files.rcsb.org/download/{clean_pdb}']
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p.communicate()
            if p.returncode != 0:
                logger.error(f'PDB download failed for: {clean_pdb}. If you believe this PDB ID is correct, there may '
                             f'only be a .cif file available for this entry, which currently can\'t be parsed')
                # Todo parse .cif file.
                #  Super easy as the names of the columns are given in a loop and the ATOM records still start with ATOM
                #  The additional benefits is that the records contain entity IDS as well as the residue index and the
                #  author residue number. I think I will prefer this format from now on once parsing is possible.

            # file_request = requests.get('https://files.rcsb.org/download/%s' % clean_pdb)
            # if file_request.status_code == 200:
            #     with open(file_name, 'wb') as f:
            #         f.write(file_request.content)
            # else:
            #     logger.error('PDB download failed for: %s' % pdb)
        file_names.append(file_name)

    return file_names


def fetch_pdb_file(pdb_code: str, asu: bool = True, location: str | bytes = pdb_db, **kwargs) -> str | bytes | None:
    #                assembly: int = 1, out_dir: str | bytes = os.getcwd()
    """Fetch PDB object from PDBdb or download from PDB server

    Args:
        pdb_code: The PDB ID/code. If the biological assembly is desired, supply 1ABC_1 where '_1' is assembly ID
        asu: Whether to fetch the ASU
        location: Location of a local PDB mirror if one is linked on disk
    Keyword Args:
        assembly=None (int): Location of a local PDB mirror if one is linked on disk
        out_dir=os.getcwd() (str | bytes): The location to save retrieved files if fetched from PDB
    Returns:
        The path to the file if located successfully
    """
    # if location == pdb_db and asu:
    if os.path.exists(location) and asu:
        file_path = os.path.join(location, f'pdb{pdb_code.lower()}.ent')
        get_pdb = (lambda *args, **kwargs: sorted(glob(file_path)))
        #                                            pdb_code, location=None, asu=None, assembly=None, out_dir=None
        logger.debug(f'Searching for PDB file at "{file_path}"')
        # Cassini format is above, KM local pdb and the escher PDB mirror is below
        # get_pdb = (lambda pdb_code, asu=None, assembly=None, out_dir=None:
        #            glob(os.path.join(pdb_db, subdirectory(pdb_code), '%s.pdb' % pdb_code)))
        # print(os.path.join(pdb_db, subdirectory(pdb_code), '%s.pdb' % pdb_code))
    else:
        get_pdb = _fetch_pdb_from_api

    # return a list where the matching file is the first (should only be one anyway)
    pdb_file = get_pdb(pdb_code, asu=asu, location=location, **kwargs)
    if not pdb_file:
        logger.warning(f'No matching file found for PDB: {pdb_code}')
    else:  # we should only find one file, therefore, return the first
        return pdb_file[0]


def query_qs_bio(pdb_entry_id: str) -> int:
    """Retrieve the first matching High/Very High confidence QSBio assembly from a PDB ID

    Args:
        pdb_entry_id: The 4 letter PDB code to query
    Returns:
        The integer of the corresponding PDB Assembly ID according to the QSBio assembly
    """
    biological_assemblies = qsbio_confirmed.get(pdb_entry_id)
    if biological_assemblies:  # first   v   assembly in matching oligomers
        assembly = biological_assemblies[0]
    else:
        assembly = 1
        logger.warning(f'No confirmed biological assembly for entry {pdb_entry_id},'
                       f' using PDB default assembly {assembly}')
    return assembly


class Database:  # Todo ensure that the single object is completely loaded before multiprocessing... Queues and whatnot
    def __init__(self, oriented: str | bytes | Path = None, oriented_asu: str | bytes | Path = None,
                 refined: str | bytes | Path = None, full_models: str | bytes | Path = None,
                 stride: str | bytes | Path = None, sequences: str | bytes | Path = None,
                 hhblits_profiles: str | bytes | Path = None, pdb_api: str | bytes | Path = None,
                 uniprot_api: str | bytes | Path = None, sql=None, log: Logger = logger):  # sql: sqlite = None,
        #                  pdb_entity_api: str | bytes | Path = None, pdb_assembly_api: str | bytes | Path = None,
        if sql:
            raise NotImplementedError('SQL set up has not been completed!')
            self.sql = sql
        else:
            self.sql = sql

        self.log = log
        self.oriented = DataStore(location=oriented, extension='.pdb*', sql=sql, log=log)
        self.oriented_asu = DataStore(location=oriented_asu, extension='.pdb', sql=sql, log=log)
        self.refined = DataStore(location=refined, extension='.pdb', sql=sql, log=log)
        self.full_models = DataStore(location=full_models, extension='_ensemble.pdb', sql=sql, log=log)
        self.stride = DataStore(location=stride, extension='.stride', sql=sql, log=log)
        self.sequences = DataStore(location=sequences, extension='.fasta', sql=sql, log=log)
        self.alignments = DataStore(location=hhblits_profiles, extension='.sto', sql=sql, log=log)
        self.hhblits_profiles = DataStore(location=hhblits_profiles, extension='.hmm', sql=sql, log=log)
        self.pdb_api = PDBDataStore(location=pdb_api, extension='.json', sql=sql, log=log)
        # self.pdb_entity_api = DataStore(location=pdb_entity_api, extension='.json', sql=sql, log=log)
        # self.pdb_assembly_api = DataStore(location=pdb_assembly_api, extension='.json', sql=sql, log=log)
        self.uniprot_api = UniProtDataStore(location=uniprot_api, extension='.json', sql=sql, log=log)
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

    def orient_structures(self, structure_identifiers: Iterable[str], symmetry: str = 'C1', by_file: bool = False) -> \
            list['Pose.Model'] | list:
        """Given entity_ids and their corresponding symmetry, retrieve .pdb files, orient and save Database files then
        return the ASU for each

        Args:
            structure_identifiers: The names of all entity_ids requiring orientation
            symmetry: The symmetry to treat each passed Entity. Default assumes no symmetry
            by_file: Whether to parse the structure_identifiers as file paths. Default treats as PDB EntryID or EntityID
        Returns:
            The resulting asymmetric oriented Poses
        """
        if not structure_identifiers:
            return []
        from Pose import Pose, Model
        # using Pose enables simple ASU writing
        # Pose isn't the best for orient since the structure isn't oriented the symmetry wouldn't work
        # It could still happen, but we certainly shouldn't pass sym_entry to it
        self.oriented.make_path()
        orient_dir = self.oriented.location
        # os.makedirs(orient_dir, exist_ok=True)
        models_dir = os.path.dirname(orient_dir)  # this is the case because it was specified this way, but not required
        self.oriented_asu.make_path()
        # orient_asu_dir = self.oriented_asu.location
        # os.makedirs(orient_asu_dir, exist_ok=True)
        self.stride.make_path()
        # stride_dir = self.stride.location
        # os.makedirs(stride_dir, exist_ok=True)
        # orient_log = start_log(name='orient', handler=1)
        orient_log = \
            start_log(name='orient', handler=2, location=os.path.join(orient_dir, orient_log_file), propagate=True)
        if by_file:
            logger.info(f'The requested files are being checked for proper orientation with symmetry {symmetry}: '
                        f'{", ".join(structure_identifiers)}')
            oriented_filepaths = [orient_structure_file(file, log=orient_log, symmetry=symmetry, out_dir=orient_dir)
                                  for file in structure_identifiers]
            # pull out the structure names and use below to retrieve the oriented file
            structure_identifiers = list(map(os.path.basename,
                                             [os.path.splitext(file)[0] for file in filter(None, oriented_filepaths)]))
        else:
            logger.info(f'The requested IDs are being checked for proper orientation with symmetry {symmetry}: '
                        f'{", ".join(structure_identifiers)}')

        # Next work through the process of orienting the selected files
        orient_names = self.oriented.retrieve_names()
        orient_asu_names = self.oriented_asu.retrieve_names()
        sym_entry = parse_symmetry_to_sym_entry(symmetry=symmetry)
        all_structures = []
        non_viable_structures = []
        for structure_identifier in structure_identifiers:
            # first, check if the structure_identifier has been oriented. this happens when files are passed
            if structure_identifier not in orient_names:  # they are missing, retrieve the proper files using PDB ID's
                entry = structure_identifier.split('_')
                # in case entry_entity is coming from a new SymDesign Directory the entity name is probably 1ABC_1
                if len(entry) == 2:
                    entry, entity = entry
                    entry_entity = structure_identifier
                    logger.debug(f'Fetching entry {entry}, entity {entity} from PDB')
                else:
                    entry = entry_entity = structure_identifier
                    entity = None
                    logger.debug(f'Fetching entry {entry} from PDB')

                if symmetry == 'C1':
                    assembly = None  # 1 is the default
                    asu = True
                else:
                    asu = False
                    assembly = query_qs_bio(entry)
                # get the specified file_path for the assembly state of interest
                file_path = fetch_pdb_file(entry, assembly=assembly, asu=asu, out_dir=models_dir)

                if not file_path:
                    logger.warning(f'Couldn\'t locate the file "{file_path}", there may have been an issue '
                                   'downloading it from the PDB. Attempting to copy from job data source...')
                    # Todo
                    raise NotImplementedError('This functionality hasn\'t been written yet. Use the canonical_pdb1/2 '
                                              'attribute of PoseDirectory to pull the pdb file source.')
                # remove any PDB Database mirror specific naming from fetch_pdb_file such as pdb1ABC.ent
                file_name = os.path.splitext(os.path.basename(file_path))[0].replace('pdb', '')
                model = Model.from_pdb(file_path, name=file_name)  # , sym_entry=sym_entry
                if entity:  # replace Structure from fetched file with the Entity Structure
                    # entry_entity will be formatted the exact same as the desired EntityID if it was provided correctly
                    entity = model.entity(entry_entity)
                    if not entity:  # we couldn't find the specified EntityID
                        logger.warning(f'For {entry_entity}, couldn\'t locate the specified Entity "{entity}". The'
                                       f' available Entities are {", ".join(entity.name for entity in pose.entities)}')
                        continue
                    entity_out_path = os.path.join(models_dir, f'{entry_entity}.pdb')
                    if symmetry == 'C1':  # write out only entity
                        entity_file_path = entity.write(out_path=entity_out_path)
                    else:  # write out the entity as parsed. since this is assembly we should get the correct state
                        entity_file_path = entity.write_oligomer(out_path=entity_out_path)
                    # pose = entity.oligomer <- not quite as desired
                    # Todo make Entity capable of orient() then don't need this mechanism, just set pose = entity
                    model = model.from_chains(entity.chains, entity_names=[entry_entity])
                    # pose.entities = [entity]
                # else:  # we will orient the whole set of chains based on orient() multicomponent solution
                #     # entry_entity = pose.name
                #     pass
                # the entry_entity name is correct for the entry or entity if we have got to this point
                # entry_entity_base = f'{entry_entity}.pdb'

                # write out file for the orient database
                if symmetry == 'C1':  # translate the Structure to the origin for the database
                    # entity = pose.entities[0]  # issue if we passed a hetero dimer and treated as a C1
                    model.translate(-model.center_of_mass)
                    # entity.name = entry_entity
                    # orient_file = entity.write(out_path=os.path.join(orient_dir, entry_entity_base))
                    # orient_file = model.write(out_path=self.oriented.store(name=entry_entity))
                    # model.symmetry = symmetry
                    # model.file_path = model.write(out_path=self.oriented_asu.store(name=entry_entity))
                    # for entity in model.entities:
                    #     entity.stride(to_file=self.stride.store(name=entity.name))
                    # all_structures.append(model)  # .entities[0]
                else:
                    try:
                        model.orient(symmetry=symmetry, log=orient_log)
                        # orient_file = model.write(assembly=True, out_path=self.oriented.store(name=entry_entity))
                    except (ValueError, RuntimeError) as err:
                        orient_log.error(str(err))
                        non_viable_structures.append(entry_entity)
                        continue
                    # all_structures.append(return_orient_asu(orient_file, entry_entity, symmetry))
                    # entity = pose.entities[0]
                    # entity.name = pose.name  # use oriented_pose.name (pdbcode_assembly), not API name

                # Extract the asu from the oriented file. Used for symmetric refinement, pose generation
                orient_file = model.write(out_path=self.oriented.store(name=entry_entity))
                orient_log.info(f'Oriented: {orient_file}')
                model.symmetry = symmetry
                # Save the ASU file as the Structure.file_path to be used later by preprocess_structures_for_design
                # model.file_path = model.write(out_path=self.oriented_asu.store(name=entry_entity))
                model.file_path = self.oriented_asu.store(name=entry_entity)
                with open(model.file_path, 'w') as f:
                    for entity in model.entities:
                        # write each Entity to asu
                        entity.write(file_handle=f)
                        # save Stride results
                        entity.stride(to_file=self.stride.store(name=entity.name))
                all_structures.append(model)
            # for those below, use structure_identifier as entry_entity isn't parsed
            elif structure_identifier not in orient_asu_names:  # orient file exists, load asu, save and create stride
                orient_file = self.oriented.retrieve_file(name=structure_identifier)
                pose = Pose.from_file(orient_file, sym_entry=sym_entry)  # , log=None, entity_names=[entry_entity])
                # entity = pose.entities[0]
                # entity.name = pose.name  # use pose.name, not API name
                # Pose already sets a symmetry, so we don't need to set one
                # pose.symmetry = symmetry
                pose.file_path = pose.write(out_path=self.oriented_asu.store(name=structure_identifier))
                # save Stride results
                for entity in pose.entities:
                    entity.stride(to_file=self.stride.store(name=entity.name))
                all_structures.append(pose)  # entry_entity,
            else:  # orient_asu file exists, stride file should as well. just load asu
                orient_asu_file = self.oriented_asu.retrieve_file(name=structure_identifier)
                pose = Pose.from_file(orient_asu_file, sym_entry=sym_entry)  # , log=None, entity_names=[entry_entity])
                # entity = pose.entities[0]
                # entity.name = entry_entity  # make explicit
                # Pose already sets a symmetry and file_path upon constriction, so we don't need to set
                # pose.symmetry = symmetry
                # oriented_pose.file_path = oriented_pose.file_path
                all_structures.append(pose)
        if non_viable_structures:
            non_str = ', '.join(non_viable_structures)
            orient_log.error(f'The Model'
                             f'{f"s {non_str} were" if len(non_viable_structures) > 1 else f" {non_str} was"} '
                             f'unable to be oriented properly')
        return all_structures

    def preprocess_structures_for_design(self, structures: list[Structure], script_out_path: str | bytes = os.getcwd(),
                                         load_resources: bool = False, batch_commands: bool = True) -> \
            tuple[list, bool, bool]:
        """Assess whether Structure objects require any processing prior to design calculations.
        Processing includes relaxation "refine" into the energy function and/or modelling missing segments "loop model"

        Args:
            structures: An iterable of Structure objects of interest with the following attributes:
                file_path, symmetry, name, make_loop_file(), make_blueprint_file()
            script_out_path: Where should Entity processing commands be written?
            load_resources: Whether resources have been specified to be loaded already
            batch_commands: Whether commands should be made for batch submission
        Returns:
            Any instructions, then booleans for whether designs are pre_refined and whether they are pre_loop_modeled
        """
        # Todo
        #  Need to move make_loop_file to Pose/Structure (with SequenceProfile superclass)
        self.refined.make_path()
        refine_names = self.refined.retrieve_names()
        refine_dir = self.refined.location
        self.full_models.make_path()
        full_model_names = self.full_models.retrieve_names()
        full_model_dir = self.full_models.location
        # Identify the entities to refine and to model loops before proceeding
        structures_to_refine, structures_to_loop_model, sym_def_files = [], [], {}
        for structure in structures:  # if structure is here, the file should've been oriented...
            sym_def_files[structure.symmetry] = \
                sdf_lookup() if structure.symmetry == 'C1' else sdf_lookup(structure.symmetry)
            if structure.name not in refine_names:  # assumes oriented_asu structure name is the same
                structures_to_refine.append(structure)
            if structure.name not in full_model_names:  # assumes oriented_asu structure name is the same
                structures_to_loop_model.append(structure)

        # query user and set up commands to perform refinement on missing entities
        info_messages = []
        pre_refine = False
        if structures_to_refine:  # if files found unrefined, we should proceed
            logger.info('The following oriented entities are not yet refined and are being set up for refinement'
                        ' into the Rosetta ScoreFunction for optimized sequence design: '
                        f'{", ".join(sorted(set(structure.name for structure in structures_to_refine)))}')
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
                refine_cmds = [script_cmd + refine_cmd + ['-in:file:s', structure.file_path, '-parser:script_vars'] +
                               [f'sdf={sym_def_files[structure.symmetry]}',
                                f'symmetry={"make_point_group" if structure.symmetry != "C1" else "asymmetric"}']
                               for structure in structures_to_refine]
                if batch_commands:
                    commands_file = \
                        write_commands([subprocess.list2cmdline(cmd) for cmd in refine_cmds], out_path=refine_dir,
                                       name=f'{starttime}-refine_entities')
                    refine_sbatch = distribute(file=commands_file, out_path=script_out_path, scale=refine,
                                               log_file=os.path.join(refine_dir, f'{refine}.log'),
                                               max_jobs=int(len(refine_cmds) / 2 + 0.5),
                                               number_of_commands=len(refine_cmds))
                    refine_sbatch_message = \
                        'Once you are satisfied%snter the following to distribute refine jobs:\n\tsbatch %s' \
                        % (', you can run this script at any time. E' if load_resources else ', e', refine_sbatch)
                    info_messages.append(refine_sbatch_message)
                else:
                    raise NotImplementedError('Refinement run_in_shell functionality hasn\'t been implemented yet')
                load_resources = True

        # query user and set up commands to perform loop modelling on missing entities
        pre_loop_model = False
        if structures_to_loop_model:
            logger.info('The following structures have not been modelled for disorder. Missing loops will '
                        'be built for optimized sequence design: %s'
                        % ', '.join(sorted(set(structure.name for structure in structures_to_loop_model))))
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
                variables = [('script_nstruct', '100')]  # generate 100 trial loops, 500 is typically sufficient
                flags.append('-parser:script_vars %s' % ' '.join(f'{var}={val}' for var, val in variables))
                with open(flags_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(flags))
                loop_model_cmd = [f'@{flags_file}', '-parser:protocol',
                                  os.path.join(rosetta_scripts, 'loop_model_ensemble.xml'), '-parser:script_vars']
                # Make all output paths and files for each loop ensemble
                # logger.info('Preparing blueprint and loop files for structure:')
                # for structure in structures_to_loop_model:
                loop_model_cmds = []
                for idx, structure in enumerate(structures_to_loop_model):
                    structure_out_path = os.path.join(full_model_dir, structure.name)
                    make_path(structure_out_path)  # make a new directory for each structure
                    structure.renumber_residues()
                    structure_loop_file = structure.make_loop_file(out_path=full_model_dir)
                    if not structure_loop_file:  # no loops found, copy input file to the full model
                        copy_cmd = ['scp', self.refined.store(structure.name), self.full_models.store(structure.name)]
                        loop_model_cmds.append(
                            write_shell_script(subprocess.list2cmdline(copy_cmd), name=structure.name,
                                               out_path=full_model_dir))
                        continue
                    structure_blueprint = structure.make_blueprint_file(out_path=full_model_dir)
                    structure_cmd = script_cmd + loop_model_cmd + \
                        [f'blueprint={structure_blueprint}', f'loop_file={structure_loop_file}',
                         '-in:file:s', self.refined.store(structure.name), '-out:path:pdb', structure_out_path] + \
                        (['-symmetry:symmetry_definition', sym_def_files[structure.symmetry]]
                         if structure.symmetry != 'C1' else [])
                    # create a multimodel from all output loop models
                    multimodel_cmd = ['python', models_to_multimodel_exe, '-d', structure_loop_file,
                                      '-o', os.path.join(full_model_dir, f'{structure.name}_ensemble.pdb')]
                    # copy the first model from output loop models to be the full model
                    copy_cmd = ['scp', os.path.join(structure_out_path, f'{structure.name}_0001.pdb'),
                                self.full_models.store(structure.name)]
                    loop_model_cmds.append(
                        write_shell_script(subprocess.list2cmdline(structure_cmd), name=structure.name,
                                           out_path=full_model_dir,
                                           additional=[subprocess.list2cmdline(multimodel_cmd),
                                                       subprocess.list2cmdline(copy_cmd)]))
                if batch_commands:
                    loop_cmds_file = \
                        write_commands(loop_model_cmds, name=f'{starttime}-loop_model_entities',
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
                    raise NotImplementedError('Loop modeling run_in_shell functionality hasn\'t been implemented yet')

        return info_messages, pre_refine, pre_loop_model


class DatabaseFactory:
    """Return a Database instance by calling the Factory instance with the Database source name

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the named Database
    """

    def __init__(self, **kwargs):
        self._databases = {}

    def __call__(self, source: str = os.path.join(os.getcwd(), f'{program_name}{data.title()}'), sql: bool = False,
                 **kwargs) -> Database:
        """Return the specified Database object singleton

        Args:
            source: The Database source path, or name if SQL Database
            sql: Whether the Database is a SQL Database
        Returns:
            The instance of the specified Database
        """
        database = self._databases.get(source)
        if database:
            return database
        elif sql:
            raise NotImplementedError('SQL set up has not been completed!')
        else:
            make_path(source)
            pdbs = os.path.join(source, 'PDBs')  # Used to store downloaded PDB's
            sequence_info_dir = os.path.join(source, sequence_info)
            external_db = os.path.join(source, 'ExternalDatabases')
            # pdbs subdirectories
            orient_dir = os.path.join(pdbs, 'oriented')
            orient_asu_dir = os.path.join(pdbs, 'oriented_asu')
            refine_dir = os.path.join(pdbs, 'refined')
            full_model_dir = os.path.join(pdbs, 'full_models')
            stride_dir = os.path.join(pdbs, 'stride')
            # sequence_info subdirectories
            sequences = os.path.join(sequence_info_dir, 'sequences')
            profiles = os.path.join(sequence_info_dir, 'profiles')
            # external database subdirectories
            pdb_api = os.path.join(external_db, 'pdb')
            # pdb_entity_api = os.path.join(external_db, 'pdb_entity')
            # pdb_assembly_api = os.path.join(external_db, 'pdb_assembly')
            uniprot_api = os.path.join(external_db, 'uniprot')
            logger.info(f'Initializing {source} {Database.__name__}')

            self._databases[source] = \
                Database(orient_dir, orient_asu_dir, refine_dir, full_model_dir, stride_dir, sequences, profiles,
                         pdb_api, uniprot_api, sql=None)

        return self._databases[source]

    def get(self, **kwargs) -> Database:
        """Return the specified Database object singleton

        Keyword Args:
            source=current_working_directory/Data (str): The Database source path, or name if SQL Database
            sql=False (bool): Whether the Database is a SQL Database
        Returns:
            The instance of the specified Database
        """
        return self.__call__(**kwargs)


database_factory: Annotated[DatabaseFactory,
                            'Calling this factory method returns the single instance of the Database class located at '
                            'the "source" keyword argument'] = \
    DatabaseFactory()
"""Calling this factory method returns the single instance of the Database class located at the "source" keyword 
argument"""


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


def read_json(file_name, **kwargs) -> dict | None:
    """Use json.load to read an object from a file

    Args:
        file_name: The location of the file to write
    Returns:
        The json data in the file
    """
    with open(file_name, 'r') as f_save:
        data = json.load(f_save, **kwargs)

    return data


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
            from Pose import Model
            self.load_file = Model.from_pdb
            self.save_file = not_implemented
        elif '.json' in extension:
            self.load_file = read_json
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

    def make_path(self, condition: bool = True):
        """Make all required directories in specified path if it doesn't exist, and optional condition is True

        Args:
            condition: A condition to check before the path production is executed
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

    def retrieve_files(self) -> list:
        """Returns the actual location of all files in the stored .location"""
        path = self.store()
        files = sorted(glob(path))
        if not files:
            self.log.info(f'No files found for "{path}"')
        return files

    def retrieve_names(self) -> list[str]:
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
            data = self._load_data(name, log=None)  # attempt to retrieve the new data
            if data:
                setattr(self, name, data)  # attempt to store the new data as an attribute
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
        """Return the data located in a particular entry specified by name"""
        if self.sql:
            dummy = True
        else:
            file = self.retrieve_file(name)
            if file:
                return self.load_file(file, **kwargs)

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


# class EntityDataStore(DataStore):
#     def retrieve_data(self, name: str = None, **kwargs) -> dict | None:
#         """Return data requested by PDB EntityID. Loads into the Database or queries the PDB API
#
#         Args:
#             name: The name of the data to be retrieved. Will be found with location and extension attributes
#         Returns:
#             If the data is available, the object requested will be returned, else None
#         """
#         data = super().retrieve_data(name=name)
#         #         data = getattr(self, name, None)
#         #         if data:
#         #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
#         #         else:
#         #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
#         #             if data:
#         #                 setattr(self, name, data)  # attempt to store the new data as an attribute
#         #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
#         #
#         #         return data
#         if not data:
#             request = query_entity_id(entity_id=name)
#             if not request:
#                 logger.warning(f'PDB API found no matching results for {name}')
#             else:
#                 data = request.json()
#                 # setattr(self, name, data)
#                 self.store_data(data, name=name)
#
#         return data
#
#
# class AssemblyDataStore(DataStore):
#     def retrieve_data(self, name: str = None, **kwargs) -> dict | None:
#         """Return data requested by PDB AssemblyID. Loads into the Database or queries the PDB API
#
#         Args:
#             name: The name of the data to be retrieved. Will be found with location and extension attributes
#         Returns:
#             If the data is available, the object requested will be returned, else None
#         """
#         data = super().retrieve_data(name=name)
#         #         data = getattr(self, name, None)
#         #         if data:
#         #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
#         #         else:
#         #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
#         #             if data:
#         #                 setattr(self, name, data)  # attempt to store the new data as an attribute
#         #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
#         #
#         #         return data
#         if not data:
#             request = query_assembly_id(assembly_id=name)
#             if not request:
#                 logger.warning(f'PDB API found no matching results for {name}')
#             else:
#                 data = request.json()
#                 # setattr(self, name, data)
#                 self.store_data(data, name=name)
#
#         return data


class PDBDataStore(DataStore):
    def __init__(self, location: str = None, extension: str = '.json', sql=None, log: Logger = logger):
        super().__init__(location=location, extension=extension, sql=sql, log=log)
        # pdb_entity_api: str | bytes | Path = os.path.join(self.location, 'pdb_entity')
        # pdb_assembly_api: str | bytes | Path = os.path.join(self.location, 'pdb_assembly')
        # self.entity_api = EntityDataStore(location=pdb_entity_api, extension='.json', sql=self.sql, log=self.log)
        # self.assembly_api = AssemblyDataStore(location=pdb_assembly_api, extension='.json', sql=self.sql, log=self.log)
        # make_path(pdb_entity_api)
        # make_path(pdb_assembly_api)

    def retrieve_entity_data(self, name: str = None, **kwargs) -> dict | None:
        """Return data requested by PDB EntityID. Loads into the Database or queries the PDB API

        Args:
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        data = super().retrieve_data(name=name)
        #         data = getattr(self, name, None)
        #         if data:
        #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
        #         else:
        #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
        #             if data:
        #                 setattr(self, name, data)  # attempt to store the new data as an attribute
        #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
        #
        #         return data
        if not data:
            request = query_entity_id(entity_id=name)
            if not request:
                logger.warning(f'PDB API found no matching results for {name}')
            else:
                data = request.json()
                # setattr(self, name, data)
                self.store_data(data, name=name)

        return data

    def retrieve_assembly_data(self, name: str = None, **kwargs) -> dict | None:
        """Return data requested by PDB AssemblyID. Loads into the Database or queries the PDB API

        Args:
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        data = super().retrieve_data(name=name)
        #         data = getattr(self, name, None)
        #         if data:
        #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
        #         else:
        #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
        #             if data:
        #                 setattr(self, name, data)  # attempt to store the new data as an attribute
        #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
        #
        #         return data
        if not data:
            request = query_assembly_id(assembly_id=name)
            if not request:
                logger.warning(f'PDB API found no matching results for {name}')
            else:
                data = request.json()
                # setattr(self, name, data)
                self.store_data(data, name=name)

        return data

    def retrieve_data(self, entry: str = None, assembly_id: str = None, assembly_integer: int | str = None,
                      entity_id: str = None, entity_integer: int | str = None, chain: str = None, **kwargs) -> \
            dict | list[list[str]] | None:
        """Return data requested by PDB identifier. Loads into the Database or queries the PDB API

        Args:
            entry: The 4 character PDB EntryID of interest
            assembly_id: The AssemblyID to query with format (1ABC-1)
            assembly_integer: The particular assembly integer to query. Must include entry as well
            entity_id: The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
            entity_integer: The entity integer from the EntryID of interest
            chain: The polymer "chain" identifier otherwise known as the "asym_id" from the PDB EntryID of interest
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        if entry:
            if len(entry) == 4:
                if entity_integer:
                    # logger.debug(f'Querying PDB API with {entry}_{entity_integer}')
                    # data = self.entity_api.retrieve_data(name=f'{entry}_{entity_integer}')
                    # return parse_entities_json([self.entity_api.retrieve_data(name=f'{entry}_{entity_integer}')])
                    return parse_entities_json([self.retrieve_entity_data(name=f'{entry}_{entity_integer}')])
                elif assembly_integer:
                    # logger.debug(f'Querying PDB API with {entry}-{assembly_integer}')
                    # data = self.assembly_api.retrieve_data(name=f'{entry}_{assembly_integer}')
                    # return parse_assembly_json(self.assembly_api.retrieve_data(name=f'{entry}-{assembly_integer}'))
                    return parse_assembly_json(self.retrieve_assembly_data(name=f'{entry}-{assembly_integer}'))
                else:
                    # logger.debug(f'Querying PDB API with {entry}')
                    # perform the normal DataStore routine with super(), however, finish with API call if no data found
                    data = super().retrieve_data(name=entry)
                    #         data = getattr(self, name, None)
                    #         if data:
                    #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
                    #         else:
                    #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
                    #             if data:
                    #                 setattr(self, name, data)  # attempt to store the new data as an attribute
                    #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
                    #
                    #         return data
                    if not data:
                        entry_request = query_entry_id(entry)
                        if not entry_request:
                            logger.warning(f'PDB API found no matching results for {entry}')
                        else:
                            data = entry_request.json()
                            # setattr(self, entry, data)
                            self.store_data(data, name=entry)

                    data = dict(entity=parse_entities_json([self.retrieve_entity_data(name=f'{entry}_{integer}')
                                                            for integer in range(1, int(data['rcsb_entry_info']
                                                                                        ['polymer_entity_count']) + 1)
                                                            ]),
                                **parse_entry_json(data))
                    if chain:
                        integer = None
                        for entity_idx, chains in data.get('entity').items():
                            if chain in chains:
                                integer = entity_idx
                                break
                        if integer:
                            # logger.debug(f'Querying PDB API with {entry}_{integer}')
                            return self.retrieve_entity_data(name=f'{entry}_{integer}')
                        else:
                            raise KeyError(f'No chain "{chain}" found in PDB ID {entry}. Possible chains '
                                           f'{", ".join(ch for chns in data.get("entity", {}).items() for ch in chns)}')
                    else:  # provide the formatted PDB API Entry ID information
                        return data
            else:
                logger.warning(
                    f'EntryID "{entry}" is not of the required format and will not be found with the PDB API')
        elif assembly_id:
            entry, assembly_integer, *extra = assembly_id.split('-')
            if not extra and len(entry) == 4:
                # logger.debug(f'Querying PDB API with {entry}-{assembly_integer}')
                # data = self.assembly_api.retrieve_data(name=f'{entry}-{assembly_integer}')
                return parse_assembly_json(self.retrieve_assembly_data(name=f'{entry}-{assembly_integer}'))

            logger.warning(
                f'AssemblyID "{entry}-{assembly_integer}" is not of the required format and will not be found '
                f'with the PDB API')

        elif entity_id:
            entry, entity_integer, *extra = entity_id.split('_')
            if not extra and len(entry) == 4:
                # logger.debug(f'Querying PDB API with {entry}_{entity_integer}')
                # data = self.entity_api.retrieve_data(name=f'{entry}_{entity_integer}')
                return parse_entities_json([self.retrieve_entity_data(name=f'{entry}_{entity_integer}')])

            logger.warning(
                f'EntityID "{entry}_{entity_integer}" is not of the required format and will not be found with '
                f'the PDB API')

        else:  # this could've been passed as name=. This case would need to be solved with some parsing of the splitter
            raise RuntimeError(f'No valid arguments passed to {self.retrieve_data.__name__}. Valid arguments include: '
                               f'entry, assembly_id, assembly_integer, entity_id, entity_integer, chain')


class UniProtDataStore(DataStore):
    def __init__(self, location: str = None, extension: str = '.json', sql=None, log: Logger = logger):
        super().__init__(location=location, extension=extension, sql=sql, log=log)

    def retrieve_data(self, name: str = None, **kwargs) -> dict | None:
        """Return data requested by UniProtID. Loads into the Database or queries the UniProt API

        Args:
            name: The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            If the data is available, the object requested will be returned, else None
        """
        data = super().retrieve_data(name=name)
        #         data = getattr(self, name, None)
        #         if data:
        #             self.log.debug(f'Info {name}{self.extension} was retrieved from DataStore')
        #         else:
        #             data = self._load_data(name, log=None)  # attempt to retrieve the new data
        #             if data:
        #                 setattr(self, name, data)  # attempt to store the new data as an attribute
        #                 self.log.debug(f'Database file {name}{self.extension} was loaded fresh')
        #
        #         return data
        if not data:
            response = query_uniprot(uniprot_id=name)
            if not response:
                logger.warning(f'UniprotKB API found no matching results for {name}')
            else:
                data = response.json()
                self.store_data(data, name=name)

        return data

    def is_thermophilic(self, uniprot_id: str) -> int:
        """Query if a UniProtID is thermophilic

        Args:
            uniprot_id: The formatted UniProtID which consists of either a 6 or 10 character code
        Returns:
            1 if the UniProtID of interest has an organism lineage from a thermophilic taxa, else 0
        """
        data = self.retrieve_data(name=uniprot_id)
        for element in data.get('organism', {}).get('lineage', []):
            if 'thermo' in element.lower():
                return 1  # True

        return 0  # False


class JobResources:
    """The intention of JobResources is to serve as a singular source of design info which is common accross all
    designs. This includes common paths, databases, and design flags which should only be set once in program operation,
    then shared across all member designs"""
    def __init__(self, program_root: str | bytes = None, **kwargs):
        """For common resources for all SymDesign outputs, ensure paths to these resources are available attributes"""
        try:
            if os.path.exists(program_root):
                self.program_root = program_root
            else:
                raise FileNotFoundError(f'Path does not exist!\n\t{program_root}')
        except TypeError:
            raise TypeError(f'Can\'t initialize {JobResources.__name__} without parameter "program_root"')

        # program_root subdirectories
        self.data = os.path.join(self.program_root, data.title())
        self.projects = os.path.join(self.program_root, projects)
        self.job_paths = os.path.join(self.program_root, 'JobPaths')
        self.sbatch_scripts = os.path.join(self.program_root, 'Scripts')
        # TODO ScoreDatabase integration
        self.all_scores = os.path.join(self.program_root, all_scores)

        # data subdirectories
        self.clustered_poses = os.path.join(self.data, 'ClusteredPoses')
        self.pdbs = os.path.join(self.data, 'PDBs')  # Used to store downloaded PDB's
        self.sequence_info = os.path.join(self.data, sequence_info)
        self.external_db = os.path.join(self.data, 'ExternalDatabases')
        # pdbs subdirectories
        self.orient_dir = os.path.join(self.pdbs, 'oriented')
        self.orient_asu_dir = os.path.join(self.pdbs, 'oriented_asu')
        self.refine_dir = os.path.join(self.pdbs, 'refined')
        self.full_model_dir = os.path.join(self.pdbs, 'full_models')
        self.stride_dir = os.path.join(self.pdbs, 'stride')
        # sequence_info subdirectories
        self.sequences = os.path.join(self.sequence_info, 'sequences')
        self.profiles = os.path.join(self.sequence_info, 'profiles')
        # external database subdirectories
        self.pdb_api = os.path.join(self.external_db, 'pdb')
        # self.pdb_entity_api = os.path.join(self.external_db, 'pdb_entity')
        # self.pdb_assembly_api = os.path.join(self.external_db, 'pdb_assembly')
        self.uniprot_api = os.path.join(self.external_db, 'uniprot')
        # try:
        # if not self.projects:  # used for subclasses
        # if not getattr(self, 'projects', None):  # used for subclasses
        #     self.projects = os.path.join(self.program_root, projects)
        # except AttributeError:
        #     self.projects = os.path.join(self.program_root, projects)
        # self.design_db = None
        # self.score_db = None
        make_path(self.pdb_api)
        # make_path(self.pdb_entity_api)
        # make_path(self.pdb_assembly_api)
        make_path(self.uniprot_api)
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
        self.resource_db = database_factory.get(source=self.data)
        # self.resource_db = Database(self.orient_dir, self.orient_asu_dir, self.refine_dir, self.full_model_dir,
        #                           self.stride_dir, self.sequences, self.profiles, self.pdb_api,
        #                           # self.pdb_entity_api, self.pdb_assembly_api,
        #                           self.uniprot_api, sql=None)  # , log=logger)
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

    # @staticmethod
    # def make_path(path: str | bytes, condition: bool = True):
    #     """Make all required directories in specified path if it doesn't exist, and optional condition is True
    #
    #     Args:
    #         path: The path to create
    #         condition: A condition to check before the path production is executed
    #     """
    #     if condition:
    #         os.makedirs(path, exist_ok=True)


class JobResourcesFactory:
    """Return a JobResource instance by calling the Factory instance

    Handles creation and allotment to other processes by making a shared pointer to the JobResource for the current Job
    """
    def __init__(self, **kwargs):
        self._resources = {}
        self._warn = False

    def __call__(self, **kwargs) -> JobResources:
        """Return the specified JobResources object singleton

        Returns:
            The instance of the specified JobResources
        """
        #         Args:
        #             source: The JobResources source name
        source = 'single'
        fragment_db = self._resources.get(source)
        if fragment_db:
            if kwargs and not self._warn:
                # try:
                #     fragment_db.update(kwargs)
                # except RuntimeError:
                self._warn = True
                logger.warning(f'Can\'t pass the new arguments {", ".join(kwargs.keys())} to JobResources '
                               f'since it was already initialized and is a singleton')
                # raise RuntimeError(f'Can\'t pass the new arguments {", ".join(kwargs.keys())} to JobResources '
                #                    f'since it was already initialized')
            return fragment_db
        else:
            logger.info(f'Initializing {JobResources.__name__}')
            self._resources[source] = JobResources(**kwargs)

        return self._resources[source]

    def get(self, **kwargs) -> JobResources:
        """Return the specified JobResources object singleton

        Returns:
            The instance of the specified JobResources
        """
        #         Keyword Args:
        #             source: The JobResource source name
        return self.__call__(**kwargs)


job_resources_factory = JobResourcesFactory()
