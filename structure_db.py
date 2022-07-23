from __future__ import annotations

import os
import subprocess
from copy import copy
from glob import glob
from logging import Logger
from pathlib import Path
from typing import Iterable, Annotated, AnyStr

from CommandDistributer import rosetta_flags, relax_flags, rosetta_variables, script_cmd, distribute
from database import Database, DataStore
from PathUtils import qs_bio, pdb_db, orient_log_file, rosetta_scripts, refine, models_to_multimodel_exe, program_name,\
    data, structure_info
import Pose
from Query.utils import boolean_choice
from Structure import Structure, parse_stride
from SymDesignUtils import unpickle, to_iterable, start_log, write_commands, starttime, make_path, write_shell_script
from classes.SymEntry import parse_symmetry_to_sym_entry, sdf_lookup

logger = start_log(name=__name__)
qsbio_confirmed = unpickle(qs_bio)


def _fetch_pdb_from_api(pdb_codes: str | list, assembly: int = 1, asu: bool = False, out_dir: AnyStr = os.getcwd(),
                        **kwargs) -> list[AnyStr]:  # Todo mmcif
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


def fetch_pdb_file(pdb_code: str, asu: bool = True, location: AnyStr = pdb_db, **kwargs) -> AnyStr | None:
    #                assembly: int = 1, out_dir: AnyStr = os.getcwd()
    """Fetch PDB object from PDBdb or download from PDB server

    Args:
        pdb_code: The PDB ID/code. If the biological assembly is desired, supply 1ABC_1 where '_1' is assembly ID
        asu: Whether to fetch the ASU
        location: Location of a local PDB mirror if one is linked on disk
    Keyword Args:
        assembly=None (int): Location of a local PDB mirror if one is linked on disk
        out_dir=os.getcwd() (AnyStr): The location to save retrieved files if fetched from PDB
    Returns:
        The path to the file if located successfully
    """
    # if location == pdb_db and asu:
    if os.path.exists(location) and asu:
        file_path = os.path.join(location, f'pdb{pdb_code.lower()}.ent')
        def get_pdb(): return sorted(glob(file_path))
        # Cassini format is above, KM local pdb and the escher PDB mirror is below
        # file_path = os.path.join(location, subdirectory(pdb_code), f'{pdb_code.lower()}.pdb')
        logger.debug(f'Searching for PDB file at "{file_path}"')
    else:
        get_pdb = _fetch_pdb_from_api

    # return a list where the matching file is the first (should only be one anyway)
    pdb_file = get_pdb(pdb_code, asu=asu, location=location, **kwargs)
    if not pdb_file:
        logger.warning(f'No matching file found for PDB: {pdb_code}')
    else:  # we should only find one file, therefore, return the first
        return pdb_file[0]


def orient_structure_files(files: Iterable[AnyStr], log: Logger = logger, symmetry: str = None,
                           out_dir: AnyStr = None) -> list[str]:
    """For a specified file and output directory, orient the file according to the provided symmetry where the
    resulting file will have the chains symmetrized and oriented in the coordinate frame as to have the major axis
    of symmetry along z, and additional axis along canonically defined vectors. If the symmetry is C1, then the monomer
    will be transformed so the center of mass resides at the origin

    Args:
        files: The location of the files to be oriented
        log: A log to report on operation success
        symmetry: The symmetry type to be oriented. Possible types in SymmetryUtils.valid_subunit_number
        out_dir: The directory that should be used to output files
    Returns:
        Filepath of oriented PDB
    """
    file_paths = []
    for file in files:
        model_name = os.path.basename(file)
        oriented_file_path = os.path.join(out_dir, model_name)
        if not os.path.exists(oriented_file_path):
            model = Pose.Model.from_file(file, log=log)  # must load entities to solve multi-component orient problem
            try:
                model.orient(symmetry=symmetry)
            except (ValueError, RuntimeError) as error:
                log.error(str(error))
                continue
            model.write(out_path=oriented_file_path)
            log.info(f'Oriented: {model_name}')

        file_paths.append(oriented_file_path)
    return file_paths


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


class StructureDatabase(Database):
    def __init__(self, full_models: AnyStr | Path = None, oriented: AnyStr | Path = None,
                 oriented_asu: AnyStr | Path = None, refined: AnyStr | Path = None, stride: AnyStr | Path = None,
                 **kwargs):
        # passed to Database
        # sql: sqlite = None, log: Logger = logger
        super().__init__(**kwargs)  # Database

        self.full_models = DataStore(location=full_models, extension='_ensemble.pdb', sql=self.sql, log=self.log,
                                     load_file=Pose.Model.from_pdb)
        self.oriented = DataStore(location=oriented, extension='.pdb', sql=self.sql, log=self.log,
                                  load_file=Pose.Model.from_pdb)
        self.oriented_asu = DataStore(location=oriented_asu, extension='.pdb', sql=self.sql, log=self.log,
                                      load_file=Pose.Model.from_pdb)
        self.refined = DataStore(location=refined, extension='.pdb', sql=self.sql, log=self.log,
                                 load_file=Pose.Model.from_pdb)
        self.stride = DataStore(location=stride, extension='.stride', sql=self.sql, log=self.log,
                                load_file=parse_stride)

        # Todo only load the necessary structural template
        self.sources = [self.oriented_asu, self.refined, self.stride]  # self.full_models

    def orient_structures(self, structure_identifiers: Iterable[str], symmetry: str = 'C1', by_file: bool = False) -> \
            list[Pose.Model] | list:
        """Given entity_ids and their corresponding symmetry, retrieve .pdb files, orient and save Database files then
        return the ASU for each

        Args:
            structure_identifiers: The names of all entity_ids requiring orientation
            symmetry: The symmetry to treat each passed Entity. Default assumes no symmetry
            by_file: Whether to parse the structure_identifiers as file paths. Default treats as PDB EntryID or EntityID
        Returns:
            The symmetrized Poses, oriented in a canonical orientation
        """
        if not structure_identifiers:
            return []
        # using Pose enables simple ASU writing
        # Pose isn't the best for orient since the structure isn't oriented the symmetry wouldn't work
        # It could still happen, but we certainly shouldn't pass sym_entry to it
        self.oriented.make_path()
        orient_dir = self.oriented.location
        models_dir = os.path.dirname(orient_dir)  # this is the case because it was specified this way, but not required
        self.oriented_asu.make_path()
        self.stride.make_path()
        # orient_log = start_log(name='orient', handler=1)
        orient_log = \
            start_log(name='orient', handler=2, location=os.path.join(orient_dir, orient_log_file), propagate=True)
        if symmetry:
            sym_entry = parse_symmetry_to_sym_entry(symmetry=symmetry)
            if by_file:
                type_ = 'files'
                oriented_filepaths = \
                    orient_structure_files(structure_identifiers, log=orient_log, symmetry=symmetry, out_dir=orient_dir)
                # pull out the structure names and use below to retrieve the oriented file
                structure_identifiers = \
                    list(map(os.path.basename, [os.path.splitext(file)[0]
                                                for file in filter(None, oriented_filepaths)]))
            else:
                type_ = 'IDs'
            logger.info(f'The requested {type_} are being checked for proper orientation with symmetry {symmetry}: '
                        f'{", ".join(structure_identifiers)}')
        else:
            sym_entry = None
            if by_file:
                type_ = 'files'
                structure_identifiers_ = []
                for file in structure_identifiers:
                    model = Pose.Model.from_file(file)
                    model.write(out_path=self.oriented.path_to(name=model.name))
                    # Write out each Entity in Model to ASU file for the oriented_asu database
                    model.file_path(out_path=self.oriented_asu.path_to(name=model.name))
                    with open(model.file_path, 'w') as f:
                        model.write_header(f)
                        for entity in model.entities:
                            # write each Entity to asu
                            entity.write(file_handle=f)
                            # save Stride results
                            entity.stride(to_file=self.stride.path_to(name=entity.name))
                    structure_identifiers_.append(model.name)
                structure_identifiers = structure_identifiers_
            else:
                type_ = 'IDs'
            logger.info(f'The requested {type_} are being set up into the DataBase: {", ".join(structure_identifiers)}')

        # Next work through the process of orienting the selected files
        orient_names = self.oriented.retrieve_names()
        orient_asu_names = self.oriented_asu.retrieve_names()
        all_structures = []
        non_viable_structures = []
        for structure_identifier in structure_identifiers:
            # First, check if the structure_identifier ASU has been processed. This happens when files are passed
            if structure_identifier in orient_asu_names:  # orient_asu file exists, stride should as well. Just load asu
                orient_asu_file = self.oriented_asu.retrieve_file(name=structure_identifier)
                pose = Pose.Pose.from_file(orient_asu_file, name=structure_identifier, sym_entry=sym_entry)
                # entity = pose.entities[0]
                # entity.name = entry_entity  # make explicit
                # Pose already sets a symmetry and file_path upon constriction, so we don't need to set
                # pose.symmetry = symmetry
                # oriented_pose.file_path = oriented_pose.file_path
                all_structures.append(pose.assembly)
            elif structure_identifier in orient_names:  # orient file exists, load asu, save and create stride
                orient_file = self.oriented.retrieve_file(name=structure_identifier)
                pose = Pose.Pose.from_file(orient_file, name=structure_identifier, sym_entry=sym_entry)
                # entity = pose.entities[0]
                # entity.name = pose.name  # use pose.name, not API name
                # Pose already sets a symmetry, so we don't need to set one
                # pose.symmetry = symmetry
                pose.file_path = pose.write(out_path=self.oriented_asu.path_to(name=structure_identifier))
                # save Stride results
                for entity in pose.entities:
                    entity.stride(to_file=self.stride.path_to(name=entity.name))
                all_structures.append(pose.assembly)  # entry_entity,
            # Use entry_entity only if not processed before
            else:  # they are missing, retrieve the proper files using PDB ID's
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
                model = Pose.Model.from_pdb(file_path, name=file_name)  # , sym_entry=sym_entry
                if entity:  # replace Structure from fetched file with the Entity Structure
                    # entry_entity will be formatted the exact same as the desired EntityID if it was provided correctly
                    entity = model.entity(entry_entity)
                    if not entity:  # we couldn't find the specified EntityID
                        logger.warning(f'For {entry_entity}, couldn\'t locate the specified Entity "{entity}". The'
                                       f' available Entities are {", ".join(entity.name for entity in pose.entities)}')
                        continue
                    else:
                        model = entity
                    entity_out_path = os.path.join(models_dir, f'{entry_entity}.pdb')
                    if symmetry == 'C1':  # Write out only the entity that was extracted
                        # Todo should we delete the source file?
                        entity_file_path = model.write(out_path=entity_out_path)
                    else:  # write out the entity as parsed. since this is assembly we should get the correct state
                        entity_file_path = model.write_oligomer(out_path=entity_out_path)

                    try:  # orient the Structure
                        model.orient(symmetry=symmetry, log=orient_log)
                    except (ValueError, RuntimeError) as err:
                        orient_log.error(str(err))
                        non_viable_structures.append(entry_entity)
                        continue
                    # Write out file for the orient database
                    orient_file = model.write_oligomer(out_path=self.oriented.path_to(name=entry_entity))
                    model.file_path = self.oriented_asu.path_to(name=entry_entity)
                    # Write out ASU file for the oriented_asu database
                    model.write(out_path=self.oriented_asu.path_to(name=entry_entity))
                    # save Stride results
                    model.stride(to_file=self.stride.path_to(name=entry_entity))

                else:  # orient the whole set of chains based on orient() multicomponent solution
                    try:  # orient the Structure
                        model.orient(symmetry=symmetry, log=orient_log)
                    except (ValueError, RuntimeError) as err:
                        orient_log.error(str(err))
                        non_viable_structures.append(entry_entity)
                        continue
                    # Write out file for the orient database
                    orient_file = model.write(out_path=self.oriented.path_to(name=entry_entity))
                    # Extract ASU from the Structure, save the file as .file_path for preprocess_structures_for_design
                    model.file_path = self.oriented_asu.path_to(name=entry_entity)
                    with open(model.file_path, 'w') as f:
                        model.write_header(f)
                        for entity in model.entities:
                            # write each Entity to asu
                            entity.write(file_handle=f)
                            # save Stride results
                            entity.stride(to_file=self.stride.path_to(name=entity.name))

                model.symmetry = symmetry
                all_structures.append(model)
                orient_log.info(f'Oriented: {orient_file}')
        if non_viable_structures:
            non_str = ', '.join(non_viable_structures)
            orient_log.error(f'The Model'
                             f'{f"s {non_str} were" if len(non_viable_structures) > 1 else f" {non_str} was"} '
                             f'unable to be oriented properly')
        return all_structures

    def preprocess_structures_for_design(self, structures: list[Structure], script_out_path: AnyStr = os.getcwd(),
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
            Any instructions, then booleans for whether to refine or whether to loop_model designs
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
        # Assume pre_refine is True until we find it isn't
        pre_refine = True
        if structures_to_refine:  # if files found unrefined, we should proceed
            logger.info('The following structures are not yet refined and are being set up for refinement'
                        ' into the Rosetta ScoreFunction for optimized sequence design: '
                        f'{", ".join(sorted(set(structure.name for structure in structures_to_refine)))}')
            print('Would you like to refine them now? If you plan on performing sequence design with models '
                  'containing them, it is highly recommended you perform refinement')
            if boolean_choice():
                run_pre_refine = True
            else:
                print('To confirm, asymmetric units are going to be generated with unrefined coordinates. Confirm '
                      'with "y" to ensure this is what you want')
                if boolean_choice():
                    run_pre_refine = False
                else:
                    run_pre_refine = True

            if run_pre_refine:
                pre_refine = False
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
            else:
                pre_refine = True

        # query user and set up commands to perform loop modelling on missing entities
        # Assume pre_loop_model is True until we find it isn't
        pre_loop_model = True
        if structures_to_loop_model:
            pre_loop_model = False
            logger.info('The following structures have not been modelled for disorder. Missing loops will '
                        'be built for optimized sequence design: %s'
                        % ', '.join(sorted(set(structure.name for structure in structures_to_loop_model))))
            print('Would you like to model loops for these structures now? If you plan on performing sequence '
                  'design with them, it is highly recommended you perform loop modelling to avoid designed clashes')
            if boolean_choice():
                run_loop_model = True
            else:
                print('To confirm, asymmetric units are going to be generated without disordered loops. Confirm '
                      'with "y" to ensure this is what you want')
                if boolean_choice():
                    run_loop_model = False
                else:
                    run_loop_model = True

            if run_loop_model:
                pre_loop_model = False
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
                        copy_cmd = ['scp', self.refined.path_to(structure.name), self.full_models.path_to(structure.name)]
                        loop_model_cmds.append(
                            write_shell_script(subprocess.list2cmdline(copy_cmd), name=structure.name,
                                               out_path=full_model_dir))
                        continue
                    structure_blueprint = structure.make_blueprint_file(out_path=full_model_dir)
                    structure_cmd = script_cmd + loop_model_cmd + \
                        [f'blueprint={structure_blueprint}', f'loop_file={structure_loop_file}',
                         '-in:file:s', self.refined.path_to(structure.name), '-out:path:pdb', structure_out_path] + \
                        (['-symmetry:symmetry_definition', sym_def_files[structure.symmetry]]
                         if structure.symmetry != 'C1' else [])
                    # create a multimodel from all output loop models
                    multimodel_cmd = ['python', models_to_multimodel_exe, '-d', structure_loop_file,
                                      '-o', os.path.join(full_model_dir, f'{structure.name}_ensemble.pdb')]
                    # copy the first model from output loop models to be the full model
                    copy_cmd = ['scp', os.path.join(structure_out_path, f'{structure.name}_0001.pdb'),
                                self.full_models.path_to(structure.name)]
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
            else:
                pre_loop_model = True

        return info_messages, pre_refine, pre_loop_model


class StructureDatabaseFactory:
    """Return a StructureDatabase instance by calling the Factory instance with the StructureDatabase source name

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the named StructureDatabase
    """

    def __init__(self, **kwargs):
        # self._databases = {}
        self._database = None

    def __call__(self, source: str = os.path.join(os.getcwd(), f'{program_name}{data.title()}'), sql: bool = False,
                 **kwargs) -> StructureDatabase:
        """Return the specified StructureDatabase object singleton

        Args:
            source: The StructureDatabase source path, or name if SQL database
            sql: Whether the StructureDatabase is a SQL database
        Returns:
            The instance of the specified StructureDatabase
        """
        # Todo potentially configure, however, we really only want a single database
        # database = self._databases.get(source)
        # if database:
        #     return database
        if self._database:
            return self._database
        elif sql:
            raise NotImplementedError('SQL set up has not been completed!')
        else:
            structure_info_dir = os.path.join(source, structure_info)
            pdbs = os.path.join(structure_info_dir, 'PDBs')  # Used to path downloaded PDB's
            # stride directory
            stride_dir = os.path.join(structure_info_dir, 'stride')
            make_path(stride_dir)
            # pdbs subdirectories
            orient_dir = os.path.join(pdbs, 'oriented')
            orient_asu_dir = os.path.join(pdbs, 'oriented_asu')
            refine_dir = os.path.join(pdbs, 'refined')
            full_model_dir = os.path.join(pdbs, 'full_models')
            make_path(orient_dir)
            make_path(orient_asu_dir)
            make_path(refine_dir)
            make_path(full_model_dir)
            logger.info(f'Initializing {source} {StructureDatabase.__name__}')

            # self._databases[source] = \
            #     StructureDatabase(orient_dir, orient_asu_dir, refine_dir, full_model_dir, stride_dir, sql=None)
            self._database = \
                StructureDatabase(full_model_dir, orient_dir, orient_asu_dir, refine_dir, stride_dir, sql=None)

        # return self._databases[source]
        return self._database

    def get(self, **kwargs) -> StructureDatabase:
        """Return the specified Database object singleton

        Keyword Args:
            source: str = 'current_working_directory/Data' - The StructureDatabase source path, or name if SQL database
            sql: bool = False - Whether the StructureDatabase is a SQL database
        Returns:
            The instance of the specified StructureDatabase
        """
        return self.__call__(**kwargs)


structure_database_factory: Annotated[StructureDatabaseFactory,
                                      'Calling this factory method returns the single instance of the Database class '
                                      'located at the "source" keyword argument'] = \
    StructureDatabaseFactory()
"""Calling this factory method returns the single instance of the Database class located at the "source" keyword 
argument
"""
