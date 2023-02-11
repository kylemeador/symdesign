from __future__ import annotations

import logging
import os
import shutil
import subprocess
from glob import glob
from logging import Logger
from pathlib import Path
from typing import Iterable, Annotated, AnyStr

from .database import Database, DataStore
from .query.utils import boolean_choice
from symdesign import flags, structure, utils
from symdesign.utils import CommandDistributer, rosetta
putils = utils.path

# Todo adjust the logging level for this module?
logger = logging.getLogger(__name__)
qsbio_confirmed: Annotated[dict[str, list[int]],
                           'PDB EntryID mapped to the correct biological assemblies as specified by a QSBio confidence'\
                           " of high or very high. Lowercase EntryID keys are mapped to a list of integer values"] = \
    utils.unpickle(putils.qs_bio)
"""PDB EntryID mapped to the correct biological assemblies as specified by a QSBio confidence of high or very high.
Lowercase EntryID keys are mapped to a list of integer values
"""


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
    for pdb_code in utils.to_iterable(pdb_codes):
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
                             "only be a .cif file available for this entry, which currently can't be parsed")
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


def fetch_pdb_file(pdb_code: str, asu: bool = True, location: AnyStr = putils.pdb_db, **kwargs) -> AnyStr | None:
    #                assembly: int = 1, out_dir: AnyStr = os.getcwd()
    """Fetch PDB object from PDBdb or download from PDB server

    Args:
        pdb_code: The PDB ID/code. If the biological assembly is desired, supply 1ABC_1 where '_1' is assembly ID
        asu: Whether to fetch the ASU
        location: Location of a local PDB mirror if one is linked on disk
    Keyword Args:
        assembly: int = None - Location of a local PDB mirror if one is linked on disk
        out_dir: AnyStr = os.getcwd() - The location to save retrieved files if fetched from PDB
    Returns:
        The path to the file if located successfully
    """
    # if location == putils.pdb_db and asu:
    if os.path.exists(location) and asu:
        file_path = os.path.join(location, f'pdb{pdb_code.lower()}.ent')
        def get_pdb(*args, **_kwargs): return sorted(glob(file_path))
        # Cassini format is above, KM local pdb and the escher PDB mirror is below
        # file_path = os.path.join(location, subdirectory(pdb_code), f'{pdb_code.lower()}.pdb')
        logger.debug(f'Searching for PDB file at "{file_path}"')
    else:
        get_pdb = _fetch_pdb_from_api

    # The matching file is the first (should only be one)
    pdb_file: list[str] = get_pdb(pdb_code, asu=asu, location=location, **kwargs)
    if not pdb_file:  # Empty list
        logger.warning(f'No matching file found for PDB: {pdb_code}')
        return None
    else:  # We should only find one file, therefore, return the first
        return pdb_file[0]


def orient_structure_files(files: Iterable[AnyStr], log: Logger = logger, symmetry: str = None,
                           out_dir: AnyStr = None) -> list[str] | list:
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
        model_name, extension = os.path.splitext(os.path.basename(file))
        # Todo refactor this file_path resolution as not very precise for mmcif files...
        if '.pdb' in extension:  # Use the original extension which may have an assembly provided
            output_extension = extension
        else:  # This could be mmcif
            output_extension = '.pdb'

        oriented_file_path = os.path.join(out_dir, f'{model_name}{output_extension}')  # .pdb')
        if not os.path.exists(oriented_file_path):
            # Must load entities to solve multi-component orient problem
            model = structure.model.Model.from_file(file, log=log)
            try:
                model.orient(symmetry=symmetry)
            except (ValueError, RuntimeError) as error:
                log.error(str(error))
                continue
            model.write(out_path=oriented_file_path)
            log.info(f'Oriented: {model_name}')

        file_paths.append(oriented_file_path)
    return file_paths


def query_qs_bio(pdb_code: str) -> int:
    """Retrieve the first matching Very High/High confidence QSBio assembly from a PDB EntryID

    Args:
        pdb_code: The 4 letter PDB code to query
    Returns:
        The integer of the corresponding PDB Assembly ID according to the QSBio assembly
    """
    biological_assemblies = qsbio_confirmed.get(pdb_code.lower())
    if biological_assemblies:
        # Get the first assembly in matching oligomers
        assembly = biological_assemblies[0]
    else:
        assembly = 1
        logger.warning(f'No confirmed biological assembly for entry {pdb_code.lower()}, '
                       f'using PDB default assembly {assembly}')
    return assembly


class StructureDatabase(Database):
    def __init__(self, full_models: AnyStr | Path = None, oriented: AnyStr | Path = None,
                 oriented_asu: AnyStr | Path = None, refined: AnyStr | Path = None, stride: AnyStr | Path = None,
                 **kwargs):
        # passed to Database
        # sql: sqlite = None, log: Logger = logger
        super().__init__(**kwargs)  # Database

        self.full_models = DataStore(location=full_models, extension='_ensemble.pdb', glob_extension='_ensemble.pdb*',
                                     sql=self.sql, log=self.log, load_file=structure.model.Model.from_pdb)
        self.oriented = DataStore(location=oriented, extension='.pdb', glob_extension='.pdb*',
                                  sql=self.sql, log=self.log, load_file=structure.model.Model.from_pdb)
        self.oriented_asu = DataStore(location=oriented_asu, extension='.pdb', glob_extension='.pdb*',
                                      sql=self.sql, log=self.log, load_file=structure.model.Model.from_pdb)
        self.refined = DataStore(location=refined, extension='.pdb', glob_extension='.pdb*',
                                 sql=self.sql, log=self.log, load_file=structure.model.Model.from_pdb)
        self.stride = DataStore(location=stride, extension='.stride', sql=self.sql, log=self.log,
                                load_file=structure.utils.parse_stride)

        # Todo only load the necessary structural template
        self.sources = [self.oriented_asu, self.refined, self.stride]  # self.full_models

    def download_structures(self, structure_identifiers: Iterable[str], out_dir: str = os.getcwd()) -> \
            list[structure.model.Model | structure.model.Entity]:
        """Given EntryIDs/EntityIDs, retrieve/save .pdb files, then return a Structure for each identifier

        Args:
            structure_identifiers: The names of all entity_ids requiring orientation
            out_dir: The directory to write downloaded files to
        Returns:
            The requested Model/Entity instances
        """
        all_models = []
        for structure_identifier in structure_identifiers:
            # Retrieve the proper files using PDB ID's
            entry = structure_identifier.split('_')
            assembly = entity = None
            if len(entry) == 2:
                entry, entity = entry
                logger.debug(f'Fetching entry {entry}, entity {entity} from PDB')
            else:
                entry = structure_identifier.split('-')
                if len(entry) == 2:
                    entry, assembly = entry
                    logger.debug(f'Fetching entry {entry}, assembly {assembly} from PDB')
                else:
                    entry = structure_identifier
                    logger.debug(f'Fetching entry {entry} from PDB')

            asu = False
            # if symmetry == 'C1':
            #     # Todo modify if monomers are removed from qs_bio
            #     assembly = query_qs_bio(entry)
            #     # assembly = None  # 1 is the default
            #     # asu = True <- NOT NECESSARILY A MONOMERIC FILE!
            # else:
            #     asu = False
            if assembly is None:
                assembly = query_qs_bio(entry)
            # Get the specified file_path for the assembly state of interest
            file_path = fetch_pdb_file(entry, assembly=assembly, asu=asu, out_dir=out_dir)

            if not file_path:
                logger.warning(f"Couldn't locate the file '{file_path}', there may have been an issue "
                               'downloading it from the PDB. Attempting to copy from job data source...')
                # Todo
                raise NotImplementedError("This functionality hasn't been written yet. Use the canonical_pdb1/2 "
                                          'attribute of PoseJob to pull the pdb file source.')
            # Remove any PDB Database mirror specific naming from fetch_pdb_file such as pdb1ABC.ent
            file_name = os.path.splitext(os.path.basename(file_path))[0].replace('pdb', '')
            model = structure.model.Model.from_pdb(file_path, name=file_name)
            # entity_out_path = os.path.join(out_dir, f'{structure_identifier}.pdb')
            if entity is not None:  # Replace Structure from fetched file with the Entity Structure
                # structure_identifier will be formatted the exact same as the desired EntityID
                # if it was provided correctly
                entity = model.entity(structure_identifier)
                if entity:
                    model = entity
                else:  # We couldn't find the specified EntityID
                    logger.warning(f"For {structure_identifier}, couldn't locate the specified Entity '{entity}'. The"
                                   f' available Entities are {", ".join(entity.name for entity in model.entities)}')
                    continue
            #     # Write out the entity as parsed. since this is assembly we should get the correct state
            #     entity_file_path = model.write_oligomer(out_path=entity_out_path)
            # elif assembly is not None:
            #     # Separate the Entity instances?
            # else:
            #     # Write out file for the orient database
            #     orient_file = model.write(out_path=entity_out_path)

            all_models.append(model)

        return all_models

    def orient_structures(self, structure_identifiers: Iterable[str], symmetry: str = 'C1', by_file: bool = False) -> \
            list[structure.model.Model | structure.model.Entity] | list:
        """Given EntryIDs/EntityIDs, and their corresponding symmetry, retrieve .pdb files, orient and save files to
        the Database, then return the symmetric Model for each

        Args:
            structure_identifiers: The names of all entity_ids requiring orientation
            symmetry: The symmetry to treat each passed Entity. Default assumes no symmetry
            by_file: Whether to parse the structure_identifiers as file paths. Default treats as PDB EntryID or EntityID
        Returns:
            The Model instances, oriented in a canonical orientation, and symmetrized
        """
        if not structure_identifiers:
            return []
        # Using Pose enables simple ASU writing
        # Pose isn't the best for orient since the structure isn't oriented the symmetry wouldn't work
        # It could still happen, but we certainly shouldn't pass sym_entry to it
        self.oriented.make_path()
        orient_dir = self.oriented.location
        models_dir = os.path.dirname(orient_dir)  # This is the case because it was specified this way, but not required
        self.oriented_asu.make_path()
        self.stride.make_path()
        # orient_logger = start_log(name='orient', handler=1)
        # orient_logger = utils.start_log(name='orient', propagate=True)
        orient_logger = logging.getLogger(putils.orient)
        # For some reason this won't also emit to stream when the log level is error
        # orient_logger = \
        #     start_log(name='orient', handler=2, location=os.path.join(orient_dir, orient_log_file), propagate=True)
        if symmetry:
            sym_entry = utils.SymEntry.parse_symmetry_to_sym_entry(symmetry=symmetry)
            logger.info(f'The requested {"files" if by_file else "IDs"} are being checked for proper orientation '
                        f'with symmetry {symmetry}: {", ".join(structure_identifiers)}')
            if by_file:
                oriented_filepaths = orient_structure_files(structure_identifiers, log=orient_logger,
                                                            symmetry=symmetry, out_dir=orient_dir)
                # Pull out the structure names and use below to retrieve the oriented file
                structure_identifiers = \
                    list(map(os.path.basename, [os.path.splitext(file)[0]
                                                for file in filter(None, oriented_filepaths)]))
        else:  # Only possible if symmetry passed as None/False, in which case we should treat as asymmetric - i.e. C1
            symmetry = 'C1'
            sym_entry = utils.SymEntry.parse_symmetry_to_sym_entry(symmetry=symmetry)
            logger.info(f'The requested {"files" if by_file else "IDs"} are being set up into the DataBase: '
                        f'{", ".join(structure_identifiers)}')
            if by_file:
                structure_identifiers_ = []
                for file in structure_identifiers:
                    model = structure.model.Model.from_file(file)
                    # Loading the file will parse the file name and set .name
                    model.write(out_path=self.oriented.path_to(name=model.name))
                    # Write out each Entity in Model to ASU file for the oriented_asu database
                    model.file_path = self.oriented_asu.path_to(name=model.name)
                    model.symmetry = symmetry
                    with open(model.file_path, 'w') as f:
                        f.write(model.format_header())
                        for entity in model.entities:
                            # Write each Entity to asu
                            entity.write(file_handle=f)
                            # Save Stride results
                            entity.stride(to_file=self.stride.path_to(name=entity.name))
                    structure_identifiers_.append(model.name)

                structure_identifiers = structure_identifiers_

        # Next work through the process of orienting the selected files
        orient_names = self.oriented.retrieve_names()
        orient_asu_names = self.oriented_asu.retrieve_names()
        all_structures = []
        non_viable_structures = []
        pose_kwargs = dict(sym_entry=sym_entry, ignore_clashes=True)
        for structure_identifier in structure_identifiers:
            # First, check if the structure_identifier ASU has been processed
            # This happens when files are passed WITHOUT symmetry
            if structure_identifier in orient_asu_names:  # orient_asu file exists, stride should as well. Just load asu
                orient_asu_file = self.oriented_asu.retrieve_file(name=structure_identifier)
                pose = structure.model.Pose.from_file(orient_asu_file, name=structure_identifier, **pose_kwargs)
                # Get the full assembly and set the symmetry as it has none
                model = pose.assembly
                model.name = structure_identifier
                model.file_path = orient_asu_file  # pose.file_path
            # This happens when files are passed WITH symmetry
            elif structure_identifier in orient_names:  # orient file exists, load, save asu, and create stride
                orient_file = self.oriented.retrieve_file(name=structure_identifier)
                pose = structure.model.Pose.from_file(orient_file, name=structure_identifier, **pose_kwargs)
                # Write out the Pose ASU
                assembly_integer = '' if pose.biological_assembly is None else pose.biological_assembly
                pose.file_path = pose.write(out_path=os.path.join(self.oriented_asu.location,
                                                                  f'{structure_identifier}.pdb{assembly_integer}'))
                # pose.file_path = pose.write(out_path=self.oriented_asu.path_to(name=structure_identifier))
                # Save Stride results
                for entity in pose.entities:
                    # Ensure we have the files written for the Entity instances as well. This is a bit of a data dump
                    entity.write(oligomer=True, out_path=os.path.join(self.oriented_asu.location,
                                                                      f'{entity.name}.pdb{assembly_integer}'))
                    entity.write(out_path=os.path.join(self.oriented_asu.location,
                                                       f'{entity.name}.pdb{assembly_integer}'))
                    entity.stride(to_file=self.stride.path_to(name=entity.name))

                # Get the full assembly
                model = pose.assembly
                model.name = structure_identifier
                model.file_path = pose.file_path
            # Use entry_entity only if not processed before
            else:  # They are missing, retrieve the proper files using PDB ID's
                # entry = structure_identifier.split('_')
                # # In case entry_entity is from a new program_output directory, the entity name is probably 1ABC_1
                # if len(entry) == 2:
                #     entry, entity = entry
                #     # entry_entity = structure_identifier
                #     logger.debug(f'Fetching entry {entry}, entity {entity} from PDB')
                # else:
                #     entry = structure_identifier
                #     entity = None
                #     logger.debug(f'Fetching entry {entry} from PDB')
                #
                # asu = False
                # # if symmetry == 'C1':
                # #     # Todo modify if monomers are removed from qs_bio
                # #     assembly = query_qs_bio(entry)
                # #     # assembly = None  # 1 is the default
                # #     # asu = True <- NOT NECESSARILY A MONOMERIC FILE!
                # # else:
                # assembly = query_qs_bio(entry)
                # # Get the specified file_path for the assembly state of interest
                # file_path = fetch_pdb_file(entry, assembly=assembly, asu=asu, out_dir=models_dir)
                #
                # if not file_path:
                #     logger.warning(f"Couldn't locate the file '{file_path}', there may have been an issue "
                #                    'downloading it from the PDB. Attempting to copy from job data source...')
                #     # Todo
                #     raise NotImplementedError("This functionality hasn't been written yet. Use the canonical_pdb1/2 "
                #                               'attribute of PoseJob to pull the pdb file source.')
                # # Remove any PDB Database mirror specific naming from fetch_pdb_file such as pdb1ABC.ent
                # file_name = os.path.splitext(os.path.basename(file_path))[0].replace('pdb', '')
                # model = structure.model.Model.from_pdb(file_path, name=file_name)  # , sym_entry=sym_entry
                # if entity:  # Replace Structure from fetched file with the Entity Structure
                #     # structure_identifier will be formatted the exact same as the desired EntityID
                #     # if it was provided correctly
                #     entity = model.entity(structure_identifier)
                #     if entity:
                #         model = entity
                #     else:  # we couldn't find the specified EntityID
                #         logger.warning(f"For {structure_identifier}, couldn't locate the specified Entity '{entity}'. The "
                #                        f'available Entities are {", ".join(entity.name for entity in model.entities)}')
                #         continue

                models = self.download_structures([structure_identifier], out_dir=models_dir)
                if models:  # Not empty list. Get the first model and throw away the rest
                    model, *_ = models
                else:  # Empty list
                    non_viable_structures.append(structure_identifier)
                    continue
                try:  # Orient the Structure
                    model.orient(symmetry=symmetry)
                except (ValueError, RuntimeError) as error:
                    orient_logger.error(str(error))
                    non_viable_structures.append(structure_identifier)
                    continue

                if isinstance(model, structure.model.Entity):
                    # Write out only the entity that was extracted to the full structure_db directory
                    # Todo should we delete the source file?
                    # if symmetry == 'C1':
                    #     entity_file_path = model.write(out_path=entity_out_path)
                    # else:  # Write out the entity as parsed. Since this is an assembly, we should get the correct state
                    # model.write(oligomer=True, out_path=entity_out_path)

                    # Write out oligomer file for the orient database
                    orient_file = model.write(oligomer=True, out_path=self.oriented.path_to(name=structure_identifier))
                    # Copy the entity that was extracted to the full structure_db directory
                    entity_out_path = os.path.join(models_dir, f'{structure_identifier}.pdb')
                    shutil.copy(orient_file, entity_out_path)
                    # Write out ASU file for the oriented_asu database
                    model.file_path = model.write(out_path=self.oriented_asu.path_to(name=structure_identifier))
                    # Save Stride results
                    model.stride(to_file=self.stride.path_to(name=structure_identifier))
                else:
                    # Write out file for the orient database
                    orient_file = model.write(out_path=self.oriented.path_to(name=structure_identifier))
                    # Extract ASU from the Structure, save the file as .file_path for preprocess_structures_for_design
                    model.file_path = self.oriented_asu.path_to(name=structure_identifier)
                    with open(model.file_path, 'w') as f:
                        # model.write_header(f)
                        f.write(model.format_header())
                        for entity in model.entities:
                            # Write each Entity to asu
                            entity.write(file_handle=f)
                            # Save Stride results
                            entity.stride(to_file=self.stride.path_to(name=entity.name))

                orient_logger.info(f'Oriented: {orient_file}')

            # For every variation, set .symmetry to ensure preprocess_structures_for_design has attribute available
            model.symmetry = symmetry
            all_structures.append(model)

        if non_viable_structures:
            non_str = ', '.join(non_viable_structures)
            orient_logger.error(f'The Model'
                                f'{f"s {non_str} were" if len(non_viable_structures) > 1 else f" {non_str} was"} '
                                f'unable to be oriented properly')
        return all_structures

    def preprocess_structures_for_design(self, structures: list[structure.base.Structure],
                                         script_out_path: AnyStr = os.getcwd(), batch_commands: bool = True) -> \
            tuple[list, bool, bool]:
        """Assess whether Structure objects require any processing prior to design calculations.
        Processing includes relaxation "refine" into the energy function and/or modelling missing segments "loop model"

        Args:
            structures: An iterable of Structure objects of interest with the following attributes:
                file_path, symmetry, name, make_loop_file(), make_blueprint_file()
            script_out_path: Where should Entity processing commands be written?
            batch_commands: Whether commands should be made for batch submission
        Returns:
            Any instructions if processing is needed, then booleans for whether refinement and loop modeling has already
            occurred (True) or if files are not reported as having this modeling yet
        """
        # and loop_modelled (True)
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
        for _structure in structures:  # If _structure is here, the file should've been oriented...
            sym_def_files[_structure.symmetry] = utils.SymEntry.sdf_lookup(_structure.symmetry)
            if _structure.name not in refine_names:  # Assumes oriented_asu structure name is the same
                structures_to_refine.append(_structure)
            if _structure.name not in full_model_names:  # Assumes oriented_asu structure name is the same
                structures_to_loop_model.append(_structure)

        # For now remove this portion of the protocol
        # Todo remove if need to model. Maybe using Alphafold
        structures_to_loop_model = []

        # Query user and set up commands to perform refinement on missing entities
        if CommandDistributer.is_sbatch_available():
            shell = CommandDistributer.sbatch
        else:
            shell = CommandDistributer.default_shell

        info_messages = []
        # Assume pre_refine is True until we find it isn't
        pre_refine = True
        if structures_to_refine:  # if files found unrefined, we should proceed
            pre_refine = False
            logger.critical('The following structures are not yet refined and are being set up for refinement'
                            ' into the Rosetta ScoreFunction for optimized sequence design:\n'
                            f'{", ".join(sorted(set(_structure.name for _structure in structures_to_refine)))}')
            print(f'If you plan on performing {flags.design} using Rosetta, it is strongly encouraged that you perform '
                  f'initial refinement. You can also refine them later using the {flags.refine} module')
            print('Would you like to refine them now?')
            if boolean_choice():
                run_pre_refine = True
            else:
                print('To confirm, asymmetric units are going to be generated with input coordinates. Confirm '
                      'with "y" to ensure this is what you want')
                if boolean_choice():
                    run_pre_refine = False
                else:
                    run_pre_refine = True

            if run_pre_refine:
                # Generate sbatch refine command
                flags_file = os.path.join(refine_dir, 'refine_flags')
                # if not os.path.exists(flags_file):
                _flags = rosetta.rosetta_flags.copy() + rosetta.relax_flags
                _flags.extend([f'-out:path:pdb {refine_dir}', '-no_scorefile true'])
                _flags.remove('-output_only_asymmetric_unit true')  # want full oligomers
                variables = rosetta.rosetta_variables.copy()
                variables.append(('dist', 0))  # Todo modify if not point groups used
                _flags.append(f'-parser:script_vars {" ".join(f"{var}={val}" for var, val in variables)}')

                with open(flags_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(_flags))

                refine_cmd = [f'@{flags_file}', '-parser:protocol',
                              os.path.join(putils.rosetta_scripts_dir, f'{putils.refine}.xml')]
                refine_cmds = [rosetta.script_cmd + refine_cmd
                               + ['-in:file:s', _structure.file_path, '-parser:script_vars']
                               + [f'sdf={sym_def_files[_structure.symmetry]}',
                                  f'symmetry={"asymmetric" if _structure.symmetry == "C1" else "make_point_group"}']
                               for _structure in structures_to_refine]
                if batch_commands:
                    commands_file = \
                        utils.write_commands([subprocess.list2cmdline(cmd) for cmd in refine_cmds], out_path=refine_dir,
                                             name=f'{utils.starttime}-refine_entities')
                    refine_script = \
                        utils.CommandDistributer.distribute(commands_file, flags.refine, out_path=script_out_path,
                                                            log_file=os.path.join(refine_dir,
                                                                                  f'{putils.refine}.log'),
                                                            max_jobs=int(len(refine_cmds)/2 + .5),
                                                            number_of_commands=len(refine_cmds))
                    refine_script_message = f'Once you are satisfied, run the following to distribute refine jobs:' \
                                            f'\n\t{shell} {refine_script}'
                    info_messages.append(refine_script_message)
                else:
                    raise NotImplementedError("Currently, refinement can't be run in the shell. "
                                              'Implement this if you would like this feature')

        # Query user and set up commands to perform loop modelling on missing entities
        # Assume pre_loop_model is True until we find it isn't
        pre_loop_model = True
        if structures_to_loop_model:
            pre_loop_model = False
            logger.info('The following structures have not been modelled for disorder. Missing loops will '
                        'be built for optimized sequence design: '
                        f'{", ".join(sorted(set(_structure.name for _structure in structures_to_loop_model)))}')
            print(f'If you plan on performing {flags.design}/{flags.predict_structure} with them, it is strongly '
                  f'encouraged that you perform loop modeling to avoid disordered region clashing/misalignment')
            print('Would you like to model loops for these structures now?')
            if boolean_choice():
                run_loop_model = True
            else:
                print('To confirm, asymmetric units are going to be generated without modeling disordered loops. '
                      'Confirm with "y" to ensure this is what you want')
                if boolean_choice():
                    run_loop_model = False
                else:
                    run_loop_model = True

            if run_loop_model:
                # Generate sbatch refine command
                flags_file = os.path.join(full_model_dir, 'loop_model_flags')
                # if not os.path.exists(flags_file):
                loop_model_flags = ['-remodel::save_top 0', '-run:chain A', '-remodel:num_trajectory 1']
                #                   '-remodel:run_confirmation true', '-remodel:quick_and_dirty',
                _flags = rosetta.rosetta_flags.copy() + loop_model_flags
                # flags.extend(['-out:path:pdb %s' % full_model_dir, '-no_scorefile true'])
                _flags.extend(['-no_scorefile true', '-no_nstruct_label true'])
                # Generate 100 trial loops, 500 is typically sufficient
                variables = [('script_nstruct', '100')]
                _flags.append(f'-parser:script_vars {" ".join(f"{var}={val}" for var, val in variables)}')
                with open(flags_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(_flags))
                loop_model_cmd = [f'@{flags_file}', '-parser:protocol',
                                  os.path.join(putils.rosetta_scripts_dir, 'loop_model_ensemble.xml'),
                                  '-parser:script_vars']
                # Make all output paths and files for each loop ensemble
                # logger.info('Preparing blueprint and loop files for structure:')
                loop_model_cmds = []
                for idx, _structure in enumerate(structures_to_loop_model):
                    # Make a new directory for each structure
                    structure_out_path = os.path.join(full_model_dir, _structure.name)
                    putils.make_path(structure_out_path)
                    # Todo is renumbering required?
                    _structure.renumber_residues()
                    structure_loop_file = _structure.make_loop_file(out_path=full_model_dir)
                    if not structure_loop_file:  # No loops found, copy input file to the full model
                        copy_cmd = ['scp', self.refined.path_to(_structure.name),
                                    self.full_models.path_to(_structure.name)]
                        loop_model_cmds.append(
                            utils.write_shell_script(subprocess.list2cmdline(copy_cmd), name=_structure.name,
                                                     out_path=full_model_dir))
                        # Can't do this v as refined path doesn't exist yet
                        # shutil.copy(self.refined.path_to(_structure.name), self.full_models.path_to(_structure.name))
                        continue
                    structure_blueprint = _structure.make_blueprint_file(out_path=full_model_dir)
                    structure_cmd = rosetta.script_cmd + loop_model_cmd \
                        + [f'blueprint={structure_blueprint}', f'loop_file={structure_loop_file}',
                           '-in:file:s', self.refined.path_to(_structure.name), '-out:path:pdb', structure_out_path] \
                        + [f'sdf={sym_def_files[_structure.symmetry]}',
                           f'symmetry={"asymmetric" if _structure.symmetry == "C1" else "make_point_group"}']
                    #     + (['-symmetry:symmetry_definition', sym_def_files[_structure.symmetry]]
                    #        if _structure.symmetry != 'C1' else [])
                    # Create a multimodel from all output loop models
                    multimodel_cmd = ['python', putils.models_to_multimodel_exe, '-d', structure_loop_file,
                                      '-o', os.path.join(full_model_dir, f'{_structure.name}_ensemble.pdb')]
                    # Copy the first model from output loop models to be the full model
                    copy_cmd = ['scp', os.path.join(structure_out_path, f'{_structure.name}_0001.pdb'),
                                self.full_models.path_to(_structure.name)]
                    loop_model_cmds.append(
                        utils.write_shell_script(subprocess.list2cmdline(structure_cmd), name=_structure.name,
                                                 out_path=full_model_dir,
                                                 additional=[subprocess.list2cmdline(multimodel_cmd),
                                                             subprocess.list2cmdline(copy_cmd)]))
                if batch_commands:
                    loop_cmds_file = \
                        utils.write_commands(loop_model_cmds, name=f'{utils.starttime}-loop_model_entities',
                                             out_path=full_model_dir)
                    loop_model_script = \
                        utils.CommandDistributer.distribute(loop_cmds_file, flags.refine, out_path=script_out_path,
                                                            log_file=os.path.join(full_model_dir, 'loop_model.log'),
                                                            max_jobs=int(len(loop_model_cmds)/2 + .5),
                                                            number_of_commands=len(loop_model_cmds))
                    multi_script_warning = "\n***Run this script AFTER completion of the refinement script***\n" \
                        if info_messages else ""
                    loop_model_sbatch_message = 'Once you are satisfied, run the following to distribute loop_modeling'\
                                                f' jobs:{multi_script_warning}\n\t{shell} {loop_model_script}'
                    info_messages.append(loop_model_sbatch_message)
                else:
                    raise NotImplementedError("Currently, loop modeling can't be run in the shell. "
                                              'Implement this if you would like this feature')

        return info_messages, pre_refine, pre_loop_model


class StructureDatabaseFactory:
    """Return a StructureDatabase instance by calling the Factory instance with the StructureDatabase source name

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the named StructureDatabase
    """

    def __init__(self, **kwargs):
        # self._databases = {}
        self._database = None

    def __call__(self, source: str = os.path.join(os.getcwd(),
                                                  f'{putils.program_name}{putils.data.title()}',
                                                  putils.structure_info),
                 sql: bool = False, **kwargs) -> StructureDatabase:
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
            # source = os.path.join(source, putils.structure_info)
            pdbs = os.path.join(source, 'PDBs')  # Used to store downloaded PDB's
            # stride directory
            stride_dir = os.path.join(source, 'stride')
            # Todo only make paths if they are needed...
            putils.make_path(stride_dir)
            # pdbs subdirectories
            orient_dir = os.path.join(pdbs, 'oriented')
            orient_asu_dir = os.path.join(pdbs, 'oriented_asu')
            refine_dir = os.path.join(pdbs, 'refined')
            full_model_dir = os.path.join(pdbs, 'full_models')
            putils.make_path(orient_dir)
            putils.make_path(orient_asu_dir)
            putils.make_path(refine_dir)
            putils.make_path(full_model_dir)
            logger.info(f'Initializing {StructureDatabase.__name__}({source})')

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
