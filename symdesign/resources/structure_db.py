from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from glob import glob
from itertools import count
from pathlib import Path
from typing import Annotated, AnyStr

import jax.numpy as jnp

from . import distribute, sql
from .database import Database, DataStore
from symdesign import flags, resources, structure, utils
from symdesign.protocols.pose import load_evolutionary_profile
from symdesign.sequence import generate_mutations, expression
from symdesign.structure.model import ContainsEntities, Entity, Pose, Structure
from symdesign.utils.SymEntry import SymEntry
from symdesign.utils.symmetry import CRYST
putils = utils.path

# Todo adjust the logging level for this module?
logger = logging.getLogger(__name__)
# rcsb_download_url = 'https://files.wwpdb.org/pub/pdb/data/'  # biounit/PDB or assemblies/mmcif
# Could add .gz to all downloads to speed transfer and load. Change the load options to use .gz version?
rcsb_download_url = 'https://files.rcsb.org/download/'


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
            pdb_file = f'{clean_pdb}.pdb'
        else:
            pdb_file = f'{clean_pdb}.pdb{assembly}'

        file_name = os.path.join(out_dir, pdb_file)
        current_file = sorted(glob(file_name))
        # logger.debug('Found the files: {', '.join(current_file)}')
        if not current_file:
            # The desired file is missing and should be retrieved
            # Always returns files in lowercase
            cmd = ['wget', '-q', '-O', file_name, f'{rcsb_download_url}{pdb_file}']
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p.communicate()
            if p.returncode != 0:
                failed_download = Path(file_name)
                failed_download.unlink(missing_ok=False)

                cif_file = f'{clean_pdb}-assembly{assembly}.cif'
                # download_cif_file = cif_file
                # Todo debug reinstate v
                download_cif_file = f'{clean_pdb}.cif{assembly}'
                file_name = os.path.join(out_dir, download_cif_file)
                cmd = ['wget', '-q', '-O', file_name, f'{rcsb_download_url}{cif_file}']
                # logger.debug(f'Pulling the .cif file with: {subprocess.list2cmdline(cmd)}')
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p.communicate()
                if p.returncode != 0:
                    continue

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
        # ^ Cassini format
        def get_pdb(*args, **_kwargs): return sorted(glob(file_path))
        # v KM local pdb and escher PDB mirror
        # file_path = os.path.join(location, pdb_code[1:3], f'{pdb_code.lower()}.pdb')
        logger.debug(f'Searching for PDB file at "{file_path}"')
    else:
        get_pdb = _fetch_pdb_from_api

    # The matching file is the first (should only be one)
    pdb_file: list[str] = get_pdb(pdb_code, asu=asu, location=location, **kwargs)
    if not pdb_file:  # Empty list
        logger.warning(f'No matching file found for PDB: {pdb_code}')
        return None
    else:  # Should only find one file, therefore, return the first
        # if len(pdb_file) != 1:
        #     logger.info(f'Found multiple file for EntryID {pdb_code}. Choosing the first: '
        #                 f'{", ".join(pdb_file)}')
        return pdb_file[0]


def download_structures(
    structure_identifiers: Iterable[str], out_dir: str = os.getcwd(), asu: bool = False
) -> list[Structure]:
    """Given EntryIDs/EntityIDs, retrieve/save .pdb files, then return a ContainsEntities for each identifier

    Defaults to fetching the biological assembly file, prioritizing the assemblies as predicted very high/high from
    QSBio, then using the first assembly if QSBio is missing

    Args:
        structure_identifiers: The names of all entity_ids requiring orientation
        out_dir: The directory to write downloaded files to
        asu: Whether to get the asymmetric unit from the PDB instead of the biological assembly
    Returns:
        The requested Pose/Entity instances
    """
    all_structures = []
    for structure_identifier in structure_identifiers:
        # Retrieve the proper files using PDB ID's
        structure_identifier_ = structure_identifier
        assembly_integer = entity_integer = None
        splitter_iter = iter('_-')  # (entity, assembly))
        idx = count(-1)
        extra = None
        while len(structure_identifier_) != 4:
            try:  # To parse the name using standard PDB API entry ID's
                structure_identifier_, *extra = structure_identifier_.split(next(splitter_iter))
            except StopIteration:
                # We didn't find an EntryID in structure_identifier_ from splitting typical PDB formatted strings
                logger.debug(f"The name '{structure_identifier}' can't be coerced to PDB API format")
                continue
            else:
                next(idx)
        # Set the index to the index that was stopped at
        idx = next(idx)

        if extra:  # Extra not None or []
            # Todo, use of elif means can't have 1ABC_1.pdb2
            # Try to parse any found extra to an integer denoting entity or assembly ID
            integer, *non_sense = extra
            if integer.isdigit() and not non_sense:
                integer = int(integer)
                entry = structure_identifier_
                if idx == 0:  # Entity integer, such as 1ABC_1.pdb
                    entity_integer = integer
                    # structure_identifier_ = f'{structure_identifier_}_{integer}'
                    logger.debug(f'Fetching EntityID {entry}_{entity_integer} from PDB')
                else:  # This is an assembly or unknown conjugation
                    if idx == 1:  # This is an assembly integer, such as 1ABC-1.pdb
                        entry = structure_identifier_
                        assembly_integer = integer
                        logger.debug(f'Fetching AssemblyID {entry}-{assembly_integer} from PDB')
                    else:
                        logger.critical("This logic happened and wasn't expected")
            else:  # This isn't an integer or there are extra characters
                logger.info(f"The name '{structure_identifier}' can't be coerced to PDB format")
                continue
        elif extra is None:  # Nothing extra as it was correct length to begin with, just query entry
            entry = structure_identifier_
        else:
            raise RuntimeError(
                f"This logic wasn't expected and shouldn't be allowed to persist: "
                f'structure_identifier={structure_identifier}, structure_identifier_={structure_identifier_}, '
                f'extra={extra}, idx={idx}')

        if assembly_integer is None:
            assembly_integer = query_qs_bio(entry)
        # Get the specified file_path for the assembly state of interest
        file_path = fetch_pdb_file(entry, assembly=assembly_integer, asu=asu, out_dir=out_dir)

        if file_path is None:
            logger.error(f"Couldn't locate the identifier '{structure_identifier}'. There may have been an issue "
                         'retrieving it from the PDB')
            continue

        # Remove any PDB Database mirror specific naming from fetch_pdb_file such as pdb1ABC.ent
        file_name = os.path.splitext(os.path.basename(file_path))[0].replace('pdb', '')
        pose = Pose.from_file(file_path, name=file_name)
        if entity_integer is not None:
            # Replace the Pose from fetched file with the Entity
            # structure_identifier is formatted the exact same as the desired EntityID
            entity = pose.entity(structure_identifier)
            if entity:
                # Set the file_path attribute on the Entity
                file_path = pose.file_path
                pose = entity
                pose.file_path = file_path
            else:  # Couldn't find the specified EntityID
                logger.warning(f"For {structure_identifier}, couldn't locate the specified {Entity.__name__} "
                               f"'{structure_identifier}'. The available {Entity.__name__} instances are "
                               f'{", ".join(entity.name for entity in pose.entities)}')
                continue

        all_structures.append(pose)

    return all_structures


def query_qs_bio(entry_id: str) -> int:
    """Retrieve the first matching Very High/High confidence QSBio assembly from a PDB EntryID

    Args:
        entry_id: The 4 character PDB EntryID (code) to query
    Returns:
        The integer of the corresponding PDB Assembly ID according to QSBio
    """
    biological_assemblies = resources.query.pdb.qsbio_confirmed.get(entry_id.lower())
    if biological_assemblies:
        # Get the first assembly in matching oligomers
        if len(biological_assemblies) != 1:
            logger.info(f'Found multiple biological assemblies for EntryID {entry_id}. Choosing the first: '
                        f'{", ".join(map(str, biological_assemblies))}')
        assembly = biological_assemblies[0]
    else:
        assembly = 1
        logger.warning(f'No QSBio confirmed biological assembly for EntryID {entry_id}. Using the default assembly 1')
    return assembly


class StructureDatabase(Database):
    def __init__(self, models: AnyStr | Path = None, full_models: AnyStr | Path = None, oriented: AnyStr | Path = None,
                 oriented_asu: AnyStr | Path = None, refined: AnyStr | Path = None, stride: AnyStr | Path = None,
                 **kwargs):
        # passed to Database
        # sql: sqlite = None, log: Logger = logger
        super().__init__(**kwargs)  # Database

        self.models = DataStore(location=models, extension='.pdb', glob_extension='.pdb*',
                                sql=self.sql, log=self.log, load_file=Pose.from_pdb)
        # Old version when loop model came from an ensemble
        # self.full_models = DataStore(location=full_models, extension='_ensemble.pdb', glob_extension='_ensemble.pdb*',
        self.full_models = DataStore(location=full_models, extension='.pdb', glob_extension='.pdb*',
                                     sql=self.sql, log=self.log, load_file=Pose.from_pdb)
        self.oriented = DataStore(location=oriented, extension='.pdb', glob_extension='.pdb*',
                                  sql=self.sql, log=self.log, load_file=Pose.from_pdb)
        self.oriented_asu = DataStore(location=oriented_asu, extension='.pdb', glob_extension='.pdb*',
                                      sql=self.sql, log=self.log, load_file=Pose.from_pdb)
        self.refined = DataStore(location=refined, extension='.pdb', glob_extension='.pdb*',
                                 sql=self.sql, log=self.log, load_file=Pose.from_pdb)
        self.stride = DataStore(location=stride, extension='.stride', sql=self.sql, log=self.log,
                                load_file=structure.utils.parse_stride)

        self.sources = [self.oriented_asu, self.refined, self.stride]  # self.full_models

    def orient_structures(self, structure_identifiers: Iterable[str], sym_entry: SymEntry = None,
                          by_file: bool = False) \
            -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, ...], list[sql.ProteinMetadata]]]:
        """Given structure identifiers and their corresponding symmetry, retrieve, orient, and save oriented files to
        the Database, then return metadata for each

        Args:
            structure_identifiers: The names of all entity_ids requiring orientation
            sym_entry: The SymEntry used to treat each passed Entity as symmetric. Default assumes no symmetry
            by_file: Whether to parse the structure_identifiers as file paths. Default treats as PDB EntryID/EntityID
        Returns:
            The tuple consisting of (
                A map of the entire Pose name to each contained Entity name,
                A mapping of the UniprotID's to their ProteinMetadata instance for every Entity loaded
            )
        """
        if not structure_identifiers:
            return {}, {}

        self.oriented.make_path()
        self.oriented_asu.make_path()
        models_dir = self.models.location

        if isinstance(sym_entry, utils.SymEntry.SymEntry):
            if sym_entry.number:
                resulting_symmetry = sym_entry.resulting_symmetry
                if resulting_symmetry in utils.symmetry.space_group_cryst1_fmt_dict:
                    # This is a crystalline symmetry, so use a TOKEN to specify use of the CRYST record
                    resulting_symmetry = CRYST
                else:
                    logger.info(f'The requested {"files" if by_file else "IDs"} are being checked for proper '
                                f'orientation with symmetry {resulting_symmetry}: {", ".join(structure_identifiers)}')
            else:  # This is entry_number 0, which is a TOKEN to use the CRYST record
                resulting_symmetry = CRYST
        else:  # Treat as asymmetric - i.e. C1
            if sym_entry:
                logger.warning(f"The passed 'sym_entry' isn't of the required type {utils.SymEntry.SymEntry.__name__}. "
                               "Treating as asymmetric")
            sym_entry = None  # Ensure not something else
            resulting_symmetry = 'C1'
            logger.info(f'The requested {"files" if by_file else "IDs"} are being set up into the DataBase: '
                        f'{", ".join(structure_identifiers)}')

        orient_logger = logging.getLogger(putils.orient)
        structure_identifier_tuples: dict[str, tuple[str, ...]] = {}
        uniprot_id_to_protein_metadata: dict[tuple[str, ...], list[sql.ProteinMetadata]] = defaultdict(list)
        non_viable_structures = []

        def create_protein_metadata(model: ContainsEntities):
            """From a ContainsEntities instance, extract the unique metadata to identify the entities involved

            Args:
                model: The Entity instances to initialize to ProteinMetadata
            """
            for entity in model.entities:
                protein_metadata = sql.ProteinMetadata(
                    entity_id=entity.name,
                    reference_sequence=entity.reference_sequence,
                    thermophilicity=entity.thermophilicity,
                    symmetry_group=entity.symmetry,
                    model_source=entity.file_path
                )
                entity.calculate_secondary_structure(to_file=self.stride.path_to(name=entity.name))
                protein_metadata.n_terminal_helix = entity.is_termini_helical()
                protein_metadata.c_terminal_helix = entity.is_termini_helical('c')

                try:
                    ''.join(entity.uniprot_ids)
                except TypeError:  # Uniprot_ids is (None,)
                    entity.uniprot_ids = (entity.name,)
                except AttributeError:  # Unable to retrieve .uniprot_ids
                    entity.uniprot_ids = (entity.name,)
                # else:  # .uniprot_ids work. Use as parsed
                uniprot_ids = entity.uniprot_ids

                uniprot_id_to_protein_metadata[uniprot_ids].append(protein_metadata)

            if resulting_symmetry == CRYST:
                structure_identifier_tuples[model.name] = tuple()
            else:
                structure_identifier_tuples[model.name] = tuple(entity.name for entity in model.entities)

        def report_non_viable_structures():
            if non_viable_structures:
                if len(non_viable_structures) > 1:
                    non_str = ', '.join(non_viable_structures[:-1]) + f' and {non_viable_structures[-1]}'
                    plural_str = f"s {non_str} weren't"
                else:
                    plural_str = f" {non_viable_structures} wasn't"
                orient_logger.error(
                    f'The structure{plural_str} able to be oriented properly')

        def write_entities_and_asu(model: ContainsEntities, assembly_integer: str):
            """Write the overall ASU, each Entity as an ASU and oligomer, and set the model.file_path attribute

            Args:
                model: The ContainsEntities instance being oriented
                assembly_integer: The integer representing the assembly number (provided from ".pdb1" type extensions)
            """
            # Save .file_path attribute
            model.file_path = os.path.join(self.oriented_asu.location, f'{model.name}.pdb{assembly_integer}')
            with open(model.file_path, 'w') as f:
                f.write(model.format_header())
                # Write out each Entity in model to form the ASU
                for entity in model.entities:
                    # Write each Entity to combined asu
                    entity.write(file_handle=f)
                    # Write each Entity to own file
                    oligomer_path = os.path.join(self.oriented.location, f'{entity.name}.pdb{assembly_integer}')
                    entity.write(assembly=True, out_path=oligomer_path)
                    # And asu
                    asu_path = os.path.join(self.oriented_asu.location, f'{entity.name}.pdb{assembly_integer}')
                    # Set the Entity.file_path for ProteinMetadata
                    entity.file_path = entity.write(out_path=asu_path)

        def _orient_existing_files(files: Iterable[str], resulting_symmetry: str, sym_entry: SymEntry = None) -> None:
            """Return the structure identifier for a file that is loaded and oriented

            Args:
                files: The files to orient in the canonical symmetry
                resulting_symmetry: The symmetry to use during orient
                sym_entry: The symmetry to use during orient protocol
            Returns:
                None
            """
            for file in files:
                # Load entities to solve multi-component orient problem
                pose = Pose.from_file(file)
                if resulting_symmetry == CRYST:
                    pose.set_symmetry(sym_entry=sym_entry)
                    pose.file_path = pose.write(out_path=self.models.path_to(name=pose.name))
                    # Set each Entity.file_path
                    for entity in pose.entities:
                        entity.file_path = entity.write(out_path=self.models.path_to(name=entity.name))
                else:
                    try:
                        pose.orient(symmetry=resulting_symmetry)
                    except (ValueError, RuntimeError, structure.utils.SymmetryError) as error:
                        orient_logger.error(str(error))
                        non_viable_structures.append(file)
                        continue
                    pose.set_symmetry(sym_entry=sym_entry)
                    assembly_integer = '' if pose.biological_assembly is None else pose.biological_assembly
                    orient_file = os.path.join(self.oriented.location, f'{pose.name}.pdb{assembly_integer}')
                    pose.write(out_path=orient_file)
                    orient_logger.info(f'Oriented: {orient_file}')  # <- This isn't ASU
                    write_entities_and_asu(pose, assembly_integer)

                create_protein_metadata(pose)

        if by_file:
            _orient_existing_files(structure_identifiers, resulting_symmetry, sym_entry)
        else:  # Orienting the selected files and save
            # First, check if using crystalline symmetry and prevent loading of existing files
            if resulting_symmetry == CRYST:
                orient_asu_names = orient_names = model_names = []
            else:
                orient_names = self.oriented.retrieve_names()
                orient_asu_names = self.oriented_asu.retrieve_names()
                model_names = self.models.retrieve_names()
            # Using Pose simplifies ASU writing, however if the Pose isn't oriented correctly SymEntry won't work
            # Todo
            #  Should clashes be warned?
            # , ignore_clashes=True)
            pose_kwargs = dict(sym_entry=sym_entry)
            for structure_identifier in structure_identifiers:
                # First, check if the structure_identifier ASU has been processed
                if structure_identifier in orient_asu_names:  # orient_asu file exists, just load
                    orient_asu_file = self.oriented_asu.retrieve_file(name=structure_identifier)
                    pose = Pose.from_file(orient_asu_file, name=structure_identifier, **pose_kwargs)
                    if pose.symmetric_assembly_is_clash(measure=self.job.design.clash_criteria,
                                                        distance=self.job.design.clash_distance, warn=True):
                        if not self.job.design.ignore_symmetric_clashes:
                            logger.critical(f"The structure '{structure_identifier}' isn't a viable symmetric assembly "
                                            f"in the symmetry {resulting_symmetry}. Couldn't initialize")
                            continue

                    # Write each Entity as well
                    for entity in pose.entities:
                        entity_asu_file = self.oriented_asu.retrieve_file(name=entity.name)
                        if entity_asu_file is None:
                            entity.write(out_path=self.oriented_asu.path_to(name=entity.name))
                        # Set the Entity.file_path for ProteinMetadata
                        entity.file_path = entity_asu_file
                elif structure_identifier in orient_names:  # ASU files don't exist. Load oriented and save asu
                    orient_file = self.oriented.retrieve_file(name=structure_identifier)
                    # These name=structure_identifier should be the default parsing method anyway...
                    pose = Pose.from_file(orient_file, name=structure_identifier, **pose_kwargs)

                    if pose.symmetric_assembly_is_clash(measure=self.job.design.clash_criteria,
                                                        distance=self.job.design.clash_distance, warn=True):
                        if not self.job.design.ignore_symmetric_clashes:
                            logger.critical(f"The structure '{structure_identifier}' isn't a viable symmetric assembly "
                                            f"in the symmetry {resulting_symmetry}. Couldn't initialize")
                            continue

                    # Write out the Pose ASU
                    assembly_integer = '' if pose.biological_assembly is None else pose.biological_assembly
                    write_entities_and_asu(pose, assembly_integer)
                else:  # orient is missing, retrieve the proper files using PDB ID's
                    if structure_identifier in model_names:
                        model_file = self.models.retrieve_file(name=structure_identifier)
                        pose = Pose.from_file(model_file, name=structure_identifier)
                    else:
                        pose_models = download_structures([structure_identifier], out_dir=models_dir)
                        if pose_models:
                            # Get the first model and throw away the rest
                            pose, *_ = pose_models
                        else:  # Empty list
                            non_viable_structures.append(structure_identifier)
                            continue

                    if resulting_symmetry == CRYST:
                        pose.set_symmetry(sym_entry=sym_entry)
                        # pose.file_path is already set
                        # Set each Entity.file_path
                        for entity in pose.entities:
                            entity.file_path = entity.write(out_path=self.models.path_to(name=entity.name))
                    else:
                        try:  # Orient the Pose
                            pose.orient(symmetry=resulting_symmetry)
                        except (ValueError, RuntimeError, structure.utils.SymmetryError) as error:
                            orient_logger.error(str(error))
                            non_viable_structures.append(structure_identifier)
                            continue

                        pose.set_symmetry(sym_entry=sym_entry)
                        assembly_integer = '' if pose.biological_assembly is None else pose.biological_assembly
                        # Write out files for the orient database
                        base_file_name = f'{structure_identifier}.pdb{assembly_integer}'
                        orient_file = os.path.join(self.oriented.location, base_file_name)

                        if isinstance(pose, Entity):
                            # The symmetry attribute should be set from parsing, so assembly=True will work
                            # and create_protein_metadata has access to .symmetry
                            pose.write(assembly=True, out_path=orient_file)
                            # Write out ASU file
                            asu_path = os.path.join(self.oriented_asu.location, base_file_name)
                            # Set the Entity.file_path for ProteinMetadata
                            pose.file_path = pose.write(out_path=asu_path)
                        else:
                            pose.write(out_path=orient_file)
                            write_entities_and_asu(pose, assembly_integer)

                        orient_logger.info(f'Oriented: {orient_file}')

                create_protein_metadata(pose)

        report_non_viable_structures()
        return structure_identifier_tuples, uniprot_id_to_protein_metadata

    # def preprocess_structures_for_design(self, structures: list[structure.base.ContainsResidues],
    #                                      script_out_path: AnyStr = os.getcwd(), batch_commands: bool = True) -> \
    #         tuple[list, bool, bool]:
    #     """Assesses whether StructureBase objects require any processing prior to design calculations.
    #     Processing includes relaxation "refine" into the energy function and/or modeling missing segments "loop model"
    #
    #     Args:
    #         structures: An iterable of StructureBase objects of interest with the following attributes:
    #             file_path, symmetry, name, make_loop_file(), make_blueprint_file()
    #         script_out_path: Where should Entity processing commands be written?
    #         batch_commands: Whether commands should be made for batch submission
    #     Returns:
    #         Any instructions if processing is needed, then booleans for whether refinement and loop modeling has already
    #         occurred (True) or if files are not reported as having this modeling yet
    #     """
    #     # and loop_modeled (True)
    #     self.refined.make_path()
    #     refine_names = self.refined.retrieve_names()
    #     refine_dir = self.refined.location
    #     self.full_models.make_path()
    #     full_model_names = self.full_models.retrieve_names()
    #     full_model_dir = self.full_models.location
    #     # Identify the entities to refine and to model loops before proceeding
    #     structures_to_refine, structures_to_loop_model, sym_def_files = [], [], {}
    #     for _structure in structures:  # If _structure is here, the file should've been oriented...
    #         sym_def_files[_structure.symmetry] = utils.SymEntry.sdf_lookup(_structure.symmetry)
    #         if _structure.name not in refine_names:  # Assumes oriented_asu structure name is the same
    #             structures_to_refine.append(_structure)
    #         if _structure.name not in full_model_names:  # Assumes oriented_asu structure name is the same
    #             structures_to_loop_model.append(_structure)
    #
    #     # For now remove this portion of the protocol
    #     # Todo remove if need to model. Maybe using Alphafold
    #     structures_to_loop_model = []
    #
    #     # Query user and set up commands to perform refinement on missing entities
    #     if distribute.is_sbatch_available():
    #         shell = distribute.sbatch
    #     else:
    #         shell = distribute.default_shell
    #
    #     info_messages = []
    #     # Assume pre_refine is True until we find it isn't
    #     pre_refine = True
    #     if structures_to_refine:  # if files found unrefined, we should proceed
    #         pre_refine = False
    #         logger.critical('The following structures are not yet refined and are being set up for refinement'
    #                         ' into the Rosetta ScoreFunction for optimized sequence design:\n'
    #                         f'{", ".join(sorted(set(_structure.name for _structure in structures_to_refine)))}')
    #         print(f'If you plan on performing {flags.design} using Rosetta, it is strongly encouraged that you perform '
    #               f'initial refinement. You can also refine them later using the {flags.refine} module')
    #         print('Would you like to refine them now?')
    #         if boolean_choice():
    #             refine_input = True
    #         else:
    #             print('To confirm, asymmetric units are going to be generated with input coordinates. Confirm '
    #                   'with "y" to ensure this is what you want')
    #             if boolean_choice():
    #                 refine_input = False
    #             else:
    #                 refine_input = True
    #
    #         if refine_input:
    #             # Generate sbatch refine command
    #             flags_file = os.path.join(refine_dir, 'refine_flags')
    #             # if not os.path.exists(flags_file):
    #             _flags = rosetta.rosetta_flags.copy() + rosetta.relax_flags
    #             _flags.extend([f'-out:path:pdb {refine_dir}', '-no_scorefile true'])
    #             _flags.remove('-output_only_asymmetric_unit true')  # want full oligomers
    #             variables = rosetta.rosetta_variables.copy()
    #             variables.append(('dist', 0))  # Todo modify if not point groups used
    #             _flags.append(f'-parser:script_vars {" ".join(f"{var}={val}" for var, val in variables)}')
    #
    #             with open(flags_file, 'w') as f:
    #                 f.write('%s\n' % '\n'.join(_flags))
    #
    #             refine_cmd = [f'@{flags_file}', '-parser:protocol',
    #                           os.path.join(putils.rosetta_scripts_dir, f'{putils.refine}.xml')]
    #             refine_cmds = [rosetta.script_cmd + refine_cmd
    #                            + ['-in:file:s', _structure.file_path, '-parser:script_vars']
    #                            + [f'sdf={sym_def_files[_structure.symmetry]}',
    #                               f'symmetry={"asymmetric" if _structure.symmetry == "C1" else "make_point_group"}']
    #                            for _structure in structures_to_refine]
    #             if batch_commands:
    #                 commands_file = \
    #                     resources.distribute.write_commands([subprocess.list2cmdline(cmd) for cmd in refine_cmds], out_path=refine_dir,
    #                                                         name=f'{utils.starttime}-refine_entities')
    #                 refine_script = \
    #                     distribute.distribute(commands_file, flags.refine, out_path=script_out_path,
    #                                           log_file=os.path.join(refine_dir, f'{putils.refine}.log'),
    #                                           max_jobs=int(len(refine_cmds)/2 + .5),
    #                                           number_of_commands=len(refine_cmds))
    #                 refine_script_message = f'Once you are satisfied, run the following to distribute refine jobs:' \
    #                                         f'\n\t{shell} {refine_script}'
    #                 info_messages.append(refine_script_message)
    #             else:
    #                 raise NotImplementedError("Currently, refinement can't be run in the shell. "
    #                                           'Implement this if you would like this feature')
    #
    #     # Query user and set up commands to perform loop modeling on missing entities
    #     # Assume pre_loop_model is True until we find it isn't
    #     pre_loop_model = True
    #     if structures_to_loop_model:
    #         pre_loop_model = False
    #         logger.info("The following structures haven't been modeled for disorder: "
    #                     f'{", ".join(sorted(set(_structure.name for _structure in structures_to_loop_model)))}')
    #         print(f'If you plan on performing {flags.design}/{flags.predict_structure} with them, it is strongly '
    #               f'encouraged that you build missing loops to avoid disordered region clashing/misalignment')
    #         print('Would you like to model loops for these structures now?')
    #         if boolean_choice():
    #             loop_model_input = True
    #         else:
    #             print('To confirm, asymmetric units are going to be generated without modeling disordered loops. '
    #                   'Confirm with "y" to ensure this is what you want')
    #             if boolean_choice():
    #                 loop_model_input = False
    #             else:
    #                 loop_model_input = True
    #
    #         if loop_model_input:
    #             # Generate sbatch refine command
    #             flags_file = os.path.join(full_model_dir, 'loop_model_flags')
    #             # if not os.path.exists(flags_file):
    #             loop_model_flags = ['-remodel::save_top 0', '-run:chain A', '-remodel:num_trajectory 1']
    #             #                   '-remodel:run_confirmation true', '-remodel:quick_and_dirty',
    #             _flags = rosetta.rosetta_flags.copy() + loop_model_flags
    #             # flags.extend(['-out:path:pdb %s' % full_model_dir, '-no_scorefile true'])
    #             _flags.extend(['-no_scorefile true', '-no_nstruct_label true'])
    #             # Generate 100 trial loops, 500 is typically sufficient
    #             variables = [('script_nstruct', '100')]
    #             _flags.append(f'-parser:script_vars {" ".join(f"{var}={val}" for var, val in variables)}')
    #             with open(flags_file, 'w') as f:
    #                 f.write('%s\n' % '\n'.join(_flags))
    #             loop_model_cmd = [f'@{flags_file}', '-parser:protocol',
    #                               os.path.join(putils.rosetta_scripts_dir, 'loop_model_ensemble.xml'),
    #                               '-parser:script_vars']
    #             # Make all output paths and files for each loop ensemble
    #             # logger.info('Preparing blueprint and loop files for structure:')
    #             loop_model_cmds = []
    #             for idx, _structure in enumerate(structures_to_loop_model):
    #                 # Make a new directory for each structure
    #                 structure_out_path = os.path.join(full_model_dir, _structure.name)
    #                 putils.make_path(structure_out_path)
    #                 # Todo is renumbering required?
    #                 _structure.renumber_residues()
    #                 structure_loop_file = _structure.make_loop_file(out_path=full_model_dir)
    #                 if not structure_loop_file:  # No loops found, copy input file to the full model
    #                     copy_cmd = ['scp', self.refined.path_to(_structure.name),
    #                                 self.full_models.path_to(_structure.name)]
    #                     loop_model_cmds.append(
    #                         resources.distribute.write_script(subprocess.list2cmdline(copy_cmd), name=_structure.name,
    #                                                           out_path=full_model_dir))
    #                     # Can't do this v as refined path doesn't exist yet
    #                     # shutil.copy(self.refined.path_to(_structure.name), self.full_models.path_to(_structure.name))
    #                     continue
    #                 structure_blueprint = _structure.make_blueprint_file(out_path=full_model_dir)
    #                 structure_cmd = rosetta.script_cmd + loop_model_cmd \
    #                     + [f'blueprint={structure_blueprint}', f'loop_file={structure_loop_file}',
    #                        '-in:file:s', self.refined.path_to(_structure.name), '-out:path:pdb', structure_out_path] \
    #                     + [f'sdf={sym_def_files[_structure.symmetry]}',
    #                        f'symmetry={"asymmetric" if _structure.symmetry == "C1" else "make_point_group"}']
    #                 #     + (['-symmetry:symmetry_definition', sym_def_files[_structure.symmetry]]
    #                 #        if _structure.symmetry != 'C1' else [])
    #                 # Create a multimodel from all output loop models
    #                 multimodel_cmd = ['python', putils.models_to_multimodel_exe, '-d', structure_loop_file,
    #                                   '-o', os.path.join(full_model_dir, f'{_structure.name}_ensemble.pdb')]
    #                 # Copy the first model from output loop models to be the full model
    #                 copy_cmd = ['scp', os.path.join(structure_out_path, f'{_structure.name}_0001.pdb'),
    #                             self.full_models.path_to(_structure.name)]
    #                 loop_model_cmds.append(
    #                     resources.distribute.write_script(
    #                         subprocess.list2cmdline(structure_cmd), name=_structure.name, out_path=full_model_dir,
    #                         additional=[subprocess.list2cmdline(multimodel_cmd), subprocess.list2cmdline(copy_cmd)]))
    #             if batch_commands:
    #                 loop_cmds_file = \
    #                     resources.distribute.write_commands(loop_model_cmds, out_path=full_model_dir,
    #                                                         name=f'{utils.starttime}-loop_model_entities')
    #                 loop_model_script = \
    #                     distribute.distribute(loop_cmds_file, flags.refine, out_path=script_out_path,
    #                                           log_file=os.path.join(full_model_dir, 'loop_model.log'),
    #                                           max_jobs=int(len(loop_model_cmds)/2 + .5),
    #                                           number_of_commands=len(loop_model_cmds))
    #                 multi_script_warning = "\n***Run this script AFTER completion of the refinement script***\n" \
    #                     if info_messages else ""
    #                 loop_model_sbatch_message = 'Once you are satisfied, run the following to distribute loop_modeling'\
    #                                             f' jobs:{multi_script_warning}\n\t{shell} {loop_model_script}'
    #                 info_messages.append(loop_model_sbatch_message)
    #             else:
    #                 raise NotImplementedError("Currently, loop modeling can't be run in the shell. "
    #                                           'Implement this if you would like this feature')
    #
    #     return info_messages, pre_refine, pre_loop_model

    def preprocess_metadata_for_design(self, metadata: list[sql.ProteinMetadata], script_out_path: AnyStr = os.getcwd(),
                                       batch_commands: bool = False) -> list[str] | list:
        """Assess whether structural data requires any processing prior to design calculations.
        Processing includes relaxation "refine" into the energy function and/or modeling missing segments "loop model"

        Args:
            metadata: An iterable of ProteinMetadata objects of interest with the following attributes:
                model_source, symmetry_group, and entity_id
            script_out_path: Where should Entity processing commands be written?
            batch_commands: Whether commands should be made for batch submission
        Returns:
            Any instructions if processing is needed, otherwise an empty list
        """
        if batch_commands:
            putils.make_path(script_out_path)
            if distribute.is_sbatch_available():
                shell = distribute.sbatch
            else:
                shell = distribute.default_shell

        api_db = self.job.api_db  # resources.wrapapi.api_database_factory()
        self.full_models.make_path()
        self.refined.make_path()
        full_model_names = self.full_models.retrieve_names()
        full_model_dir = self.full_models.location
        # Identify the entities to refine and to model loops before proceeding
        protein_data_to_loop_model = []
        for data in metadata:
            if not data.model_source:
                logger.debug(f"{self.preprocess_metadata_for_design.__name__}: Couldn't find the "
                             f"ProteinMetadata.model_source for {data.entity_id}. Skipping loop model preprocessing")
                continue
            # If data is here, it's model_source file should've been oriented...
            if data.entity_id not in full_model_names:  # Assumes oriented_asu structure name is the same
                protein_data_to_loop_model.append(data)

        info_messages = []
        if protein_data_to_loop_model:
            logger.info("The following structures haven't been modeled for disorder: "
                        f'{", ".join(sorted(set(protein.entity_id for protein in protein_data_to_loop_model)))}')
            # Files found unloop_modeled, check to see if work should be done
            if self.job.init.loop_model_input:  # is not None:
            #     loop_model_input = self.job.init.loop_model_input
            # else:  # Query user and set up commands to perform loop modeling on missing entities
            #     print(f'If you plan on performing {flags.design}/{flags.predict_structure} with them, it is strongly '
            #           f'encouraged that you build missing loops to avoid disordered region clashing/misalignment')
            #     print('Would you like to model loops for these structures now?')
            #     if boolean_choice():
            #         loop_model_input = True
            #     else:
            #         print('To confirm, asymmetric units are going to be generated without modeling disordered loops. '
            #               'Confirm with "y" to ensure this is what you want')
            #         if boolean_choice():
            #             loop_model_input = False
            #         else:
            #             loop_model_input = True
            #
            # if loop_model_input:
                # Generate loop model commands
                use_alphafold = True
                if use_alphafold and self.job.gpu_available:
                    if batch_commands:
                        # Write all commands to a file to perform in batches
                        cmd = [*putils.program_command_tuple, flags.initialize_building_blocks,
                               f'--{flags.loop_model_input}', flags.pdb_codes_args[-1]]
                        commands = [cmd + [protein.entity_id, f'--symmetry', protein.symmetry_group]
                                    for idx, protein in enumerate(protein_data_to_loop_model)]

                        loop_cmds_file = resources.distribute.write_commands(
                            [subprocess.list2cmdline(cmd) for cmd in commands], out_path=script_out_path,
                            name=f'{utils.starttime}-loop_model_entities',)
                        loop_model_script = distribute.distribute(
                            loop_cmds_file, flags.predict_structure, out_path=script_out_path,
                            log_file=os.path.join(full_model_dir, 'loop_model.log'),
                            max_jobs=int(len(commands)/2 + .5), number_of_commands=len(commands))
                        loop_model_script_message = 'Once you are satisfied, run the following to distribute ' \
                                                    f'loop-modeling jobs:\n\t{shell} {loop_model_script}'
                        info_messages.append(loop_model_script_message)
                        # This prevents refinement from trying as it will be called upon distribution of the script
                        return info_messages
                    else:
                        # # Hard code in parameters
                        # model_type = 'monomer'
                        relaxed = self.job.predict.models_to_relax is not None
                        # Set up the various model_runners to supervise the prediction task for each sequence
                        monomer_runners = \
                            resources.ml.set_up_model_runners(model_type='monomer', development=self.job.development)
                        multimer_runners = \
                            resources.ml.set_up_model_runners(model_type='multimer', development=self.job.development)
                        # I don't suppose I need to reinitialize these for different length inputs, but I'm sure I will

                        # Predict each
                        for idx, protein in enumerate(protein_data_to_loop_model):
                            entity_name = protein.entity_id
                            # .model_source should be a file containing an oriented, asymmetric version of the structure
                            entity = Entity.from_file(protein.model_source, metadata=protein)

                            # Using the protein.uniprot_entity.reference_sequence would be preferred, however, it should
                            # be realigned to the structure.reference_sequence or .sequence in order to not have large
                            # insertions well beyond the indicated structural domain
                            # In a similar mechanism to load_evolutionary_profile(), these need to be combined...
                            # Example:
                            # for entity in protein.uniprot_entities:
                            #     entity.reference_sequence

                            # Remove tags from reference_sequence
                            clean_reference_sequence = expression.remove_terminal_tags(entity.reference_sequence)
                            logger.debug(f'Found the .reference_sequence:\n{entity.reference_sequence}')
                            logger.debug(f'Found the clean_reference_sequence:\n{clean_reference_sequence}')
                            source_gap_mutations = generate_mutations(clean_reference_sequence, entity.sequence,
                                                                      zero_index=True, only_gaps=True)
                            # Format the Pose to have the proper sequence to predict loops/disorder
                            logger.debug(f'Found the source_gap_mutations: {source_gap_mutations}')
                            for residue_index, mutation in source_gap_mutations.items():
                                # residue_index is zero indexed
                                new_aa_type = mutation['from']
                                # What happens if Entity has resolved tag density?
                                #  mutation_index: {'from': '-', 'to: LETTER}}
                                if new_aa_type == '-':
                                    # This could be removed from the structure but that seems implicitly bad
                                    continue
                                entity.insert_residue_type(residue_index, new_aa_type, chain_id=entity.chain_id)

                            # If the msa features are present, the prediction should succeed with high probability...
                            # Attach evolutionary info to the entity
                            evolution_loaded, alignment_loaded = load_evolutionary_profile(api_db, entity)

                            # After all sequence modifications, create the entity.assembly
                            entity.make_oligomer(symmetry=protein.symmetry_group)
                            if entity.number_of_symmetry_mates > 1:
                                af_symmetric = True
                                model_runners = multimer_runners
                                previous_position_coords = jnp.asarray(entity.assembly.alphafold_coords)
                            else:
                                af_symmetric = False
                                model_runners = monomer_runners
                                previous_position_coords = jnp.asarray(entity.alphafold_coords)
                            # Don't get the msa (no_msa=True) if the alignment_loaded is missing (False)
                            features = entity.get_alphafold_features(symmetric=af_symmetric,
                                                                     no_msa=not alignment_loaded,
                                                                     templates=True)
                            # Put the entity oligomeric coordinates in as a prior to bias the prediction
                            features['prev_pos'] = previous_position_coords
                            # Run the prediction
                            entity_structures, entity_scores = \
                                resources.ml.af_predict(features, model_runners,  # {**features, **template_features},
                                                        gpu_relax=self.job.predict.use_gpu_relax,
                                                        models_to_relax='best')  # self.job.predict.models_to_relax)
                            if relaxed:
                                structures_to_load = entity_structures.get('relaxed', [])
                            else:
                                structures_to_load = entity_structures.get('unrelaxed', [])

                            pose_kwargs = dict(name=entity_name, entity_info=protein.entity_info,
                                               symmetry=protein.symmetry_group)
                            folded_entities = {
                                model_name: Pose.from_pdb_lines(structure_.splitlines(), **pose_kwargs)
                                for model_name, structure_ in structures_to_load.items()}
                            if relaxed:  # Set b-factor data as relaxed get overwritten
                                model_plddts = {model_name: scores['plddt'][:entity.number_of_residues]
                                                for model_name, scores in entity_scores.items()}
                                for model_name, entity_ in folded_entities.items():
                                    entity_.set_b_factor_data(model_plddts[model_name])
                            # Check for the rmsd between the backbone of the provided Entity and
                            # the Alphafold prediction
                            # If the model were to be multimeric, then use this...
                            # if multimer:
                            #     entity_cb_coords = np.concatenate([mate.cb_coords for mate in entity.chains])
                            #     Tod0 entity_backbone_and_cb_coords = entity.assembly.cb_coords

                            # Only use the original indices to align
                            new_indices = list(source_gap_mutations.keys())
                            align_indices = [idx for idx in entity.residue_indices if idx not in new_indices]
                            template_cb_coords = entity.cb_coords[align_indices]
                            min_rmsd = float('inf')
                            min_entity = None
                            for af_model_name, entity_ in folded_entities.items():
                                rmsd, rot, tx = structure.coords.superposition3d(
                                    template_cb_coords, entity_.cb_coords[align_indices])
                                if rmsd < min_rmsd:
                                    min_rmsd = rmsd
                                    # Move the Alphafold model into the Pose reference frame
                                    entity_.transform(rotation=rot, translation=tx)
                                    min_entity = entity_

                            # Indicate that this ProteinMetadata has been processed for loop modeling
                            protein.loop_modeled = True
                            protein.refined = relaxed
                            # Save the min_model asu (now aligned with entity, which was oriented prior)
                            full_model_file = self.full_models.path_to(name=entity_name)
                            min_entity.write(out_path=full_model_file)
                            if relaxed:
                                refined_path = self.refined.path_to(name=entity_name)
                                shutil.copy(full_model_file, refined_path)
                else:  # rosetta_loop_model
                    raise NotImplementedError(f"Rosetta loop model hasn't been updated to use ProteinMetadata")
                    flags_file = os.path.join(full_model_dir, 'loop_model_flags')
                    # if not os.path.exists(flags_file):
                    loop_model_flags = ['-remodel::save_top 0', '-run:chain A', '-remodel:num_trajectory 1']
                    #                   '-remodel:run_confirmation true', '-remodel:quick_and_dirty',
                    _flags = utils.rosetta.flags.copy() + loop_model_flags
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
                    sym_def_files = {}
                    for idx, protein_data in enumerate(protein_data_to_loop_model):
                        if data.symmetry_group not in sym_def_files:
                            sym_def_files[data.symmetry_group] = utils.SymEntry.sdf_lookup(data.symmetry_group)
                        # Make a new directory for each structure
                        structure_out_path = os.path.join(full_model_dir, protein_data.name)
                        putils.make_path(structure_out_path)
                        structure_ = Pose.from_file(protein_data.model_source)
                        structure_.renumber_residues()
                        structure_loop_file = structure_.make_loop_file(out_path=full_model_dir)
                        if not structure_loop_file:  # No loops found, copy input file to the full model
                            copy_cmd = ['scp', self.refined.path_to(protein_data.name),
                                        self.full_models.path_to(protein_data.name)]
                            loop_model_cmds.append(
                                resources.distribute.write_script(
                                    subprocess.list2cmdline(copy_cmd), name=protein_data.name, out_path=full_model_dir))
                            # Can't do this v as refined path doesn't exist yet
                            # shutil.copy(self.refined.path_to(protein_data.name),
                            #             self.full_models.path_to(protein_data.name))
                            continue
                        structure_blueprint = structure_.make_blueprint_file(out_path=full_model_dir)
                        structure_cmd = utils.rosetta.script_cmd + loop_model_cmd \
                            + [f'blueprint={structure_blueprint}', f'loop_file={structure_loop_file}',
                               '-in:file:s', self.refined.path_to(protein_data.name),
                               '-out:path:pdb', structure_out_path] \
                            + [f'sdf={sym_def_files[protein_data.symmetry_group]}',
                               f'symmetry={"asymmetric" if protein_data.symmetry_group == "C1" else "make_point_group"}']
                        #     + (['-symmetry:symmetry_definition', sym_def_files[protein_data.symmetry]]
                        #        if protein_data.symmetry != 'C1' else [])
                        # Create a multimodel from all output loop models
                        multimodel_cmd = ['python', putils.models_to_multimodel_exe, '-d', structure_loop_file,
                                          '-o', os.path.join(full_model_dir, f'{protein_data.name}_ensemble.pdb')]
                        # Copy the first model from output loop models to be the full model
                        copy_cmd = ['scp', os.path.join(structure_out_path, f'{protein_data.name}_0001.pdb'),
                                    self.full_models.path_to(protein_data.name)]
                        loop_model_cmds.append(
                            resources.distribute.write_script(
                                subprocess.list2cmdline(structure_cmd), name=protein_data.name, out_path=full_model_dir,
                                additional=[subprocess.list2cmdline(multimodel_cmd),
                                            subprocess.list2cmdline(copy_cmd)]))
                    if batch_commands:
                        loop_cmds_file = \
                            resources.distribute.write_commands(
                                loop_model_cmds, name=f'{utils.starttime}-loop_model_entities', out_path=full_model_dir)
                        loop_model_script = \
                            distribute.distribute(loop_cmds_file, flags.refine, out_path=script_out_path,
                                                  log_file=os.path.join(full_model_dir, 'loop_model.log'),
                                                  max_jobs=int(len(loop_model_cmds)/2 + .5),
                                                  number_of_commands=len(loop_model_cmds))
                        loop_model_script_message = 'Once you are satisfied, run the following to distribute ' \
                                                    f'loop_modeling jobs:\n\t{shell} {loop_model_script}'
                        info_messages.append(loop_model_script_message)
                    else:
                        raise NotImplementedError("Currently, loop modeling can't be run in the shell. "
                                                  'Implement this if you would like this feature')
                    # Todo this is sloppy as this doesn't necessarily indicate that work will be done (batch_command)
                    # Indicate that this ProteinMetadata has been processed
                    for protein in protein_data_to_refine:
                        protein.loop_modeled = True
            else:  # Indicate that this ProteinMetadata hasn't been processed
                for protein in protein_data_to_loop_model:
                    protein.loop_modeled = False

        refine_names = self.refined.retrieve_names()
        refine_dir = self.refined.location
        # Identify the entities to refine before proceeding
        protein_data_to_refine = []
        for data in metadata:
            if not data.model_source:
                logger.debug(f"{self.preprocess_metadata_for_design.__name__}: Couldn't find the "
                             f"ProteinMetadata.model_source for {data.entity_id}. Skipping refine preprocessing")
                continue
            # If data is here, it's model_source file should've been oriented...
            if data.entity_id not in refine_names:  # Assumes oriented_asu structure name is the same
                protein_data_to_refine.append(data)

        if protein_data_to_refine:
            # Files found unrefined, check to see if work should be done
            logger.info("The following structures haven't been refined: "
                        f'{", ".join(sorted(set(protein.entity_id for protein in protein_data_to_refine)))}')
            if self.job.init.refine_input:  # is not None:
            #     refine_input = self.job.init.refine_input
            # else:  # Query user and set up commands to perform refinement on missing entities
            #     print(f'If you plan on performing {flags.design} using Rosetta, it is strongly encouraged that you '
            #           f'perform initial refinement. You can also refine them later using the {flags.refine} module')
            #     print('Would you like to refine them now?')
            #     if boolean_choice():
            #         refine_input = True
            #     else:
            #         print('To confirm, asymmetric units are going to be generated with input coordinates. Confirm '
            #               'with "y" to ensure this is what you want')
            #         if boolean_choice():
            #             refine_input = False
            #         else:
            #             refine_input = True
            #
            # if refine_input:
                if not sym_def_files:
                    sym_def_files = {}
                    for data in protein_data_to_refine:
                        if data.symmetry_group not in sym_def_files:
                            sym_def_files[data.symmetry_group] = utils.SymEntry.sdf_lookup(data.symmetry_group)
                # Generate sbatch refine command
                flags_file = os.path.join(refine_dir, 'refine_flags')
                # if not os.path.exists(flags_file):
                _flags = utils.rosetta.flags.copy() + utils.rosetta.relax_flags
                _flags.extend([f'-out:path:pdb {refine_dir}', '-no_scorefile true'])
                _flags.remove('-output_only_asymmetric_unit true')  # want full oligomers
                variables = utils.rosetta.variables.copy()
                variables.append(('dist', 0))  # Todo modify if not point groups used
                _flags.append(f'-parser:script_vars {" ".join(f"{var}={val}" for var, val in variables)}')

                with open(flags_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(_flags))

                refine_cmd = [f'@{flags_file}', '-parser:protocol',
                              os.path.join(putils.rosetta_scripts_dir, f'refine.xml')]
                refine_cmds = [utils.rosetta.script_cmd + refine_cmd
                               + ['-in:file:s', protein.model_source, '-parser:script_vars']
                               + [f'sdf={sym_def_files[protein.symmetry_group]}',
                                  f'symmetry={"asymmetric" if protein.symmetry_group == "C1" else "make_point_group"}']
                               for protein in protein_data_to_refine]
                if batch_commands:
                    commands_file = \
                        resources.distribute.write_commands(
                            [subprocess.list2cmdline(cmd) for cmd in refine_cmds], out_path=refine_dir,
                            name=f'{utils.starttime}-refine_entities')
                    refine_script = \
                        distribute.distribute(commands_file, flags.refine, out_path=script_out_path,
                                              log_file=os.path.join(refine_dir, f'{putils.refine}.log'),
                                              max_jobs=int(len(refine_cmds)/2 + .5),
                                              number_of_commands=len(refine_cmds))
                    multi_script_warning = "\n***Run this script AFTER completion of the loop modeling script***\n" \
                        if info_messages else ""
                    refine_script_message = f'Once you are satisfied, run the following to distribute refine jobs:' \
                                            f'{multi_script_warning}\n\t{shell} {refine_script}'
                    info_messages.append(refine_script_message)
                else:
                    raise NotImplementedError("Currently, refinement can't be run in the shell. "
                                              'Implement this if you would like this feature')
                # Todo this is sloppy as this doesn't necessarily indicate that this work will be done (batch_command)
                # Indicate that this ProteinMetadata has been processed
                for protein in protein_data_to_refine:
                    protein.refined = True
            else:  # Indicate that this ProteinMetadata hasn't been processed
                for protein in protein_data_to_refine:
                    protein.refined = False

        return info_messages


class StructureDatabaseFactory:
    """Return a StructureDatabase instance by calling the Factory instance with the StructureDatabase source name

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the named StructureDatabase
    """

    def __init__(self, **kwargs):
        self._database = None

    def destruct(self, **kwargs):
        self._database = None

    def __call__(self, source: str = None, sql: bool = False, **kwargs) -> StructureDatabase:
        """Return the specified StructureDatabase object singleton

        Args:
            source: The StructureDatabase source path, or name if SQL database
            sql: Whether the StructureDatabase is a SQL database
        Returns:
            The instance of the specified StructureDatabase
        """
        if self._database:
            return self._database
        elif sql:
            raise NotImplementedError('SQL set up has not been completed')
        else:
            pdbs = os.path.join(source, 'PDBs')  # Used to store downloaded PDB's
            # stride directory
            stride_dir = os.path.join(source, 'stride')
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

            self._database = \
                StructureDatabase(pdbs, full_model_dir, orient_dir, orient_asu_dir, refine_dir, stride_dir, sql=None)

        return self._database

    def get(self, source: str = None, **kwargs) -> StructureDatabase:
        """Return the specified Database object singleton

        Keyword Args:
            source: str = None - The StructureDatabase source path, or name if SQL database
            sql: bool = False - Whether the StructureDatabase is a SQL database
        Returns:
            The instance of the specified StructureDatabase
        """
        return self.__call__(source, **kwargs)


structure_database_factory: Annotated[StructureDatabaseFactory,
                                      'Calling this factory method returns the single instance of the Database class '
                                      'located at the "source" keyword argument'] = \
    StructureDatabaseFactory()
"""Calling this factory method returns the single instance of the Database class located at the "source" keyword 
argument
"""
