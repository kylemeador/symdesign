import os
from shutil import copy
from typing import AnyStr

from symdesign.structure import base, model
from symdesign.structure.fragment.db import ClusterInfo, FragmentDatabase, Representative, RELOAD_DB, PARTNER_CHAIN
from symdesign.structure.fragment.resources import biological_fragment_db_pickle, fragment_database_source_literal, \
    biological_interfaces, monofrag_cluster_rep_dirpath, intfrag_cluster_rep_dirpath
from symdesign import utils
putils = utils.path
logger = utils.start_log(name=__name__, no_log_name=True)


def create_fragment_db_from_raw_files(source: AnyStr) -> FragmentDatabase:
    """Load the FragmentDatabase data from disk and return the instance

    Args:
        source: Which database to use?

    Returns:
        The loaded FragmentDatabase
    """
    fragment_db = FragmentDatabase(source=source, fragment_length=5)  # Todo dynamic...
    representatives_path = monofrag_cluster_rep_dirpath
    paired_fragments_path = intfrag_cluster_rep_dirpath
    logger.info(f"Initializing {repr(fragment_db)} from disk at '{paired_fragments_path}'. "
                'This may take awhile...')
    fragment_db.representatives = \
        {int(os.path.splitext(os.path.basename(file))[0]):
         Representative(model.Chain.from_file(file, log=None), fragment_db=fragment_db)
         for file in utils.get_file_paths_recursively(representatives_path)}
    fragment_db.paired_frags = load_paired_fragment_representatives(paired_fragments_path)
    info = {}
    for cluster_file in utils.get_file_paths_recursively(fragment_db.location, extension='.pkl'):
        name = os.path.splitext(os.path.basename(cluster_file))[0]
        info[tuple(map(int, name.split('_', maxsplit=2)))] = ClusterInfo(name=name, **utils.unpickle(cluster_file))
    fragment_db.info = info
    fragment_db.index_ghosts()

    return fragment_db


def load_paired_fragment_representatives(cluster_representatives_path: AnyStr) \
        -> dict[tuple[int, int, int], tuple[model.Model, str]]:
    """From a directory containing cluster representatives, load the representatives to a dictionary

    Args:
        cluster_representatives_path: The directory containing the paired fragment representative model files

    Returns:
        A mapping between the cluster type and the loaded representative model
    """
    identified_files = \
        {os.path.basename(os.path.dirname(representative_file)): representative_file
         for representative_file in utils.get_file_paths_recursively(cluster_representatives_path)}

    paired_frags = {}
    for cluster_name, file_path in identified_files.items():
        # The token RELOAD_DB is passed to ensure loading happens without default loading
        cluster_representative = model.Model.from_pdb(file_path, log=None, fragment_db=RELOAD_DB)
        # Load as Model as we must look up the partner coords later by using chain_id stored in file_name
        partner_chain_idx = file_path.find(PARTNER_CHAIN)
        ijk_cluster_rep_partner_chain_id = file_path[partner_chain_idx + 13:partner_chain_idx + 14]
        # Store in the dictionary
        i_j_k_type: tuple[int, int, int] = tuple(map(int, cluster_name.split('_', maxsplit=2)))  # type: ignore
        paired_frags[i_j_k_type] = cluster_representative, ijk_cluster_rep_partner_chain_id

    return paired_frags


def main(dev: bool = False, source: fragment_database_source_literal = biological_interfaces):
    """Creates a FragmentDatabase instance using raw ijk cluster representatives"""
    original_backbone_atoms = base.protein_backbone_atom_types
    # Removing 'C' for fragment library guide atoms...
    base.protein_backbone_atom_types = {'N', 'CA', 'O'}
    ijk_frag_db = create_fragment_db_from_raw_files(source=source)

    if dev and biological_fragment_db_pickle.exists():
        backup_path = f'{biological_fragment_db_pickle}.bak-{utils.timestamp()}'
        logger.info(f'Backing of the old fragment_db: {biological_fragment_db_pickle} '
                    f'-> {backup_path}')
        copy(biological_fragment_db_pickle, backup_path)
    logger.info(f'Saving the new fragment_db: {biological_fragment_db_pickle}')
    utils.pickle_object(ijk_frag_db, out_path=biological_fragment_db_pickle)
    # Reset backbone atoms
    base.protein_backbone_atom_types = original_backbone_atoms


if __name__ == '__main__':
    main()
