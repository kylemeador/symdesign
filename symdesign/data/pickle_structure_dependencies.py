import os
import sys
from shutil import copy, move
from typing import AnyStr
# Insert the local symdesign directory at the front of the PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from symdesign.structure import base, model
from symdesign.structure.fragment.db import FragmentDatabase, Representative, RELOAD_DB, PARTNER_CHAIN
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
    logger.info(f'Initializing FragmentDatabase({source}) from disk at {fragment_db.cluster_representatives_path}. '
                'This may take awhile...')
    fragment_db.representatives = \
        {int(os.path.splitext(os.path.basename(file))[0]):
         Representative(model.Chain.from_file(file, log=None), fragment_db=fragment_db)
         for file in utils.get_file_paths_recursively(fragment_db.monofrag_representatives_path)}
    fragment_db.paired_frags = load_paired_fragment_representatives(fragment_db.cluster_representatives_path)
    fragment_db.load_cluster_info()
    fragment_db._index_ghosts()

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
        cluster_representative = model.Model.from_file(file_path, log=None, fragment_db=RELOAD_DB)
        # Load as Model as we must look up the partner coords later by using chain_id stored in file_name
        partner_chain_idx = file_path.find(PARTNER_CHAIN)
        ijk_cluster_rep_partner_chain_id = file_path[partner_chain_idx + 13:partner_chain_idx + 14]
        # Store in the dictionary
        i_j_k_type: tuple[int, int, int] = tuple(map(int, cluster_name.split('_', maxsplit=2)))  # type: ignore
        paired_frags[i_j_k_type] = cluster_representative, ijk_cluster_rep_partner_chain_id

    return paired_frags


def main():
    ref_aa = model.Chain.from_file(putils.reference_aa_file)
    if os.path.exists(putils.reference_residues_pkl):
        move(putils.reference_residues_pkl, f'{putils.reference_residues_pkl}.bak')

    utils.pickle_object([res.make_parent() for res in ref_aa.residues],
                        name=putils.reference_residues_pkl, out_path='')

    # Create fragment database for all ijk cluster representatives
    # This should work now
    base.protein_backbone_atom_types = {'N', 'CA', 'O'}  # 'C', Removing 'C' for fragment library guide atoms...
    ijk_frag_db = create_fragment_db_from_raw_files(source=putils.biological_interfaces)

    if os.path.exists(putils.biological_fragment_db_pickle):
        logger.info(f'Making a backup of the old fragment_db: {putils.biological_fragment_db_pickle} '
                    f'-> {putils.biological_fragment_db_pickle}.bak-{utils.timestamp()}')
        copy(putils.biological_fragment_db_pickle, f'{putils.biological_fragment_db_pickle}.bak-{utils.timestamp()}')
    logger.info(f'Saving the new fragment_db: {putils.biological_fragment_db_pickle}')
    utils.pickle_object(ijk_frag_db, name=putils.biological_fragment_db_pickle, out_path='')


if __name__ == '__main__':
    main()
