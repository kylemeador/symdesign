import os
from shutil import copy, move
from typing import AnyStr

import structure.model
from structure.fragment.fragment import FragmentDatabase
from utils.path import biological_fragment_db_pickle, reference_aa_file, reference_residues_pkl, biological_interfaces
from utils import timestamp, pickle_object, start_log, get_file_paths_recursively

logger = start_log(name=__name__)


def create_fragment_db_from_raw_files(source: AnyStr) -> FragmentDatabase:
    """Load the FragmentDatabase data from disk and return the instance

    Args:
        source: Which database to use?
    Returns:
        The loaded FragmentDatabase
    """
    fragment_db = FragmentDatabase(source=source)
    logger.info(f'Initializing {source} FragmentDatabase from disk. This may take awhile...')
    # self.get_monofrag_cluster_rep_dict()
    fragment_db.reps = {int(os.path.splitext(os.path.basename(file))[0]):
                        structure.base.Structure.from_file(file, entities=False, log=None).ca_coords
                        for file in get_file_paths_recursively(fragment_db.monofrag_representatives_path)}
    fragment_db.paired_frags = load_paired_fragment_representatives(fragment_db.cluster_representatives_path)
    fragment_db.load_cluster_info()  # Using my generated data instead of Josh's for future compatibility and size
    # fragment_db.load_cluster_info_from_text()
    fragment_db._index_ghosts()

    return fragment_db


def load_paired_fragment_representatives(cluster_representatives_path) \
        -> dict[tuple[int, int, int], tuple['structure.model.Model', str]]:
    """From a directory containing cluster representatives, load the representatives to a dictionary

    Args:
        cluster_representatives_path: The directory containing the paired fragment representative model files
    Returns:
        A mapping between the cluster type and the loaded representative model
    """
    identified_files = \
        {os.path.basename(os.path.dirname(representative_file)): representative_file
         for representative_file in get_file_paths_recursively(cluster_representatives_path)}

    paired_frags = {}
    for cluster_name, file_path in identified_files.items():
        i_type, j_type, k_type = map(int, cluster_name.split('_'))

        ijk_frag_cluster_rep_pdb = structure.model.Model.from_file(file_path, entities=False, log=None)
        # Load as Model as we must look up the partner coords later by using chain_id stored in file_name
        partner_chain_idx = file_path.find('partnerchain')
        ijk_cluster_rep_partner_chain = file_path[partner_chain_idx + 13:partner_chain_idx + 14]
        # Store in the dictionary
        paired_frags[(i_type, j_type, k_type)] = ijk_frag_cluster_rep_pdb, ijk_cluster_rep_partner_chain
        # # OR
        # i_dict = self.paired_frags.get(i_type)
        # if not i_dict:
        #     self.paired_frags[i_type] = \
        #         {j_type: {k_type: (ijk_frag_cluster_rep_pdb, ijk_cluster_rep_partner_chain)}}
        # else:
        #     j_dict = i_dict.get(j_type)
        #     if not j_dict:
        #         self.paired_frags[i_type][j_type] = \
        #             {k_type: (ijk_frag_cluster_rep_pdb, ijk_cluster_rep_partner_chain)}
        # # self.paired_frags[i_type][j_type][k_type] = ijk_frag_cluster_rep_pdb, ijk_cluster_rep_partner_chain
    return paired_frags

    # for root, dirs, files in os.walk(self.cluster_representatives_path):
    #     if not dirs:
    #         i_cluster_type, j_cluster_type, k_cluster_type = map(int, os.path.basename(root).split('_'))
    #
    #         if i_cluster_type not in self.paired_frags:
    #             self.paired_frags[i_cluster_type] = {}
    #         if j_cluster_type not in self.paired_frags[i_cluster_type]:
    #             self.paired_frags[i_cluster_type][j_cluster_type] = {}
    #
    #         for file in files:
    #             ijk_frag_cluster_rep_pdb = \
    #                 structure.model.Model.from_file(os.path.join(root, file), entities=False, log=None)
    #             # mapped_chain_idx = file.find('mappedchain')
    #             # ijk_cluster_rep_mapped_chain = file[mapped_chain_idx + 12:mapped_chain_idx + 13]
    #             # must look up the partner coords later by using chain_id stored in file
    #             partner_chain_idx = file.find('partnerchain')
    #             ijk_cluster_rep_partner_chain = file[partner_chain_idx + 13:partner_chain_idx + 14]
    #             self.paired_frags[i_cluster_type][j_cluster_type][k_cluster_type] = \
    #                 (ijk_frag_cluster_rep_pdb, ijk_cluster_rep_partner_chain)  # ijk_cluster_rep_mapped_chain,


try:
    from structure import base
except Exception as error:  # If something goes wrong, we should remake this too
    # Create and save the new reference_residues_pkl from scratch
    # Todo if this never catches then these aren't updated
    ref_aa = base.Structure.from_file(reference_aa_file)
    move(reference_residues_pkl, f'{reference_residues_pkl}.bak')
    pickle_object(ref_aa.residues, name=reference_residues_pkl, out_path='')

# Create fragment database for all ijk cluster representatives
# This should work now
from structure import base
base.protein_backbone_atom_types = {'N', 'CA', 'O'}  # 'C', Removing 'C' for fragment library guide atoms...
ijk_frag_db = create_fragment_db_from_raw_files(source=biological_interfaces)

copy(biological_fragment_db_pickle, f'{biological_fragment_db_pickle}.bak-{timestamp}')
pickle_object(ijk_frag_db, name=biological_fragment_db_pickle, out_path='')
