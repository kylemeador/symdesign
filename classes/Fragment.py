import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([parent_dir])
import numpy as np

from SymDesignUtils import start_log
from BioPDBUtils import biopdb_aligned_chain, biopdb_superimposer


logger = start_log(name=__name__)
index_offset = 1


class GhostFragment:
    def __init__(self, structure, i_frag_type, j_frag_type, k_frag_type, ijk_rmsd, aligned_surf_frag_central_res_tup,
                 guide_coords=None):
        self.structure = structure
        self.i_frag_type = i_frag_type
        self.j_frag_type = j_frag_type
        self.k_frag_type = k_frag_type
        self.rmsd = ijk_rmsd
        self.aligned_surf_frag_central_res_tup = aligned_surf_frag_central_res_tup

        if not guide_coords:
            self.guide_coords = self.structure.chain('9').get_coords()
        else:
            self.guide_coords = guide_coords

    def get_ijk(self):
        """Return the fragments corresponding cluster index information

        Returns:
            (tuple[str, str, str]): I cluster index, J cluster index, K cluster index
        """
        return self.i_frag_type, self.j_frag_type, self.k_frag_type

    def get_aligned_surf_frag_central_res_tup(self):
        """Return the fragment information the GhostFragment instance is aligned to
        Returns:
            (tuple): aligned chain, aligned residue_number"""
        return self.aligned_surf_frag_central_res_tup

    def get_i_type(self):
        return self.i_frag_type

    def get_j_type(self):
        return self.j_frag_type

    def get_k_type(self):
        return self.k_frag_type

    def get_rmsd(self):
        return self.rmsd

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure

    # def get_pdb_coords(self):
    #     return self.pdb.get_coords()

    def get_guide_coords(self):
        return self.guide_coords

    def get_center_of_mass(self):
        return np.matmul(np.array([0.33333, 0.33333, 0.33333]), self.guide_coords)


class MonoFragment:
    def __init__(self, pdb=None, monofrag_cluster_rep_dict=None, fragment_type=None, guide_coords=None,
                 central_res_num=None, central_res_chain_id=None, rmsd_thresh=0.75):
        self.structure = pdb
        self.type = fragment_type
        self.guide_coords = guide_coords
        self.central_res_num = central_res_num
        self.central_res_chain_id = central_res_chain_id

        if self.structure and monofrag_cluster_rep_dict:
            frag_ca_atoms = self.structure.get_ca_atoms()
            central_residue = frag_ca_atoms[2]  # Todo integrate this to be the main object identifier
            self.central_res_num = central_residue.residue_number
            self.central_res_chain_id = central_residue.chain
            min_rmsd = float('inf')
            # min_rmsd_cluster_rep_type = None
            for cluster_type, cluster_rep in monofrag_cluster_rep_dict.items():
                rmsd, rot, tx = biopdb_superimposer(frag_ca_atoms, cluster_rep.get_ca_atoms())

                if rmsd <= rmsd_thresh and rmsd <= min_rmsd:
                    self.type = cluster_type
                    min_rmsd, min_rot, min_tx = rmsd, rot, tx

            if self.type:
                # self.type = min_rmsd_cluster_rep_type
                guide_coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
                # t_vec = np.array(min_tx)
                # r_mat = np.transpose(np.array(min_rot))
                self.rot = min_rot
                self.tx = min_tx
                self.guide_coords = np.matmul(guide_coords, min_rot) + min_tx

    @classmethod
    def from_residue(cls):
        return cls()

    @classmethod
    def from_database(cls, pdb, representative_dict):
        return cls(pdb=pdb, monofrag_cluster_rep_dict=representative_dict)

    @classmethod
    def from_fragment(cls, pdb=None, fragment_type=None, guide_coords=None, central_res_num=None,
                      central_res_chain_id=None):
        return cls(pdb=pdb, fragment_type=fragment_type, guide_coords=guide_coords, central_res_num=central_res_num,
                   central_res_chain_id=central_res_chain_id)

    def get_central_res_tup(self):
        return self.central_res_chain_id, self.central_res_num  # self.central_residue.number

    def get_guide_coords(self):
        return self.guide_coords

    def get_center_of_mass(self):
        if self.guide_coords:
            return np.matmul(np.array([0.33333, 0.33333, 0.33333]), self.guide_coords)
        else:
            return None

    def get_i_type(self):
        return self.type

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure

    # def get_pdb_coords(self):
    #     return self.structure.get_coords()

    def get_central_res_num(self):
        return self.central_res_num  # self.central_residue.number

    def get_central_res_chain_id(self):
        return self.central_res_chain_id

    def get_ghost_fragments(self, intfrag_cluster_rep_dict, kdtree_oligomer_backbone, intfrag_cluster_info_dict,
                            clash_dist=2.2):
        if self.type in intfrag_cluster_rep_dict:
            ghost_fragments = []
            for j_type in intfrag_cluster_rep_dict[self.type]:
                for k_type in intfrag_cluster_rep_dict[self.type][j_type]:
                    intfrag = intfrag_cluster_rep_dict[self.type][j_type][k_type]
                    frag_pdb = intfrag[0]
                    frag_paired_chain = intfrag[1]
                    # # frag_mapped_chain = intfrag[1]
                    # # intfrag_mapped_chain_central_res_num = intfrag[2]
                    # # intfrag_partner_chain_id = intfrag[3]
                    # # intfrag_partner_chain_central_res_num = intfrag[4]

                    # aligned_ghost_frag_pdb = biopdb_aligned_chain(self.structure, frag_pdb.chain(frag_mapped_chain))
                    # Only keep ghost fragments that don't clash with oligomer backbone
                    # Note: guide atoms, mapped chain atoms and non-backbone atoms not included
                    # ghost_frag_chain = (set(frag_pdb.chain_id_list) - {'9', frag_mapped_chain}).pop()
                    aligned_ghost_frag_pdb = frag_pdb.return_transformed_copy(rotation=self.rot, translation=self.tx)
                    # print(aligned_ghost_frag_pdb.filepath)
                    # print(aligned_ghost_frag_pdb.chain_id_list)
                    # for chain in aligned_ghost_frag_pdb.chains:
                    #     print(chain.name)
                    # print(frag_paired_chain)
                    g_frag_bb_coords = aligned_ghost_frag_pdb.chain(frag_paired_chain).get_backbone_coords()
                    cb_clash_count = kdtree_oligomer_backbone.two_point_correlation(g_frag_bb_coords, [clash_dist])

                    if cb_clash_count[0] == 0:
                        rmsd = intfrag_cluster_info_dict[self.type][j_type][k_type].get_rmsd()
                        ghost_fragments.append(GhostFragment(aligned_ghost_frag_pdb, self.type, j_type, k_type, rmsd,
                                                             self.get_central_res_tup()))

            return ghost_fragments

        else:
            return None


