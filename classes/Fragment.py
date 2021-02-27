import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([parent_dir])
import numpy as np

# from classes.Atom import Atom
# from classes.PDB import PDB
from Structure import Structure
from PDB import PDB
from BioPDBUtils import biopdb_aligned_chain, biopdb_superimposer
from PathUtils import intfrag_cluster_rep_dirpath, monofrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath


def get_surface_fragments(pdb):  # Todo to PDB.py
    surface_frags = []
    for (chain, res_num) in pdb.get_surface_residue_info():
        frag_res_nums = [res_num - 2, res_num - 1, res_num, res_num + 1, res_num + 2]
        ca_count = 0

        # for atom in pdb.get_chain_atoms(chain):
        # for atom in pdb.chain(chain):
        # frag_atoms = pdb.chain(chain).get_residue_atoms(numbers=frag_res_nums, pdb=True)  # Todo
        frag_atoms = []
        for atom in pdb.chain(chain).get_atoms():
            # if atom.residue_number in frag_res_nums:  # TODO
            if atom.pdb_residue_number in frag_res_nums:
                frag_atoms.append(atom)
                if atom.is_CA():
                    ca_count += 1
        if ca_count == 5:
            surface_frags.append(Structure.from_atoms(frag_atoms))

    return surface_frags


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

        if pdb and monofrag_cluster_rep_dict:
            frag_ca_atoms = self.structure.get_ca_atoms()
            central_residue = frag_ca_atoms[2]  # Todo integrate this to be the main object identifier
            self.central_res_num = central_residue.residue_number
            self.central_res_chain_id = central_residue.chain
            min_rmsd = float('inf')
            min_rmsd_cluster_rep_type = None
            for cluster_type, cluster_rep in monofrag_cluster_rep_dict.items():
                rmsd, rot, tx = biopdb_superimposer(frag_ca_atoms, cluster_rep.get_ca_atoms())

                if rmsd <= min_rmsd and rmsd <= rmsd_thresh:
                    min_rmsd_cluster_rep_type = cluster_type
                    min_rmsd, min_rot, min_tx = rmsd, rot, tx

            if min_rmsd_cluster_rep_type:
                self.type = min_rmsd_cluster_rep_type
                guide_coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
                # t_vec = np.array(min_tx)
                # r_mat = np.transpose(np.array(min_rot))
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
                    intfrag_pdb = intfrag[0]
                    intfrag_mapped_chain = intfrag[1]
                    # intfrag_mapped_chain_central_res_num = intfrag[2]
                    # intfrag_partner_chain_id = intfrag[3]
                    # intfrag_partner_chain_central_res_num = intfrag[4]

                    aligned_ghost_frag_pdb = biopdb_aligned_chain(self.pdb, self.pdb.chain_id_list[0], intfrag_pdb,
                                                                  intfrag_mapped_chain)

                    # Only keep ghost fragments that don't clash with oligomer backbone
                    # Note: guide atoms, mapped chain atoms and non-backbone atoms not included
                    ghost_frag_chain = (set(aligned_ghost_frag_pdb.chain_id_list) - {'9', intfrag_mapped_chain}).pop()
                    g_frag_bb_coords = aligned_ghost_frag_pdb.chain(ghost_frag_chain).get_backbone_coords()

                    cb_clash_count = kdtree_oligomer_backbone.two_point_correlation(g_frag_bb_coords, [clash_dist])

                    if cb_clash_count[0] == 0:
                        rmsd = intfrag_cluster_info_dict[self.type][j_type][k_type].get_rmsd()
                        ghost_fragments.append(GhostFragment(aligned_ghost_frag_pdb, self.type, j_type, k_type, rmsd,
                                                             self.get_central_res_tup()))

            return ghost_fragments

        else:
            return None


class ClusterInfoFile:
    def __init__(self, infofile_path):
        self.infofile_path = infofile_path
        self.name = None
        self.size = None
        self.rmsd = None
        self.representative_filename = None
        self.central_residue_pair_freqs = []
        self.central_residue_pair_counts = []
        self.load_info()

    def load_info(self):
        infofile = open(self.infofile_path, "r")
        info_lines = infofile.readlines()
        infofile.close()
        is_res_freq_line = False
        for line in info_lines:

            if line.startswith("CLUSTER NAME:"):
                self.name = line.split()[2]
            if line.startswith("CLUSTER SIZE:"):
                self.size = int(line.split()[2])
            if line.startswith("CLUSTER RMSD:"):
                self.rmsd = float(line.split()[2])
            if line.startswith("CLUSTER REPRESENTATIVE NAME:"):
                self.representative_filename = line.split()[3]

            if line.startswith("CENTRAL RESIDUE PAIR COUNT:"):
                is_res_freq_line = False
            if is_res_freq_line:
                res_pair_type = (line.split()[0][0], line.split()[0][1])
                res_pair_freq = float(line.split()[1])
                self.central_residue_pair_freqs.append((res_pair_type, res_pair_freq))
            if line.startswith("CENTRAL RESIDUE PAIR FREQUENCY:"):
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
    def __init__(self):  # , monofrag_cluster_rep_dirpath, intfrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath):
        self.monofrag_cluster_rep_dirpath = monofrag_cluster_rep_dirpath
        self.intfrag_cluster_rep_dirpath = intfrag_cluster_rep_dirpath
        self.intfrag_cluster_info_dirpath = intfrag_cluster_info_dirpath
        self.reps = None
        self.paired_frags = None
        self.info = None

    def get_monofrag_cluster_rep_dict(self):
        cluster_rep_pdb_dict = {}
        for root, dirs, files in os.walk(self.monofrag_cluster_rep_dirpath):
            for filename in files:
                # if ".pdb" in filename:  # Todo remove this check as all files are .pdb
                pdb = PDB.from_file(os.path.join(self.monofrag_cluster_rep_dirpath, filename), remove_alt_location=True)
                cluster_rep_pdb_dict[os.path.splitext(filename)[0]] = pdb

        self.reps = cluster_rep_pdb_dict
        # return cluster_rep_pdb_dict

    def get_intfrag_cluster_rep_dict(self):
        i_j_k_intfrag_cluster_rep_dict = {}
        for dirpath1, dirnames1, filenames1 in os.walk(self.intfrag_cluster_rep_dirpath):
            if not dirnames1:
                ijk_cluster_name = dirpath1.split("/")[-1]
                i_cluster_type = ijk_cluster_name.split("_")[0]
                j_cluster_type = ijk_cluster_name.split("_")[1]
                k_cluster_type = ijk_cluster_name.split("_")[2]

                if i_cluster_type not in i_j_k_intfrag_cluster_rep_dict:
                    i_j_k_intfrag_cluster_rep_dict[i_cluster_type] = {}

                if j_cluster_type not in i_j_k_intfrag_cluster_rep_dict[i_cluster_type]:
                    i_j_k_intfrag_cluster_rep_dict[i_cluster_type][j_cluster_type] = {}

                for dirpath2, dirnames2, filenames2 in os.walk(dirpath1):
                    for filename in filenames2:
                        # if ".pdb" in filename:  # Todo remove this check as all files are .pdb
                        ijk_frag_cluster_rep_pdb = PDB.from_file(os.path.join(dirpath1, filename))
                        ijk_frag_cluster_rep_mapped_chain_id = \
                            filename[filename.find("mappedchain") + 12:filename.find("mappedchain") + 13]
                        ijk_frag_cluster_rep_partner_chain_id = \
                            filename[filename.find("partnerchain") + 13:filename.find("partnerchain") + 14]

                        # Get central residue number of mapped interface fragment chain
                        intfrag_mapped_chain_central_res_num = None
                        mapped_chain_res_count = 0
                        for atom in ijk_frag_cluster_rep_pdb.chain(ijk_frag_cluster_rep_mapped_chain_id):
                            if atom.is_CA():
                                mapped_chain_res_count += 1
                                if mapped_chain_res_count == 3:
                                    intfrag_mapped_chain_central_res_num = atom.residue_number

                        # Get central residue number of partner interface fragment chain
                        intfrag_partner_chain_central_res_num = None
                        partner_chain_res_count = 0
                        for atom in ijk_frag_cluster_rep_pdb.chain(ijk_frag_cluster_rep_partner_chain_id):
                            if atom.is_CA():
                                partner_chain_res_count += 1
                                if partner_chain_res_count == 3:
                                    intfrag_partner_chain_central_res_num = atom.residue_number

                        i_j_k_intfrag_cluster_rep_dict[i_cluster_type][j_cluster_type][k_cluster_type] = \
                            (ijk_frag_cluster_rep_pdb, ijk_frag_cluster_rep_mapped_chain_id)  # ,
                             # intfrag_mapped_chain_central_res_num, ijk_frag_cluster_rep_partner_chain_id,
                             # intfrag_partner_chain_central_res_num)

        self.paired_frags = i_j_k_intfrag_cluster_rep_dict
        # return i_j_k_intfrag_cluster_rep_dict

    def get_intfrag_cluster_info_dict(self):
        intfrag_cluster_info_dict = {}
        for dirpath1, dirnames1, filenames1 in os.walk(self.intfrag_cluster_info_dirpath):
            if not dirnames1:
                ijk_cluster_name = dirpath1.split("/")[-1]
                i_cluster_type = ijk_cluster_name.split("_")[0]
                j_cluster_type = ijk_cluster_name.split("_")[1]
                k_cluster_type = ijk_cluster_name.split("_")[2]

                if i_cluster_type not in intfrag_cluster_info_dict:
                    intfrag_cluster_info_dict[i_cluster_type] = {}

                if j_cluster_type not in intfrag_cluster_info_dict[i_cluster_type]:
                    intfrag_cluster_info_dict[i_cluster_type][j_cluster_type] = {}

                for dirpath2, dirnames2, filenames2 in os.walk(dirpath1):
                    for filename in filenames2:
                        # if ".txt" in filename:
                        intfrag_cluster_info_dict[i_cluster_type][j_cluster_type][k_cluster_type] = ClusterInfoFile(
                            dirpath1 + "/" + filename)

        self.info = intfrag_cluster_info_dict
        # return intfrag_cluster_info_dict
