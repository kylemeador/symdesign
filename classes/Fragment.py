import os

import numpy as np

import PathUtils as PUtils
from classes.Atom import Atom
from classes.PDB import PDB
from utils.BioPDBUtils import biopdb_aligned_chain
from utils.BioPDBUtils import biopdb_superimposer
from PathUtils import intfrag_cluster_rep_dirpath, monofrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath


def get_interface_fragments(pdb, chain_res_info, fragment_length=5):
    interface_frags = []

    for (chain, res_num) in chain_res_info:
        frag_atoms = []
        frag_res_nums = [res_num + i for i in range(-2, 3)]
        ca_count = 0
        # for atom in pdb.get_chain_atoms(chain):  # TODO
        for atom in pdb.chain(chain):
            if atom.residue_number in frag_res_nums:
                frag_atoms.append(atom)
                if atom.is_CA():
                    ca_count += 1
        if ca_count == 5:
            surf_frag_pdb = PDB()
            surf_frag_pdb.read_atom_list(frag_atoms)
            interface_frags.append(surf_frag_pdb)

    return interface_frags


def get_surface_fragments(pdb):  # Todo to PDB.py
    surface_frags = []
    surf_res_info = pdb.get_surface_residue_info()

    for (chain, res_num) in surf_res_info:
        frag_atoms = []
        frag_res_nums = [res_num - 2, res_num - 1, res_num, res_num + 1, res_num + 2]
        ca_count = 0
        # for atom in pdb.chain(chain).get_atoms():  # TODO
        # for atom in pdb.get_chain_atoms(chain):
        for atom in pdb.chain(chain):
            if atom.residue_number in frag_res_nums:
                frag_atoms.append(atom)
                if atom.is_CA():
                    ca_count += 1
        if ca_count == 5:
            surf_frag_pdb = PDB()
            surf_frag_pdb.read_atom_list(frag_atoms)
            surface_frags.append(surf_frag_pdb)

    return surface_frags


def get_surface_fragments_chain(pdb, chain_id):  # DEPRECIATE
    surface_frags = []
    surf_res_info = pdb.get_surface_residue_info()

    for (chain, res_num) in surf_res_info:
        if chain == chain_id:
            frag_atoms = []
            frag_res_nums = [res_num - 2, res_num - 1, res_num, res_num + 1, res_num + 2]
            ca_count = 0
            # for atom in pdb.get_chain_atoms(chain):
            for atom in pdb.chain(chain):
                if atom.residue_number in frag_res_nums:
                    frag_atoms.append(atom)
                    if atom.is_CA():
                        ca_count += 1
            if ca_count == 5:
                surf_frag_pdb = PDB()
                surf_frag_pdb.read_atom_list(frag_atoms)
                surface_frags.append(surf_frag_pdb)

    return surface_frags


class GhostFragment:
    def __init__(self, pdb, i_frag_type, j_frag_type, k_frag_type, ijk_rmsd, aligned_surf_frag_central_res_tup,
                 guide_coords=None):  # aligned_surf_frag_central_res_tup, guide_atoms=None, pdb_coords=None
        self.pdb = pdb
        self.i_frag_type = i_frag_type
        self.j_frag_type = j_frag_type
        self.k_frag_type = k_frag_type
        self.rmsd = ijk_rmsd
        self.central_res_tup = aligned_surf_frag_central_res_tup
        # self.aligned_surf_frag_central_res_tup = aligned_surf_frag_central_res_tup  # (chain, res_number, ch, res#)

        if not guide_coords:  # guide_atoms, , pdb_coords] == [None, None]
            # self.guide_atoms = []
            self.guide_coords = []
            # self.pdb_coords = []
            for atom in self.pdb.all_atoms:
                # self.pdb_coords.append([atom.x, atom.y, atom.z])
                if atom.chain == "9":
                    # self.guide_atoms.append(atom)
                    self.guide_coords.append([atom.x, atom.y, atom.z])

        else:
            # self.guide_atoms = guide_atoms
            self.guide_coords = guide_coords
            # self.pdb_coords = pdb_coords

    def get_ijk(self):
        """Return the fragments corresponding cluster index information

        Returns:
            (tuple): I cluster index, J cluster index, K cluster index
        """
        return self.i_frag_type, self.j_frag_type, self.k_frag_type

    def get_central_res_tup(self):
        """Get the representative chain and residue information from the underlying observation

        Returns:
            (tuple): Ghost Fragment Mapped Chain ID, Central Residue Number, Partner Chain ID, Central Residue Number
        """
        return self.central_res_tup

    # def get_aligned_surf_frag_central_res_tup(self):
    #     """Return the fragment information the GhostFragment instance is aligned to
    #     Returns:
    #         (tuple): aligned chain, aligned residue_number"""
    #     return self.aligned_surf_frag_central_res_tup

    def get_aligned_central_res_info(self):
        """Return the cluster representative and aligned fragment information for the GhostFragment instance

        Returns:
            (tuple): mapped_chain, mapped_central_res_number, partner_chain, partner_central_residue_number,
            chain, residue_number
        """
        return self.central_res_tup  # + self.aligned_surf_frag_central_res_tup

    def get_i_frag_type(self):
        return self.i_frag_type

    def get_j_frag_type(self):
        return self.j_frag_type

    def get_k_frag_type(self):
        return self.k_frag_type

    def get_rmsd(self):
        return self.rmsd

    def get_pdb(self):
        return self.pdb

    # def get_pdb_coords(self):
    #     return self.pdb_coords

    # def get_guide_atoms(self):
    #     return self.guide_atoms

    def get_guide_coords(self):
        return self.guide_coords

    def get_center_of_mass(self):
        return np.matmul(np.array([0.33333, 0.33333, 0.33333]), np.array(self.guide_coords))


class MonoFragment:
    def __init__(self, pdb, monofrag_cluster_rep_dict=None, type=None, guide_coords=None, central_res_num=None,
                 central_res_chain_id=None, rmsd_thresh=0.75):  # pdb_coords=None,
        self.pdb = None
        # self.pdb_coords = None
        self.type = None
        self.guide_coords = None
        # self.guide_atoms = None
        self.central_res_num = None
        self.central_res_chain_id = None

        if monofrag_cluster_rep_dict is None and type is not None and guide_coords is not None and \
                central_res_num is not None and central_res_chain_id is not None:  #  and pdb_coords is not None:
            self.pdb = pdb
            # self.pdb_coords = pdb_coords
            self.type = type
            self.guide_coords = guide_coords
            # a1 = Atom(1, "CA", " ", "GLY", "9", 0, " ", guide_coords[0][0], guide_coords[0][1], guide_coords[0][2],
            #           1.00, 20.00, "C", "")
            # a2 = Atom(2, "N", " ", "GLY", "9", 0, " ", guide_coords[1][0], guide_coords[1][1], guide_coords[1][2], 1.00,
            #           20.00, "N", "")
            # a3 = Atom(3, "O", " ", "GLY", "9", 0, " ", guide_coords[2][0], guide_coords[2][1], guide_coords[2][2], 1.00,
            #           20.00, "O", "")
            # self.guide_atoms = [a1, a2, a3]
            self.central_res_num = central_res_num
            self.central_res_chain_id = central_res_chain_id

        # elif monofrag_cluster_rep_dict is not None and pdb is not None:  # TODO
        elif monofrag_cluster_rep_dict is not None and type is None and guide_coords is None and \
                central_res_num is None and central_res_chain_id is None:  # and pdb_coords is None:
            self.pdb = pdb
            # self.pdb_coords = self.pdb.extract_all_coords()
            frag_ca_atoms = self.pdb.get_CA_atoms()
            # frag_ca_atoms = self.pdb.get_ca_atoms()  # TODO
            self.central_res_num = frag_ca_atoms[2].residue_number
            self.central_res_chain_id = self.pdb.chain_id_list[0]

            # a1 = Atom(1, "CA", " ", "GLY", "9", 0, " ", 0.0, 0.0, 0.0, 1.00, 20.00, "C", "")
            # a2 = Atom(2, "N", " ", "GLY", "9", 0, " ", 3.0, 0.0, 0.0, 1.00, 20.00, "N", "")
            # a3 = Atom(3, "O", " ", "GLY", "9", 0, " ", 0.0, 3.0, 0.0, 1.00, 20.00, "O", "")
            guide_coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
            min_rmsd = float('inf')
            min_rmsd_cluster_rep_rot_tx = None
            min_rmsd_cluster_rep_type = None
            for cluster_type in monofrag_cluster_rep_dict:
                cluster_rep = monofrag_cluster_rep_dict[cluster_type]
                cluster_rep_ca_atoms = cluster_rep.get_CA_atoms()
                # cluster_rep_ca_atoms = cluster_rep.get_ca_atoms()  # TODO

                rmsd, rot, tx = biopdb_superimposer(frag_ca_atoms, cluster_rep_ca_atoms)

                if rmsd <= min_rmsd and rmsd <= rmsd_thresh:
                    min_rmsd = rmsd
                    min_rot, min_tx = rot, tx
                    min_rmsd_cluster_rep_type = cluster_type

            if min_rmsd_cluster_rep_rot_tx is not None:
                self.type = min_rmsd_cluster_rep_type
                # guide_atoms_pdb = PDB()
                # guide_atoms_pdb.read_atom_list([a1, a2, a3])
                # guide_atoms_pdb.rotate_translate(min_rmsd_cluster_rep_rot_tx[0],
                #                                  min_rmsd_cluster_rep_rot_tx[1])  # *args
                # # coord_rot = self.mat_vec_mul3(rot, coord)
                # # coord_tr = coord_rot + tx
                # self.guide_atoms = guide_atoms_pdb.all_atoms
                # self.guide_coords = guide_atoms_pdb.extract_all_coords()

                t_vec = np.array(min_tx)
                r_mat = np.transpose(np.array(min_rot))

                r_guide_coords = np.matmul(guide_coords, r_mat)
                rt_guide_coords = r_guide_coords + t_vec
                self.guide_coords = rt_guide_coords

    def get_central_res_tup(self):
        return self.central_res_chain_id, self.central_res_num

    def get_guide_coords(self):
        return self.guide_coords

    def get_center_of_mass(self):
        if self.guide_coords is not None:
            return np.matmul(np.array([0.33333, 0.33333, 0.33333]), np.array(self.guide_coords))
        else:
            return None

    def get_type(self):
        return self.type

    def get_pdb(self):
        return self.pdb

    # def get_pdb_coords(self):
    #     return self.pdb_coords

    def get_central_res_num(self):
        return self.central_res_num

    def get_central_res_chain_id(self):
        return self.central_res_chain_id

    def set_pdb(self, pdb):
        self.pdb = pdb
        # self.pdb_coords = pdb.extract_all_coords()
        # self.pdb_coords = pdb.extract_coords()  # TODO

    # def set_guide_atoms(self, guide_coords):
    #     self.guide_coords = guide_coords
    #     a1 = Atom(1, "CA", " ", "GLY", "9", 0, " ", guide_coords[0][0], guide_coords[0][1], guide_coords[0][2], 1.00,
    #               20.00, "C", "")
    #     a2 = Atom(2, "N", " ", "GLY", "9", 0, " ", guide_coords[1][0], guide_coords[1][1], guide_coords[1][2], 1.00,
    #               20.00, "N", "")
    #     a3 = Atom(3, "O", " ", "GLY", "9", 0, " ", guide_coords[2][0], guide_coords[2][1], guide_coords[2][2], 1.00,
    #               20.00, "O", "")
    #     self.guide_atoms = [a1, a2, a3]

    def get_ghost_fragments(self, intfrag_cluster_rep_dict, kdtree_oligomer_backbone, intfrag_cluster_info_dict,
                            clash_dist=2.2):
        if self.type in intfrag_cluster_rep_dict:
            ghost_fragments = []
            for j_type in intfrag_cluster_rep_dict[self.type]:
                for k_type in intfrag_cluster_rep_dict[self.type][j_type]:
                    intfrag = intfrag_cluster_rep_dict[self.type][j_type][k_type]
                    rmsd = intfrag_cluster_info_dict[self.type][j_type][k_type].get_rmsd()
                    intfrag_pdb = intfrag[0]
                    intfrag_mapped_chain_id = intfrag[1]
                    intfrag_mapped_chain_central_res_num = intfrag[2]
                    intfrag_partner_chain_id = intfrag[3]
                    intfrag_partner_chain_central_res_num = intfrag[4]

                    aligned_ghost_frag_pdb = biopdb_aligned_chain(self.pdb, self.pdb.chain_id_list[0], intfrag_pdb,
                                                                  intfrag_mapped_chain_id)

                    # Ghost Fragment Mapped Chain ID, Central Residue Number and Partner Chain ID, Partner Central Residue Number
                    # ghostfrag_central_res_tup = (
                    #     intfrag_mapped_chain_id, intfrag_mapped_chain_central_res_num, intfrag_partner_chain_id,
                    #     intfrag_partner_chain_central_res_num)

                    # Only keep ghost fragments that don't clash with oligomer backbone
                    # Note: guide atoms, mapped chain atoms and non-backbone atoms not included
                    g_frag_bb_coords = []
                    for atom in aligned_ghost_frag_pdb.all_atoms:
                        if atom.chain != "9" and atom.chain != intfrag_mapped_chain_id and atom.is_backbone():
                            g_frag_bb_coords.append([atom.x, atom.y, atom.z])

                    cb_clash_count = kdtree_oligomer_backbone.two_point_correlation(g_frag_bb_coords, [clash_dist])

                    if cb_clash_count[0] == 0:
                        ghost_fragments.append(
                            GhostFragment(aligned_ghost_frag_pdb, self.type, j_type, k_type, rmsd,
                                          self.get_central_res_tup()))  # ghostfrag_central_res_tup,

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
                pdb = PDB()
                pdb.readfile(self.monofrag_cluster_rep_dirpath + "/" + filename, remove_alt_location=True)
                cluster_rep_pdb_dict[os.path.splitext(filename)[0]] = pdb

        self.reps = cluster_rep_pdb_dict
        return cluster_rep_pdb_dict

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
                        ijk_frag_cluster_rep_pdb = PDB()
                        ijk_frag_cluster_rep_pdb.readfile(dirpath1 + "/" + filename)
                        ijk_frag_cluster_rep_mapped_chain_id = filename[
                                                               filename.find("mappedchain") + 12:filename.find(
                                                                   "mappedchain") + 13]
                        ijk_frag_cluster_rep_partner_chain_id = filename[
                                                                filename.find("partnerchain") + 13:filename.find(
                                                                    "partnerchain") + 14]

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

                        i_j_k_intfrag_cluster_rep_dict[i_cluster_type][j_cluster_type][k_cluster_type] = (
                        ijk_frag_cluster_rep_pdb, ijk_frag_cluster_rep_mapped_chain_id,
                        intfrag_mapped_chain_central_res_num, ijk_frag_cluster_rep_partner_chain_id,
                        intfrag_partner_chain_central_res_num)

        self.paired_frags = i_j_k_intfrag_cluster_rep_dict
        return i_j_k_intfrag_cluster_rep_dict

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
        return intfrag_cluster_info_dict
