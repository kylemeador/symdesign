import copy
import os
import pickle
from glob import glob
from itertools import chain, combinations

import numpy as np
from sklearn.neighbors import BallTree

import PathUtils as PUtils
from FragDock import filter_euler_lookup_by_zvalue, calculate_overlap
from PDB import PDB
from SequenceProfile import SequenceProfile
from Structure import Coords
# Globals
from SymDesignUtils import to_iterable, logger
from classes.Fragment import MonoFragment
from utils.ExpandAssemblyUtils import sg_cryst1_fmt_dict, pg_cryst1_fmt_dict, zvalue_dict
from utils.SymmUtils import valid_subunit_number

# Fragment Database Directory Paths
# frag_db = os.path.join(main_script_path, 'data', 'databases', 'fragment_db', 'biological_interfaces')
frag_db = PUtils.frag_directory['biological_interfaces']
monofrag_cluster_rep_dirpath = os.path.join(frag_db, "Top5MonoFragClustersRepresentativeCentered")
ijk_intfrag_cluster_rep_dirpath = os.path.join(frag_db, "Top75percent_IJK_ClusterRepresentatives_1A")
intfrag_cluster_info_dirpath = os.path.join(frag_db, "IJK_ClusteredInterfaceFragmentDBInfo_1A")

# # Create fragment database for all ijk cluster representatives
# ijk_frag_db = FragmentDB(monofrag_cluster_rep_dirpath, ijk_intfrag_cluster_rep_dirpath,
#                          intfrag_cluster_info_dirpath)
# # Get complete IJK fragment representatives database dictionaries
# ijk_monofrag_cluster_rep_pdb_dict = ijk_frag_db.get_monofrag_cluster_rep_dict()
# ijk_intfrag_cluster_rep_dict = ijk_frag_db.get_intfrag_cluster_rep_dict()
# ijk_intfrag_cluster_info_dict = ijk_frag_db.get_intfrag_cluster_info_dict()
#
# # Initialize Euler Lookup Class
# eul_lookup = EulerLookup()

config_directory = PUtils.pdb_db
sym_op_location = PUtils.sym_op_location
# sg_cryst1_fmt_dict = {'F222': 'F 2 2 2', 'P6222': 'P 62 2 2', 'I4132': 'I 41 3 2', 'P432': 'P 4 3 2',
#                       'P6322': 'P 63 2 2', 'I4122': 'I 41 2 2', 'I213': 'I 21 3', 'I422': 'I 4 2 2',
#                       'I432': 'I 4 3 2', 'P4222': 'P 42 2 2', 'F23': 'F 2 3', 'P23': 'P 2 3', 'P213': 'P 21 3',
#                       'F432': 'F 4 3 2', 'P622': 'P 6 2 2', 'P4232': 'P 42 3 2', 'F4132': 'F 41 3 2',
#                       'P4132': 'P 41 3 2', 'P422': 'P 4 2 2', 'P312': 'P 3 1 2', 'R32': 'R 3 2'}
# pg_cryst1_fmt_dict = {'p3': 'P 3', 'p321': 'P 3 2 1', 'p622': 'P 6 2 2', 'p4': 'P 4', 'p222': 'P 2 2 2',
#                       'p422': 'P 4 2 2', 'p4212': 'P 4 21 2', 'p6': 'P 6', 'p312': 'P 3 1 2', 'c222': 'C 2 2 2'}
# zvalue_dict = {'P 2 3': 12, 'P 42 2 2': 8, 'P 3 2 1': 6, 'P 63 2 2': 12, 'P 3 1 2': 12, 'P 6 2 2': 12, 'F 2 3': 48,
#                'F 2 2 2': 16, 'P 62 2 2': 12, 'I 4 2 2': 16, 'I 21 3': 24, 'R 3 2': 6, 'P 4 21 2': 8, 'I 4 3 2': 48,
#                'P 41 3 2': 24, 'I 41 3 2': 48, 'P 3': 3, 'P 6': 6, 'I 41 2 2': 16, 'P 4': 4, 'C 2 2 2': 8,
#                'P 2 2 2': 4, 'P 21 3': 12, 'F 41 3 2': 96, 'P 4 2 2': 8, 'P 4 3 2': 24, 'F 4 3 2': 96,
#                'P 42 3 2': 24}


class Model:  # (PDB)
    """Keep track of different variations of the same PDB object whether they be mutated in sequence or have
    their coordinates perturbed
    """
    def __init__(self, pdb=None, models=None):
        # if isinstance(pdb, list):
        # self.models = pdb  # list or dict of PDB objects
        # self.pdb = self.models[0]
        # elif isinstance(pdb, PDB):
        self.pdb = None
        self.models = None

    def set_models(self, models):
        self.models = models
        self.pdb = self.models[0]

    def add_pdb(self, pdb):
        self.models.append(pdb)

    # def add_atoms_to_pdb(self, index=0, atoms=None):
    #     """Add atoms to a PDB object in the model. Zero indexed"""
    #     self.models[index].read_atom_list(atoms)

    def get_ca_atoms(self):
        return [pdb.get_ca_atoms() for pdb in self.models]

    def chain(self, chain_id):
        return [pdb.chain(chain_id) for pdb in self.models]

    def extract_all_coords(self):  # TODO
        model_coords = [pdb.extract_coords() for pdb in self.models]
        return np.array(model_coords)  # arr.reshape(-1, arr.shape[-1])

    def extract_backbone_coords(self):  # TODO
        return [pdb.extract_backbone_coords() for pdb in self.models]

    def extract_backbone_and_cb_coords(self):  # TODO
        return [pdb.extract_backbone_and_cb_coords() for pdb in self.models]

    def extract_CA_coords(self):  # TODO
        return [pdb.extract_CA_coords() for pdb in self.models]

    def extract_CB_coords(self, InclGlyCA=False):  # TODO
        return [pdb.extract_CB_coords(InclGlyCA=InclGlyCA) for pdb in self.models]

    def extract_CB_coords_chain(self, chain_id, InclGlyCA=False):  # TODO
        return [pdb.extract_CB_coords_chain(chain_id, InclGlyCA=InclGlyCA) for pdb in self.models]

    def get_CB_coords(self, ReturnWithCBIndices=False, InclGlyCA=False):  # TODO
        return [pdb.get_CB_coords(ReturnWithCBIndices=ReturnWithCBIndices, InclGlyCA=InclGlyCA) for pdb in self.models]

    def set_atom_coordinates(self, new_coords):  # Todo
        for i, pdb in enumerate(self.models):
            pdb.coords = Coords(new_coords[i])
        # return [pdb.set_atom_coordinates(new_cords[i]) for i, pdb in enumerate(self.models)]


class SymmetricModel:  # (Model)
    def __init__(self, models=None, symmetry=None, **kwargs):
        super().__init__(**kwargs)
        # super().__init__()  # for Model Todo should I have this?
        self.pdb = None  # Model
        self.models = []  # Model
        self.asu = None  # the pose specific asu
        self.coords = []
        self.model_coords = []
        self.symmetry = None  # symmetry  # also defined in PDB as self.space_group
        self.dimension = None
        self.uc_dimensions = None  # also defined in PDB
        self.expand_matrices = None  # Todo make expand_matrices numpy
        self.number_of_models = None

    def set_symmetry(self, cryst1=None, uc_dimensions=None, symmetry=None, generate_assembly=True):
        """Set the model symmetry using the CRYST1 record, or the unit cell dimensions and the Hermann–Mauguin symmetry
        notation (in CRYST1 format, ex P 4 3 2) for the Model assembly. If the assembly is a point group,
        only the symmetry is required"""
        if cryst1:
            self.uc_dimensions, self.symmetry = PDB.parse_cryst_record(cryst1_string=cryst1)
        else:
            if uc_dimensions:
                self.uc_dimensions = uc_dimensions

            if symmetry:
                self.symmetry = symmetry
                # self.symmetry = ''.join(symmetry.split())  # ensure the symmetry is NO SPACES Hermann–Mauguin notation

        if self.symmetry in ['T', 'O', 'I']:
            self.dimension = 0
            self.expand_matrices = self.get_ptgrp_sym_op(self.symmetry)  # Todo numpy expand_matrices
            # self.expand_matrices = np.array(self.get_ptgrp_sym_op(self.symmetry))
        else:
            if self.symmetry in pg_cryst1_fmt_dict.values():  # not available yet for non-Nanohedra PG's
                self.dimension = 2
            elif self.symmetry in sg_cryst1_fmt_dict.values():  # not available yet for non-Nanohedra SG's
                self.dimension = 3
            self.expand_matrices = self.get_sg_sym_op(''.join(self.symmetry.split()))
            # self.expand_matrices = np.array(self.get_sg_sym_op(self.symmetry))  # Todo numpy expand_matrices

        if generate_assembly:
            self.generate_symmetric_assembly()

    def generate_symmetric_assembly(self, return_side_chains=True, surrounding_uc=False, generate_symmetry_mates=False):
        """Expand an asu in self.pdb using self.symmetry for the symmetry specification, and optional unit cell
        dimensions if self.dimension > 0. Expands assembly to complete point group, or the unit cell

        Keyword Args:
            return_side_chains=True (bool): Whether to return all side chain atoms. False returns backbone and CB atoms
            surrounding_uc=False (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        """
        if self.dimension == 0:  # symmetry in ['T', 'O', 'I']: Todo add other point groups
            self.get_point_group_coords(return_side_chains=return_side_chains)
        else:
            self.expand_uc_coords(return_side_chains=return_side_chains, surrounding_uc=surrounding_uc)

        if generate_symmetry_mates:
            self.get_assembly_symmetry_mates()

    def expand_uc_coords(self, surrounding_uc, **kwargs):  # return_side_chains=False, surrounding_uc=False
        """Expand the backbone coordinates for every symmetric copy within the unit cells surrounding a central cell
        """
        self.get_unit_cell_coords(**kwargs)  # # return_side_chains=return_side_chains
        if surrounding_uc:
            self.get_surrounding_unit_cell_coords()  # return_side_chains=return_side_chains FOR Coord expansion
            # self.get_surrounding_unit_cell_coords()  # return_side_chains=return_side_chains FOR PDB expansion

    def cart_to_frac(self, cart_coords):
        """Takes a numpy array of coordinates (and finds the fractional coordinates from cartesian coordinates
        From http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm

        Returns:
            (Numpy.array):
        """
        if not self.uc_dimensions:
            return None

        a2r = np.pi / 180.0
        a, b, c, alpha, beta, gamma = self.uc_dimensions
        alpha *= a2r
        beta *= a2r
        gamma *= a2r

        # unit cell volume
        v = a * b * c * np.sqrt((1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 +
                                 2 * (np.cos(alpha) * np.cos(beta) * np.cos(gamma))))

        # deorthogonalization matrix M
        m0 = [1 / a, -(np.cos(gamma) / float(a * np.sin(gamma))),
              (((b * np.cos(gamma) * c * (np.cos(alpha) - (np.cos(beta) * np.cos(gamma)))) / float(np.sin(gamma))) -
               (b * c * np.cos(beta) * np.sin(gamma))) * (1 / float(v))]
        m1 = [0, 1 / (b * np.sin(gamma)),
              -((a * c * (np.cos(alpha) - (np.cos(beta) * np.cos(gamma)))) / float(v * np.sin(gamma)))]
        m2 = [0, 0, (a * b * np.sin(gamma)) / float(v)]
        m = np.array([m0, m1, m2])

        return np.matmul(cart_coords, np.transpose(m))

    def frac_to_cart(self, frac_coords):
        """Takes a numpy array of coordinates and finds the cartesian coordinates from fractional coordinates
        From http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm
        """
        if not self.uc_dimensions:
            return None

        a2r = np.pi / 180.0
        a, b, c, alpha, beta, gamma = self.uc_dimensions
        alpha *= a2r
        beta *= a2r
        gamma *= a2r

        # unit cell volume
        v = a * b * c * np.sqrt((1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 +
                                 2 * (np.cos(alpha) * np.cos(beta) * np.cos(gamma))))

        # orthogonalization matrix m_inv
        m_inv_0 = [a, b * np.cos(gamma), c * np.cos(beta)]
        m_inv_1 = [0, b * np.sin(gamma),
                   (c * (np.cos(alpha) - (np.cos(beta) * np.cos(gamma)))) / float(np.sin(gamma))]
        m_inv_2 = [0, 0, v / float(a * b * np.sin(gamma))]
        m_inv = np.array([m_inv_0, m_inv_1, m_inv_2])

        return np.matmul(frac_coords, np.transpose(m_inv))

    def get_point_group_coords(self, return_side_chains=True):
        """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
        self.number_of_models = valid_subunit_number[self.symmetry]
        if return_side_chains:  # get different function calls depending on the return type
            get_pdb_coords = getattr(PDB, 'get_coords')
        else:
            get_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')

        self.coords = get_pdb_coords(self.asu)
        self.model_coords = np.empty((len(self.coords) * self.number_of_models, 3), dtype=float)
        # self.model_coords[:len(self.coords)] = self.coords
        for idx, rot in enumerate(self.expand_matrices):  # Todo numpy incomming expand_matrices
            r_mat = np.transpose(np.array(rot))
            r_asu_coords = np.matmul(self.coords, r_mat)
            self.model_coords[idx * len(self.coords): (idx * len(self.coords)) + len(self.coords)] = r_asu_coords
        # self.model_coords = np.array(self.model_coords)

    def get_unit_cell_coords(self, return_side_chains=True):
        """Generates unit cell coordinates for a symmetry group. Modifies model_coords to include all in a unit cell"""
        # self.models = [self.asu]
        self.number_of_models = zvalue_dict[self.symmetry]
        if return_side_chains:  # get different function calls depending on the return type
            get_pdb_coords = getattr(PDB, 'get_coords')
        else:
            get_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')

        self.coords = get_pdb_coords(self.asu)
        # asu_cart_coords = get_pdb_coords(self.pdb)
        # asu_cart_coords = self.pdb.get_coords()  # returns a numpy array
        # asu_frac_coords = self.cart_to_frac(np.array(asu_cart_coords))
        asu_frac_coords = self.cart_to_frac(self.coords)
        self.model_coords = np.empty((len(self.coords) * self.number_of_models, 3), dtype=float)
        for idx, (rot, tx) in enumerate(self.expand_matrices):
            t_vec = np.array(tx)  # Todo numpy incomming expand_matrices
            # t_vec = t
            r_mat = np.transpose(np.array(rot))
            # r_mat = np.transpose(r)

            r_asu_frac_coords = np.matmul(asu_frac_coords, r_mat)
            rt_asu_frac_coords = r_asu_frac_coords + t_vec
            self.model_coords[idx * len(self.coords): (idx * len(self.coords)) + len(self.coords)] = \
                self.frac_to_cart(rt_asu_frac_coords)

            # self.model_coords.extend(self.frac_to_cart(rt_asu_frac_coords).tolist())
        # self.model_coords = np.array(self.model_coords)

    def get_surrounding_unit_cell_coords(self):
        """Generates a grid of unit cell coordinates for a symmetry group. Modifies model_coords from a unit cell
        representation to a grid of unit cells, either 3x3 for a layer group or 3x3x3 for a space group"""
        if self.dimension == 3:
            z_shifts, uc_number = [-1, 0, 1], 9
        elif self.dimension == 2:
            z_shifts, uc_number = [0], 27
        else:
            return None

        central_uc_frac_coords = self.cart_to_frac(self.model_coords)
        uc_coord_length = self.model_coords.shape[0]
        self.model_coords = np.empty((uc_coord_length * uc_number, 3), dtype=float)
        idx = 0
        for x_shift in [-1, 0, 1]:
            for y_shift in [-1, 0, 1]:
                for z_shift in z_shifts:
                    # add central uc_coords to the model coords after applying the correct tx of frac coords & convert
                    self.model_coords[(idx * uc_coord_length): (idx * uc_coord_length) + uc_coord_length] = \
                        self.frac_to_cart(central_uc_frac_coords + [x_shift, y_shift, z_shift])
                    idx += 1

        self.number_of_models = zvalue_dict[self.symmetry] * uc_number

    def get_assembly_symmetry_mates(self, return_side_chains=True):  # For getting PDB copies
        """Return all symmetry mates as a list of PDB objects. Chain names will match the ASU"""
        if return_side_chains:  # get different function calls depending on the return type
            extract_pdb_atoms = getattr(PDB, 'get_atoms')
            # extract_pdb_coords = getattr(PDB, 'extract_coords')
        else:
            extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
            # extract_pdb_coords = getattr(PDB, 'extract_backbone_and_cb_coords')

        # self.models.append(copy.copy(self.asu))
        # prior_idx = self.asu.number_of_atoms  # TODO modify by extract_pdb_atoms!
        for model_idx in range(self.number_of_models):  # range(1,
            symmetry_mate_pdb = copy.copy(self.asu)
            symmetry_mate_pdb.coords = Coords(self.model_coords[(model_idx * self.asu.number_of_atoms):
                                                                ((model_idx + 1) * self.asu.number_of_atoms)])
            # symmetry_mate_pdb.set_atom_coordinates(self.model_coords[(model_idx * self.asu.number_of_atoms):
            #                                                          (model_idx * self.asu.number_of_atoms)
            #                                                          + self.asu.number_of_atoms])
            self.models.append(symmetry_mate_pdb)

    def find_asu_equivalent_symmetry_mate(self, residue_query_number=1):
        """Find the asu equivalent model in the SymmetricModel. Zero-indexed

        model_coords must be from all atoms which by default is True
        """
        if not self.symmetry:
            return None

        template_atom_coords = self.asu.residue(residue_query_number).ca.coords
        template_atom_index = self.asu.residue(residue_query_number).ca.index
        for model_number in range(self.number_of_models):
            if template_atom_coords == self.model_coords[(model_number * len(self.coords)) + template_atom_index]:
                #                      self.pdb.number_of_atoms
                return model_number

        print('%s is FAILING' % self.find_asu_equivalent_symmetry_mate.__name__)

    def return_symmetry_mates(self, pdb, **kwargs):  # return_side_chains=False, surrounding_uc=False):
        """Expand an asu in self.pdb using self.symmetry for the symmetry specification, and optional unit cell
        dimensions if self.dimension > 0. Expands assembly to complete point group, or the unit cell

        Keyword Args:
            return_side_chains=True (bool): Whether to return all side chain atoms. False gives backbone and CB atoms
            surrounding_uc=False (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        """
        if self.dimension == 0:
            return self.return_point_group_symmetry_mates(pdb, **kwargs)  # return_side_chains=return_side_chains)
        else:
            return self.return_crystal_symmetry_mates(pdb, **kwargs)  # return_side_chains=return_side_chains,
            #                                                           surrounding_uc=surrounding_uc)

    def return_crystal_symmetry_mates(self, pdb, surrounding_uc, **kwargs):  # return_side_chains=False, surrounding_uc=False
        """Expand the backbone coordinates for every symmetric copy within the unit cells surrounding a central cell
        """
        if surrounding_uc:
            self.return_surrounding_unit_cell_symmetry_mates(pdb, **kwargs)  # return_side_chains=return_side_chains FOR Coord expansion
        else:
            self.return_unit_cell_symmetry_mates(pdb, **kwargs)  # # return_side_chains=return_side_chains

    # @staticmethod
    def return_point_group_symmetry_mates(self, pdb, return_side_chains=True):  # For returning PDB copies
        """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
        if return_side_chains:  # get different function calls depending on the return type
            extract_pdb_atoms = getattr(PDB, 'get_atoms')  # Not using. The copy() versus PDB() changes residue objs
            extract_pdb_coords = getattr(PDB, 'extract_coords')
        else:
            extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
            extract_pdb_coords = getattr(PDB, 'extract_backbone_and_cb_coords')

        asu_symm_mates = []
        pdb_coords = extract_pdb_coords(pdb)
        for rot in self.expand_matrices:
            r_mat = np.transpose(np.array(rot))  # Todo numpy incomming expand_matrices
            r_asu_coords = np.matmul(pdb_coords, r_mat)
            symmetry_mate_pdb = copy.copy(pdb)
            symmetry_mate_pdb.coords = Coords(r_asu_coords)
            # symmetry_mate_pdb.set_atom_coordinates(r_asu_coords)
            asu_symm_mates.append(symmetry_mate_pdb)

        return asu_symm_mates

    def return_unit_cell_symmetry_mates(self, pdb, return_side_chains=True):  # For returning PDB copies
        """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
        if return_side_chains:  # get different function calls depending on the return type
            extract_pdb_atoms = getattr(PDB, 'get_atoms')  # Not using. The copy() versus PDB() changes residue objs
            extract_pdb_coords = getattr(PDB, 'extract_coords')
        else:
            extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
            extract_pdb_coords = getattr(PDB, 'extract_backbone_and_cb_coords')

        # asu_cart_coords = self.pdb.get_coords()  # returns a numpy array
        asu_cart_coords = extract_pdb_coords(pdb)
        # asu_frac_coords = self.cart_to_frac(np.array(asu_cart_coords))
        asu_frac_coords = self.cart_to_frac(asu_cart_coords)
        asu_symm_mates = []
        for r, t in self.expand_matrices:
            t_vec = np.array(t)  # Todo numpy incomming expand_matrices
            r_mat = np.transpose(np.array(r))

            r_asu_frac_coords = np.matmul(asu_frac_coords, r_mat)
            tr_asu_frac_coords = r_asu_frac_coords + t_vec
            tr_asu_cart_coords = self.frac_to_cart(tr_asu_frac_coords)  # We want numpy array, not .tolist()

            symmetry_mate_pdb = copy.copy(pdb)
            symmetry_mate_pdb.coords = Coords(tr_asu_cart_coords)
            # symmetry_mate_pdb.set_atom_coordinates(tr_asu_cart_coords)
            asu_symm_mates.append(symmetry_mate_pdb)

        return asu_symm_mates

    def return_surrounding_unit_cell_symmetry_mates(self, pdb, return_side_chains=True):  # For returning PDB copies
        """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
        if return_side_chains:  # get different function calls depending on the return type
            extract_pdb_atoms = getattr(PDB, 'get_atoms')  # Not using. The copy() versus PDB() changes residue objs
            extract_pdb_coords = getattr(PDB, 'extract_coords')
        else:
            extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
            extract_pdb_coords = getattr(PDB, 'extract_backbone_and_cb_coords')

        # could move the next block to a get all uc frac coords and remove redundancy from here and return_uc_sym_mates
        asu_cart_coords = extract_pdb_coords(pdb)
        asu_frac_coords = self.cart_to_frac(asu_cart_coords)
        all_uc_frac_coords = []
        for r, t in self.expand_matrices:
            t_vec = np.array(t)  # Todo numpy incomming expand_matrices
            r_mat = np.transpose(np.array(r))

            r_asu_frac_coords = np.matmul(asu_frac_coords, r_mat)
            tr_asu_frac_coords = r_asu_frac_coords + t_vec
            all_uc_frac_coords.extend(tr_asu_frac_coords)
            # tr_asu_cart_coords = self.frac_to_cart(tr_asu_frac_coords)  # We want numpy array, not .tolist()

        if self.dimension == 3:
            z_shifts, uc_number = [-1, 0, 1], 9
        elif self.dimension == 2:
            z_shifts, uc_number = [0], 27
        else:
            return None

        number_coords = len(asu_frac_coords)
        asu_symm_mates = []
        uc_idx = 0
        for x_shift in [-1, 0, 1]:
            for y_shift in [-1, 0, 1]:
                for z_shift in z_shifts:
                    symmetry_mate_pdb = copy.copy(pdb)
                    symmetry_mate_pdb.coords = Coords(self.frac_to_cart(all_uc_frac_coords
                                                                        [(uc_idx * number_coords):
                                                                         ((uc_idx + 1) * number_coords)]
                                                                        + [x_shift, y_shift, z_shift]))
                    # symmetry_mate_pdb.set_atom_coordinates(self.frac_to_cart(all_uc_frac_coords
                    #                                                          [(uc_idx * number_coords):
                    #                                                           (uc_idx * number_coords) + number_coords]
                    #                                                          + [x_shift, y_shift, z_shift]))
                    asu_symm_mates.append(symmetry_mate_pdb)

        assert len(asu_symm_mates) == uc_number * zvalue_dict[self.symmetry], \
            'Number of models %d is incorrect! Should be %d' % (len(asu_symm_mates), uc_number *
                                                                zvalue_dict[self.symmetry])

        return asu_symm_mates

    # def get_point_group_symmetry_mates(self, return_side_chains=False):  # For getting PDB copies
    #     """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
    #     if return_side_chains:  # get different function calls depending on the return type
    #         extract_pdb_atoms = getattr(PDB, 'get_atoms')
    #         # extract_pdb_coords = getattr(PDB, 'extract_coords')
    #     else:
    #         extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
    #         # extract_pdb_coords = getattr(PDB, 'extract_backbone_and_cb_coords')
    #
    #     self.models.append(copy.copy(self.asu))
    #     prior_idx = self.asu.number_of_atoms  # TODO modify by extract_pdb_atoms!
    #     for model in range(1, self.number_of_models):
    #         symmetry_mate_pdb = copy.copy(self.asu)
    #         symmetry_mate_pdb.set_atom_coordinates(self.model_coords[prior_idx: prior_idx + self.asu.number_of_atoms])
    #         self.models.append(symmetry_mate_pdb)
    #
    # def get_unit_cell_symmetry_mates(self, return_side_chains=False):  # For getting PDB copies # TODO model after PG ^
    #     """Return all symmetry mates as a list of PDB objects. Chain names will match the ASU"""
    #     if return_side_chains:  # get different function calls depending on the return type
    #         extract_pdb_atoms = getattr(PDB, 'get_atoms')
    #         # extract_pdb_coords = getattr(PDB, 'extract_coords')
    #     else:
    #         extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
    #         # extract_pdb_coords = getattr(PDB, 'extract_backbone_and_cb_coords')
    #
    #     # self.models.append(copy.copy(self.asu))
    #     # prior_idx = self.asu.number_of_atoms  # TODO modify by extract_pdb_atoms!
    #     for model_idx in range(self.number_of_models):  # range(1,
    #         symmetry_mate_pdb = copy.copy(self.asu)
    #         symmetry_mate_pdb.set_atom_coordinates(self.model_coords[(model_idx * self.asu.number_of_atoms):
    #                                                                  (model_idx * self.asu.number_of_atoms)
    #                                                                  + self.asu.number_of_atoms])
    #         self.models.append(symmetry_mate_pdb)
    #
    # def get_surrounding_unit_cell_symmetry_mates(self):  # For getting PDB copies  # TODO model after UC ^
    #     """Returns a grid of unit cells for a symmetry group. Each unit cell is a list of ASU's in total grid list"""
    #     # self.number_of_models = zvalue_dict[self.symmetry] * uc_number  # Todo this figure out in context
    #     if self.dimension == 3:
    #         z_shifts, uc_number = [-1, 0, 1], 9
    #     elif self.dimension == 2:
    #         z_shifts, uc_number = [0], 27
    #     else:
    #         return None
    #
    #     # if return_side_chains:  # get different function calls depending on the return type
    #     #     extract_pdb_atoms = getattr(PDB, 'get_atoms')
    #     #     extract_pdb_coords = getattr(PDB, 'extract_coords')
    #     # else:
    #     #     extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
    #     #     extract_pdb_coords = getattr(PDB, 'extract_backbone_and_cb_coords')
    #
    #     asu_atom_template = PDB.from_atoms(self.models[0].get_atoms())  # Todo may need some other mods with return_side_chains gone
    #     # asu_atom_template = extract_pdb_atoms(self.pdb)
    #     # asu_bb_atom_template = uc_sym_mates[0].get_backbone_atoms()
    #
    #     central_uc_cart_coords = extract_pdb_coords(self.models)
    #     # central_uc_cart_coords = []
    #     # for unit_cell_sym_mate_pdb in self.uc_sym_mates:
    #     #     central_uc_cart_coords.extend(extract_pdb_coords(unit_cell_sym_mate_pdb))
    #     #     # central_uc_bb_cart_coords.extend(unit_cell_sym_mate_pdb.extract_backbone_coords())
    #     central_uc_frac_coords = self.cart_to_frac(central_uc_cart_coords)  # Todo flatten list?
    #     # central_uc_frac_coords = self.cart_to_frac(np.array(central_uc_cart_coords))
    #
    #     all_surrounding_uc_frac_coords = []
    #     for x_shift in [-1, 0, 1]:
    #         for y_shift in [-1, 0, 1]:
    #             for z_shift in z_shifts:
    #                 if [x_shift, y_shift, z_shift] != [0, 0, 0]:
    #                     all_surrounding_uc_frac_coords.extend(central_uc_frac_coords + [x_shift, y_shift, z_shift])
    #
    #     all_surrounding_uc_cart_coords = np.split(self.frac_to_cart(np.array(all_surrounding_uc_frac_coords)),
    #                                               number_of_models)
    #     # all_surrounding_uc_cart_coords = np.split(all_surrounding_uc_cart_coords, number_of_models)
    #
    #     for surrounding_uc_cart_coords in all_surrounding_uc_cart_coords:
    #         all_uc_sym_mates_cart_coords = np.split(surrounding_uc_cart_coords, len(self.models))
    #         # one_surrounding_unit_cell = []
    #         for uc_sym_mate_cart_coords in all_uc_sym_mates_cart_coords:
    #             uc_sym_mate_pdb = copy.copy(asu_atom_template)
    #             uc_sym_mate_pdb.set_atom_coordinates(uc_sym_mate_cart_coords)
    #
    #             # uc_sym_mate_atoms = []
    #             # for atom_count, atom in enumerate(asu_atom_template):
    #             #     x_transformed = uc_sym_mate_cart_coords[atom_count][0]
    #             #     y_transformed = uc_sym_mate_cart_coords[atom_count][1]
    #             #     z_transformed = uc_sym_mate_cart_coords[atom_count][2]
    #             #     atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
    #             #                             atom.get_residue_type(), atom.get_chain(),
    #             #                             atom.get_residue_number(),
    #             #                             atom.get_code_for_insertion(), x_transformed, y_transformed,
    #             #                             z_transformed,
    #             #                             atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
    #             #                             atom.get_atom_charge())
    #             #     uc_sym_mate_atoms.append(atom_transformed)
    #
    #             # uc_sym_mate_pdb = PDB(atoms=uc_sym_mate_atoms)
    #             # one_surrounding_unit_cell.append(uc_sym_mate_pdb)
    #
    #             self.models.extend(uc_sym_mate_pdb)
    #         # self.models.extend(one_surrounding_unit_cell)

    def write(self, name, location=os.getcwd(), cryst1=None):  # Todo write model, write symmetry
        out_path = os.path.join(location, '%s.pdb' % name)
        with open(out_path, 'w') as f:
            if cryst1 and isinstance(cryst1, str) and cryst1.startswith('CRYST1'):
                f.write('%s\n' % cryst1)
            for i, model in enumerate(self.models, 1):
                f.write('{:9s}{:>4d}\n'.format('MODEL', i))
                for _chain in model.chain_id_list:
                    chain_atoms = model.get_chain_atoms(_chain)
                    f.write(''.join(str(atom) for atom in chain_atoms))
                    f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', chain_atoms[-1].number + 1,
                                                                          chain_atoms[-1].residue_type, _chain,
                                                                          chain_atoms[-1].residue_number))
                # f.write(''.join(str(atom) for atom in model.atoms))
                # f.write('\n'.join(str(atom) for atom in model.atoms))
                f.write('ENDMDL\n')

    @staticmethod
    def get_ptgrp_sym_op(sym_type, expand_matrix_dir=os.path.join(sym_op_location, 'POINT_GROUP_SYMM_OPERATORS')):
        expand_matrix_filepath = os.path.join(expand_matrix_dir, sym_type)
        with open(expand_matrix_filepath, "r") as expand_matrix_f:
            expand_matrix_lines = expand_matrix_f.readlines()

        # Todo pickle these
        line_count = 0
        expand_matrices = []
        mat = []
        for line in expand_matrix_lines:
            line = line.split()
            if len(line) == 3:
                line_float = [float(s) for s in line]
                mat.append(line_float)
                line_count += 1
                if line_count % 3 == 0:
                    expand_matrices.append(mat)
                    mat = []

        return expand_matrices

    @staticmethod
    def get_sg_sym_op(sym_type, expand_matrix_dir=os.path.join(sym_op_location, 'SPACE_GROUP_SYMM_OPERATORS')):
        sg_op_filepath = os.path.join(expand_matrix_dir, sym_type + '.pickle')
        with open(sg_op_filepath, "rb") as sg_op_file:
            sg_sym_op = pickle.load(sg_op_file)

        return sg_sym_op


class Pose(Model, SymmetricModel, SequenceProfile):  # PDB Todo get rid of Model? could be imported from SymmetricModel
    """A Pose is made of multiple PDB objects all sharing a common feature such as symmetric copies or modifications to
    the PDB sequence
    """
    def __init__(self, asu=None, pdb=None, pdb_file=None, asu_file=None, symmetry=None, **kwargs):
        # super().__init__(**kwargs)
        super().__init__()

        if pdb:
            self.pdb = pdb
            # self.set_pdb(pdb)
            # self.initialize_symmetry()

        if pdb_file:
            self.pdb = PDB.from_file(pdb_file)
            # Depending on the extent of PDB class initialization, I could copy the PDB info into self.pdb
            # this would be:
            # coords, atoms, residues, chains, entities, design, (host of others read from file)
            # self.set_pdb(pdb)
            # self.initialize_symmetry()

        if asu:  # Todo ensure a Structure/PDB object
            self.asu = asu
            self.pdb = self.asu
            # self.initialize_symmetry()

        if asu_file:
            self.asu = PDB.from_file(asu_file)
            # Depending on the extent of PDB class initialization, I could copy the PDB info into self.pdb
            # this would be:
            # coords, atoms, residues, chains, entities, design, (host of others read from file)
            # self.set_pdb(pdb)
            self.pdb = self.asu
            # self.initialize_symmetry()

        self.initialize_symmetry()

        # super().__init__()  # structure=self)  # SequenceProfile init
        # self.pdb = None  # the pose specific pdb  # Model
        # self.models = []  # from Model or SymmetricModel
        self.pdbs = []  # the member pdbs which make up the pose
        # self.entities = []  # from PDB
        self.pdbs_d = {}
        self.pose_pdb_accession_map = {}

        # self.uc_sym_mates = []
        # self.surrounding_uc_sym_mates = []

        # # Model
        # super().__init__()  # Model init, PDB? init is handled above for all Super Classes
        # # self.pdb = None  # Model
        # # self.models = []  # Model

        # super().__init__()  # SymmetricModel init is handled above
        # if symmetry:
        #     self.set_symmetry(symmetry=symmetry)
        # self.generate_symmetric_assembly(return_side_chains=False, surrounding_uc=False)
        # SymmetricModel
        # self.asu = None  # the pose specific asu
        self.symmetric_assembly = Model()  # the symmetry expanded attribute of self.asu
        # self.coords = []
        # self.model_coords = []
        # self.symmetry = symmetry  # also in PDB as self.space_group
        # self.dimension = None
        # self.uc_dimensions = None
        # self.expand_matrices = None
        # self.number_of_models = None

    @classmethod
    def from_pdb(cls, pdb):
        return cls(pdb=pdb)

    @classmethod
    def pdb_from_file(cls, pdb_file):
        return cls(pdb_file=pdb_file)

    @classmethod
    def from_asu(cls, asu):
        return cls(asu=asu)

    @classmethod
    def asu_from_file(cls, asu_file):
        return cls(asu_file=asu_file)

    # @classmethod  # In PDB class
    # def from_file(cls, file):
    #     return cls(file=file)

    # @property
    # def asu(self):
    #     return self._asu
    #
    # @asu.setter
    # def asu(self, pdb):
    #     self_.asu = pdb
    #
    # @property
    # def pdb(self):
    #     return self._pdb
    #
    # @pdb.setter
    # def pdb(self, pdb):
    #     self._pdb = pdb

    @property
    def entities(self):
        return self.pdb.entities

    @entities.setter
    def entities(self, entities):
        self.pdb.entities = entities

    def add_pdb(self, pdb):
        """Add a PDB to the PosePDB as well as the member PDB list"""
        self.pdbs.append(pdb)
        self.pdbs_d[pdb.name] = pdb
        # self.pdbs_d[id(pdb)] = pdb
        self.add_entities_to_pose(self, pdb)
        # Todo turn multiple PDB's into one structure representative
        # self.pdb.add_atoms(pdb.get_atoms())

    def add_entities_to_pose(self, pdb):
        """Add each unique entity in a pdb to the pose, updating all metadata"""
        # self.pose_pdb_accession_map[pdb.name] = pdb.entity_accession_map
        self.pose_pdb_accession_map[pdb.name] = pdb.entity_d
        # for entity in pdb.accession_entity_map:
        for idx, entity in enumerate(pdb.entities):
            self.add_entity(entity, name='%s_%d' % (pdb.name, idx))

    def add_entity(self, entity, name=None):
        # Todo Fix this garbage... Entity()
        self.entities = None
        entity_chain = pdb.entities[entity]['representative']
        self.asu.add_atoms(entity.get_atoms())

    def initialize_symmetry(self):
        if self.pdb.space_group and self.pdb.uc_dimensions:
            self.set_symmetry(symmetry=self.pdb.space_group, uc_dimensions=self.pdb.uc_dimensions)
        elif self.pdb.cryst_record:
            self.set_symmetry(cryst1=self.pdb.cryst_record)
        else:
            print('No symmetry present in the Pose PDB')

    def construct_cb_atom_tree(self, entity1, entity2, distance=8):  # TODO UNUSED
        """Create a atom tree using CB atoms from two PDB's

        Args:
            pdb1 (PDB): First PDB to query against
            pdb2 (PDB): Second PDB which will be tested against pdb1
        Keyword Args:
            distance=8 (int): The distance to query in Angstroms
            include_glycine=True (bool): Whether glycine CA should be included in the tree
        Returns:
            query (list()): sklearn query object of pdb2 coordinates within dist of pdb1 coordinates
            pdb1_cb_indices (list): List of all CB indices from pdb1
            pdb2_cb_indices (list): List of all CB indices from pdb2
        """
        # Get CB Atom Coordinates including CA coordinates for Gly residues
        entity1_indecies = np.array(self.asu.entity(entity1).get_cb_indices())
        # mask = np.ones(self.asu.number_of_atoms, dtype=int)  # mask everything
        # mask[index_array] = 0  # we unmask the useful coordinates
        entity1_coords = self.coords[entity1_indecies]  # only get the coordinate indices we want!

        entity2_indecies = np.array(self.asu.entity(entity2).get_cb_indices())
        entity2_coords = self.coords[entity2_indecies]  # only get the coordinate indices we want!

        # Construct CB Tree for PDB1
        entity1_tree = BallTree(entity1_coords)

        # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
        return entity1_tree.query_radius(entity2_coords, distance)

    def find_interface_pairs(self, entity1=None, entity2=None, distance=8, include_glycine=True):
        """Get pairs of residues across an interface within a certain distance. Symmetry aware

        If symmetry is used, entity2 needs to be symmeterized with all atomic coordinates, by default this is True
        Keyword Args:
            entity1=None (str): First entity name to measure interface between
            entity2=None (str): Second entity name to measure interface between
            distance=8 (int): The distance to query the interface in Angstroms
            include_glycine=True (bool): Whether glycine CA should be included in the tree
        Returns:
            interface_pairs (list(tuple): A list of interface residue numbers across the interface
        """
        # entity2_query = construct_cb_atom_tree(entity1, entity2, distance=distance)

        # Get CB Atom Coordinates including CA coordinates for Gly residues
        entity1_atoms = self.asu.entity(entity1).get_atoms()  # if passing by name
        # entity1_atoms = entity1.get_atoms()  # if passing by Structure object
        entity1_indecies = np.array(self.asu.entity(entity1).get_cb_indices(InclGlyCA=include_glycine))
        entity1_coords = self.coords[entity1_indecies]  # only get the coordinate indices we want!

        entity2_atoms = self.asu.entity(entity2).get_atoms()  # if passing by name
        # entity2_atoms = entity2.get_atoms()  # if passing by Structure object
        entity2_indecies = np.array(self.asu.entity(entity2).get_cb_indices(InclGlyCA=include_glycine))
        if self.symmetry:  # self.model_coords:
            # get all symmetric indices if the pose is symmetric
            entity2_indecies = np.array([idx + (self.asu.number_of_atoms * model_number)
                                         for model_number in range(self.number_of_models) for idx in entity2_indecies])
            # Todo mask=[residue_numbers?] default parameter
            if entity2 == entity1:  # the entity is the same however, we don't want interactions with the same sym mate
                model_number = self.find_asu_equivalent_symmetry_mate()
                start_idx = self.asu.number_of_atoms * model_number
                end_idx = self.asu.number_of_atoms * (model_number + 1)
                # Todo test logic, have to offset the index by 1 from the start and end_idx values
                entity2_indecies = [idx for idx in entity2_indecies if idx >= end_idx or idx < start_idx]
            entity2_atoms = [atom for model_number in range(self.number_of_models) for atom in entity2_atoms]
            entity2_coords = self.model_coords[entity2_indecies]  # only get the coordinate indices we want!
        else:
            entity2_coords = self.coords[entity2_indecies]  # only get the coordinate indices we want!

        # Construct CB tree for entity1 and query entity2 CBs for a distance less than a threshold
        entity1_tree = BallTree(entity1_coords)
        entity2_query = entity1_tree.query_radius(entity2_coords, distance)

        # Return residue numbers of identified coordinates
        # interface_pairs = []
        # for entity2_idx in range(entity2_query.size):
        #     # if entity2_query[entity2_idx].size > 0:
        #     #     entity2_residue_number = entity2_atoms[entity2_indecies[entity2_idx]].residue_number
        #         for entity1_idx in entity2_query[entity2_idx]:
        #             # entity1_residue_number = entity1_atoms[entity1_indecies[entity1_idx]].residue_number
        #             interface_pairs.append((entity1_atoms[entity1_indecies[entity1_idx]].residue_number,
        #                                     entity2_atoms[entity2_indecies[entity2_idx]].residue_number))
        #             # interface_pairs.append((entity1_residue_number, entity2_residue_number))
        # return interface_pairs
        return [(entity1_atoms[entity1_indecies[entity1_idx]].residue_number,
                 entity2_atoms[entity2_indecies[entity2_idx]].residue_number)
                for entity2_idx in range(entity2_query.size) for entity1_idx in entity2_query[entity2_idx]]

    @staticmethod
    def split_interface_pairs(interface_pairs):
        residues1, residues2 = zip(*interface_pairs)
        return sorted(set(residues1), key=int), sorted(set(residues2), key=int)

    def find_interface_residues(self, entity1=None, entity2=None, distance=8, include_glycine=True):
        """Get unique residues from each pdb across an interface

            Keyword Args:
                entity1=None (str): First entity name to measure interface between
                entity2=None (str): Second entity name to measure interface between
                distance=8 (int): The distance to query the interface in Angstroms
                include_glycine=True (bool): Whether glycine CA should be included in the tree
            Returns:
                (tuple(set): A tuple of interface residue sets across an interface
        """
        return split_interface_pairs(self.find_interface_pairs(entity1=entity1, entity2=entity2, distance=distance,
                                                               include_glycine=include_glycine))

    def query_interface_for_fragments(self, entity1=None, entity2=None):
        # interface_name = self.asu
        entity1_residue_numbers, entity2_residue_numbers = self.find_interface_residues(entity1=entity1,
                                                                                        entity2=entity2)
        # pdb1_interface_sa = pdb1.get_surface_area_residues(entity1_residue_numbers)
        # pdb2_interface_sa = pdb2.get_surface_area_residues(entity2_residue_numbers)
        # interface_buried_sa = pdb1_interface_sa + pdb2_interface_sa

        entity1_pdb = self.asu.get_entity(entity1)
        # if self.symmetry:  # self.model_coords:  # find the interface using symmetry
        #     if entity2 == entity1:  # check to ensure we only measure the interchain contacts
        #         entity2_pdb =
        #     else:
        #         entity2_pdb = self.asu.get_entity(entity2)
        # else:
        entity2_pdb = self.asu.get_entity(entity2)
        interface_frags1 = entity1_pdb.get_fragments(entity1_residue_numbers)
        interface_frags2 = entity2_pdb.get_fragments(entity2_residue_numbers)
        if self.symmetry:
            interface_frags2_nested = [self.return_symmetry_mates(frag) for frag in interface_frags2]
            interface_frags2 = list(chain.from_iterable(interface_frags2_nested))

        entity1_coords = entity1.extract_coords()
        fragment_matches = find_fragment_overlap_at_interface(entity1_coords, interface_frags1, interface_frags2)
        self.fragment_queries[(entity1, entity2)] = fragment_matches

    def return_interface_metrics(self):
        """From the reported fragment queries, find the interface scores"""
        interface_metrics_d = {}
        for query, fragment_matches in self.fragment_queries.items():
            res_level_sum_score, center_level_sum_score, number_residues_with_fragments, \
            number_fragment_central_residues, multiple_frag_ratio, total_residues, percent_interface_matched, \
            percent_interface_covered, fragment_content_d = get_fragment_metrics(fragment_matches)

            interface_metrics_d[query] = {'nanohedra_score': res_level_sum_score,
                                          'nanohedra_score_central': center_level_sum_score,
                                          'fragments': fragment_matches,
                                          # 'fragment_cluster_ids': ','.join(fragment_indices),
                                          # 'interface_area': interface_buried_sa,
                                          'multiple_fragment_ratio': multiple_frag_ratio,
                                          'number_fragment_residues_all': number_residues_with_fragments,
                                          'number_fragment_residues_central': number_fragment_central_residues,
                                          'total_interface_residues': total_residues,
                                          'number_fragments': len(fragment_matches),
                                          'percent_residues_fragment_all': percent_interface_covered,
                                          'percent_residues_fragment_center': percent_interface_matched,
                                          'percent_fragment_helix': fragment_content_d['1'],
                                          'percent_fragment_strand': fragment_content_d['2'],
                                          'percent_fragment_coil': fragment_content_d['3'] + fragment_content_d['4'] +
                                          fragment_content_d['5']}

        return interface_metrics_d

    def initialize_pose(self, design_dir=None, symmetry=None, frag_db='biological_interfaces'):
        # Todo ensure ASU
        # Todo fix chains/entities
        # Todo connect design_dir obj or query user for their files

        if symmetry and isinstance(symmetry, dict):  # otherwise done on __init__()
            self.set_symmetry(**symmetry)

        self.connect_fragment_database(location=frag_db)
        for entity_pair in combinations(self.entities, 2):
            self.query_interface_for_fragments(*entity_pair)

        query_idx_to_alignment_type = {0: 'mapped', 1: 'paired'}
        for query_pair, fragments in self.fragment_queries.items():
            for query_idx, entity_name in enumerate(query_pair):
                # if entity_name == entity.get_name():
                self.pdb.entity(entity_name).assign_fragments(fragments=fragments,
                                                              alignment_type=query_idx_to_alignment_type[query_idx])

        for entity in self.entities:
            # must provide the list from des_dir.gather_fragment_metrics or InterfaceScoring.py then specify whether the
            # Entity in question is from mapped or paired
            # such as fragment_source = des_dir.fragment_observations
            entity.connect_fragment_database(location=self.frag_db)
            entity.add_profile(out_path=design_dir.sequences, pdb_numbering=True)  # fragment_source=design_dir.fragment_observations, frag_alignment_type=fragment_info_source,


        # Extract PSSM for each protein and combine into single PSSM
        # set pose.pssm
        self.combine_pssm([entity.pssm for entity in self.entities])
        logger.debug('Position Specific Scoring Matrix: %s' % str(self.pssm))
        self.pssm_file = self.make_pssm_file(self.pssm, PUtils.msa_pssm, outpath=design_dir.path)  # staticmethod

        # set pose.fragment_profile
        self.combine_fragment_profile([entity.fragment_profile for entity in self.entities])
        logger.debug('Fragment Specific Scoring Matrix: %s' % str(self.fragment_profile))
        # Todo, remove missing fragment entries here or add where they are loaded keep_extras = False  # =False added for pickling 6/14/20
        # interface_data_file = SDUtils.pickle_object(final_issm, frag_db + PUtils.frag_profile, out_path=des_dir.data)
        interface_data_file = SDUtils.pickle_object(self.fragment_profile, self.frag_db + PUtils.frag_profile, out_path=design_dir.data)

        self.combine_dssm([entity.dssm for entity in self.asu])  # sets pose.pssm
        logger.debug('Design Specific Scoring Matrix: %s' % str(self.dssm))
        self.dssm_file = self.make_pssm_file(self.dssm, PUtils.dssm, outpath=design_dir.path)  # static

        self.solve_consensus()

        # Update DesignDirectory with design information
        design_dir.info['pssm'] = self.pssm_file
        design_dir.info['issm'] = interface_data_file
        design_dir.info['dssm'] = self.dssm_file
        design_dir.info['db'] = self.frag_db
        design_dir.info['des_residues'] = [j for name in names for j in int_res_numbers[name]]
        # TODO add oligomer data to .info
        info_pickle = SDUtils.pickle_object(design_dir.info, 'info', out_path=design_dir.data)


def subdirectory(name):  # TODO PDBdb
    return name


def download_pdb(pdb, location=os.getcwd(), asu=False):
    """Download a pdbs from a file, a supplied list, or a single entry

    Args:
        pdb (str, list): PDB's of interest. If asu=False, code_# is format for biological assembly specific pdb.
            Ex: 1bkh_2 fetches biological assembly 2
    Keyword Args:
        asu=False (bool): Whether or not to download the asymmetric unit file
    Returns:
        (None)
    """
    clean_list = to_iterable(pdb)

    failures = []
    for pdb in clean_list:
        clean_pdb = pdb[0:4]  # .upper() redundant call
        if asu:
            assembly = ''
        else:
            assembly = pdb[-3:]
            try:
                assembly = assembly.split('_')[1]
            except IndexError:
                assembly = '1'

        clean_pdb = '%s.pdb%s' % (clean_pdb, assembly)
        file_name = os.path.join(location, clean_pdb)
        current_file = glob(file_name)
        # current_files = os.listdir(location)
        # if clean_pdb not in current_files:
        if not current_file:  # glob will return an empty list if the file is missing and therefore should be downloaded
            # TODO subprocess.POPEN()
            status = os.system('wget -q -O %s http://files.rcsb.org/download/%s' % (file_name, clean_pdb))
            if status != 0:
                failures.append(pdb)

    if failures:
        logger.error('PDB download ran into the following failures:\n%s' % ', '.join(failures))

    return file_name  # if list then will only return the last file


def fetch_pdbs(codes, location=PUtils.pdb_db):  # UNUSED
    """Fetch PDB object of each chain from PDBdb or PDB server

    Args:
        codes (iter): Any iterable of PDB codes
    Keyword Args:
        location= : Location of the  on disk
    Returns:
        (dict): {pdb_code: PDB.py object, ...}
    """
    if PUtils.pdb_source == 'download_pdb':
        get_pdb = download_pdb
        # doesn't return anything at the moment
    else:
        get_pdb = (lambda pdb_code, dummy: glob(os.path.join(PUtils.pdb_location, subdirectory(pdb_code),
                                                             '%s.pdb' % pdb_code)))
        # returns a list with matching file (should only be one)
    oligomers = {}
    for code in codes:
        pdb_file_name = get_pdb(code, location=des_dir.pdbs)
        assert len(pdb_file_name) == 1, 'More than one matching file found for pdb code %s' % code
        oligomers[code] = PDB(file=pdb_file_name[0])
        oligomers[code].set_name(code)
        oligomers[code].reorder_chains()

    return oligomers


def retrieve_pdb_file_path(code, directory=PUtils.pdb_db):
    """Fetch PDB object of each chain from PDBdb or PDB server

        Args:
            code (iter): Any iterable of PDB codes
        Keyword Args:
            location= : Location of the  on disk
        Returns:
            (str): path/to/your_pdb.pdb
        """
    if PUtils.pdb_source == 'download_pdb':
        get_pdb = download_pdb
        # doesn't return anything at the moment
    else:
        get_pdb = (lambda pdb_code, location=None: glob(os.path.join(location, 'pdb%s.ent' % pdb_code.lower())))
        # The below set up is my local pdb and the format of escher. cassini is slightly different, ughhh
        # get_pdb = (lambda pdb_code, dummy: glob(os.path.join(PUtils.pdb_db, subdirectory(pdb_code),
        #                                                      '%s.pdb' % pdb_code)))
        # returns a list with matching file (should only be one)

    # pdb_file = get_pdb(code, location)
    pdb_file = get_pdb(code, location=directory)
    # pdb_file = get_pdb(code, location=des_dir.pdbs)
    assert len(pdb_file) == 1, 'More than one matching file found for PDB: %s' % code
    assert pdb_file != list(), 'No matching file found for PDB: %s' % code

    return pdb_file[0]


def construct_cb_atom_tree(pdb1, pdb2, distance=8):  # Todo Pose.py
    """Create a atom tree using CB atoms from two PDB's

    Args:
        pdb1 (PDB): First PDB to query against
        pdb2 (PDB): Second PDB which will be tested against pdb1
    Keyword Args:
        distance=8 (int): The distance to query in Angstroms
        include_glycine=True (bool): Whether glycine CA should be included in the tree
    Returns:
        query (list()): sklearn query object of pdb2 coordinates within dist of pdb1 coordinates
        pdb1_cb_indices (list): List of all CB indices from pdb1
        pdb2_cb_indices (list): List of all CB indices from pdb2
    """
    # Get CB Atom Coordinates including CA coordinates for Gly residues
    pdb1_coords = np.array(pdb1.extract_CB_coords(InclGlyCA=gly_ca))
    pdb2_coords = np.array(pdb2.extract_CB_coords(InclGlyCA=gly_ca))

    # Construct CB Tree for PDB1
    pdb1_tree = BallTree(pdb1_coords)

    # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
    return pdb1_tree.query_radius(pdb2_coords, distance)


def find_interface_pairs(pdb1, pdb2, distance=8, gly_ca=True):  # Todo Pose.py
    """Get pairs of residues across an interface within a certain distance

        Args:
            pdb1 (PDB): First pdb to measure interface between
            pdb2 (PDB): Second pdb to measure interface between
        Keyword Args:
            distance=8 (int): The distance to query in Angstroms
        Returns:
            interface_pairs (list(tuple): A list of interface residue pairs across the interface
    """
    query = construct_cb_atom_tree(pdb1, pdb2, distance=distance)

    # Map Coordinates to Atoms
    pdb1_cb_indices = pdb1.get_cb_indices(InclGlyCA=gly_ca)
    pdb2_cb_indices = pdb2.get_cb_indices(InclGlyCA=gly_ca)

    # Map Coordinates to Residue Numbers
    interface_pairs = []
    for pdb2_index in range(len(query)):
        if query[pdb2_index].tolist() != list():
            pdb2_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_index]].residue_number
            for pdb1_index in query[pdb2_index]:
                pdb1_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_index]].residue_number
                interface_pairs.append((pdb1_res_num, pdb2_res_num))

    return interface_pairs


def split_interface_pairs(interface_pairs):  # Todo Pose.py
    residues1, residues2 = zip(*interface_pairs)
    return sorted(set(residues1), key=int), sorted(set(residues2), key=int)


def find_interface_residues(pdb1, pdb2, distance=8):  # Todo Pose.py
    """Get unique residues from each pdb across an interface

        Args:
            pdb1 (PDB): First pdb to measure interface between
            pdb2 (PDB): Second pdb to measure interface between
        Keyword Args:
            distance=8 (int): The distance to query in Angstroms
        Returns:
            (tuple(set): A tuple of interface residue sets across an interface
    """
    return split_interface_pairs(find_interface_pairs(pdb1, pdb2, distance=distance))


def get_fragments(pdb, chain_res_info, fragment_length=5):
    interface_frags = []
    ca_count = 0
    for residue_number in chain_res_info:
        frag_residue_numbers = [residue_number + i for i in range(-2, 3)]
        frag_atoms, ca_present = [], []
        for residue in pdb.residue(frag_residue_numbers):
            frag_atoms.extend(residue.get_atoms())
            if residue.get_ca():
                ca_count += 1

        if ca_count == 5:
            interface_frags.append(PDB.from_atoms(frag_atoms))

    return interface_frags


def find_fragment_overlap_at_interface(entity1_coords, interface_frags1, interface_frags2, max_z_value=2):  # entity1, entity2, entity1_interface_residue_numbers, entity2_interface_residue_numbers, max_z_value=2):
    """From a Structure Entity, score the interface between them according to Nanohedra's fragment matching"""
    # Get entity1 interface fragments with guide coordinates using fragment database
    # interface_frags1 = get_fragments(entity1, entity1_interface_residue_numbers)

    kdtree_oligomer1_backbone = BallTree(entity1_coords)
    # kdtree_oligomer1_backbone = BallTree(np.array(entity1.extract_backbone_coords()))
    complete_int1_ghost_frag_l, interface_ghostfrag_guide_coords_list = [], []
    for frag1 in interface_frags1:
        complete_monofrag1 = MonoFragment(frag1, ijk_monofrag_cluster_rep_pdb_dict)
        complete_monofrag1_ghostfrag_list = complete_monofrag1.get_ghost_fragments(
            ijk_intfrag_cluster_rep_dict, kdtree_oligomer1_backbone, ijk_intfrag_cluster_info_dict)
        if complete_monofrag1_ghostfrag_list:
            complete_int1_ghost_frag_l.extend(complete_monofrag1_ghostfrag_list)
            for ghost_frag in complete_monofrag1_ghostfrag_list:
                interface_ghostfrag_guide_coords_list.append(ghost_frag.get_guide_coords())

    # Get entity2 interface fragments with guide coordinates using complete fragment database
    # interface_frags2 = get_fragments(entity2, entity2_interface_residue_numbers)

    complete_int2_frag_l, interface_surf_frag_guide_coords_list = [], []
    for frag2 in interface_frags2:
        complete_monofrag2 = MonoFragment(frag2, ijk_monofrag_cluster_rep_pdb_dict)
        complete_monofrag2_guide_coords = complete_monofrag2.get_guide_coords()
        if complete_monofrag2_guide_coords:
            complete_int2_frag_l.append(complete_monofrag2)
            interface_surf_frag_guide_coords_list.append(complete_monofrag2_guide_coords)

    # Check for matching Euler angles
    eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(interface_ghostfrag_guide_coords_list,
                                                               interface_surf_frag_guide_coords_list)
    eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]

    all_fragment_overlap = filter_euler_lookup_by_zvalue(eul_lookup_true_list, complete_int1_ghost_frag_l,
                                                         interface_ghostfrag_guide_coords_list,
                                                         complete_int2_frag_l, interface_surf_frag_guide_coords_list,
                                                         z_value_func=calculate_overlap, max_z_value=max_z_value)
    passing_fragment_overlap = list(filter(None, all_fragment_overlap))
    ghostfrag_surffrag_pairs = [(complete_int1_ghost_frag_l[eul_lookup_true_list[idx][0]],
                                 complete_int2_frag_l[eul_lookup_true_list[idx][1]])
                                for idx, boolean in enumerate(all_fragment_overlap) if boolean]
    fragment_matches = []
    for frag_idx, (interface_ghost_frag, interface_mono_frag) in enumerate(ghostfrag_surffrag_pairs):
        ghost_frag_i_type = interface_ghost_frag.get_i_frag_type()
        ghost_frag_j_type = interface_ghost_frag.get_j_frag_type()
        ghost_frag_k_type = interface_ghost_frag.get_k_frag_type()
        cluster_id = "%s_%s_%s" % (ghost_frag_i_type, ghost_frag_j_type, ghost_frag_k_type)

        entity1_surffrag_ch, entity1_surffrag_resnum = interface_ghost_frag.get_aligned_surf_frag_central_res_tup()
        entity2_surffrag_ch, entity2_surffrag_resnum = interface_mono_frag.get_central_res_tup()
        score_term = passing_fragment_overlap[frag_idx][0]
        fragment_matches.append({'mapped': entity1_surffrag_resnum, 'match_score': score_term,
                                 'paired': entity2_surffrag_resnum, 'culster': cluster_id})

    return fragment_matches


def get_fragment_metrics(fragment_matches):
    # fragment_matches = [{'mapped': entity1_surffrag_resnum, 'match_score': score_term,
    #                          'paired': entity2_surffrag_resnum, 'culster': cluster_id}, ...]
    fragment_i_index_count_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    fragment_j_index_count_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    entity1_match_scores, entity2_match_scores = {}, {}
    interface_residues_with_fragment_overlap = {'mapped': set(), 'paired': set()}
    for fragment in fragment_matches:

        interface_residues_with_fragment_overlap['mapped'].add(fragment['mapped'])
        interface_residues_with_fragment_overlap['paired'].add(fragment['paired'])
        covered_residues_pdb1 = [(fragment['mapped'] + j) for j in range(-2, 3)]
        covered_residues_pdb2 = [(fragment['paired'] + j) for j in range(-2, 3)]
        for k in range(5):
            resnum1 = covered_residues_pdb1[k]
            resnum2 = covered_residues_pdb2[k]
            if resnum1 not in entity1_match_scores:
                entity1_match_scores[resnum1] = [fragment['match_score']]
            else:
                entity1_match_scores[resnum1].append(fragment['match_score'])

            if resnum2 not in entity2_match_scores:
                entity2_match_scores[resnum2] = [fragment['match_score']]
            else:
                entity2_match_scores[resnum2].append(fragment['match_score'])

        fragment_i_index_count_d[fragment['cluster'].split('_')[0]] += 1
        fragment_j_index_count_d[fragment['cluster'].split('_')[0]] += 1

    # Generate Nanohedra score for center and all residues
    all_residue_score, center_residue_score = 0, 0
    for residue_number, res_scores in entity1_match_scores.items():
        n = 1
        res_scores_sorted = sorted(res_scores, reverse=True)
        if residue_number in interface_residues_with_fragment_overlap['mapped']:  # interface_residue_numbers: <- may be at termini
            for central_score in res_scores_sorted:
                center_residue_score += central_score * (1 / float(n))
                n *= 2
        else:
            for peripheral_score in res_scores_sorted:
                all_residue_score += peripheral_score * (1 / float(n))
                n *= 2

    # doing this twice seems unnecessary as there is no new fragment information, but residue observations are
    # weighted by n, number of observations which differs between entities across the interface
    for residue_number, res_scores in entity2_match_scores.items():
        n = 1
        res_scores_sorted = sorted(res_scores, reverse=True)
        if residue_number in interface_residues_with_fragment_overlap['paired']:  # interface_residue_numbers: <- may be at termini
            for central_score in res_scores_sorted:
                center_residue_score += central_score * (1 / float(n))
                n *= 2
        else:
            for peripheral_score in res_scores_sorted:
                all_residue_score += peripheral_score * (1 / float(n))
                n *= 2

    all_residue_score += center_residue_score

    # Get the number of central residues with overlapping fragments identified given z_value criteria
    number_unique_residues_with_fragment_obs = len(interface_residues_with_fragment_overlap['mapped']) + \
        len(interface_residues_with_fragment_overlap['paired'])

    # Get the number of residues with fragments overlapping given z_value criteria
    number_residues_in_fragments = len(entity1_match_scores) + len(entity2_match_scores)

    if number_unique_residues_with_fragment_obs > 0:
        multiple_frag_ratio = (len(fragment_matches) * 2) / number_unique_residues_with_fragment_obs  # paired fragment
    else:
        multiple_frag_ratio = 0

    interface_residue_count = len(interface_residues_with_fragment_overlap['mapped']) + len(interface_residues_with_fragment_overlap['paired'])
    if interface_residue_count > 0:
        percent_interface_matched = number_unique_residues_with_fragment_obs / float(interface_residue_count)
        percent_interface_covered = number_residues_in_fragments / float(interface_residue_count)
    else:
        percent_interface_matched, percent_interface_covered = 0, 0

    # Sum the total contribution from each fragment type on both sides of the interface
    fragment_content_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    for index in fragment_i_index_count_d:
        fragment_content_d[index] += fragment_i_index_count_d[index]
        fragment_content_d[index] += fragment_j_index_count_d[index]

    if len(fragment_matches) > 0:
        for index in fragment_content_d:
            fragment_content_d[index] = fragment_content_d[index] / (len(fragment_matches) * 2)  # paired fragment

    return all_residue_score, center_residue_score, number_residues_in_fragments, \
        number_unique_residues_with_fragment_obs, multiple_frag_ratio, interface_residue_count, \
        percent_interface_matched, percent_interface_covered, fragment_content_d


def calculate_interface_score(interface_pdb):
    """Takes as input a single PDB with two chains"""
    interface_name = interface_pdb.name

    entity1 = PDB(atoms=interface_pdb.chain(interface_pdb.chain_id_list[0]).get_atoms())
    entity1.update_attributes_from_pdb(interface_pdb)
    entity2 = PDB(atoms=interface_pdb.chain(interface_pdb.chain_id_list[-1]).get_atoms())
    entity2.update_attributes_from_pdb(interface_pdb)

    entity1_ch_interface_residue_numbers, entity2_ch_interface_residue_numbers = \
        get_interface_fragment_chain_residue_numbers(entity1, entity2)

    entity1_interface_sa = entity1.get_surface_area_residues(entity1_ch_interface_residue_numbers)
    entity2_interface_sa = entity2.get_surface_area_residues(entity2_ch_interface_residue_numbers)
    interface_buried_sa = entity1_interface_sa + entity2_interface_sa

    interface_frags1 = get_fragments(entity1, entity1_ch_interface_residue_numbers)
    interface_frags2 = get_fragments(entity2, entity2_ch_interface_residue_numbers)
    entity1_coords = entity1.extract_coords()

    fragment_matches = find_fragment_overlap_at_interface(entity1_coords, interface_frags1, interface_frags2)
    # fragment_matches = find_fragment_overlap_at_interface(entity1, entity2, entity1_interface_residue_numbers,
    #                                                       entity2_interface_residue_numbers)

    res_level_sum_score, center_level_sum_score, number_residues_with_fragments, number_fragment_central_residues, \
        multiple_frag_ratio, total_residues, percent_interface_matched, percent_interface_covered, \
        fragment_content_d = get_fragment_metrics(fragment_matches)

    interface_metrics = {'nanohedra_score': res_level_sum_score, 'nanohedra_score_central': center_level_sum_score,
                         'fragments': fragment_matches,
                         # 'fragment_cluster_ids': ','.join(fragment_indices),
                         'interface_area': interface_buried_sa,
                         'multiple_fragment_ratio': multiple_frag_ratio,
                         'number_fragment_residues_all': number_residues_with_fragments,
                         'number_fragment_residues_central': number_fragment_central_residues,
                         'total_interface_residues': total_residues, 'number_fragments': len(fragment_matches),
                         'percent_residues_fragment_all': percent_interface_covered,
                         'percent_residues_fragment_center': percent_interface_matched,
                         'percent_fragment_helix': fragment_content_d['1'],
                         'percent_fragment_strand': fragment_content_d['2'],
                         'percent_fragment_coil': fragment_content_d['3'] + fragment_content_d['4']
                         + fragment_content_d['5']}

    return interface_name, interface_metrics


def get_interface_fragment_chain_residue_numbers(pdb1, pdb2, cb_distance=8):
    """Given two PDBs, return the unique chain and interacting residue lists"""
    # Get the interface residues
    pdb1_cb_coords, pdb1_cb_indices = pdb1.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)
    pdb2_cb_coords, pdb2_cb_indices = pdb2.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)

    pdb1_cb_kdtree = sklearn.neighbors.BallTree(np.array(pdb1_cb_coords))

    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    query = pdb1_cb_kdtree.query_radius(pdb2_cb_coords, cb_distance)

    # Get ResidueNumber, ChainID for all Interacting PDB1 CB, PDB2 CB Pairs
    interacting_pairs = []
    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].tolist() != list():
            pdb2_cb_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].residue_number
            pdb2_cb_chain_id = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].chain
            for pdb1_query_index in query[pdb2_query_index]:
                pdb1_cb_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].residue_number
                pdb1_cb_chain_id = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].chain
                interacting_pairs.append(((pdb1_cb_res_num, pdb1_cb_chain_id), (pdb2_cb_res_num, pdb2_cb_chain_id)))

    # Get interface fragment information
    pdb1_central_chainid_resnum_unique_list, pdb2_central_chainid_resnum_unique_list = [], []
    for pair in interacting_pairs:

        pdb1_central_res_num = pair[0][0]
        pdb1_central_chain_id = pair[0][1]
        pdb2_central_res_num = pair[1][0]
        pdb2_central_chain_id = pair[1][1]

        pdb1_res_num_list = [pdb1_central_res_num - 2, pdb1_central_res_num - 1, pdb1_central_res_num,
                             pdb1_central_res_num + 1, pdb1_central_res_num + 2]
        pdb2_res_num_list = [pdb2_central_res_num - 2, pdb2_central_res_num - 1, pdb2_central_res_num,
                             pdb2_central_res_num + 1, pdb2_central_res_num + 2]

        frag1_ca_count = 0
        for atom in pdb1.all_atoms:
            if atom.chain == pdb1_central_chain_id:
                if atom.residue_number in pdb1_res_num_list:
                    if atom.is_CA():
                        frag1_ca_count += 1

        frag2_ca_count = 0
        for atom in pdb2.all_atoms:
            if atom.chain == pdb2_central_chain_id:
                if atom.residue_number in pdb2_res_num_list:
                    if atom.is_CA():
                        frag2_ca_count += 1

        if frag1_ca_count == 5 and frag2_ca_count == 5:
            if (pdb1_central_chain_id, pdb1_central_res_num) not in pdb1_central_chainid_resnum_unique_list:
                pdb1_central_chainid_resnum_unique_list.append((pdb1_central_chain_id, pdb1_central_res_num))

            if (pdb2_central_chain_id, pdb2_central_res_num) not in pdb2_central_chainid_resnum_unique_list:
                pdb2_central_chainid_resnum_unique_list.append((pdb2_central_chain_id, pdb2_central_res_num))

    return pdb1_central_chainid_resnum_unique_list, pdb2_central_chainid_resnum_unique_list