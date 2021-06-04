import os
from copy import copy
import math
import pickle
from glob import glob
from itertools import chain as iter_chain, combinations_with_replacement, combinations
from math import sqrt, cos, sin
# from operator import itemgetter

import numpy as np
# from numba import njit, jit
from sklearn.neighbors import BallTree
# import requests

import PathUtils as PUtils
from SymDesignUtils import to_iterable, pickle_object, DesignError, calculate_overlap, z_value_from_match_score, \
    start_log, null_log, possible_symmetries, match_score_from_z_value, split_interface_pairs, dictionary_lookup
from classes.SymEntry import get_rot_matrices, RotRangeDict, get_degen_rotmatrices, SymEntry, flip_x_matrix
from utils.GeneralUtils import write_frag_match_info_file, transform_coordinate_sets
from utils.SymmetryUtils import valid_subunit_number, sg_cryst1_fmt_dict, pg_cryst1_fmt_dict, sg_zvalues
from classes.EulerLookup import EulerLookup
from PDB import PDB
from SequenceProfile import SequenceProfile
from DesignMetrics import calculate_match_metrics, fragment_metric_template, format_fragment_metrics
from Structure import Coords, Structure
from interface_analysis.Database import FragmentDB, FragmentDatabase

# Globals
logger = start_log(name=__name__)
config_directory = PUtils.pdb_db
sym_op_location = PUtils.sym_op_location


class Model:  # (PDB)
    """Keep track of different variations of the same PDB object whether they be mutated in sequence or have
    their coordinates perturbed
    """
    def __init__(self, pdb=None, models=None, log=None, **kwargs):
        super().__init__()  # without passing **kwargs, there is no need to ensure base Object class is protected
        # self.pdb = self.models[0]
        # elif isinstance(pdb, PDB):
        if log:
            self.log = log
        elif log is None:
            self.log = null_log
        else:  # When log is explicitly passed as False, use the module logger
            self.log = logger

        if pdb and isinstance(pdb, Structure):
            self.pdb = pdb
        if models and isinstance(models, list):
            self.models = models
        else:
            self.models = []

        self.number_of_models = len(self.models)

    @property
    def pdb(self):
        return self._pdb

    @pdb.setter
    def pdb(self, pdb):
        self._pdb = pdb
        # set up coordinate information
        self.coords = pdb._coords

    @property
    def number_of_atoms(self):
        return self.pdb.number_of_atoms

    @property
    def number_of_residues(self):
        return self.pdb.number_of_residues

    @property
    def coords(self):
        """Return a view of the representative Coords from the Model. These may be the ASU if a SymmetricModel"""
        return self._coords.coords

    @coords.setter
    def coords(self, coords):
        if isinstance(coords, Coords):
            self._coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Structure, use the private attribute _coords')

    @property
    def coords_indexed_to_residues(self):
        try:
            return self._coords_indexed_to_residues
        except AttributeError:
            self._coords_indexed_to_residues = [residue for residue, atom_idx in self.pdb.coords_indexed_residues]
            return self._coords_indexed_to_residues

    @property
    def center_of_mass(self):
        """Returns: (Numpy.ndarray)"""
        return np.matmul(np.full(self.number_of_atoms, 1 / self.number_of_atoms), self.coords)

    @property
    def model_coords(self):
        """Return a view of the modelled Coords. These may be symmetric if a SymmetricModel"""
        return self._model_coords.coords

    @model_coords.setter
    def model_coords(self, coords):
        if isinstance(coords, Coords):
            self._model_coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

    def set_models(self, models):
        self.models = models
        self.pdb = self.models[0]

    def add_model(self, pdb):
        self.models.append(pdb)

    # def add_atoms_to_pdb(self, index=0, atoms=None):
    #     """Add atoms to a PDB object in the model. Zero indexed"""
    #     self.models[index].read_atom_list(atoms)

    def get_ca_atoms(self):
        return [pdb.get_ca_atoms() for pdb in self.models]

    def chain(self, chain_id):
        return [pdb.chain(chain_id) for pdb in self.models]

    def get_coords(self):  # TODO
        return [pdb.coords for pdb in self.models]

    def get_backbone_coords(self):  # TODO
        return [pdb.get_backbone_coords() for pdb in self.models]

    def get_backbone_and_cb_coords(self):  # TODO
        return [pdb.get_backbone_and_cb_coords() for pdb in self.models]

    def get_ca_coords(self):  # TODO
        return [pdb.get_ca_coords() for pdb in self.models]

    def get_cb_coords(self, InclGlyCA=False):  # TODO
        return [pdb.get_cb_coords(InclGlyCA=InclGlyCA) for pdb in self.models]

    # def extract_cb_coords_chain(self, chain_id, InclGlyCA=False):
    #     return [pdb.extract_CB_coords_chain(chain_id, InclGlyCA=InclGlyCA) for pdb in self.models]

    def set_atom_coordinates(self, new_coords):  # Todo
        for i, pdb in enumerate(self.models):
            pdb.coords = Coords(new_coords[i])
        # return [pdb.set_atom_coordinates(new_cords[i]) for i, pdb in enumerate(self.models)]


class SymmetricModel(Model):
    def __init__(self, asu=None, **kwargs):
        super().__init__(**kwargs)  # log=log,
        if asu and isinstance(asu, Structure):
            self.asu = asu  # the pose specific asu
        # self.pdb = pdb
        # self.models = []
        # self.number_of_models = number_of_models
        # self.coords = []
        # self.model_coords = []
        self.coords_type = None  # coords_type
        self.sym_entry = None
        self.symmetry = None  # symmetry  # also defined in PDB as self.space_group
        self.symmetry_point_group = None
        self.dimension = None  # dimension
        self.uc_dimensions = None  # uc_dimensions  # also defined in PDB
        self.expand_matrices = None  # expand_matrices  # Todo make expand_matrices numpy
        self.asu_equivalent_model_idx = None
        self.oligomeric_equivalent_model_idxs = {}

        # if models:
        #     self.models = models
        #     self.number_of_models = len(models)
        #     self.set_symmetry(symmetry)

        # if symmetry:
        # if self.asu:
        # kwargs.update({})
        # symmetry_kwargs = self.asu.symmetry.copy()
        kwargs.update(self.asu.symmetry.copy())
        self.set_symmetry(**kwargs)
        #    self.set_symmetry(symmetry)

    @classmethod
    def from_assembly(cls, assembly, symmetry=None):
        assert symmetry, 'Currently, can\'t initialize a symmetric model without the symmetry! Pass symmetry during ' \
                         'Class initialization. Or add a scout symmetry Class method to SymmetricModel.'
        return cls(models=assembly, **symmetry)

    @property
    def asu(self):
        return self._asu

    @asu.setter
    def asu(self, asu):
        self._asu = asu
        self.pdb = asu

    @property
    def symmetric_center_of_mass(self):
        """Returns: (Numpy.ndarray)"""
        if self.symmetry:
            return np.matmul(np.full(self.number_of_atoms * self.number_of_models,
                                     1 / self.number_of_atoms * self.number_of_models),
                             self.model_coords)

    @property
    def symmetric_centers_of_mass(self):
        """Returns: (Numpy.ndarray)"""
        if self.symmetry:
            return np.matmul(np.full(self.number_of_atoms, 1 / self.number_of_atoms),
                             np.split(self.model_coords, self.number_of_models))

    def set_symmetry(self, sym_entry=None, expand_matrices=None, symmetry=None, cryst1=None, uc_dimensions=None,
                     generate_assembly=True, generate_symmetry_mates=False, **kwargs):
        """Set the model symmetry using the CRYST1 record, or the unit cell dimensions and the Hermann–Mauguin symmetry
        notation (in CRYST1 format, ex P 4 3 2) for the Model assembly. If the assembly is a point group,
        only the symmetry is required"""
        # if not expand_matrices:  # or not self.symmetry:
        if cryst1:
            uc_dimensions, symmetry = PDB.parse_cryst_record(cryst1_string=cryst1)

        if sym_entry and isinstance(sym_entry, SymEntry):
            self.sym_entry = sym_entry
            self.symmetry = sym_entry.result
            self.dimension = sym_entry.dim
            if self.dimension > 0:
                self.symmetry_point_group = sym_entry.pt_grp
                self.uc_dimensions = uc_dimensions

        elif symmetry:
            if uc_dimensions:
                self.uc_dimensions = uc_dimensions
                self.symmetry = ''.join(symmetry.split())

            if symmetry in pg_cryst1_fmt_dict:  # not available yet for non-Nanohedra PG's
                self.dimension = 2
                self.symmetry = symmetry
            elif symmetry in sg_cryst1_fmt_dict:  # not available yet for non-Nanohedra SG's
                self.dimension = 3
                self.symmetry = symmetry
            elif symmetry in possible_symmetries:  # ['T', 'O', 'I']:
                self.symmetry = possible_symmetries[symmetry]
                self.symmetry_point_group = possible_symmetries[symmetry]
                self.dimension = 0

            elif self.uc_dimensions:
                raise DesignError('Symmetry %s is not available yet! If you didn\'t provide it, the symmetry was likely'
                                  'set from a PDB file. Get the symmetry operations from the international'
                                  ' tables and add to the pickled operators if this displeases you!' % symmetry)
            else:  # when a point group besides T, O, or I is provided
                raise DesignError('Symmetry %s is not available yet! Get the canonical symm operators from %s and add '
                                  'to the pickled operators if this displeases you!' % (symmetry, PUtils.orient_dir))
        elif not symmetry:
            return None  # no symmetry was provided

        if expand_matrices:
            self.expand_matrices = expand_matrices
        else:
            self.expand_matrices = self.get_ptgrp_sym_op(self.symmetry) if self.dimension == 0 \
                else self.get_sg_sym_op(self.symmetry)  # ensure symmetry is Hermann–Mauguin notation
            # Todo numpy expand_matrices
        if self.asu and generate_assembly:
            self.generate_symmetric_assembly()  # **kwargs
            if generate_symmetry_mates:  # todo combine duplication with below
                self.get_assembly_symmetry_mates()

    def generate_symmetric_assembly(self, return_side_chains=True, surrounding_uc=False, generate_symmetry_mates=False,
                                    **kwargs):
        """Expand an asu in self.pdb using self.symmetry for the symmetry specification, and optional unit cell
        dimensions if self.dimension > 0. Expands assembly to complete point group, or the unit cell

        Keyword Args:
            return_side_chains=True (bool): Whether to return all side chain atoms. False returns backbone and CB atoms
            surrounding_uc=False (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        """
        if self.dimension == 0:
            self.get_point_group_coords(return_side_chains=return_side_chains)
        else:
            self.expand_uc_coords(surrounding_uc=surrounding_uc, return_side_chains=return_side_chains)

        self.log.info('Generated %d Symmetric Models' % self.number_of_models)
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
        """Takes a numpy array of coordinates and finds the fractional coordinates from cartesian coordinates
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
        v = a * b * c * \
            sqrt((1 - cos(alpha) ** 2 - cos(beta) ** 2 - cos(gamma) ** 2 + 2 * (cos(alpha) * cos(beta) * cos(gamma))))

        # deorthogonalization matrix M
        m0 = [1 / a, -(cos(gamma) / float(a * sin(gamma))),
              (((b * cos(gamma) * c * (cos(alpha) - (cos(beta) * cos(gamma)))) / float(sin(gamma))) -
               (b * c * cos(beta) * sin(gamma))) * (1 / float(v))]
        m1 = [0, 1 / (b * np.sin(gamma)),
              -((a * c * (cos(alpha) - (cos(beta) * cos(gamma)))) / float(v * sin(gamma)))]
        m2 = [0, 0, (a * b * sin(gamma)) / float(v)]
        m = [m0, m1, m2]

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
        v = a * b * c * \
            sqrt((1 - cos(alpha) ** 2 - cos(beta) ** 2 - cos(gamma) ** 2 + 2 * (cos(alpha) * cos(beta) * cos(gamma))))

        # orthogonalization matrix m_inv
        m_inv_0 = [a, b * cos(gamma), c * cos(beta)]
        m_inv_1 = [0, b * sin(gamma),
                   (c * (cos(alpha) - (cos(beta) * cos(gamma)))) / float(sin(gamma))]
        m_inv_2 = [0, 0, v / float(a * b * sin(gamma))]
        m_inv = [m_inv_0, m_inv_1, m_inv_2]

        return np.matmul(frac_coords, np.transpose(m_inv))

    def get_point_group_coords(self, return_side_chains=True):
        """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
        self.number_of_models = valid_subunit_number[self.symmetry]
        if return_side_chains:  # get different function calls depending on the return type # todo
            # get_pdb_coords = getattr(PDB, 'coords')
            self.coords_type = 'all'
        else:
            # get_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')
            self.coords_type = 'bb_cb'

        coords_length = len(self.coords)
        # print('Length of Coords: %d' % coords_length)
        # print('Number of Models: %d' % self.number_of_models)
        model_coords = np.empty((coords_length * self.number_of_models, 3), dtype=float)
        for idx, rot in enumerate(self.expand_matrices):
            # r_asu_coords = np.matmul(self.coords, np.transpose(rot))
            model_coords[idx * coords_length: (idx + 1) * coords_length] = np.matmul(self.coords, np.transpose(rot))
        self.model_coords = Coords(model_coords)
        # print('Length of Model Coords: %d' % len(self.model_coords))

    def get_unit_cell_coords(self, return_side_chains=True):
        """Generates unit cell coordinates for a symmetry group. Modifies model_coords to include all in a unit cell"""
        # self.models = [self.asu]
        self.number_of_models = sg_zvalues[self.symmetry]
        if return_side_chains:  # get different function calls depending on the return type  # todo
            # get_pdb_coords = getattr(PDB, 'coords')
            self.coords_type = 'all'
        else:
            # get_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')
            self.coords_type = 'bb_cb'

        # asu_frac_coords = self.cart_to_frac(self.coords)
        # coords_length = len(self.coords)
        # model_coords = np.empty((coords_length * self.number_of_models, 3), dtype=float)
        # for idx, (rot, tx) in enumerate(self.expand_matrices):
        #     rt_asu_frac_coords = np.matmul(asu_frac_coords, np.transpose(rot)) + tx
        #     model_coords[idx * coords_length: (idx + 1) * coords_length] = rt_asu_frac_coords
        uc_coords = self.return_unit_cell_coords(self.coords)
        self.model_coords = Coords(uc_coords)

    def get_surrounding_unit_cell_coords(self):
        """Generates a grid of unit cell coordinates for a symmetry group. Modifies model_coords from a unit cell
        representation to a grid of unit cells, either 3x3 for a layer group or 3x3x3 for a space group"""
        if self.dimension == 3:
            z_shifts, uc_number = [0, 1, -1], 9
        elif self.dimension == 2:
            z_shifts, uc_number = [0], 27
        else:
            return None

        uc_frac_coords = self.cart_to_frac(self.model_coords)
        # coords_length = len(self.model_coords)
        # model_coords = np.empty((coords_length * uc_number, 3), dtype=float)
        # idx = 0
        # for x_shift in [-1, 0, 1]:
        #     for y_shift in [-1, 0, 1]:
        #         for z_shift in z_shifts:
        #             add central uc_coords to the model coords after applying the correct tx of frac coords & convert
        #             model_coords[idx * coords_length: (idx + 1) * coords_length] = \
        #                 uc_frac_coords + [x_shift, y_shift, z_shift]
        #             idx += 1
        # self.model_coords = Coords(self.frac_to_cart(model_coords))

        surrounding_frac_coords = [uc_frac_coords + [x_shift, y_shift, z_shift] for x_shift in [0, 1, -1]
                                   for y_shift in [0, 1, -1] for z_shift in z_shifts]
        self.model_coords = Coords(self.frac_to_cart(surrounding_frac_coords))
        self.number_of_models = sg_zvalues[self.symmetry] * uc_number

    def return_assembly_symmetry_mates(self, **kwargs):
        """Return all symmetry mates in self.models (list[Structure]). Chain names will match the ASU"""
        count = 0
        while len(self.models) != self.number_of_models:
            if count == 1:
                raise DesignError('%s: The assembly couldn\'t be returned'
                                  % self.return_assembly_symmetry_mates.__name__)
            self.get_assembly_symmetry_mates(**kwargs)
            count += 1

        return self.models

    def get_assembly_symmetry_mates(self, surrounding_uc=False):  # , return_side_chains=True):
        """Set all symmetry mates in self.models (list[Structure]). Chain names will match the ASU"""
        if not self.symmetry:
            # self.log.critical('%s: No symmetry set for %s! Cannot get symmetry mates'  # Todo
            #                   % (self.get_assembly_symmetry_mates.__name__, self.asu.name))
            raise DesignError('%s: No symmetry set for %s! Cannot get symmetry mates'
                              % (self.get_assembly_symmetry_mates.__name__, self.asu.name))
        # if return_side_chains:  # get different function calls depending on the return type
        #     extract_pdb_atoms = getattr(PDB, 'get_atoms')
        # else:
        #     extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')

        # prior_idx = self.asu.number_of_atoms  # TODO modify by extract_pdb_atoms!
        if not surrounding_uc and self.symmetry in sg_zvalues:
            number_of_models = sg_zvalues[self.symmetry]  # set to the uc only
        else:
            number_of_models = self.number_of_models

        for model_idx in range(number_of_models):
            symmetry_mate_pdb = copy(self.asu)
            symmetry_mate_pdb.replace_coords(self.model_coords[(model_idx * self.asu.number_of_atoms):
                                                               ((model_idx + 1) * self.asu.number_of_atoms)])
            self.models.append(symmetry_mate_pdb)

    def find_asu_equivalent_symmetry_model(self):
        """Find the asu equivalent model in the SymmetricModel. Zero-indexed

        Returns:
            (int): The index of the number of models where the ASU can be found
        """
        if self.asu_equivalent_model_idx:
            self.log.debug('Skipping ASU identification as information already exists')
            return  # we already found this information

        template_atom_coords = self.asu.residues[0].ca_coords
        template_atom_index = self.asu.residues[0].ca_index
        coords_length = len(self.coords)
        for model_num in range(self.number_of_models):
            # print(self.model_coords[(model_num * coords_length) + template_atom_index])
            # print(template_atom_coords ==
            #         self.model_coords[(model_num * coords_length) + template_atom_index])
            if np.allclose(template_atom_coords, self.model_coords[(model_num * coords_length) + template_atom_index]):
                # if (template_atom_coords ==
                #         self.model_coords[(model_num * coords_length) + template_atom_index]).all():
                self.asu_equivalent_model_idx = model_num
                # return model_num
                return

        self.log.error('%s is FAILING' % self.find_asu_equivalent_symmetry_model.__name__)

    def find_intra_oligomeric_equivalent_symmetry_models(self, entity, epsilon=3):  # todo put back to 0.5 after SUCCES
        """From an Entity's Chain members, find the SymmetricModel equivalent models using Chain center or mass
        compared to the symmetric model center of mass

        Args:
            entity (Entity): The Entity with oligomeric chains that should be queried
        Keyword Args:
            epsilon=0.5 (float): The distance measurement tolerance to find similar symmetric models to the oligomer
        """
        if self.oligomeric_equivalent_model_idxs.get(entity):
            self.log.debug('Skipping oligomeric identification as information already exists')
            return None  # we already found this information
        asu_size = len(self.coords)
        # need to slice through the specific Entity coords once we have the model
        entity_start, entity_end = entity.atom_indices[0], entity.atom_indices[-1]
        entity_length = entity.number_of_atoms
        entity_center_of_mass_divisor = np.full(entity_length, 1 / entity_length)
        equivalent_models = []
        for chain in entity.chains:
            # chain_length = chain.number_of_atoms
            # chain_center_of_mass = np.matmul(np.full(chain_length, 1 / chain_length), chain.coords)
            chain_center_of_mass = chain.center_of_mass
            # print('Chain', chain_center_of_mass.astype(int))
            for model in range(self.number_of_models):
                sym_model_center_of_mass = \
                    np.matmul(entity_center_of_mass_divisor,  # '                            have to add 1 for slice v
                              self.model_coords[(model * asu_size) + entity_start: (model * asu_size) + entity_end + 1])
                # print('Sym Model', sym_model_center_of_mass)
                # if np.allclose(chain_center_of_mass.astype(int), sym_model_center_of_mass.astype(int)):
                # if np.allclose(chain_center_of_mass, sym_model_center_of_mass):  # using np.rint()
                if np.linalg.norm(chain_center_of_mass - sym_model_center_of_mass) < epsilon:
                    equivalent_models.append(model)
                    break
        assert len(equivalent_models) == len(entity.chains), \
            'The number of equivalent models (%d) does not equal the expected number of chains (%d)!'\
            % (len(equivalent_models), len(entity.chains))

        self.oligomeric_equivalent_model_idxs[entity] = equivalent_models
        # return equivalent_models

    def find_asu_equivalent_symmetry_mate_indices(self):
        """Find the asu equivalent model in the SymmetricModel. Zero-indexed

        self.model_coords must be from all atoms which by default is True
        Returns:
            (list): The indices in the SymmetricModel where the ASU is also located
        """
        self.find_asu_equivalent_symmetry_model()
        start_idx = self.asu.number_of_atoms * self.asu_equivalent_model_idx
        end_idx = self.asu.number_of_atoms * (self.asu_equivalent_model_idx + 1)
        return list(range(start_idx, end_idx))

    def find_intra_oligomeric_symmetry_mate_indices(self, entity):
        """Find the intra-oligomeric equivalent models in the SymmetricModel. Zero-indexed

        self.model_coords must be from all atoms which is True by default
        Args:
            entity (Entity): The Entity with oligomeric chains to query for corresponding symmetry mates
        Returns:
            (list): The indices in the SymmetricModel where the intra-oligomeric contacts are located
        """
        self.find_intra_oligomeric_equivalent_symmetry_models(entity)
        oligomeric_indices = []
        for model_number in self.oligomeric_equivalent_model_idxs.get(entity):
            start_idx = self.asu.number_of_atoms * model_number
            end_idx = self.asu.number_of_atoms * (model_number + 1)
            oligomeric_indices.extend(list(range(start_idx, end_idx)))

        return oligomeric_indices

    def return_symmetry_mates(self, pdb, **kwargs):  # return_side_chains=False, surrounding_uc=False):
        """Expand an asu in self.pdb using self.symmetry for the symmetry specification, and optional unit cell
        dimensions if self.dimension > 0. Expands assembly to complete point group, or the unit cell

        Keyword Args:
            return_side_chains=True (bool): Whether to return all side chain atoms. False gives backbone and CB atoms
            surrounding_uc=False (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        """
        if self.dimension == 0:
            return self.return_point_group_symmetry_mates(pdb)
        else:
            return self.return_crystal_symmetry_mates(pdb, **kwargs)  # return_side_chains=return_side_chains,
            #                                                           surrounding_uc=surrounding_uc)

    def return_point_group_symmetry_mates(self, pdb):
        """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
        return [pdb.return_transformed_copy(rotation=rot) for rot in self.expand_matrices]

    def return_crystal_symmetry_mates(self, pdb, surrounding_uc=False, **kwargs):
        """Expand the backbone coordinates for every symmetric copy within the unit cells surrounding a central cell
        """
        if surrounding_uc:
            return self.return_surrounding_unit_cell_symmetry_mates(pdb, **kwargs)  # return_side_chains
        else:
            return self.return_unit_cell_symmetry_mates(pdb, **kwargs)  # return_side_chains

    def return_symmetric_coords(self, coords):
        """Return the unit cell coordinates from a set of coordinates for the specified SymmetricModel"""
        if self.dimension == 0:
            coords_length = len(coords)
            model_coords = np.empty((coords_length * self.number_of_models, 3), dtype=float)
            for idx, rot in enumerate(self.expand_matrices):
                rot_coords = np.matmul(coords, np.transpose(rot))
                model_coords[idx * coords_length: (idx + 1) * coords_length] = rot_coords

            return model_coords
        else:
            return self.return_unit_cell_coords(coords)

    def return_unit_cell_coords(self, coords, fractional=False):
        """Return the unit cell coordinates from a set of coordinates for the specified SymmetricModel"""
        asu_frac_coords = self.cart_to_frac(coords)
        coords_length = len(coords)
        model_coords = np.empty((coords_length * self.number_of_models, 3), dtype=float)
        # Todo pickled operators don't have identity, so we should add the asu.
        model_coords[:coords_length] = asu_frac_coords
        for idx, (rot, tx) in enumerate(self.expand_matrices, 1):  # since no identity, start idx at 1
            rt_asu_frac_coords = np.matmul(asu_frac_coords, np.transpose(rot)) + tx
            model_coords[idx * coords_length: (idx + 1) * coords_length] = rt_asu_frac_coords

        if fractional:
            return model_coords
        else:
            return self.frac_to_cart(model_coords)

    def return_unit_cell_symmetry_mates(self, pdb, return_side_chains=True):  # For returning PDB copies
        """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
        if return_side_chains:  # get different function calls depending on the return type
            # extract_pdb_atoms = getattr(PDB, 'get_atoms')  # Not using. The copy() versus PDB() changes residue objs
            pdb_coords = pdb.coords
        else:
            # extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
            pdb_coords = pdb.get_backbone_and_cb_coords()

        # # asu_cart_coords = self.pdb.get_coords()  # returns a numpy array
        # asu_cart_coords = extract_pdb_coords(pdb)
        # # asu_frac_coords = self.cart_to_frac(np.array(asu_cart_coords))
        # asu_frac_coords = self.cart_to_frac(asu_cart_coords)
        # sym_frac_coords = []
        # for rot, tx in self.expand_matrices:
        #     r_asu_frac_coords = np.matmul(asu_frac_coords, np.transpose(rot))
        #     tr_asu_frac_coords = r_asu_frac_coords + tx
        #     sym_frac_coords.extend(tr_asu_frac_coords)
        #     # tr_asu_cart_coords = self.frac_to_cart(tr_asu_frac_coords)
        # sym_cart_coords = self.frac_to_cart(sym_frac_coords)
        # pdb_coords = extract_pdb_coords(pdb)
        sym_cart_coords = self.return_unit_cell_coords(pdb_coords)

        coords_length = len(pdb_coords)
        sym_mates = []
        # for coord_set in sym_cart_coords:
        for model in range(self.number_of_models):
            symmetry_mate_pdb = copy(pdb)
            symmetry_mate_pdb.replace_coords(sym_cart_coords[model * coords_length: (model + 1) * coords_length])
            sym_mates.append(symmetry_mate_pdb)

        return sym_mates

    def return_surrounding_unit_cell_symmetry_mates(self, pdb, return_side_chains=True, **kwargs):
        """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
        if return_side_chains:  # get different function calls depending on the return type
            # extract_pdb_atoms = getattr(PDB, 'get_atoms')  # Not using. The copy() versus PDB() changes residue objs
            pdb_coords = pdb.coords
        else:
            # extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
            pdb_coords = pdb.get_backbone_and_cb_coords()

        if self.dimension == 3:
            z_shifts, uc_number = [0, 1, -1], 9
        elif self.dimension == 2:
            z_shifts, uc_number = [0], 27
        else:
            return None

        # pdb_coords = extract_pdb_coords(pdb)
        uc_frac_coords = self.return_unit_cell_coords(pdb_coords, fractional=True)
        surrounding_frac_coords = [uc_frac_coords + [x_shift, y_shift, z_shift] for x_shift in [0, 1, -1]
                                   for y_shift in [0, 1, -1] for z_shift in z_shifts]

        surrounding_cart_coords = self.frac_to_cart(surrounding_frac_coords)

        coords_length = len(uc_frac_coords)
        sym_mates = []
        for coord_set in surrounding_cart_coords:
            # for model in self.number_of_models:
            for model in range(sg_zvalues[self.symmetry]):
                symmetry_mate_pdb = copy(pdb)
                symmetry_mate_pdb.replace_coords(coord_set[(model * coords_length): ((model + 1) * coords_length)])
                sym_mates.append(symmetry_mate_pdb)

        assert len(sym_mates) == uc_number * sg_zvalues[self.symmetry], \
            'Number of models %d is incorrect! Should be %d' % (len(sym_mates), uc_number * sg_zvalues[self.symmetry])
        return sym_mates

    def assign_entities_to_sub_symmetry(self):
        """From a symmetry entry, find the entities which belong to each sub-symmetry (the component groups) which make
        the global symmetry. Construct the sub-symmetry by copying each symmetric chain to the Entity's .chains
        attribute
        """
        if not self.symmetry:
            raise DesignError('Must set a global symmetry to assign entities to sub symmetry!')

        # Get the rotation matrices for each group then orient along the setting matrix "axis"
        if self.sym_entry.group1 in ['D2', 'D3', 'D4', 'D6'] or self.sym_entry.group2 in ['D2', 'D3', 'D4', 'D6']:
            group1 = self.sym_entry.group1.replace('D', 'C')
            group2 = self.sym_entry.group1.replace('D', 'C')
            rotation_matrices_only1 = get_rot_matrices(RotRangeDict[group1], 'z', 360)
            rotation_matrices_only2 = get_rot_matrices(RotRangeDict[group2], 'z', 360)
            # provide a 180 degree rotation along x (all D orient symmetries have axis here)
            # apparently passing the degeneracy matrix first without any specification towards the row/column major
            # worked for Josh. I am not sure that I understand his degeneracy (rotation) matrices orientation enough to
            # understand if he hardcoded the column "majorness" into situations with rot and degen np.matmul(rot, degen)
            rotation_matrices_group1 = get_degen_rotmatrices(flip_x_matrix, rotation_matrices_only1)
            rotation_matrices_group2 = get_degen_rotmatrices(flip_x_matrix, rotation_matrices_only2)
            # group_set_rotation_matrices = {1: np.matmul(degen_rot_mat_1, np.transpose(set_mat1)),
            #                                2: np.matmul(degen_rot_mat_2, np.transpose(set_mat2))}
            raise DesignError('Using dihedral symmetry has not been implemented yet! It is required to change the code'
                              ' before continuing with design of symmetry entry %d!' % self.sym_entry.entry_number)
        else:
            group1 = self.sym_entry.group1
            group2 = self.sym_entry.group1
            rotation_matrices_group1 = get_rot_matrices(RotRangeDict[group1], 'z', 360)
            rotation_matrices_group2 = get_rot_matrices(RotRangeDict[group2], 'z', 360)

        # Assign each Entity to a symmetry group
        # entity_coms = [entity.center_of_mass for entity in self.asu]
        # all_entities_com = np.matmul(np.full(len(entity_coms), 1 / len(entity_coms)), entity_coms)
        all_entities_com = self.center_of_mass
        origin = np.array([0., 0., 0.])
        # check if global symmetry is centered at the origin. If not, translate to the origin with ext_tx
        self.log.debug('The symmetric center of mass is: %s' % str(self.symmetric_center_of_mass))
        if np.isclose(self.symmetric_center_of_mass, origin):  # is this threshold loose enough?
            # the com is at the origin
            self.log.debug('The symmetric center of mass is at the origin')
            ext_tx = origin
            expand_matrices = self.expand_matrices
        else:
            self.log.debug('The symmetric center of mass is NOT at the origin')
            # Todo find ext_tx from input without Nanohedra input? There is a difficulty when the symmetry is a crystal
            #  and the external translation should be to the center of a component point group. The difficulty is
            #  finding that point group a priori due to a random collection of centers of mass that belong to it and
            #  their orientation with respect to the cell origin. In Nanohedra, the origin will work for many symmetries
            if self.dimension > 0:
                # Todo we have different set up required here. The expand matrices can be derived from a point group in
                #  the layer or space setting, however we must ensure that the required external tx is respected
                #  (i.e. subtracted) at the required steps such as from coms_group1/2 in return_symmetric_coords
                #  (generated using self.expand_matrices) and/or the entity_com as this is set up within a
                #  cartesian expand matrix environment is going to yield wrong results on the expand matrix indexing
                assert self.number_of_models == sg_zvalues[self.symmetry], 'Cannot have more models than a single UC!'
                expand_matrices = self.get_ptgrp_sym_op(self.symmetry_point_group)
            else:
                expand_matrices = self.expand_matrices
            ext_tx = self.symmetric_center_of_mass  # only works for unit cell or point group NOT surrounding UC
            # This is typically centered at the origin for the symmetric assembly... NEED rigourous testing.
            # Maybe this route of generation is too flawed for layer/space? Nanohedra framework gives a comprehensive
            # handle on all these issues though
        # find the approximate scalar translation of the asu center of mass from the reference symmetry origin
        approx_entity_com_reference = np.linalg.norm(all_entities_com - ext_tx)
        approx_entity_z_tx = [0., 0., approx_entity_com_reference]
        # apply the setting matrix for each group to the approximate translation
        set_mat1 = self.sym_entry.get_rot_set_mat_group1()
        set_mat2 = self.sym_entry.get_rot_set_mat_group2()
        # TODO test transform_coordinate_sets has the correct input format (numpy.ndarray)
        com_group1 = \
            transform_coordinate_sets(origin, translation=approx_entity_z_tx, rotation2=set_mat1, translation2=ext_tx)
        com_group2 = \
            transform_coordinate_sets(origin, translation=approx_entity_z_tx, rotation2=set_mat2, translation2=ext_tx)
        # expand the tx'd, setting matrix rot'd, approximate coms for each group using self.expansion operators
        coms_group1 = self.return_symmetric_coords(com_group1)
        coms_group2 = self.return_symmetric_coords(com_group2)

        # measure the closest distance from each entity com to the setting matrix transformed approx group coms to find
        # which group the entity belongs to. Save the group and the operation index of the expansion matrices. With both
        # of these, it is possible to find a new setting matrix that is symmetry equivalent and will generate the
        # correct sub-symmetry symmetric copies for each provided Entity
        group_entity_rot_ops = {1: {}, 2: {}}
        # min_dist1, min_dist2, min_1_entity, min_2_entity = float('inf'), float('inf'), None, None
        for entity in self.asu.entities:
            entity_com = entity.center_of_mass
            min_dist, min_entity_group_operator = float('inf'), None
            for idx in range(len(expand_matrices)):  # has the length of the symmetry operations
                com1_distance = np.linalg.norm(entity_com - coms_group1[idx])
                com2_distance = np.linalg.norm(entity_com - coms_group2[idx])
                if com1_distance < com2_distance:
                    if com1_distance < min_dist:
                        min_dist = com1_distance
                        min_entity_group_operator = (group2, expand_matrices[idx])
                    # # entity_min_group = 1
                    # entity_group_d[1].append(entity)
                else:
                    if com2_distance < min_dist:
                        min_dist = com2_distance
                        min_entity_group_operator = (group2, expand_matrices[idx])
                    # # entity_min_group = 2
                    # entity_group_d[2].append(entity)
            if min_entity_group_operator:
                group, operation = min_entity_group_operator
                group_entity_rot_ops[group][entity] = operation
                # {1: {entity1: [[],[],[]]}, 2: {entity2: [[],[],[]]}}

        set_mat = {1: set_mat1, 2: set_mat2}
        inv_set_matrix = {1: np.linalg.inv(set_mat1), 2: np.linalg.inv(set_mat2)}
        group_rotation_matrices = {1: rotation_matrices_group1, 2: rotation_matrices_group2}
        # Multiplication is not possible in this way apparently!
        # group_set_rotation_matrices = {1: np.matmul(rotation_matrices_group1, np.transpose(set_mat1)),
        #                                2: np.matmul(rotation_matrices_group2, np.transpose(set_mat2))}

        # Apply the rotation matrices to the identified group Entities. First modify the Entity by the inverse expansion
        # and setting matrices to orient along Z axis. Apply the rotation matrix, then reverse operations back to start
        for idx, (group, entity_ops) in enumerate(group_entity_rot_ops.items()):
            for entity, rot_op in entity_ops.items():
                dummy_rotation = False
                dummy_translation = False
                # Todo need to reverse the expansion matrix first to get the entity coords to the "canonical" setting
                #  matrix as expected by Nanohedra. I can then make_oligomers
                entity.make_oligomer(sym=group, **dict(rotation=dummy_rotation, translation=dummy_translation,
                                                       rotation2=set_mat[idx], translation2=ext_tx))
                # # Todo if this is a fractional rot/tx pair this won't work
                # #  I converted the space group external tx and design_pg_symmetry to rot_matrices so I should
                # #  test if the change to local point group symmetry in a layer or space group is sufficient
                # inv_expand_matrix = np.linalg.inv(rot_op)
                # inv_rotation_matrix = np.linalg.inv(dummy_rotation)
                # # entity_inv = entity.return_transformed_copy(rotation=inv_expand_matrix, rotation2=inv_set_matrix[group])
                # # need to reverse any external transformation to the entity coords so rotation occurs at the origin...
                # centered_coords = transform_coordinate_sets(entity.coords, translation=-ext_tx)
                # sym_on_z_coords = transform_coordinate_sets(centered_coords, rotation=inv_expand_matrix,
                #                                             rotation2=inv_set_matrix[group])
                # TODO                                        NEED DIHEDRAl rotation v back to canonical
                # sym_on_z_coords = transform_coordinate_sets(centered_coords, rotation=inv_rotation_matrix,
                # TODO                                        as well as v translation (not approx, dihedral won't work)
                #                                             translation=approx_entity_z_tx)
                # # now rotate, then undo symmetry expansion matrices
                # # for rot in group_rotation_matrices[group][1:]:  # exclude the first rotation matrix as it is identity
                # for rot in group_rotation_matrices[group]:
                #     temp_coords = transform_coordinate_sets(sym_on_z_coords, rotation=np.array(rot), rotation2=set_mat[group])
                #     # rot_centered_coords = transform_coordinate_sets(sym_on_z_coords, rotation=rot)
                #     # final_coords = transform_coordinate_sets(rot_centered_coords, rotation=rotation,
                #     #                                          translation=translation, <-NEED^ for DIHEDRAL
                #     #                                          rotation2=rotation2, translation2=translation2)
                #     final_coords = transform_coordinate_sets(temp_coords, rotation=rot_op, translation=ext_tx)
                #     # Entity representative stays in the .chains attribute as chain[0] given the iterator slice above
                #     sub_symmetry_mate_pdb = copy(entity.chain_representative)
                #     sub_symmetry_mate_pdb.replace_coords(final_coords)
                #     entity.chains.append(sub_symmetry_mate_pdb)
                #     # need to take the cyclic system generated and somehow transpose it on the dihedral group.
                #     # an easier way would be to grab the assembly from the SymDesignOutput/Data/PDBs and set the
                #     # oligomer onto the ASU. The .chains would then be populated for the non-transposed chains
                #     # if dihedral:  # TODO
                #     #     dummy = True

    def symmetric_assembly_is_clash(self, distance=2.1):  # Todo design_selector
        """Returns True if the SymmetricModel presents any clashes. Checks only backbone and CB atoms

        Keyword Args:
            distance=2.2 (float): The cutoff distance for the coordinate overlap

        Returns:
            (bool)
        """
        if not self.symmetry:
            raise DesignError('[Error] Cannot check if the assembly is clashing as it has no symmetry!')
        elif not self.number_of_models:
            raise DesignError('[Error] Cannot check if the assembly is clashing without first calling %s'
                              % self.generate_symmetric_assembly.__name__)

        model_asu_indices = self.find_asu_equivalent_symmetry_mate_indices()
        if self.coords_type != 'bb_cb':
            # print('reducing coords to bb_cb')
            # Need to only select the coords that are BB or CB from the model coords
            number_asu_atoms = self.asu.number_of_atoms
            asu_indices = self.asu.get_backbone_and_cb_indices()
            # print('number of asu_residues: %d' % len(self.asu.residues))
            # print('asu_indices: %s' % asu_indices)
            # print('length asu', len(asu_indices))
            # We have all the BB/CB indices from ASU now need to multiply this by every integer in self.number_of_models
            # to get every BB/CB coord in the model
            # Finally we take out those indices that are inclusive of the model_asu_indices like below
            model_indices_filter = np.array([idx + (model_number * number_asu_atoms)
                                             for model_number in range(self.number_of_models)
                                             for idx in asu_indices])
        else:  # we will grab every coord in the model
            model_indices_filter = np.array(list(range(len(self.model_coords))))
            asu_indices = None

        # make a boolean mask where the model indices of interest are True
        without_asu_mask = np.logical_or(model_indices_filter < model_asu_indices[0],
                                         model_indices_filter > model_asu_indices[-1])
        # take the boolean mask and filter the model indices mask to leave only symmetry mate bb/cb indices, NOT asu
        model_indices_without_asu = model_indices_filter[without_asu_mask]
        # print(model_indices_without_asu)
        # print('length model_indices_without_asu', len(model_indices_without_asu))
        # print(asu_indices)
        selected_assembly_coords = len(model_indices_without_asu) + len(asu_indices)
        all_assembly_coords_length = len(asu_indices) * self.number_of_models
        assert selected_assembly_coords == all_assembly_coords_length, \
            '%s: Ran into an issue indexing' % self.symmetric_assembly_is_clash.__name__

        asu_coord_tree = BallTree(self.coords[asu_indices])
        clash_count = asu_coord_tree.two_point_correlation(self.model_coords[model_indices_without_asu], [distance])
        if clash_count[0] > 0:
            self.log.warning('%s: Found %d clashing sites! Pose is not a viable symmetric assembly'
                             % (self.pdb.name, clash_count[0]))
            return True  # clash
        else:
            return False  # no clash

    def write(self, out_path=os.getcwd(), header=None, increment_chains=False):  # , cryst1=None):  # Todo write symmetry, name, location
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle. Return the filename if
        one was written"""
        with open(out_path, 'w') as f:
            if header:
                if isinstance(header, str):
                    f.write(header)
                # if isinstance(header, Iterable):

            if increment_chains:
                idx = 0
                # for idx, model in enumerate(self.models):
                for model in self.models:
                    for entity in model.entities:
                        chain = PDB.available_letters[idx]
                        entity.write(file_handle=f, chain=chain)
                        chain_terminal_atom = entity.atoms[-1]
                        f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', chain_terminal_atom.number + 1,
                                                                              chain_terminal_atom.residue_type, chain,
                                                                              chain_terminal_atom.residue_number))
                        idx += 1
            else:
                for model_number, model in enumerate(self.models, 1):
                    f.write('{:9s}{:>4d}\n'.format('MODEL', model_number))
                    for entity in model.entities:
                        entity.write(file_handle=f)
                        chain_terminal_atom = entity.atoms[-1]
                        f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', chain_terminal_atom.number + 1,
                                                                              chain_terminal_atom.residue_type,
                                                                              entity.chain_id,
                                                                              chain_terminal_atom.residue_number))
                    f.write('ENDMDL\n')

    @staticmethod
    def get_ptgrp_sym_op(sym_type, expand_matrix_dir=os.path.join(sym_op_location, 'POINT_GROUP_SYMM_OPERATORS')):
        """Get the symmetry operations for a specified point group oriented in the cannonical orientation
        Returns:
            (list[list])
        """
        expand_matrix_filepath = os.path.join(expand_matrix_dir, '%s.txt' % sym_type)
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
        sg_op_filepath = os.path.join(expand_matrix_dir, '%s.pickle' % sym_type)
        with open(sg_op_filepath, 'rb') as sg_op_file:
            sg_sym_op = pickle.load(sg_op_file)

        return sg_sym_op


class Pose(SymmetricModel, SequenceProfile):  # Model
    """A Pose is made of single or multiple PDB objects such as Entities, Chains, or other tsructures.
    All objects share a common feature such as the same symmetric system or the same general atom configuration in
    separate models across the Structure or sequence.
    """
    def __init__(self, asu_file=None, pdb_file=None, **kwargs):  # asu=None, pdb=None,
        # the member pdbs which make up the pose. todo, combine with self.models?
        # self.pdbs = []
        self.pdbs_d = {}
        self.fragment_pairs = []
        self.fragment_metrics = {}
        self.design_selector_entities = set()
        self.design_selector_indices = set()
        self.required_indices = set()
        self.required_residues = None
        self.interface_residues = {}
        self.source_db = kwargs.get('source_db', None)
        self.split_interface_residues = {}  # {1: '23A,45A,46A,...' , 2: '234B,236B,239B,...'}
        self.split_interface_ss_elements = {}  # {1: [0,1,2] , 2: [9,13,19]]}
        self.ss_index_array = []  # stores secondary structure elements by incrementing index
        self.ss_type_array = []  # stores secondary structure type ('H', 'S', ...)
        # self.handle_flags(**kwargs)
        # self.ignore_clashes = False
        self.ignore_clashes = kwargs.get('ignore_clashes', False)
        # self.design_selector = {}
        # if kwargs.get('design_selector'):
        self.design_selector = kwargs.get('design_selector', {})
        # else:
        #     self.design_selector = {}
        # if asu and isinstance(asu, Structure):
        #     self.asu = asu
        if asu_file:
            self.asu = PDB.from_file(asu_file, log=self.log)  # **kwargs todo create kwarg collector to protect object
        # elif pdb and isinstance(pdb, Structure):
        #     self.pdb = pdb
        elif pdb_file:
            self.pdb = PDB.from_file(pdb_file, log=self.log)  # **kwargs

        super().__init__(**kwargs)  # will only generate an assembly if an ASU is present
        # super().__init__(**kwargs)
        # self.set_symmetry(**symmetry_kwargs)

        frag_db = kwargs.get('frag_db')
        if frag_db:  # Attach existing FragmentDB to the Pose
            self.attach_fragment_database(db=frag_db)
            # self.frag_db = frag_db  # Todo property
            for entity in self.entities:
                entity.attach_fragment_database(db=frag_db)

        self.euler_lookup = kwargs.get('euler_lookup', None)
        # for entity in self.entities:  # No need to attach to entities
        #     entity.euler_lookup = euler_lookup

    @classmethod
    def from_pdb(cls, pdb, **kwargs):
        return cls(pdb=pdb, **kwargs)

    @classmethod
    def from_pdb_file(cls, pdb_file, **kwargs):
        return cls(pdb_file=pdb_file, **kwargs)

    @classmethod
    def from_asu(cls, asu, **kwargs):
        return cls(asu=asu, **kwargs)

    @classmethod
    def from_asu_file(cls, asu_file, **kwargs):
        return cls(asu_file=asu_file, **kwargs)

    # @property
    # def asu(self):
    #     return self._asu
    #
    # @asu.setter
    # def asu(self, asu):
    #     self._asu = asu
    #     self.pdb = asu

    # @property
    # def pdb(self):
    #     return self._pdb

    @Model.pdb.setter
    def pdb(self, pdb):
        self.log.debug('Adding PDB \'%s\' to Pose' % pdb.name)
        # super(Model, self).pdb = pdb
        self._pdb = pdb
        self.coords = pdb._coords
        if not self.ignore_clashes:
            if pdb.is_clash():
                raise DesignError('%s contains Backbone clashes as is not being considered further!' % self.name)
        self.pdbs_d[pdb.name] = pdb
        self.create_design_selector()  # **self.design_selector)

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            return self.pdb.name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def active_entities(self):
        return [entity for entity in self.pdb.entities if entity in self.design_selector_entities]

    @property
    def entities(self):
        return self.pdb.entities

    @property
    def chains(self):
        return [chain for entity in self.pdb.entities for chain in entity.chains]

    @property
    def active_chains(self):
        return [chain for entity in self.active_entities for chain in entity.chains]

    @property
    def residues(self):
        return self.pdb.residues

    @property
    def reference_sequence(self):
        return ''.join(self.pdb.reference_sequence.values())

    # def find_center_of_mass(self):
    #     """Retrieve the center of mass for the specified Structure"""
    #     if self.symmetry:
    #         divisor = 1 / len(self.model_coords)
    #         self._center_of_mass = np.matmul(np.full(self.number_of_atoms, divisor), self.model_coords)
    #     else:
    #         divisor = 1 / len(self.coords)  # must use coords as can have reduced view of the full coords, i.e. BB & CB
    #         self._center_of_mass = np.matmul(np.full(self.number_of_atoms, divisor), self.coords)

    def add_pdb(self, pdb):
        """Add a PDB to the Pose PDB as well as the member PDB container"""
        # self.debug_pdb(tag='add_pdb')  # here, pdb chains are still in the oriented configuration
        self.pdbs_d[pdb.name] = pdb
        self.add_entities_to_pose(pdb)

    def add_entities_to_pose(self, pdb):
        """Add each unique Entity in a PDB to the Pose PDB. Multiple PDB's become one PDB representative"""
        current_pdb_entities = self.entities
        for idx, entity in enumerate(pdb.entities):
            current_pdb_entities.append(entity)
        self.log.debug('Adding the entities \'%s\' to the Pose'
                       % ', '.join(list(entity.name for entity in current_pdb_entities)))

        self.pdb = PDB.from_entities(current_pdb_entities, metadata=self.pdb, log=self.log)

    def get_contacting_asu(self, distance=8):
        """From the Pose PDB and the associated active Entities, find the maximal contacting ASU for each of the
        entities

        Keyword Args:
            distance=8 (int): The distance to check for contacts
        Returns:
            (PDB): The PDB object with the minimal set of Entities containing the maximally touching configuration
        """
        # self.debug_pdb(tag='get_contacting')
        idx = 0
        chain_combinations, entity_combinations = [], []
        contact_count = np.zeros((math.prod([len(entity.chains) for entity in self.active_entities])))
        for entity1, entity2 in combinations(self.active_entities, 2):
            for chain1 in entity1.chains:
                chain_cb_coord_tree = BallTree(chain1.get_cb_coords())
                for chain2 in entity2.chains:
                    chain_combinations.append((chain1, chain2))
                    entity_combinations.append((entity1, entity2))
                    contact_count[idx] = chain_cb_coord_tree.two_point_correlation(chain2.get_cb_coords(),
                                                                                   [distance])[0]
                    idx += 1
        max_contact_idx = contact_count.argmax()
        additional_chains = []
        max_chains = list(chain_combinations[max_contact_idx])
        if len(max_chains) != len(self.active_entities):
            selected_chain_indices = [idx for idx, chain_pair in enumerate(chain_combinations)
                                      if max_chains[0] in chain_pair or max_chains[1] in chain_pair]
            remaining_entities = set(self.active_entities).difference(entity_combinations[max_contact_idx])
            for entity in remaining_entities:  # get the maximum contacts and the associated entity and chain indices
                remaining_indices = [idx for idx, entity_pair in enumerate(entity_combinations)
                                     if entity in entity_pair]
                pair_position = [0 if entity_pair[0] == entity else 1
                                 for idx, entity_pair in enumerate(entity_combinations) if entity in entity_pair]
                viable_remaining_indices = list(set(remaining_indices).intersection(selected_chain_indices))
                max_index = contact_count[viable_remaining_indices].argmax()
                additional_chains.append(chain_combinations[max_index][pair_position[max_index]])

        return PDB.from_chains(max_chains + additional_chains, name='asu', log=self.log)

    def entity(self, entity):
        return self.pdb.entity(entity)

    def chain(self, chain):
        return self.pdb.entity_from_chain(chain)

    # def handle_flags(self, design_selector=None, frag_db=None, ignore_clashes=False, **kwargs):
    #     self.ignore_clashes = ignore_clashes
    #     if design_selector:
    #         self.design_selector = design_selector
    #     else:
    #         self.design_selector = {}
    #     # if design_selector:
    #     #     self.create_design_selector(**design_selector)
    #     # else:
    #     #     # self.create_design_selector(selection={}, mask={}, required={})
    #     #     self.design_selector_entities = self.design_selector_entities.union(set(self.entities))
    #     #     self.design_selector_indices = self.design_selector_indices.union(set(self.pdb.atom_indices))
    #     if frag_db:
    #         # Attach an existing FragmentDB to the Pose
    #         self.attach_fragment_database(db=frag_db)
    #         for entity in self.entities:
    #             entity.attach_fragment_database(db=frag_db)

    def create_design_selector(self):  # , selection=None, mask=None, required=None):
        """Set up a design selector for the Pose including selctions, masks, and required Entities and Atoms

        Sets:
            self.design_selector_indices (set[int])
            self.design_selector_entities (set[Entity])
            self.required_indices (set[int])
        """
        if len(self.pdbs_d) > 1:
            self.log.debug('The design_selector may be incorrect as the Pose was initialized with multiple PDB '
                           'files. Proceed with caution if this is not what you expected!')

        def grab_indices(pdbs=None, entities=None, chains=None, residues=None, pdb_residues=None, atoms=None,
                         start_with_none=False):
            if start_with_none:
                entity_set = set()
                atom_indices = set()
                set_function = getattr(set, 'union')
            else:  # start with all indices and include those of interest
                entity_set = set(self.entities)
                atom_indices = set(self.pdb.atom_indices)
                set_function = getattr(set, 'intersection')

            if pdbs:
                # atom_selection = set(self.pdb.get_residue_atom_indices(numbers=residues))
                raise DesignError('Can\'t select residues by PDB yet!')
            if entities:
                atom_indices = set_function(atom_indices, iter_chain.from_iterable([self.entity(entity).atom_indices
                                                                                   for entity in entities]))
                entity_set = set_function(entity_set, [self.entity(entity) for entity in entities])
            if chains:
                # vv This is for the intersectional model
                atom_indices = set_function(atom_indices, iter_chain.from_iterable([self.chain(chain_id).atom_indices
                                                                                   for chain_id in chains]))
                # atom_indices.union(iter_chain.from_iterable(self.chain(chain_id).get_residue_atom_indices(numbers=residues)
                #                                     for chain_id in chains))
                # ^^ This is for the additive model
                entity_set = set_function(entity_set, [self.chain(chain_id) for chain_id in chains])
            if residues:
                atom_indices = set_function(atom_indices, self.pdb.get_residue_atom_indices(numbers=residues))
            if pdb_residues:
                atom_indices = set_function(atom_indices, self.pdb.get_residue_atom_indices(numbers=residues, pdb=True))
            if atoms:
                atom_indices = set_function(atom_indices, [idx for idx in self.pdb.atom_indices if idx in atoms])

            return entity_set, atom_indices

        if 'selection' in self.design_selector:
            self.log.debug('The design_selection includes: %s' % self.design_selector['selection'])
            entity_selection, atom_selection = grab_indices(**self.design_selector['selection'])
        else:
            entity_selection, atom_selection = set(self.entities), set(self.pdb.atom_indices)

        if 'mask' in self.design_selector:
            self.log.debug('The design_mask includes: %s' % self.design_selector['mask'])
            entity_mask, atom_mask = grab_indices(**self.design_selector['mask'], start_with_none=True)
        else:
            entity_mask, atom_mask = set(), set()

        self.design_selector_entities = entity_selection.difference(entity_mask)
        self.design_selector_indices = atom_selection.difference(atom_mask)

        if 'required' in self.design_selector:
            self.log.debug('The required_residues includes: %s' % self.design_selector['required'])
            entity_required, self.required_indices = grab_indices(**self.design_selector['required'],
                                                                  start_with_none=True)
            if self.required_indices:  # only if indices are specified should we grab them
                self.required_residues = self.pdb.get_residues_by_atom_indices(indices=self.required_indices)
        else:
            entity_required, self.required_indices = set(), set()

    def construct_cb_atom_tree(self, entity1, entity2, distance=8):  # TODO UNUSED
        """Create a atom tree using CB atoms from two PDB's

        Args:
            entity1 (Structure): First PDB to query against
            entity2 (Structure): Second PDB which will be tested against pdb1
        Keyword Args:
            distance=8 (int): The distance to query in Angstroms
            include_glycine=True (bool): Whether glycine CA should be included in the tree
        Returns:
            query (list()): sklearn query object of pdb2 coordinates within dist of pdb1 coordinates
            pdb1_cb_indices (list): List of all CB indices from pdb1
            pdb2_cb_indices (list): List of all CB indices from pdb2
        """
        # Get CB Atom Coordinates including CA coordinates for Gly residues
        entity1_indices = np.array(self.asu.entity(entity1).get_cb_indices())
        # mask = np.ones(self.asu.number_of_atoms, dtype=int)  # mask everything
        # mask[index_array] = 0  # we unmask the useful coordinates
        entity1_coords = self.coords[entity1_indices]  # only get the coordinate indices we want!

        entity2_indices = np.array(self.asu.entity(entity2).get_cb_indices())
        entity2_coords = self.coords[entity2_indices]  # only get the coordinate indices we want!

        # Construct CB Tree for PDB1
        entity1_tree = BallTree(entity1_coords)

        # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
        return entity1_tree.query_radius(entity2_coords, distance)

    def find_interface_pairs(self, entity1=None, entity2=None, distance=8):
        """Get pairs of residue numbers that have CB atoms within a certain distance between two named Entities

        Caution: Pose must have Coords representing all atoms! Residue pairs are found using CB indices from all atoms
        Symmetry aware. If symmetry is used, by default all atomic coordinates for entity2 are symmeterized.
        design_selector aware. Will remove interface residues if not active under the design selector

        Keyword Args:
            entity1=None (Entity): First entity to measure interface between
            entity2=None (Entity): Second entity to measure interface between
            distance=8 (int): The distance to query the interface in Angstroms
        Returns:
            (list[tuple]): A list of interface residue numbers across the interface
        """
        # entity2_query = construct_cb_atom_tree(entity1, entity2, distance=distance)
        self.log.debug('Entity %s | Entity %s interface query' % (entity1.name, entity2.name))
        # Get CB Atom Coordinates including CA coordinates for Gly residues
        # entity1_atoms = entity1.get_atoms()  # if passing by Structure
        entity1_indices = entity1.get_cb_indices()  # InclGlyCA=include_glycine)
        # print('PDB CB indices: %s' % self.pdb.get_cb_indices())
        # print('Number of PDB residues: %d' % len(pdb_residues))
        # print('Entity1 CB indices: %s' % entity1_indices)
        # entity2_atoms = entity2.get_atoms()  # if passing by Structure
        entity2_indices = entity2.get_cb_indices()  # InclGlyCA=include_glycine)
        # print('Entity2 CB indices: %s' % entity2_indices)

        if self.design_selector_indices:  # subtract the masked atom indices from the entity indices
            before = len(entity1_indices) + len(entity2_indices)
            entity1_indices = list(set(entity1_indices).intersection(self.design_selector_indices))
            entity2_indices = list(set(entity2_indices).intersection(self.design_selector_indices))
            self.log.debug('Applied design selection to interface identification. Number of indices before '
                           'selection = %d. Number after = %d' % (before, len(entity1_indices) + len(entity2_indices)))

        if not entity1_indices or not entity2_indices:
            return

        # pdb_atoms = self.pdb.atoms
        coords_length = len(self.coords)
        if self.symmetry:
            sym_string = 'symmetric '
            self.log.debug('Number of Atoms in Pose: %s' % coords_length)
            # get all symmetric indices
            entity2_indices = [idx + (coords_length * model_number) for model_number in range(self.number_of_models)
                               for idx in entity2_indices]
            # pdb_atoms = [atom for _ in range(self.number_of_models) for atom in pdb_atoms]
            # self.log.debug('Number of Atoms in Pose expanded assembly: %s' % len(pdb_atoms))
            # pdb_residues = [residue for model in range(self.number_of_models) for residue in pdb_residues]
            # entity2_atoms = [atom for model_number in range(self.number_of_models) for atom in entity2_atoms]
            if entity1 == entity2:
                # We don't want interactions with the symmetric asu model or intra-oligomeric contacts
                if entity1.is_oligomeric:  # remove oligomer protomers (has asu)
                    remove_indices = self.find_intra_oligomeric_symmetry_mate_indices(entity1)
                    self.log.debug('Removing indices from models %s due to detected oligomer'
                                   % self.oligomeric_equivalent_model_idxs.get(entity1))
                    self.log.debug('Removing %d indices from symmetric query due to detected oligomer'  # %s'
                                   % (len(remove_indices)))  # , remove_indices))
                else:  # remove asu
                    remove_indices = self.find_asu_equivalent_symmetry_mate_indices()
                self.log.debug('Number of indices before removal of \'self\' indices: %s' % len(entity2_indices))
                entity2_indices = list(set(entity2_indices).difference(remove_indices))
                self.log.debug('Final indices remaining after removing \'self\': %s' % len(entity2_indices))
            entity2_coords = self.model_coords[entity2_indices]  # only get the coordinate indices we want
        elif entity1 == entity2:
            # without symmetry, we can't measure this, unless intra-oligomeric contacts are desired
            self.log.warning('Entities are the same, but no symmetry is present. The interface between them will not be'
                             ' detected!')
            return None
        else:
            sym_string = ''
            entity2_coords = self.coords[entity2_indices]  # only get the coordinate indices we want

        # Construct CB tree for entity1 and query entity2 CBs for a distance less than a threshold
        entity1_coords = self.coords[entity1_indices]  # only get the coordinate indices we want
        entity1_tree = BallTree(entity1_coords)
        entity2_query = entity1_tree.query_radius(entity2_coords, distance)

        # Return residue numbers of identified coordinates
        self.log.info('Querying %d CB residues in Entity %s versus, %d CB residues in %sEntity %s'
                      % (len(entity1_indices), entity1.name, len(entity2_indices), sym_string, entity2.name))
        # self.log.debug('Entity2 Query size: %d' % entity2_query.size)
        # contacting_pairs = [(pdb_residues[entity1_indices[entity1_idx]],
        #                      pdb_residues[entity2_indices[entity2_idx]])
        #                    for entity2_idx in range(entity2_query.size) for entity1_idx in entity2_query[entity2_idx]]

        # contacting_pairs = [(pdb_atoms[entity1_indices[entity1_idx]].residue_number,
        #                      pdb_atoms[entity2_indices[entity2_idx]].residue_number)
        #                     for entity2_idx, entity1_contacts in enumerate(entity2_query)
        #                     for entity1_idx in entity1_contacts]
        contacting_pairs = [(self.coords_indexed_to_residues[entity1_indices[entity1_idx]],
                             self.coords_indexed_to_residues[entity2_indices[entity2_idx] % coords_length])
                            for entity2_idx, entity1_contacts in enumerate(entity2_query)
                            for entity1_idx in entity1_contacts]
        if entity1 != entity2:
            return contacting_pairs
        else:  # solve symmetric results for asymmetric contacts
            asymmetric_contacting_pairs, found_pairs = [], []
            for pair1, pair2 in contacting_pairs:
                # only add to contacting pair if we have never observed either
                if (pair1, pair2) not in found_pairs or (pair2, pair1) not in found_pairs:
                    asymmetric_contacting_pairs.append((pair1, pair2))
                # add both pair orientations (1, 2) or (2, 1) regardless
                found_pairs.extend([(pair1, pair2), (pair2, pair1)])

            return asymmetric_contacting_pairs

    def find_interface_residues(self, entity1=None, entity2=None, **kwargs):  # distance=8
        """Get unique residues from each pdb across an interface provided by two Entity names

        Keyword Args:
            entity1=None (Entity): First Entity to measure interface between
            entity2=None (Entity): Second Entity to measure interface between
            distance=8 (int): The distance to query the interface in Angstroms
        Returns:
            (tuple[set]): A tuple of interface residue sets across an interface
        """
        entity1_residues, entity2_residues = \
            split_interface_pairs(self.find_interface_pairs(entity1=entity1, entity2=entity2, **kwargs))
        # entity1_residue_numbers, entity2_residue_numbers = \
        #     split_interface_pairs(self.find_interface_pairs(entity1=entity1, entity2=entity2, **kwargs))
        # if not entity1_residue_numbers or not entity2_residue_numbers:
        if not entity1_residues or not entity2_residues:
            self.log.info('Interface search at %s | %s found no interface residues' % (entity1.name, entity2.name))
            self.fragment_queries[(entity1, entity2)] = []
            self.interface_residues[(entity1, entity2)] = ([], [])
            return None
        else:
            if entity1 == entity2:
                # separate the residue numbers so that only one interface gets the numbers?
                dum = True
                # for number in entity1_residue_numbers:

            self.log.info(
                'At Entity %s | Entity %s interface:\n\t%s found residue numbers: %s\n\t%s found residue numbers: %s'
                # % (entity1.name, entity2.name, entity1.name, entity1_residue_numbers, entity2.name,
                #    entity2_residue_numbers))
                % (entity1.name, entity2.name, entity1.name, ', '.join(str(res.number) for res in entity1_residues),
                   entity2.name, ', '.join(str(res.number) for res in entity2_residues)))

        self.interface_residues[(entity1, entity2)] = (entity1_residues, entity2_residues)
        # self.interface_residues[(entity1, entity2)] = (entity1.get_residues(numbers=entity1_residue_numbers),
        #                                                entity2.get_residues(numbers=entity2_residue_numbers))
        entities = [entity1, entity2]
        self.log.debug('Added interface_residues: %s' % ['%d%s' % (residue.number, entities[idx].chain_id)
                       for idx, entity_residues in enumerate(self.interface_residues[(entity1, entity2)])
                       for residue in entity_residues])

    def query_interface_for_fragments(self, entity1=None, entity2=None):
        """For all found interface residues in a Entity/Entity interface, search for corresponding fragment pairs

        Keyword Args:
            entity1=None (Structure): The first Entity to measure for an interface
            entity2=None (Structure): The second Entity to measure for an interface
        Sets:
            self.fragment_queries (dict[mapping[tuple, dict]])
        """
        entity1_residues, entity2_residues = self.interface_residues.get((entity1, entity2))
        if not entity1_residues or not entity2_residues:
            self.log.debug('At Entity %s | Entity %s interface, NO residues found')
            return
        if entity1 == entity2 and entity1.is_oligomeric:
            # surface_frags1.extend(surface_frags2)
            # surface_frags2 = surface_frags1
            entity1_residues = set(entity1_residues + entity2_residues)
            entity2_residues = entity1_residues
        # else:
        #     _entity1_residues, _entity2_residues = entity1_residues, entity2_residues
        # Todo make so that the residue objects support fragments instead of converting back
        entity1_res_numbers = sorted(residue.number for residue in entity1_residues)
        entity2_res_numbers = sorted(residue.number for residue in entity2_residues)
        self.log.debug('At Entity %s | Entity %s interface, searching for fragments at the surface of:%s%s'
                       % (entity1.name, entity2.name,
                          '\n\tEntity %s: Residues %s' % (entity1.name, ', '.join(map(str, entity1_res_numbers))),
                          '\n\tEntity %s: Residues %s' % (entity2.name, ', '.join(map(str, entity2_res_numbers)))))

        surface_frags1 = entity1.get_fragments(residue_numbers=entity1_res_numbers, representatives=self.frag_db.reps)
        surface_frags2 = entity2.get_fragments(residue_numbers=entity2_res_numbers, representatives=self.frag_db.reps)

        if not surface_frags1 or not surface_frags2:
            self.log.debug('At interface Entity %s | Entity %s\tNO fragments found' % (entity1.name, entity2.name))
            return
        else:
            self.log.debug(
                'At Entity %s | Entity %s interface:\t%s has %d interface fragments\t%s has %d interface fragments'
                % (entity1.name, entity2.name, entity1.name, len(surface_frags1), entity2.name, len(surface_frags2)))

        if self.symmetry:
            # even if entity1 == entity2, only need to expand the entity2 fragments due to surface/ghost frag mechanics
            # asu frag subtraction is unnecessary THIS IS ALL WRONG DEPENDING ON THE CONTEXT
            if entity1 == entity2:
                skip_models = self.oligomeric_equivalent_model_idxs[entity1]  # + [self.asu_equivalent_model_idx]
                self.log.debug('Skipping oligomeric models %s' % skip_models)
            else:
                skip_models = []
            surface_frags2_nested = [self.return_symmetry_mates(frag) for frag in surface_frags2]
            surface_frags2.clear()
            for frag_mates in surface_frags2_nested:
                surface_frags2.extend(frag for sym_idx, frag in enumerate(frag_mates) if sym_idx not in skip_models)
            self.log.debug('Entity %s has %d symmetric fragments' % (entity2.name, len(surface_frags2)))

        entity1_coords = entity1.get_backbone_and_cb_coords()  # for clash check, we only want the backbone and CB
        ghostfrag_surfacefrag_pairs = \
            find_fragment_overlap_at_interface(entity1_coords, surface_frags1, surface_frags2, fragdb=self.frag_db,
                                               euler_lookup=self.euler_lookup)
        self.log.info('Found %d overlapping fragment pairs at the %s | %s interface.'
                      % (len(ghostfrag_surfacefrag_pairs), entity1.name, entity2.name))
        self.fragment_queries[(entity1, entity2)] = get_matching_fragment_pairs_info(ghostfrag_surfacefrag_pairs)
        # add newly found fragment pairs to the existing fragment observations
        self.fragment_pairs.extend(ghostfrag_surfacefrag_pairs)

    def score_interface(self, entity1=None, entity2=None):
        """Generate the fragment metrics for a specified interface between two entities

        Returns:
            (dict): Fragment metrics as key (metric type) value (measurement) pairs
        """
        if (entity1, entity2) not in self.fragment_queries or (entity2, entity1) not in self.fragment_queries:
            self.find_interface_residues(entity1=entity1, entity2=entity2)
            self.query_interface_for_fragments(entity1=entity1, entity2=entity2)

        return self.return_fragment_metrics(by_interface=True, entity1=entity1, entity2=entity2)

    def find_and_split_interface(self):
        """Locate the interface residues for the designable entities and split into two interfaces

        Sets:
            self.split_interface_residues (dict): Residue/Entity id of each residue at the interface identified by interface id
            as split by topology
        """
        self.log.debug('Find and split interface using active_entities: %s' %
                       [entity.name for entity in self.active_entities])
        for entity_pair in combinations_with_replacement(self.active_entities, 2):
            self.find_interface_residues(*entity_pair)

        self.check_interface_topology()

    def check_interface_topology(self):
        """From each pair of entities that share an interface, split the identified residues into two distinct groups.
        If an interface can't be composed into two distinct groups, raise a DesignError
        """
        first, second = 0, 1
        interface_residue_d = {first: {}, second: {}, 'self': [False, False]}
        terminate = False
        # self.log.debug('Pose contains interface residues: %s' % self.interface_residues)
        for entity_pair, entity_residues in self.interface_residues.items():
            if not entity_residues:
                continue
            else:
                if entity_pair[first] == entity_pair[second]:  # if query is with self, have to record it
                    _self = True
                else:
                    _self = False

                # idx - 1 grabs the last index if at the first index, or grabs the first index if it's the last
                if not interface_residue_d[first]:
                    # on first observation, add the pair to the dictionary in their indexed order
                    interface_residue_d[first][entity_pair[first]] = copy(entity_residues[first])  # list
                    interface_residue_d[second][entity_pair[second]] = copy(entity_residues[second])  # list
                    if _self:
                        interface_residue_d['self'][second] = _self
                else:
                    # >= Second observation, divide the residues in each entity to the correct interface index
                    # Need to check if the Entity is in either side before adding
                    #           for idx, residues in enumerate(entity_residues):
                    if entity_pair[first] in interface_residue_d[first]:
                        if interface_residue_d['self'][first]:
                            # Ex4 - self Entity was added to index 0 while ASU added to index 1.
                            # Now, flip Entity at index 0 to the other side, add new Entity to index 1
                            interface_residue_d[second][entity_pair[first]].extend(entity_residues[first])
                            interface_residue_d[first][entity_pair[second]] = copy(entity_residues[second])
                        else:
                            # Entities are properly indexed, extend the first index
                            interface_residue_d[first][entity_pair[first]].extend(entity_residues[first])
                            # Because of combinations with replacement entity search, the second entity is not in
                            # the second index, UNLESS the self Entity (SYMMETRY) is in FIRST (as above)
                            # Therefore we add below without checking for overwrite
                            interface_residue_d[second][entity_pair[second]] = copy(entity_residues[second])
                            # if _self:  # This can't happen, it would VIOLATES RULES
                            #     interface_residue_d['self'][1] = _self
                    # we have interface assigned and the entity is not in the first index, which means it may
                    # be in the second, it may not
                    elif entity_pair[first] in interface_residue_d[second]:
                        # it is, add it to the second index
                        interface_residue_d[second][entity_pair[first]].extend(entity_residues[first])
                        # also add it's partner entity to the first index
                        # if entity_pair[1] in interface_residue_d[first]:  # Can this ever be True? Can't find a case
                        #     interface_residue_d[first][entity_pair[1]].extend(entity_residues[1])
                        # else:  # Ex5
                        interface_residue_d[first][entity_pair[second]] = copy(entity_residues[second])
                        if _self:
                            interface_residue_d['self'][first] = _self
                    # CHECK INDEX 2
                    elif entity_pair[second] in interface_residue_d[second]:
                        # this is possible (A:D) (C:D)
                        interface_residue_d[second][entity_pair[second]].extend(entity_residues[second])
                        # if entity_pair[first] in interface_residue_d[first]: # NOT POSSIBLE ALREADY CHECKED
                        #     interface_residue_d[first][entity_pair[first]].extend(entity_residues[first])
                        # else:
                        interface_residue_d[first][entity_pair[first]] = copy(entity_residues[first])
                        if _self:  # Ex3
                            interface_residue_d['self'][first] = _self
                        # interface_residue_d['self'][first] = _self  # NOT POSSIBLE ALREADY CHECKED
                    elif entity_pair[second] in interface_residue_d[first]:
                        # the first Entity wasn't found in either, but both are already set, therefore it can't be a
                        # self, so the only way this works is if entity_pair[first] is further in the iterative process
                        # which is impossible, this violates the rules
                        interface_residue_d[second][entity_pair[first]] = False
                        terminate = True
                        break
                    # Neither of our indices are in the dictionary yet. We are going to add 2 entities to each interface
                    else:
                        # the first and second Entity weren't found in either, but both are already set, violation
                        interface_residue_d[first][entity_pair[first]] = False
                        interface_residue_d[second][entity_pair[second]] = False

                        terminate = True
                        break

            interface1, interface2, self_check = tuple(interface_residue_d.values())
            if len(interface1) == 2 and len(interface2) == 2 and all(self_check):
                pass
            elif len(interface1) == 1 or len(interface2) == 1:
                pass
            else:
                terminate = True
                break

        if terminate:
            self.log.critical('%s: The set of interfaces found during interface search generated a topologically '
                              'disallowed combination.\n\t %s\n This cannot be modelled by a simple split for residues '
                              'on either side while respecting the requirements of polymeric Entities. '
                              '%sPlease correct your design_selectors to reduce the number of Entities you are '
                              'attempting to design. This issue may be global if your designs are very similar'
                              % (self.name,
                                 ' | '.join(':'.join(entity.name for entity in interface_entities)
                                            for key, interface_entities in interface_residue_d.items()
                                            if key != 'self'),
                                 'Symmetry was set which may have influenced this unfeasible topology, you can try to '
                                 'set it False. ' if self.symmetry else ''))
            raise DesignError('The specified interfaces generated a topologically disallowed combination! Check the log'
                              ' for more information.')

        for key, entity_residues in interface_residue_d.items():
            if key == 'self':
                continue
            all_residues = [(residue, entity) for entity, residues in entity_residues.items() for residue in residues]
            self.split_interface_residues[key + 1] = sorted(all_residues, key=lambda res_ent: res_ent[0].number)

        # self.split_interface_residues = \
        #     {key + 1: [(residue, entity) for entity, residues in entity_residues.items() for residue in residues]
        #      for key, entity_residues in interface_residue_d.items() if key != 'self'}
        # self.split_interface_residues = {number: sorted(residue_entities, key=lambda tup: tup[0].number)
        #                                  for number, residue_entities in self.split_interface_residues.items()}
        #
        # self.split_interface_residues = {number: ','.join('%d%s' % residue_entity
        #                                 for residue_entity in sorted(residue_entities, key=lambda tup: tup[0].number))
        #                                  for number, residue_entities in self.split_interface_residues.items()}
        if self.split_interface_residues[1] == '':
            raise DesignError('Interface was unable to be split because no residues were found on one side of the'
                              ' interface!')
        else:
            self.log.debug('The interface is split as:\n\tinterface 1: %s'
                           % '\n\tinterface 2: '.join(','.join('%d%s' % (res.number, ent.chain_id)
                                                               for res, ent in residues_entities)
                                                      for residues_entities in self.split_interface_residues.values()))

    def interface_secondary_structure(self):
        """From a split interface, curate the secondary structure topology for each

        Keyword Args:
            source_db=None (Database): A Database object connected to secondary structure db
            source_dir=None (str): The location of the directory containing Stride files
        """
        pose_secondary_structure = ''
        for entity in self.active_entities:
            if not entity.secondary_structure:
                if self.source_db:
                    parsed_secondary_structure = self.source_db.stride.retrieve_data(name=entity.name)
                    if parsed_secondary_structure:
                        entity.fill_secondary_structure(secondary_structure=parsed_secondary_structure)
                    else:
                        entity.stride(to_file=self.source_db.stride.store(entity.name))
                # if source_dir:
                #     entity.parse_stride(os.path.join(source_dir, '%s.stride' % entity.name))
                else:
                    entity.stride()
            pose_secondary_structure += entity.secondary_structure

        # increment a secondary structure index which changes with every secondary structure transition
        # simultaneously, map the secondary structure type to an array of pose length (offset for residue number)
        self.ss_index_array.clear(), self.ss_type_array.clear()  # clear any information if it exists
        self.ss_type_array.append(pose_secondary_structure[0])
        ss_increment_index = 0
        self.ss_index_array.append(ss_increment_index)
        for prior_idx, ss_type in enumerate(pose_secondary_structure[1:], 0):
            if ss_type != pose_secondary_structure[prior_idx]:
                self.ss_type_array.append(ss_type)
                ss_increment_index += 1
            self.ss_index_array.append(ss_increment_index)

        for number, residues_entities in self.split_interface_residues.items():
            self.split_interface_ss_elements[number] = []
            for residue, entity in residues_entities:
                self.split_interface_ss_elements[number].append(self.ss_index_array[residue.number - 1])

        self.log.debug('Found interface secondary structure: %s' % self.split_interface_ss_elements)

    def interface_design(self, evolution=True, fragments=True, query_fragments=False, fragment_source=None,
                         write_fragments=True, frag_db='biological_interfaces', des_dir=None):  # Todo deprec. des_dir
        """Take the provided PDB, and use the ASU to compute calculations relevant to interface design.

        This process identifies the ASU (if one is not explicitly provided, enables Pose symmetry,

        Sets:
            design_dir.info['fragments'] to True is fragments are queried
        """
        # Ensure ASU. This should be done on loading from PDB file with Pose.from_asu()/Pose.from_pdb()
        # save self.asu to design_dir.asu now that we have cleaned any chain issues and renumbered residues
        # self.pdb.write(out_path=design_dir.asu)

        # if design_dir.nano:
        #     # num_chains = len(self.pdb.chain_id_list)
        #     # if num_chains != len(design_dir.oligomers):
        #     #     # oligomer_file = glob(os.path.join(des_dir.path, pdb_codes[0] + '_tx_*.pdb'))
        #     #     # assert len(oligomer_file) == 1, 'More than one matching file found with %s' % pdb_codes[0] + '_tx_*.pdb'
        #     #     # # assert len(oligomer_file) == 1, '%s: More than one matching file found with %s' % \
        #     #     # #                                 (des_dir.path, pdb_codes[0] + '_tx_*.pdb')
        #     #     # first_oligomer = PDB(file=oligomer_file[0])
        #     #     # # first_oligomer = SDUtils.read_pdb(oligomer_file[0])
        #     #     # # find the number of ATOM records for template_pdb chain1 using the same oligomeric chain as model
        #     #     # for atom_idx in range(len(first_oligomer.chain(template_pdb.chain_id_list[0]))):
        #     #     for atom_idx in range(len(design_dir.oligomers.chain(self.pdb.entity(1)))):
        #     #         self.pdb.atoms[atom_idx].chain = self.pdb.chain_id_list[0].lower()
        #     #     self.pdb.chain_id_list = [self.pdb.chain_id_list[0].lower(), self.pdb.chain_id_list[0]]
        #     #     num_chains = len(self.pdb.chain_id_list)
        #     #     logger.warning(
        #     #         '%s: Incorrect chain count: %d. Chains probably have the same id! Temporarily changing IDs\'s to'
        #     #         ' %s' % (design_dir.path, num_chains, self.pdb.chain_id_list))
        #     #     # Save the renamed chain PDB to the source.pdb
        #     #     self.pdb.write(out_path=design_dir.source)
        #
        #     assert len(design_dir.oligomers) == num_chains, \
        #         'Number of chains \'%d\' in ASU doesn\'t match number of building blocks \'%d\'' \
        #         % (num_chains, len(design_dir.oligomers))
        # else:
        #     # sdf_file_name = os.path.join(os.path.dirname(oligomer[name].filepath), '%s.sdf' % oligomer[name].name)
        #     # sym_definition_files[name] = oligomer[name].make_sdf(out_path=sdf_file_name, modify_sym_energy=True)
        self.log.debug('Entities: %s' % ', '.join(entity.name for entity in self.entities))
        self.log.debug('Active Entities: %s' % ', '.join(entity.name for entity in self.active_entities))
        # self.log.debug('Designable Residues: %s' % ', '.join(entity.name for entity in self.design_selector_indices))
        # self.log.info('Symmetry is: %s' % symmetry)
        # if symmetry and isinstance(symmetry, dict):
        #     self.set_symmetry(**symmetry)

        # we get interface residues for the designable entities as well as interface_topology at DesignDirectory level
        if fragments:
            if query_fragments:  # search for new fragment information
                self.generate_interface_fragments(out_path=des_dir.frags, write_fragments=write_fragments)
            else:  # No fragment query, add existing fragment information to the pose
                if not self.frag_db:
                    self.connect_fragment_database(source=frag_db, init=False)  # no need to initialize
                    # Attach an existing FragmentDB to the Pose
                    self.attach_fragment_database(db=self.frag_db)
                    for entity in self.entities:
                        entity.attach_fragment_database(db=self.frag_db)

                if fragment_source is None:
                    raise DesignError('Fragments were set for design but there were none found! Try including '
                                      '--query_fragments in your input flags and rerun this command or generate '
                                      'them separately with \'%s %s\''
                                      % (PUtils.program_command, PUtils.generate_fragments))

                # Must provide des_dir.fragment_observations then specify whether the Entity in question is from the
                # mapped or paired chain (entity1 is mapped, entity2 is paired from Nanohedra). Then, need to renumber
                # fragments to Pose residue numbering when added to fragment queries
                # if design_dir.nano:  # Todo depreciate this check as Nanohedra outputs will be in Pose Numbering
                #     if len(self.entities) > 2:  # Todo compatible with > 2 entities
                #         raise DesignError('Not able to solve fragment/residue membership with more than 2 Entities!')
                #     self.log.debug('Fragment data found in Nanohedra docking. Solving fragment membership for '
                #                    'Entity\'s: %s by PDB numbering correspondence'
                #                    % ','.join(entity.name for entity in self.entities))
                #     self.add_fragment_query(entity1=self.entities[0], entity2=self.entities[1], query=fragment_source,
                #                             pdb_numbering=True)
                # else:  # assuming the input is in Pose numbering!
                self.log.debug('Fragment data found from prior query. Solving query index by Pose numbering/Entity '
                               'matching')
                self.add_fragment_query(query=fragment_source)

            for query_pair, fragment_info in self.fragment_queries.items():
                self.log.debug('Query Pair: %s, %s\n\tFragment Info:%s' % (query_pair[0].name, query_pair[1].name,
                                                                           fragment_info))
                for query_idx, entity in enumerate(query_pair):
                    # # Attach an existing FragmentDB to the Pose
                    # entity.connect_fragment_database(location=frag_db, db=design_dir.frag_db)
                    entity.attach_fragment_database(db=self.frag_db)
                    entity.assign_fragments(fragments=fragment_info,
                                            alignment_type=SequenceProfile.idx_to_alignment_type[query_idx])
        for entity in self.entities:
            # entity.retrieve_sequence_from_api(entity_id=entity)  # Todo
            # TODO Insert loop identifying comparison of SEQRES and ATOM before SeqProf.calculate_design_profile()
            if entity not in self.active_entities:  # we shouldn't design, add a null profile instead
                entity.add_profile(null=True)
            else:
                if self.source_db:
                    entity.sequence_file = self.source_db.sequences.retrieve_file(name=entity.name)
                    entity.evolutionary_profile = self.source_db.hhblits_profiles.retrieve_data(name=entity.name)
                    profiles_path = self.source_db.hhblits_profiles.location
                else:
                    profiles_path = des_dir.profiles
                if not entity.sequence_file:
                    entity.write_fasta_file(entity.reference_sequence, name=entity.name, out_path=des_dir.sequences)
                entity.add_profile(evolution=evolution, fragments=fragments, out_path=profiles_path)

        # Update DesignDirectory with design information # Todo include in DesignDirectory initialization by args?
        # This info is pulled out in AnalyzeOutput from Rosetta currently

        if fragments:  # set pose.fragment_profile by combining entity frag profile into single profile
            self.combine_fragment_profile([entity.fragment_profile for entity in self.entities])
            self.fragment_pssm_file = self.write_pssm_file(self.fragment_profile, PUtils.fssm, out_path=des_dir.data)
            # design_dir.info['fragment_profile'] = self.fragment_pssm_file
            # design_dir.info['fragment_database'] = frag_db
            # self.log.debug('Fragment Specific Scoring Matrix: %s' % str(self.fragment_profile))
            # this dictionary is removed of all entries that are not fragment populated.
            clean_fragment_profile = dict(item for item in self.fragment_profile.items()
                                          if item[1].get('stats', (None,))[0])  # must be a fragment observation
            self.interface_data_file = \
                pickle_object(clean_fragment_profile, '%s_fragment_profile' % self.frag_db.source,
                              out_path=des_dir.data)
            # design_dir.info['fragment_data'] = self.interface_data_file

        if evolution:  # set pose.evolutionary_profile by combining entity evo profile into single profile
            self.combine_pssm([entity.evolutionary_profile for entity in self.entities])
            # self.log.debug('Position Specific Scoring Matrix: %s' % str(self.evolutionary_profile))
            self.pssm_file = self.write_pssm_file(self.evolutionary_profile, PUtils.pssm, out_path=des_dir.data)
            # design_dir.info['evolutionary_profile'] = self.pssm_file

        self.combine_profile([entity.profile for entity in self.entities])
        # self.log.debug('Design Specific Scoring Matrix: %s' % str(self.profile))
        self.design_pssm_file = self.write_pssm_file(self.profile, PUtils.dssm, out_path=des_dir.data)
        # design_dir.info['design_profile'] = self.design_pssm_file
        # -------------------------------------------------------------------------
        # self.solve_consensus()
        # -------------------------------------------------------------------------

    def return_fragment_observations(self):
        """Return the fragment observations identified on the pose regardless of Entity binding

        Returns:
            (list[dict]): [{'mapped': int, 'paired': int, 'cluster': str, 'match': float}, ...]
        """
        observations = []
        # {(ent1, ent2): [{mapped: res_num1, paired: res_num2, cluster: id, match: score}, ...], ...}
        for query_pair, fragment_matches in self.fragment_queries.items():
            observations.extend(fragment_matches)

        return observations

    def return_fragment_metrics(self, fragments=None, by_interface=False, entity1=None, entity2=None, by_entity=False):
        """From self.fragment_queries, return the specified fragment metrics. By default returns the entire Pose

        Keyword Args:
            metrics=None (list): A list of calculated metrics
            by_interface=False (bool): Return fragment metrics for each particular interface found in the Pose
            entity1=None (Entity): The first Entity object to identify the interface if per_interface=True
            entity2=None (Entity): The second Entity object to identify the interface if per_interface=True
            by_entity=False (bool): Return fragment metrics for each Entity found in the Pose
        Returns:
            (dict): {query1: {all_residue_score (Nanohedra), center_residue_score, total_residues_with_fragment_overlap,
            central_residues_with_fragment_overlap, multiple_frag_ratio, fragment_content_d}, ... }
        """
        # Todo consolidate return to (dict[(dict)]) like by_entity
        # Todo once moved to pose, incorporate these?
        #  'fragment_cluster_ids': ','.join(clusters),
        #  'total_interface_residues': total_residues,
        #  'percent_residues_fragment_total': percent_interface_covered,
        #  'percent_residues_fragment_center': percent_interface_matched,

        if fragments:
            return format_fragment_metrics(calculate_match_metrics(fragments))

        # self.calculate_fragment_query_metrics()  # populates self.fragment_metrics
        if not self.fragment_metrics:
            for query_pair, fragment_matches in self.fragment_queries.items():
                self.fragment_metrics[query_pair] = calculate_match_metrics(fragment_matches)

        if by_interface:
            if entity1 and entity2:
                for query_pair, metrics in self.fragment_metrics.items():
                    if not metrics:
                        continue
                    if (entity1, entity2) in query_pair or (entity2, entity1) in query_pair:
                        return format_fragment_metrics(metrics)
                self.log.info('Couldn\'t locate query metrics for Entity pair %s, %s' % (entity1.name, entity2.name))
            else:
                self.log.error('%s: entity1 or entity1 can\'t be None!' % self.return_fragment_metrics.__name__)

            return fragment_metric_template
        elif by_entity:
            metric_d = {}
            for query_pair, metrics in self.fragment_metrics.items():
                if not metrics:
                    continue
                for idx, entity in enumerate(query_pair):
                    if entity not in metric_d:
                        metric_d[entity] = fragment_metric_template

                    align_type = SequenceProfile.idx_to_alignment_type[idx]
                    metric_d[entity]['center_residues'].union(metrics[align_type]['center']['residues'])
                    metric_d[entity]['total_residues'].union(metrics[align_type]['total']['residues'])
                    metric_d[entity]['nanohedra_score'] += metrics[align_type]['total']['score']
                    metric_d[entity]['nanohedra_score_center'] += metrics[align_type]['center']['score']
                    metric_d[entity]['multiple_fragment_ratio'] += metrics[align_type]['multiple_ratio']
                    metric_d[entity]['number_fragment_residues_total'] += metrics[align_type]['total']['number']
                    metric_d[entity]['number_fragment_residues_center'] += metrics[align_type]['center']['number']
                    metric_d[entity]['number_fragments'] += metrics['total']['observations']
                    metric_d[entity]['percent_fragment_helix'] += metrics[align_type]['index_count'][1]
                    metric_d[entity]['percent_fragment_strand'] += metrics[align_type]['index_count'][2]
                    metric_d[entity]['percent_fragment_coil'] += (metrics[align_type]['index_count'][3] +
                                                                  metrics[align_type]['index_count'][4] +
                                                                  metrics[align_type]['index_count'][5])
            for entity in metric_d:
                metric_d[entity]['percent_fragment_helix'] /= metric_d[entity]['number_fragments']
                metric_d[entity]['percent_fragment_strand'] /= metric_d[entity]['number_fragments']
                metric_d[entity]['percent_fragment_coil'] /= metric_d[entity]['number_fragments']

            return metric_d
        else:
            metric_d = fragment_metric_template
            for query_pair, metrics in self.fragment_metrics.items():
                if not metrics:
                    continue
                metric_d['center_residues'].union(
                    metrics['mapped']['center']['residues'].union(metrics['paired']['center']['residues']))
                metric_d['total_residues'].union(
                    metrics['mapped']['total']['residues'].union(metrics['paired']['total']['residues']))
                metric_d['nanohedra_score'] += metrics['total']['total']['score']
                metric_d['nanohedra_score_center'] += metrics['total']['center']['score']
                metric_d['multiple_fragment_ratio'] += metrics['total']['multiple_ratio']
                metric_d['number_fragment_residues_total'] += metrics['total']['total']['number']
                metric_d['number_fragment_residues_center'] += metrics['total']['center']['number']
                metric_d['number_fragments'] += metrics['total']['observations']
                metric_d['percent_fragment_helix'] += metrics['total']['index_count'][1]
                metric_d['percent_fragment_strand'] += metrics['total']['index_count'][2]
                metric_d['percent_fragment_coil'] += (metrics['total']['index_count'][3] +
                                                      metrics['total']['index_count'][4] +
                                                      metrics['total']['index_count'][5])
            try:
                metric_d['percent_fragment_helix'] /= (metric_d['number_fragments'] * 2)  # account for 2x observations
                metric_d['percent_fragment_strand'] /= (metric_d['number_fragments'] * 2)  # account for 2x observations
                metric_d['percent_fragment_coil'] /= (metric_d['number_fragments'] * 2)  # account for 2x observations
            except ZeroDivisionError:
                metric_d['percent_fragment_helix'], metric_d['percent_fragment_strand'], \
                    metric_d['percent_fragment_coil'] = 0, 0, 0

            return metric_d

    # def calculate_fragment_query_metrics(self):
    #     """From the profile's fragment queries, calculate and store the query metrics per query"""
    #     for query_pair, fragment_matches in self.fragment_queries.items():
    #         self.fragment_metrics[query_pair] = calculate_match_metrics(fragment_matches)

    # def return_fragment_info(self):
    #     clusters, residue_numbers, match_scores = [], [], []
    #     for query_pair, fragments in self.fragment_queries.items():
    #         for query_idx, entity_name in enumerate(query_pair):
    #             clusters.extend([fragment['cluster'] for fragment in fragments])

    def renumber_fragments_to_pose(self, fragments):
        for idx, fragment in enumerate(fragments):
            # if self.pdb.residue_from_pdb_numbering():
            # only assign the new fragment number info to the fragments if the residue is found
            map_pose_number = self.pdb.residue_number_from_pdb(fragment['mapped'])
            fragment['mapped'] = map_pose_number if map_pose_number else fragment['mapped']
            pair_pose_number = self.pdb.residue_number_from_pdb(fragment['paired'])
            fragment['paired'] = pair_pose_number if pair_pose_number else fragment['paired']
            # fragment['mapped'] = self.pdb.residue_number_from_pdb(fragment['mapped'])
            # fragment['paired'] = self.pdb.residue_number_from_pdb(fragment['paired'])
            fragments[idx] = fragment

        return fragments

    def add_fragment_query(self, entity1=None, entity2=None, query=None, pdb_numbering=False):
        """For a fragment query loaded from disk between two entities, add the fragment information to the Pose"""
        # Todo This function has logic pitfalls if residue numbering is in PDB format. How easy would
        #  it be to refactor fragment query to deal with the chain info from the frag match file?
        if pdb_numbering:  # Renumber self.fragment_map and self.fragment_profile to Pose residue numbering
            query = self.renumber_fragments_to_pose(query)
            # for idx, fragment in enumerate(fragment_source):
            #     fragment['mapped'] = self.pdb.residue_number_from_pdb(fragment['mapped'])
            #     fragment['paired'] = self.pdb.residue_number_from_pdb(fragment['paired'])
            #     fragment_source[idx] = fragment
            if entity1 and entity2 and query:
                self.fragment_queries[(entity1, entity2)] = query
        else:
            entity_pairs = [(self.pdb.entity_from_residue(fragment['mapped']),
                             self.pdb.entity_from_residue(fragment['paired'])) for fragment in query]
            if all([all(pair) for pair in entity_pairs]):
                for entity_pair, fragment in zip(entity_pairs, query):
                    if entity_pair in self.fragment_queries:
                        self.fragment_queries[entity_pair].append(fragment)
                    else:
                        self.fragment_queries[entity_pair] = [fragment]
            else:
                raise DesignError('%s: Couldn\'t locate Pose Entities passed by residue number. Are the residues in '
                                  'Pose Numbering? This may be occurring due to fragment queries performed on the PDB '
                                  'and not explicitly searching using pdb_numbering = True. Retry with the appropriate'
                                  ' modifications' % self.add_fragment_query.__name__)

    def connect_fragment_database(self, source=None, init=False, **kwargs):  # Todo Clean up
        """Generate a new connection. Initialize the representative library by passing init=True"""
        if not source:  # Todo fix once multiple are available
            source = 'biological_interfaces'
        self.frag_db = FragmentDatabase(source=source, init_db=init)
        #                               source=source, init_db=init_db)

    def generate_interface_fragments(self, write_fragments=True, out_path=None, new_db=False):
        """Using the attached fragment database, generate interface fragments between the Pose interfaces

        Keyword Args:
            write_fragments=True (bool): Whether or not to write the located fragments
            out_path=None (str): The location to write each fragment file
            new_db=False (bool): Whether a fragment database should be initialized for the interface fragment search
        """
        if not self.frag_db:  # There is no fragment database connected
            # Connect to a new DB, Todo parameterize which one should be used with source=
            self.connect_fragment_database(init=True)  # default init=False, we need an initiated one to generate frags

        if not self.interface_residues:
            self.find_and_split_interface()  # shouldn't occur with the set up on 3/17/21, upstream funcs call this 1st
            # for entity_pair in combinations_with_replacement(self.active_entities, 2):
            #     self.find_interface_residues(*entity_pair)

        for entity_pair in combinations_with_replacement(self.active_entities, 2):
            self.log.debug('Querying Entity pair: %s, %s for interface fragments'
                           % tuple(entity.name for entity in entity_pair))
            self.query_interface_for_fragments(*entity_pair)

        if write_fragments:
            self.write_fragment_pairs(self.fragment_pairs, out_path=out_path)
            frag_file = os.path.join(out_path, PUtils.frag_text_file)
            if os.path.exists(frag_file):
                os.system('rm %s' % frag_file)  # ensure old file is removed before new write
            for match_count, (ghost_frag, surface_frag, match) in enumerate(self.fragment_pairs, 1):
                write_frag_match_info_file(ghost_frag=ghost_frag, matched_frag=surface_frag,
                                           overlap_error=z_value_from_match_score(match),
                                           match_number=match_count, out_path=out_path)

    def write_fragment_pairs(self, ghostfrag_surffrag_pairs, out_path=os.getcwd()):
        for idx, (interface_ghost_frag, interface_mono_frag, match_score) in enumerate(ghostfrag_surffrag_pairs, 1):
            fragment, _ = dictionary_lookup(self.frag_db.paired_frags, interface_ghost_frag.get_ijk())
            trnsfmd_fragment = fragment.return_transformed_copy(**interface_ghost_frag.aligned_fragment.transformation)
            trnsfmd_fragment.write(out_path=os.path.join(out_path, '%d_%d_%d_fragment_match_%d.pdb'
                                                         % (*interface_ghost_frag.get_ijk(), idx)))
            # interface_ghost_frag.structure.write(out_path=os.path.join(out_path, '%d_%d_%d_fragment_overlap_match_%d.pdb'
            #                                                            % (*interface_ghost_frag.get_ijk(), idx)))

    def return_symmetry_parameters(self):
        """Return the symmetry parameters from a SymmetricModel

        Returns:
            (dict): {symmetry: (str), dimension: (int), uc_dimensions: (list), expand_matrices: (list[list])}
        """
        return {'symmetry': self.__dict__['symmetry'],
                'uc_dimensions': self.__dict__['uc_dimensions'],
                'expand_matrices': self.__dict__['expand_matrices'],
                'dimension': self.__dict__['dimension']}

    def debug_pdb(self, tag=None):
        """Write out all Structure objects for the Pose PDB"""
        with open('%sDEBUG_POSE_PDB_%s.pdb' % ('%s_' % tag if tag else '', self.pdb.name), 'w') as f:
            idx = 0
            for ent_idx, entity in enumerate(self.pdb.entities, 1):
                f.write('REMARK 999   Entity %d - ID %s\n' % (ent_idx, entity.name))
                entity.write(file_handle=f, chain=Structure.available_letters[idx])
                idx += 1
                for ch_idx, chain in enumerate(entity.chains, 1):
                    f.write('REMARK 999   Entity %d - ID %s   Chain %d - ID %s\n'
                            % (ent_idx, entity.name, ch_idx, chain.chain_id))
                    chain.write(file_handle=f, chain=Structure.available_letters[idx])
                    idx += 1

    # def get_interface_surface_area(self):
    #     # pdb1_interface_sa = entity1.get_surface_area_residues(entity1_residue_numbers)
    #     # pdb2_interface_sa = entity2.get_surface_area_residues(self.interface_residues or entity2_residue_numbers)
    #     # interface_buried_sa = pdb1_interface_sa + pdb2_interface_sa
    #     return


def subdirectory(name):  # TODO PDBdb
    return name


def fetch_pdb(pdb, out_dir=os.getcwd(), asu=False):
    """Download pdbs from a file, a supplied list, or a single entry

    Args:
        pdb (union[str, list]): PDB's of interest. If asu=False, code_# is format for biological assembly specific pdb.
            Ex: 1bkh_2 fetches 1bkh biological assembly 2
    Keyword Args:
        out_dir=os.getcwd() (str): The location to download files to
        asu=False (bool): Whether or not to download the asymmetric unit file
    Returns:
        (str): Filename of the retrieved file, if pdb is a list then will only return the last filename
    """
    file_name = None
    for pdb in to_iterable(pdb):
        clean_pdb = pdb[0:4].lower()
        if asu:
            assembly = ''
        else:
            assembly = pdb[-3:]
            try:
                assembly = assembly.split('_')[1]
            except IndexError:
                assembly = '1'

        clean_pdb = '%s.pdb%s' % (clean_pdb, assembly)
        file_name = os.path.join(out_dir, clean_pdb)
        current_file = glob(file_name)
        # current_files = os.listdir(location)
        # if clean_pdb not in current_files:
        if not current_file:  # glob will return an empty list if the file is missing and therefore should be downloaded
            # Always returns files in lowercase
            status = os.system('wget -q -O %s https://files.rcsb.org/download/%s' % (file_name, clean_pdb))
            # TODO subprocess.POPEN()
            if status != 0:
                logger.error('PDB download failed for: %s' % pdb)

            # file_request = requests.get('https://files.rcsb.org/download/%s' % clean_pdb)
            # if file_request.status_code == 200:
            #     with open(file_name, 'wb') as f:
            #         f.write(file_request.content)
            # else:
            #     logger.error('PDB download failed for: %s' % pdb)

    return file_name


# def fetch_pdbs(codes, location=PUtils.pdb_db):  # UNUSED
#     """Fetch PDB object of each chain from PDBdb or PDB server
#
#     Args:
#         codes (iter): Any iterable of PDB codes
#     Keyword Args:
#         location= : Location of the  on disk
#     Returns:
#         (dict): {pdb_code: PDB.py object, ...}
#     """
#     if PUtils.pdb_source == 'download_pdb':
#         get_pdb = download_pdb
#         # doesn't return anything at the moment
#     else:
#         get_pdb = (lambda pdb_code, dummy: glob(os.path.join(PUtils.pdb_location, subdirectory(pdb_code),
#                                                              '%s.pdb' % pdb_code)))
#         # returns a list with matching file (should only be one)
#     oligomers = {}
#     for code in codes:
#         pdb_file_name = get_pdb(code, location=des_dir.pdbs)
#         assert len(pdb_file_name) == 1, 'More than one matching file found for pdb code %s' % code
#         oligomers[code] = PDB(file=pdb_file_name[0])
#         oligomers[code].name = code
#         oligomers[code].reorder_chains()
#
#     return oligomers


def fetch_pdb_file(pdb_code, location=PUtils.pdb_db, out_dir=os.getcwd(), asu=True):
    """Fetch PDB object of each chain from PDBdb or PDB server

    Args:
        pdb_code (iter): The PDB ID/code. If the biological assembly is desired, supply 1ABC_1 where '_1' is assembly ID
    Keyword Args:
        location=PathUtils.pdb_db (str): Location of a local PDB mirror if one is linked on disk
        out_dir=os.getcwd() (str): The location to save retrieved files if fetched from PDB
        asu=False (bool): Whether to fetch the ASU
    Returns:
        (str): path/to/your_pdb.pdb if located/downloaded successfully (alphabetical characters in lowercase)
    """
    if location == PUtils.pdb_db and asu:
        get_pdb = (lambda pdb_code, out_dir=None, asu=None:
                   glob(os.path.join(out_dir, 'pdb%s.ent' % pdb_code.split('_')[0].lower())))
        #                                      remove any biological assembly data and make lowercase
        # Cassini format is above, KM local pdb and the escher PDB mirror is below
        # get_pdb = (lambda pdb_code, dummy: glob(os.path.join(PUtils.pdb_db, subdirectory(pdb_code),
        #                                                      '%s.pdb' % pdb_code)))
    else:
        get_pdb = fetch_pdb

    # return a list with matching files (should only be one)
    pdb_file = get_pdb(pdb_code, out_dir=out_dir, asu=asu)
    if not pdb_file:
        logger.warning('No matching file found for PDB: %s' % pdb_code)
    # elif len(pdb_file) > 1:
    #     logger.warning('More than one matching file found for PDB \'%s\'. Retrieving %s' % (pdb_code, pdb_file[0]))
    else:
        return pdb_file  # [0]


def construct_cb_atom_tree(pdb1, pdb2, distance=8):
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
    pdb1_coords = np.array(pdb1.get_cb_coords())  # InclGlyCA=gly_ca))
    pdb2_coords = np.array(pdb2.get_cb_coords())  # InclGlyCA=gly_ca))

    # Construct CB Tree for PDB1
    pdb1_tree = BallTree(pdb1_coords)

    # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
    return pdb1_tree.query_radius(pdb2_coords, distance)


def find_interface_pairs(pdb1, pdb2, distance=8):  # , gly_ca=True):
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
    pdb1_cb_indices = pdb1.get_cb_indices()  # InclGlyCA=gly_ca)
    pdb2_cb_indices = pdb2.get_cb_indices()  # InclGlyCA=gly_ca)

    # Map Coordinates to Residue Numbers
    interface_pairs = []
    for pdb2_index in range(len(query)):
        if query[pdb2_index].tolist() != list():
            pdb2_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_index]].residue_number
            for pdb1_index in query[pdb2_index]:
                pdb1_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_index]].residue_number
                interface_pairs.append((pdb1_res_num, pdb2_res_num))

    return interface_pairs


# def split_interface_pairs(interface_pairs):
#     residues1, residues2 = zip(*interface_pairs)
#     return sorted(set(residues1), key=int), sorted(set(residues2), key=int)
#
#
# def find_interface_residues(pdb1, pdb2, distance=8):
#     """Get unique residues from each pdb across an interface
#
#         Args:
#             pdb1 (PDB): First pdb to measure interface between
#             pdb2 (PDB): Second pdb to measure interface between
#         Keyword Args:
#             distance=8 (int): The distance to query in Angstroms
#         Returns:
#             (tuple(set): A tuple of interface residue sets across an interface
#     """
#     return split_interface_pairs(find_interface_pairs(pdb1, pdb2, distance=distance))


# @njit
def find_fragment_overlap_at_interface(entity1_coords, interface_frags1, interface_frags2, fragdb=None,
                                       euler_lookup=None, max_z_value=2):
    #           entity1, entity2, entity1_interface_residue_numbers, entity2_interface_residue_numbers, max_z_value=2):
    """From two Structure's, score the interface between them according to Nanohedra's fragment matching"""
    if not fragdb:
        fragdb = FragmentDB()
        fragdb.get_monofrag_cluster_rep_dict()
        fragdb.get_intfrag_cluster_rep_dict()
        fragdb.get_intfrag_cluster_info_dict()
    if not euler_lookup:
        euler_lookup = EulerLookup()

    # logger.debug('Starting Ghost Frag Lookup')
    oligomer1_bb_tree = BallTree(entity1_coords)
    interface_ghost_frags1 = []
    for frag1 in interface_frags1:
        ghostfrags = frag1.get_ghost_fragments(fragdb.indexed_ghosts, oligomer1_bb_tree)
        if ghostfrags:
            interface_ghost_frags1.extend(ghostfrags)
    logger.debug('Finished Ghost Frag Lookup')

    # Get fragment guide coordinates
    interface_ghostfrag_guide_coords = np.array([ghost_frag.guide_coords for ghost_frag in interface_ghost_frags1])
    interface_surf_frag_guide_coords = np.array([frag2.guide_coords for frag2 in interface_frags2])

    # Check for matching Euler angles
    # TODO create a stand alone function
    # logger.debug('Starting Euler Lookup')
    overlapping_ghost_indices, overlapping_surf_indices = \
        euler_lookup.check_lookup_table(interface_ghostfrag_guide_coords, interface_surf_frag_guide_coords)
    # logger.debug('Finished Euler Lookup')
    logger.debug('Found %d overlapping fragments' % len(overlapping_ghost_indices))
    # filter array by matching type for surface (i) and ghost (j) frags
    surface_type_i_array = np.array([interface_frags2[idx].i_type for idx in overlapping_surf_indices.tolist()])
    ghost_type_j_array = np.array([interface_ghost_frags1[idx].j_type for idx in overlapping_ghost_indices.tolist()])
    ij_type_match = np.where(surface_type_i_array == ghost_type_j_array, True, False)

    passing_ghost_indices = overlapping_ghost_indices[ij_type_match]
    passing_ghost_coords = interface_ghostfrag_guide_coords[passing_ghost_indices]

    passing_surf_indices = overlapping_surf_indices[ij_type_match]
    passing_surf_coords = interface_surf_frag_guide_coords[passing_surf_indices]
    logger.debug('Found %d overlapping fragments in the same i/j type' % len(passing_ghost_indices))
    # precalculate the reference_rmsds for each ghost fragment
    reference_rmsds = np.array([interface_ghost_frags1[ghost_idx].rmsd for ghost_idx in passing_ghost_indices.tolist()])
    reference_rmsds = np.where(reference_rmsds == 0, 0.01, reference_rmsds)

    # logger.debug('Calculating passing fragment overlaps by RMSD')
    all_fragment_overlap = calculate_overlap(passing_ghost_coords, passing_surf_coords, reference_rmsds,
                                             max_z_value=max_z_value)
    # logger.debug('Finished calculating fragment overlaps')
    passing_overlap_indices = np.flatnonzero(all_fragment_overlap)
    logger.debug('Found %d overlapping fragments under the %f threshold' % (len(passing_overlap_indices), max_z_value))

    interface_ghostfrags = [interface_ghost_frags1[idx] for idx in passing_ghost_indices[passing_overlap_indices].tolist()]
    interface_monofrags2 = [interface_frags2[idx] for idx in passing_surf_indices[passing_overlap_indices].tolist()]
    passing_z_values = all_fragment_overlap[passing_overlap_indices]
    match_scores = match_score_from_z_value(passing_z_values)

    return list(zip(interface_ghostfrags, interface_monofrags2, match_scores))


def get_matching_fragment_pairs_info(ghostfrag_surffrag_pairs):
    """From a ghost fragment/surface fragment pair and corresponding match score, return the pertinent interface
    information
    Args:
        ghostfrag_surffrag_pairs (list[tuple]): Observed ghost and surface fragment overlaps and their match score
    Returns:
        (list[dict])
    """
    fragment_matches = []
    for interface_ghost_frag, interface_mono_frag, match_score in ghostfrag_surffrag_pairs:
        surffrag_ch1, surffrag_resnum1 = interface_ghost_frag.get_aligned_chain_and_residue()
        surffrag_ch2, surffrag_resnum2 = interface_mono_frag.get_central_res_tup()
        fragment_matches.append(dict(zip(('mapped', 'paired', 'match', 'cluster'),
                                     (surffrag_resnum1, surffrag_resnum2,  match_score,
                                      '%d_%d_%d' % interface_ghost_frag.get_ijk()))))
    logger.debug('Fragments for Entity1 found at residues: %s' % [fragment['mapped'] for fragment in fragment_matches])
    logger.debug('Fragments for Entity2 found at residues: %s' % [fragment['paired'] for fragment in fragment_matches])

    return fragment_matches


def calculate_interface_score(interface_pdb, write=False, out_path=os.getcwd()):
    """Takes as input a single PDB with two chains and scores the interface using fragment decoration"""
    interface_name = interface_pdb.name

    entity1 = PDB.from_atoms(interface_pdb.chain(interface_pdb.chain_id_list[0]).atoms)
    entity1.update_attributes_from_pdb(interface_pdb)
    entity2 = PDB.from_atoms(interface_pdb.chain(interface_pdb.chain_id_list[-1]).atoms)
    entity2.update_attributes_from_pdb(interface_pdb)

    interacting_residue_pairs = find_interface_pairs(entity1, entity2)

    entity1_interface_residue_numbers, entity2_interface_residue_numbers = \
        get_interface_fragment_residue_numbers(entity1, entity2, interacting_residue_pairs)
    # entity1_ch_interface_residue_numbers, entity2_ch_interface_residue_numbers = \
    #     get_interface_fragment_chain_residue_numbers(entity1, entity2)

    entity1_interface_sa = entity1.get_surface_area_residues(entity1_interface_residue_numbers)
    entity2_interface_sa = entity2.get_surface_area_residues(entity2_interface_residue_numbers)
    interface_buried_sa = entity1_interface_sa + entity2_interface_sa

    interface_frags1 = entity1.get_fragments(residue_numbers=entity1_interface_residue_numbers)
    interface_frags2 = entity2.get_fragments(residue_numbers=entity2_interface_residue_numbers)
    entity1_coords = entity1.coords

    ghostfrag_surfacefrag_pairs = find_fragment_overlap_at_interface(entity1_coords, interface_frags1, interface_frags2)
    # fragment_matches = find_fragment_overlap_at_interface(entity1, entity2, entity1_interface_residue_numbers,
    #                                                       entity2_interface_residue_numbers)
    fragment_matches = get_matching_fragment_pairs_info(ghostfrag_surfacefrag_pairs)
    if write:
        write_fragment_pairs(ghostfrag_surfacefrag_pairs, out_path=out_path)

    # all_residue_score, center_residue_score, total_residues_with_fragment_overlap, \
    #     central_residues_with_fragment_overlap, multiple_frag_ratio, fragment_content_d = \
    #     calculate_match_metrics(fragment_matches)

    match_metrics = calculate_match_metrics(fragment_matches)
    # Todo
    #   'mapped': {'center': {'residues' (int): (set), 'score': (float), 'number': (int)},
    #                         'total': {'residues' (int): (set), 'score': (float), 'number': (int)},
    #                         'match_scores': {residue number(int): (list[score (float)]), ...},
    #                         'index_count': {index (int): count (int), ...},
    #                         'multiple_ratio': (float)}
    #              'paired': {'center': , 'total': , 'match_scores': , 'index_count': , 'multiple_ratio': },
    #              'total': {'center': {'score': , 'number': },
    #                        'total': {'score': , 'number': },
    #                        'index_count': , 'multiple_ratio': , 'observations': (int)}
    #              }

    total_residues = {'A': set(), 'B': set()}
    for pair in interacting_residue_pairs:
        total_residues['A'].add(pair[0])
        total_residues['B'].add(pair[1])

    total_residues = len(total_residues['A']) + len(total_residues['B'])

    percent_interface_matched = central_residues_with_fragment_overlap / total_residues
    percent_interface_covered = total_residues_with_fragment_overlap / total_residues

    interface_metrics = {'nanohedra_score': all_residue_score,
                         'nanohedra_score_central': center_residue_score,
                         'fragments': fragment_matches,
                         'multiple_fragment_ratio': multiple_frag_ratio,
                         'number_fragment_residues_central': central_residues_with_fragment_overlap,
                         'number_fragment_residues_all': total_residues_with_fragment_overlap,
                         'total_interface_residues': total_residues,
                         'number_fragments': len(fragment_matches),
                         'percent_residues_fragment_total': percent_interface_covered,
                         'percent_residues_fragment_center': percent_interface_matched,
                         'percent_fragment_helix': fragment_content_d['1'],
                         'percent_fragment_strand': fragment_content_d['2'],
                         'percent_fragment_coil': fragment_content_d['3'] + fragment_content_d['4']
                         + fragment_content_d['5'],
                         'interface_area': interface_buried_sa}

    return interface_name, interface_metrics


def get_interface_fragment_residue_numbers(pdb1, pdb2, interacting_pairs):
    # Get interface fragment information
    pdb1_residue_numbers, pdb2_residue_numbers = set(), set()
    for pdb1_central_res_num, pdb2_central_res_num in interacting_pairs:
        pdb1_res_num_list = [pdb1_central_res_num - 2, pdb1_central_res_num - 1, pdb1_central_res_num,
                             pdb1_central_res_num + 1, pdb1_central_res_num + 2]
        pdb2_res_num_list = [pdb2_central_res_num - 2, pdb2_central_res_num - 1, pdb2_central_res_num,
                             pdb2_central_res_num + 1, pdb2_central_res_num + 2]

        frag1_ca_count = 0
        for atom in pdb1.all_atoms:
            if atom.residue_number in pdb1_res_num_list:
                if atom.is_CA():
                    frag1_ca_count += 1

        frag2_ca_count = 0
        for atom in pdb2.all_atoms:
            if atom.residue_number in pdb2_res_num_list:
                if atom.is_CA():
                    frag2_ca_count += 1

        if frag1_ca_count == 5 and frag2_ca_count == 5:
            pdb1_residue_numbers.add(pdb1_central_res_num)
            pdb2_residue_numbers.add(pdb2_central_res_num)

    return pdb1_residue_numbers, pdb2_residue_numbers


def get_interface_fragment_chain_residue_numbers(pdb1, pdb2, cb_distance=8):
    """Given two PDBs, return the unique chain and interacting residue lists"""
    pdb1_cb_coords = pdb1.get_cb_coords()
    pdb1_cb_indices = pdb1.get_cb_indices()
    pdb2_cb_coords = pdb2.get_cb_coords()
    pdb2_cb_indices = pdb2.get_cb_indices()

    pdb1_cb_kdtree = BallTree(np.array(pdb1_cb_coords))

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


def get_multi_chain_interface_fragment_residue_numbers(pdb1, pdb2, interacting_pairs):
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
