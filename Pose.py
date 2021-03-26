import copy
import os
import pickle
from glob import glob
from itertools import chain as iter_chain, combinations_with_replacement
from math import sqrt, cos, sin

import numpy as np
from sklearn.neighbors import BallTree

import PathUtils as PUtils
from SymDesignUtils import to_iterable, pickle_object, DesignError, calculate_overlap, z_value_from_match_score, \
    start_log, null_log, possible_symmetries, match_score_from_z_value
from utils.GeneralUtils import write_frag_match_info_file
from utils.SymmetryUtils import valid_subunit_number, sg_cryst1_fmt_dict, pg_cryst1_fmt_dict, zvalue_dict
from classes.EulerLookup import EulerLookup
from PDB import PDB
from SequenceProfile import SequenceProfile, calculate_match_metrics
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
        super().__init__()  # **kwargs
        # self.pdb = self.models[0]
        # elif isinstance(pdb, PDB):
        if log:
            self.log = log
        elif log is None:
            self.log = null_log
        else:  # When log is explicitly passed as False, use the module logger
            self.log = logger

        if pdb:
            self.pdb = pdb
        if models and isinstance(models, list):
            self.models = models
        else:
            self.models = []

        self.number_of_models = len(self.models)

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

    def extract_cb_coords_chain(self, chain_id, InclGlyCA=False):  # TODO
        return [pdb.extract_CB_coords_chain(chain_id, InclGlyCA=InclGlyCA) for pdb in self.models]

    def set_atom_coordinates(self, new_coords):  # Todo
        for i, pdb in enumerate(self.models):
            pdb.coords = Coords(new_coords[i])
        # return [pdb.set_atom_coordinates(new_cords[i]) for i, pdb in enumerate(self.models)]


class SymmetricModel(Model):
    def __init__(self, asu=None, **kwargs):  # coords_type=None, symmetry=None, dimension=None,
        #        uc_dimensions=None, expand_matrices=None, pdb=None, models=None, log=None, number_of_models=None,
        super().__init__(**kwargs)  # log=log,
        self.asu = asu  # the pose specific asu
        # self.pdb = pdb
        # self.models = []
        # self.number_of_models = number_of_models
        # self.coords = []
        # self.model_coords = []
        self.coords_type = None  # coords_type
        self.symmetry = None  # symmetry  # also defined in PDB as self.space_group
        self.dimension = None  # dimension
        self.uc_dimensions = None  # uc_dimensions  # also defined in PDB
        self.expand_matrices = None  # expand_matrices  # Todo make expand_matrices numpy

        # if log:
        #     self.log = log
        # else:
        #     print('SymmetricModel starting log')
        #     self.log = start_log()

        # if models:
        #     self.models = models
        #     self.number_of_models = len(models)
        #     self.set_symmetry(symmetry)

        # if symmetry:
        # if self.asu:
        # kwargs.update({})
        self.set_symmetry(**kwargs)
        #    self.set_symmetry(symmetry)

    @classmethod
    def from_assembly(cls, assembly, symmetry=None):
        assert symmetry, 'Currently, can\'t initialize a symmetric model without the symmetry! Pass symmetry during ' \
                         'Class initialization. Or add a scout symmetry Class method to SymmetricModel.'
        return cls(models=assembly, **symmetry)

    @property
    def coords(self):  # Todo Model
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
    def model_coords(self):  # Todo Model
        """Return a view of the modelled Coords. These may be symmetric if a SymmetricModel"""
        return self._model_coords.coords

    @model_coords.setter
    def model_coords(self, coords):
        if isinstance(coords, Coords):
            self._model_coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

    def set_symmetry(self, expand_matrices=None, symmetry=None, cryst1=None, uc_dimensions=None, generate_assembly=True,
                     generate_symmetry_mates=False, **kwargs):
        """Set the model symmetry using the CRYST1 record, or the unit cell dimensions and the Hermann–Mauguin symmetry
        notation (in CRYST1 format, ex P 4 3 2) for the Model assembly. If the assembly is a point group,
        only the symmetry is required"""
        # if not expand_matrices:  # or not self.symmetry:
        if cryst1:
            uc_dimensions, symmetry = PDB.parse_cryst_record(cryst1_string=cryst1)

        if symmetry:
            if uc_dimensions:
                self.uc_dimensions = uc_dimensions
                self.symmetry = ''.join(symmetry.split())

            if symmetry in pg_cryst1_fmt_dict.values():  # not available yet for non-Nanohedra PG's
                self.dimension = 2
            elif symmetry in sg_cryst1_fmt_dict.values():  # not available yet for non-Nanohedra SG's
                self.dimension = 3
            elif symmetry in possible_symmetries:  # ['T', 'O', 'I']:
                self.symmetry = possible_symmetries[symmetry]
                self.dimension = 0

            elif self.uc_dimensions:
                raise DesignError('Symmetry %s is not available yet! If you didn\'t provide it, the symmetry was likely'
                                  'set from a PDB file. Get the symmetry operations from the international'
                                  ' tables and add to the pickled operators if this displeases you!' % symmetry)
            else:
                raise DesignError('Symmetry %s is not available yet! Get the cannonical symm operators from %s and add '
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
            if generate_symmetry_mates:
                self.get_assembly_symmetry_mates()

    def generate_symmetric_assembly(self, return_side_chains=True, surrounding_uc=False, generate_symmetry_mates=False,
                                    **kwargs):
        """Expand an asu in self.pdb using self.symmetry for the symmetry specification, and optional unit cell
        dimensions if self.dimension > 0. Expands assembly to complete point group, or the unit cell

        Keyword Args:
            return_side_chains=True (bool): Whether to return all side chain atoms. False returns backbone and CB atoms
            surrounding_uc=False (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        """
        if self.dimension == 0:  # symmetry in ['T', 'O', 'I']: Todo add other point groups
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
        self.number_of_models = zvalue_dict[self.symmetry]
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
        self.number_of_models = zvalue_dict[self.symmetry] * uc_number

    def return_assembly_symmetry_mates(self, **kwargs):
        """Return all symmetry mates in self.models (list[Structure]). Chain names will match the ASU"""
        count = 0
        while len(self.models) != self.number_of_models:  # Todo clarify we haven't generated the mates yet
            self.get_assembly_symmetry_mates(**kwargs)
            if count == 1:
                raise DesignError('%s: The assembly couldn\'t be returned'
                                  % self.return_assembly_symmetry_mates.__name__)
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
        if not surrounding_uc and self.symmetry in zvalue_dict:
            number_of_models = zvalue_dict[self.symmetry]  # set to the uc only
        else:
            number_of_models = self.number_of_models

        for model_idx in range(number_of_models):
            symmetry_mate_pdb = copy.copy(self.asu)
            symmetry_mate_pdb.replace_coords(self.model_coords[(model_idx * self.asu.number_of_atoms):
                                                               ((model_idx + 1) * self.asu.number_of_atoms)])
            self.models.append(symmetry_mate_pdb)

    def find_asu_equivalent_symmetry_model(self):
        """Find the asu equivalent model in the SymmetricModel. Zero-indexed

        Returns:
            (int): The index of the number of models where the ASU can be found
        """
        template_atom_coords = self.asu.residues[0].ca_coords
        template_atom_index = self.asu.residues[0].ca_index
        for model_number in range(self.number_of_models):
            if (template_atom_coords ==
                    self.model_coords[(model_number * len(self.coords)) + template_atom_index]).all():
                return model_number

        self.log.error('%s is FAILING' % self.find_asu_equivalent_symmetry_model.__name__)

    def find_intra_oligomeric_equivalent_symmetry_models(self, entity, distance=3):  # too lenient, put back to 0.5 soon
        """From an Entities Chain members, find the SymmetricModel equivalent models using Chain center or mass
        compared to the symmetric model center of mass"""
        asu_length = len(self.coords)
        entity_start, entity_end = entity.atom_indices[0], entity.atom_indices[-1]
        entity_length = entity.number_of_atoms
        entity_center_of_mass_divisor = np.full(entity_length, 1 / entity_length)
        equivalent_models = []
        for chain in entity.chains:
            chain_length = chain.number_of_atoms
            chain_center_of_mass = np.matmul(np.full(chain_length, 1 / chain_length), chain.coords)
            # print('Chain', chain_center_of_mass.astype(int))
            for model in range(self.number_of_models):
                sym_model_center_of_mass = np.matmul(entity_center_of_mass_divisor,
                                                     self.model_coords[(model * asu_length) + entity_start:
                                                                       (model * asu_length) + entity_end + 1])
                # print('Sym Model', sym_model_center_of_mass)
                # if np.allclose(chain_center_of_mass.astype(int), sym_model_center_of_mass.astype(int)):
                # if np.allclose(chain_center_of_mass, sym_model_center_of_mass):  # using np.rint()
                if np.linalg.norm(chain_center_of_mass - sym_model_center_of_mass) < distance:
                    equivalent_models.append(model)
                    break
        assert len(equivalent_models) == len(entity.chains), 'The number of equivalent models (%d) does not equal the '\
                                                             'expected number of chains (%d)!' \
                                                             % (len(equivalent_models), len(entity.chains))

        return equivalent_models

    def find_asu_equivalent_symmetry_mate_indices(self):
        """Find the asu equivalent model in the SymmetricModel. Zero-indexed

        self.model_coords must be from all atoms which by default is True
        Returns:
            (list): The indices in the SymmetricModel where the ASU is also located
        """
        model_number = self.find_asu_equivalent_symmetry_model()
        start_idx = self.asu.number_of_atoms * model_number
        end_idx = self.asu.number_of_atoms * (model_number + 1)
        return list(range(start_idx, end_idx))

    def find_intra_oligomeric_symmetry_mate_indices(self, entity):
        """Find the intra-oligomeric equivalent models in the SymmetricModel. Zero-indexed

        self.model_coords must be from all atoms which by default is True
        Returns:
            (list): The indices in the SymmetricModel where the intra-oligomeric contacts are located
        """
        model_numbers = self.find_intra_oligomeric_equivalent_symmetry_models(entity)
        oligomeric_indices = []
        for model_number in model_numbers:
            start_idx = self.pdb.number_of_atoms * model_number
            end_idx = self.pdb.number_of_atoms * (model_number + 1)
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
        return [pdb.return_transformed_copy(rotation=rot) for rot in self.expand_matrices]  # Todo change below as well

    def return_crystal_symmetry_mates(self, pdb, surrounding_uc, **kwargs):  # return_side_chains=False, surrounding_uc=False
        """Expand the backbone coordinates for every symmetric copy within the unit cells surrounding a central cell
        """
        if surrounding_uc:
            return self.return_surrounding_unit_cell_symmetry_mates(pdb, **kwargs)  # return_side_chains=return_side_chains FOR Coord expansion
        else:
            return self.return_unit_cell_symmetry_mates(pdb, **kwargs)  # # return_side_chains=return_side_chains

    def return_unit_cell_coords(self, coords, fractional=False):
        """Return the cartesian unit cell coordinates from a set of coordinates for the specified SymmetricModel"""
        asu_frac_coords = self.cart_to_frac(coords)
        coords_length = len(coords)
        model_coords = np.empty((coords_length * self.number_of_models, 3), dtype=float)
        for idx, (rot, tx) in enumerate(self.expand_matrices):
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
            extract_pdb_coords = getattr(PDB, 'coords')
        else:
            # extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
            extract_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')

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
        pdb_coords = extract_pdb_coords(pdb)
        sym_cart_coords = self.return_unit_cell_coords(pdb_coords)

        coords_length = len(pdb_coords)
        sym_mates = []
        # for coord_set in sym_cart_coords:
        for model in self.number_of_models:
            symmetry_mate_pdb = copy.copy(pdb)
            symmetry_mate_pdb.replace_coords(sym_cart_coords[model * coords_length: (model + 1) * coords_length])
            sym_mates.append(symmetry_mate_pdb)

        return sym_mates

    def return_surrounding_unit_cell_symmetry_mates(self, pdb, return_side_chains=True, **kwargs):
        """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
        if return_side_chains:  # get different function calls depending on the return type
            # extract_pdb_atoms = getattr(PDB, 'get_atoms')  # Not using. The copy() versus PDB() changes residue objs
            extract_pdb_coords = getattr(PDB, 'coords')
        else:
            # extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
            extract_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')

        if self.dimension == 3:
            z_shifts, uc_number = [0, 1, -1], 9
        elif self.dimension == 2:
            z_shifts, uc_number = [0], 27
        else:
            return None

        pdb_coords = extract_pdb_coords(pdb)
        uc_frac_coords = self.return_unit_cell_coords(pdb_coords, fractional=True)
        surrounding_frac_coords = [uc_frac_coords + [x_shift, y_shift, z_shift] for x_shift in [0, 1, -1]
                                   for y_shift in [0, 1, -1] for z_shift in z_shifts]

        surrounding_cart_coords = self.frac_to_cart(surrounding_frac_coords)

        coords_length = len(uc_frac_coords)
        sym_mates = []
        for coord_set in surrounding_cart_coords:
            for model in self.number_of_models:
                symmetry_mate_pdb = copy.copy(pdb)
                symmetry_mate_pdb.replace_coords(coord_set[(model * coords_length): ((model + 1) * coords_length)])
                sym_mates.append(symmetry_mate_pdb)

        assert len(sym_mates) == uc_number * zvalue_dict[self.symmetry], \
            'Number of models %d is incorrect! Should be %d' % (len(sym_mates), uc_number * zvalue_dict[self.symmetry])
        return sym_mates

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
        assert selected_assembly_coords == all_assembly_coords_length, '%s: Ran into an issue indexing.  ' \
                                                                       % self.symmetric_assembly_is_clash.__name__

        asu_coord_tree = BallTree(self.coords[asu_indices])
        clash_count = asu_coord_tree.two_point_correlation(self.model_coords[model_indices_without_asu], [distance])
        if clash_count[0] > 0:
            self.log.warning('%s: Found %d clashing sites! Pose is not a viable symmetric assembly'
                             % (self.pdb.name, clash_count[0]))
            return True  # clash
        else:
            return False  # no clash

    def write(self, out_path=os.getcwd(), header=None):  # , cryst1=None):  # Todo write symmetry, name, location
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle. Return the filename if
        one was written"""
        with open(out_path, 'w') as f:
            if header:
                if isinstance(header, str):
                    f.write(header)
                # if isinstance(header, Iterable):

            for model_number, model in enumerate(self.models, 1):
                f.write('{:9s}{:>4d}\n'.format('MODEL', model_number))
                for entity in model.entities:
                    chain_terminal_atom = entity.atoms[-1]
                    entity.write(file_handle=f)
                    # f.write('\n'.join(str(atom) % '{:8.3f}{:8.3f}{:8.3f}'.format(*tuple(coord))
                    #                   for atom, coord in zip(chain.atoms, self.model_coords.tolist())))
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
        sg_op_filepath = os.path.join(expand_matrix_dir, sym_type + '.pickle')
        with open(sg_op_filepath, "rb") as sg_op_file:
            sg_sym_op = pickle.load(sg_op_file)

        return sg_sym_op


class Pose(SymmetricModel, SequenceProfile):  # Model
    """A Pose is made of single or multiple PDB objects such as Entities, Chains, or other tsructures.
    All objects share a common feature such as the same symmetric system or the same general atom configuration in
    separate models across the Structure or sequence.
    """
    def __init__(self, asu=None, asu_file=None, pdb=None, pdb_file=None, **kwargs):
        #        symmetry=None, log=None,
        super().__init__(**kwargs)  # log=None,
        # the member pdbs which make up the pose. todo, combine with self.models?
        # self.pdbs = []
        self.pdbs_d = {}
        self.fragment_pairs = []
        self.design_selector_entities = set()
        self.design_selector_indices = set()
        self.required_indices = set()
        self.required_residues = None
        self.interface_residues = {}
        self.interface_split = {}
        # self.handle_flags(**kwargs)
        # self.ignore_clashes = False
        self.ignore_clashes = kwargs.get('ignore_clashes', False)
        # self.design_selector = {}
        # if kwargs.get('design_selector'):
        self.design_selector = kwargs.get('design_selector', {})
        # else:
        #     self.design_selector = {}

        if asu and isinstance(asu, Structure):
            self.asu = asu
            self.pdb = self.asu
        elif asu_file:
            self.asu = PDB.from_file(asu_file, log=self.log)  # **kwargs
            self.pdb = self.asu
        elif pdb and isinstance(pdb, Structure):
            self.pdb = pdb
        elif pdb_file:
            self.pdb = PDB.from_file(pdb_file, log=self.log)  # **kwargs

        frag_db = kwargs.get('frag_db')
        if frag_db:
            # Attach existing FragmentDB to the Pose
            self.attach_fragment_database(db=frag_db)
            # self.frag_db = frag_db  # Todo property
            for entity in self.entities:
                entity.attach_fragment_database(db=frag_db)

        euler_lookup = kwargs.get('euler_lookup')
        if euler_lookup:
            self.euler_lookup = euler_lookup
        else:
            self.euler_lookup = None
            # for entity in self.entities:  # No need to attach to entities
            #     entity.euler_lookup = euler_lookup

        symmetry_kwargs = self.pdb.symmetry.copy()
        symmetry_kwargs.update(kwargs)
        self.set_symmetry(**symmetry_kwargs)  # this will only generate an assembly if an ASU is present

    @classmethod
    def from_pdb(cls, pdb, **kwargs):  # symmetry=None,
        return cls(pdb=pdb, **kwargs)  # symmetry=None,

    @classmethod
    def from_pdb_file(cls, pdb_file, **kwargs):  # symmetry=None,
        return cls(pdb_file=pdb_file, **kwargs)  # symmetry=None,

    @classmethod
    def from_asu(cls, asu, **kwargs):  # symmetry=None,
        return cls(asu=asu, **kwargs)  # symmetry=None,

    @classmethod
    def from_asu_file(cls, asu_file, **kwargs):  # symmetry=None,
        return cls(asu_file=asu_file, **kwargs)  # symmetry=None,

    @property
    def asu(self):
        return self._asu

    @asu.setter
    def asu(self, asu):
        self._asu = asu

    @property
    def pdb(self):
        return self._pdb

    @pdb.setter
    def pdb(self, pdb):
        # self.log.debug('Adding PDB \'%s\' to pose' % pdb.name)
        self._pdb = pdb
        if not self.ignore_clashes:
            if pdb.is_clash():
                raise DesignError('%s contains Backbone clashes! See the log for more details' % self.name)
        # add structure to the SequenceProfile
        self.set_structure(pdb)
        # set up coordinate information for SymmetricModel
        self.coords = pdb._coords
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
    def number_of_atoms(self):
        return self.pdb.number_of_atoms

    @property
    def number_of_residues(self):
        return self.pdb.number_of_residues

    @property
    def center_of_mass(self):  # Todo move to Model
        """Returns: (Numpy.ndarray)"""
        return np.matmul(np.full(self.number_of_atoms, 1 / self.number_of_atoms), self.coords)

    @property
    def symmetric_centers_of_mass(self):  # Todo move to SymmetricModel
        """Returns: (Numpy.ndarray)"""
        if self.symmetry:
            return np.matmul(np.full(self.number_of_atoms, 1 / self.number_of_atoms),
                             np.split(self.model_coords, self.number_of_models))
            # for model in self.number_of_models:
            #     np.matmul(np.full(coords_length, divisor),
            #               self.model_coords[model * coords_length: (model + 1) * coords_length])
        # try:
        #     return self._center_of_mass
        # except AttributeError:
        #     self.find_center_of_mass()
        #     return self._center_of_mass

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
        self.pdbs_d[pdb.name] = pdb
        self.add_entities_to_pose(pdb)

    def add_entities_to_pose(self, pdb):
        """Add each unique Entity in a PDB to the Pose PDB. Multiple PDB's become one PDB representative"""
        current_pdb_entities = self.entities
        for idx, entity in enumerate(pdb.entities):
            current_pdb_entities.append(entity)
        self.log.debug('Found entities: %s' % list(entity.name for entity in current_pdb_entities))

        self.pdb = PDB.from_entities(current_pdb_entities, metadata=self.pdb, log=self.log)

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
            entity_required, self.required_indices = grab_indices(**self.design_selector['required'], start_with_none=True)
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

    def find_interface_pairs(self, entity1=None, entity2=None, distance=8):  # , include_glycine=True):
        """Get pairs of residue numbers that have CB atoms within a certain distance (in contact) between two named
        Entities.
        Caution!: Pose must have Coords representing all atoms as residue pairs are found using CB indices from all atoms

        Symmetry aware. If symmetry is used, by default all atomic coordinates for entity2 are symmeterized.
        design_selector aware

        Keyword Args:
            entity1=None (Entity): First entity to measure interface between
            entity2=None (Entity): Second entity to measure interface between
            distance=8 (int): The distance to query the interface in Angstroms
            include_glycine=True (bool): Whether glycine CA should be included in the tree
        Returns:
            list[tuple]: A list of interface residue numbers across the interface
        """
        # entity2_query = construct_cb_atom_tree(entity1, entity2, distance=distance)
        pdb_atoms = self.pdb.atoms
        number_of_atoms = self.number_of_atoms
        self.log.debug('Number of atoms in PDB: %s' % number_of_atoms)

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
            return None

        if self.symmetry:
            sym_string = 'symmetric '
            # get all symmetric indices
            entity2_indices = [idx + (number_of_atoms * model_number) for model_number in range(self.number_of_models)
                               for idx in entity2_indices]
            pdb_atoms = [atom for model in range(self.number_of_models) for atom in pdb_atoms]
            self.log.debug('Number of atoms in expanded assembly PDB: %s' % len(pdb_atoms))
            # pdb_residues = [residue for model in range(self.number_of_models) for residue in pdb_residues]
            # entity2_atoms = [atom for model_number in range(self.number_of_models) for atom in entity2_atoms]
            if entity2 == entity1:
                # the queried entity is the same, however we don't want interactions with the same symmetry mate or
                # intra-oligomeric contacts. Both should be removed from symmetry mate coords
                remove_indices = self.find_asu_equivalent_symmetry_mate_indices()
                # entity2_indices = [idx for idx in entity2_indices if asu_indices[0] > idx or idx > asu_indices[-1]]
                remove_indices += self.find_intra_oligomeric_symmetry_mate_indices(entity2)
                entity2_indices = list(set(entity2_indices) - set(remove_indices))
                # self.log.info('Number of Entity2 indices: %s' % len(entity2_indices))
            entity2_coords = self.model_coords[entity2_indices]  # only get the coordinate indices we want
        elif entity1 == entity2:
            # without symmetry, we can't measure this, unless intra-oligomeric contacts are desired
            return None
        else:
            sym_string = ' '
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
        #                     for entity2_idx in range(entity2_query.size) for entity1_idx in entity2_query[entity2_idx]]

        contacting_pairs = [(pdb_atoms[entity1_indices[entity1_idx]].residue_number,
                             pdb_atoms[entity2_indices[entity2_idx]].residue_number)
                            for entity2_idx in range(entity2_query.size) for entity1_idx in entity2_query[entity2_idx]]
        if entity2 != entity1:
            return contacting_pairs
        else:  # solve symmetric results for asymmetric contacts
            asymmetric_contacting_pairs, found_pairs = [], []
            for pair1, pair2 in contacting_pairs:
                # add both pair orientations (1, 2) or (2, 1) regardless
                found_pairs.extend([(pair1, pair2), (pair2, pair1)])
                # only add to contacting pair if we have never observed either
                if (pair1, pair2) not in found_pairs or (pair2, pair1) not in found_pairs:
                    asymmetric_contacting_pairs.append((pair1, pair2))

            return asymmetric_contacting_pairs

    # @staticmethod
    # def split_interface_pairs(interface_pairs):
    #     if interface_pairs:
    #         residues1, residues2 = zip(*interface_pairs)
    #         return sorted(set(residues1), key=lambda residue: residue.number), \
    #             sorted(set(residues2), key=lambda residue: residue.number)
    #     else:
    #         return [], []

    # for residue numbers
    @staticmethod
    def split_interface_pairs(interface_pairs):
        if interface_pairs:
            residues1, residues2 = zip(*interface_pairs)
            return sorted(set(residues1), key=int), sorted(set(residues2), key=int)
        else:
            return [], []

    def find_interface_residues(self, entity1=None, entity2=None, **kwargs):  # distance=8, include_glycine=True):
        """Get unique residues from each pdb across an interface provided by two Entity names

        Keyword Args:
            entity1=None (Entity): First Entity to measure interface between
            entity2=None (Entity): Second Entity to measure interface between
            distance=8 (int): The distance to query the interface in Angstroms
            include_glycine=True (bool): Whether glycine CA should be included in the tree
        Returns:
            (tuple[set]): A tuple of interface residue sets across an interface
        """
        # entity1_residues, entity2_residues = \
        entity1_residue_numbers, entity2_residue_numbers = \
            self.split_interface_pairs(self.find_interface_pairs(entity1=entity1, entity2=entity2, **kwargs))
        if not entity1_residue_numbers or not entity2_residue_numbers:
        # if not entity1_residues or not entity2_residues:
            self.log.info('Interface search at %s | %s found no interface residues' % (entity1.name, entity2.name))
            self.fragment_queries[(entity1, entity2)] = []
            self.interface_residues[(entity1, entity2)] = ([], [])
            return None
        else:
            self.log.info('At Entity %s | Entity %s interface:\t%s found residue numbers: %s'
                          # % (entity1.name, entity2.name, entity1.name, ', '.join(str(res.number)
                          #                                                        for res in entity1_residues)))
                          % (entity1.name, entity2.name, entity1.name, entity1_residue_numbers))
            self.log.info('At Entity %s | Entity %s interface:\t%s found residue numbers: %s'
                          # % (entity1.name, entity2.name, entity2.name, ', '.join(str(res.number)
                          #                                                        for res in entity2_residues)))
                          % (entity1.name, entity2.name, entity2.name, entity2_residue_numbers))

        # self.interface_residues[(entity1, entity2)] = (entity1_residues, entity2_residues)
        self.interface_residues[(entity1, entity2)] = (entity1.get_residues(numbers=entity1_residue_numbers),
                                                       entity2.get_residues(numbers=entity2_residue_numbers))
        entity_d = {1: entity1, 2: entity2}  # Todo clean
        self.log.debug('Added interface_residues: %s' % ['%d%s' % (residue.number, entity_d[idx].chain_id)
                       for idx, entity_residues in enumerate(self.interface_residues[(entity1, entity2)], 1)
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
        # if not self.interface_residues[(entity1, entity2)]:
        #     self.log.debug('At interface Entity %s | Entity %s\tMISSING interface residues'
        #                    % (entity1.name, entity2.name))
        #     return None
        surface_frags1 = entity1.get_fragments([residue.number for residue in entity1_residues],
                                               representatives=self.frag_db.reps)
        surface_frags2 = entity2.get_fragments([residue.number for residue in entity2_residues],
                                               representatives=self.frag_db.reps)

        if not surface_frags1 or not surface_frags2:
            self.log.debug('At interface Entity %s | Entity %s\tMISSING interface residues with matching fragments'
                           % (entity1.name, entity2.name))
            return None
        else:
            self.log.debug('At interface Entity %s | Entity %s\t%s has %d interface fragments'
                           % (entity1.name, entity2.name, entity1.name, len(surface_frags1)))
            self.log.debug('At interface Entity %s | Entity %s\t%s has %d interface fragments'
                           % (entity1.name, entity2.name, entity2.name, len(surface_frags2)))
        if self.symmetry:
            # even if entity1 == entity2, only need to expand the entity2 fragments due to surface/ghost frag mechanics
            # asu frag subtraction is unnecessary
            surface_frags2_nested = [self.return_symmetry_mates(frag) for frag in surface_frags2]
            surface_frags2 = list(iter_chain.from_iterable(surface_frags2_nested))
            self.log.debug('Entity 2 Symmetry expanded fragment count: %d' % len(surface_frags2))

        entity1_coords = entity1.get_backbone_and_cb_coords()  # for clash check, we only want the backbone and CB
        ghostfrag_surfacefrag_pairs = find_fragment_overlap_at_interface(entity1_coords, surface_frags1, surface_frags2,
                                                                         fragdb=self.frag_db,
                                                                         euler_lookup=self.euler_lookup)
        self.log.info('Found %d overlapping fragment pairs at the %s | %s interface.'
                      % (len(ghostfrag_surfacefrag_pairs), entity1.name, entity2.name))
        fragment_matches = get_matching_fragment_pairs_info(ghostfrag_surfacefrag_pairs)
        self.fragment_queries[(entity1, entity2)] = fragment_matches
        # add newly found fragment pairs to the existing fragment observations
        self.fragment_pairs.extend(ghostfrag_surfacefrag_pairs)  # Todo, change so not same as DesignDirectory

    def score_interface(self, entity1=None, entity2=None):
        if (entity1, entity2) not in self.fragment_queries and (entity2, entity1) not in self.fragment_queries:
            self.find_interface_residues(entity1=entity1, entity2=entity2)
            self.query_interface_for_fragments(entity1=entity1, entity2=entity2)
            self.calculate_fragment_query_metrics()

        return self.return_fragment_query_metrics(entity1=entity1, entity2=entity2, per_interface=True)

    def find_and_split_interface(self):
        """Locate the interface residues for the designable entities and split into two interfaces

        Sets:
            self.interface_split (dict): Residue/Entity id of each residue at the interface identified by interface id
            as split by topology
        """
        self.log.debug('Find and split interface using active_entities: %s' %
                       [entity.name for entity in self.active_entities])
        for entity_pair in combinations_with_replacement(self.active_entities, 2):
            self.find_interface_residues(*entity_pair)

        self.check_interface_topology()

    def check_interface_topology(self):
        first, second = 0, 1
        interface_residue_d = {first: {}, second: {}, 'self': [False, False]}
        terminate = False
        self.log.debug('Pose contains interface residues: %s' % self.interface_residues)
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
                    interface_residue_d[first][entity_pair[first]] = entity_residues[first]  # list
                    interface_residue_d[second][entity_pair[second]] = entity_residues[second]  # list
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
                            interface_residue_d[first][entity_pair[second]] = entity_residues[second]
                        else:
                            # Entities are properly indexed, extend the first index
                            interface_residue_d[first][entity_pair[first]].extend(entity_residues[first])
                            # Because of combinations with replacement entity search, the second entity is not in
                            # the second index, UNLESS the self Entity (SYMMETRY) is in FIRST (as above)
                            # Therefore we add below without checking for overwrite
                            interface_residue_d[second][entity_pair[second]] = entity_residues[second]
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
                        interface_residue_d[first][entity_pair[second]] = entity_residues[second]
                        if _self:
                            interface_residue_d['self'][first] = _self
                    # CHECK INDEX 2
                    elif entity_pair[second] in interface_residue_d[second]:
                        # this is possible (A:D) (C:D)
                        interface_residue_d[second][entity_pair[second]].extend(entity_residues[second])
                        # if entity_pair[first] in interface_residue_d[first]: # NOT POSSIBLE ALREADY CHECKED
                        #     interface_residue_d[first][entity_pair[first]].extend(entity_residues[first])
                        # else:
                        interface_residue_d[first][entity_pair[first]] = entity_residues[first]
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

        self.interface_split = \
            {key + 1: ','.join('%d%s' % (residue.number, entity.chain_id)
                               for entity, residues in interface_entities.items()
                               for residue in residues) for key, interface_entities in interface_residue_d.items()
             if key != 'self'}
        self.log.debug('The interface is split as: %s' % self.interface_split)
        if self.interface_split[1] == '':
            raise DesignError('No residues were found in an interface!')

    def interface_design(self, design_dir=None, symmetry=None, evolution=True,
                         fragments=True, query_fragments=False, write_fragments=True, fragments_exist=False,
                         frag_db='biological_interfaces',  # mask=None, output_assembly=False,
                         ):  # Todo initialize without DesignDirectory
        """Take the provided PDB, and use the ASU to compute calculations relevant to interface design.

        This process identifies the ASU (if one is not explicitly provided, enables Pose symmetry,

        Sets:
            design_dir.info['fragments'] to True is fragments are queried
        """
        if not design_dir:  # Todo
            dummy = True
            return None

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
        self.log.info('Symmetry is: %s' % symmetry)  # Todo resolve duplication with below self.symmetry
        if symmetry and isinstance(symmetry, dict):  # Todo with crysts. Not sure about the dict. Also done on __init__
            self.set_symmetry(**symmetry)

        # # get interface residues for the designable entities done at DesignDirectory level
        # self.find_and_split_interface()
        # # for entity_pair in combinations_with_replacement(self.active_entities, 2):
        # #     self.find_interface_residues(*entity_pair)
        # #
        # # self.check_interface_topology()

        if fragments:
            if query_fragments:  # search for new fragment information
                # inherently gets interface residues for the designable entities
                self.generate_interface_fragments(out_path=design_dir.frags, write_fragments=write_fragments)
                # self.check_interface_topology()  # already done above
                design_dir.info['fragments'] = True
            else:  # No fragment query, add existing fragment information to the pose
                # if fragments_exist:
                if not self.frag_db:
                    self.connect_fragment_database(init=False)  # location='biological_interfaces' inherent in call
                    # Attach an existing FragmentDB to the Pose
                    self.attach_fragment_database(db=self.frag_db)
                    for entity in self.entities:
                        entity.attach_fragment_database(db=self.frag_db)
                    # self.handle_flags(frag_db=self.frag_db)  # attach to all entities

                fragment_source = design_dir.fragment_observations
                if not fragment_source:
                    raise DesignError('%s: Fragments were set for design but there were none found in the Design '
                                      'Directory! Fix your input flags if this is not what you expected or generate '
                                      'them with \'%s %s\''
                                      % (str(design_dir), PUtils.program_command, PUtils.generate_fragments))

                # Must provide des_dir.fragment_observations then specify whether the Entity in question is from the
                # mapped or paired chain (entity1 is mapped, entity2 is paired from Nanohedra). Then, need to renumber
                # fragments to Pose residue numbering when added to fragment queries
                if design_dir.nano:
                    if len(self.entities) > 2:  # Todo compatible with > 2 entities
                        raise DesignError('Not able to solve fragment/residue membership with more than 2 Entities!')
                    entity_ids = tuple(entity.name for entity in self.entities)
                    self.log.debug('Fragment data found in Nanohedra docking. Solving fragment membership for '
                                   'Entity ID\'s: %s by PDB numbering correspondence' % str(entity_ids))
                    self.add_fragment_query(entity1=self.entities[0], entity2=self.entities[1], query=fragment_source,
                                            pdb_numbering=True)
                else:  # assuming the input is in Pose numbering
                    self.log.debug('Fragment data found from prior query. Solving query index by Pose number/Entity '
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
            # Todo check this assumption...
            #  DesignDirectory.path was removed from evol as the input oligomers should be perfectly symmetric so the tx
            #  won't matter for each entity. right?
            # TODO Insert loop identifying comparison of SEQRES and ATOM before SeqProf.calculate_design_profile()
            if entity not in self.active_entities:  # we shouldn't design, add a null profile instead
                entity.add_profile(null=True, out_path=design_dir.sequences)
            else:
                entity.add_profile(evolution=evolution, fragments=fragments, out_path=design_dir.sequences)

        # Update DesignDirectory with design information # Todo include in DesignDirectory initialization by args?
        # This info is pulled out in AnalyzeOutput from Rosetta currently

        if fragments:  # set pose.fragment_profile by combining entity frag profile into single profile
            self.combine_fragment_profile([entity.fragment_profile for entity in self.entities])
            # self.log.debug('Fragment Specific Scoring Matrix: %s' % str(self.fragment_profile))
            self.interface_data_file = pickle_object(self.fragment_profile, frag_db + PUtils.frag_profile,
                                                     out_path=design_dir.data)
            design_dir.info['fragment_database'] = frag_db
            design_dir.info['fragment_profile'] = self.interface_data_file

        if evolution:  # set pose.evolutionary_profile by combining entity evo profile into single profile
            self.combine_pssm([entity.evolutionary_profile for entity in self.entities])
            # self.log.debug('Position Specific Scoring Matrix: %s' % str(self.evolutionary_profile))
            self.pssm_file = self.write_pssm_file(self.evolutionary_profile, PUtils.pssm, out_path=design_dir.data)
            design_dir.info['evolutionary_profile'] = self.pssm_file

        self.combine_profile([entity.profile for entity in self.entities])
        # self.log.debug('Design Specific Scoring Matrix: %s' % str(self.profile))
        self.design_pssm_file = self.write_pssm_file(self.profile, PUtils.dssm, out_path=design_dir.data)
        design_dir.info['design_profile'] = self.design_pssm_file
        # -------------------------------------------------------------------------
        # self.solve_consensus()  # Todo
        # -------------------------------------------------------------------------

    def connect_fragment_database(self, location=None, init=False, **kwargs):  # Todo Clean up
        """Generate a new connection. Initialize the representative library by passing init=True"""
        if not location:  # Todo fix once multiple are available
            location = 'biological_interfaces'
        self.frag_db = FragmentDatabase(location=location, init_db=init)
        #                               source=source, location=location, init_db=init_db)

    def generate_interface_fragments(self, write_fragments=True, out_path=None, new_db=False):
        """Using the attached fragment database, generate interface fragments between the Pose interfaces

        Keyword Args:
            write_fragments=True (bool): Whether or not to write the located fragments
            out_path=None (str): The location to write each fragment file
            new_db=False (bool): Whether a fragment database should be initialized for the interface fragment search
        """
        if not self.frag_db:  # There is no fragment database connected
            # Connect to a new DB, Todo parameterize which one should be used with location=
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
            write_fragment_pairs(self.fragment_pairs, out_path=out_path)
            for match_count, (ghost_frag, surface_frag, match) in enumerate(self.fragment_pairs, 1):
                write_frag_match_info_file(ghost_frag=ghost_frag, matched_frag=surface_frag,
                                           overlap_error=z_value_from_match_score(match),
                                           match_number=match_count, out_path=out_path)

    def return_symmetry_parameters(self):
        """Return the symmetry parameters from a SymmetricModel

        Returns:
            (dict): {symmetry: (str), dimension: (int), uc_dimensions: (list), expand_matrices: (list[list])}
        """
        return {'symmetry': self.__dict__['symmetry'],
                'uc_dimensions': self.__dict__['uc_dimensions'],
                'expand_matrices': self.__dict__['expand_matrices'],
                'dimension': self.__dict__['dimension']}

    # def get_interface_surface_area(self):
    #     # pdb1_interface_sa = entity1.get_surface_area_residues(entity1_residue_numbers)
    #     # pdb2_interface_sa = entity2.get_surface_area_residues(self.interface_residues or entity2_residue_numbers)
    #     # interface_buried_sa = pdb1_interface_sa + pdb2_interface_sa
    #     return


def subdirectory(name):  # TODO PDBdb
    return name


def download_pdb(pdb, location=os.getcwd(), asu=False):
    """Download a pdbs from a file, a supplied list, or a single entry

    Args:
        pdb (str, list): PDB's of interest. If asu=False, code_# is format for biological assembly specific pdb.
            Ex: 1bkh_2 fetches 1BKH biological assembly 2
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
            status = os.system('wget -O %s http://files.rcsb.org/download/%s' % (file_name, clean_pdb))
            if status != 0:
                failures.append(pdb)

    if failures:
        logger.error('PDB download ran into the following failures:\n%s' % ', '.join(failures))

    return file_name  # Todo if list then will only return the last file


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


def split_interface_pairs(interface_pairs):
    residues1, residues2 = zip(*interface_pairs)
    return sorted(set(residues1), key=int), sorted(set(residues2), key=int)


def find_interface_residues(pdb1, pdb2, distance=8):
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


def get_fragments(pdb, chain_res_info, fragment_length=5):  # Todo depreciate
    interface_frags = []
    ca_count = 0
    for residue_number in chain_res_info:
        frag_residue_numbers = [residue_number + i for i in range(-2, 3)]
        frag_atoms, ca_present = [], []
        for residue in pdb.residue(frag_residue_numbers):
            frag_atoms.extend(residue.atoms())
            if residue.ca:
                ca_count += 1

        if ca_count == 5:
            interface_frags.append(PDB.from_atoms(frag_atoms))

    return interface_frags


def find_fragment_overlap_at_interface(entity1_coords, interface_frags1, interface_frags2, fragdb=None,
                                       euler_lookup=None, max_z_value=2):
    #           entity1, entity2, entity1_interface_residue_numbers, entity2_interface_residue_numbers, max_z_value=2):
    """From a Structure Entity, score the interface between them according to Nanohedra's fragment matching"""
    if not fragdb:
        fragdb = FragmentDB()
        fragdb.get_monofrag_cluster_rep_dict()
        fragdb.get_intfrag_cluster_rep_dict()
        fragdb.get_intfrag_cluster_info_dict()
    if not euler_lookup:
        euler_lookup = EulerLookup()

    kdtree_oligomer1_backbone = BallTree(entity1_coords)
    interface_ghost_frags1 = []
    for frag1 in interface_frags1:
        ghostfrags = frag1.get_ghost_fragments(fragdb.paired_frags, kdtree_oligomer1_backbone, fragdb.info)
        if ghostfrags:
            interface_ghost_frags1.extend(ghostfrags)

    # Get fragment guide coordinates
    interface_ghostfrag_guide_coords = np.array([ghost_frag.guide_coords for ghost_frag in interface_ghost_frags1])
    interface_surf_frag_guide_coords = np.array([frag2.get_guide_coords() for frag2 in interface_frags2])

    # Check for matching Euler angles
    # TODO create a stand alone function
    overlapping_ghost_indices, overlapping_surf_indices = \
        euler_lookup.check_lookup_table(interface_ghostfrag_guide_coords, interface_surf_frag_guide_coords)
    # filter array by matching type for surface (i) and ghost (j) frags
    surface_type_i_array = np.array([interface_frags2[idx].i_type for idx in overlapping_surf_indices.tolist()])
    ghost_type_j_array = np.array([interface_ghost_frags1[idx].j_type for idx in overlapping_ghost_indices.tolist()])
    ij_type_match = np.where(surface_type_i_array == ghost_type_j_array, True, False)

    passing_ghost_indices = overlapping_ghost_indices[ij_type_match]
    passing_ghost_coords = interface_ghostfrag_guide_coords[passing_ghost_indices]

    passing_surf_indices = overlapping_surf_indices[ij_type_match]
    passing_surf_coords = interface_surf_frag_guide_coords[passing_surf_indices]
    # precalculate the reference_rmsds for each ghost fragment
    reference_rmsds = np.array([interface_ghost_frags1[ghost_idx].rmsd for ghost_idx in passing_ghost_indices.tolist()])
    reference_rmsds = np.where(reference_rmsds == 0, 0.01, reference_rmsds)

    all_fragment_overlap = calculate_overlap(passing_ghost_coords, passing_surf_coords, reference_rmsds,
                                             max_z_value=max_z_value)
    passing_overlap_indices = np.flatnonzero(all_fragment_overlap)

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
        entity1_surffrag_ch, entity1_surffrag_resnum = interface_ghost_frag.get_aligned_chain_and_residue()
        entity2_surffrag_ch, entity2_surffrag_resnum = interface_mono_frag.get_central_res_tup()
        fragment_matches.append({'mapped': entity1_surffrag_resnum, 'match': match_score,
                                 'paired': entity2_surffrag_resnum, 'cluster': '%d_%d_%d'
                                                                               % interface_ghost_frag.get_ijk()})
    logger.debug('Fragments for Entity1 found at residues: %s' % [fragment['mapped'] for fragment in fragment_matches])
    logger.debug('Fragments for Entity2 found at residues: %s' % [fragment['paired'] for fragment in fragment_matches])

    return fragment_matches


def write_fragment_pairs(ghostfrag_surffrag_pairs, out_path=os.getcwd()):
    for idx, (interface_ghost_frag, interface_mono_frag, match_score) in enumerate(ghostfrag_surffrag_pairs):
        interface_ghost_frag.structure.write(out_path=os.path.join(out_path, '%d_%d_%d_fragment_overlap_match_%d.pdb'
                                                                   % (*interface_ghost_frag.get_ijk(), idx)))


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

    interface_frags1 = get_fragments(entity1, entity1_interface_residue_numbers)
    interface_frags2 = get_fragments(entity2, entity2_interface_residue_numbers)
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
    # Get the interface residues
    pdb1_cb_coords, pdb1_cb_indices = pdb1.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)
    pdb2_cb_coords, pdb2_cb_indices = pdb2.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)

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
