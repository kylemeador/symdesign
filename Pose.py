import copy
import os
import pickle
from glob import glob
from itertools import chain, combinations, combinations_with_replacement

# import ipdb

import numpy as np
from sklearn.neighbors import BallTree

import PathUtils as PUtils
from PDB import PDB
from SequenceProfile import SequenceProfile, calculate_match_metrics
from Structure import Coords, Structure
# Globals
from SymDesignUtils import to_iterable, pickle_object, DesignError, calculate_overlap,  \
    filter_euler_lookup_by_zvalue, z_value_from_match_score, start_log, point_group_sdf_map, possible_symmetries # logger,
from FragDock import write_frag_match_info_file
from classes.EulerLookup import EulerLookup
from classes.Fragment import MonoFragment, FragmentDB
from utils.ExpandAssemblyUtils import sg_cryst1_fmt_dict, pg_cryst1_fmt_dict, zvalue_dict
from utils.SymmUtils import valid_subunit_number


logger = start_log(name=__name__, level=2)  # was from SDUtils logger, but moved here per standard suggestion

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
    def __init__(self, pdb=None, models=None, log=None, **kwargs):
        super().__init__()  # **kwargs
        # if isinstance(pdb, list):
        # self.models = pdb  # list or dict of PDB objects
        # self.pdb = self.models[0]
        # elif isinstance(pdb, PDB):
        self.pdb = pdb
        if models and isinstance(models, list):
            self.models = models
        else:
            self.models = []

        self.number_of_models = len(self.models)
        if log:
            self.log = log
        else:
            print('Model starting log')
            self.log = start_log()

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
    def coords(self):
        """Return a view of the Coords from the Model"""
        return self._coords.coords  # [self.atom_indices]
        # return self._coords.get_indices(self.atom_indices)

    @coords.setter
    def coords(self, coords):
        # assert len(self.atoms) == coords.shape[0], '%s: ERROR number of Atoms (%d) != number of Coords (%d)!' \
        #                                                 % (self.name, len(self.atoms), self.coords.shape[0])
        if isinstance(coords, Coords):
            self._coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

    @property
    def model_coords(self):
        """Return a view of the Coords from the Model"""
        return self._model_coords.coords  # [self.atom_indices]
        # return self._coords.get_indices(self.atom_indices)

    @model_coords.setter
    def model_coords(self, coords):
        # assert len(self.atoms) == coords.shape[0], '%s: ERROR number of Atoms (%d) != number of Coords (%d)!' \
        #                                                 % (self.name, len(self.atoms), self.coords.shape[0])
        if isinstance(coords, Coords):
            self._model_coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

    def set_symmetry(self, symmetry=None, cryst1=None, uc_dimensions=None, generate_assembly=True,
                     generate_symmetry_mates=False, **kwargs):
        """Set the model symmetry using the CRYST1 record, or the unit cell dimensions and the Hermann–Mauguin symmetry
        notation (in CRYST1 format, ex P 4 3 2) for the Model assembly. If the assembly is a point group,
        only the symmetry is required"""
        if cryst1:
            uc_dimensions, symmetry = PDB.parse_cryst_record(cryst1_string=cryst1)

        if uc_dimensions and symmetry:
            self.uc_dimensions = uc_dimensions
            # self.symmetry = ''.join(symmetry.split())  # ensure the symmetry is NO SPACES Hermann–Mauguin notation
            if symmetry in pg_cryst1_fmt_dict.values():  # not available yet for non-Nanohedra PG's
                self.dimension = 2
            elif symmetry in sg_cryst1_fmt_dict.values():  # not available yet for non-Nanohedra SG's
                self.dimension = 3
            else:
                raise DesignError('Symmetry %s is not available yet! You likely set the symmetry from a PDB file. '
                                  'Get the symmetry operations from international'
                                  ' tables and add to the pickled operators if this displeases you!' % symmetry)
            self.expand_matrices = self.get_sg_sym_op(''.join(symmetry.split()))
            # self.expand_matrices = np.array(self.get_sg_sym_op(self.symmetry))  # Todo numpy expand_matrices
        elif not symmetry:
            return None  # no symmetry was provided
        elif symmetry in possible_symmetries:  # ['T', 'O', 'I']:
            # symmetry = point_group_sdf_map[symmetry][0]
            symmetry = possible_symmetries[symmetry]
            self.dimension = 0
            self.expand_matrices = self.get_ptgrp_sym_op(symmetry)  # Todo numpy expand_matrices
            # self.expand_matrices = np.array(self.get_ptgrp_sym_op(self.symmetry))
        else:
            raise DesignError('Symmetry %s is not available yet! Get the cannonical symm operators from %s and add to'
                              ' the pickled operators if this displeases you!' % (symmetry, PUtils.orient_dir))

        self.symmetry = symmetry
        if self.asu and generate_assembly:
            self.generate_symmetric_assembly()
            if generate_symmetry_mates:
                self.get_assembly_symmetry_mates()

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
        if return_side_chains:  # get different function calls depending on the return type # todo
            get_pdb_coords = getattr(PDB, 'get_coords')
            self.coords_type = 'all'
        else:
            get_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')
            self.coords_type = 'bb_cb'

        # self.coords = Coords(get_pdb_coords(self.asu))
        self.log.debug('Coords length at symmetry coordinate expansion: %d' % len(self.coords))
        # self.model_coords = np.empty((len(self.coords) * self.number_of_models, 3), dtype=float)
        model_coords = np.empty((len(self.coords) * self.number_of_models, 3), dtype=float)
        self.log.debug('model coords length at coord expansion: %d' % len(model_coords))
        # self.model_coords[:len(self.coords)] = self.coords
        for idx, rot in enumerate(self.expand_matrices):  # Todo numpy incomming expand_matrices
            r_mat = np.transpose(np.array(rot))
            r_asu_coords = np.matmul(self.coords, r_mat)
            model_coords[idx * len(self.coords): (idx + 1) * len(self.coords)] = r_asu_coords
        self.model_coords = Coords(model_coords)

    def get_unit_cell_coords(self, return_side_chains=True):
        """Generates unit cell coordinates for a symmetry group. Modifies model_coords to include all in a unit cell"""
        # self.models = [self.asu]
        self.number_of_models = zvalue_dict[self.symmetry]
        if return_side_chains:  # get different function calls depending on the return type  # todo
            get_pdb_coords = getattr(PDB, 'get_coords')
            self.coords_type = 'all'
        else:
            get_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')
            self.coords_type = 'bb_cb'

        # self.coords = Coords(get_pdb_coords(self.asu))
        # asu_cart_coords = get_pdb_coords(self.pdb)
        # asu_cart_coords = self.pdb.get_coords()  # returns a numpy array
        # asu_frac_coords = self.cart_to_frac(np.array(asu_cart_coords))
        asu_frac_coords = self.cart_to_frac(self.coords)
        # self.model_coords = np.empty((len(self.coords) * self.number_of_models, 3), dtype=float)
        model_coords = np.empty((len(self.coords) * self.number_of_models, 3), dtype=float)
        for idx, (rot, tx) in enumerate(self.expand_matrices):
            t_vec = np.array(tx)  # Todo numpy incomming expand_matrices
            # t_vec = t
            r_mat = np.transpose(np.array(rot))
            # r_mat = np.transpose(r)

            r_asu_frac_coords = np.matmul(asu_frac_coords, r_mat)
            rt_asu_frac_coords = r_asu_frac_coords + t_vec
            model_coords[idx * len(self.coords): (idx + 1) * len(self.coords)] = self.frac_to_cart(rt_asu_frac_coords)

            # self.model_coords.extend(self.frac_to_cart(rt_asu_frac_coords).tolist())
        self.model_coords = Coords(model_coords)

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
        # uc_coord_length = self.model_coords.shape[0]
        uc_coord_length = len(self.model_coords)
        # self.model_coords = np.empty((uc_coord_length * uc_number, 3), dtype=float)
        model_coords = np.empty((uc_coord_length * uc_number, 3), dtype=float)
        idx = 0
        for x_shift in [-1, 0, 1]:
            for y_shift in [-1, 0, 1]:
                for z_shift in z_shifts:
                    # add central uc_coords to the model coords after applying the correct tx of frac coords & convert
                    model_coords[idx * uc_coord_length: (idx + 1) * uc_coord_length] = \
                        self.frac_to_cart(central_uc_frac_coords + [x_shift, y_shift, z_shift])
                    idx += 1
        self.model_coords = Coords(model_coords)
        self.number_of_models = zvalue_dict[self.symmetry] * uc_number

    def return_assembly_symmetry_mates(self):
        count = 0
        while len(self.models) != self.number_of_models:  # Todo clarify we haven't generated the mates yet
            self.get_assembly_symmetry_mates()
            if count == 1:
                raise DesignError('%s: The assembly couldn\'t be returned'
                                  % self.return_assembly_symmetry_mates.__name__)
            count += 1

        return self.models

    def get_assembly_symmetry_mates(self, return_side_chains=True):  # For getting PDB copies
        """Return all symmetry mates as a list of PDB objects. Chain names will match the ASU"""
        if not self.symmetry:
            raise DesignError('%s: No symmetry set for %s! Cannot get symmetry mates'
                              % (self.get_assembly_symmetry_mates.__name__, self.asu.name))
        if return_side_chains:  # get different function calls depending on the return type
            extract_pdb_atoms = getattr(PDB, 'get_atoms')
            # extract_pdb_coords = getattr(PDB, 'extract_coords')
        else:
            extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
            # extract_pdb_coords = getattr(PDB, 'extract_backbone_and_cb_coords')

        # self.models.append(copy.copy(self.asu))
        # prior_idx = self.asu.number_of_atoms  # TODO modify by extract_pdb_atoms!
        for model_idx in range(self.number_of_models):  # range(1,
            symmetry_mate_pdb = copy.deepcopy(self.asu)
            # self.log.info(len(self.model_coords[(model_idx * self.asu.number_of_atoms):
            #                             ((model_idx + 1) * self.asu.number_of_atoms)]))
            # self.log.info('Asu ATOM number: %d' % self.asu.number_of_atoms)
            symmetry_mate_pdb.coords = Coords(self.model_coords[(model_idx * self.asu.number_of_atoms):
                                                                ((model_idx + 1) * self.asu.number_of_atoms)])
            # symmetry_mate_pdb.set_atom_coordinates(self.model_coords[(model_idx * self.asu.number_of_atoms):
            #                                                          (model_idx * self.asu.number_of_atoms)
            #                                                          + self.asu.number_of_atoms])
            # self.log.info('Coords copy of length %d: %s' % (len(symmetry_mate_pdb.coords), symmetry_mate_pdb.coords))
            # self.log.info('Atom 1 coordinates: %s' % symmetry_mate_pdb.get_atoms()[0].x)
            # self.log.info('Atom 1: %s' % str(symmetry_mate_pdb.get_atoms()[0]))
            self.models.append(symmetry_mate_pdb)
        # for model_idx in range(len(self.models)):
            # self.log.info('Atom 1: %s' % str(self.models[model_idx].get_atoms()[0]))

    def find_asu_equivalent_symmetry_model(self, residue_query_number=1):
        """Find the asu equivalent model in the SymmetricModel. Zero-indexed

        Returns:
            (int): The index of the number of models where the ASU can be found
        """
        if not self.symmetry:
            return None

        template_atom_coords = self.asu.residue(residue_query_number).ca.coords
        template_atom_index = self.asu.residue(residue_query_number).ca.index
        for model_number in range(self.number_of_models):
            if (template_atom_coords ==
                    self.model_coords[(model_number * len(self.coords)) + template_atom_index]).all():
                return model_number

        self.log.error('%s is FAILING' % self.find_asu_equivalent_symmetry_model.__name__)

    def find_asu_equivalent_symmetry_mate_indices(self):
        """Find the asu equivalent model in the SymmetricModel. Zero-indexed

        model_coords must be from all atoms which by default is True
        Returns:
            (list): The index of the number of models where the ASU can be found
        """
        model_number = self.find_asu_equivalent_symmetry_model()
        start_idx = self.pdb.number_of_atoms * model_number
        end_idx = self.pdb.number_of_atoms * (model_number + 1)
        # Todo test logic, we offset the index by 1 from the end_idx values (0 to 100 -> 0:99 or 100 to 200 -> 100:199)
        return [idx for idx in range(start_idx, end_idx)]

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
            return self.return_surrounding_unit_cell_symmetry_mates(pdb, **kwargs)  # return_side_chains=return_side_chains FOR Coord expansion
        else:
            return self.return_unit_cell_symmetry_mates(pdb, **kwargs)  # # return_side_chains=return_side_chains

    # @staticmethod
    def return_point_group_symmetry_mates(self, pdb, return_side_chains=True):  # For returning PDB copies
        """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
        # if return_side_chains:  # get different function calls depending on the return type
        #     extract_pdb_atoms = getattr(PDB, 'get_atoms')  # Not using. The copy() versus PDB() changes residue objs
        #     extract_pdb_coords = getattr(PDB, 'extract_coords')
        # else:
        #     extract_pdb_atoms = getattr(PDB, 'get_backbone_and_cb_atoms')
        #     extract_pdb_coords = getattr(PDB, 'extract_backbone_and_cb_coords')
        # ipdb.set_trace()
        return [pdb.return_transformed_copy(rotation=rot) for rot in self.expand_matrices]  # Todo change below as well
        # asu_symm_mates = []
        # pdb_coords = np.array(extract_pdb_coords(pdb))
        # for rot in self.expand_matrices:
        #     r_mat = np.transpose(np.array(rot))
        #     r_asu_coords = np.matmul(pdb_coords, r_mat)
        #     symmetry_mate_pdb = pdb.return_transformed_copy(rotation=rot)
        #     symmetry_mate_pdb = copy.deepcopy(pdb)
        #     symmetry_mate_pdb.coords = Coords(r_asu_coords)
        #     # symmetry_mate_pdb.set_atom_coordinates(r_asu_coords)
        #     asu_symm_mates.append(symmetry_mate_pdb)
        #
        # return asu_symm_mates

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

    def symmetric_assembly_is_clash(self, clash_distance=2.2):  # Todo design_selector
        """Returns True if the SymmetricModel presents any clashes. Checks only backbone and CB atoms

        Keyword Args:
            clash_distance=2.2 (float): The cutoff distance for the coordinate overlap

        Returns:
            (bool)
        """
        if not self.number_of_models:
            raise DesignError('Cannot check if the assembly is clashing without first calling %s'
                              % self.generate_symmetric_assembly.__name__)

        model_asu_indices = self.find_asu_equivalent_symmetry_mate_indices()
        # print('ModelASU Indices: %s' % model_asu_indices)
        # print('Equivalent ModelASU: %d' % self.find_asu_equivalent_symmetry_model())
        if self.coords_type != 'bb_cb':
            asu_indices = self.asu.get_backbone_and_cb_indices()
            # model_indices_filter = np.array(asu_indices * self.number_of_models)
            # print('BB/CB ASU Indices: %s' % asu_indices)

            # Need to only select the coords that are not BB or CB from the model coords.
            # We have all the BB/CB indices from ASU now need to multiply this by every integer in self.number_of_models
            # to get every BB/CB coord.
            # Finally we take out those indices that are inclusive of the model_asu_indices like below
            number_asu_atoms = self.asu.number_of_atoms
            model_indices_filter = np.array([idx + (model_number * number_asu_atoms)
                                             for model_number in range(self.number_of_models)
                                             for idx in asu_indices])
            # print('Model indices factor: %s' % model_indices_factor)
            # print('Model ASU indices[0] & [-1]: %d & %d' % (model_asu_indices[0], model_asu_indices[-1]))
        else:  # we will frag every coord in the model
            model_indices_filter = np.array([idx for idx in range(len(self.model_coords))])
            asu_indices = None

        # print('Model indices filter length: %d' % len(model_indices_filter))
        # print('Model indices filter: %s' % model_indices_filter)
        asu_coord_kdtree = BallTree(self.coords[asu_indices])
        without_asu_mask = np.logical_or(model_indices_filter < model_asu_indices[0],
                                         model_indices_filter > model_asu_indices[-1])
        # print('Model without asu mask length: %d' % len(without_asu_mask))
        # print('Asu filter length: %d' % number_asu_indices)
        # take the boolean mask and filter the model indices mask to leave only symmetry mate bb/cb indices, NOT asu
        model_indices_without_asu = model_indices_filter[without_asu_mask]

        # print('Model Coord length: %d' % len(self.model_coords))
        # print('Model minus ASU indices: %s' % model_indices_without_asu)
        # print('Model Coord filtered length: %d' % len(self.model_coords[model_indices_without_asu]))
        # print('Model Coord filtered: %s' % self.model_coords[model_indices_without_asu])
        selected_assembly_coords = len(self.model_coords[model_indices_without_asu]) + len(self.coords[asu_indices])
        all_assembly_coords_length = len(asu_indices) * self.number_of_models
        assert selected_assembly_coords == all_assembly_coords_length, '%s: Ran into an issue with indexing.' \
                                                                       % self.symmetric_assembly_is_clash()

        clash_count = asu_coord_kdtree.two_point_correlation(self.model_coords[model_indices_without_asu],
                                                             [clash_distance])
        if clash_count[0] > 0:
            self.log.warning('%s: Found %d clashing sites! Pose is not a viable symmetric assembly'
                             % (self.pdb.name, clash_count[0]))
            return True  # clash
        else:
            return False  # no clash

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

    def write(self, out_path=os.getcwd()):  #, cryst1=None):  # Todo write model, write symmetry.    name, location
        # out_path = os.path.join(location, '%s.pdb' % name)
        with open(out_path, 'w') as f:
            # if cryst1 and isinstance(cryst1, str) and cryst1.startswith('CRYST1'):
            #     f.write('%s\n' % cryst1)
            for i, model in enumerate(self.models, 1):
                f.write('{:9s}{:>4d}\n'.format('MODEL', i))
                for chain in model.chains:
                    chain_atoms = chain.get_atoms()
                    f.write(''.join(str(atom) for atom in chain_atoms))
                    f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', chain_atoms[-1].number + 1,
                                                                          chain_atoms[-1].residue_type, chain.name,
                                                                          chain_atoms[-1].residue_number))
                # f.write(''.join(str(atom) for atom in model.atoms))
                # f.write('\n'.join(str(atom) for atom in model.atoms))
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


class Pose(SymmetricModel, SequenceProfile):  # Model, PDB
    """A Pose is made of multiple PDB objects all sharing a common feature such as symmetric copies or modifications to
    the PDB sequence
    """
    def __init__(self, asu=None, pdb=None, pdb_file=None, asu_file=None, **kwargs):  # symmetry=None, log=None,
        super().__init__(**kwargs)  # log=None,
        # super().__init__(log=log)
        # if log:  # from Super SymmetricModel
        #     self.log = log
        # else:
        #     self.log = start_log()

        if asu and isinstance(asu, Structure):
            self.asu = asu
            self.pdb = self.asu
        elif asu_file:
            self.asu = PDB.from_file(asu_file, log=self.log)
            self.pdb = self.asu
        elif pdb and isinstance(asu, Structure):
            self.pdb = pdb
            # self.set_pdb(pdb)
        elif pdb_file:
            self.pdb = PDB.from_file(pdb_file, log=self.log)
            # Depending on the extent of PDB class initialization, I could copy the PDB info into self.pdb
            # this would be:
            # coords, atoms, residues, chains, entities, design, (host of others read from file)
            # self.set_pdb(pdb)
        # else:
        #     nothing = True
        if self.pdb:
            # add structure to the SequenceProfile
            # self.pdb.is_clash()  # Todo
            self.set_structure(self.pdb)
            # set up coordinate information for SymmetricModel
            self.coords = Coords(self.pdb.get_coords())

        self.design_selector_entities = set()
        self.design_selector_indices = set()
        self.interface_residues = {}
        self.fragment_observations = []

        symmetry_kwargs = self.pdb.symmetry
        symmetry_kwargs.update(kwargs)
        # self.log.debug('Pose symmetry_kwargs: %s' % symmetry_kwargs)
        self.set_symmetry(**symmetry_kwargs)  # this will only generate an assembly if the ASU was passed
        # self.initialize_symmetry(symmetry=symmetry)

        self.handle_flags(**kwargs)
        self.pdbs = []  # the member pdbs which make up the pose
        self.pdbs_d = {}
        self.pose_pdb_accession_map = {}

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

    # @entities.setter
    # def entities(self, entities):
    #     self.pdb.entities = entities

    @property
    def chains(self):
        return self.pdb.chains

    @property
    def active_chains(self):
        return [_chain for entity in self.active_entities for _chain in entity.chains]

    # @chains.setter
    # def chains(self, chains):
    #     self.pdb.chains = chains

    def entity(self, entity):
        return self.pdb.entity(entity)

    def chain(self, _chain):  # no not mess with chain.from_iterable namespace
        return self.pdb.entity_from_chain(_chain)

    @property
    def number_of_atoms(self):
        return len(self.pdb.get_atoms())
    #     try:
    #         return self._number_of_atoms
    #     except AttributeError:
    #         self.set_length()
    #         return self._number_of_atoms
    #
    # @number_of_atoms.setter
    # def number_of_atoms(self, length):
    #     self._number_of_atoms = length

    @property
    def number_of_residues(self):
        return len(self.pdb.get_residues())
    #     try:
    #         return self._number_of_residues
    #     except AttributeError:
    #         self.set_length()
    #         return self._number_of_residues
    #
    # @number_of_residues.setter
    # def number_of_residues(self, length):
    #     self._number_of_residues = length

    @property
    def center_of_mass(self):
        try:
            return self._center_of_mass
        except AttributeError:
            self.find_center_of_mass()
            return self._center_of_mass

    # def set_length(self):
    #     self.number_of_atoms = len(self.pdb.get_atoms())
    #     self.number_of_residues = len(self.pdb.get_residues())

    def find_center_of_mass(self):
        """Retrieve the center of mass for the specified Structure"""
        if self.symmetry:
            divisor = 1 / len(self.model_coords)
            self._center_of_mass = np.matmul(np.full(self.number_of_atoms, divisor), self.model_coords)
        else:
            divisor = 1 / len(self.coords)  # must use coords as can have reduced view of the full coords, i.e. BB & CB
            self._center_of_mass = np.matmul(np.full(self.number_of_atoms, divisor), self.coords)

    def handle_flags(self, design_selector=None, frag_db=None, **kwargs):
        if design_selector:
            self.create_design_selector(**design_selector)
        if frag_db:
            # Attach an existing FragmentDB to the Pose
            self.attach_fragment_database(db=frag_db)
            for entity in self.entities:
                entity.attach_fragment_database(db=frag_db)

    def create_design_selector(self, selection=None, mask=None):
        # def mask(self, pdbs=None, entities=None, chains=None, residues=None, atoms=None):
        def grab_indices(pdbs=None, entities=None, chains=None, residues=None, atoms=None):
            entity_union = set()
            atom_intersect = set(self.pdb.atom_indices)
            if pdbs:
                # atom_selection = set(self.pdb.get_residue_atom_indices(numbers=residues))
                raise DesignError('Can\'t select residues by PDB yet!')
            if entities:
                atom_intersect = atom_intersect.intersection(chain.from_iterable([self.entity(entity).atom_indices
                                                                                  for entity in entities]))
                entity_union = entity_union.union([self.entity(entity) for entity in entities])
            if chains:
                # vv This is for the intersectional model
                atom_intersect = atom_intersect.intersection(chain.from_iterable([self.chain(chain_id).atom_indices
                                                                                  for chain_id in chains]))
                # atom_selection.union(chain.from_iterable(self.chain(chain_id).get_residue_atom_indices(numbers=residues)
                #                                     for chain_id in chains))
                # ^^ This is for the additive model
                entity_union = entity_union.union([self.chain(chain_id) for chain_id in chains])
            if residues:
                atom_intersect = atom_intersect.intersection(self.pdb.get_residue_atom_indices(numbers=residues))
            if atoms:
                atom_intersect = atom_intersect.intersection(self.pdb.get_atom_indices(numbers=atoms))

            return entity_union, atom_intersect

        entity_selection, atom_selection = grab_indices(**selection)
        entity_mask, atom_mask = grab_indices(**mask)
        entity_selection = entity_selection.difference(entity_mask)
        atom_selection = atom_selection.difference(atom_mask)

        # self.design_selector_entities = set(self.entities)
        # self.design_selector_entities = self.design_selector_entities.intersection(entity_union)
        self.design_selector_entities = self.design_selector_entities.union(entity_selection)
        self.design_selector_indices = self.design_selector_indices.union(atom_selection)

    # def add_pdb(self, pdb):  # Todo
    #     """Add a PDB to the PosePDB as well as the member PDB list"""
    #     self.pdbs.append(pdb)
    #     self.pdbs_d[pdb.name] = pdb
    #     # self.pdbs_d[id(pdb)] = pdb
    #     self.add_entities_to_pose(self, pdb)
    #     # Todo turn multiple PDB's into one structure representative
    #     # self.pdb.add_atoms(pdb.get_atoms())
    #
    # def add_entities_to_pose(self, pdb):  # Unused Todo
    #     """Add each unique entity in a pdb to the pose, updating all metadata"""
    #     # self.pose_pdb_accession_map[pdb.name] = pdb.entity_accession_map
    #     self.pose_pdb_accession_map[pdb.name] = pdb.entity_d
    #     # for entity in pdb.accession_entity_map:
    #     for idx, entity in enumerate(pdb.entities):
    #         self.add_entity(entity, name='%s_%d' % (pdb.name, idx))
    #
    # def add_entity(self, entity, name=None):  # Unused
    #     # Todo Fix this garbage... Entity()
    #     self.entities = None
    #     entity_chain = pdb.entities[entity]['representative']
    #     self.asu.add_atoms(entity.get_atoms())

    def initialize_symmetry(self, symmetry=None):  # Unused
        if symmetry:
            self.set_symmetry(symmetry=symmetry)
        elif self.pdb.space_group and self.pdb.uc_dimensions:  # Todo from ASU has the old symmetry indicators attached!
            self.set_symmetry(symmetry=self.pdb.space_group, uc_dimensions=self.pdb.uc_dimensions)
        elif self.pdb.cryst_record:
            self.set_symmetry(cryst1=self.pdb.cryst_record)
        else:
            self.log.info('No symmetry present in the Pose PDB')

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

    def find_interface_pairs(self, entity1=None, entity2=None, distance=8, include_glycine=True):
        """Get pairs of residue numbers that have CB atoms within a certain distance (in contact) between two named
        Entities.
        Caution!: Pose must have Coords representing all atoms as residue pairs are found using CB indices from all atoms

        Symmetry aware. If symmetry is used, by default all atomic coordinates for entity2 are symmeterized.
        design_selector aware

        Keyword Args:
            entity1=None (Entity): First entity to measure interface between
            # entity1=None (str): First entity name to measure interface between
            entity2=None (Entity): Second entity to measure interface between
            # entity2=None (str): Second entity name to measure interface between
            distance=8 (int): The distance to query the interface in Angstroms
            include_glycine=True (bool): Whether glycine CA should be included in the tree
        Returns:
            list[tuple]: A list of interface residue numbers across the interface
        """
        # entity2_query = construct_cb_atom_tree(entity1, entity2, distance=distance)
        pdb_atoms = self.pdb.get_atoms()
        number_of_atoms = self.pdb.number_of_atoms
        self.log.debug('Number of atoms in PDB: %s' % number_of_atoms)

        # Get CB Atom Coordinates including CA coordinates for Gly residues
        # entity1_atoms = entity1.get_atoms()  # if passing by Structure
        entity1_indices = entity1.get_cb_indices(InclGlyCA=include_glycine)
        # entity1_atoms = self.pdb.entity(entity1).get_atoms()  # if passing by name
        # entity1_indices = np.array(self.pdb.entity(entity1).get_cb_indices(InclGlyCA=include_glycine))

        # entity2_atoms = entity2.get_atoms()  # if passing by Structure
        entity2_indices = entity2.get_cb_indices(InclGlyCA=include_glycine)
        # entity2_atoms = self.pdb.entity(entity2).get_atoms()  # if passing by name
        # entity2_indices = self.pdb.entity(entity2).get_cb_indices(InclGlyCA=include_glycine)

        if self.design_selector_indices:  # subtract the masked atom indices from the entity indices
            before = len(entity1_indices) + len(entity2_indices)
            entity1_indices = list(set(entity1_indices).intersection(self.design_selector_indices))
            entity2_indices = list(set(entity2_indices).intersection(self.design_selector_indices))
            self.log.debug('Applied design selection to interface identification. Number of indices before '
                           'selection = %d. Number after = %d' % (before, len(entity1_indices) + len(entity2_indices)))

        if not entity1_indices or not entity2_indices:
            return None

        if self.symmetry:
            # get all symmetric indices if the pose is symmetric
            entity2_indices = [idx + (number_of_atoms * model_number) for model_number in range(self.number_of_models)
                               for idx in entity2_indices]
            pdb_atoms = [atom for model_number in range(self.number_of_models) for atom in pdb_atoms]
            self.log.debug('Number of atoms in expanded assembly PDB: %s' % len(pdb_atoms))
            # entity2_atoms = [atom for model_number in range(self.number_of_models) for atom in entity2_atoms]
            if entity2 == entity1:
                # the queried entity is the same, however we don't want interactions with the same symmetry mate
                asu_indices = self.find_asu_equivalent_symmetry_mate_indices()
                entity2_indices = [idx for idx in entity2_indices if asu_indices[0] > idx or idx > asu_indices[-1]]
                # self.log.info('Number of Entity2 indices: %s' % len(entity2_indices))
            entity2_coords = self.model_coords[np.array(entity2_indices)]  # only get the coordinate indices we want
        else:
            entity2_coords = self.coords[np.array(entity2_indices)]  # only get the coordinate indices we want

        # Construct CB tree for entity1 and query entity2 CBs for a distance less than a threshold
        entity1_indices = np.array(entity1_indices)
        entity1_coords = self.coords[entity1_indices]  # only get the coordinate indices we want
        entity1_tree = BallTree(entity1_coords)
        entity2_query = entity1_tree.query_radius(entity2_coords, distance)

        # Return residue numbers of identified coordinates
        self.log.info('Querying %d Entity %s CB residues versus, %d Entity %s CB residues'
                      % (len(entity1_indices), entity1.name, len(entity2_indices), entity2.name))
        # self.log.debug('Entity2 Query size: %d' % entity2_query.size)

        return [(pdb_atoms[entity1_indices[entity1_idx]].residue_number,  # Todo return residue object from atom?
                 pdb_atoms[entity2_indices[entity2_idx]].residue_number)
                for entity2_idx in range(entity2_query.size) for entity1_idx in entity2_query[entity2_idx]]

    @staticmethod
    def split_interface_pairs(interface_pairs):
        if interface_pairs:
            residues1, residues2 = zip(*interface_pairs)
            return sorted(set(residues1), key=int), sorted(set(residues2), key=int)
        else:
            return [], []

    def find_interface_residues(self, entity1=None, entity2=None, **kwargs):  # entity1=None, entity2=None, distance=8, include_glycine=True):
        """Get unique residues from each pdb across an interface provide two Entity names

        Keyword Args:
            entity1=None (str): First entity name to measure interface between
            entity2=None (str): Second entity name to measure interface between
            distance=8 (int): The distance to query the interface in Angstroms
            include_glycine=True (bool): Whether glycine CA should be included in the tree
        Returns:
            tuple[set]: A tuple of interface residue sets across an interface
        """
        entity1_residue_numbers, entity2_residue_numbers = \
            self.split_interface_pairs(self.find_interface_pairs(entity1=entity1, entity2=entity2, **kwargs))
        if not entity1_residue_numbers or not entity2_residue_numbers:
            self.log.info('Interface %s | %s, no interface found' % (entity1.name, entity2.name))
            self.fragment_queries[(entity1, entity2)] = []
            if entity1 not in self.interface_residues:
                self.interface_residues[entity1] = []
            if entity2 not in self.interface_residues:
                self.interface_residues[entity2] = []
            return None
        else:
            self.log.info('At interface Entity %s | Entity %s\t%s has interface residue numbers: %s'
                          % (entity1.name, entity2.name, entity1.name, entity1_residue_numbers))
            self.log.info('At interface Entity %s | Entity %s\t%s has interface residue numbers: %s'
                          % (entity1.name, entity2.name, entity2.name, entity2_residue_numbers))

        # entity1_structure = self.pdb.entity(entity1)
        if entity1 not in self.interface_residues:
            self.interface_residues[entity1] = entity1.get_residues(numbers=entity1_residue_numbers)
            # self.interface_residues[entity1_structure] = entity1_residue_numbers
        else:
            self.interface_residues[entity1].extend(entity1.get_residues(numbers=entity1_residue_numbers))
            # self.interface_residues[entity1_structure].extend(entity1_residue_numbers)

        # entity2_structure = self.pdb.entity(entity2)
        if entity2 not in self.interface_residues:
            self.interface_residues[entity2] = entity2.get_residues(numbers=entity2_residue_numbers)
        else:
            self.interface_residues[entity2].extend(entity2.get_residues(numbers=entity2_residue_numbers))

        self.log.debug('interface_residues: %s' % self.interface_residues)
        # return entity1_residue_numbers, entity2_residue_numbers

    def query_interface_for_fragments(self, entity1=None, entity2=None):
        # Todo identity frag type by residue, not by calculation
        surface_frags1 = entity1.get_fragments([residue.number for residue in self.interface_residues[entity1]])
        surface_frags2 = entity2.get_fragments([residue.number for residue in self.interface_residues[entity2]])
        if not surface_frags1 or not surface_frags2:
            self.log.debug('At interface Entity %s | Entity %s\tMISSING interface residues with matching fragments'
                           % (entity1.name, entity2.name))
            return None
        else:
            self.log.debug('At interface Entity %s | Entity %s\t%s has %s interface fragments'
                           % (entity1.name, entity2.name, len(surface_frags1), entity1.name))
            self.log.debug('At interface Entity %s | Entity %s\t%s has %s interface fragments'
                           % (entity1.name, entity2.name, len(surface_frags2), entity2.name))
        if self.symmetry:
            # even if entity1 == entity2, only need to expand the entity2 fragments due to surface/ghost frag mechanics
            # asu frag subtraction is unnecessary
            surface_frags2_nested = [self.return_symmetry_mates(frag) for frag in surface_frags2]
            surface_frags2 = list(chain.from_iterable(surface_frags2_nested))
            self.log.debug('Entity 2 Symmetry expanded fragment count: %d' % len(surface_frags2))

        entity1_coords = entity1.get_backbone_and_cb_coords()  # for clash check, we only want the backbone and CB
        ghostfrag_surfacefrag_pairs = find_fragment_overlap_at_interface(entity1_coords, surface_frags1, surface_frags2,
                                                                         fragdb=self.frag_db)
        self.log.info('Found %d overlapping fragment pairs at the %s | %s interface.'
                      % (len(ghostfrag_surfacefrag_pairs), entity1.name, entity2.name))
        self.fragment_observations.extend(ghostfrag_surfacefrag_pairs)  # Todo, change so not same as DesignDirectory
        fragment_matches = get_matching_fragment_pairs_info(ghostfrag_surfacefrag_pairs)
        self.fragment_queries[(entity1, entity2)] = fragment_matches

    def score_interface(self, entity1=None, entity2=None):
        if (entity1, entity2) not in self.fragment_queries and (entity2, entity1) not in self.fragment_queries:
            self.find_interface_residues(entity1=entity1, entity2=entity2)
            self.query_interface_for_fragments(entity1=entity1, entity2=entity2)
            self.calculate_fragment_query_metrics()

        return self.return_fragment_query_metrics(entity1=entity1, entity2=entity2, per_interface=True)

    def interface_design(self, design_dir=None, symmetry=None, evolution=True,
                         fragments=True, query_fragments=False, write_fragments=True,
                         frag_db='biological_interfaces',  # mask=None, output_assembly=False,
                         ):  # Todo initialize without DesignDirectory
        """Take the provided PDB, and use the ASU to compute calculations relevant to interface design.

        This process identifies the ASU (if one is not explicitly provided, enables Pose symmetry,
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
        self.log.debug('Entities: %s' % str(self.entities))
        self.log.info('Symmetry is: %s' % symmetry)  # Todo resolve duplication with below self.symmetry
        if symmetry and isinstance(symmetry, dict):  # Todo with crysts. Not sure about the dict. Also done on __init__
            self.set_symmetry(**symmetry)

        if fragments:
            if query_fragments:  # search for new fragment information
                # inherently gets interface residues for the designable entities
                self.generate_interface_fragments(out_path=design_dir.frags, write_fragments=write_fragments)
            else:  # add existing fragment information to the pose
                # must provide des_dir.fragment_observations from des_dir.gather_fragment_metrics then specify whether
                # the Entity in question is from mapped or paired (entity1 is mapped, entity2 is paired from Nanohedra)
                # Need to renumber fragments to Pose residue numbering when we add to the available queries
                # if existing_fragments:  # add provided fragment information to the pose
                #     # Todo DesignDirectory.gather_fragment_info(self)
                #     #   design_dir.fragment_observations?
                #     with open(existing_fragments, 'r') as f:
                #         fragment_source = f.readlines()
                # else:
                fragment_source = design_dir.fragment_observations
                if not fragment_source:
                    raise DesignError('%s: Fragments were set for design but there were none found in the Design '
                                      'Directory! Fix your input flags if this is not what you expected or generate '
                                      'them with \'%s generate_frags\'' % (str(design_dir), PUtils.program_command))

                # self.fragment_queries[tuple(entity.name for entity in self.entities)] = fragment_source
                if design_dir.nano:
                    entity_ids = tuple(entity.name for entity in self.entities)  # Todo compatible with > 2 entities
                    self.log.debug('Entity ID\'s: %s' % str(entity_ids))
                    self.add_fragment_query(entity1=entity_ids[0], entity2=entity_ids[1], query=fragment_source,
                                            pdb_numbering=True)
                else:  # assuming the input is in Pose numbering
                    self.log.debug('Fragment data being pulled from file')  #. Data:\n%s' % fragment_source)
                    self.add_fragment_query(query=fragment_source)
                # for entity in self.entities:
                #     entity.add_fragment_query(entity1=entity_ids[0], entity2=entity_ids[1], query=fragment_source,
                #                               pdb_numbering=True)

            for query_pair, fragment_info in self.fragment_queries.items():
                self.log.debug('Query Pair: %s, %s\nFragment Info: %s' % (query_pair[0].name, query_pair[1].name, fragment_info))
                for query_idx, entity in enumerate(query_pair):
                    # Attach an existing FragmentDB to the Pose
                    entity.attach_fragment_database(db=design_dir.frag_db)
                    # entity.connect_fragment_database(location=frag_db, db=design_dir.frag_db)
                    entity.assign_fragments(fragments=fragment_info,
                                            alignment_type=SequenceProfile.idx_to_alignment_type[query_idx])
        else:
            # get interface residues for the designable entities
            for entity_pair in combinations_with_replacement(self.active_entities, 2):
                self.find_interface_residues(*entity_pair)

        for entity in self.entities:
            # entity.retrieve_sequence_from_api(entity_id=entity)  # Todo
            # Todo check this assumption...
            #  DesignDirectory.path was removed from evol as the input oligomers should be perfectly symmetric so the tx
            #  won't matter for each entity. right?
            # TODO Insert loop identifying comparison of SEQRES and ATOM before SeqProf.calculate_design_profile()
            if entity not in self.active_entities:  # we shouldn't design
                entity.add_profile(null=True, out_path=design_dir.sequences)
            else:
                entity.add_profile(evolution=evolution, fragments=fragments, out_path=design_dir.sequences)

        # Update DesignDirectory with design information # Todo include in DesignDirectory initialization by args?
        # This info is pulled out in AnalyzeOutput from Rosetta currently

        if fragments:  # set pose.fragment_profile by combining entity frag profile into single profile
            self.combine_fragment_profile([entity.fragment_profile for entity in self.entities])
            self.log.debug('Fragment Specific Scoring Matrix: %s' % str(self.fragment_profile))
            self.interface_data_file = pickle_object(self.fragment_profile, frag_db + PUtils.frag_profile,
                                                     out_path=design_dir.data)
            design_dir.info['fragment_database'] = frag_db
            design_dir.info['fragment_profile'] = self.interface_data_file

        if evolution:  # set pose.evolutionary_profile by combining entity evo profile into single profile
            self.combine_pssm([entity.evolutionary_profile for entity in self.entities])
            self.log.debug('Position Specific Scoring Matrix: %s' % str(self.evolutionary_profile))
            self.pssm_file = self.write_pssm_file(self.evolutionary_profile, PUtils.pssm, out_path=design_dir.data)
            design_dir.info['evolutionary_profile'] = self.pssm_file

        self.combine_profile([entity.profile for entity in self.entities])
        self.log.debug('Design Specific Scoring Matrix: %s' % str(self.profile))
        self.design_pssm_file = self.write_pssm_file(self.profile, PUtils.dssm, out_path=design_dir.data)
        design_dir.info['design_profile'] = self.design_pssm_file
        design_dir.info['design_residues'] = self.interface_residues

        # -------------------------------------------------------------------------
        # self.solve_consensus()  # Todo
        # -------------------------------------------------------------------------

    def generate_interface_fragments(self, write_fragments=True, out_path=None, new_db=False):
        """Using the attached fragment database, generate interface fragments between the Pose interfaces

        Keyword Args:
            write_fragments=True (bool): Whether or not to write the located fragments
            out_path=None (str): The location to write each fragment file
            new_db=False (bool): Whether a fragment database should be initialized for the interface fragment search
        """
        if new_db:  # Connect to a new DB
            self.connect_fragment_database()  # default, init=True. other args are source= , location= ,
        elif not self.frag_db:  # There is no fragment database connected
            raise DesignError('%s: A fragment database is required to add fragments to the profile. Ensure you '
                              'initialized the Pose with a database (pass FragmentDatabase obj by \'frag_db\')! '
                              'Alternatively pass new_db=True to %s'
                              % (self.generate_interface_fragments.__name__,
                                 self.generate_interface_fragments.__name__))

        for entity_pair in combinations_with_replacement(self.active_entities, 2):
            self.find_interface_residues(*entity_pair)
            self.log.debug('Querying Entity pair: %s, %s for interface fragments'
                           % tuple(entity.name for entity in entity_pair))
            self.query_interface_for_fragments(*entity_pair)

        if write_fragments:
            write_fragment_pairs(self.fragment_observations, out_path=out_path)
            for match_count, frag_obs in enumerate(self.fragment_observations):
                write_frag_match_info_file(ghost_frag=frag_obs[0], matched_frag=frag_obs[1],
                                           overlap_error=z_value_from_match_score(frag_obs[2]),
                                           match_number=match_count, out_path=out_path)

    def return_symmetry_parameters(self):
        """Return the symmetry parameters from a SymmetricModel

        Returns:
            (dict): {symmetry: (str), dimension: (int), uc_dimensions: (list), expand_matrices: (list[list])}
        """
        temp_dict = copy.copy(self.__dict__)
        temp_dict.pop('models')
        temp_dict.pop('pdb')
        temp_dict.pop('asu')
        temp_dict.pop('coords_type')
        temp_dict.pop('number_of_models')

        return temp_dict

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
        oligomers[code].name = code
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
    pdb1_coords = np.array(pdb1.extract_CB_coords(InclGlyCA=gly_ca))
    pdb2_coords = np.array(pdb2.extract_CB_coords(InclGlyCA=gly_ca))

    # Construct CB Tree for PDB1
    pdb1_tree = BallTree(pdb1_coords)

    # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
    return pdb1_tree.query_radius(pdb2_coords, distance)


def find_interface_pairs(pdb1, pdb2, distance=8, gly_ca=True):
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


def find_fragment_overlap_at_interface(entity1_coords, interface_frags1, interface_frags2, fragdb=None, max_z_value=2):
    #           entity1, entity2, entity1_interface_residue_numbers, entity2_interface_residue_numbers, max_z_value=2):
    """From a Structure Entity, score the interface between them according to Nanohedra's fragment matching"""
    # Get entity1 interface fragments with guide coordinates using fragment database
    # interface_frags1 = get_fragments(entity1, entity1_interface_residue_numbers)

    if not fragdb:
        fragdb = FragmentDB()
        fragdb.get_monofrag_cluster_rep_dict()
        fragdb.get_intfrag_cluster_rep_dict()
        fragdb.get_intfrag_cluster_info_dict()

    kdtree_oligomer1_backbone = BallTree(entity1_coords)
    # kdtree_oligomer1_backbone = BallTree(np.array(entity1.extract_backbone_coords()))
    complete_int1_ghost_frag_l, interface_ghostfrag_guide_coords_list = [], []
    for frag1 in interface_frags1:
        complete_monofrag1 = MonoFragment(frag1, fragdb.reps)  # ijk_monofrag_cluster_rep_pdb_dict)
        complete_monofrag1_ghostfrag_list = complete_monofrag1.get_ghost_fragments(fragdb.paired_frags,
                                                                                   kdtree_oligomer1_backbone,
                                                                                   fragdb.info)
        if complete_monofrag1_ghostfrag_list:
            complete_int1_ghost_frag_l.extend(complete_monofrag1_ghostfrag_list)
            for ghost_frag in complete_monofrag1_ghostfrag_list:
                interface_ghostfrag_guide_coords_list.append(ghost_frag.get_guide_coords())

    # Get entity2 interface fragments with guide coordinates using complete fragment database
    # interface_frags2 = get_fragments(entity2, entity2_interface_residue_numbers)

    complete_int2_frag_l, interface_surf_frag_guide_coords_list = [], []
    for frag2 in interface_frags2:
        complete_monofrag2 = MonoFragment(frag2, fragdb.reps)
        complete_monofrag2_guide_coords = complete_monofrag2.get_guide_coords()
        if complete_monofrag2_guide_coords is not None:
            complete_int2_frag_l.append(complete_monofrag2)
            interface_surf_frag_guide_coords_list.append(complete_monofrag2_guide_coords)

    eul_lookup = EulerLookup()
    # Check for matching Euler angles
    # Todo prefilter guide_coords_list with i/j type true array
    eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(interface_ghostfrag_guide_coords_list,
                                                               interface_surf_frag_guide_coords_list)
    eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]

    all_fragment_overlap = filter_euler_lookup_by_zvalue(eul_lookup_true_list, complete_int1_ghost_frag_l,
                                                         interface_ghostfrag_guide_coords_list,
                                                         complete_int2_frag_l, interface_surf_frag_guide_coords_list,
                                                         z_value_func=calculate_overlap, max_z_value=max_z_value)
    # passing_fragment_overlap = list(filter(None, all_fragment_overlap))
    ghostfrag_surffrag_pairs = [(complete_int1_ghost_frag_l[eul_lookup_true_list[idx][0]],
                                 complete_int2_frag_l[eul_lookup_true_list[idx][1]], all_fragment_overlap[idx][0])
                                for idx, boolean in enumerate(all_fragment_overlap) if boolean]

    return ghostfrag_surffrag_pairs


def get_matching_fragment_pairs_info(ghostfrag_surffrag_pairs):
    fragment_matches = []
    for interface_ghost_frag, interface_mono_frag, match_score in ghostfrag_surffrag_pairs:
        entity1_surffrag_ch, entity1_surffrag_resnum = interface_ghost_frag.get_aligned_surf_frag_central_res_tup()
        entity2_surffrag_ch, entity2_surffrag_resnum = interface_mono_frag.get_central_res_tup()
        fragment_matches.append({'mapped': entity1_surffrag_resnum, 'match': match_score,
                                 'paired': entity2_surffrag_resnum, 'cluster': '%s_%s_%s'
                                                                               % interface_ghost_frag.get_ijk()})
    # if log:
    #     log.debug('Fragments for Entity1 found at residues: %s' % [fragment['mapped'] for fragment in fragment_matches])
    #     log.debug('Fragments for Entity2 found at residues: %s' % [fragment['paired'] for fragment in fragment_matches])
    logger.debug('Fragments for Entity1 found at residues: %s' % [fragment['mapped'] for fragment in fragment_matches])
    logger.debug('Fragments for Entity2 found at residues: %s' % [fragment['paired'] for fragment in fragment_matches])

    return fragment_matches


def write_fragment_pairs(ghostfrag_surffrag_pairs, out_path=os.getcwd()):
    for idx, (interface_ghost_frag, interface_mono_frag, match_score) in enumerate(ghostfrag_surffrag_pairs):
        interface_ghost_frag.pdb.write(out_path=os.path.join(out_path, '%s_%s_%s_fragment_overlap_match_%d.pdb'
                                                             % (*interface_ghost_frag.get_ijk(), idx)))


def calculate_interface_score(interface_pdb, write=False, out_path=os.getcwd()):
    """Takes as input a single PDB with two chains and scores the interface using fragment decoration"""
    interface_name = interface_pdb.name

    entity1 = PDB.from_atoms(interface_pdb.chain(interface_pdb.chain_id_list[0]).get_atoms())
    entity1.update_attributes_from_pdb(interface_pdb)
    entity2 = PDB.from_atoms(interface_pdb.chain(interface_pdb.chain_id_list[-1]).get_atoms())
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
    entity1_coords = entity1.extract_coords()

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
                         'percent_residues_fragment_all': percent_interface_covered,
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