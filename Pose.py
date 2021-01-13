import os
import pickle
from glob import glob

import numpy as np

import PathUtils as PUtils
from PDB import PDB
from SequenceProfile import SequenceProfile
# Globals
from SymDesignUtils import to_iterable, logger

config_directory = PUtils.pdb_db
sym_op_location = PUtils.sym_op_location
sg_cryst1_fmt_dict = {'F222': 'F 2 2 2', 'P6222': 'P 62 2 2', 'I4132': 'I 41 3 2', 'P432': 'P 4 3 2',
                      'P6322': 'P 63 2 2', 'I4122': 'I 41 2 2', 'I213': 'I 21 3', 'I422': 'I 4 2 2',
                      'I432': 'I 4 3 2', 'P4222': 'P 42 2 2', 'F23': 'F 2 3', 'P23': 'P 2 3', 'P213': 'P 21 3',
                      'F432': 'F 4 3 2', 'P622': 'P 6 2 2', 'P4232': 'P 42 3 2', 'F4132': 'F 41 3 2',
                      'P4132': 'P 41 3 2', 'P422': 'P 4 2 2', 'P312': 'P 3 1 2', 'R32': 'R 3 2'}
pg_cryst1_fmt_dict = {'p3': 'P 3', 'p321': 'P 3 2 1', 'p622': 'P 6 2 2', 'p4': 'P 4', 'p222': 'P 2 2 2',
                      'p422': 'P 4 2 2', 'p4212': 'P 4 21 2', 'p6': 'P 6', 'p312': 'P 3 1 2', 'c222': 'C 2 2 2'}


class Model:
    def __init__(self, pdb, symmetry=None):
        if isinstance(pdb, list):
            self.model_list = pdb  # list of PDB objects
            self.pdb = self.model_list[0]
        elif isinstance(pdb, PDB):
            self.pdb = pdb
            self.model_list = None

    def add_pdb(self, pdb):
        self.model_list.append(pdb)

    def add_atoms_to_pdb(self, index=0, atoms=None):
        """Add atoms to a PDB object in the model. Zero indexed"""
        self.model_list[index].read_atom_list(atoms)

    def get_CA_atoms(self):
        return Model([pdb.get_CA_atoms() for pdb in self.model_list])

    def select_chain(self, chain_id):
        return Model([pdb.get_chain_atoms(chain_id) for pdb in self.model_list])

    def extract_all_coords(self):
        return [pdb.extract_all_coords() for pdb in self.model_list]

    def extract_backbone_coords(self):
        return [pdb.extract_backbone_coords() for pdb in self.model_list]

    def extract_CA_coords(self):
        return [pdb.extract_CA_coords() for pdb in self.model_list]

    def extract_CB_coords(self, InclGlyCA=False):
        return [pdb.extract_CB_coords(InclGlyCA=InclGlyCA) for pdb in self.model_list]

    def extract_CB_coords_chain(self, chain_id, InclGlyCA=False):
        return [pdb.extract_CB_coords_chain(chain_id, InclGlyCA=InclGlyCA) for pdb in self.model_list]

    def get_CB_coords(self, ReturnWithCBIndices=False, InclGlyCA=False):
        return [pdb.get_CB_coords(ReturnWithCBIndices=ReturnWithCBIndices, InclGlyCA=InclGlyCA) for pdb in self.model_list]

    def replace_coords(self, new_cords):
        return [pdb.replace_coords(new_cords[i]) for i, pdb in enumerate(self.model_list)]


class Pose(PDB, SequenceProfile):
    # made of multiple pdb objects
    def __init__(self, initial_pdb=None, file=None, symmetry=None):
        super().__init__()  # PDB init
        super().__init__(structure=self)  # SequenceProfile init
        self.pdb = None  # the pose specific pdb
        self.pdbs = []  # the member pdbs which make up the pose
        # self.entities = []  # from PDB
        self.pdbs_d = {}
        self.pose_pdb_accession_map = {}
        self.uc_dimensions = []
        self.dimension = None
        self.expand_matrices = None
        self.uc_sym_mates = []
        self.surrounding_uc_sym_mates = []
        self.symmetry = None

        self.set_symmetry(symmetry)

        if initial_pdb:
            self.add_pdb(initial_pdb)

        if file:
            pdb = PDB(file=file)
            # Depending on the extent of PDB class initialization, I could copy the PDB info into self.pdb
            # this would be:
            # coords, atoms, residues, chains, entities, design, (host of others read from file)
            self.add_pdb(pdb)

    @classmethod
    def from_pdb(cls, pdb):
        return cls(initial_pdb=pdb)

    # @classmethod  # In PDB class
    # def from_file(cls, file):
    #     return cls(file=file)

    def add_pdb(self, pdb):
        """Add a PDB to the PosePDB as well as the member PDB list"""
        self.pdbs.append(pdb)
        self.pdbs_d[pdb.name] = pdb
        # self.pdbs_d[id(pdb)] = pdb
        self.add_entities_to_pose(self, pdb)

    def add_entities_to_pose(self, pdb):
        """Add each unique entity in a pdb to the pose, updating all metadata"""
        self.pose_pdb_accession_map[pdb.name] = pdb.entity_accession_map
        for entity in pdb.accession_entity_map:
            self.add_entity(pdb, entity)

    def add_entity(self, pdb, entity):
        # Todo Entity()
        entity_chain = pdb.entities[entity]['representative']
        self.pdb.add_atoms(pdb.get_chain_atoms(entity_chain))

    def initialize_pose(self):

        for entity in pose:
            # must provide the list from des_dir.gather_fragment_metrics or InterfaceScoring.py then specify whether the
            # Entity in question is from mapped or paired
            # such as fragment_source = des_dir.fragment_observations
            fragment_info_source = 'mapped'  # Todo store in variable like des_dir.oligomer1?
            entity.connect_fragment_database(location='biological_interfaces')
            entity.add_profile(fragment_source=des_dir.fragment_observations,
                               frag_alignment_type=fragment_info_source,
                               out_path=des_dir.sequences, pdb_numbering=True)

        # Extract PSSM for each protein and combine into single PSSM TODO full pose!
        pose.combine_pssm([entity.pssm for entity in pose])  # sets pose.pssm
        logger.debug('Position Specific Scoring Matrix: %s' % str(pose.pssm))
        pssm_file = pose.make_pssm_file(pose.pssm, PUtils.msa_pssm, outpath=des_dir.path)  # static

        pose.combine_fragment_profile([entity.fragment_profile for entity in pose])  # sets pose.pssm
        logger.debug('Fragment Specific Scoring Matrix: %s' % str(pose.fragment_profile))
        # Todo, remove missing fragment entries here or add where they are loaded keep_extras = False  # =False added for pickling 6/14/20
        interface_data_file = SDUtils.pickle_object(final_issm, frag_db + PUtils.frag_type, out_path=des_dir.data)

        pose.combine_dssm([entity.dssm for entity in pose])  # sets pose.pssm
        logger.debug('Design Specific Scoring Matrix: %s' % str(pose.dssm))
        dssm_file = pose.make_pssm_file(pose.dssm, PUtils.dssm, outpath=des_dir.path)  # static

        self.solve_consensus()

        # Update DesignDirectory with design information
        des_dir.info['pssm'] = pssm_file
        des_dir.info['issm'] = interface_data_file
        des_dir.info['dssm'] = dssm_file
        des_dir.info['db'] = frag_db
        des_dir.info['des_residues'] = [j for name in names for j in int_res_numbers[name]]
        # TODO add oligomer data to .info
        info_pickle = SDUtils.pickle_object(des_dir.info, 'info', out_path=des_dir.data)

    def set_symmetry(self, symmetry):
        self.symmetry = ''.join(symmetry.split())  # ensure the symmetry is Hermann–Mauguin notation
        if self.symmetry:
            if self.symmetry in ['T', 'O', 'I']:
                self.dimension = 0
                self.expand_matrices = self.get_ptgrp_sym_op(self.symmetry)
            else:
                if self.symmetry in pg_cryst1_fmt_dict:
                    self.dimension = 2
                elif self.symmetry in sg_cryst1_fmt_dict:
                    self.dimension = 3

                self.expand_matrices = self.get_sg_sym_op(self.symmetry)

    def expand_asu(self, return_side_chains=False):
        """Return the expanded material from the input ASU, symmetry specification, and unit cell dimensions. Expanded
        to entire point group, 3x3 layer group, or 3x3x3 space group

        Keyword Args:
            return_side_chains=False (bool): Whether to return all side chain atoms
        """
        if self.symmetry in ['T', 'O', 'I']:
            self.get_expanded_ptgrp_pdb(return_side_chains=return_side_chains)
        else:
            self.expand_uc(return_side_chains=return_side_chains)

    def expand_uc(self, return_side_chains=False):
        """Expand the backbone coordinates for every symmetric copy within the unit cells surrounding a central cell
        """
        self.get_unit_cell_sym_mates()
        self.get_surrounding_unit_cells(return_side_chains=return_side_chains)

    def cart_to_frac(self, cart_coords):
        """Takes a numpy array of coordinates and finds the fractional coordinates from cartesian coordinates
        From http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm
        """
        a2r = np.pi / 180.0
        a, b, c, alpha, beta, gamma = self.uc_dimensions
        alpha *= a2r
        beta *= a2r
        gamma *= a2r

        # unit cell volume
        v = a * b * c * np.sqrt((1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * (
                np.cos(alpha) * np.cos(beta) * np.cos(gamma))))

        # deorthogonalization matrix M
        M_0 = [1 / a, -(np.cos(gamma) / float(a * np.sin(gamma))), (((b * np.cos(gamma) * c * (
                np.cos(alpha) - (np.cos(beta) * np.cos(gamma)))) / float(np.sin(gamma))) - (
                                                                            b * c * np.cos(beta) * np.sin(
                                                                        gamma))) * (1 / float(v))]
        M_1 = [0, 1 / (b * np.sin(gamma)),
               -((a * c * (np.cos(alpha) - (np.cos(beta) * np.cos(gamma)))) / float(v * np.sin(gamma)))]
        M_2 = [0, 0, (a * b * np.sin(gamma)) / float(v)]
        M = np.array([M_0, M_1, M_2])

        return np.matmul(cart_coords, np.transpose(M))

    def frac_to_cart(self, frac_coords):
        """Takes a numpy array of coordinates and finds the cartesian coordinates from fractional coordinates
        From http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm
        """
        a2r = np.pi / 180.0
        a, b, c, alpha, beta, gamma = self.uc_dimensions
        alpha *= a2r
        beta *= a2r
        gamma *= a2r

        # unit cell volume
        v = a * b * c * np.sqrt((1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * (
                np.cos(alpha) * np.cos(beta) * np.cos(gamma))))

        # orthogonalization matrix M_inv
        M_inv_0 = [a, b * np.cos(gamma), c * np.cos(beta)]
        M_inv_1 = [0, b * np.sin(gamma),
                   (c * (np.cos(alpha) - (np.cos(beta) * np.cos(gamma)))) / float(np.sin(gamma))]
        M_inv_2 = [0, 0, v / float(a * b * np.sin(gamma))]
        M_inv = np.array([M_inv_0, M_inv_1, M_inv_2])

        return np.matmul(frac_coords, np.transpose(M_inv))

    def get_expanded_ptgrp_pdb(self, return_side_chains=False):
        """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
        asu_coords = self.pdb.extract_all_coords()
        for r in self.expand_matrices:
            r_mat = np.transpose(np.array(r))
            r_asu_coords = np.matmul(asu_coords, r_mat)

            asu_sym_mate_pdb = PDB()
            asu_sym_mate_pdb_atom_list = []
            atom_count = 0
            for atom in self.pdb.get_atoms():
                x_transformed = r_asu_coords[atom_count][0]
                y_transformed = r_asu_coords[atom_count][1]
                z_transformed = r_asu_coords[atom_count][2]
                atom_transformed = Atom(atom_count, atom.get_type(), atom.get_alt_location(),
                                        atom.get_residue_type(), atom.get_chain(),
                                        atom.get_residue_number(),
                                        atom.get_code_for_insertion(), x_transformed, y_transformed,
                                        z_transformed,
                                        atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                        atom.get_atom_charge())
                atom_count += 1
                asu_sym_mate_pdb_atom_list.append(atom_transformed)

            asu_sym_mate_pdb.set_all_atoms(asu_sym_mate_pdb_atom_list)
            self.uc_sym_mates.append(asu_sym_mate_pdb)

    def get_unit_cell_sym_mates(self, return_side_chains=True):
        """Return all symmetry mates as a list of PDB objects. Chain names will match the ASU"""
        self.uc_sym_mates = [self.pdb]

        if return_side_chains:  # get different function calls depending on the return type
            extract_pdb_atoms = getattr(PDB, 'get_atoms')
            extract_pdb_coords = getattr(PDB, '.extract_all_coords')
        else:
            extract_pdb_atoms = getattr(PDB, 'get_backbone_atoms')
            extract_pdb_coords = getattr(PDB, 'extract_backbone_coords')

        asu_cart_coords = extract_pdb_coords(self.pdb)
        asu_frac_coords = self.cart_to_frac(np.array(asu_cart_coords))

        for r, t in self.expand_matrices:
            t_vec = np.array(t)
            r_mat = np.transpose(np.array(r))

            r_asu_frac_coords = np.matmul(asu_frac_coords, r_mat)
            tr_asu_frac_coords = r_asu_frac_coords + t_vec

            tr_asu_cart_coords = self.frac_to_cart(np.array(tr_asu_frac_coords)).tolist()

            unit_cell_sym_mate_pdb = PDB()
            unit_cell_sym_mate_pdb_atom_list = []
            # Todo replace routine with pdb.replace_coords ?
            atom_count = 0
            for atom in extract_pdb_atoms(self.pdb):
                x_transformed = tr_asu_cart_coords[atom_count][0]
                y_transformed = tr_asu_cart_coords[atom_count][1]
                z_transformed = tr_asu_cart_coords[atom_count][2]
                atom_transformed = Atom(atom_count, atom.get_type(), atom.get_alt_location(),
                                        atom.get_residue_type(), atom.get_chain(),
                                        atom.get_residue_number(),
                                        atom.get_code_for_insertion(), x_transformed, y_transformed,
                                        z_transformed,
                                        atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                        atom.get_atom_charge())
                atom_count += 1
                unit_cell_sym_mate_pdb_atom_list.append(atom_transformed)

            unit_cell_sym_mate_pdb.set_all_atoms(unit_cell_sym_mate_pdb_atom_list)
            self.uc_sym_mates.append(unit_cell_sym_mate_pdb)

    def get_surrounding_unit_cells(self, return_side_chains=False):
        """Returns a grid of unit cells for a symmetry group. Each unit cell is a list of ASU's in total grid list"""
        if self.dimension == 3:
            z_shifts, uc_copy_number = [-1, 0, 1], 8
        elif self.dimension == 2:
            z_shifts, uc_copy_number = [0], 26
        else:
            return None

        if return_side_chains:  # get different function calls depending on the return type
            extract_pdb_atoms = getattr(PDB, 'get_atoms')
            extract_pdb_coords = getattr(PDB, '.extract_all_coords')
        else:
            extract_pdb_atoms = getattr(PDB, 'get_backbone_atoms')
            extract_pdb_coords = getattr(PDB, 'extract_backbone_coords')

        asu_atom_template = extract_pdb_atoms(self.uc_sym_mates[0])
        # asu_bb_atom_template = uc_sym_mates[0].get_backbone_atoms()

        central_uc_cart_coords = []
        for unit_cell_sym_mate_pdb in self.uc_sym_mates:
            central_uc_cart_coords.extend(extract_pdb_coords(unit_cell_sym_mate_pdb))
            # central_uc_bb_cart_coords.extend(unit_cell_sym_mate_pdb.extract_backbone_coords())
        central_uc_frac_coords = self.cart_to_frac(np.array(central_uc_cart_coords))

        all_surrounding_uc_frac_coords = []
        for x_shift in [-1, 0, 1]:
            for y_shift in [-1, 0, 1]:
                for z_shift in z_shifts:
                    if [x_shift, y_shift, z_shift] != [0, 0, 0]:
                        shifted_uc_frac_coords = central_uc_frac_coords + [x_shift, y_shift, z_shift]
                        all_surrounding_uc_frac_coords.extend(shifted_uc_frac_coords)

        all_surrounding_uc_cart_coords = self.frac_to_cart(np.array(all_surrounding_uc_frac_coords))
        all_surrounding_uc_cart_coords = np.split(all_surrounding_uc_cart_coords, uc_copy_number)

        for surrounding_uc_cart_coords in all_surrounding_uc_cart_coords:
            all_uc_sym_mates_cart_coords = np.split(surrounding_uc_cart_coords, len(self.uc_sym_mates))
            one_surrounding_unit_cell = []
            for uc_sym_mate_cart_coords in all_uc_sym_mates_cart_coords:
                uc_sym_mate_atoms = []
                for atom_count, atom in enumerate(asu_atom_template):
                    x_transformed = uc_sym_mate_cart_coords[atom_count][0]
                    y_transformed = uc_sym_mate_cart_coords[atom_count][1]
                    z_transformed = uc_sym_mate_cart_coords[atom_count][2]
                    atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
                                            atom.get_residue_type(), atom.get_chain(),
                                            atom.get_residue_number(),
                                            atom.get_code_for_insertion(), x_transformed, y_transformed,
                                            z_transformed,
                                            atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                            atom.get_atom_charge())
                    uc_sym_mate_atoms.append(atom_transformed)

                uc_sym_mate_pdb = PDB(atoms=uc_sym_mate_atoms)
                one_surrounding_unit_cell.append(uc_sym_mate_pdb)

            self.surrounding_uc_sym_mates.extend(one_surrounding_unit_cell)

    def write(self, name, location=os.getcwd(), cryst1=None):  # Todo write model, write symmetry
        out_path = os.path.join(location, '%s.pdb' % name)
        with open(out_path, 'w') as f:
            if cryst1 and isinstance(cryst1, str) and cryst1.startswith('CRYST1'):
                f.write('%s\n' % cryst1)
            for i, model in enumerate(self.model_list, 1):
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