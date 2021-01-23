import math

import numpy as np
from Bio.SeqUtils import IUPACData


# from Bio.Alphabet import IUPAC


class Structure:  # (Coords):
    def __init__(self, atoms=None, residues=None, name=None, coords=None, **kwargs):
        super().__init__(**kwargs)
        # self.coords = coords
        # super().__init__(coords=coords)  # gets self.coords
        self.atoms = []  # atoms
        self.residues = []  # residues
        # self.id = None
        self.name = name
        self.secondary_structure = None
        self.center_of_mass = None
        # self.sequence = None

        if atoms:
            self.set_atoms(atoms)
        if residues:
            self.set_residues(residues)
        if isinstance(coords, np.ndarray) and coords.any():
            self.coords = coords

    @classmethod
    def from_atoms(cls, atoms):
        return cls(atoms=atoms)

    @classmethod
    def from_residues(cls, residues):
        return cls(residues=residues)

    # @property Todo
    def get_name(self):
        # return self.id
        return self.name

    def set_name(self, name):
        self.name = name

    @property
    def coords(self):
        return self._coords.get_indicies(self.atom_indecies)

    @coords.setter
    def coords(self, coords):
        # assert len(self.atoms) == coords.shape[0], '%s: ERROR number of Atoms (%d) != number of Coords (%d)!' \
        #                                                 % (self.name, len(self.atoms), self.coords.shape[0])
        self._coords = coords
        for atom in self.atoms:
            atom.coords = coords

    @property
    def atom_indecies(self):  # Todo has relevance to Residue
        # if self._atom_indecies:
        try:
            # self._atom_indecies:
            return self._atom_indecies
        except AttributeError:
        # else:
            self.atom_indecies = [atom.index for atom in self.atoms]
            return self._atom_indecies

    @atom_indecies.setter
    def atom_indecies(self, indecies):
        self._atom_indecies = np.array(indecies)

    def get_coords(self):
        """Return the numpy array of Coords from the Structure"""
        return self.coords

    def get_backbone_coords(self):
        """Return a view of the numpy array of Coords from the Structure with only backbone atom coordinates"""
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_backbone()]
        return self.coords[index_mask]

    def get_backbone_and_cb_coords(self):
        """Return a view of the numpy array of Coords from the Structure with backbone and CB atom coordinates
        inherently gets all glycine CA's"""
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_backbone() or atom.is_CB()]
        return self.coords[index_mask]

    def get_ca_coords(self):
        """Return a view of the numpy array of Coords from the Structure with CA atom coordinates"""
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_CA()]
        return self.coords[index_mask]

    def get_cb_coords(self, InclGlyCA=True):
        """Return a view of the numpy array of Coords from the Structure with CB atom coordinates"""
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_CB(InclGlyCA=InclGlyCA)]
        return self.coords[index_mask]

    def extract_coords(self):
        """Grab all the coordinates from the Structure's Coords, returns a list with views of the Coords array"""
        return [atom.coords for atom in self.get_atoms()]

    def extract_backbone_coords(self):
        return [atom.coords for atom in self.get_atoms() if atom.is_backbone()]

    def extract_backbone_and_cb_coords(self):
        # inherently gets all glycine CA's
        return [atom.coords for atom in self.get_atoms() if atom.is_backbone() or atom.is_CB()]

    def extract_CA_coords(self):
        return [atom.coords for atom in self.get_atoms() if atom.is_CA()]

    def extract_CB_coords(self, InclGlyCA=False):
        return [atom.coords for atom in self.get_atoms() if atom.is_CB(InclGlyCA=InclGlyCA)]

    # @property Todo
    def get_atoms(self, numbers=None):
        """Retrieve Atoms in structure. Returns all by default. If numbers=(list) selected Atom numbers are returned"""
        if numbers:
            return [atom for atom in self.atoms if atom.number in numbers]
        else:
            return self.atoms

    def set_atoms(self, atom_list):
        """Set the Structure atoms to Atoms in atom_list"""
        self.atoms = atom_list
        self.create_residues()
        # self.update_structure(atom_list)
        self.set_length()

    def add_atoms(self, atom_list):
        """Add Atoms in atom_list to the structure instance"""
        self.atoms.extend(atom_list)
        self.create_residues()
        # self.update_structure(atom_list)
        self.set_length()

    def update_structure(self, atom_list):
        # self.reindex_atoms()
        # self.coords = np.append(self.coords, [atom.coords for atom in atom_list])
        # self.set_atom_coordinates(self.coords)
        # self.create_residues()
        self.set_length()

    def get_atom_indices(self):
        return [atom.index for atom in self.get_atoms()]

    def get_backbone_indices(self):
        return [atom.index for atom in self.get_atoms() if atom.is_backbone()]
        # return [idx for idx, atom in enumerate(self.get_atoms()) if atom.is_backbone()]  # for a structure/atom 1:1 correspondance

    def get_cb_indices(self, InclGlyCA=False):
        return [atom.index for atom in self.get_atoms() if atom.is_CB(InclGlyCA=InclGlyCA)]

    def get_helix_cb_indices(self):
        """Only works on secondary structure assigned structures!"""
        h_cb_indices = []
        for idx, residue in enumerate(self.get_residues()):
            if not residue.get_secondary_structure():
                print('Secondary Structures must be set before finding helical CB\'s! Error at Residue %s' % residue.number)
                return None
            elif residue.get_secondary_structure() == 'H':
                h_cb_indices.append(residue.get_cb())

        return h_cb_indices

    def get_ca_atoms(self):
        return [atom for atom in self.get_atoms() if atom.is_CA()]

    def get_cb_atoms(self):
        return [atom for atom in self.get_atoms() if atom.is_CB()]

    def get_backbone_atoms(self):
        return [atom for atom in self.get_atoms() if atom.is_backbone()]

    def get_backbone_and_cb_atoms(self):
        return [atom for atom in self.get_atoms() if atom.is_backbone() or atom.is_CB()]

    def atom(self, atom_number):
        """Retrieve the Atom specified"""
        for atom in self.atoms:
            if atom.number == atom_number:
                return atom
        return None

    def renumber_atoms(self):
        """Renumber all atom entries one-indexed according to list order"""
        for idx, atom in enumerate(self.atoms, 1):
            atom.number = idx

    def reindex_atoms(self):
        for idx, atom in enumerate(self.atoms):
            atom.index = idx

    # def set_atom_coordinates(self, coords):
    #     """Set/Replace all Atom coordinates with coords specified. Must be in the same order to apply correctly!"""
    #     assert len(self.atoms) == coords.shape[0], '%s: ERROR setting Atom coordinates, # Atoms (%d) !=  # Coords (%d)'\
    #                                                % (self.name, len(self.atoms), coords.shape[0])
    #     self.coords = coords
    #     for idx, atom in enumerate(self.get_atoms()):
    #         atom.coords = coords[idx]
    #         # atom.x, atom.y, atom.z = coords[idx][0], coords[idx][1], coords[idx][2]

    # @property Todo
    def get_residues(self, numbers=None):
        """Retrieve Residues in structure. Returns all by default. If numbers=(list) selected Residues numbers are
        returned"""
        if numbers:
            return [residue for residue in self.residues if residue.number in numbers]
        else:
            return self.residues

    def set_residues(self, residue_list):
        """Set the Structure residues to Residues provided in a residue list"""
        self.residues = residue_list  # []
        atom_list = [atom for residue in residue_list for atom in residue.get_atoms()]
        self.atoms = atom_list
        # self.update_structure(atom_list)
        self.set_length()

    def add_residues(self, residue_list):
        """Add Residues in a residue list to the Structure instance"""
        self.residues.extend(residue_list)
        atom_list = [atom for atom in residue_list.get_atoms()]
        self.atoms.extend(atom_list)
        # self.update_structure(atom_list)
        self.set_length()

    # update_structure():
    #  self.reindex_atoms() -> self.coords = np.append(self.coords, [atom.coords for atom in atom_list]) ->
    #  self.set_atom_coordinates(self.coords) -> self.create_residues() -> self.set_length()

    def create_residues(self):
        """For the Structure, create all possible Residue instances"""
        current_residue_number = self.atoms[0].residue_number
        current_residue = []
        for atom in self.atoms:
            if atom.residue_number == current_residue_number:
                current_residue.append(atom)
            else:
                self.residues.append(Residue(current_residue))
                current_residue = [atom]
                current_residue_number = atom.residue_number
        # ensure last residue is added after iteration is complete
        self.residues.append(Residue(current_residue))

    def residue(self, residue_number):
        """Retrieve the Residue specified"""
        for residue in self.residues:
            if residue.number == residue_number:
                return residue
        return None

    def residue_from_pdb(self, residue_number):
        """Returns the Residue object from the Structure according to PDB residue number"""
        for residue in self.residues:
            if residue.number_pdb == residue_number:
                return residue
        return None

    def renumber_residues(self):
        """Starts numbering Residues at 1 and number sequentially until last Residue"""
        last_atom_index = len(self.atoms)
        idx = 0  # offset , 1
        for i, residue in enumerate(self.residues, 1):
            # current_res_num = self.atoms[idx].residue_number
            current_res_num = residue.number
            while self.atoms[idx].residue_number == current_res_num:
                self.atoms[idx].residue_number = i  # + offset
                idx += 1
                if idx == last_atom_index:
                    break
        # self.renumber_atoms()  # should be unnecessary

    def residue_number_from_pdb(self, residue_number):
        """Returns the pose residue number from the queried .pdb number"""
        for residue in self.residues:
            if residue.number_pdb == residue_number:
                return residue.number
        return None

    def residue_number_to_pdb(self, residue_number):
        """Returns the .pdb residue number from the queried pose number"""
        for residue in self.residues:
            if residue.number == residue_number:
                return residue.number_pdb
        return None

    def set_length(self):
        self.number_of_atoms = len(self.get_atoms())
        self.number_of_residues = len(self.get_residues())

    @property
    def number_of_atoms(self):
        return self._number_of_atoms

    @number_of_atoms.setter
    def number_of_atoms(self, length):
        self._number_of_atoms = length

    @property
    def number_of_residues(self):
        return self._number_of_residues

    @number_of_residues.setter
    def number_of_residues(self, length):
        self._number_of_residues = length

    def get_center_of_mass(self):
        return self.center_of_mass

    def find_center_of_mass(self):
        """Given a numpy array of 3D coordinates, return the center of mass"""
        divisor = 1 / len(self.atom_indecies)
        return np.matmul(np.full((1, 3), divisor), np.transpose(self.coords))

    def get_structure_sequence(self):
        """Returns the single AA sequence of Residues found in the Structure. Handles odd residues by marking with '-'
        """
        sequence_list = [residue.type for residue in self.get_residues()]
        sequence = ''.join([IUPACData.protein_letters_3to1_extended[k.title()]
                            if k.title() in IUPACData.protein_letters_3to1_extended else '-'
                            for k in sequence_list])

        return sequence

    # def stride(self, chain=None):
    #     # REM  -------------------- Secondary structure summary -------------------  XXXX
    #     # REM                .         .         .         .         .               XXXX
    #     # SEQ  1    IVQQQNNLLRAIEAQQHLLQLTVWGIKQLQAGGWMEWDREINNYTSLIHS   50          XXXX
    #     # STR       HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH  HHHHHHHHHHHHHHHHH               XXXX
    #     # REM                                                                        XXXX
    #     # SEQ  51   LIEESQN                                              57          XXXX
    #     # STR       HHHHHH                                                           XXXX
    #     # REM                                                                        XXXX
    #     # LOC  AlphaHelix   ILE     3 A      ALA     33 A                            XXXX
    #     # LOC  AlphaHelix   TRP    41 A      GLN     63 A                            XXXX
    #     # REM                                                                        XXXX
    #     # REM  --------------- Detailed secondary structure assignment-------------  XXXX
    #     # REM                                                                        XXXX
    #     # REM  |---Residue---|    |--Structure--|   |-Phi-|   |-Psi-|  |-Area-|      XXXX
    #     # ASG  ILE A    3    1    H    AlphaHelix    360.00    -29.07     180.4      XXXX
    #     # ASG  VAL A    4    2    H    AlphaHelix    -64.02    -45.93      99.8      XXXX
    #     # ASG  GLN A    5    3    H    AlphaHelix    -61.99    -39.37      82.2      XXXX
    #
    #     # try:
    #         # with open(os.devnull, 'w') as devnull:
    #     stride_cmd = [stride_exe_path, '%s' % self.filepath]
    #     #   -rId1Id2..  Read only chains Id1, Id2 ...
    #     #   -cId1Id2..  Process only Chains Id1, Id2 ...
    #     if chain:
    #         stride_cmd.append('-c%s' % chain_id)
    #
    #     p = subprocess.Popen(stride_cmd, stderr=subprocess.DEVNULL)
    #     out, err = p.communicate()
    #     out_lines = out.decode('utf-8').split('\n')
    #     # except:
    #     #     stride_out = None
    #
    #     # if stride_out is not None:
    #     #     lines = stride_out.split('\n')
    #
    #     for line in out_lines:
    #         if line[0:3] == 'ASG' and line[10:15].strip().isdigit():   # Todo sort out chain issues
    #             self.chain(line[9:10]).residue(int(line[10:15].strip())).set_secondary_structure(line[24:25])
    #     self.secondary_structure = [residue.get_secondary_structure() for residue in self.get_residues()]
    #     # self.secondary_structure = {int(line[10:15].strip()): line[24:25] for line in out_lines
    #     #                             if line[0:3] == 'ASG' and line[10:15].strip().isdigit()}
    #
    def is_n_term_helical(self, window=5):
        """Using assigned secondary structure, probe for a helical N-termini using a sequence seqment of five residues

        Keyword Args:
            window=5 (int): The segment size to search
        Returns:
            (bool): Whether the termini has a stretch of helical residues with length of the window
        """
        if self.secondary_structure and len(self.secondary_structure) >= 2 * window:
            for idx, residue_secondary_structure in enumerate(self.secondary_structure):
                temp_window = ''.join(self.secondary_structure[idx + j] for j in range(window))
                # res_number = self.secondary_structure[0 + i:5 + i][0][0]
                if 'H' * window in temp_window:
                    return True  # , res_number
                if idx == window:
                    break
        return False  # , None

    def is_c_term_helical(self, window=5):
        """Using assigned secondary structure, probe for a helical C-termini using a sequence seqment of five residues

        Keyword Args:
            window=5 (int): The segment size to search
        Returns:
            (bool): Whether the termini has a stretch of helical residues with length of the window
        """
        if self.secondary_structure and len(self.secondary_structure) >= 2 * window:
            # for i in range(5):
            for idx, residue_secondary_structure in enumerate(reversed(self.secondary_structure)):
                # reverse_ss_asg = self.secondary_structure[::-1]
                temp_window = ''.join(self.secondary_structure[idx + j] for j in range(-window + 1, 1))
                # res_number = reverse_ss_asg[0+i:5+i][4][0]
                if 'H' * window in temp_window:
                    return True  # , res_number
                if idx == window:
                    break
        return False  # ,

    def get_secondary_structure(self):
        if self.secondary_structure:
            return self.secondary_structure
        else:
            self.fill_secondary_structure()
            if list(filter(None, self.secondary_structure)):  # check if there is at least 1 secondary struc assignment
                return self.secondary_structure
            else:
                return None

    def fill_secondary_structure(self, secondary_structure=None):
        if secondary_structure:
            self.secondary_structure = secondary_structure
        else:
            self.secondary_structure = [residue.get_secondary_structure() for residue in self.get_residues()]

    def write(self, out_path=None, file_handle=None):
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle"""
        if file_handle:
            file_handle.write('\n'.join(str(atom) for atom in self.get_atoms()))

        if out_path:
            with open(out_path, 'w') as outfile:
                outfile.write('\n'.join(str(atom) for atom in self.get_atoms()))

    def get_fragments(self, residue_numbers, fragment_length=5):
        interface_frags = []
        for residue_number in residue_numbers:
            frag_residue_numbers = [residue_number + i for i in range(-2, 3)]  # Todo parameterize
            # frag_atoms, ca_present = [], []
            ca_count = 0
            for residue in self.residue(frag_residue_numbers):
                # frag_atoms.extend(residue.get_atoms())
                if residue.get_ca():
                    ca_count += 1

            if ca_count == 5:
                interface_frags.append(Structure.from_residues(self.residue(frag_residue_numbers)))

        for structure in interface_frags:
            structure.chain_id_list = [structure.get_residues()[0].chain]  # Todo test if I can add attribute

        return interface_frags

    # @staticmethod
    # def index_to_mask(length, indices, index_masked=False):
    #     mask = [0 for i in range(length)]


class Chain(Structure):
    def __init__(self, chain_name=None, residues=None, coords=None):
        super().__init__(residues=residues, name=chain_name, coords=coords)
        # self.residues = residues
        # self.id = name

    # def set_id(self, _id):
    #     self.id = _id

    def __eq__(self, other):
        return self.name == other


class Residue:
    def __init__(self, atom_list):
        self.atom_list = atom_list
        self.ca = self.get_ca()
        self.cb = self.get_cb()
        self.number = self.ca.residue_number  # get_number()  # Todo test accessors, maybe make property
        self.number_pdb = self.ca.pdb_residue_number  # get_pdb_residue_number()
        self.type = self.ca.residue_type  # get_type()
        self.chain = self.ca.chain  # get_chain()
        self.secondary_structure = None

    def coords(self):
        return [atom.coords for atom in self.atom_list]

    def get_atoms(self):
        return self.atom_list

    def get_secondary_structure(self):
        return self.secondary_structure

    def set_secondary_structure(self, ss_code):
        self.secondary_structure = ss_code

    def get_ca(self):
        for atom in self.atom_list:
            if atom.is_CA():
                return atom
        return None

    def get_cb(self, include_glycine=True):  # KM added 7/25/20 to retrieve CB for atom_tree
        for atom in self.atom_list:
            if atom.is_CB(InclGlyCA=include_glycine):
                return atom
        return None

    def distance(self, other_residue):
        min_dist = float('inf')
        for self_atom in self.atom_list:
            for other_atom in other_residue.atom_list:
                d = self_atom.distance(other_atom, intra=True)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def in_contact(self, other_residue, distance_thresh=4.5, side_chain_only=False):
        if side_chain_only:
            for self_atom in self.atom_list:
                if not self_atom.is_backbone():
                    for other_atom in other_residue.atom_list:
                        if not other_atom.is_backbone():
                            if self_atom.distance(other_atom, intra=True) < distance_thresh:
                                return True
            return False
        else:
            for self_atom in self.atom_list:
                for other_atom in other_residue.atom_list:
                    if self_atom.distance(other_atom, intra=True) < distance_thresh:
                        return True
            return False

    def in_contact_residuelist(self, residuelist, distance_thresh=4.5, side_chain_only=False):
        for residue in residuelist:
            if self.in_contact(residue, distance_thresh, side_chain_only):
                return True
        return False

    @staticmethod
    def get_residue(number, chain, residue_type, residuelist):
        for residue in residuelist:
            if residue.number == number and residue.chain == chain and residue.type == residue_type:
                return residue
        # print("NO RESIDUE FOUND")
        return None

    def __key(self):
        return self.number, self.chain, self.type

    def __eq__(self, other):
        # return self.ca == other_residue.ca
        if isinstance(other, Residue):
            return self.__key() == other.__key()
        return NotImplemented

    def __str__(self):
        return_string = ""
        for atom in self.atom_list:
            return_string += str(atom)
        return return_string

    def __hash__(self):  # Todo current key is mutable so this hash is invalid
        return hash(self.__key())


class Atom:  # (Coords):
    """An Atom container with the full Structure coordinates and the Atom unique data. Pass a reference to the full
    Structure coordinates for Keyword Arg coords=self.coords"""
    def __init__(self, index=None, number=None, atom_type=None, alt_location=None, residue_type=None, chain=None,
                 residue_number=None, code_for_insertion=None, occ=None, temp_fact=None, element_symbol=None,
                 atom_charge=None, coords=None):  # x, y, z,
        self.index = index
        self.number = number
        self.type = atom_type
        self.alt_location = alt_location
        self.residue_type = residue_type
        self.chain = chain
        self.pdb_residue_number = residue_number
        self.residue_number = residue_number
        self.code_for_insertion = code_for_insertion
        # super().__init__(coords=coords)  # init from Coords class
        self.coords = coords
        # self.x = x
        # self.y = y
        # self.z = z
        # self.coordinates =
        self.occ = occ
        self.temp_fact = temp_fact
        self.element_symbol = element_symbol
        self.atom_charge = atom_charge

    @classmethod
    def from_info(cls, *args):  # number, atom_type, alt_location, residue_type, chain, residue_number, code_for_insertion, occ, temp_fact, element_symbol, atom_charge
        """Initialize without coordinates"""
        return cls(*args)

    def is_backbone(self):
        # returns True if atom is part of the proteins backbone and False otherwise
        backbone_specific_atom_type = ["N", "CA", "C", "O"]
        if self.type in backbone_specific_atom_type:
            return True
        else:
            return False

    def is_CB(self, InclGlyCA=False):
        if InclGlyCA:
            return self.type == "CB" or (self.type == "CA" and self.residue_type == "GLY")
        else:
            return self.type == "CB" or (self.type == "H" and self.residue_type == "GLY")

    def is_CA(self):
        return self.type == "CA"

    def distance(self, atom, intra=False):
        """returns distance (type float) between current instance of Atom and another instance of Atom"""
        if self.chain == atom.chain and not intra:
            print('Atoms Are In The Same Chain')
            return None
        else:
            distance = math.sqrt((self.x - atom.x)**2 + (self.y - atom.y)**2 + (self.z - atom.z)**2)
            return distance

    def distance_squared(self, atom, intra=False):
        """returns squared distance (type float) between current instance of Atom and another instance of Atom"""
        if self.chain == atom.chain and not intra:
            print('Atoms Are In The Same Chain')
            return None
        else:
            distance = (self.x - atom.x)**2 + (self.y - atom.y)**2 + (self.z - atom.z)**2
            return distance

    def translate(self, tx):
        self.coords = self.coords + tx

    @property
    def coords(self):  # , transformation_operator=None):
        """This holds the atomic coords which is a view from the Structure that created them"""
        # if transformation_operator:
        #     return np.matmul([self.x, self.y, self.z], transformation_operator)
        # else:
        return self._coords[self.index]  # [self.x, self.y, self.z]

    @coords.setter
    def coords(self, coords):
        self._coords = coords

    def get_index(self):
        return self.index

    def get_number(self):
        return self.number

    def get_type(self):
        return self.type

    def get_alt_location(self):
        return self.alt_location

    def get_residue_type(self):
        return self.residue_type

    def get_chain(self):
        return self.chain

    def get_pdb_residue_number(self):
        return self.pdb_residue_number

    def get_residue_number(self):
        return self.residue_number

    def get_code_for_insertion(self):
        return self.code_for_insertion

    @property
    def x(self):
        return self.coords[0]  # x

    @x.setter
    def x(self, x):
        self._coords[self.index][0] = x
        # self.coords[0] = x

    @property
    def y(self):
        return self.coords[1]  # y

    @y.setter
    def y(self, y):
        self._coords[self.index][1] = y
        # self.coords[1] = y

    @property
    def z(self):
        return self.coords[2]  # z

    @z.setter
    def z(self, z):
        self._coords[self.index][2] = z
        # self.coords[2] = z

    def get_occ(self):
        return self.occ

    def get_temp_fact(self):
        return self.temp_fact

    def get_element_symbol(self):
        return self.element_symbol

    def get_atom_charge(self):
        return self.atom_charge

    def __key(self):
        return self.number, self.type

    def __str__(self):
        """Represent Atom in PDB format"""
        return '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}'\
               '\n'.format('ATOM', self.number, self.type, self.alt_location, self.residue_type, self.chain,
                           self.residue_number, self.code_for_insertion, self.x, self.y, self.z, self.occ,
                           self.temp_fact, self.element_symbol, self.atom_charge)

    def __eq__(self, other):
        return (self.number == other.number and self.chain == other.chain and self.type == other.type and
                self.residue_type == other.residue_type)

    def __hash__(self):  # Todo current key is mutable so this hash is invalid
        return hash(self.__key())