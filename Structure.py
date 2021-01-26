import math
from collections.abc import Iterable

import numpy as np
from Bio.SeqUtils import IUPACData
from numpy.linalg import eigh, LinAlgError

# from Bio.Alphabet import IUPAC
from SequenceProfile import SequenceProfile


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
        # if isinstance(coords, np.ndarray) and coords.any():
        if coords:  # isinstance(coords, Coords):
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
        """From the larger array of Coords attached to a PDB object, get the specific Coords for the subset of Atoms
        belonging to the specific Structure instance"""
        return self._coords.get_indices(self.atom_indices)

    @coords.setter
    def coords(self, coords):
        # assert len(self.atoms) == coords.shape[0], '%s: ERROR number of Atoms (%d) != number of Coords (%d)!' \
        #                                                 % (self.name, len(self.atoms), self.coords.shape[0])
        if isinstance(coords, Coords):
            self._coords = coords
            for atom in self.atoms:
                atom.coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

    @property
    def atom_indices(self):  # Todo has relevance to Residue
        try:
            return self._atom_indices
        except AttributeError:
            self.atom_indices = [atom.index for atom in self.atoms]
            return self._atom_indices

    @atom_indices.setter
    def atom_indices(self, indices):
        self._atom_indices = np.array(indices)

    def get_coords(self):
        """Return the numpy array of Coords from the Structure"""
        return self.coords

    def get_backbone_coords(self):
        """Return a view of the numpy array of Coords from the Structure with only backbone atom coordinates"""
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_backbone()]
        return self._coords[index_mask]
        # return self.coords[index_mask]

    def get_backbone_and_cb_coords(self):
        """Return a view of the numpy array of Coords from the Structure with backbone and CB atom coordinates
        inherently gets all glycine CA's"""
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_backbone() or atom.is_CB()]
        return self._coords[index_mask]
        # return self.coords[index_mask]

    def get_ca_coords(self):
        """Return a view of the numpy array of Coords from the Structure with CA atom coordinates"""
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_CA()]
        return self._coords[index_mask]
        # return self.coords[index_mask]

    def get_cb_coords(self, InclGlyCA=True):
        """Return a view of the numpy array of Coords from the Structure with CB atom coordinates"""
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_CB(InclGlyCA=InclGlyCA)]
        return self._coords[index_mask]
        # return self.coords[index_mask]

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
        if numbers and isinstance(numbers, Iterable):
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

    def get_atom_indices(self, numbers=None):
        """Retrieve Atom indices for Atoms in the Structure. Returns all by default. If atom numbers are provided
         the selected Atoms are returned"""
        # if numbers and isinstance(numbers, Iterable):
        return [atom.index for atom in self.get_atoms(numbers=numbers)]
        # else:
        #     return [atom.index for atom in self.get_atoms()]

    def get_residue_indices(self, numbers=None):
        """Retrieve Atom indices for Residues in the Structure. Returns all by default. If residue numbers are provided
         the selected Residues are returned"""
        return [atom.index for atom in self.get_residue_atoms(numbers=numbers)]

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
        """Retrieve the Atom specified by atom number"""
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
        """Retrieve Residues in Structure. Returns all by default. If a list of numbers is provided, the selected
        Residues numbers are returned"""
        if numbers and isinstance(numbers, Iterable):
            return [residue for residue in self.residues if residue.number in numbers]
        else:
            return self.residues

    def set_residues(self, residue_list):
        """Set the Structure residues to Residue objects provided in a list"""
        self.residues = residue_list  # []
        atom_list = [atom for residue in residue_list for atom in residue.get_atoms()]
        self.atoms = atom_list
        # self.update_structure(atom_list)
        self.set_length()

    def add_residues(self, residue_list):
        """Add Residue objects in a list to the Structure instance"""
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

    def get_residue_atoms(self, numbers=None):
        """Return the Atoms contained in the Residue objects matching a set of residue numbers"""
        atoms = []
        for residue in self.get_residues(numbers=numbers):
            atoms.extend(residue.get_atoms())
        return atoms
        # return [residue.get_atoms() for residue in self.get_residues(numbers=residue_numbers)]

    def residue_from_pdb_numbering(self, residue_number):
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
        try:
            return self._number_of_atoms
        except AttributeError:
            self.set_length()
            return self._number_of_atoms

    @number_of_atoms.setter
    def number_of_atoms(self, length):
        self._number_of_atoms = length

    @property
    def number_of_residues(self):
        try:
            return self._number_of_residues
        except AttributeError:
            self.set_length()
            return self._number_of_residues

    @number_of_residues.setter
    def number_of_residues(self, length):
        self._number_of_residues = length

    def get_center_of_mass(self):
        return self.center_of_mass

    def find_center_of_mass(self):
        """Retrieve the center of mass for the specified Structure"""
        divisor = 1 / self.number_of_atoms
        self.center_of_mass = np.matmul(np.full(self.number_of_atoms, divisor), self.coords)

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

    def write(self, out_path=None, header=None, file_handle=None):
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle"""
        def write_header(location):
            if header and isinstance(header, Iterable):
                if isinstance(header, str):
                    location.write(header)
                else:
                    location.write('\n'.join(line for line in header))

        if file_handle:
            write_header(file_handle)
            file_handle.write('\n'.join(str(atom) for atom in self.get_atoms()))

        if out_path:
            with open(out_path, 'w') as outfile:
                write_header(outfile)
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


class Entity(SequenceProfile, Structure):  # Chain
    """Entity
    Initialize with Keyword Args:
        representative=None (Chain): The Chain that should represent the Entity
        chains=None (list): A list of all Chain objects that match the Entity
        name=None (str): The name for the Entity. Typically PDB.name is used to make PDB compatible form
        PDB EntryID_EntityID
        uniprot_id=None (str): The unique UniProtID for the Entity
    """
    def __init__(self, representative=None, chains=None, name=None, uniprot_id=None, coords=None):
        self.chains = chains  # [Chain objs]
        # self.representative_chain = representative_chain
        # use the self.structure __init__ from SequenceProfile for the structure identifier
        # Chain init
        super().__init__(name=name, residues=representative.get_residues(), coords=coords, structure=self)
        # super().__init__(chain_name=representative_chain.name, residues=representative_chain.get_residues(),
        #                  coords=representative_chain.coords)
        # super().__init__(chain_name=entity_id, residues=self.chain(representative_chain).get_residues(), coords=coords)
        # SequenceProfile init
        # super().__init__(structure=self)
        # self.representative = representative  # Chain obj
        # super().__init__(structure=self.representative)  # SequenceProfile init
        # self.residues = self.chain(representative).get_residues()  # reflected above in super() call to Chain
        # self.name = entity_id  # reflected above in super() call to Chain

        # self.entity_id = entity_id
        self.uniprot_id = uniprot_id

    @classmethod
    def from_representative(cls, representative=None, chains=None, name=None, uniprot_id=None, coords=None):
        return cls(representative=representative, chains=chains, name=name,
                   uniprot_id=uniprot_id, coords=coords)  # **kwargs

    # def get_representative_chain(self):
    #     return self.representative

    def chain(self, chain_id):  # Also in PDB
        for chain in self.chains:
            if chain.name == chain_id:
                return chain
        return None

    # Todo set up captain chain and mate chains dependence

    # FROM CHAIN super().__init__()
    #
    # def set_id(self, _id):
    #     self.id = _id
    #
    # def get_atoms(self):
    #     atoms = []
    #     for residue in self.get_residues():
    #         atoms.extend(residue.get_atoms())
    #     return atoms
    #
    # def get_residues(self):
    #     return self.residues
    #
    # def residue(self, residue_number):
    #     for residue in self.residues:
    #         if residue.number == residue_number:
    #             return residue


class Chain(Structure):
    def __init__(self, name=None, residues=None, coords=None):
        super().__init__(residues=residues, name=name, coords=coords)
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

    @property
    def number_of_atoms(self):
        try:
            return self._number_of_atoms
        except AttributeError:
            self.number_of_atoms = len(self.get_atoms())
            return self._number_of_atoms

    @number_of_atoms.setter
    def number_of_atoms(self, length):
        self._number_of_atoms = length

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


class Coords:
    def __init__(self, coords=None):
        if coords is not None:
            self.coords = coords
        else:
            self.coords = []
        # self.indices = None

    @property
    def coords(self):  # , transformation_operator=None):
        """This holds the atomic coords which is a view from the Structure that created them"""
        # if transformation_operator:
        #     return np.matmul([self.x, self.y, self.z], transformation_operator)
        # else:
        return self._coords  # [self.indices]  # [self.x, self.y, self.z]

    @coords.setter
    def coords(self, coords):
        self._coords = np.array(coords)

    def get_indices(self, indicies=None):
        if indicies.any():
            return self._coords[indicies]
        else:
            return self.coords

    def __len__(self):
        return self.coords.shape[0]

    # @property
    # def x(self):
    #     return self.coords[0]  # x
    #
    # @x.setter
    # def x(self, x):
    #     self.coords[0] = x
    #
    # @property
    # def y(self):
    #     return self.coords[1]  # y
    #
    # @y.setter
    # def y(self, y):
    #     self.coords[1] = y
    #
    # @property
    # def z(self):
    #     return self.coords[2]  # z
    #
    # @z.setter
    # def z(self, z):
    #     self.coords[2] = z


def superposition3d(aa_xf_orig, aa_xm_orig, a_weights=None, allow_rescale=False, report_quaternion=False):
    """
    MIT License. Copyright (c) 2016, Andrew Jewett
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
    Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
    OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Superpose3D() takes two lists of xyz coordinates (same length), and attempts to superimpose them using rotations,
     translations, and (optionally) rescale operations in order to minimize the root-mean-squared-distance (RMSD)
     between them. These operations should be applied to the "aa_xf_orig" argument.

    This function implements a more general variant of the method from:
    R. Diamond, (1988) "A Note on the Rotational Superposition Problem", Acta Cryst. A44, pp. 211-216
    This version has been augmented slightly. The version in the original paper only considers rotation and translation
    and does not allow the coordinates of either object to be rescaled (multiplication by a scalar).
    (Additional documentation can be found at https://pypi.org/project/superpose3d/ )

    Args:
        aa_xf_orig (numpy.array): The coordinates for the "frozen" object
        aa_xm_orig (numpy.array): The coordinates for the "mobile" object
    Keyword Args:
        aWeights=None (numpy.array): The optional weights for the calculation of RMSD
        allow_rescale=False (bool): Attempt to rescale the mobile point cloud in addition to translation/rotation?
        report_quaternion=False (bool): Whether to report the rotation angle and axis in typical quaternion fashion
    Returns:
        (float, numpy.array, numpy.array, float): Corresponding to the rmsd, optimal rotation_matrix or
        quaternion_matrix (if report_quaternion=True), optimal_translation_vector, and optimal_scale_factor.
        The quaternion_matrix has the first row storing cos(θ/2) (where θ is the rotation angle). The following 3 rows
        form a vector (of length sin(θ/2)), pointing along the axis of rotation.
        Details here: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    # convert input lists as to numpy arrays

    aa_xf_orig = np.array(aa_xf_orig)
    aa_xm_orig = np.array(aa_xm_orig)

    if aa_xf_orig.shape[0] != aa_xm_orig.shape[0]:
        raise ValueError("Inputs should have the same size.")

    number_of_points = aa_xf_orig.shape[0]
    # Find the center of mass of each object:
    """ # old code (using for-loops)
    if (aWeights == None) or (len(aWeights) == 0):
        aWeights = np.full(number_of_points, 1.0)
    a_center_f = np.zeros(3)
    a_center_m = np.zeros(3)
    sum_weights = 0.0
    for n in range(0, number_of_points):
        for d in range(0, 3):
            a_center_f[d] += aaXf_orig[n][d]*aWeights[n]
            a_center_m[d] += aaXm_orig[n][d]*aWeights[n]
        sum_weights += aWeights[n]
    """
    # new code (avoiding for-loops)
    # convert weights into array
    if not a_weights or (len(a_weights) == 0):
        a_weights = np.full((number_of_points, 1), 1.0)
    else:
        # reshape aWeights so multiplications are done column-wise
        a_weights = np.array(a_weights).reshape(number_of_points, 1)

    a_center_f = np.sum(aa_xf_orig * a_weights, axis=0)
    a_center_m = np.sum(aa_xm_orig * a_weights, axis=0)
    sum_weights = np.sum(a_weights, axis=0)

    # Subtract the centers-of-mass from the original coordinates for each object
    """ # old code (using for-loops)
    if sum_weights != 0:
        for d in range(0, 3):
            a_center_f[d] /= sum_weights
            a_center_m[d] /= sum_weights
    for n in range(0, number_of_points):
        for d in range(0, 3):
            aa_xf[n][d] = aaXf_orig[n][d] - a_center_f[d]
            aa_xm[n][d] = aaXm_orig[n][d] - a_center_m[d]
    """
    # new code (avoiding for-loops)
    if sum_weights != 0:
        a_center_f /= sum_weights
        a_center_m /= sum_weights
    aa_xf = aa_xf_orig - a_center_f
    aa_xm = aa_xm_orig - a_center_m

    # Calculate the "M" array from the Diamond paper (equation 16)
    """ # old code (using for-loops)
    M = np.zeros((3,3))
    for n in range(0, number_of_points):
        for i in range(0, 3):
            for j in range(0, 3):
                M[i][j] += aWeights[n] * aa_xm[n][i] * aa_xf[n][j]
    """
    M = np.matmul(aa_xm.T, (aa_xf * a_weights))

    # Calculate Q (equation 17)

    """ # old code (using for-loops)
    traceM = 0.0
    for i in range(0, 3):
        traceM += M[i][i]
    Q = np.empty((3,3))
    for i in range(0, 3):
        for j in range(0, 3):
            Q[i][j] = M[i][j] + M[j][i]
            if i==j:
                Q[i][j] -= 2.0 * traceM
    """
    Q = M + M.T - 2 * np.eye(3) * np.trace(M)

    # Calculate v (equation 18)
    v = np.empty(3)
    v[0] = M[1][2] - M[2][1]
    v[1] = M[2][0] - M[0][2]
    v[2] = M[0][1] - M[1][0]

    # Calculate "P" (equation 22)
    """ # old code (using for-loops)
    P = np.empty((4,4))
    for i in range(0,3):
        for j in range(0,3):
            P[i][j] = Q[i][j]
    P[0][3] = v[0]
    P[3][0] = v[0]
    P[1][3] = v[1]
    P[3][1] = v[1]
    P[2][3] = v[2]
    P[3][2] = v[2]
    P[3][3] = 0.0
    """
    P = np.zeros((4, 4))
    P[:3, :3] = Q
    P[3, :3] = v
    P[:3, 3] = v

    # Calculate "p".
    # "p" contains the optimal rotation (in backwards-quaternion format)
    # (Note: A discussion of various quaternion conventions is included below.)
    # First, specify the default value for p:
    p = np.zeros(4)
    p[3] = 1.0           # p = [0,0,0,1]    default value
    pPp = 0.0            # = p^T * P * p    (zero by default)
    singular = (number_of_points < 2)   # (it doesn't make sense to rotate a single point)

    try:
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html
        a_eigenvals, aa_eigenvects = eigh(P)
    except LinAlgError:
        singular = True  # (I have never seen this happen.)

    if not singular:  # (don't crash if the caller supplies nonsensical input)
        """ # old code (using for-loops)
        eval_max = a_eigenvals[0]
        i_eval_max = 0
        for i in range(1, 4):
            if a_eigenvals[i] > eval_max:
                eval_max = a_eigenvals[i]
                i_eval_max = i
        p[0] = aa_eigenvects[0][i_eval_max]
        p[1] = aa_eigenvects[1][i_eval_max]
        p[2] = aa_eigenvects[2][i_eval_max]
        p[3] = aa_eigenvects[3][i_eval_max]
        pPp = eval_max
        """
        # new code (avoiding for-loops)
        i_eval_max = np.argmax(a_eigenvals)
        pPp = np.max(a_eigenvals)
        p[:] = aa_eigenvects[:, i_eval_max]

    # normalize the vector
    # (It should be normalized already, but just in case it is not, do it again)
    p /= np.linalg.norm(p)

    # Finally, calculate the rotation matrix corresponding to "p"
    # (p is in backwards-quaternion format)

    aa_rotate = np.empty((3, 3))
    aa_rotate[0][0] = (p[0]*p[0])-(p[1]*p[1])-(p[2]*p[2])+(p[3]*p[3])
    aa_rotate[1][1] = -(p[0]*p[0])+(p[1]*p[1])-(p[2]*p[2])+(p[3]*p[3])
    aa_rotate[2][2] = -(p[0]*p[0])-(p[1]*p[1])+(p[2]*p[2])+(p[3]*p[3])
    aa_rotate[0][1] = 2*(p[0]*p[1] - p[2]*p[3])
    aa_rotate[1][0] = 2*(p[0]*p[1] + p[2]*p[3])
    aa_rotate[1][2] = 2*(p[1]*p[2] - p[0]*p[3])
    aa_rotate[2][1] = 2*(p[1]*p[2] + p[0]*p[3])
    aa_rotate[0][2] = 2*(p[0]*p[2] + p[1]*p[3])
    aa_rotate[2][0] = 2*(p[0]*p[2] - p[1]*p[3])

    # Alternatively, in modern python versions, this code also works:
    """
    from scipy.spatial.transform import Rotation as R
    the_rotation = R.from_quat(p)
    aa_rotate = the_rotation.as_matrix()
    """

    # Optional: Decide the scale factor, c
    c = 1.0   # by default, don't rescale the coordinates
    if allow_rescale and (not singular):
        """ # old code (using for-loops)
        Waxaixai = 0.0
        WaxaiXai = 0.0
        for a in range(0, number_of_points):
            for i in range(0, 3):
                Waxaixai += aWeights[a] * aa_xm[a][i] * aa_xm[a][i]
                WaxaiXai += aWeights[a] * aa_xm[a][i] * aa_xf[a][i]
        """
        # new code (avoiding for-loops)
        Waxaixai = np.sum(a_weights * aa_xm ** 2)
        WaxaiXai = np.sum(a_weights * aa_xf ** 2)

        c = (WaxaiXai + pPp) / Waxaixai

    # Finally compute the RMSD between the two coordinate sets:
    # First compute E0 from equation 24 of the paper

    """ # old code (using for-loops)
    E0 = 0.0
    for n in range(0, number_of_points):
        for d in range(0, 3):
            # (remember to include the scale factor "c" that we inserted)
            E0 += aWeights[n] * ((aa_xf[n][d] - c*aa_xm[n][d])**2)
    sum_sqr_dist = E0 - c*2.0*pPp
    if sum_sqr_dist < 0.0: #(edge case due to rounding error)
        sum_sqr_dist = 0.0
    """
    # new code (avoiding for-loops)
    E0 = np.sum((aa_xf - c * aa_xm) ** 2)
    sum_sqr_dist = max(0, E0 - c * 2.0 * pPp)

    rmsd = 0.0
    if sum_weights != 0.0:
        rmsd = np.sqrt(sum_sqr_dist/sum_weights)

    # Lastly, calculate the translational offset:
    # Recall that:
    #RMSD=sqrt((Σ_i  w_i * |X_i - (Σ_j c*R_ij*x_j + T_i))|^2) / (Σ_j w_j))
    #    =sqrt((Σ_i  w_i * |X_i - x_i'|^2) / (Σ_j w_j))
    #  where
    # x_i' = Σ_j c*R_ij*x_j + T_i
    #      = Xcm_i + c*R_ij*(x_j - xcm_j)
    #  and Xcm and xcm = center_of_mass for the frozen and mobile point clouds
    #                  = a_center_f[]       and       a_center_m[],  respectively
    # Hence:
    #  T_i = Xcm_i - Σ_j c*R_ij*xcm_j  =  a_translate[i]

    """ # old code (using for-loops)
    a_translate = np.empty(3)
    for i in range(0,3):
        a_translate[i] = a_center_f[i]
        for j in range(0,3):
            a_translate[i] -= c*aa_rotate[i][j]*a_center_m[j]
    """
    # new code (avoiding for-loops)
    a_translate = a_center_f - np.matmul(c * aa_rotate, a_center_m).T.reshape(3,)

    if report_quaternion:  # does the caller want the quaternion?
        # The p array is a quaternion that uses this convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        # However it seems that the following convention is much more popular:
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        # https://mathworld.wolfram.com/Quaternion.html
        # So I return "q" (a version of "p" using the more popular convention).
        q = np.empty(4)
        q[0] = p[3]
        q[1] = p[0]
        q[2] = p[1]
        q[3] = p[2]
        return rmsd, q, a_translate, c
    else:
        return rmsd, aa_rotate, a_translate, c