import math

import numpy as np
from Bio.SeqUtils import IUPACData


# from Bio.Alphabet import IUPAC


class Structure:
    def __init__(self, atoms=None, residues=None, name=None):
        super().__init__()
        self.atoms = atoms
        self.residues = residues
        # self.id = None
        self.name = name
        self.secondary_structure = None
        self.center_of_mass = None
        # self.sequence = None

    def get_name(self):
        # return self.id
        return self.name

    def set_name(self, name):
        self.name = name

    def get_atoms(self, numbers=None):
        """Retrieve Atoms in structure. Returns all by default. If numbers=(list) selected Atom numbers are returned"""
        if numbers:
            return [atom for atom in self.atoms if atom.number in numbers]
        else:
            return self.atoms

    def get_CA_atoms(self):
        return [atom for atom in self.get_atoms() if atom.is_CA()]

    def add_atoms(self, atom_list):
        """Add Atoms in atom_list to the structure instance"""
        self.atoms.extend(atom_list)
        self.create_residues()

    def set_atoms(self, atom_list):
        """Set the Structure atoms to Atoms in atom_list"""
        self.atoms = atom_list
        self.create_residues()

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

    def get_residues(self, numbers=None):
        """Retrieve Residues in structure. Returns all by default. If numbers=(list) selected Residues numbers are
        returned"""
        if numbers:
            return [residue for residue in self.residues if residue.number in numbers]
        else:
            return self.residues

    def add_residues(self, residue_list):
        """Add Residues in residue_list to the structure instance"""
        self.residues.extend(residue_list)
        self.atoms.extend(atom for atom in residue_list.get_atoms())

    def set_residues(self, residue_list):
        """Set the Structure residues to Residues in residue_list"""
        self.residues = residue_list
        self.atoms = [atom for atom in residue_list.get_atoms()]

    def create_residues(self):
        """For the structure, create all possible Residue instances"""
        self.residues = []
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
        self.renumber_atoms()  # should be unnecessary

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

    def number_of_atoms(self):
        return len(self.get_atoms())

    def number_of_residues(self):
        return len(self.get_residues())

    def get_bb_indices(self):
        return [idx for idx, atom in enumerate(self.get_atoms()) if atom.is_backbone()]

    def get_cb_indices(self, InclGlyCA=False):
        return [idx for idx, atom in enumerate(self.get_atoms()) if atom.is_CB(InclGlyCA=InclGlyCA)]

    def get_helix_cb_indices(self):
        """Only works for monomers or homo-complexes on secondary structure assigned structures!"""
        h_cb_indices = []
        for idx, residue in enumerate(self.get_residues()):
            if not residue.get_secondary_structure():
                print('Secondary Structures must be set before finding helical CB\'s! Error at Residue %s' % residue.number)
                return None
            elif residue.get_secondary_structure() == 'H':
                h_cb_indices.append(residue.get_cb())

        return h_cb_indices

    def get_center_of_mass(self):
        return self.center_of_mass

    def find_center_of_mass(self, coords):  # Todo, make reference self.coords
        """Given a numpy array of 3D coordinates, return the center of mass"""
        divisor = 1 / len(coords)
        return np.matmul(np.array.full((1, 3), divisor), coords)

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
    # def is_n_term_helical(self, window=5):
    #     if len(self.secondary_structure) >= 2 * window:
    #         for idx, residue_number in enumerate(sorted(self.secondary_structure.keys())):
    #             temp_window = ''.join(self.secondary_structure[residue_number + j] for j in range(window))
    #             # res_number = self.secondary_structure[0 + i:5 + i][0][0]
    #             if 'H' * window in temp_window:
    #                 return True  # , res_number
    #             if idx == 6:
    #                 break
    #     return False  # , None
    #
    # def is_c_term_helical(self, window=5):
    #     if len(self.secondary_structure) >= 2 * window:
    #         # for i in range(5):
    #         for idx, residue_number in enumerate(sorted(self.secondary_structure.keys(), reverse=True)):
    #             # reverse_ss_asg = self.secondary_structure[::-1]
    #             temp_window = ''.join(self.secondary_structure[residue_number + j] for j in range(-window + 1, 1))
    #             # res_number = reverse_ss_asg[0+i:5+i][4][0]
    #             if 'H' * window in temp_window:
    #                 return True  # , res_number
    #             if idx == 6:
    #                 break
    #     return False  # , None
    #
    # def get_secondary_structure(self, chain_id=None):  # different from Josh PDB
    #     if not self.secondary_structure:
    #         self.stride(chain=chain_id)
    #
    #     return self.secondary_structure

    def write(self, out_path=None, file_handle=None):
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle"""
        if file_handle:
            file_handle.write('\n'.join(str(atom) for atom in self.get_atoms()))

        if out_path:
            with open(out_path, 'w') as outfile:
                outfile.write('\n'.join(str(atom) for atom in self.get_atoms()))


class Chain(Structure):
    def __init__(self, residues=None, name=None):
        super().__init__(residues=residues, name=name)
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
        self.number = self.ca.get_number()  # Todo test accessors
        self.number_pdb = self.ca.get_pdb_residue_number()
        self.type = self.ca.get_type()
        self.chain = self.ca.get_chain()
        self.secondary_structure = None

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
        else:
            # print('RESIDUE OBJECT REQUIRES CA ATOM. No CA found in: %s\nSelecting CB instead' % str(self.atom_list[0]))
            for atom in self.atom_list:
                if atom.is_CB():
                    return atom
            # print('RESIDUE OBJECT MISSING CB ATOM. Severely flawed residue, fix your PDB input!')
            return None

    def get_cb(self):  # KM added 7/25/20 to retrieve CB for atom_tree
        for atom in self.atom_list:
            if atom.is_CB():
                return atom
        else:
            # print('No CB found in: %s\nSelecting CB instead' % str(self.atom_list[0]))
            for atom in self.atom_list:
                if atom.is_CA():
                    return atom
            # print('RESIDUE OBJECT MISSING CB ATOM. Severely flawed residue, fix your PDB input!')
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
        #print "NO RESIDUE FOUND"
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


class Atom:
    def __init__(self, number, atom_type, alt_location, residue_type, chain, residue_number, code_for_insertion, x, y,
                 z, occ, temp_fact, element_symbol, atom_charge):
        self.number = number
        self.type = atom_type
        self.alt_location = alt_location
        self.residue_type = residue_type
        self.chain = chain
        self.pdb_residue_number = residue_number
        self.residue_number = residue_number
        self.code_for_insertion = code_for_insertion
        self.x = x
        self.y = y
        self.z = z
        self.occ = occ
        self.temp_fact = temp_fact
        self.element_symbol = element_symbol
        self.atom_charge = atom_charge

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
        # returns distance (type float) between current instance of Atom and another instance of Atom
        if self.chain == atom.chain and not intra:
            print('Atoms Are In The Same Chain')
            return None
        else:
            distance = math.sqrt((self.x - atom.x)**2 + (self.y - atom.y)**2 + (self.z - atom.z)**2)
            return distance

    def distance_squared(self, atom, intra=False):
        # returns squared distance (type float) between current instance of Atom and another instance of Atom
        if self.chain == atom.chain and not intra:
            print('Atoms Are In The Same Chain')
            return None
        else:
            distance = (self.x - atom.x)**2 + (self.y - atom.y)**2 + (self.z - atom.z)**2
            return distance

    def translate3d(self, tx):
        coord = [self.x, self.y, self.z]
        translated = []
        for i in range(3):
            coord[i] += tx[i]
            translated.append(coord[i])
        self.x, self.y, self.z = translated

    def coords(self):
        return [self.x, self.y, self.z]

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

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

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