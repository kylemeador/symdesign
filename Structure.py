import subprocess

from Bio.SeqUtils import IUPACData

from Residue import Residue


# from Bio.Alphabet import IUPAC


class Structure:
    def __init__(self, atoms=None, residues=None, name=None):
        super().__init__()
        self.atoms = atoms
        self.residues = residues
        # self.id = None
        self.name = name
        self.secondary_structure = None
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

    def get_structure_sequence(self):  # , aa_code=1):
        """Returns the single AA sequence of Residues found in the Structure. Handles odd residues by marking with '-'
        """
        sequence_list = [residue.type for residue in self.get_residues()]

        # if aa_code == 1:
        # self.sequence = ''.join([IUPACData.protein_letters_3to1_extended[k.title()]
        sequence = ''.join([IUPACData.protein_letters_3to1_extended[k.title()]
                            if k.title() in IUPACData.protein_letters_3to1_extended else '-'
                            for k in sequence_list])
        # else:
        #     sequence = ' '.join(sequence_list)
        return sequence

    def stride(self, chain=None):
        # REM  -------------------- Secondary structure summary -------------------  XXXX
        # REM                .         .         .         .         .               XXXX
        # SEQ  1    IVQQQNNLLRAIEAQQHLLQLTVWGIKQLQAGGWMEWDREINNYTSLIHS   50          XXXX
        # STR       HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH  HHHHHHHHHHHHHHHHH               XXXX
        # REM                                                                        XXXX
        # SEQ  51   LIEESQN                                              57          XXXX
        # STR       HHHHHH                                                           XXXX
        # REM                                                                        XXXX
        # LOC  AlphaHelix   ILE     3 A      ALA     33 A                            XXXX
        # LOC  AlphaHelix   TRP    41 A      GLN     63 A                            XXXX
        # REM                                                                        XXXX
        # REM  --------------- Detailed secondary structure assignment-------------  XXXX
        # REM                                                                        XXXX
        # REM  |---Residue---|    |--Structure--|   |-Phi-|   |-Psi-|  |-Area-|      XXXX
        # ASG  ILE A    3    1    H    AlphaHelix    360.00    -29.07     180.4      XXXX
        # ASG  VAL A    4    2    H    AlphaHelix    -64.02    -45.93      99.8      XXXX
        # ASG  GLN A    5    3    H    AlphaHelix    -61.99    -39.37      82.2      XXXX

        # try:
            # with open(os.devnull, 'w') as devnull:
        stride_cmd = [self.stride_exe_path, '%s' % self.filepath]
        #   -rId1Id2..  Read only chains Id1, Id2 ...
        #   -cId1Id2..  Process only Chains Id1, Id2 ...
        if chain:
            stride_cmd.append('-c%s' % chain_id)

        p = subprocess.Popen(stride_cmd, stderr=subprocess.DEVNULL)
        out, err = p.communicate()
        out_lines = out.decode('utf-8').split('\n')
        # except:
        #     stride_out = None

        # if stride_out is not None:
        #     lines = stride_out.split('\n')

        for line in out_lines:
            if line[0:3] == 'ASG' and line[10:15].strip().isdigit():
                self.chain(line[9:10]).residue(int(line[10:15].strip())).set_secondary_structure(line[24:25])
        self.secondary_structure = [residue.get_secondary_structure() for residue in self.get_residues()]
        # self.secondary_structure = {int(line[10:15].strip()): line[24:25] for line in out_lines
        #                             if line[0:3] == 'ASG' and line[10:15].strip().isdigit()}

    def is_n_term_helical(self, window=5):
        if len(self.secondary_structure) >= 2 * window:
            for idx, residue_number in enumerate(sorted(self.secondary_structure.keys())):
                temp_window = ''.join(self.secondary_structure[residue_number + j] for j in range(window))
                # res_number = self.secondary_structure[0 + i:5 + i][0][0]
                if 'H' * window in temp_window:
                    return True  # , res_number
                if idx == 6:
                    break
        return False  # , None

    def is_c_term_helical(self, window=5):
        if len(self.secondary_structure) >= 2 * window:
            # for i in range(5):
            for idx, residue_number in enumerate(sorted(self.secondary_structure.keys(), reverse=True)):
                # reverse_ss_asg = self.secondary_structure[::-1]
                temp_window = ''.join(self.secondary_structure[residue_number + j] for j in range(-window + 1, 1))
                # res_number = reverse_ss_asg[0+i:5+i][4][0]
                if 'H' * window in temp_window:
                    return True  # , res_number
                if idx == 6:
                    break
        return False  # , None

    def get_secondary_structure(self, chain_id=None):  # different from Josh PDB
        if not self.secondary_structure:
            self.stride(chain=chain_id)

        return self.secondary_structure

    def write(self, out_path=None, file_handle=None):
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle"""
        if file_handle:
            file_handle.write('\n'.join(str(atom) for atom in self.get_atoms()))

        if out_path:
            with open(out_path, 'w') as outfile:
                outfile.write('\n'.join(str(atom) for atom in self.get_atoms()))
