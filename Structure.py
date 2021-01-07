from Bio.SeqUtils import IUPACData
# from Bio.Alphabet import IUPAC


class Structure:
    def __init__(self, atoms=None, residues=None, name=None):
        super().__init__()
        self.atoms = atoms
        self.residues = residues
        # self.id = None
        self.name = name
        # self.sequence = None

    def get_name(self):
        # return self.id
        return self.name

    def set_name(self, name):
        self.name = name

    def get_atoms(self):
        return self.atoms
        # atoms = []
        # for residue in self.get_residues():
        #     atoms.extend(residue.get_atoms())
        # return atoms

    def add_atoms(self, atom_list):
        """Add Atoms in atom_list to the structure instance"""
        self.atoms.extend(atom_list)

    def set_atoms(self, atom_list):
        """Set the Structure atoms to Atoms in atom_list"""
        self.atoms = atom_list

    def atom(self, atom_number):
        for atom in self.atoms:
            if atom.number == atom_number:
                return atom
        return None

    def get_residues(self):
        return self.residues

    def get_residue(self, residue_number):
        for residue in self.residues:
            if residue.number == residue_number:
                return residue
        return None

    def residues(self, residue_numbers):
        return [residue for residue in self.residues if residue.number in residue_numbers]

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

    def write(self, out_path=None, file_handle=None):
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle"""
        if file_handle:
            file_handle.write('\n'.join(str(atom) for atom in self.get_atoms()))

        if out_path:
            with open(out_path, 'w') as outfile:
                outfile.write('\n'.join(str(atom) for atom in self.get_atoms()))
