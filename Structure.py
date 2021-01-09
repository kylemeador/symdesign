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

    def write(self, out_path=None, file_handle=None):
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle"""
        if file_handle:
            file_handle.write('\n'.join(str(atom) for atom in self.get_atoms()))

        if out_path:
            with open(out_path, 'w') as outfile:
                outfile.write('\n'.join(str(atom) for atom in self.get_atoms()))
