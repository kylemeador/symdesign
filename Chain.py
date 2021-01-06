class Chain:
    def __init__(self, residues=None, name=None):
        self.residues = residues
        self.id = name

    def set_id(self, _id):
        self.id = _id

    def get_atoms(self):
        atoms = []
        for residue in self.get_residues():
            atoms.extend(residue.get_atoms())
        return atoms

    def get_residues(self):
        return self.residues

    def residue(self, residue_number):
        for residue in self.residues:
            if residue.number == residue_number:
                return residue
        return None

    def number_of_atoms(self):
        return len(self.get_atoms())

    def number_of_residues(self):
        return len(self.get_residues())

    def __eq__(self, other):
        return self.id == other
