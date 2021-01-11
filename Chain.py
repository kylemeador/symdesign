from Structure import Structure


class Chain(Structure):
    def __init__(self, residues=None, name=None):
        super().__init__(residues=residues, name=name)
        # self.residues = residues
        # self.id = name

    # def set_id(self, _id):
    #     self.id = _id

    def __eq__(self, other):
        return self.id == other
