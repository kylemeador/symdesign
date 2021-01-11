from Chain import Chain
from SequenceProfile import SequenceProfile


class Entity(Chain, SequenceProfile):
    def __init__(self, chains=None, representative=None, entity_id=None, uniprot_id=None):
        self.chains = chains  # [Chain objs]
        self.representative = representative  # Chain obj
        # Todo grab a unique residue list from the chain representative
        residues = representative.get_residues()  # A stub. This is kinda the behvior I would want. fill a self.residues with the master entity residues
        super().__init__(residues=residues, name=entity_id)  # Chain init
        super().__init__(structure=self.representative)  # SequenceProfile init
        # self.residues = representative.get_residues()  # reflected above in super() call to Chain
        # self.id = entity_id  # reflected above in super() call to Chain

        # self.entity_id = entity_id
        self.uniprot_id = uniprot_id

    def get_chain_name(self):
        return self.representative.get_name()


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
