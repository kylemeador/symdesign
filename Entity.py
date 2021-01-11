from Chain import Chain
from SequenceProfile import SequenceProfile


class Entity(Chain, SequenceProfile):
    def __init__(self, representative_chain=None, chains=None, entity_id=None, uniprot_id=None):
        self.chains = chains  # [Chain objs]
        self.representative_chain = representative_chain
        # use the self.structure __init__ from SequenceProfile for the structure identifier
        # Todo grab a unique residue list from the chain representative
        # residues = representative.get_residues()  # A stub. This is kinda the behvior I would want. fill a self.residues with the master entity residues
        super().__init__(residues=self.chain(representative_chain).get_residues(), name=entity_id)  # Chain init
        super().__init__(structure=self)  # SequenceProfile init
        # self.representative = representative  # Chain obj
        # super().__init__(structure=self.representative)  # SequenceProfile init
        # self.residues = self.chain(representative).get_residues()  # reflected above in super() call to Chain
        # self.name = entity_id  # reflected above in super() call to Chain

        # self.entity_id = entity_id
        self.uniprot_id = uniprot_id

    @classmethod
    def from_representative(cls, representative_chain=None, chains=None, entity_id=None, uniprot_id=None):
        return cls(representative_chain=representative_chain, chains=chains, entity_id=entity_id,
                   uniprot_id=uniprot_id)

    def get_representative_chain(self):
        return self.representative_chain

    def chain(self, chain_id):  # Also in PDB
        for chain in self.chains:
            if chain.id == chain_id:
                return chain

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
