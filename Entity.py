from SequenceProfile import SequenceProfile
from Structure import Structure


class Entity(SequenceProfile, Structure):  # Chain
    def __init__(self, representative=None, chains=None, entity_id=None, uniprot_id=None, coords=None):
        self.chains = chains  # [Chain objs]
        # self.representative_chain = representative_chain
        # use the self.structure __init__ from SequenceProfile for the structure identifier
        # Chain init
        super().__init__(name=entity_id, residues=representative.get_residues(), coords=coords, structure=self)  # representative.coords
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
    def from_representative(cls, representative=None, chains=None, entity_id=None, uniprot_id=None, coords=None):
        return cls(representative=representative, chains=chains, entity_id=entity_id,
                   uniprot_id=uniprot_id, coords=coords)  # **kwargs

    def get_representative_chain(self):
        return self.representative_chain

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
