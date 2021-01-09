class Residue:
    def __init__(self, atom_list):
        self.atom_list = atom_list
        self.ca = self.get_ca()
        self.cb = self.get_cb()
        self.number = self.ca.get_number()  # Todo test accessors
        self.number_pdb = self.ca.get_pdb_residue_number()
        self.type = self.ca.get_type()
        self.chain = self.ca.get_chain()

    def get_atoms(self):
        return self.atom_list

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
