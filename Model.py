import os


class Model:
    def __init__(self):
        self.model_list = []  # list of PDB objects

    def add_model(self, pdb):
        self.model_list.append(pdb)

    def write(self, name, location=os.getcwd(), cryst1=None):
        out_path = os.path.join(location, '%s.pdb' % name)
        with open(out_path, 'w') as f:
            if cryst1 and isinstance(cryst1, str) and cryst1.startswith('CRYST1'):
                f.write('%s\n' % cryst1)
            for i, model in enumerate(self.model_list, 1):
                f.write('{:9s}{:>4d}\n'.format('MODEL', i))
                for _chain in model.chain_id_list:
                    chain_atoms = model.chain(_chain)
                    f.write(''.join(str(atom) for atom in chain_atoms))
                    f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', chain_atoms[-1].number + 1,
                                                                          chain_atoms[-1].residue_type, _chain,
                                                                          chain_atoms[-1].residue_number))
                # f.write(''.join(str(atom) for atom in model.all_atoms))
                # f.write('\n'.join(str(atom) for atom in model.all_atoms))
                f.write('ENDMDL\n')

