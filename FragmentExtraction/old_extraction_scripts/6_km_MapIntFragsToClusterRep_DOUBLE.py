import sys
import os
import Bio.PDB
import copy

# Globals
module = 'Map Paired Fragments to IJ Cluster and Add Guide Atoms:'


def add_guide_atoms(biopdb_structure):
    # Create Guide Atoms
    _a1 = Bio.PDB.Atom
    a1 = _a1.Atom("CA", (0.0, 0.0, 0.0), 20.00, 1.0, " ", " CA ", 1, element="C")
    _a2 = Bio.PDB.Atom
    a2 = _a2.Atom("N", (3.0, 0.0, 0.0), 20.00, 1.0, " ", " N  ", 2, element="N")
    _a3 = Bio.PDB.Atom
    a3 = _a3.Atom("O", (0.0, 3.0, 0.0), 20.00, 1.0, " ", " O  ", 3, element="O")

    # Create Residue for Guide Atoms
    _r = Bio.PDB.Residue
    r = _r.Residue((' ', 0, ' '), "GLY", "    ")

    # Create Chain for Guide Atoms
    _c = Bio.PDB.Chain
    c = _c.Chain("9")

    # Add Guide Atoms to Residue
    r.add(a1)
    r.add(a2)
    r.add(a3)

    # Add Residue to Chain
    c.add(r)

    # Add Chain to BioPDB Structure
    biopdb_structure[0].add(c)


def get_biopdb_ca(biopdb_structure):
    ca_atoms = []
    for atom in biopdb_structure.get_atoms():
        if atom.get_id() == 'CA':
            ca_atoms.append(atom)

    return ca_atoms


def main():
    print(module, 'Beginning')
    # Match RMSD Threshold
    clust_rmsd_thresh = 0.75
    number_fragment_types = 5

    # Centered Split Fragment Cluster Representatives (1 Chain) Directory - "Top5MonoFragClustersRepresentativeCentered"
    cent_i_frag = os.path.join(os.getcwd(), 'i_clusters')

    # Paired Fragment Directory
    paired_frag_dir = os.path.join(os.getcwd(), 'all_paired_frags')

    # Out Directory - Containing Interface Fragments Where one of the Chains is Mapped Onto the Centered Coordinates
    # of the Corresponding Fragment Type Cluster Representative - "MappedIntFragsWithGuideAtoms"
    outdir = os.path.join(os.getcwd(), 'ij_mapped_paired_frags')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Read In The Centered Cluster Representative Fragments and Create Bio.PDB Structure Objects
    # And Create Cluster Representative Directories in outdir
    cent_split_frag_rep_biopdb = []
    i_cluster_dict = {}
    while len(cent_split_frag_rep_biopdb) <= number_fragment_types:
        for root1, dirs1, files1 in os.walk(cent_i_frag):
            if not dirs1:
                for file1 in files1:
                    if file1.endswith('_representative.pdb'):
                        parser = Bio.PDB.PDBParser()
                        cent_i_frag_biopdb = parser.get_structure(os.path.splitext(file1)[0], os.path.join(cent_i_frag, file1))
                        cent_split_frag_rep_biopdb.append(cent_i_frag_biopdb)
                        i_cluster_dict[os.path.splitext(file1)[0]] = os.path.basename(root1)

                        if not os.path.exists(os.path.join(outdir, os.path.basename(root1))):  # i_cluster_dict[file1])):
                            os.makedirs(os.path.join(outdir, os.path.basename(root1)))
    print(module, 'Finished Setting Up I Directories, Fetched I Fragments')

    # Get Paths for Interface Fragment PDB files
    paired_frag_paths = []
    for root2, dirs2, files2 in os.walk(paired_frag_dir):
        for file2 in files2:
            if file2.endswith('.pdb'):
                paired_frag_paths.append(os.path.join(paired_frag_dir, file2))

    print(module, 'Mapping Pairs to I Fragments, Setting Up Guide Atoms, and Making J Directories. '
                  'This may take some time...')
    for int_frag_path in paired_frag_paths:
        try:
            parser = Bio.PDB.PDBParser()
            frag_biopdb = parser.get_structure(os.path.splitext(os.path.basename(int_frag_path))[0], int_frag_path)
            int_frag_chains_info = []
            for chain in frag_biopdb[0]:
                min_rms = sys.maxint
                min_rms_biopdb = None
                min_sup = None

                # Getting CA Atoms From One Interface Fragment Chain
                chain_frag_ca = get_biopdb_ca(chain)
                # int_frag_mono_frag_ca_atoms = []
                # for atom in chain.get_atoms():
                #     if atom.get_id() == "CA":
                #         int_frag_mono_frag_ca_atoms.append(atom)

                # Find the Cluster/Fragment Type that the Interface Fragment Chain Belongs to if any
                for cent_i_frag_biopdb in cent_split_frag_rep_biopdb:
                    # Getting CA Atoms From Centered Cluster Representative Fragment
                    cent_I_frag_rep_ca = get_biopdb_ca(cent_i_frag_biopdb)
                    # cent_split_frag_rep_ca_atoms = []
                    # for atom in cent_i_frag_biopdb.get_atoms():
                    #     if atom.get_id() == "CA":
                    #         cent_split_frag_rep_ca_atoms.append(atom)

                    # Comparing Interface Fragment Chain to Cluster Representatives
                    sup = Bio.PDB.Superimposer()
                    sup.set_atoms(cent_I_frag_rep_ca, chain_frag_ca)

                    if sup.rms < min_rms:
                        min_rms = sup.rms
                        min_rms_biopdb = cent_i_frag_biopdb
                        min_sup = sup

                if min_rms_biopdb is not None and min_rms <= clust_rmsd_thresh:
                    # Getting Cluster Representative Fragment Type
                    frag_type = i_cluster_dict[min_rms_biopdb.id]

                    # Aligning Interface Fragment Chain to Cluster Representative
                    int_frag_biopdb_copy = copy.deepcopy(frag_biopdb)
                    min_sup.apply(int_frag_biopdb_copy)
                    # TODO get rid of min_rms_biopdb.id. Will require shifting the index down for everything below
                    int_frag_chains_info.append((int_frag_biopdb_copy, min_rms_biopdb.id, frag_biopdb.id, chain.id, frag_type, copy.deepcopy(min_rms_biopdb)))
                else:
                    break

            if len(int_frag_chains_info) == 2:
                # There is an intrinsic error here. Overcounting occurs because saving of both chains happens for each
                # fragment. Each fragment has a copy with mapped/pair and the reverse

                subtype_dir_1 = os.path.join(outdir, int_frag_chains_info[0][4], int_frag_chains_info[0][4] + '_' + int_frag_chains_info[1][4])
                if not os.path.exists(subtype_dir_1):
                    os.makedirs(subtype_dir_1)

                sup_1 = Bio.PDB.Superimposer()
                # superimpose CA atoms from fragment, second chain_id, with second fragment min_rms_biopdb
                sup_1.set_atoms(get_biopdb_ca(int_frag_chains_info[0][0][0][int_frag_chains_info[1][3]]), get_biopdb_ca(int_frag_chains_info[1][5]))
                add_guide_atoms(int_frag_chains_info[1][5])
                sup_1.apply(int_frag_chains_info[1][5])
                guide_moved_atom_chain_1 = int_frag_chains_info[1][5][0]['9']
                int_frag_chains_info[0][0][0].add(guide_moved_atom_chain_1)

                io_1 = Bio.PDB.PDBIO()
                io_1.set_structure(int_frag_chains_info[0][0])
                io_1_path = os.path.join(subtype_dir_1, int_frag_chains_info[0][2] + '_mappedchain_' +
                                         int_frag_chains_info[0][3] + '_fragtype_' + int_frag_chains_info[0][4] +
                                         '_partnerchain_' + int_frag_chains_info[1][3] + '_fragtype_' +
                                         int_frag_chains_info[1][4] + '.pdb')
                io_1.save(io_1_path)

                subtype_dir_2 = os.path.join(outdir, int_frag_chains_info[1][4], int_frag_chains_info[1][4] + '_' + int_frag_chains_info[0][4])
                if not os.path.exists(subtype_dir_2):
                    os.makedirs(subtype_dir_2)

                sup_2 = Bio.PDB.Superimposer()
                sup_2.set_atoms(get_biopdb_ca(int_frag_chains_info[1][0][0][int_frag_chains_info[0][3]]), get_biopdb_ca(int_frag_chains_info[0][5]))
                add_guide_atoms(int_frag_chains_info[0][5])
                sup_2.apply(int_frag_chains_info[0][5])
                guide_moved_atom_chain_2 = int_frag_chains_info[0][5][0]['9']
                int_frag_chains_info[1][0][0].add(guide_moved_atom_chain_2)

                io_2 = Bio.PDB.PDBIO()
                io_2.set_structure(int_frag_chains_info[1][0])
                io_2_path = os.path.join(subtype_dir_2, int_frag_chains_info[1][2] + '_mappedchain_' +
                                         int_frag_chains_info[1][3] + '_fragtype_' + int_frag_chains_info[1][4] +
                                         '_partnerchain_' + int_frag_chains_info[0][3] + '_fragtype_' +
                                         int_frag_chains_info[0][4] + '.pdb')
                io_2.save(io_2_path)

        except Exception as ex:
            print('\n')
            print('Error mapping', os.path.splitext(os.path.basename(int_frag_path))[0], 'to a cluster representative')
            print('Exception: ' + str(ex))
            print('\n')

    print(module, 'Finished')


if __name__ == '__main__':
    main()
