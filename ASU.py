import os
from csv import reader
import argparse
import SymDesignUtils as SDUtils
import PathUtils as PUtils
import AnalyzeMutatedSequences as Ams


def make_asu(files, chain, destination=os.getcwd):
    return [SDUtils.extract_asu(file, chain=chain, outpath=destination) for file in files]


def design_recapitulation(design_file, pdb_dir, output_dir):
    qsbio_assemblies = SDUtils.unpickle(PUtils.qsbio)
    with open(design_file, 'r') as f:
        reading_csv = reader(f)
        design_file_input = {os.path.splitext(row[0])[0]:
                             {'design_pdb': SDUtils.read_pdb(os.path.join(pdb_dir, row[0])),
                              'source_pdb': {(row[1], row[3]), (row[2], row[4])}, 'final_sym': row[5]}
                             for row in reading_csv}  # 'pdb1': 'sym1': 'pdb2': 'sym2':
    # all_pdbs = {file: {'design': SDUtils.read_pdb(pdb_file) for pdb_file in all_pdb_files} for file in design_input
    # all_pdb_files = SDUtils.get_all_pdb_file_paths(pdb_dir)
    if not os.path.exists(os.path.join(output_dir, 'design_asus')):
        os.makedirs(os.path.join(output_dir, 'design_asus'))
    if not os.path.exists(os.path.join(output_dir, 'biological_assemblies')):
        os.makedirs(os.path.join(output_dir, 'biological_assemblies'))

    chain_correspondence = {design: {} for design in design_file_input}  # capture all the design and chain info
    # Final format:
    # {design: {'pdb1': {'asu_chain': None, 'dock_chains': []},
    #           'pdb2': {'asu_chain': None, 'dock_chains': []}}, ...}
    for design in design_file_input:
        asu = design_file_input[design]['design_pdb'].return_asu()
        asu.reorder_chains()
        # asu.pose_numbering()
        for chain in asu.chain_id_list:
            asu.reindex_chain_residues(chain)
        asu.get_all_entities()

        if not os.path.exists(os.path.join(output_dir, design)):
            os.makedirs(os.path.join(output_dir, design))

        used_chains = []
        for i, (pdb, sym) in enumerate(design_file_input[design]['source_pdb'], 1):
            if pdb in qsbio_assemblies:
                biological_assembly = qsbio_assemblies[pdb][0]  # find a verified biological assembly from QSBio
            else:
                print('%s: %s is not verified in QSBio! Arbitrarily choosing assembly 1' % (design, pdb))
                biological_assembly = 1
            new_file = SDUtils.download_pdb('%s_%s' % (pdb, biological_assembly),
                                            location=os.path.join(output_dir, 'biological_assemblies'))
            downloaded_pdb = SDUtils.read_pdb(new_file)
            oriented_pdb = downloaded_pdb.orient(sym, PUtils.orient_dir)  # , generate_oriented_pdb=False)
            # oriented_pdb.name = pdb.lower()
            # oriented_pdb.pose_numbering()  # Residue numbering needs to be same for each chain...
            for chain in oriented_pdb.chain_id_list:
                oriented_pdb.reindex_chain_residues(chain)
            oriented_pdb.reorder_chains(exclude_chains_list=used_chains)
            used_chains += oriented_pdb.chain_id_list
            oriented_pdb_seq_a = oriented_pdb.atom_sequences[oriented_pdb.chain_id_list[0]]
            chain_in_asu = asu.match_entity_by_seq(other_seq=oriented_pdb_seq_a, force_closest=True)
            print('ASU\t: %s' % asu.atom_sequences[chain_in_asu])
            print('Orient\t: %s' % oriented_pdb_seq_a)
            des_mutations_asu = Ams.generate_mutations_from_seq(oriented_pdb_seq_a, asu.atom_sequences[chain_in_asu],
                                                                blanks=True)
            des_mutations_orient = Ams.generate_mutations_from_seq(asu.atom_sequences[chain_in_asu], oriented_pdb_seq_a,
                                                                   blanks=True)
            # TODO need to ensure that the design mutations have the right index! This must be adjusted for the offset
            #  of both sequences
            print('ASU: %s' % des_mutations_asu)
            print('Orient: %s' % des_mutations_orient)
            asu_offset, orient_offset = 0, 0
            for residue in des_mutations_asu:
                if des_mutations_asu[residue]['from'] == '-':
                    asu_offset += 1
                if des_mutations_asu[residue]['to'] == '-':
                    # oriented_pdb.delete_residue(chain_in_asu, residue)
                    # print('Design %s: Deleted residue %d from Oriented Input' % (design, residue))
                    asu.delete_residue(chain_in_asu, residue - asu_offset)
                    print('Design %s: Deleted residue %d from Design ASU' % (design, residue - asu_offset))
            for residue in des_mutations_orient:
                # if design_mutations[residue]['to'] == '-':
                #     asu.delete_residue(chain_in_asu, residue)
                #     print('Design %s: Deleted residue %d from Design ASU' % (design, residue))
                if des_mutations_orient[residue]['from'] == '-':
                    orient_offset += 1
                elif des_mutations_orient[residue]['to'] == '-':
                    for chain in oriented_pdb.chain_id_list:
                        # asu.delete_residue(chain, residue)
                        # print('Design %s: Deleted residue %d from Design ASU' % (design, residue))
                        oriented_pdb.delete_residue(chain, residue - orient_offset)
                        print('Design %s: Deleted residue %d from Oriented Input' % (design, residue - orient_offset))
                else:
                    for chain in oriented_pdb.chain_id_list:
                        oriented_pdb.mutate_to(chain, residue - orient_offset,
                                               res_id=des_mutations_orient[residue]['to'])
            # fix the residue numbering to account for deletions
            # asu.pose_numbering()
            for chain in asu.chain_id_list:
                asu.reindex_chain_residues(chain)
            for chain in oriented_pdb.chain_id_list:
                oriented_pdb.reindex_chain_residues(chain)
            # Get the updated sequences
            asu.update_chain_sequences()
            oriented_pdb.update_chain_sequences()
            oriented_pdb_seq_final = oriented_pdb.atom_sequences[oriented_pdb.chain_id_list[0]]
            final_mutations = Ams.generate_mutations_from_seq(oriented_pdb_seq_final, asu.atom_sequences[chain_in_asu],
                                                              offset=False, blanks=True)
            print('ASU\t: %s' % asu.atom_sequences[chain_in_asu])
            print('Orient\t: %s' % oriented_pdb_seq_final)
            if final_mutations != dict():
                exit('There is an error with indexing for Design %s, PDB %s. The index is %s' %
                     (design, pdb, final_mutations))

            if not os.path.exists(os.path.join(output_dir, design, sym)):
                os.makedirs(os.path.join(output_dir, design, sym))
            oriented_pdb.write(os.path.join(output_dir, design, sym, '%s.pdb' % pdb.lower()))  # pdb.lower()

            chain_correspondence[design]['pdb%s' % i] = {'asu_chain': chain_in_asu,
                                                         'dock_chains': oriented_pdb.chain_id_list}
            # chain_correspondence[chain_in_asu] = oriented_pdb.chain_id_list

        asu.write(os.path.join(output_dir, 'design_asus', '%s.pdb' % design))

        with open(os.path.join(output_dir, design, '%s_components.dock' % design)) as f:
            f.write('\n'.join('%s %s' % (pdb, sym) for pdb, sym in design_file_input[design]['source_pdb']))
            f.write('%s %s' % ('final_symmetry', design_file_input[design]['final_sym']))

    SDUtils.pickle_object(chain_correspondence, 'asu_to_oriented_oligomer_chain_correspondance', out_path=output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='\nTurn file(s) from a full PDB biological assembly into an ASU containing one copy of all entities'
                    ' in contact with the chain specified by chain')
    parser.add_argument('-d', '--directory', type=str, help='Directory where \'.pdb\' files are located.\nDefault=None',
                        default=None)
    parser.add_argument('-f', '--file', type=str, help='File with list of pdb files of interest\nDefault=None',
                        default=None)
    parser.add_argument('-c', '--chain', type=str, help='What chain would you like to leave?\nDefault=A', default='A')
    parser.add_argument('-o', '--output_destination', type=str, help='Where should new files be saved?\nDefault=CWD')

    args = parser.parse_args()
    logger = SDUtils.start_log()
    # if args.directory and args.file:
    #     logger.error('Can only specify one of -d or -f not both. If you want to use a file list from a different '
    #                  'directory than the one you are in, you will have to navigate there!')
    #     exit()
    # elif not args.directory and not args.file:
    #     logger.error('No file list specified. Please specify one of -d or -f to collect the list of files')
    #     exit()
    # elif args.directory:
    #     file_list = SDUtils.get_all_pdb_file_paths(args.directory)
    #     # location = SDUtils.get_all_pdb_file_paths(args.directory)
    # else:  # args.file
    #     file_list = SDUtils.to_iterable(args.file)
    #     # location = args.file

    # file_list = SDUtils.to_iterable(location)
    # new_files = make_asu(file_list, args.chain, destination=args.output_destination)
    # logger.info('ASU files were written to:\n%s' % '\n'.join(new_files))

    if not args.directory:
        logger.error('No pdb directory specified. Please specify -d to collect the design PDBs')
        exit()
    if not args.file:
        logger.error('No file specified. Please specify -f to collect the files')
        exit()

    design_recapitulation(args.file, args.directory, args.output_destination)
