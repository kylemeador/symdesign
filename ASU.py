import os
import sys
import subprocess
from glob import glob
from csv import reader
import argparse
import SymDesignUtils as SDUtils
import PathUtils as PUtils
import AnalyzeMutatedSequences as Ams
# sys.path.append(PUtils.nanohedra_source)
# print(sys.path)
# from utils.BioPDBUtils import biopdb_aligned_chain
# from Josh_push.BioPDBUtils import biopdb_aligned_chain  # removed for rmsd because of path issues


def make_asu(files, chain, destination=os.getcwd):
    return [SDUtils.extract_asu(file, chain=chain, outpath=destination) for file in files]


def make_asu_oligomer(asu, chain_map, location=os.getcwd):
    """Transform oriented oligomers to the ASU pose

    Args:
        asu (PDB): a PDB instance with the correctly oriented ASU for design
        chain_map (dict): Relation of the chains in the oriented oligomer to corresponding chain name in the asu PDB &
            the paths to the oriented oligomers
    Keyword Args:
        location=os.getcwd() (str): Location on disk to write the ASU oriented oligomer files
    Returns:
        (dict): {'nanohedra_output': /path/to/directory, 'pdb1': /path/to/design_asu/pdb1_oligomer.pdb, 'pdb1': ...}
            The locations of the oligomeric asu and the Nanohedra directory
    """
    # chain_map = {'pdb1': {'asu_chain': chain_in_asu, 'dock_chains': oriented_pdb.chain_id_list, 'path': path/to/.pdb}, {}}
    # move each oligomer to correct asu chain

    # for design in chain_map:  # TODO make correspondance design specific
    moved_oligomer = {}
    for pdb in chain_map:
        asu_chain = chain_map[pdb]['asu_chain']
        oriented_oligomer = SDUtils.read_pdb(chain_map[pdb]['path'])
        oligomer_chain = chain_map[pdb]['dock_chains'][0]
        # moved_oligomer[pdb] = biopdb_aligned_chain(asu, asu_chain, oriented_oligomer, oligomer_chain) #uncomment when path is solved
        moved_oligomer[pdb] = None  # remove when path is solved
        # moved_oligomer = biopdb_aligned_chain(pdb_fixed, chain_id_fixed, pdb_moving, chain_id_moving)
    final_comparison = {'nanohedra_output': glob(os.path.join(os.path.dirname(location), 'NanohedraEntry*DockedPoses'))[0]}
    # final_comparison = {'nanohedra_output': os.path.join(os.path.dirname(location), 'NanohedraEntry%sDockedPoses' % entry_num)}
    for pdb in moved_oligomer:
        moved_oligomer[pdb].write(os.path.join(location, '%s_oligomer.pdb' % pdb))  # design/design_asu/pdb1_oligomer.pdb
        final_comparison[pdb] = os.path.join(location, '%s_oligomer.pdb' % pdb)

    return final_comparison


def design_recapitulation(design_file, pdb_dir, output_dir, oligomer=False):
    qsbio_assemblies = SDUtils.unpickle(PUtils.qsbio)
    with open(design_file, 'r') as f:
        reading_csv = reader(f)
        design_file_input = {os.path.splitext(row[0])[0]:
                             # {'design_pdb': SDUtils.read_pdb(os.path.join(pdb_dir, row[0])),  # TODO reinstate upon recap exp termination
                             {'design_pdb': row[0],
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
    rmsd_comp_commands = {}
    for design in design_file_input:
        # asu = design_file_input[design]['design_pdb'].return_asu()  # TODO reinstate upon recap exp termination
        asu = SDUtils.read_pdb(os.path.join(output_dir, 'design_asus', design + '.pdb'))
        asu.reorder_chains()
        # asu.pose_numbering()
        # for chain in asu.chain_id_list:
        #     asu.reindex_chain_residues(chain)
        asu.reindex_all_chain_residues()
        asu.get_all_entities()

        design_dir = os.path.join(output_dir, design)
        if not os.path.exists(design_dir):
            os.makedirs(design_dir)

        used_chains, success = [], True
        for i, (pdb, sym) in enumerate(design_file_input[design]['source_pdb'], 1):  # TODO Not necessarily in order!!!
            if pdb in qsbio_assemblies:
                biological_assembly = qsbio_assemblies[pdb][0]  # find a verified biological assembly from QSBio
                if pdb == '3E6Q':
                    biological_assembly = 2
            else:
                print('%s: %s is not verified in QSBio! Arbitrarily choosing assembly 1' % (design, pdb))
                biological_assembly = 1
            new_file = SDUtils.download_pdb('%s_%s' % (pdb, biological_assembly),
                                            location=os.path.join(output_dir, 'biological_assemblies'))
            downloaded_pdb = SDUtils.read_pdb(new_file)
            oriented_pdb = downloaded_pdb.orient(sym, PUtils.orient_dir)  # , generate_oriented_pdb=False)
            if oriented_pdb.all_atoms == list():
                print('%s failed! Skipping design %s' % (pdb, design))
                success = False
                break
            # oriented_pdb.name = pdb.lower()
            # oriented_pdb.pose_numbering()  # Residue numbering needs to be same for each chain...
            try:  # Some pdb's are messed up and don't have CA or CB?!
                for chain in oriented_pdb.chain_id_list:
                    oriented_pdb.reindex_chain_residues(chain)
            except AttributeError:
                print('%s failed! Missing residue information in re-indexing' % pdb)
            oriented_pdb.reorder_chains(exclude_chains_list=used_chains)  # redoes sequences
            used_chains += oriented_pdb.chain_id_list
            oriented_pdb_seq_a = oriented_pdb.atom_sequences[oriented_pdb.chain_id_list[0]]
            chain_in_asu = asu.match_entity_by_seq(other_seq=oriented_pdb_seq_a, force_closest=True)
            # print('ASU\t: %s' % asu.atom_sequences[chain_in_asu])
            # print('Orient\t: %s' % oriented_pdb_seq_a)
            des_mutations_asu = Ams.generate_mutations_from_seq(oriented_pdb_seq_a, asu.atom_sequences[chain_in_asu],
                                                                blanks=True)
            des_mutations_orient = Ams.generate_mutations_from_seq(asu.atom_sequences[chain_in_asu], oriented_pdb_seq_a,
                                                                   blanks=True)
            # print('ASU: %s' % des_mutations_asu)
            # print('Orient: %s' % des_mutations_orient)
            # Ensure that the design mutations have the right index, must be adjusted for the offset of both sequences
            asu_offset, orient_offset = 0, 0
            for residue in des_mutations_asu:
                if des_mutations_asu[residue]['from'] == '-' and residue > 0:
                    asu_offset += 1
                if des_mutations_asu[residue]['to'] == '-':
                    asu.delete_residue(chain_in_asu, residue - asu_offset)
                    # print('Design %s: Deleted residue %d from Design ASU' % (design, residue - asu_offset))
            # for k in range(1, 34):
            #     print(oriented_pdb.get_residue(oriented_pdb.chain_id_list[0], k).type)
            #     print(oriented_pdb.get_residue(oriented_pdb.chain_id_list[0], k).number)
            # print(oriented_pdb.get_residue(chain, 0).type)
            # print(oriented_pdb.get_residue(chain, 0).number)
            for residue in des_mutations_orient:
                # if design_mutations[residue]['to'] == '-':
                #     asu.delete_residue(chain_in_asu, residue)
                #     print('Design %s: Deleted residue %d from Design ASU' % (design, residue))
                if des_mutations_orient[residue]['from'] == '-' and residue > 0:
                    orient_offset += 1
                elif des_mutations_orient[residue]['to'] == '-':  # all need to be removed from reference
                    for chain in oriented_pdb.chain_id_list:
                        oriented_pdb.delete_residue(chain, residue - orient_offset)
                        # print('Design %s: Deleted residue %d from Oriented Input' % (design, residue - orient_offset))
                else:
                    for chain in oriented_pdb.chain_id_list:
                        oriented_pdb.mutate_to(chain, residue - orient_offset,
                                               res_id=des_mutations_orient[residue]['to'])
            # fix the residue numbering to account for deletions
            # asu.pose_numbering()
            # for chain in asu.chain_id_list:
            #     asu.reindex_chain_residues(chain)
            asu.reindex_all_chain_residues()
            # for chain in oriented_pdb.chain_id_list:
            #     oriented_pdb.reindex_chain_residues(chain)
            oriented_pdb.reindex_all_chain_residues()
            # Get the updated sequences
            asu.update_chain_sequences()
            oriented_pdb.update_chain_sequences()
            oriented_pdb_seq_final = oriented_pdb.atom_sequences[oriented_pdb.chain_id_list[0]]
            final_mutations = Ams.generate_mutations_from_seq(oriented_pdb_seq_final, asu.atom_sequences[chain_in_asu],
                                                              offset=False, blanks=True)
            # print('ASU\t: %s' % asu.atom_sequences[chain_in_asu])
            # print('Orient\t: %s' % oriented_pdb_seq_final)
            if final_mutations != dict():
                print('There is an error with indexing for Design %s, PDB %s. The index is %s' %
                      (design, pdb, final_mutations))
                break

            if not os.path.exists(os.path.join(output_dir, design, '%s_%s' % (i, sym))):
                os.makedirs(os.path.join(output_dir, design, '%s_%s' % (i, sym)))

            out_path = os.path.join(output_dir, design, '%s_%s' % (i, sym), '%s.pdb' % pdb.lower())
            oriented_pdb.write(out_path)

            # when sym of directory is not the same
            # if not os.path.exists(os.path.join(output_dir, design, sym)):
            #     os.makedirs(os.path.join(output_dir, design, sym))
            # oriented_pdb.write(os.path.join(output_dir, design, sym, '%s.pdb' % pdb.lower()))

            chain_correspondence[design]['pdb%s' % i] = {'asu_chain': chain_in_asu,
                                                         'dock_chains': oriented_pdb.chain_id_list,
                                                         'path': out_path}
            # chain_correspondence[chain_in_asu] = oriented_pdb.chain_id_list

        # asu_path = os.path.join(output_dir, 'design_asus', '%s' % design)
        if success:
            asu_path = os.path.join(design_dir, 'design_asu')  # New as of oligomeric processing
            if not os.path.exists(asu_path):
                os.makedirs(asu_path)
            if oligomer:  # requires an ASU PDB instance beforehand
                rmsd_comp_commands[design] = make_asu_oligomer(asu, chain_correspondence[design], location=asu_path)
                # {'nanohedra_output': /path/to/directory, 'pdb1': /path/to/design_asu/pdb1_oligomer.pdb, 'pdb2': ...}
            else:
                asu.write(os.path.join(asu_path, '%s_asu.pdb' % design))

            # {1_Sym: PDB1, 1_Sym2: PDB2, 'final_symmetry': I}
            sym_d = {'%s_%s' % (i, sym): pdb.lower() for i, (pdb, sym) in enumerate(design_file_input[design]['source_pdb'])}
            sym_d['final_symmetry'] = design_file_input[design]['final_sym']
            SDUtils.pickle_object(sym_d, name='%s_dock' % design, out_path=os.path.join(output_dir, design))

        # with open(os.path.join(output_dir, design, '%s_components.dock' % design), 'w') as f:
        #     f.write('\n'.join('%s %s' % (pdb.lower(), sym) for pdb, sym in design_file_input[design]['source_pdb']))
        #     f.write('\n%s %s' % ('final_symmetry', design_file_input[design]['final_sym']))

    if rmsd_comp_commands != dict():
        SDUtils.pickle_object(rmsd_comp_commands, name='recap_rmsd_command_paths', out_path=output_dir)

    missing = []
    for i, design in enumerate(chain_correspondence):
        if len(chain_correspondence[design]) != 2:
            missing.append(i)
    if missing != list():
        missing_str = map(str, missing)
        print('Designs missing one of two chains:\n%s' % ', '.join(missing_str))

    SDUtils.pickle_object(chain_correspondence, 'asu_to_oriented_oligomer_chain_correspondance', out_path=output_dir)


def run_rmsd_calc(design_list, design_map_pickle):
    design_map = SDUtils.unpickle(design_map_pickle)
    logger.info('Starting RMSD calculation')
    for design in design_list:
        design = design.strip()
        # rmsd_cmd = ['python', '/home/kmeador/Nanohedra/crystal_vs_docked_v2.py', design_map[design]['pdb1'],
        #             design_map[design]['pdb2'], design_map[design]['nanohedra_output'],
        #             design_map[design]['nanohedra_output']]
        rmsd_cmd_flip = ['python', '/home/kmeador/Nanohedra/crystal_vs_docked_v2.py', design_map[design]['pdb2'],
                         design_map[design]['pdb1'], design_map[design]['nanohedra_output'],
                         design_map[design]['nanohedra_output']]
        # p = subprocess.Popen(rmsd_cmd)  # , capture_output=True)
        p = subprocess.Popen(rmsd_cmd_flip)  # , capture_output=True)
        p.communicate()
        logger.info('%s finished RMSD calculation' % design)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='\nTurn file(s) from a full PDB biological assembly into an ASU containing one copy of all entities'
                    ' in contact with the chain specified by chain')
    parser.add_argument('-d', '--directory', type=str, help='Directory where \'.pdb\' files to set up ASU extraction'
                                                            'are located.\nDefault=CWD',
                        default=os.getcwd())
    parser.add_argument('-f', '--file', type=str, help='File with list of pdb files of interest\nDefault=None',
                        default=None)
    parser.add_argument('-c', '--chain', type=str, help='What chain would you like to leave?\nDefault=A', default='A')
    parser.add_argument('-p', '--out_path', type=str, help='Where should new files be saved?\nDefault=CWD')
    parser.add_argument('-o', '--oligomer_asu', action='store_true', help='Whether the full oligomer used for docking '
                                                                          'should be saved in the ASU?\nDefault=False')
    parser.add_argument('-m', '--design_map', type=str, help='The location of a file to map the design directory to '
                                                             'lower and higher symmetry\nDefault=None', default=None)

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

    if args.design_map:
        with open(args.file, 'r') as f:
            all_design_directories = f.readlines()
            design_d_names = map(os.path.basename, all_design_directories)

        run_rmsd_calc(design_d_names, args.design_map)
    else:
        design_recapitulation(args.file, args.directory, args.out_path, args.oligomer_asu)
