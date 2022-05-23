import argparse
import os
import subprocess
from csv import reader
from glob import glob
from itertools import repeat

import pandas as pd

import PathUtils as PUtils
import SequenceProfile
import SymDesignUtils as SDUtils
# sys.path.append(PUtils.nanohedra_source)
# print(sys.path)
# from utils.BioPDBUtils import biopdb_aligned_chain
from utils.PDBUtils import biopdb_aligned_chain
from PDB import PDB, fetch_pdb

# if sys.version[0] < 3:
pickle_prot = 2
# else:
#     pickle_prot = pickle.HIGHEST_PROTOCOL
sym_dict = {'4NWN': 'T', '4NWO': 'T', '4NWP': 'T', '4ZK7': 'T', '5CY5': 'T', '5IM4': 'I', '5IM5': 'I', '5IM6': 'I',
            '6P6F': 'I', '6VFH': 'T', '6VFI': 'O', '6VFJ': 'I', '6VL6': 'T'}


def make_asu(pdb_file, chain=None, out_path=os.getcwd, center=True):
    if '4NWR' in pdb_file:
        return None
    pdb = PDB.from_file(pdb_file)
    if center:
        print(pdb.center_of_mass)
        pdb.translate(-pdb.center_of_mass)
        pdb.center_of_mass
        print(pdb.center_of_mass)
        pdb.write(out_path=os.path.join(out_path, 'expanded', 'centered' + os.path.basename(pdb.filepath)))
    # asu = pdb.return_asu(chain)  # no chain needed, just use the default
    # # asu = PDB.from_atoms(pdb.get_asu(chain))  # no chain needed, just use the default
    # asu.write(out_path=os.path.join(out_path, os.path.basename(pdb.filepath)), header=None)  # Todo make symmetry for point groups
    # print(sym_dict[pdb.name])
    # pose = Pose.from_asu(asu, symmetry=sym_dict[pdb.name])
    # print('Total Atoms: %d' % pose.asu.number_of_atoms)
    # print('Coords of length %d: %s' % (pose.asu.coords.shape[0], pose.asu.coords))
    # pose.get_assembly_symmetry_mates()
    # # pose.set_symmetry(symmetry=sym_dict[pdb.name], generate_symmetry_mates=True)
    # pose.write(out_path=os.path.join(out_path, 'expanded', os.path.basename(pdb.filepath)))

    return out_path
    # pose = Pose.from_asu()
    # return [SDUtils.extract_asu(file, chain=chain, outpath=destination) for file in files]


def make_asu_oligomer(asu, chain_map, location=os.getcwd()):
    """Transform oriented oligomers to the ASU pose

    Args:
        asu (PDB): a PDB instance with the correctly oriented ASU for design
        chain_map (dict): {'pdb1': {'asu_chain': 'A', 'dock_chains': ['A', 'B', 'C'], 'path': path/to/.pdb}, {}}
            Relation of the chains in the oriented oligomer to corresponding chain name in the asu PDB & the paths to
            the oriented oligomers
    Keyword Args:
        location=os.getcwd() (str): Location on disk to write the ASU oriented oligomer files
    Returns:
        (dict): {'nanohedra_output': /path/to/directory, 'pdb1': /path/to/design_asu/pdb1_oligomer.pdb, 'pdb1': ...}
            The locations of the oligomeric asu and the Nanohedra directory
    """
    moved_oligomer = {}
    for pdb in chain_map:
        asu_chain = chain_map[pdb]['asu_chain']
        oriented_oligomer = PDB.from_file(chain_map[pdb]['path'])
        # oriented_oligomer = SDUtils.read_pdb(chain_map[pdb]['path'])
        oligomer_chain = chain_map[pdb]['dock_chains'][0]
        moved_oligomer[pdb] = biopdb_aligned_chain(asu.chain(asu_chain), oriented_oligomer, oligomer_chain)
        # moved_oligomer[pdb] = biopdb_aligned_chain(asu, asu_chain, oriented_oligomer, oligomer_chain)
        # moved_oligomer = biopdb_aligned_chain(pdb_fixed, chain_id_fixed, pdb_moving, chain_id_moving)
    final_comparison = {'nanohedra_output': glob(os.path.join(os.path.dirname(location), 'NanohedraEntry*DockedPoses'))[0]}
    for pdb in moved_oligomer:
        moved_oligomer[pdb].write(
            out_path=os.path.join(location, '%s_oligomer.pdb' % pdb))  # design/design_asu/pdb1_oligomer.pdb
        final_comparison[pdb] = os.path.join(location, '%s_oligomer.pdb' % pdb)

    return final_comparison


def design_recapitulation(design_file, output_dir, pdb_dir=None, oligomer=False):
    """Recapitulate a published design using Nanohedra docking

    Args
        design_file (str): Location of .csv file containing design information including:
            design_name, pdb_code1, pdb_code2, symm1, symm2, final_symm
        output_dir (str): Location where all files processed for Nanohedra should be written
    Keyword Args
        pdb_dir=None (str): Location on disk of all design .pdb files. If not provided, then the PDB will be sourced
            from the default directory structure, i.e. output_dir/design_asus/design_name.pdb
        oligomer=False (bool): Whether or not the output ASU should contain the oligomer (True) or just the ASU (False)
    Returns:
        None
    """
    qsbio_assemblies = SDUtils.unpickle(PUtils.qsbio)
    with open(design_file, 'r') as f_csv:
        reading_csv = reader(f_csv)
        if pdb_dir:
            design_file_input = {os.path.splitext(row[0])[0]:
                                 {'design_pdb': PDB.from_file(os.path.join(pdb_dir, row[0])),
                                 # {'design_pdb': SDUtils.read_pdb(os.path.join(pdb_dir, row[0])),
                                  'source_pdb': [(row[1], row[3]), (row[2], row[4])], 'final_sym': row[5]}
                                 for row in reading_csv}  # 'pdb1': 'sym1': 'pdb2': 'sym2':
        else:
            design_file_input = {os.path.splitext(row[0])[0]:
                                 {'design_pdb': row[0], 'source_pdb': [(row[1], row[3]), (row[2], row[4])],
                                  'final_sym': row[5]} for row in reading_csv}  # 'pdb1': 'sym1': 'pdb2': 'sym2':

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
        if pdb_dir:
            print('CAUTION FINDING THE ASU USING THE INPUT PDB HAS BEEN REWORKED AND MAY NOT BE STABLE. '
                  'CONSIDER USING THE Pose.return_contacting_asu() instead!')
            asu = design_file_input[design]['design_pdb'].return_asu()
        else:
            asu = PDB.from_file(os.path.join(output_dir, 'design_asus', design + '.pdb'))  # old, design_asus outside
            # asu = PDB(file=os.path.join(output_dir, design, 'design_asus', design  + '.pdb'))  # TODO in new, asu is inside design directory
        asu.reorder_chains()
        # asu.renumber_residues()
        asu.renumber_residues_by_chain()
        asu.get_entity_info_from_atoms()

        design_dir = os.path.join(output_dir, design)
        if not os.path.exists(design_dir):
            os.makedirs(design_dir)

        used_chains, success = [], True
        # for i, sym_order in enumerate(design_file_input[design]['source_pdb'], 1):
        #     pdb, sym = design_file_input[design]['source_pdb'][sym_order]
        for i, (pdb, sym) in enumerate(design_file_input[design]['source_pdb'], 1):
            if pdb in qsbio_assemblies:
                biological_assembly = qsbio_assemblies[pdb][0]  # find a verified biological assembly from QSBio
                if pdb == '3E6Q':  # manual override list
                    biological_assembly = 2
            else:
                logger.warning('%s: %s is not verified in QSBio! Arbitrarily choosing assembly 1' % (design, pdb))
                biological_assembly = 1
            new_file = \
                fetch_pdb(pdb, assembly=biological_assembly, out_dir=os.path.join(output_dir, 'biological_assemblies'))
            if new_file:
                downloaded_pdb = PDB.from_file(new_file[0])
            else:
                continue
            downloaded_pdb.orient(symmetry=sym)
            if not downloaded_pdb.atoms:
                logger.error('%s failed! Skipping design %s' % (pdb, design))
                success = False
                break
            # downloaded_pdb.name = pdb.lower()
            # downloaded_pdb.renumber_residues()  # Residue numbering needs to be same for each chain...
            try:  # Some pdb's are messed up and don't have CA or CB?!
                for chain in downloaded_pdb.chain_ids:
                    downloaded_pdb.chain(chain).renumber_residues()
                    # downloaded_pdb.reindex_chain_residues(chain)
            except AttributeError:
                logger.error('%s failed! Missing residue information in re-indexing' % pdb)
                success = False
                break
            downloaded_pdb.reorder_chains(exclude_chains=used_chains)  # redoes sequences
            used_chains += downloaded_pdb.chain_ids
            oriented_pdb_seq_a = downloaded_pdb.chains[0].sequence
            chain_in_asu = asu.match_entity_by_seq(other_seq=oriented_pdb_seq_a)
            asu_sequence = asu.chain(chain_in_asu).sequence
            logger.debug('ASU\t: %s' % asu_sequence)
            logger.debug('Orient\t: %s' % oriented_pdb_seq_a)
            des_mutations_asu = \
                SequenceProfile.generate_mutations(asu_sequence, oriented_pdb_seq_a, blanks=True)
            des_mutations_orient = \
                SequenceProfile.generate_mutations(oriented_pdb_seq_a, asu_sequence, blanks=True)
            logger.debug('ASU: %s' % des_mutations_asu)
            logger.debug('Orient: %s' % des_mutations_orient)
            # Ensure that the design mutations have the right index, must be adjusted for the offset of both sequences
            asu_offset, orient_offset = 0, 0
            for residue in des_mutations_asu:
                if des_mutations_asu[residue]['from'] == '-' and residue > 0:
                    asu_offset += 1
                if des_mutations_asu[residue]['to'] == '-':
                    asu.delete_residue(chain_in_asu, residue - asu_offset)
                    # logger.debug'Design %s: Deleted residue %d from Design ASU' % (design, residue - asu_offset))
            # for k in range(1, 34):
            #     logger.debug(downloaded_pdb.get_residue(downloaded_pdb.chain_ids[0], k).type)
            #     logger.debug(downloaded_pdb.get_residue(downloaded_pdb.chain_ids[0], k).number)
            # logger.debug(downloaded_pdb.get_residue(chain, 0).type)
            # logger.debug(downloaded_pdb.get_residue(chain, 0).number)
            for residue in des_mutations_orient:
                # if design_mutations[residue]['to'] == '-':
                #     asu.delete_residue(chain_in_asu, residue)
                #     logger.debug('Design %s: Deleted residue %d from Design ASU' % (design, residue))
                if des_mutations_orient[residue]['from'] == '-' and residue > 0:
                    orient_offset += 1
                elif des_mutations_orient[residue]['to'] == '-':  # all need to be removed from reference
                    for chain in downloaded_pdb.chain_ids:
                        downloaded_pdb.delete_residue(chain, residue - orient_offset)
                        # logger.debug('Design %s: Deleted residue %d from Oriented Input' % (design, residue - orient_offset))
                else:
                    for chain in downloaded_pdb.chain_ids:
                        downloaded_pdb.chain(chain).mutate_residue(number=residue - orient_offset,
                                                                   to=des_mutations_orient[residue]['to'])
                        # downloaded_pdb.mutate_to(chain, residue - orient_offset,
                        #                        res_id=des_mutations_orient[residue]['to'])
            # fix the residue numbering to account for deletions
            # asu.renumber_residues()
            # for chain in asu.chain_ids:
            #     asu.reindex_chain_residues(chain)
            asu.renumber_residues_by_chain()
            # for chain in downloaded_pdb.chain_ids:
            #     downloaded_pdb.reindex_chain_residues(chain)
            downloaded_pdb.renumber_residues_by_chain()
            # Get the updated sequences
            # asu.get_chain_sequences()  # 1/29/21 KM updated this function which might affect this routine
            # downloaded_pdb.get_chain_sequences()
            oriented_pdb_seq_final = downloaded_pdb.chains[0].sequence
            # Todo I think that this is way wrong
            final_mutations = SequenceProfile.generate_mutations(asu.chain(chain_in_asu).sequence,
                                                                 oriented_pdb_seq_final, offset=False, blanks=True)
            logger.debug('ASU\t: %s' % asu.chain(chain_in_asu).sequence)
            logger.debug('Orient\t: %s' % oriented_pdb_seq_final)
            if final_mutations != dict():
                logger.error('There is an error with indexing for Design %s, PDB %s. The index is %s' %
                             (design, pdb, final_mutations))
                break

            if not os.path.exists(os.path.join(output_dir, design, '%s_%s' % (i, sym))):
                os.makedirs(os.path.join(output_dir, design, '%s_%s' % (i, sym)))

            out_path = os.path.join(output_dir, design, '%s_%s' % (i, sym), '%s.pdb' % pdb.lower())
            downloaded_pdb.write(out_path=out_path)

            # when sym of directory is not the same
            # if not os.path.exists(os.path.join(output_dir, design, sym)):
            #     os.makedirs(os.path.join(output_dir, design, sym))
            # downloaded_pdb.write(os.path.join(output_dir, design, sym, '%s.pdb' % pdb.lower()))

            chain_correspondence[design]['pdb%s' % i] = {'asu_chain': chain_in_asu,
                                                         'dock_chains': downloaded_pdb.chain_ids,
                                                         'path': out_path}
            # chain_correspondence[chain_in_asu] = downloaded_pdb.chain_ids

        # asu_path = os.path.join(output_dir, 'design_asus', '%s' % design)
        if success:
            asu_path = os.path.join(design_dir, 'design_asu')  # New as of oligomeric processing
            if not os.path.exists(asu_path):
                os.makedirs(asu_path)
            if oligomer:  # requires an ASU PDB instance beforehand
                rmsd_comp_commands[design] = make_asu_oligomer(asu, chain_correspondence[design], location=asu_path)
            else:
                asu.write(out_path=os.path.join(asu_path, '%s_asu.pdb' % design))

            # {1_Sym1: PDB1, 2_Sym2: PDB2, 'final_symmetry': I}
            sym_d = {'%s_%s' % (i, sym): pdb.lower() for i, (pdb, sym) in enumerate(design_file_input[design]['source_pdb'])}
            sym_d['final_symmetry'] = design_file_input[design]['final_sym']
            # 10/6/20 removed _vflip
            SDUtils.pickle_object(sym_d, name='%s_dock' % design, out_path=os.path.join(output_dir, design),
                                  protocol=pickle_prot)

        # with open(os.path.join(output_dir, design, '%s_components.dock' % design), 'w') as f:
        #     f.write('\n'.join('%s %s' % (pdb.lower(), sym) for pdb, sym in design_file_input[design]['source_pdb']))
        #     f.write('\n%s %s' % ('final_symmetry', design_file_input[design]['final_sym']))
    # 10/6/20 removed _vflip
    if rmsd_comp_commands:
        # {design: {'nanohedra_output': /path/to/directory, 'pdb1': /path/to/design_asu/pdb1_oligomer.pdb, 'pdb2': ...}}
        SDUtils.pickle_object(rmsd_comp_commands, name='recap_rmsd_command_paths', out_path=output_dir,
                              protocol=pickle_prot)

    missing = []
    for design in chain_correspondence:
        if len(chain_correspondence[design]) != 2:
            missing.append(design)
            chain_correspondence.pop(design)
    if missing != list():
        missing_str = map(str, missing)
        logger.critical('Designs missing one of two chains:\n%s' % ', '.join(missing_str))
    # 10/6/20 removed _vflip
    SDUtils.pickle_object(chain_correspondence, 'asu_to_oriented_oligomer_chain_correspondance',
                          out_path=output_dir, protocol=pickle_prot)


def run_rmsd_calc(design_list, design_map_pickle, command_only=False):
    """Calculate the interface RMSD between a reference pose and a docked pose

    Args:
        design_list (list): List of designs to search for that have an entry in the design_map_pickle
        design_map_pickle (str): The path of a serialized file to unpickle into a dictionary with directory maps
    Returns:
        None
    """
    design_map = SDUtils.unpickle(design_map_pickle)
    logger.info('Design Analysis mode: RMSD calculation')
    log_file = os.path.join(os.getcwd(), 'RMSD_calc.log')
    rmsd_commands = []
    with open(log_file, 'a+') as log_f:
        for design in design_list:
            logger.info('%s Starting RMSD calculation' % design)
            design = design.strip()
            outdir = os.path.join(design_map[design]['nanohedra_output'], 'rmsd_calculation')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            rmsd_cmd = ['python', '/home/kmeador/symdesign/design_recap_scripts/crystal_vs_docked_v2.py',  # Nanohedra/
                        design_map[design]['pdb1'], design_map[design]['pdb2'], design_map[design]['nanohedra_output'],
                        outdir]
            if command_only:
                rmsd_commands.append(
                    SDUtils.write_shell_script(subprocess.list2cmdline(rmsd_cmd), name='rmsd_calculation',
                                               out_path=outdir))
            else:
                p = subprocess.Popen(rmsd_cmd, stdout=log_f, stderr=log_f)
                p.communicate()
                # rmsd_cmd_flip = ['python', '/home/kmeador/Nanohedra/crystal_vs_docked_v2.py', design_map[design]['pdb2'],
                #                  design_map[design]['pdb1'], design_map[design]['nanohedra_output'],
                #                  design_map[design]['nanohedra_output']]
                # p = subprocess.Popen(rmsd_cmd_flip, stdout=log_f, stderr=log_f)
                logger.info('%s finished' % design)
            # log_f.write(design)

    return rmsd_commands


def run_all_to_all_calc(design_list, design_map_pickle, command_only=False):
    """Calculate the all to all interface RMSD between a each docked pose in the top matching directories

    Args:
        design_list (list): List of designs to search for that have an entry in the design_map_pickle
        design_map_pickle (str): The path of a serialized file to unpickle into a dictionary with directory maps
    Returns:
        None
    """
    design_map = SDUtils.unpickle(design_map_pickle)
    logger.info('Design Analysis mode: All to All calculations')
    log_file = os.path.join(os.getcwd(), 'RMSD_calc.log')
    all_to_all_commands = []
    with open(log_file, 'a+') as log_f:
        for design in design_list:
            logger.info('%s: Starting All to All calculation' % design)
            design = design.strip()
            outdir = os.path.join(design_map[design]['nanohedra_output'], 'rmsd_calculation')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            _cmd = ['python', '/home/kmeador/symdesign/design_recap_scripts/top_n_all_to_all_docked_poses_irmsd.py',
                    design_map[design]['nanohedra_output'], os.path.join(outdir, 'crystal_vs_docked_irmsd.txt')]
            if command_only:
                all_to_all_commands.append(
                    SDUtils.write_shell_script(subprocess.list2cmdline(_cmd), name='all_to_all', out_path=outdir))
            else:
                p = subprocess.Popen(_cmd, stdout=log_f, stderr=log_f)
                p.communicate()
                logger.info('%s finished' % design)
            # log_f.write(design)

    return all_to_all_commands


def run_cluster_calc(design_list, design_map_pickle, command_only=False):
    """Calculate the all to all interface RMSD between a each docked pose in the top matching directories

    Args:
        design_list (list): List of designs to search for that have an entry in the design_map_pickle
        design_map_pickle (str): The path of a serialized file to unpickle into a dictionary with directory maps
    Returns:
        None
    """
    design_map = SDUtils.unpickle(design_map_pickle)
    logger.info('Design Analysis mode: Clustering RMSD\'s')
    log_file = os.path.join(os.getcwd(), 'RMSD_calc.log')
    cluster_commands = []
    with open(log_file, 'a+') as log_f:
        for design in design_list:
            logger.info('%s: Starting Clustering calculation' % design)
            design = design.strip()
            outdir = os.path.join(design_map[design]['nanohedra_output'], 'rmsd_calculation')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            _cmd = ['python', '/home/kmeador/symdesign/design_recap_scripts/cluster_all_to_all_docked_poses_irmsd_v0.py',
                    os.path.join(outdir, 'top2000_all_to_all_docked_poses_irmsd.txt'),
                    os.path.join(outdir, 'crystal_vs_docked_irmsd.txt'), design]
            if command_only:
                cluster_commands.append(
                    SDUtils.write_shell_script(subprocess.list2cmdline(_cmd), name='rmsd_clustering', out_path=outdir))
            else:
                p = subprocess.Popen(_cmd, stdout=log_f, stderr=log_f)
                p.communicate()
                logger.info('%s finished' % design)
            # log_f.write(design)

    return cluster_commands


def collect_rmsd_calc(design_list, number=10, location=os.getcwd()):
    """Returns a RMSD dictionary for the top (number) of all designs of interest in (design_list)

    Returns:
        (dict): {'design': {1: {'pose': DEGEN_1_1_ROT_47_44_tx_22, 'iRMSD': 2.167, 'score': 13.414 'rank': 556}, ...}
    """
    entry_d = {'I': {('C2', 'C3'): 8, ('C2', 'C5'): 14, ('C3', 'C5'): 56}, 'T': {('C2', 'C3'): 4, ('C3', 'C3'): 52}}
    top_rmsd_d, missing_designs = {}, []
    for design in design_list:
        design_sym = design[:1]
        design_components = design[1:3]
        entry = entry_d[design_sym][('C%s' % design_components[1], 'C%s' % design_components[0])]
        try:
            with open(os.path.join(location, '%s' % design, 'NanohedraEntry%dDockedPoses' % entry, 'rmsd_calculation',
                                   'crystal_vs_docked_irmsd.txt')) as f_irmsd:
                top_10 = []
                top_rmsd_d[design] = {}
                for i in range(number):
                    top_10.append(f_irmsd.readline())
                    top_10[i] = top_10[i].split()
                    top_rmsd_d[design][i + 1] = {'pose': top_10[i][0], 'iRMSD': top_10[i][1], 'score': top_10[i][2],
                                                 'rank': top_10[i][3]}
        except FileNotFoundError:
            logger.info('Design %s has no RMSD file' % design)
            missing_designs.append(design)

    logger.info('All missing designs %d:\n%s' % (len(missing_designs), missing_designs))

    return top_rmsd_d


def report_top_rmsd(rmsd_d):
    top_rmsd_s = pd.Series()
    top_rank_s = pd.Series()
    top_score_s = pd.Series()
    motif_library_s = pd.Series()
    for design in rmsd_d:
        top_rmsd_s['%s_%s' % (design, rmsd_d[design][1]['pose'])] = rmsd_d[design][1]['iRMSD']
        top_rank_s['%s_%s' % (design, rmsd_d[design][1]['pose'])] = rmsd_d[design][1]['rank']
        top_score_s['%s_%s' % (design, rmsd_d[design][1]['pose'])] = rmsd_d[design][1]['score']
        if design[:3] in ['I32', 'I52']:
            motif_library_s['%s_%s' % (design, rmsd_d[design][1]['pose'])] = True

    top_df = pd.concat([top_rmsd_s, top_score_s, top_rank_s, motif_library_s], axis=1)
    top_df.columns = ['iRMSD', 'Nanohedra Score', 'Nanohedra Rank', 'TC Dock Motifs']
    top_df.sort_values('iRMSD', inplace=True)

    pd.options.display.max_rows = 182
    print(top_df)
    # top_rmsd_s.sort_values(inplace=True)
    # print(top_rmsd_s)


modes = ['report', 'reference_rmsd', 'all_to_all_rmsd', 'cluster_rmsd', 'all_rmsd']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='\nTurn file(s) from a full PDB biological assembly into an ASU containing one copy of all entities'
                    ' in contact with the chain specified by chain')
    parser.add_argument('--debug', action='store_true', help='Debug all steps to standard out?\nDefault=False')
    parser.add_argument('-c', '--command_only', action='store_true', help='Whether to only write commands, not run '
                                                                          'them')
    parser.add_argument('-d', '--directory', type=str, help='Directory where \'.pdb\' files to set up ASU extraction'
                                                            'are located.\n')
    parser.add_argument('-f', '--file', type=str, help='File with list of pdb files of interest\nDefault=None',
                        default=None)
    # parser.add_argument('-c', '--chain', type=str, help='What chain would you like to leave?\nDefault=A', default='A')
    parser.add_argument('-O', '--oligomer_asu', action='store_true', help='Whether the full oligomer used for docking '
                                                                          'should be saved in the ASU?\nDefault=False')
    parser.add_argument('-m', '--mode', type=str, help='Which mode of RMSD processing to use. Chose on of %s'
                                                       % ','.join(modes))
    parser.add_argument('-o', '--out_path', type=str, help='Where should new files be saved?\nDefault=CWD')
    parser.add_argument('-r', '--recap', action='store_true', help='Special mode to run the design recap protocol')
    parser.add_argument('-p', '--design_map', type=str, help='The location of a file to map the design directory to '
                                                             'lower and higher symmetry\nDefault=None', default=None)
    parser.add_argument('-rot', '--flip', action='store_true', help='Whether to flip the orientation of a design')

    args = parser.parse_args()
    # Start logging output
    if args.debug:
        logger = SDUtils.start_log(level=1)
        SDUtils.set_logging_to_debug()
        logger.debug('Debug mode. Produces verbose output and not written to any .log files')
    else:
        logger = SDUtils.start_log(name=os.path.basename(__file__), propagate=True)

    logger.info('Starting %s with options:\n\t%s' %
                (os.path.basename(__file__),
                 '\n\t'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    # if args.directory and args.file:
    #     logger.error('Can only specify one of -d or -f not both. If you want to use a file list from a different '
    #                  'directory than the one you are in, you will have to navigate there!')
    #     exit()
    # elif not args.directory and not args.file:
    #     logger.error('No file list specified. Please specify one of -d or -f to collect the list of files')
    #     exit()
    # elif args.directory:
    #     file_list = SDUtils.get_all_file_paths(args.directory, extension='.pdb')
    #     # location = SDUtils.get_all_file_paths(args.directory, extension='.pdb')
    # else:  # args.file
    #     file_list = SDUtils.to_iterable(args.file)
    #     # location = args.file

    # file_list = SDUtils.to_iterable(location)
    # new_files = make_asu(file_list, args.chain, destination=args.output_destination)
    # logger.info('ASU files were written to:\n%s' % '\n'.join(new_files))

    # if not args.directory:
    #     logger.error('No pdb directory specified. Please specify -d to collect the design PDBs')
    #     exit()

    if args.recap:
        if not args.file:
            logger.error('No file specified. Please specify -f to collect the files')
            exit()

        if args.design_map:
            if args.flip:
                chain_map = SDUtils.unpickle(args.design_map)
                for design in chain_map:
                    try:
                        input_pdb = chain_map[design]['pdb2']['path']
                    except KeyError:
                        print('No \'pdb2\' found for design %s' % design)
                        continue
                    p = subprocess.Popen(['python', '/home/kmeador/Nanohedra/flip_pdb_180deg_y.py', input_pdb])
                    # update the file name in the docking pickle
                    dock_instructions = glob(os.path.join(os.path.dirname(os.path.dirname(input_pdb)), '*_vflip_dock.pkl'))
                    try:
                        dock_d = SDUtils.unpickle(dock_instructions[0])
                    except IndexError:
                        print('No %s found for design %s' % (os.path.join(os.path.dirname(input_pdb), '*_vflip_dock.pkl'),
                                                             design))
                        continue

                    # max_sym, max_name = 0, None
                    for sym in list(set(dock_d.keys()) - {'final_symmetry'}):
                        # sym_l = sym.split('_')
                        # sym_l[0] = int(sym_l[0])
                        # if sym_l[0] >= max_sym:
                        #     max_sym = sym_l[0]
                        #     max_name = sym
                        if sym.split('_')[0] == 1:  # The higher symmetry
                            dock_d[sym] = os.path.splitext(input_pdb)[0] + '_flipped_180y'  # .pdb'
                    SDUtils.pickle_object(dock_d, os.path.splitext(dock_instructions[0])[0], protocol=pickle_prot)

                    # update the path in the chain pickle
                    chain_map[design]['pdb2']['path'] = os.path.splitext(input_pdb)[0] + '_flipped_180y.pdb'
                    p.communicate()
                SDUtils.pickle_object(chain_map, 'asu_to_oriented_oligomer_chain_correspondance_vflip_flipped',
                                      protocol=pickle_prot)
                exit('All second oligomer chains flipped')

            with open(args.file, 'r') as f:
                all_design_directories = f.readlines()
                design_d_names = map(str.strip, all_design_directories)
                design_d_names = list(map(os.path.basename, design_d_names))

            if args.mode == 'report':
                rmsd_d = collect_rmsd_calc(design_d_names, location=args.directory)
                report_top_rmsd(rmsd_d)
                all_commands = []
            elif args.mode == 'reference_rmsd':
                all_commands = run_rmsd_calc(design_d_names, args.design_map, args.command_only)
            elif args.mode == 'all_to_all_rmsd':
                all_commands = run_all_to_all_calc(design_d_names, args.design_map, args.command_only)
            elif args.mode == 'cluster_rmsd':
                all_commands = run_cluster_calc(design_d_names, args.design_map, args.command_only)
            elif args.mode == 'all_to_cluster':
                commands1 = run_all_to_all_calc(design_d_names, args.design_map, args.command_only)
                commands2 = run_cluster_calc(design_d_names, args.design_map, args.command_only)
                modified_commands1 = map(subprocess.list2cmdline, zip(repeat('bash'), commands1))
                modified_commands2 = list(map(subprocess.list2cmdline, zip(repeat('bash'), commands2)))
                all_commands = [
                    SDUtils.write_shell_script(cmd1, name='all_to_cluster', out_path=os.path.dirname(commands1[l]),
                                               additional=[modified_commands2[l]])
                    for l, cmd1 in enumerate(modified_commands1)]
            elif args.mode == 'all_rmsd':
                commands1 = run_rmsd_calc(design_d_names, args.design_map, args.command_only)
                commands2 = run_all_to_all_calc(design_d_names, args.design_map, args.command_only)
                commands3 = run_cluster_calc(design_d_names, args.design_map, args.command_only)
                modified_commands1 = map(subprocess.list2cmdline, zip(repeat('bash'), commands1))
                modified_commands2 = list(map(subprocess.list2cmdline, zip(repeat('bash'), commands2)))
                modified_commands3 = list(map(subprocess.list2cmdline, zip(repeat('bash'), commands3)))
                all_commands = [
                    SDUtils.write_shell_script(cmd1, name='rmsd_to_cluster', out_path=os.path.dirname(commands1[l]),
                                               additional=[modified_commands2[l], modified_commands3[l]])
                    for l, cmd1 in enumerate(modified_commands1)]
            else:
                exit('Invalid Input: \'-mode\' must be specified if using design_map!')

            if args.command_only:
                all_command_locations = list(map(os.path.dirname, all_commands))  # TODO remove command distributer naming
                SDUtils.write_commands(all_command_locations, name=args.mode, out_path=args.directory)
        else:
            design_recapitulation(args.file, args.out_path, pdb_dir=args.directory, oligomer=args.oligomer_asu)
    else:
        if args.directory or args.file:
            file_paths, location = SDUtils.collect_designs(files=args.file, directory=args.directory)
        else:
            exit('Specify either a file or a directory to locate the files!')

        new_files = [make_asu(file, out_path=args.out_path) for file in file_paths]

        with open(os.path.join(args.out_path, 'asu_files.txt'), 'w') as f:
            f.write('\n'.join(new_files))
