import argparse
import os
from glob import glob
from itertools import chain

import DesignDirectory
import PathUtils as PUtils
import SymDesignUtils as SDUtils
from PDB import PDB
from Pose import Model


def create_trajectory(design_directories, name='docking_trajectory', output_dir=os.getcwd()):
    trajectory_model = Model()
    # TODO How to sort?
    for des_dir in design_directories:
        trajectory_model.add_model(merge_pose_pdbs(des_dir))

    return trajectory_model.write(name=name, location=output_dir)


def merge_pose_pdbs(des_dir, frags=True):
    # all_pdbs = SDUtils.get_all_pdb_file_paths(des_dir.composition)

    # pdb_codes = str(os.path.basename(des_dir.composition)).split('_')
    pdb_codes = des_dir.entity_names
    oligomers, taken_chains = {}, []
    for name in pdb_codes:
        name_pdb_file = glob(os.path.join(des_dir.path, name + '_tx_*.pdb'))
        assert len(name_pdb_file) == 1, 'More than one matching file found with %s_tx_*.pdb' % name
        oligomers[name] = PDB(file=name_pdb_file[0])
        oligomers[name].name = name
        oligomers[name].reorder_chains(exclude_chains=taken_chains)
        taken_chains += oligomers[name].chain_id_list
    new_pdb = PDB(atoms=[oligomers[oligomer].atoms for oligomer in oligomers])

    if frags:
        frag_pdbs = glob(os.path.join(des_dir.frags, '*.pdb'))
        frags = []
        for frags in frag_pdbs:
            frag_pdb = PDB.from_file(frags)
            frag_pdb.reorder_chains(exclude_chains=taken_chains)
            taken_chains += frag_pdb.chain_id_list
            frags.append(frag_pdb)
        new_pdb = PDB.from_atoms(list(chain.from_iterable(pdb.atoms for pdb in frags)))

    return new_pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='%s\nCreate multimodel PDBS with multiple docked orientations from '
                                                 '%s output' % (__name__, PUtils.nano))
    parser.add_argument('-d', '--directory', type=str, help='Where is the docked PDB directory located?',
                        default=os.getcwd())
    parser.add_argument('-f', '--file', type=str, help='File with location(s) of %s poses' % PUtils.program_name,
                        default=None)
    parser.add_argument('-s', '--design_string', type=str, help='If pose names are specified by design string instead '
                                                                'of directories, which directory path to '
                                                                'prefix with?\nDefault=None', default=None)

    parser.add_argument('-o', '--out_path', type=str, help='Where should the trajectory file be written?', default=None)
    parser.add_argument('-n', '--name', type=str, help='What is the name of the trajectory? Default=docking_trajectory')
    # parser.add_argument('-s', '--score', type=str, help='Where is the score file located?', default=None)
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out? Default=False')

    args = parser.parse_args()
    # Start logging output
    if args.debug:
        logger = SDUtils.start_log(name=os.path.basename(__file__), level=1)
        SDUtils.set_logging_to_debug()
        logger.debug('Debug mode. Verbose output')
    else:
        logger = SDUtils.start_log(name=os.path.basename(__file__), propagate=True)

    all_poses, location = SDUtils.collect_designs(files=args.file, directory=args.directory)
    assert all_poses, 'No %s directories found within \'%s\'! Please ensure correct location' \
                      % (PUtils.nano.title(), location)

    all_design_directories = DesignDirectory.set_up_directory_objects(all_poses, symmetry=args.design_string)
    logger.info('%d Poses found in \'%s\'' % (len(all_poses), location))
    logger.info('All pose specific logs are located in corresponding directories, ex:\n%s' %
                os.path.join(all_design_directories[0].path, os.path.basename(all_design_directories[0].path) + '.log'))

    # print('\n'.join(des_dir.path for des_dir in all_design_directories))
    create_trajectory(all_design_directories, name=args.name, output_dir=args.out_path)
