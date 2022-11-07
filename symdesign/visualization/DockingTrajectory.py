import argparse
import os
from glob import glob
from itertools import chain

from symdesign import protocols, utils
from symdesign.utils import path as putils
from symdesign.structure.model import Model, MultiModel


def create_trajectory(design_directories, name='docking_trajectory', output_dir=os.getcwd()):
    trajectory_model = MultiModel()
    # TODO How to sort?
    for des_dir in design_directories:
        trajectory_model.append_model(merge_pose_pdbs(des_dir))

    return trajectory_model.write(name=name, location=output_dir)


def merge_pose_pdbs(des_dir, frags=True):
    # all_pdbs = SDUtils.get_directory_file_paths(des_dir.composition, extension='.pdb')

    # pdb_codes = str(os.path.basename(des_dir.composition)).split('_')
    pdb_codes = des_dir.entity_names
    oligomers, taken_chains = {}, []
    for name in pdb_codes:
        name_pdb_file = glob(os.path.join(des_dir.path, name + '_tx_*.pdb'))
        assert len(name_pdb_file) == 1, 'More than one matching file found with %s_tx_*.pdb' % name
        oligomers[name] = Model.from_file(name_pdb_file[0])
        oligomers[name].name = name
        oligomers[name].rename_chains(exclude_chains=taken_chains)
        taken_chains += oligomers[name].chain_ids
    new_pdb = Model.from_atoms(list(chain.from_iterable(oligomers[oligomer].atoms for oligomer in oligomers)))

    if frags:
        frag_pdbs = glob(os.path.join(des_dir.frags, '*.pdb'))
        frags = []
        for frag_file in frag_pdbs:
            frag_pdb = Model.from_file(frag_file)
            frag_pdb.rename_chains(exclude_chains=taken_chains)
            taken_chains += frag_pdb.chain_ids
            frags.append(frag_pdb)
        new_pdb = Model.from_atoms(list(chain.from_iterable(pdb.atoms for pdb in frags)))

    return new_pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='%s\nCreate multimodel PDBS with multiple docked orientations from '
                                                 '%s output' % (__name__, putils.nanohedra))
    parser.add_argument('-d', '--directory', type=str, help='Where is the docked PDB directory located?',
                        default=os.getcwd())
    parser.add_argument('-f', '--file', type=str, help='File with location(s) of %s poses' % putils.program_name,
                        default=None)
    parser.add_argument('-s', '--design_string', type=str, help='If pose names are specified by design string instead '
                                                                'of directories, which directory path to '
                                                                'prefix with?\nDefault=None', default=None)

    parser.add_argument('-o', '--out_path', type=str, help='Where should the trajectory file be written?', default=None)
    parser.add_argument('-n', '--name', type=str, help='What is the name of the trajectory? Default=docking_trajectory')
    # parser.add_argument('-s', '--score', type=str, help='Where is the score file located?', default=None)
    parser.add_argument('--debug', action='store_true', help='Debug all steps to standard out? Default=False')

    args = parser.parse_args()
    # Start logging output
    if args.debug:
        logger = utils.start_log(name=os.path.basename(__file__), level=1)
        utils.set_logging_to_level()
        logger.debug('Debug mode. Produces verbose output and not written to any .log files')
    else:
        logger = utils.start_log(name=os.path.basename(__file__))

    all_poses, location = utils.collect_designs(files=args.file, directory=args.directory)
    assert all_poses, 'No %s directories found within \'%s\'! Please ensure correct location' \
                      % (putils.nanohedra.title(), location)

    all_design_directories = [protocols.protocols.PoseDirectory.from_nanohedra(design_path, symmetry=args.design_string)
                              for design_path in all_poses]
    logger.info('%d Poses found in \'%s\'' % (len(all_poses), location))
    logger.info('All pose specific logs are located in corresponding directories, ex:\n%s' %
                os.path.join(all_design_directories[0].path, os.path.basename(all_design_directories[0].path) + '.log'))

    # print('\n'.join(des_dir.path for des_dir in all_design_directories))
    create_trajectory(all_design_directories, name=args.name, output_dir=args.out_path)
