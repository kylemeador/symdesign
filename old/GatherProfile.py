import argparse
import logging
import os
import sys
from itertools import repeat

from Bio.Data.IUPACData import protein_letters_3to1

import PathUtils as PUtils
import SequenceProfile
import SymDesignUtils
import SymDesignUtils as SDUtils
from PDB import PDB

logging.getLogger().setLevel(logging.DEBUG)


def check_pssm_v_pose(d_dir, pssm, template_residues):
    # Check length for equality before proceeding
    if len(template_residues) != len(pssm):
        logging.warning('%s: The Pose seq and the Pose profile are different length Profile=%d, Pose=%d. '
                        % (d_dir.path, len(template_residues), len(pssm)))
        return False
    pose_res, pssm_res, = {}, {}
    for n in range(len(template_residues)):
        # res = n + SDUtils.index_offset
        pose_res[n] = protein_letters_3to1[template_residues[n].type.title()]
        pssm_res[n] = pssm[n]['type']
        if pssm_res[n] != pose_res[n]:
            logging.warning('%s: The profile and the Pose seq are different. Residue %d, PSSM: %s, POSE: %s. '
                            % (d_dir.path, n, pssm_res[n], pose_res[n]))
            # des_logger.warning()
            # raise SDUtils.DesignError('%s: Pose length is the same, but residues different!' % des_dir.path)
            # rerun = True
            return False

    return True


def check_for_errors(des_dir):
    pose_pssm, template_pdb = None, None
    for file in os.listdir(des_dir.path):
        if file.endswith(PUtils.dssm):
            pose_pssm = SequenceProfile.parse_pssm(os.path.join(des_dir.path, file))
        if file.endswith(PUtils.clean_asu):
            template_pdb = PDB.from_file(os.path.join(des_dir.path, file))

    if pose_pssm and template_pdb:
        template_residues = template_pdb.residues
        pose_correct = check_pssm_v_pose(des_dir, pose_pssm, template_residues)
        return pose_correct
    else:
        raise SymDesignUtils.DesignError('Directory missing crucial files')


# def generate_profile(pdb, des_dir, debug):  # DEPRECIATED
#     # pose_pssm, template_pdb = None, None
#     # for file in os.listdir(des_dir.path):
#     #     if file.endswith('pose.dssm'):
#     #         pose_pssm = SDUtils.parse_pssm(os.path.join(des_dir.path, file))
#     #     if file.endswith(PUtils.clean_asu):
#     #         template_pdb = PDB(file=os.path.join(des_dir.path, file))
#     #
#     # if pose_pssm and template_pdb:
#     #     template_residues = template_pdb.get_residues()
#     #     pose_correct = check_pssm_v_pose(des_dir, pose_pssm, template_residues)
#     #     return pose_correct
#     # else:
#     #     raise SDUtils.DesignError('Directory missing crucial files')
#
#     # Check to see if other poses have collected design sequence info and grab PSSM
#     temp_file = os.path.join(des_dir.composition, PUtils.temp)
#     rerun = False
#     if PUtils.clean_asu not in os.listdir(des_dir.composition):
#         shutil.copy(pdb, des_dir.composition)
#         with open(temp_file, 'w') as tf:
#             tf.write('Still fetching data. Process will resume once data is gathered\n')
#
#         pssm_files, pdb_seq, errors, sequence_file, pssm_process = {}, {}, {}, {}, {}
#         des_logger.debug('Fetching PSSM Files')
#
#         # Check if other design combinations have already collected sequence info about design candidates
#         for name in names:
#             for file in os.listdir(des_dir.sequences):
#                 if fnmatch.fnmatch(file, name + '*'):
#                     if file == name + '.hmm':
#                         pssm_files[name] = os.path.join(des_dir.sequences, file)
#                         des_logger.debug('%s PSSM Files=%s' % (name, pssm_files[name]))
#                         break
#                     elif file == name + '.fasta':
#                         pssm_files[name] = PUtils.temp
#             if name not in pssm_files:
#                 pssm_files[name] = {}
#                 des_logger.debug('%s PSSM File not created' % name)
#
#         # Extract/Format Sequence Information
#         for n, name in enumerate(names):
#             if pssm_files[name] == dict():
#                 des_logger.debug('%s is chain %s in ASU' % (name, names[name](n)))
#                 pdb_seq[name], errors[name] = SDUtils.extract_aa_seq(template_pdb, chain=names[name](n))
#                 # pdb_seq[name], errors[name] = SDUtils.extract_aa_seq(oligomer[name], chain=names[name](n))
#                 des_logger.debug('%s Sequence=%s' % (name, pdb_seq[name]))
#                 if errors[name]:
#                     des_logger.warning('Sequence generation ran into the following residue errors: %s'
#                                        % ', '.join(errors[name]))
#                 sequence_file[name] = SDUtils.write_fasta_file(pdb_seq[name], name, outpath=des_dir.sequences)
#                 if not sequence_file[name]:
#                     des_logger.error('Unable to parse sequence. Check if PDB \'%s\' is valid.' % name)
#                     # logger.critical('Unable to parse sequence. Check if PDB \'%s\' is valid.' % name)
#                     raise DesignDirectory.DesignError('Unable to parse sequence in %s' % des_dir.path)
#             else:
#                 sequence_file[name] = os.path.join(des_dir.sequences, name + '.fasta')
#
#         # Make PSSM of PDB sequence POST-SEQUENCE EXTRACTION
#         for name in names:
#             if pssm_files[name] == dict():
#                 des_logger.info('Generating PSSM file for %s' % name)
#                 pssm_files[name], pssm_process[name] = SequenceProfile.hhblits(sequence_file[name],
#                                                                                outpath=des_dir.sequences)
#                 des_logger.debug('%s seq file: %s' % (name, sequence_file[name]))
#             elif pssm_files[name] == PUtils.temp:
#                 des_logger.info('Waiting for profile generation...')
#                 while True:
#                     time.sleep(20)
#                     if os.path.exists(os.path.join(des_dir.sequences, name + '.hmm')):
#                         pssm_files[name] = os.path.join(des_dir.sequences, name + '.hmm')
#                         pssm_process[name] = done_process
#                         break
#             else:
#                 des_logger.info('Found PSSM file for %s' % name)
#                 pssm_process[name] = done_process
#
#         # Wait for PSSM command to complete
#         for name in names:
#             pssm_process[name].communicate()
#         if os.path.exists(temp_file):
#             os.remove(temp_file)
#
#         # Extract PSSM for each protein and combine into single PSSM
#         pssm_dict = {}
#         for name in names:
#             pssm_dict[name] = SequenceProfile.parse_hhblits_pssm(pssm_files[name])
#         full_pssm = SequenceProfile.combine_pssm([pssm_dict[name] for name in pssm_dict])
#         pssm_file = SequenceProfile.make_pssm_file(full_pssm, PUtils.pssm, outpath=des_dir.composition)
#     else:
#         time.sleep(1)
#         des_logger.info('Waiting for profile generation...')
#         while True:
#             if os.path.exists(temp_file):
#                 time.sleep(20)
#                 continue
#             break
#
#         pssm_file = os.path.join(des_dir.composition, PUtils.pssm)
#         full_pssm = SequenceProfile.parse_pssm(pssm_file)
#
#     # Check length for equality before proceeding
#     if len(template_residues) != len(full_pssm):
#         logging.warning('%s: The Pose seq and the Pose profile are different length Profile=%d, Pose=%d.'
#                         'Generating Rot/Tx specific profile'
#                         % (des_dir.path, len(template_residues), len(full_pssm)))
#         # des_logger
#         rerun = True
#
#     if not rerun:
#         # Check sequence from Pose and PSSM to compare identity before proceeding
#         pose_res, pssm_res, = {}, {}
#         for n in range(len(template_residues)):
#             res = jump - SDUtils.index_offset + n
#             pose_res[n] = protein_letters_3to1[template_residues[res].type.title()]
#             pssm_res[n] = full_pssm[res]['type']
#             if pssm_res[n] != pose_res[n]:
#                 logging.warning('%s: The profile and the Pose seq are different. Residue %d, PSSM: %s, POSE: %s. '
#                                 'Generating Rot/Tx specific profile' % (
#                                 des_dir.path, res, pssm_res[n], pose_res[n]))
#                 # des_logger.warning()
#                 # raise SDUtils.DesignError('%s: Pose length is the same, but residues different!' % des_dir.path)
#                 rerun = True
#                 break
#     raise DesignDirectory.DesignError('%s: Messed up pose')
#     if rerun:
#         pssm_file, full_pssm = SDUtils.gather_profile_info(template_pdb, des_dir, names, des_logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gathers evolutionary profiles of designs.')
    parser.add_argument('-d', '--directory', type=str, help='Directory where SymDock output is located. Default = CWD',
                        default=os.getcwd())
    parser.add_argument('-m', '--multi_processing', action='store_true', help='Should job be run with multiprocessing? '
                                                                              'Default = False')
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out? Default = False')
    args = parser.parse_args()

    # Start logging output
    if args.debug:
        logger = SDUtils.start_log(name=os.path.basename(__file__), level=1)
        SDUtils.set_logging_to_debug()
        logger.debug('Debug mode. Produces verbose output and not written to any .log files')
    else:
        logger = SDUtils.start_log(name=os.path.basename(__file__), propagate=True)
        # warn_logger = SDUtils.start_log(__name__, level=3)
    logger.info('Starting design with options:\n%s' %
                ('\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    all_designs = SDUtils.get_design_directories(args.directory)
    if all_designs == list():
        logger.critical('No SymDock directories found in \'%s\'. Please ensure correct location' % args.directory)
        sys.exit(1)

    bad_designs = []
    if args.multi_processing:
        # calculate the number of threads to use depending on computer resources
        mp_threads = 4
        logger.info('Multiprocessing with %s multiprocessing threads' % str(mp_threads))
        zipped_args = zip(all_designs, repeat(args.debug))
        results, exceptions = SDUtils.mp_starmap(generate_profile, zipped_args, mp_threads)
        if exceptions:
            logger.warning('The following exceptions were thrown. Design for these directories is inaccurate.')
            for exception in exceptions:
                logger.warning(exception)
        for i, good_design in enumerate(results):
            if not good_design:
                bad_designs.append(all_designs[i])
    else:
        logger.info('If single thread processing is taking a while, use -m during submission '
                    '(especially with 100\'s of poses)')

        for des_directory in all_designs:
            good_design = generate_profile(des_directory, args.debug)
            if not good_design:
                bad_designs.append(des_directory)

    if bad_designs:
        logging.critical('%d directories have bad poses, including:' % len(bad_designs))
        for des in bad_designs:
            logging.critical('%s' % des)
        with open(os.path.join(args.directory, 'BAD_DESIGNS.txt'), 'w') as f:
            for des in bad_designs:
                f.write(str(des.path) + '\n')
