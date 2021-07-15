import sys
import time
import os
from glob import glob

from SymDesignUtils import start_log

file_types = \
    ['gradients_h_%d.bin', 'ergo_%d.txt', 'learning_rates_J_%d.bin', 'stat_MC_1p_sigma_%d.bin', 'gradients_J_%d.bin',
     'MC_samples_%d.txt', 'stat_MC_2p_sigma_%d.bin', 'learning_rates_h_%d.bin', 'stat_MC_1p_%d.bin', 'overlap_%d.txt',
     'stat_MC_2p_%d.bin', 'overlap_inf_%d.txt', 'parameters_J_%d.bin', 'parameters_h_%d.bin']
final_files = \
    ['stat_MC_2p_final.bin', 'learning_rates_J_final.bin', 'overlap_final.txt', 'stat_MC_1p_sigma_final.bin',
     'ergo_final.txt', 'stat_MC_1p_final.bin', 'learning_rates_h_final.bin', 'stat_MC_2p_sigma_final.bin',
     'gradients_J_final.bin', 'parameters_h_final.bin', 'overlap_inf_final.txt', 'parameters_J_final.bin',
     'MC_samples_final.txt', 'gradients_h_final.bin']
constant_files = \
    ['stat_align_2p.bin', 'stat_align_1p.bin', 'msa_numerical.txt', 'sequence_weights.txt', 'bmdca_params.conf',
     'rel_ent_grad_align_1p.bin', 'bmdca_run.log']
# 'parameters_final.txt',
logger = start_log(name=os.path.basename(__file__).split('.')[0], set_logger_level=True)


def clean_bmdca_directory(bmdca_dir):
    """From a directory with bmDCA output's clean all unnecessary files"""
    logger.info('Starting directory %s' % bmdca_dir)
    check_checkpoint_file = os.path.join(bmdca_dir, 'FinalFileIsCheckpoint*.README')
    final_h_file = os.path.join(bmdca_dir, 'parameters_h_final.bin')
    if glob(check_checkpoint_file):  # we have processed this directory previously and saved a checkpoint as final
        os.system('rm %s' % check_checkpoint_file)  # remove this checkpoint indicator
        if os.path.exists(final_h_file) and \
                int(os.path.getmtime(check_checkpoint_file)) > int(os.path.getmtime(final_h_file)):
            # the final file was made after the checkpoint descriptor was set, this job has finished
            final_file_is_not_checkpoint = True  # real final file
        else:  # the final file is a checkpoint still, and this job isn't done, we could update it
            final_file_is_not_checkpoint = False
    else:  # we haven't processed, there may exist a final file, but we need to check
        if os.path.exists(final_h_file):
            final_file_is_not_checkpoint = True  # real final file
        else:
            final_file_is_not_checkpoint = False

    sorted_checkpoints = sorted(map(int, [os.path.splitext(file)[0].split('_')[-1]
                                          for file in glob(os.path.join(bmdca_dir, 'parameters_h_[0-9]*.bin'))]))
    if sorted_checkpoints:
        checkpoint_iterator = iter(sorted_checkpoints)
    else:
        logger.info('\tNo checkpoint files found')
        return

    if final_file_is_not_checkpoint:
        logger.info('Found final checkpoint')
        number_to_keep = 0
        for file in final_files:
            final_file = os.path.join(bmdca_dir, file)
            if not os.path.exists(final_file) or os.stat(final_file).st_size <= 0:
                logger.info('Final checkpoint is missing: %s' % final_file)
                number_to_keep = next(checkpoint_iterator)
                logger.info('Next checkpoint: %d' % number_to_keep)
                break
    else:
        number_to_keep = next(checkpoint_iterator)
        logger.info('Furthest checkpoint: %d' % number_to_keep)

    start_number = 0
    while number_to_keep != start_number:  # if not final checkpoint, all req. files must be present and contain data
        start_number = number_to_keep
        for file in file_types:
            checkpoint_file = os.path.join(bmdca_dir, file % number_to_keep)
            # if int(time.time()) - int(os.path.getmtime(log_file)) < inactive_time:
            if not os.path.exists(checkpoint_file) or os.stat(checkpoint_file).st_size <= 0:
                logger.info('Checkpoint %d is missing: %s' % (number_to_keep, checkpoint_file))
                number_to_keep = next(checkpoint_iterator)
                logger.info('Next checkpoint: %d' % number_to_keep)
                break

    keep_files = constant_files
    if start_number == 0:
        keep_files += final_files
    else:
        keep_files += [file % number_to_keep for file in file_types]
    logger.info('Keeping files %s' % keep_files)

    for file in os.listdir(bmdca_dir):
        if file not in keep_files:
            remove_file = os.path.join(bmdca_dir, file)
            logger.info('Removing: %s' % remove_file)
            # new_location = os.path.join(bmdca_dir, 'checkpoint_back_up', file)
            # logger.info('New location %s' % new_location)
            # os.system('scp %s %s' % (remove_file, new_location))
            os.system('rm %s' % remove_file)

    if start_number != 0:
        logger.info('Copying checkpoint files from iteration %d to final file position' % number_to_keep)
        os.system('scp %s %s' % (os.path.join(bmdca_dir, 'parameters_h_%d.bin' % number_to_keep),
                                 final_h_file))  # os.path.join(bmdca_dir, 'parameters_h_final.bin')))
        os.system('scp %s %s' % (os.path.join(bmdca_dir, 'parameters_J_%d.bin' % number_to_keep),
                                 os.path.join(bmdca_dir, 'parameters_J_final.bin')))
        time.sleep(5)
        os.system('touch %s' % os.path.join(bmdca_dir, 'FinalFileIsCheckpoint%d.README' % number_to_keep))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit('Usage: python clean_bmDCA.py path/to/master_directory_with_bmDCA_directories')

    # for root, dirs, files in os.walk(sys.argv[1]):  # profile_directory
    profile_directory = sys.argv[1]
    for directory in next(os.walk(profile_directory))[1]:
        # for dir in dirs:
        clean_bmdca_directory(os.path.join(profile_directory, directory))
