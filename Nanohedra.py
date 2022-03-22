import os
import sys
import time
from itertools import product, combinations

import PathUtils as PUtils
from FragDock import nanohedra_dock
# from interface_analysis.Database import FragmentDB
from classes.EulerLookup import EulerLookup
from classes.SymEntry import SymEntry
from PDB import orient_pdb_file
from SymDesignUtils import get_all_pdb_file_paths, start_log, unpickle
from utils.GeneralUtils import write_docking_parameters, get_rotation_step
from utils.CmdLineArgParseUtils import get_docking_parameters, query_mode, postprocess_mode
from utils.NanohedraManualUtils import print_usage

# Copyright 2020 Joshua Laniado and Todd O. Yeates.
__author__ = "Joshua Laniado and Todd O. Yeates"
__copyright__ = "Copyright 2020, Nanohedra"
__version__ = "1.0"

if __name__ == '__main__':
    start_time = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == '-query':
        query_mode(sys.argv)

    elif len(sys.argv) > 1 and sys.argv[1] == '-postprocess':
        postprocess_mode(sys.argv)

    elif len(sys.argv) > 1 and sys.argv[1] == '-dock':

        # Parsing Command Line Input
        sym_entry_number, pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, master_outdir, \
            output_assembly, output_surrounding_uc, min_matched, timer, initial, debug = \
            get_docking_parameters(sys.argv)

        # Master Output Directory and Master Log File
        os.makedirs(master_outdir, exist_ok=True)
        master_log_filepath = os.path.join(master_outdir, PUtils.master_log)
        if debug:
            # Root logs to stream with level debug
            logger = start_log(level=1, set_logger_level=True)
            master_logger, bb_logger = logger, logger
            logger.debug('Debug mode. Verbose output')
        else:
            master_logger = start_log(name=__name__, handler=2, location=master_log_filepath)
        master_logger.info('Nanohedra\nMODE: DOCK\n')

        try:
            # Orient Oligomer Fortran Executable Path
            orient_executable_path = PUtils.orient_exe_path
            orient_assert_error_message = 'Could not locate orient_oligomer executable at: %s\n' \
                                          'Check README file for instructions on how to compile ' \
                                          'orient_oligomer.f' % orient_executable_path
            assert os.path.exists(orient_executable_path), orient_assert_error_message

            # SymEntry Parameters
            sym_entry = SymEntry(sym_entry_number)  # sym_map inclusion?

            # initialize the main Nanohedra log
            write_docking_parameters(pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, sym_entry, master_outdir,
                                     log=master_logger)
            rot_step_deg1, rot_step_deg2 = get_rotation_step(sym_entry, rot_step_deg1, rot_step_deg2)

            # Get PDB1 and PDB2 File paths
            # with open(master_log_filepath, 'a+') as master_log_file:
            if '.pdb' in pdb1_path:  # files are not in pdb_dir, for Nanohedra_wrap generated commands...
                pdb1_filepaths = [pdb1_path]
            else:
                pdb1_filepaths = get_all_pdb_file_paths(pdb1_path)

            if '.pdb' in pdb2_path:
                pdb2_filepaths = [pdb2_path]
            else:
                pdb2_filepaths = get_all_pdb_file_paths(pdb2_path)

            # Orient Input Oligomers to Canonical Orientation
            if sym_entry.group1 == sym_entry.group2:
                oligomer_input = 'Oligomer Input'
                master_logger.info('ORIENTING INPUT OLIGOMER PDB FILES')
            else:
                oligomer_input = 'Oligomer 1 Input'
                master_logger.info('ORIENTING OLIGOMER 1 INPUT PDB FILE(S)')

            oriented_pdb1_outdir = os.path.join(master_outdir, '%s_oriented' % sym_entry.group1)
            if not os.path.exists(oriented_pdb1_outdir):
                os.makedirs(oriented_pdb1_outdir)
            pdb1_oriented_filepaths = [orient_pdb_file(pdb1_path, log=master_logger, sym=sym_entry.group1,
                                                       out_dir=oriented_pdb1_outdir) for pdb1_path in pdb1_filepaths]

            if len(pdb1_oriented_filepaths) == 0:
                master_logger.info('COULD NOT ORIENT %s PDB FILES. CHECK %s%sorient_oligomer_log.txt FOR '
                                   'MORE INFORMATION' % (oligomer_input.upper(), oriented_pdb1_outdir, os.sep))
                master_logger.info('NANOHEDRA DOCKING RUN ENDED\n')
                # master_log_file.close()
                exit(1)
            elif len(pdb1_oriented_filepaths) == 1 and sym_entry.group1 == sym_entry.group2 \
                    and '.pdb' not in pdb2_path:
                master_logger.info('AT LEAST 2 OLIGOMERS ARE REQUIRED WHEN THE 2 OLIGOMERIC COMPONENTS OF '
                                   'A SCM OBEY THE SAME POINT GROUP SYMMETRY (IN THIS CASE: %s)\nHOWEVER '
                                   'ONLY 1 INPUT OLIGOMER PDB FILE COULD BE ORIENTED\nCHECK '
                                   '%s%sorient_oligomer_log.txt FOR MORE INFORMATION\n'
                                   % (sym_entry.group1, oriented_pdb1_outdir, os.sep))
                master_logger.info('NANOHEDRA DOCKING RUN ENDED\n')
                # master_log_file.close()
                exit(1)
            else:
                master_logger.info('Successfully Oriented %d out of the %d Oligomer Input PDB File(s)\n==> %s\n'
                                   % (len(pdb1_oriented_filepaths), len(pdb1_filepaths), oriented_pdb1_outdir))

            if sym_entry.group1 == sym_entry.group2 and '.pdb' not in pdb2_path:
                # in case two paths have same sym, otherwise we need to orient pdb2_filepaths as well
                pdb_filepaths = combinations(pdb1_oriented_filepaths, 2)
            else:
                master_logger.info('ORIENTING OLIGOMER 2 INPUT PDB FILE(S)')
                oriented_pdb2_outdir = os.path.join(master_outdir, '%s_oriented' % sym_entry.group2)
                if not os.path.exists(oriented_pdb2_outdir):
                    os.makedirs(oriented_pdb2_outdir)
                pdb2_oriented_filepaths = [orient_pdb_file(pdb2_path, log=master_logger, sym=sym_entry.group2,
                                                           out_dir=oriented_pdb2_outdir)
                                           for pdb2_path in pdb2_filepaths]

                if len(pdb2_oriented_filepaths) == 0:
                    master_logger.info('COULD NOT ORIENT OLIGOMER 2 INPUT PDB FILE(S) CHECK '
                                       '%s%sorient_oligomer_log.txt FOR MORE INFORMATION'
                                       % (oriented_pdb2_outdir, os.sep))
                    master_logger.info('NANOHEDRA DOCKING RUN ENDED\n')
                    # master_log_file.close()
                    exit(1)

                master_logger.info('Successfully Oriented %d out of the %d Oligomer 2 Input PDB File(s)\n==> %s\n'
                                   % (len(pdb2_oriented_filepaths), len(pdb2_filepaths), oriented_pdb2_outdir))
                pdb_filepaths = product(pdb1_oriented_filepaths, pdb2_oriented_filepaths)

            # Create fragment database for all ijk cluster representatives
            ijk_frag_db = unpickle(PUtils.biological_fragment_db_pickle)
            # Load Euler Lookup table for each instance
            euler_lookup = EulerLookup()
            # ijk_frag_db = FragmentDB()
            #
            # # Get complete IJK fragment representatives database dictionaries
            # ijk_frag_db.get_monofrag_cluster_rep_dict()
            # ijk_frag_db.get_intfrag_cluster_rep_dict()
            # ijk_frag_db.get_intfrag_cluster_info_dict()

            for pdb1_path, pdb2_path in pdb_filepaths:
                pdb1_name = os.path.splitext(os.path.basename(pdb1_path))[0]
                pdb2_name = os.path.splitext(os.path.basename(pdb2_path))[0]
                # with open(master_log_filepath, 'a+') as master_log_file:
                master_logger.info('Docking %s / %s' % (pdb1_name, pdb2_name))

                # Output Directory  # Todo DesignDirectory
                # outdir = os.path.join(master_outdir, '%s_%s' % (pdb1_name, pdb2_name))
                # if not os.path.exists(outdir):
                #     os.makedirs(outdir)

                building_blocks = '%s_%s' % (pdb1_name, pdb2_name)
                # log_file_path = os.path.join(outdir, '%s_log.txt' % building_blocks)
                # if os.path.exists(log_file_path):
                #     resume = True
                # else:
                #     resume = False
                # bb_logger = start_log(name=building_blocks, handler=2, location=log_file_path, format_log=False)
                # bb_logger.info('Found a prior incomplete run! Resuming from last sampled transformation.\n') \
                #     if resume else None

                # Write to Logfile
                # if not resume:
                #     # with open(log_file_path, 'w') as log_file:
                #     bb_logger.info('DOCKING %s TO %s\nOligomer 1 Path: %s\nOligomer 2 Path: %s\n\n'
                #                    % (pdb1_name, pdb2_name, pdb1_path, pdb2_path))

                nanohedra_dock(sym_entry, ijk_frag_db, euler_lookup, master_outdir, pdb1_path, pdb2_path,
                               rot_step_deg1=rot_step_deg1, rot_step_deg2=rot_step_deg2,
                               output_assembly=output_assembly, output_surrounding_uc=output_surrounding_uc,
                               min_matched=min_matched, keep_time=timer)  # log=bb_logger,

                # with open(master_log_filepath, 'a+') as master_log_file:
                master_logger.info('COMPLETE ==> %s' % os.path.join(master_outdir, building_blocks))

            # with open(master_log_filepath, "a+") as master_log_file:
            total_time = time.time() - start_time
            master_logger.info('TOTAL TIME: %s' % str(total_time))
            master_logger.info('COMPLETED FRAGMENT-BASED SYMMETRY DOCKING PROTOCOL\n\nDONE\n')
            exit(0)

        except KeyboardInterrupt:
            # with open(master_log_filepath, "a+") as master_log_file:
            master_logger.info('\nRun Ended By KeyboardInterrupt\n')
            exit(2)
    else:
        print_usage()
