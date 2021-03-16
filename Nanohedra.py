import os
import sys
from itertools import product, combinations

import PathUtils as PUtils
from FragDock import nanohedra_dock
from interface_analysis.Database import FragmentDB
from classes.SymEntry import SymEntry
from SymDesignUtils import get_all_pdb_file_paths
from utils.GeneralUtils import write_docking_parameters, get_rotation_step
from utils.PDBUtils import orient_pdb_file
from utils.CmdLineArgParseUtils import get_docking_parameters, query_mode, postprocess_mode
from utils.NanohedraManualUtils import print_usage

# Copyright 2020 Joshua Laniado and Todd O. Yeates.
__author__ = "Joshua Laniado and Todd O. Yeates"
__copyright__ = "Copyright 2020, Nanohedra"
__version__ = "1.0"

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '-query':
        query_mode(sys.argv)

    elif len(sys.argv) > 1 and sys.argv[1] == '-postprocess':
        postprocess_mode(sys.argv)

    elif len(sys.argv) > 1 and sys.argv[1] == '-dock':

        # Parsing Command Line Input
        sym_entry_number, pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, master_outdir, \
            output_assembly, output_surrounding_uc, min_matched, timer, initial = \
            get_docking_parameters(sys.argv)

        # Making Master Output Directory
        if not os.path.exists(master_outdir):
            os.makedirs(master_outdir)
        # Master Log File
        master_log_filepath = os.path.join(master_outdir, PUtils.master_log)

        try:
            # Orient Oligomer Fortran Executable Path
            orient_executable_path = PUtils.orient_exe_path
            orient_assert_error_message = "Could not locate orient_oligomer executable here: %s\n" \
                                          "Check README file for instructions on how to compile " \
                                          "orient_oligomer.f" % orient_executable_path
            assert os.path.exists(orient_executable_path), orient_assert_error_message

            # SymEntry Parameters
            sym_entry = SymEntry(sym_entry_number)

            write_docking_parameters(pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, sym_entry, master_outdir,
                                     master_log_filepath)

            # Get PDB1 and PDB2 File paths
            with open(master_log_filepath, 'a+') as master_log_file:
                if '.pdb' in pdb1_path:  # files are not in pdb_dir, for Nanohedra_wrap generated commands...
                    pdb1_filepaths = [pdb1_path]
                else:
                    pdb1_filepaths = get_all_pdb_file_paths(pdb1_path)

                if '.pdb' in pdb2_path:
                    pdb2_filepaths = [pdb2_path]
                else:
                    pdb2_filepaths = get_all_pdb_file_paths(pdb2_path)

                # Orient Input Oligomers to Canonical Orientation
                if sym_entry.get_group1_sym() == sym_entry.get_group2_sym():
                    oligomer_input = 'Oligomer Input'
                    master_log_file.write("ORIENTING INPUT OLIGOMER PDB FILES\n")
                else:
                    oligomer_input = 'Oligomer 1 Input'
                    master_log_file.write("ORIENTING OLIGOMER 1 INPUT PDB FILE(S)\n")

                oriented_pdb1_outdir = os.path.join(master_outdir, "%s_oriented" % sym_entry.get_group1_sym())
                if not os.path.exists(oriented_pdb1_outdir):
                    os.makedirs(oriented_pdb1_outdir)
                pdb1_oriented_filepaths = [orient_pdb_file(pdb1_path, master_log_filepath,
                                                           sym=sym_entry.get_group1_sym(), out_dir=oriented_pdb1_outdir)
                                           for pdb1_path in pdb1_filepaths]

                if len(pdb1_oriented_filepaths) == 0:
                    master_log_file.write("\nCOULD NOT ORIENT %s PDB FILES\nCHECK %s/orient_oligomer_log.txt FOR "
                                          "MORE INFORMATION\n" % (oligomer_input.upper(), oriented_pdb1_outdir))
                    master_log_file.write("NANOHEDRA DOCKING RUN ENDED\n")
                    master_log_file.close()
                    exit(1)
                elif len(pdb1_oriented_filepaths) == 1 and sym_entry.get_group1_sym() == sym_entry.get_group2_sym() \
                        and '.pdb' not in pdb2_path:
                    master_log_file.write("\nAT LEAST 2 OLIGOMERS ARE REQUIRED WHEN THE 2 OLIGOMERIC COMPONENTS OF "
                                          "A SCM OBEY THE SAME POINT GROUP SYMMETRY (IN THIS CASE: %s)\nHOWEVER "
                                          "ONLY 1 INPUT OLIGOMER PDB FILE COULD BE ORIENTED\nCHECK "
                                          "%s/orient_oligomer_log.txt FOR MORE INFORMATION\n"
                                          % (sym_entry.get_group1_sym(), oriented_pdb1_outdir))
                    master_log_file.write("NANOHEDRA DOCKING RUN ENDED\n")
                    master_log_file.close()
                    exit(1)
                else:
                    master_log_file.write("Successfully Oriented %s out of the %s Oligomer Input PDB Files\n==> %s\n\n"
                                          % (str(len(pdb1_oriented_filepaths)), str(len(pdb1_filepaths)),
                                             oriented_pdb1_outdir))

                if sym_entry.get_group1_sym() == sym_entry.get_group2_sym() and '.pdb' not in pdb2_path:
                    # in case two paths have same sym, otherwise we need to orient pdb2_filepaths as well
                    pdb_filepaths = combinations(pdb1_oriented_filepaths, 2)
                else:
                    master_log_file.write("\nORIENTING OLIGOMER 2 INPUT PDB FILE(S)\n")
                    oriented_pdb2_outdir = os.path.join(master_outdir, "%s_oriented" % sym_entry.get_group2_sym())
                    if not os.path.exists(oriented_pdb2_outdir):
                        os.makedirs(oriented_pdb2_outdir)
                    pdb2_oriented_filepaths = [orient_pdb_file(pdb2_path, master_log_filepath,
                                                               sym=sym_entry.get_group2_sym(),
                                                               out_dir=oriented_pdb2_outdir)
                                               for pdb2_path in pdb2_filepaths]

                    if len(pdb2_oriented_filepaths) == 0:
                        master_log_file.write("\nCOULD NOT ORIENT OLIGOMER 2 INPUT PDB FILE(S)\nCHECK "
                                              "%s/orient_oligomer_log.txt FOR MORE INFORMATION\n"
                                              % oriented_pdb2_outdir)
                        master_log_file.write("NANOHEDRA DOCKING RUN ENDED\n")
                        master_log_file.close()
                        exit(1)

                    master_log_file.write("Successfully Oriented %s out of the %s Oligomer 2 Input PDB File(s)\n==> "
                                          "%s\n\n" % (str(len(pdb2_oriented_filepaths)), str(len(pdb2_filepaths)),
                                                      oriented_pdb2_outdir))
                    pdb_filepaths = product(pdb1_oriented_filepaths, pdb2_oriented_filepaths)

                # Create fragment database for all ijk cluster representatives
                # frag_db = PUtils.frag_directory['biological_interfaces']  # Todo make dynamically start/use all fragDB
                ijk_frag_db = FragmentDB()
                # Get complete IJK fragment representatives database dictionaries
                ijk_frag_db.get_monofrag_cluster_rep_dict()
                ijk_frag_db.get_intfrag_cluster_rep_dict()
                ijk_frag_db.get_intfrag_cluster_info_dict()

            for pdb1_path, pdb2_path in pdb_filepaths:
                pdb1_filename = os.path.splitext(os.path.basename(pdb1_path))[0]
                pdb2_filename = os.path.splitext(os.path.basename(pdb2_path))[0]
                with open(master_log_filepath, "a+") as master_log_file:
                    master_log_file.write("Docking %s / %s \n" % (pdb1_filename, pdb2_filename))

                nanohedra_dock(sym_entry, ijk_frag_db, master_outdir, pdb1_path, pdb2_path,
                               rot_step_deg_pdb1=rot_step_deg1, rot_step_deg_pdb2=rot_step_deg2,
                               output_assembly=output_assembly, output_surrounding_uc=output_surrounding_uc,
                               min_matched=min_matched, keep_time=timer)

            with open(master_log_filepath, "a+") as master_log_file:
                master_log_file.write("\nCOMPLETED FRAGMENT-BASED SYMMETRY DOCKING PROTOCOL\n\nDONE\n")
            exit(0)

        except KeyboardInterrupt:
            with open(master_log_filepath, "a+") as master_log_file:
                master_log_file.write("\nRun Ended By KeyboardInterrupt\n")
            exit(2)
    else:
        print_usage()
