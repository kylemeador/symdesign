import sys
import sys
import warnings
from glob import glob

from Bio.PDB.Atom import PDBConstructionWarning

import SymDesignUtils as SDUtils
from classes.PDB import *
from top_n_all_to_all_docked_poses_irmsd import map_align_interface_chains, standardize_oligomer_chain_lengths, \
    interface_chains_and_resnums

warnings.simplefilter('ignore', PDBConstructionWarning)


################################################### RANKING FUNCTION ###################################################
def res_lev_sum_score_rank(all_design_directories):
    designid_metric_tup_list = [(str(des_dir), SDUtils.gather_fragment_metrics(des_dir, score=True))
                                for des_dir in all_design_directories]
    designid_metric_tup_list_sorted = sorted(designid_metric_tup_list, key=lambda tup: tup[1], reverse=True)
    designid_metric_rank_dict = {d: (m, r) for r, (d, m) in enumerate(designid_metric_tup_list_sorted, 1)}

    return designid_metric_rank_dict
########################################################################################################################


############################################### Crystal VS Docked ######################################################
def get_docked_pdb_pairs(all_design_directories):

    docked_pdb_pairs = []
    for des_dir in all_design_directories:
        docked_pdbs_d = []
        for building_block in os.path.basename(des_dir.building_blocks).split('_'):
            docked_pdb = PDB()
            docked_pdb.readfile(glob(os.path.join(des_dir.path, building_block + '_tx_*.pdb'))[0])
            docked_pdbs_d.append(docked_pdb)
        docked_pdb_pairs.append((str(des_dir), tuple(docked_pdbs_d)))

    return docked_pdb_pairs


def crystal_vs_docked_irmsd(xtal_pdb1, xtal_pdb2, all_design_directories):

    return_list = []

    # get all (docked_pdb1, docked_pdb2) pairs
    docked_pdb_pairs = get_docked_pdb_pairs(all_design_directories)

    for (design_id, (docked_pdb1, docked_pdb2)) in docked_pdb_pairs:
        # standardize oligomer chain lengths such that every 'symmetry related' subunit in an oligomer has the same number
        # of CA atoms and only contains residues (based on residue number) that are present in all 'symmetry related'
        # subunits. Also, standardize oligomer chain lengths such that oligomers being compared have the same number of CA
        # atoms and only contain residues (based on residue number) that are present in all chains of both oligomers.
        stand_docked_pdb1, stand_xtal_pdb_1 = standardize_oligomer_chain_lengths(docked_pdb1, xtal_pdb1)
        stand_docked_pdb2, stand_xtal_pdb_2 = standardize_oligomer_chain_lengths(docked_pdb2, xtal_pdb2)

        # store residue number(s) of amino acid(s) that constitute the interface between xtal_pdb_1 and xtal_pdb_2
        # (i.e. 'reference interface') by their chain id in two dictionaries. One for xtal_pdb_1 and one for xtal_pdb_2.
        # {'chain_id': [residue_number(s)]}
        xtal1_int_chids_resnums_dict, xtal2_int_chids_resnums_dict = interface_chains_and_resnums(stand_xtal_pdb_1,
                                                                                                  stand_xtal_pdb_2,
                                                                                                  cb_distance=9.0)

        # find correct chain mapping between crystal structure and docked pose
        # perform a structural alignment of xtal_pdb_1 onto docked_pdb1 using correct chain mapping
        # transform xtal_pdb_2 using the rotation and translation obtained from the alignment above
        # calculate RMSD between xtal_pdb_2 and docked_pdb2 using only 'reference interface' CA atoms from xtal_pdb_2
        # and corresponding mapped CA atoms in docked_pdb2 ==> interface RMSD or iRMSD
        irmsd = map_align_interface_chains(stand_docked_pdb1, stand_docked_pdb2, stand_xtal_pdb_1, stand_xtal_pdb_2,
                                           xtal1_int_chids_resnums_dict, xtal2_int_chids_resnums_dict,
                                           return_aligned_ref_pdbs=False)

        return_list.append((design_id, irmsd))

    return return_list
########################################################################################################################


def main():
    ############################################## INPUT PARAMETERS ####################################################
    xtal_pdb1_path = sys.argv[1]
    xtal_pdb2_path = sys.argv[2]
    docked_poses_dirpath = sys.argv[3]
    outdir = sys.argv[4]
    ####################################################################################################################

    # read in crystal structure oligomer 1 and oligomer 2 PDB files
    # get the PDB file names without '.pdb' extension
    # create name for combined crystal structure oligomers
    xtal_pdb1 = PDB()
    xtal_pdb1.readfile(xtal_pdb1_path, remove_alt_location=True)
    xtal_pdb1_name = os.path.splitext(os.path.basename(xtal_pdb1_path))[0]

    xtal_pdb2 = PDB()
    xtal_pdb2.readfile(xtal_pdb2_path, remove_alt_location=True)
    xtal_pdb2_name = os.path.splitext(os.path.basename(xtal_pdb2_path))[0]

    xtal_pdb_name = xtal_pdb1_name + "_" + xtal_pdb2_name

    # retrieve Residue Level Summation Score and Scoring Rank for all docked poses
    all_poses, location = SDUtils.collect_directories(docked_poses_dirpath)  # , file=args.file)
    # assert all_poses != list(), print 'No %s directories found within \'%s\'! Please ensure correct location' % \
    #                                   (PUtils.nano.title(), location)
    all_design_directories = SDUtils.set_up_directory_objects(all_poses)  # , symmetry=args.design_string)

    designid_score_dict = res_lev_sum_score_rank(all_design_directories)  # {design_id: (score, score_rank)}

    # align crystal structure to docked poses and calculate interface RMSD
    aligned_xtal_pdbs = crystal_vs_docked_irmsd(xtal_pdb1, xtal_pdb2, all_design_directories)  # [(design_id, aligned_xtal_pdb, irmsd)]

    # sort by RMSD value from lowest to highest
    aligned_xtal_pdbs_sorted = sorted(aligned_xtal_pdbs, key=lambda tup: tup[1], reverse=False)  # was [2], removed files

    # output a PDB file for all of the aligned crystal structures
    # and a text file containing all of the corresponding:
    # iRMSD values, Residue Level Summation Scores and Score Rankings
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfile = open(outdir + "/crystal_vs_docked_irmsd.txt", "w")
    for design_id, irmsd in aligned_xtal_pdbs_sorted:
        # aligned_xtal_pdb.write(outdir + "/%s_AlignedTo_%s.pdb" % (xtal_pdb_name, design_id))
        design_score, design_score_rank = designid_score_dict[design_id]
        out_str = "{:35s} {:8.3f} {:8.3f} {:10d}\n".format(design_id, irmsd, design_score, design_score_rank)
        outfile.write(out_str)
    outfile.close()


if __name__ == "__main__":
    main()
