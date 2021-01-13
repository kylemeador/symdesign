import itertools
import math
import multiprocessing as mp
import os
import time
from random import shuffle

from Bio.PDB import *
from Bio.PDB import PDBParser


def guide_atom_rmsd(biopdb_guide_atoms_tup):
    rmsd_thresh = 1.3

    biopdb_1_guide_atoms = biopdb_guide_atoms_tup[0]
    biopdb_2_guide_atoms = biopdb_guide_atoms_tup[1]
    biopdb_1_id = biopdb_1_guide_atoms[0].get_full_id()[0]
    biopdb_2_id = biopdb_2_guide_atoms[0].get_full_id()[0]

    if len(biopdb_1_guide_atoms) == len(biopdb_2_guide_atoms):
        if len(biopdb_1_guide_atoms) != 0:
            # Calculate RMSD
            e1 = (biopdb_1_guide_atoms[0] - biopdb_2_guide_atoms[0]) ** 2
            e2 = (biopdb_1_guide_atoms[1] - biopdb_2_guide_atoms[1]) ** 2
            e3 = (biopdb_1_guide_atoms[2] - biopdb_2_guide_atoms[2]) ** 2
            s = e1 + e2 + e3
            m = s / float(3)
            r = math.sqrt(m)

            if r <= rmsd_thresh:
                return biopdb_1_id, biopdb_2_id, r
            else:
                return None
        else:
            print("\nNO ATOMS TO COMPARE %s %s\n" (biopdb_1_id, biopdb_2_id))
            return -1
    else:
        print("\nLENGTH MISMATCH %s %s\n" (biopdb_1_id, biopdb_2_id))
        return -1


def mp_all_to_all_rmsd(frag_db_dir, thread_count, cluster_size_limit):
    # Cluster Directory Paths
    cluster_dir_paths = []
#    no_go = ['2WW8_12_frag_376_1_vs_1XT5_12_frag_128_1', '2WW8_12_frag_376_1_vs_4W65_12_frag_176_2', '2WW8_12_frag_376_1_vs_1NE9_12_frag_238_2', '2WW8_12_frag_376_1_vs_2WW8_12_frag_376_1', '2WW8_12_frag_376_1_vs_1Q79_12_frag_495_1', '1Q79_12_frag_495_1_vs_1NE9_12_frag_238_2', '1Q79_12_frag_495_1_vs_4W65_12_frag_176_2', '1Q79_12_frag_495_1_vs_1XT5_12_frag_128_1', '1Q79_12_frag_495_1_vs_2WW8_12_frag_376_1', '1Q79_12_frag_495_1_vs_1Q79_12_frag_495_1']
#    no_go= ['1Q79_12_frag_495_1_vs_2WW8_12_frag_376_1']
    for dirpath, dirnames, filenames in os.walk(frag_db_dir):
        if not dirnames:
#	    if dirpath.split('/')[-1] in no_go:
#		print "nogo"
#		continue
#	        print dirpath.split('/')[-1]
#	    else:
            cluster_dir_paths.append(dirpath)

    for cluster_dir in cluster_dir_paths:
        #print "STARTING WITH " + cluster_dir
        #start = time.time()
        # Get Fragment Guide Atoms for all Fragments in Cluster
        frag_db_biopdb_guide_atoms = []
        intfrag_filenames = []
        for root, dirs, files in os.walk(cluster_dir):
            for filename in files:
                if filename.endswith(".pdb"):
                    intfrag_filenames.append(filename)
        # print "GOT FILE NAMES "

        if len(intfrag_filenames) > cluster_size_limit:
            shuffle(intfrag_filenames)
            rand_intfragset_filenames = intfrag_filenames[0:cluster_size_limit]
            # print "CLUSTER SIZE " + str(len(intfrag_filenames)) + " OVER CLUSTERSIZE LIMIT " + str(cluster_size_limit) + " SET CLUSTER SIZE TO: " + str(len(rand_intfragset_filenames))
            for intfrag_filename in rand_intfragset_filenames:
                frag_pdb_name = os.path.splitext(intfrag_filename)[0]
                frag_pdb_biopdb_parser = PDBParser()
                frag_pdb_biopdb = frag_pdb_biopdb_parser.get_structure(frag_pdb_name, cluster_dir + "/" + intfrag_filename)
                guide_atoms = []
                for atom in frag_pdb_biopdb.get_atoms():
                    if atom.get_full_id()[2] == "9":
                        guide_atoms.append(atom)
                if len(guide_atoms) == 3:
                    frag_db_biopdb_guide_atoms.append(guide_atoms)

            # print "GOT GUIDE ATOMS "

        else:
            # print "CLUSTER SIZE " + str(len(intfrag_filenames))
            for intfrag_filename in intfrag_filenames:
                frag_pdb_name = os.path.splitext(intfrag_filename)[0]
                frag_pdb_biopdb_parser = PDBParser()
                frag_pdb_biopdb = frag_pdb_biopdb_parser.get_structure(frag_pdb_name, cluster_dir + "/" + intfrag_filename)
                guide_atoms = []
                for atom in frag_pdb_biopdb.get_atoms():
                    if atom.get_full_id()[2] == "9":
                        guide_atoms.append(atom)
                if len(guide_atoms) == 3:
                    frag_db_biopdb_guide_atoms.append(guide_atoms)

            # print "GOT GUIDE ATOMS "

        # All to All Cluster Fragment Guide Atom RMSD (no redundant calculations) Processes Argument
        if frag_db_biopdb_guide_atoms != list():
            # processes_arg = []
            # for i in range(len(frag_db_biopdb_guide_atoms) - 1):
            #     for j in range(i + 1, len(frag_db_biopdb_guide_atoms)):
            #         processes_arg.append((frag_db_biopdb_guide_atoms[i], frag_db_biopdb_guide_atoms[j]))
	
	    processes_arg = itertools.combinations(frag_db_biopdb_guide_atoms, 2)

            # print "GOT PROCESSES ARGUMENTS "

            p = mp.Pool(processes=thread_count)
            results = p.map(guide_atom_rmsd, processes_arg)
            p.close()
            p.join()
            outfile_path = cluster_dir + "/rand_limitedto_" + str(cluster_size_limit) + "_iter_all_to_all_guide_atom_rmsd.txt"
            outfile = open(outfile_path, "w")
            for result in results:
                if result is not None:
                    outfile.write("%s %s %s\n" %(result[0], result[1], result[2]))
            outfile.close()

        end = time.time()
        #print "DONE WITH " + cluster_dir
        #print end - start


def main():
    frag_db_dir = "/home/kmeador/yeates/interface_extraction/mapped_paired_xtal_frags"
    thread_count = 6
    cluster_size_limit = 15000
    mp_all_to_all_rmsd(frag_db_dir, thread_count, cluster_size_limit)


if __name__ == '__main__':
    main()
