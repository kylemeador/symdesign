from PDB import *
import sys
import os


def main():

    if len(sys.argv) != 3:
        print "USAGE: python orient.py symmetry_type pdb_directory_path"

    else:
        symm = sys.argv[1]
        pdb_dir_path = sys.argv[2]

        orient_dir = "/Users/jlaniado/Desktop/software/orient"

        out_dir = pdb_dir_path

        for root, dirs, files in os.walk(pdb_dir_path):
            for filename in files:
                if ".pdb" in filename:
                    pdb_path = pdb_dir_path + "/" + filename
                    pdb = PDB()
                    pdb.readfile(pdb_path, remove_alt_location=True)
                    oriented_pdb = pdb.orient(symm, orient_dir, generate_oriented_pdb=False)
                    oriented_pdb.write(out_dir + "/" + os.path.splitext(filename)[0] + "_oriented.pdb")


main()

