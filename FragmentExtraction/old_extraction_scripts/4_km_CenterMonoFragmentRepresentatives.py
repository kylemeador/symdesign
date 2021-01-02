import sys
import os
from PDB import PDB


def main():
    # Directory Containing Structures to Center
    top_representatives_dir = os.path.join(os.getcwd(), 'i_clusters')

    # Directory to Output Centered Structures
    outdir = os.path.join(os.getcwd(), 'i_clusters', 'centered_representatives')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Read In Cluster Representative Fragments and create PDB instance
    # i = 0  # Cluster Representative Type Index Name i
    for root, dirs, files in os.walk(top_representatives_dir):
        for filename in files:
            if filename.endswith("_representative.pdb"):

                pdb = PDB()
                pdb.readfile(top_representatives_dir + "/" + filename, remove_alt_location=True)

                # Get Central Residue (5 Residue Fragment => 3rd Residue) CA Coordinates
                pdb_ca_atoms = pdb.get_CA_atoms()
                central_ca_atom = pdb_ca_atoms[2]
                central_ca_coords = central_ca_atom.coords()

                # Center Such That Central Residue CA is at Origin
                tx = [-central_ca_coords[0], -central_ca_coords[1], -central_ca_coords[2]]
                pdb.translate(tx)

                pdb.write(outdir + "/" + filename[:-18] + '.pdb')


if __name__ == '__main__':
    main()
