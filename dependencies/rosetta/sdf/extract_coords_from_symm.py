import sys
import os
import numpy as np

from PDB import PDB
from Structure import superposition3d


def coords_to_pdb(coords):
    chains = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '=', '_', '+', ';', ':', ',', '<', '.', '>', '/', '?', '|', '[', '{', ']', '}', '`', '~']

    atom_line = "ATOM  {:5d}  CA  GLY {:1s}{:4d}   {:8.3f}{:8.3f}{:8.3f}  1.00 20.00           C\n"
    file_lines = []

# make work for transformation of x vector by y vector. Throw Z away for the 0 dimension point groups
    for i in range(len(coords)):
        temp = []
        for j in coords[i]:
            temp.append(float(coords[i][j]))
        atom_line.format(i, chains[i], i, temp[0], temp[1], temp[3])
        file_lines.append(atom_line)

    return file_lines


if __name__ == '__main__':
    file = sys.argv[1]
    t32_rosetta_file = sys.argv[2]
    t32_orient_file = sys.argv[3]
    with open(file, 'r') as f:
        new_coords = f.readlines()

    coords = []
    for line in new_coords:
        if line[:3] == 'xyz':
            temp_coords = line.strip().replace('+', '').split()
            coords.append([list(map(float, temp_coords[-3].split(','))), list(map(float, temp_coords[-2].split(','))),
                           list(map(float, temp_coords[-1].split(',')))])
    print(coords)
    final_coords = np.array(coords)
    t32_rosetta = PDB.from_file(t32_rosetta_file)
    t32_orient = PDB.from_file(t32_orient_file)
    _, rot, tx, _ = superposition3d(t32_orient.entities[0].get_cb_coords(), t32_rosetta.entities[0].get_cb_coords())
    final_coords = np.matmul(final_coords, np.transpose(rot)) + tx

    # with open(new_file, 'w') as f:
    #     f.write('%s\n' % '\n'.join(' '.join(','.join(coord_triplet) for coord_triplet in coord_group) for coord_group in final_coords.tolist()))
    print('%s\n' % '\n'.join(' '.join(','.join(list(map(str, coord_triplet))) for coord_triplet in coord_group) for coord_group in final_coords.tolist()))
    # final_coords = np.zeros((len(coords), 3))
    # # final_coords = []
    # for i in range(len(coords)):
    #     internal_coords = []
    #     for j in range(len(3)):
    #         coord = coords[i][j].strip().strip('+').split(',')
    #         # new = new.strip('+')
    #         # coord = new.split(',')
    #         # internal_coords.append(coord)
    #         final_coords[i][j] = coord
    # # final_coords.append(internal_coords)
    # new_lines = coords_to_pdb(final_coords)

