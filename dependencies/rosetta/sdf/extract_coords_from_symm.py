import sys

import numpy as np

from symdesign.structure.model import Model
from symdesign.structure.coordinates import superposition3d


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
    if len(sys.argv) != 4:
        print(f"USAGE: python {__file__} "
              "T32.sdf T32.pdb(in-rosetta-orientation) T32-canonical.pdb(in-canonical-orientation)\n\n"
              "The files should have the same exact structure present and in the same order")
        sys.exit(1)

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
    print(f"Found coords:\n{coords}")
    final_coords = np.array(coords)
    t32_rosetta = Model.from_file(t32_rosetta_file)
    t32_orient = Model.from_file(t32_orient_file)
    _, rot, tx = superposition3d(t32_orient.chains[0].cb_coords, t32_rosetta.chains[0].cb_coords)
    final_coords = np.matmul(final_coords, np.transpose(rot)) + tx

    print('%s\n' % '\n'.join(' '.join(','.join(list(map(str, coord_triplet))) for coord_triplet in coord_group) for coord_group in final_coords.tolist()))
