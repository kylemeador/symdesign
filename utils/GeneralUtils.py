import numpy as np


def euclidean_squared_3d(coordinates_1, coordinates_2):
    if len(coordinates_1) != 3 or len(coordinates_2) != 3:
        raise ValueError("len(coordinate list) != 3")

    elif type(coordinates_1) is not list or type(coordinates_2) is not list:
        raise TypeError("input parameters are not of type list")

    else:
        x1, y1, z1 = coordinates_1[0], coordinates_1[1], coordinates_1[2]
        x2, y2, z2 = coordinates_2[0], coordinates_2[1], coordinates_2[2]
        return (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2


def center_of_mass_3d(coordinates):
    n = len(coordinates)
    if n != 0:
        cm = [0. for j in range(3)]
        for i in range(n):
            for j in range(3):
                cm[j] = cm[j] + coordinates[i][j]
        for j in range(3):
            cm[j] = cm[j] / n
        return cm
    else:
        print "ERROR CALCULATING CENTER OF MASS"
        return None


def rot_txint_set_txext_frag_coord_sets(coord_sets, rot_mat=None, internal_tx_vec=None, set_mat=None, ext_tx_vec=None):

    if coord_sets != list():
        # Get the length of each coordinate set
        coord_set_lens = []
        for coord_set in coord_sets:
            coord_set_lens.append(len(coord_set))

        # Stack coordinate set arrays in sequence vertically (row wise)
        coord_sets_vstacked = np.vstack(coord_sets)

        # Rotate stacked coordinates if rotation matrix is provided
        if rot_mat is not None:
            rot_mat_T = np.transpose(rot_mat)
            coord_sets_vstacked = np.matmul(coord_sets_vstacked, rot_mat_T)

        # Translate stacked coordinates if internal translation vector is provided
        if internal_tx_vec is not None:
            coord_sets_vstacked = coord_sets_vstacked + internal_tx_vec

        # Set stacked coordinates if setting matrix is provided
        if set_mat is not None:
            set_mat_T = np.transpose(set_mat)
            coord_sets_vstacked = np.matmul(coord_sets_vstacked, set_mat_T)

        # Translate stacked coordinates if external translation vector is provided
        if ext_tx_vec is not None:
            coord_sets_vstacked = coord_sets_vstacked + ext_tx_vec

        # Slice stacked coordinates back into coordinate sets
        transformed_coord_sets = []
        slice_index_1 = 0
        for coord_set_len in coord_set_lens:
            slice_index_2 = slice_index_1 + coord_set_len

            transformed_coord_sets.append(coord_sets_vstacked[slice_index_1:slice_index_2].tolist())

            slice_index_1 += coord_set_len

        return transformed_coord_sets

    else:
        return []




