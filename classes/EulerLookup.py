import os

import numpy as np


class EulerLookup:
    def __init__(self, scale=3.0):
        nanohedra_dirpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        binary_lookup_table_path = os.path.join(nanohedra_dirpath, 'euler_lookup', 'euler_lookup_40.npz')

        self.eul_lookup_40 = np.load(binary_lookup_table_path)['a']  # 6-d bool array [[[[[[True, False, ...], ...]]]]]
        self.scale = scale

    @staticmethod
    def get_eulerint10_from_rot_vector(v1_a, v2_a, v3_a):
        """Convert rotation matrix to euler angles in the form of an integer triplet (integer values are degrees
        divided by 10; these become indices for a lookup table)
        """
        tolerance = 1.e-6
        one_tolerance = 1. - tolerance
        # set maximum at 1 and minimum at -1 to obey the domain of arccos
        # v3_a2 = v3_a[:, 2]
        v3_a2 = np.maximum(-1, v3_a[:, 2])
        v3_a2 = np.minimum(1, v3_a2)

        # for the np.where statements below use the vector conditional
        third_angle_degenerate = np.logical_or(v3_a2 > one_tolerance, v3_a2 < -one_tolerance)

        e1_v = np.where(third_angle_degenerate, np.arctan2(v2_a[:, 0], v1_a[:, 0]), np.arctan2(v1_a[:, 2], -v2_a[:, 2]))
        e2_v = np.where(~third_angle_degenerate, np.arccos(v3_a2), 0)
        e2_v = np.where(v3_a2 < -one_tolerance, np.pi, e2_v)
        e3_v = np.where(~third_angle_degenerate, np.arctan2(v3_a[:, 0], v3_a[:, 1]), 0)

        eulint1 = ((np.rint(e1_v * 180. / np.pi * 0.1 * 0.999999) + 36) % 36).astype(int)
        eulint2 = np.rint(e2_v * 180. / np.pi * 0.1 * 0.999999).astype(int)
        eulint3 = ((np.rint(e3_v * 180. / np.pi * 0.1 * 0.999999) + 36) % 36).astype(int)

        return eulint1, eulint2, eulint3

    def get_eulint_from_guides(self, guide_ats):
        """Take a set of guide atoms (3 xyz positions) and return integer indices for the euler angles describing the
        orientations of the axes they form. Note that the positions are in a 3D array. Each guide_ats[i,:,:] is a 3x3
        array with the vectors stored *in columns*, i.e. one vector is in [i,:,j]. Use known scale value to normalize,
        to save repeated sqrt calculations
        """
        # ensure the atoms are passed as an array of 3x3 matrices
        if guide_ats.ndim != 3 or guide_ats.shape[1] != 3 or guide_ats.shape[2] != 3:
            print('ERROR: Guide atom array with wrong dimensions. Calculation failed!!!')  # Todo logger

        # for fast array multiplication
        normalization = 1. / self.scale
        v1_a = (guide_ats[:, 1, :] - guide_ats[:, 0, :]) * normalization
        v2_a = (guide_ats[:, 2, :] - guide_ats[:, 0, :]) * normalization
        v3_a = np.cross(v1_a, v2_a)
        return self.get_eulerint10_from_rot_vector(v1_a, v2_a, v3_a)

    def check_lookup_table(self, guide_coords1, guide_coords2):
        """Returns a tuple with the index of the first fragment, second fragment, and a bool whether their guide coords
        overlap
        """
        eulintarray1_1, eulintarray1_2, eulintarray1_3 = self.get_eulint_from_guides(guide_coords1)
        eulintarray2_1, eulintarray2_2, eulintarray2_3 = self.get_eulint_from_guides(guide_coords2)

        indices1 = np.arange(len(guide_coords1))
        indices2 = np.arange(len(guide_coords2))
        # index_array = np.column_stack([np.repeat(indices1, indices2.shape[0]),
        #                                np.tile(indices2, indices1.shape[0])])
        index_array1 = np.repeat(indices1, indices2.shape[0])
        index_array2 = np.tile(indices2, indices1.shape[0])

        # Construct the correctly sized arrays to lookup euler space matching pairs from the all to all guide_coords
        eulintarray1_1_r = np.repeat(eulintarray1_1, indices2.shape[0])
        eulintarray1_2_r = np.repeat(eulintarray1_2, indices2.shape[0])
        eulintarray1_3_r = np.repeat(eulintarray1_3, indices2.shape[0])
        eulintarray2_1_r = np.tile(eulintarray2_1, indices1.shape[0])
        eulintarray2_2_r = np.tile(eulintarray2_2, indices1.shape[0])
        eulintarray2_3_r = np.tile(eulintarray2_3, indices1.shape[0])
        # check lookup table
        overlap = self.eul_lookup_40[eulintarray1_1_r, eulintarray1_2_r, eulintarray1_3_r,
                                     eulintarray2_1_r, eulintarray2_2_r, eulintarray2_3_r]

        return index_array1[overlap], index_array2[overlap]  # these are the overlapping ij pairs
