import os

import numpy as np


class EulerLookup:
    def __init__(self, scale=3.0):
        nanohedra_dirpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        binary_lookup_table_path = os.path.join(nanohedra_dirpath, 'euler_lookup', 'euler_lookup_40.npz')

        self.eul_lookup_40 = np.load(binary_lookup_table_path)['a']
        self.scale = scale

    @staticmethod
    def get_eulerint10_from_rot_vector(v1_a, v2_a, v3_a):
        """Convert rotation matrix to euler angles in the form of an integer triplet (integer values are degrees
        divided by 10; these become indices for a lookup table)
        """
        tolerance = 1.e-6
        v3_a2 = v3_a[:, 2]

        v3_a2 = np.maximum(-1, v3_a2)
        v3_a2 = np.minimum(1, v3_a2)

        # for the np.where statements below
        third_angle_degenerate = np.logical_or(v3_a2 > 1. - tolerance, v3_a2 < -(1. - tolerance))

        e1_v = np.where(third_angle_degenerate, np.arctan2(v2_a[:, 0], v1_a[:, 0]), np.arctan2(v1_a[:, 2], -v2_a[:, 2]))
        e2_v = np.where(~third_angle_degenerate, np.arccos(v3_a2), 0)
        e2_v = np.where(v3_a2 < -(1. - tolerance), np.pi, e2_v)
        # for third vector, set equal to the arctan of the v3_a array or 0
        e3_v = np.where(~third_angle_degenerate, np.arctan2(v3_a[:, 0], v3_a[:, 1]), 0)

        eulint1 = (np.rint(e1_v * 180. / np.pi * 0.1 * 0.999999) + 36) % 36
        eulint2 = np.rint(e2_v * 180. / np.pi * 0.1 * 0.999999)
        eulint3 = (np.rint(e3_v * 180. / np.pi * 0.1 * 0.999999) + 36) % 36

        return np.column_stack([eulint1, eulint2, eulint3])

    @staticmethod
    def get_eulerint10_from_rot(rot):
        """Convert rotation matrix to euler angles in the form of an integer triplet (integer values are degrees
        divided by 10; these become indices for a lookup table)
        """
        tolerance = 1.e-6
        eulint = np.zeros(3, dtype=int)
        # sets the cross vector, z-coord with a max of 1 and min of -1
        rot[2, 2] = min(rot[2, 2], 1.)
        rot[2, 2] = max(rot[2, 2], -1.)

        # if |rot[2,2]| ~ 1 (1. - tolerance), let the 3rd angle (which becomes degenerate with the 1st) be zero
        if rot[2, 2] > 1. - tolerance:
            e1 = np.arctan2(rot[1, 0], rot[0, 0])  # find the angle of the two guide coordinates by arctan of x coords
            e2 = 0.  # eulint[1] becomes 0
            e3 = 0.
        else:
            if rot[2, 2] < -(1. - tolerance):
                e1 = np.arctan2(rot[1, 0], rot[0, 0])
                e2 = np.pi  # eulint[1] becomes 18
                e3 = 0.
            else:
                e1 = np.arctan2(rot[0, 2], -rot[1, 2])
                e2 = np.arccos(rot[2, 2])
                e3 = np.arctan2(rot[2, 0], rot[2, 1])

        eulint[0] = (np.rint(e1 * 180. / np.pi * 0.1 * 0.999999) + 36) % 36
        eulint[1] = np.rint(e2 * 180. / np.pi * 0.1 * 0.999999)
        eulint[2] = (np.rint(e3 * 180. / np.pi * 0.1 * 0.999999) + 36) % 36

        return eulint

    def get_eulint_from_guides(self, guide_ats):
        """Take a set of guide atoms (3 xyz positions) and return integer indices for the euler angles describing the
        orientations of the axes they form. Note that the positions are in a 3D array. Each guide_ats[i,:,:] is a 3x3
        array with the vectors stored *in columns*, i.e. one vector is in [i,:,j]. Use known scale value to normalize,
        to save repeated sqrt calculations
        """
        # ensure the atoms are passed as an array of 3x3 matrices
        if guide_ats.ndim != 3 or guide_ats.shape[1] != 3 or guide_ats.shape[2] != 3:
            print('ERROR: Guide atom array with wrong dimensions. Calculation failed!!!')

        nfrags = guide_ats.shape[0]
        eulintarray = np.zeros((nfrags, 3), dtype=int)

        # form the 2 difference vectors (N or O - CA), normalize by vector scale, then cross product
        normalization = 1. / self.scale
        for i in range(nfrags):
            # v1 = (guide_ats[i, :, 1] - guide_ats[i, :, 0]) * normalization  # from np.transpose/swapaxes
            # v2 = (guide_ats[i, :, 2] - guide_ats[i, :, 0]) * normalization
            v1 = (guide_ats[i, 1, :] - guide_ats[i, 0, :]) * normalization
            v2 = (guide_ats[i, 2, :] - guide_ats[i, 0, :]) * normalization
            v3 = np.cross(v1, v2)
            rot = np.array([v1, v2, v3])

            # get the euler indices
            eulintarray[i, :] = self.get_eulerint10_from_rot(rot)
        print(eulintarray[:5], eulintarray.dtype, eulintarray.shape)

        # return eulintarray

        # the transpose done in the check_lookup_table is unnecessary if indexed as below
        # for fast array multiplication
        normalization = 1. / self.scale
        v1_a = (guide_ats[:, 1, :] - guide_ats[:, 0, :]) * normalization
        v2_a = (guide_ats[:, 2, :] - guide_ats[:, 0, :]) * normalization
        v3_a = np.cross(v1_a, v2_a)
        eulintarray2 = self.get_eulerint10_from_rot_vector(v1_a, v2_a, v3_a)
        print(eulintarray2[:5], eulintarray2.dtype, eulintarray.shape)
        return eulintarray2

    def check_lookup_table(self, guide_coords1, guide_coords2):
        """Returns a tuple with the index of the first fragment, second fragment, and a bool whether their guide coords
        overlap
        """
        # # guide_list_1_np = np.array(guide_coords1)  # required to take the transpose, could use Fortran order...
        # # guide_list_1_np_t = np.array([atoms_coords_1.T for atoms_coords_1 in guide_list_1_np])
        #
        # # guide_list_2_np = np.array(guide_coords2)  # required to take the transpose
        # # guide_list_2_np_t = np.array([atoms_coords_2.T for atoms_coords_2 in guide_list_2_np])
        #
        # eulintarray1 = self.get_eulint_from_guides(guide_coords1.swapaxes(1, 2))  # swapaxes takes the inner transpose
        # eulintarray2 = self.get_eulint_from_guides(guide_coords2.swapaxes(1, 2))  # swapaxes takes the inner transpose
        eulintarray1 = self.get_eulint_from_guides(guide_coords1)
        eulintarray2 = self.get_eulint_from_guides(guide_coords2)
        print(len(eulintarray2), eulintarray2.shape)
        print(eulintarray2[1, :])
        # check lookup table
        try:
            euler_bool_l = []
            for i in range(len(eulintarray1)):
                for j in range(len(eulintarray2)):
                    (e1, e2, e3) = eulintarray1[i, :]  # .flatten()
                    (f1, f2, f3) = eulintarray2[j, :]  # .flatten()
                    euler_bool_l.append((i, j, self.eul_lookup_40[e1, e2, e3, f1, f2, f3]))

        # return [(i, j) for i in range(len(eulintarray1)) for j in range(len(eulintarray2))
        #         if self.eul_lookup_40[(*eulintarray1[i, :].flatten(), *eulintarray2[j, :].flatten())]]
        except IndexError as e:
            print(e)
            print('i is:', i, 'j is:', j)
            print(eulintarray1[0, :])
