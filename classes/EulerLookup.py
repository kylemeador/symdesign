import os

import numpy as np


class EulerLookup:
    def __init__(self, scale=3.0):
        nanohedra_dirpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        binary_lookup_table_path = nanohedra_dirpath + "/euler_lookup/euler_lookup_40.npz"

        self.eul_lookup_40 = np.load(binary_lookup_table_path)['a']
        self.scale = scale

    @staticmethod
    def get_eulerint10_from_rot(rot):
        """Convert rotation matrix to euler angles in the form of an integer triplet (integer values are degrees
        divided by 10; these become indices for a lookup table)
        """
        tolerance = 1.e-6
        eulint = np.zeros(3, dtype=int)
        rot[2, 2] = min(rot[2, 2], 1.)  # sets the z coord with a max of 1 and min of -1
        rot[2, 2] = max(rot[2, 2], -1.)

        # if |rot[2,2]|~1, let the 3rd angle (which becomes degenerate with the 1st) be zero
        if rot[2, 2] > 1. - tolerance:
            e1 = np.arctan2(rot[1, 0], rot[0, 0])  # find the angle of the two guide coordinates by arctan of x coords
            e2 = 0.
            e3 = 0.
        else:
            if rot[2, 2] < -(1. - tolerance):
                e1 = np.arctan2(rot[1, 0], rot[0, 0])
                e2 = np.pi
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
        for i in range(nfrags):
            v1 = (guide_ats[i, :, 1] - guide_ats[i, :, 0]) * 1. / self.scale
            v2 = (guide_ats[i, :, 2] - guide_ats[i, :, 0]) * 1. / self.scale
            v3 = np.cross(v1, v2)
            rot = np.array([v1, v2, v3])

            # get the euler indices
            eulintarray[i, :] = self.get_eulerint10_from_rot(rot)

        return eulintarray

    def check_lookup_table(self, guide_coords_list1, guide_coords_list2):
        """Returns a tuple with the index of the first fragment, second fragment, and a bool whether their guide coords
        overlap
        """
        guide_list_1_np = np.array(guide_coords_list1)  # required to take the transpose could use Fortan order...
        guide_list_1_np_T = np.array([atoms_coords_1.T for atoms_coords_1 in guide_list_1_np])

        guide_list_2_np = np.array(guide_coords_list2)  # required to take the transpose
        guide_list_2_np_T = np.array([atoms_coords_2.T for atoms_coords_2 in guide_list_2_np])

        eulintarray1 = self.get_eulint_from_guides(guide_list_1_np_T)
        eulintarray2 = self.get_eulint_from_guides(guide_list_2_np_T)

        # check lookup table
        # euler_bool_l = []
        # for i in range(len(eulintarray1)):
        #     for j in range(len(eulintarray2)):
        #         (e1, e2, e3) = eulintarray1[i, :].flatten()
        #         (f1, f2, f3) = eulintarray2[j, :].flatten()
        #         euler_bool_l.append((i, j, self.eul_lookup_40[e1, e2, e3, f1, f2, f3]))

        euler_bool_l = [(i, j) for i in range(len(eulintarray1)) for j in range(len(eulintarray2))
                        if self.eul_lookup_40[*eulintarray1[i, :].flatten(), *eulintarray2[j, :].flatten()]]

        return euler_bool_l
