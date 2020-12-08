import os

import numpy as np


class EulerLookup:
    def __init__(self, scale=3.0, binary_lookup_table_path=os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))) + "/euler_lookup_table_40/euler_lookup_40_binary.npy"):
        self.eul_lookup_40 = np.load(binary_lookup_table_path)
        self.scale = scale

    @staticmethod
    def get_eulerint10_from_rot(rot):
        # convert rotation matrix to euler angles in the form of an integer triplet
        # (integer values are degrees divided by 10; these become indices for a lookup table)
        tolerance = 1.e-6
        eulint = np.zeros(3, dtype=int)
        rot[2, 2] = min(rot[2, 2], 1.)
        rot[2, 2] = max(rot[2, 2], -1.)

        # if |rot[2,2]|~1, let the 3rd angle (which becomes degernate with the 1st) be zero
        if rot[2, 2] > 1. - tolerance:
            e3 = 0.
            e1 = np.arctan2(rot[1, 0], rot[0, 0])
            e2 = 0.
        else:
            if rot[2, 2] < -(1. - tolerance):
                e3 = 0.
                e1 = np.arctan2(rot[1, 0], rot[0, 0])
                e2 = np.pi
            else:
                e2 = np.arccos(rot[2, 2])
                e1 = np.arctan2(rot[0, 2], -rot[1, 2])
                e3 = np.arctan2(rot[2, 0], rot[2, 1])

        eulint[0] = (np.rint(e1 * 180. / np.pi * 0.1 * 0.999999) + 36) % 36
        eulint[1] = np.rint(e2 * 180. / np.pi * 0.1 * 0.999999)
        eulint[2] = (np.rint(e3 * 180. / np.pi * 0.1 * 0.999999) + 36) % 36

        return eulint

    def get_eulint_from_guides(self, guide_ats):
        # take a set of guide atoms (3 xyz positions) and return integer indices
        # for the euler angles describing the orientations of the axes they form
        # Note that the positions are in a 3D array. Each guide_ats[i,:,:] is a
        # 3x3 array with the vectors stored *in columns*, i.e. one vector is in [i,:,j]
        # use known scale value to normalize, to save repeated sqrt calculations

        if guide_ats.ndim != 3 or guide_ats.shape[1] != 3 or guide_ats.shape[2] != 3:
            print('ERROR: guide atom array with wrong dimensions')

        nfrags = guide_ats.shape[0]
        rot = np.zeros((3, 3))
        eulintarray = np.zeros((nfrags, 3), dtype=int)

        # form the 2 difference vectors, normalize, then cross product
        for i in range(nfrags):
            v1 = (guide_ats[i, :, 1] - guide_ats[i, :, 0]) * 1. / self.scale
            v2 = (guide_ats[i, :, 2] - guide_ats[i, :, 0]) * 1. / self.scale
            v3 = np.cross(v1, v2)
            rot = np.array([v1, v2, v3])

            # get the euler indices
            eulintarray[i, :] = self.get_eulerint10_from_rot(rot)

        return eulintarray

    def check_lookup_table(self, guide_coords_list1, guide_coords_list2):
        return_tup_list = []

        guide_list_1_np = np.array(guide_coords_list1)
        guide_list_1_np_T = np.array([atoms_coords_1.T for atoms_coords_1 in guide_list_1_np])

        guide_list_2_np = np.array(guide_coords_list2)
        guide_list_2_np_T = np.array([atoms_coords_2.T for atoms_coords_2 in guide_list_2_np])

        eulintarray1 = self.get_eulint_from_guides(guide_list_1_np_T)
        eulintarray2 = self.get_eulint_from_guides(guide_list_2_np_T)

        # check lookup table
        for i in range(len(eulintarray1)):
            for j in range(len(eulintarray2)):
                (e1, e2, e3) = eulintarray1[i, :].flatten()
                (f1, f2, f3) = eulintarray2[j, :].flatten()
                return_tup_list.append((i, j, self.eul_lookup_40[e1, e2, e3, f1, f2, f3]))

        return return_tup_list
