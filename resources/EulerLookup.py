from typing import Annotated

import numpy as np

from utils.path import binary_lookup_table_path
from utils import start_log

# from numba import njit
# from numba.experimental import jitclass
logger = start_log(name=__name__)


# @jitclass
class EulerLookup:
    def __init__(self, scale: float = 3.):
        # 6-d bool array [[[[[[True, False, ...], ...]]]]] with shape (37, 19, 37, 37, 19, 37)
        self.eul_lookup_40 = np.load(binary_lookup_table_path)['a']
        """Indicates whether two sets of triplet integer values for each Euler angle are within an acceptable angular 
        offset. Acceptable offset is approximately +/- 40 degrees, or +/- 3 integers in one of the rotation angles and
        a couple of integers in another i.e.
        eul_lookup_40[1,5,1,1,8,1] -> True
        eul_lookup_40[1,5,1,1,9,1] -> False
        eul_lookup_40[1,5,1,1,7,4] -> True
        KM doesn't completely know how Todd made these
        """
        self.indices_lens = [0, 0]
        self.normalization = 1. / scale
        self.one_tolerance = 1. - 1.e-6
        self.eulint_divisor = 180. / np.pi * 0.1 * self.one_tolerance

    # @njit
    def get_eulint_from_guides_as_array(self, guide_coords: np.ndarray) -> np.ndarray:
        """Take a set of guide atoms (3 xyz positions) and return integer indices for the euler angles describing the
        orientations of the axes they form. Note that the positions are in a 3D array. Each guide_ats[i,:,:] is a 3x3
        array with the vectors stored *in columns*, i.e. one vector is in [i,:,j]. Use known scale value to normalize,
        to save repeated sqrt calculations
        """
        # Todo Alternative
        e_array = np.zeros(guide_coords.shape)
        e_array[:, :2, :] = (guide_coords[:, 1:, :] - guide_coords[:, :1, :]) * self.normalization
        e_array[:, 2, :] = np.cross(e_array[:, 0], e_array[:, 1])
        # Bound by a min of -1 and max of 1 as arccos is valid in the domain of [1 to -1]
        e_array[:, 2, 2] = np.minimum(1, e_array[:, 2, 2])
        e_array[:, 2, 2] = np.maximum(-1, e_array[:, 2, 2])
        third_angle_degenerate = np.abs(e_array[:, 2, 2]) > self.one_tolerance
        # Put the results in the first position of every instances first vector
        # Second and third position are disregarded
        e_array[:, 0, 0] = np.where(third_angle_degenerate,
                                    np.arctan2(*e_array[:, :2, 0].T),
                                    np.arctan2(e_array[:, 0, 2], -e_array[:, 1, 2]))
        # Put the results in the first position of every instances second vector
        # arccos returns values of [0 to pi]
        e_array[:, 1, 0] = np.arccos(e_array[:, 2, 2])
        # Put the results in the first position of every instances third vector
        e_array[:, 2, 0] = np.where(third_angle_degenerate, 0, np.arctan2(*e_array[:, 2, :2].T))
        # Values in range (-18 to 18) (not inclusive) v. Then add 36 and divide by 36 to get the abs
        np.rint(e_array * self.eulint_divisor, out=e_array)
        e_array[:, [0, 2]] += 36
        e_array[:, [0, 2]] %= 36
        # # This way allows unpacking for check_lookup_table()
        # return e_array[:, :, 0].T.astype(int)
        # This returns in the intuitive column orientation with "C" array ordering
        return e_array[:, :, 0].astype(int)
        # Todo Alternative

    # @njit
    def get_eulint_from_guides(self, guide_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Take a set of guide atoms (3 xyz positions) and return integer indices for the euler angles describing the
        orientations of the axes they form. Note that the positions are in a 3D array. Each guide_ats[i,:,:] is a 3x3
        array with the vectors stored *in columns*, i.e. one vector is in [i,:,j]. Use known scale value to normalize,
        to save repeated sqrt calculations
        """
        """
        v1_a: An array of vectors containing the first vector which is orthogonal to v2_a (canonically on x)
        v2_a: An array of vectors containing the second vector which is orthogonal to v1_a (canonically on y)
        v3_a: An array of vectors containing the third vector which is the cross product of v1_a and v2_a
        """
        v1_a = (guide_coords[:, 1, :] - guide_coords[:, 0, :]) * self.normalization
        v2_a = (guide_coords[:, 2, :] - guide_coords[:, 0, :]) * self.normalization
        v3_a = np.cross(v1_a, v2_a)

        """Convert rotation matrix to euler angles in the form of an integer triplet (integer values are degrees
        divided by 10; these become indices for a lookup table)
        """
        # Bound by a min of -1 and max of 1 as arccos is valid in the domain of [1 to -1]
        e2_v = np.minimum(1, v3_a[:, 2])
        np.maximum(-1, e2_v, out=e2_v)

        # Check if the third angle is degenerate
        third_angle_degenerate = np.abs(e2_v) > self.one_tolerance
        e1_v = np.where(third_angle_degenerate,
                        np.arctan2(v2_a[:, 0], v1_a[:, 0]),
                        np.arctan2(v1_a[:, 2], -v2_a[:, 2]))

        # arccos returns values of [0 to pi]
        np.arccos(e2_v, out=e2_v)
        e3_v = np.where(third_angle_degenerate, 0, np.arctan2(*v3_a[:, :2].T))

        # np.floor(e1_v*self.eulint_divisor + .5, out=e1_v)  # More accurate
        np.rint(e1_v * self.eulint_divisor, out=e1_v)
        e1_v += 36
        e1_v %= 36
        # np.floor(e3_v*self.eulint_divisor + .5, out=e3_v)  # More accurate
        np.rint(e3_v * self.eulint_divisor, out=e3_v)
        e3_v += 36
        e3_v %= 36
        # Values in range (0 to 18) (not inclusive)
        # np.floor(e2_v*self.eulint_divisor + .5, out=e2_v)  # More accurate
        np.rint(e2_v * self.eulint_divisor, out=e2_v)

        return e1_v.astype(int), e2_v.astype(int), e3_v.astype(int)

    # @njit
    def lookup_by_euler_integers(self, *args: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns a tuple with the index of the first fragment and second fragment where they overlap
        """
        # Unpack each of the integer arrays
        eulintarray1_1, eulintarray1_2, eulintarray1_3, eulintarray2_1, eulintarray2_2, eulintarray2_3 = args

        indices1_len, indices2_len = eulintarray1_1.shape[0], eulintarray2_1.shape[0]

        index_array1 = np.repeat(np.arange(indices1_len), indices2_len)
        index_array2 = np.tile(np.arange(indices2_len), indices1_len)

        # Construct the correctly sized arrays to lookup euler space matching pairs from the all to all guide_coords
        # there may be some solution where numpy.meshgrid is used to broadcast the euler ints
        # check lookup table
        # start = time.time()
        overlap = self.eul_lookup_40[np.repeat(eulintarray1_1, indices2_len),
                                     np.repeat(eulintarray1_2, indices2_len),
                                     np.repeat(eulintarray1_3, indices2_len),
                                     np.tile(eulintarray2_1, indices1_len),
                                     np.tile(eulintarray2_2, indices1_len),
                                     np.tile(eulintarray2_3, indices1_len)]
        # logger.debug(f'took {time.time() - start:8f}s')
        # overlap = self.eul_lookup_40[*stacked_eulintarray1.T,
        #                              stacked_eulintarray2.T]
        # there may be some solution where numpy.ix_ is used to broadcast the index operation

        return index_array1[overlap], index_array2[overlap]  # these are the overlapping ij pairs

    # @njit
    def lookup_by_euler_integers_as_array(self, eulintarray1: np.ndarray, eulintarray2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns a tuple with the index of the first fragment and second fragment where they overlap
        """
        indices1_len, indices2_len = eulintarray1.shape[0], eulintarray2.shape[0]

        index_array1 = np.repeat(np.arange(indices1_len), indices2_len)
        index_array2 = np.tile(np.arange(indices2_len), indices1_len)

        # Construct the correctly sized arrays to lookup euler space matching pairs from the all to all guide_coords
        # there may be some solution where numpy.meshgrid is used to broadcast the euler ints
        # stacked_eulintarray1 = np.tile(eulintarray1, (indices2_len, 1))
        # stacked_eulintarray2 = np.tile(eulintarray2, (indices1_len, 1))
        # # check lookup table
        # overlap = self.eul_lookup_40[stacked_eulintarray1[:, 0],
        #                              stacked_eulintarray1[:, 1],
        #                              stacked_eulintarray1[:, 2],
        #                              stacked_eulintarray2[:, 0],
        #                              stacked_eulintarray2[:, 1],
        #                              stacked_eulintarray2[:, 2]]
        # Transpose the intarray to have each of the columns unpacked as individual arrays
        eulintarray1_1, eulintarray1_2, eulintarray1_3 = eulintarray1.T
        eulintarray2_1, eulintarray2_2, eulintarray2_3 = eulintarray2.T

        # Construct the correctly sized arrays to lookup euler space matching pairs from the all to all guide_coords
        # there may be some solution where numpy.meshgrid is used to broadcast the euler ints
        # check lookup table
        # start = time.time()
        overlap = self.eul_lookup_40[np.repeat(eulintarray1_1, indices2_len),
                                     np.repeat(eulintarray1_2, indices2_len),
                                     np.repeat(eulintarray1_3, indices2_len),
                                     np.tile(eulintarray2_1, indices1_len),
                                     np.tile(eulintarray2_2, indices1_len),
                                     np.tile(eulintarray2_3, indices1_len)]
        # logger.debug(f'took {time.time() - start:8f}s')
        # overlap = self.eul_lookup_40[*stacked_eulintarray1.T,
        #                              stacked_eulintarray2.T]
        # there may be some solution where numpy.ix_ is used to broadcast the index operation

        return index_array1[overlap], index_array2[overlap]  # these are the overlapping ij pairs

    def check_lookup_table(self, guide_coords1: np.ndarray, guide_coords2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        #                    return_bool: bool = False
        """Returns a tuple with the index of the first fragment and second fragment where they overlap
        """
        # ensure the atoms are passed as an array of (n, 3x3) matrices
        try:
            for idx, guide_coords in enumerate([guide_coords1, guide_coords2]):
                indices_len, *remainder = guide_coords.shape
                if remainder != [3, 3]:
                    logger.error(f'ERROR: guide coordinate array with wrong dimensions. '
                                 f'{guide_coords.shape} != (n, 3, 3)')
                    return np.array([]), np.array([])
                self.indices_lens[idx] = indices_len
        except (AttributeError, ValueError):  # guide_coords are the wrong format or the shape couldn't be unpacked
            logger.error(f'ERROR: guide coordinate array wrong type {type(guide_coords).__name__} != (n, 3, 3)')
            return np.array([]), np.array([])

        eulintarray1_1, eulintarray1_2, eulintarray1_3 = self.get_eulint_from_guides(guide_coords1)
        eulintarray2_1, eulintarray2_2, eulintarray2_3 = self.get_eulint_from_guides(guide_coords2)

        indices1_len, indices2_len = self.indices_lens
        index_array1 = np.repeat(np.arange(indices1_len), indices2_len)
        index_array2 = np.tile(np.arange(indices2_len), indices1_len)

        # Construct the correctly sized arrays to lookup euler space matching pairs from the all to all guide_coords
        # eulintarray1_1_r = np.repeat(eulintarray1_1, indices2_len)
        # eulintarray1_2_r = np.repeat(eulintarray1_2, indices2_len)
        # eulintarray1_3_r = np.repeat(eulintarray1_3, indices2_len)
        # eulintarray2_1_r = np.tile(eulintarray2_1, indices1_len)
        # eulintarray2_2_r = np.tile(eulintarray2_2, indices1_len)
        # eulintarray2_3_r = np.tile(eulintarray2_3, indices1_len)
        # check lookup table
        overlap = self.eul_lookup_40[np.repeat(eulintarray1_1, indices2_len),
                                     np.repeat(eulintarray1_2, indices2_len),
                                     np.repeat(eulintarray1_3, indices2_len),
                                     np.tile(eulintarray2_1, indices1_len),
                                     np.tile(eulintarray2_2, indices1_len),
                                     np.tile(eulintarray2_3, indices1_len)]
        # if return_bool:
        #     return overlap
        # else:
        return index_array1[overlap], index_array2[overlap]  # these are the overlapping ij pairs


class EulerLookupV1:
    def __init__(self, scale=3.0):
        self.eul_lookup_40 = np.load(binary_lookup_table_path)['a']  # 6-d bool array [[[[[[True, False, ...], ...]]]]]
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
            print ('ERROR: guide atom array with wrong dimensions')

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
                if self.eul_lookup_40[e1, e2, e3, f1, f2, f3]:
                    return_tup_list.append((i, j))

        return map(np.array, zip(*return_tup_list))


class EulerLookupFactory:
    """Return an EulerLookup instance by calling the Factory instance

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer
    """

    def __init__(self, **kwargs):
        self._lookup_tables = {}

    def __call__(self, **kwargs) -> EulerLookup:
        """Return the specified EulerLookup object singleton

        Returns:
            The instance of the specified EulerLookup
        """
        lookup = self._lookup_tables.get('euler')
        if lookup:
            return lookup
        else:
            logger.info(f'Initializing {EulerLookup.__name__}')
            self._lookup_tables['euler'] = EulerLookup(**kwargs)

        return self._lookup_tables['euler']

    def get(self, **kwargs) -> EulerLookup:
        """Return the specified EulerLookup object singleton

        Returns:
            The instance of the specified EulerLookup
        """
        return self.__call__(**kwargs)


euler_factory: Annotated[EulerLookupFactory,
                         'Calling this factory method returns the single instance of the EulerLookup class'] = \
    EulerLookupFactory()
"""Calling this factory method returns the single instance of the EulerLookup class"""