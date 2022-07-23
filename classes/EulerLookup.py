from typing import Annotated

import numpy as np

from PathUtils import binary_lookup_table_path
from SymDesignUtils import start_log

# from numba import njit
# from numba.experimental import jitclass
logger = start_log(name=__name__)


# @jitclass
class EulerLookup:
    def __init__(self, scale: float = 3.):
        self.eul_lookup_40 = np.load(binary_lookup_table_path)['a']  # 6-d bool array [[[[[[True, False, ...], ...]]]]]
        self.indices_lens = [0, 0]
        self.scale = scale
        self.normalization = 1. / self.scale
        self.one_tolerance = 1. - 1.e-6
        self.eulint_divisor = 180. / np.pi * 0.1 * self.one_tolerance

    # @staticmethod
    # @njit
    def get_eulerint10_from_rot_vector(self, v1_a: np.ndarray, v2_a: np.ndarray, v3_a: np.ndarray) -> \
            tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert stacked rotation matrices to euler angles in the form of an integer triplet
        (integer values are degrees divided by 10) These become indices for a lookup table
        """
        # tolerance = 1.e-6
        # self.one_tolerance = 1. - 1.e-6
        # set maximum at 1 and minimum at -1 to obey the domain of arccos
        # v3_a2 = v3_a[:, 2]
        v3_a2 = np.maximum(-1, v3_a[:, 2])
        v3_a2 = np.minimum(1, v3_a2)

        # for the np.where statements below use the vector conditional
        third_angle_degenerate = np.logical_or(v3_a2 > self.one_tolerance, v3_a2 < -self.one_tolerance)

        e1_v = np.where(third_angle_degenerate, np.arctan2(v2_a[:, 0], v1_a[:, 0]), np.arctan2(v1_a[:, 2], -v2_a[:, 2]))
        # e2_v = np.where(~third_angle_degenerate, np.arccos(v3_a2), 0)
        e2_v = np.where(v3_a2 < -self.one_tolerance, np.pi, np.where(~third_angle_degenerate, np.arccos(v3_a2), 0))
        e3_v = np.where(~third_angle_degenerate, np.arctan2(v3_a[:, 0], v3_a[:, 1]), 0)

        eulint1 = ((np.rint(e1_v * self.eulint_divisor) + 36) % 36).astype(int)
        eulint2 = np.rint(e2_v * self.eulint_divisor).astype(int)
        eulint3 = ((np.rint(e3_v * self.eulint_divisor) + 36) % 36).astype(int)

        return eulint1, eulint2, eulint3

    # @njit
    def get_eulint_from_guides(self, guide_coords: np.ndarray):
        """Take a set of guide atoms (3 xyz positions) and return integer indices for the euler angles describing the
        orientations of the axes they form. Note that the positions are in a 3D array. Each guide_ats[i,:,:] is a 3x3
        array with the vectors stored *in columns*, i.e. one vector is in [i,:,j]. Use known scale value to normalize,
        to save repeated sqrt calculations
        """
        # for fast array multiplication
        v1_a = (guide_coords[:, 1, :] - guide_coords[:, 0, :]) * self.normalization
        v2_a = (guide_coords[:, 2, :] - guide_coords[:, 0, :]) * self.normalization
        # v3_a = np.cross(v1_a, v2_a)

        return self.get_eulerint10_from_rot_vector(v1_a, v2_a, np.cross(v1_a, v2_a))

    # @njit
    def check_lookup_table(self, guide_coords1: np.ndarray, guide_coords2: np.ndarray):
        """Returns a tuple with the index of the first fragment and second fragment where they overlap
        """
        # ensure the atoms are passed as an array of 3x3 matrices
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
        # indices1 = np.arange(indices1_len)
        # indices2 = np.arange(indices2_len)
        # index_array = np.column_stack([np.repeat(indices1, indices2.shape[0]),
        #                                np.tile(indices2, indices1.shape[0])])

        index_array1 = np.repeat(np.arange(indices1_len), indices2_len)
        index_array2 = np.tile(np.arange(indices2_len), indices1_len)

        # Construct the correctly sized arrays to lookup euler space matching pairs from the all to all guide_coords
        # eulintarray1_1_r = np.repeat(eulintarray1_1, indices2_len)
        # eulintarray1_2_r = np.repeat(eulintarray1_2, indices2_len)
        # eulintarray1_3_r = np.repeat(eulintarray1_3, indices2_len)
        # eulintarray2_1_r = np.tile(eulintarray2_1, indices1_len)
        # eulintarray2_2_r = np.tile(eulintarray2_2, indices1_len)
        # eulintarray2_3_r = np.tile(eulintarray2_3, indices1_len)
        # Check lookup table
        overlap = self.eul_lookup_40[np.repeat(eulintarray1_1, indices2_len),
                                     np.repeat(eulintarray1_2, indices2_len),
                                     np.repeat(eulintarray1_3, indices2_len),
                                     np.tile(eulintarray2_1, indices1_len),
                                     np.tile(eulintarray2_2, indices1_len),
                                     np.tile(eulintarray2_3, indices1_len)]

        return index_array1[overlap], index_array2[overlap]  # these are the overlapping ij pairs


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
