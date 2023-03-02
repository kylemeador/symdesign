from __future__ import annotations

from numba import jit
import numpy as np

fragment_metric_template = \
    dict(center_indices=set(),
         total_indices=set(),
         nanohedra_score=0.,
         nanohedra_score_center=0.,
         multiple_fragment_ratio=0.,
         number_residues_fragment_total=0,
         number_residues_fragment_center=0,
         number_fragments_interface=0,
         percent_fragment_helix=0.,
         percent_fragment_strand=0.,
         percent_fragment_coil=0.)


@jit(nopython=True)  # , cache=True)
def calculate_match(coords1: float | np.ndarray = None, coords2: float | np.ndarray = None,
                    coords_rmsd_reference: float | np.ndarray = None) -> float | np.ndarray:
    """Calculate the match score(s) between two sets of coordinates given a reference rmsd

    Args:
        coords1: The first set of coordinates
        coords2: The second set of coordinates
        coords_rmsd_reference: The reference RMSD to compare each pair of coordinates against
    Returns:
        The match score(s)
    """
    # rmsds = _rmsd(coords1, coords2)
    # # Calculate Guide Atom Overlap Z-Value
    # z_values = rmsds / coords_rmsd_reference
    # # filter z_values by passing threshold
    return match_score_from_z_value(_rmsd(coords1, coords2) / coords_rmsd_reference)


@jit(nopython=True)  # , cache=True)
def rmsd_z_score(coords1: float | np.ndarray = None, coords2: float | np.ndarray = None,
                 coords_rmsd_reference: float | np.ndarray = None) -> float | np.ndarray:
    """Calculate the overlap between two sets of coordinates given a reference rmsd

    Args:
        coords1: The first set of coordinates
        coords2: The second set of coordinates
        coords_rmsd_reference: The reference RMSD to compare each pair of coordinates against
    Returns:
        The overlap z-value
    """
    #         max_z_value: The z-score deviation threshold of the overlap to be considered a match
    # Calculate Guide Atom Overlap Z-Value
    return _rmsd(coords1, coords2) / coords_rmsd_reference
    # z_values = _rmsd(coords1, coords2) / coords_rmsd_reference
    # filter z_values by passing threshold
    # return np.where(z_values < max_z_value, z_values, False)


@jit(nopython=True)  # , cache=True)
def _rmsd(coords1: float | np.ndarray = None, coords2: float | np.ndarray = None) -> float | np.ndarray:
    """Calculate the root-mean-square deviation (RMSD). Arguments can be single vectors or array-like

    If calculation is over two sets of numpy.arrays. The first axis (0) contains instances of coordinate sets,
    the second axis (1) contains a set of coordinates, and the third axis (2) contains the xyz coordinates

    Returns:
        The RMSD value(s) which is equal to the length of coords1
    """
    # difference_squared = (coords1 - coords2) ** 2
    # # axis 2(-1) gets the sum of the rows 0[1[2[],2[],2[]], 1[2[],2[],2[]], ...]
    # sum_difference_squared = ((coords1 - coords2) ** 2).sum(axis=-1)
    # # axis 1(-1) gets the mean of the rows 0[1[], 1[], ...]
    # mean_sum_difference_squared = ((coords1 - coords2) ** 2).sum(axis=-1).mean(axis=-1)

    # coord_len, set_len, *_ = coords1.shape[::-1]
    # shape = coords1.shape
    # if len(shape) == 3:
    instance_len, set_len, coord_len = coords1.shape  # shape
    # elif len(shape) == 2:
    #     set_len, coord_len = shape
    # else:
    #     raise ValueError('Input length less than 2 not supported')
    return np.sqrt(((coords1 - coords2) ** 2).sum(axis=-1).sum(axis=-1) / set_len)


@jit(nopython=True)  # , cache=True)
def z_value_from_match_score(match_score: float | np.ndarray) -> float | np.ndarray:
    """Given a match score, convert to a z-value. sqrt(1/match_score - 1)"""
    return np.sqrt(1/match_score - 1)


@jit(nopython=True)  # , cache=True)
def match_score_from_z_value(z_value: float | np.ndarray) -> float | np.ndarray:
    """Return the match score from a fragment z-value -> 1 / (1 + z_value**2). Bounded between 0 and 1"""
    return 1 / (1 + z_value**2)
