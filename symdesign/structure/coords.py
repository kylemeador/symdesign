from __future__ import annotations

from typing import Sequence, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

# from symdesign import utils
from symdesign.utils.symmetry import identity_matrix


class Coords:
    """Responsible for handling StructureBase coordinates by storing in a numpy.ndarray with shape (n, 3) where n is the
     number of atoms in the structure and the 3 dimensions represent x, y, and z coordinates

    Args:
        coords: The coordinates to store. If none are passed an empty container will be generated
    """
    coords: np.ndarray

    def __init__(self, coords: np.ndarray | list[list[float]] = None):
        if coords is None:
            self.coords = np.array([])
        elif not isinstance(coords, (np.ndarray, list)):
            raise TypeError(f'Can\'t initialize {type(self).__name__} with {type(coords).__name__}. Type must be a '
                            f'numpy.ndarray of float with shape (n, 3) or list[list[float]]')
        else:
            self.coords = np.array(coords, np.float_)

    def delete(self, indices: Sequence[int]):
        """Delete coordinates from the Coords container

        Args:
            indices: The indices to delete from the Coords array
        Sets:
            self.coords = numpy.delete(self.coords, indices)
        """
        self.coords = np.delete(self.coords, indices, axis=0)

    def insert(self, at: int, new_coords: np.ndarray | list[list[float]]):
        """Insert additional coordinates into the Coords container

        Args:
            at: The index to perform the insert at
            new_coords: The coords to include into Coords
        Sets:
            self.coords = numpy.concatenate(self.coords[:at] + new_coords + self.coords[at:])
        """
        self.coords = np.concatenate((self.coords[:at], new_coords, self.coords[at:]))

    def append(self, new_coords: np.ndarray | list[list[float]]):
        """Append additional coordinates into the Coords container

        Args:
            new_coords: The coords to include into Coords
        Sets:
            self.coords = numpy.concatenate(self.coords[:at] + new_coords + self.coords[at:])
        """
        self.coords = np.concatenate((self.coords, new_coords))

    def replace(self, indices: Sequence[int], new_coords: np.ndarray | list[list[float]]):
        """Replace existing coordinates in the Coords container with new coordinates

        Args:
            indices: The indices to delete from the Coords array
            new_coords: The coordinate values to replace in Coords
        Sets:
            self.coords[indices] = new_coords
        """
        try:
            self.coords[indices] = new_coords
        except ValueError as error:  # they are probably different lengths or another numpy indexing/setting issue
            if self.coords.shape[0] == 0:  # there are no coords, lets use set mechanism
                self.coords = new_coords
            else:
                raise ValueError(f'The new_coords are not the same shape as the selected indices: {error}')

    def set(self, coords: np.ndarray | list[list[float]]):
        """Set self.coords to the provided coordinates

        Args:
            coords: The coordinate values to set
        Sets:
            self.coords = coords
        """
        self.coords = coords

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __iter__(self) -> list[float, float, float]:
        yield from self.coords.tolist()

    def __copy__(self) -> Coords:  # -> Self Todo python3.11
        cls = self.__class__
        other = cls.__new__(cls)
        # other.__dict__.update(self.__dict__)
        other.coords = self.coords.copy()
        return other

    copy = __copy__


def superposition3d(fixed_coords: np.ndarray, moving_coords: np.ndarray, a_weights: np.ndarray = None,
                    quaternion: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """Takes two xyz coordinate sets (same length), and attempts to superimpose them using rotations, translations,
    and (optionally) rescale operations to minimize the root mean squared distance (RMSD) between them. The found
    transformation operations should be applied to the "moving_coords" to place them in the setting of the fixed_coords

    This function implements a more general variant of the method from:
    R. Diamond, (1988) "A Note on the Rotational Superposition Problem", Acta Cryst. A44, pp. 211-216
    This version has been augmented slightly. The version in the original paper only considers rotation and translation
    and does not allow the coordinates of either object to be rescaled (multiplication by a scalar).
    (Additional documentation can be found at https://pypi.org/project/superpose3d/ )

    The quaternion_matrix has the last entry storing cos(θ/2) (where θ is the rotation angle). The first 3 entries
    form a vector (of length sin(θ/2)), pointing along the axis of rotation.
    Details: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    MIT License. Copyright (c) 2016, Andrew Jewett
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
    Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
    OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Args:
        fixed_coords: The coordinates for the 'frozen' object
        moving_coords: The coordinates for the 'mobile' object
        quaternion: Whether to report the rotation angle and axis in Scipy.Rotation quaternion format
    Raises:
        AssertionError: If coordinates are not the same length
    Returns:
        rmsd, rotation/quaternion_matrix, translation_vector
    """
    number_of_points = fixed_coords.shape[0]
    if number_of_points != moving_coords.shape[0]:
        raise ValueError(f'{superposition3d.__name__}: Inputs should have the same size. '
                         f'Input 1={number_of_points}, 2={moving_coords.shape[0]}')

    # convert weights into array
    # if a_weights is None or len(a_weights) == 0:
    # a_weights = np.full((number_of_points, 1), 1.)
    # sum_weights = float(number_of_points)
    # else:  # reshape a_eights so multiplications are done column-wise
    #     a_weights = np.array(a_weights).reshape(number_of_points, 1)
    #     sum_weights = np.sum(a_weights, axis=0)

    # Find the center of mass of each object:
    center_of_mass_fixed = fixed_coords.sum(axis=0)
    center_of_mass_moving = moving_coords.sum(axis=0)

    # Subtract the centers-of-mass from the original coordinates for each object
    # if sum_weights != 0:
    try:
        center_of_mass_fixed /= number_of_points
        center_of_mass_moving /= number_of_points
    except ZeroDivisionError:
        pass  # the weights are a total of zero which is allowed algorithmically, but not possible

    # Translate the center of mass to the origin
    fixed_coords_at_origin = fixed_coords - center_of_mass_fixed
    moving_coords_at_origin = moving_coords - center_of_mass_moving

    # Calculate the "m" array from the Diamond paper (equation 16)
    m = np.matmul(moving_coords_at_origin.T, fixed_coords_at_origin)

    # Calculate "v" (equation 18)
    # v = np.empty(3)
    # v[0] = m[1][2] - m[2][1]
    # v[1] = m[2][0] - m[0][2]
    # v[2] = m[0][1] - m[1][0]
    v = [m[1][2] - m[2][1], m[2][0] - m[0][2], m[0][1] - m[1][0]]

    # Calculate "P" (equation 22)
    matrix_p = np.zeros((4, 4))
    # Calculate "q" (equation 17)
    # q = m + m.T - 2*utils.symmetry.identity_matrix*np.trace(m)
    matrix_p[:3, :3] = m + m.T - 2*identity_matrix*np.trace(m)
    # matrix_p[:3, :3] = m + m.T - 2*utils.symmetry.identity_matrix*np.trace(m)
    matrix_p[3, :3] = v
    matrix_p[:3, 3] = v
    # [[ q[0][0] q[0][1] q[0][2] v[0] ]
    #  [ q[1][0] q[1][1] q[1][2] v[1] ]
    #  [ q[2][0] q[2][1] q[2][2] v[2] ]
    #  [ v[0]    v[1]    v[2]    0    ]]

    # Calculate "p" - optimal_quat
    # "p" contains the optimal rotation (in backwards-quaternion format)
    # (Note: A discussion of various quaternion conventions is included below)
    if number_of_points < 2:
        # Specify the default values for p, pPp
        optimal_quat = np.array([0., 0., 0., 1.])  # p = [0,0,0,1]    default value
        pPp = 0.  # = p^T * P * p    (zero by default)
    else:
        # try:
        # The a_eigenvals are returned as 1D array in ascending order; largest is last
        a_eigenvals, aa_eigenvects = np.linalg.eigh(matrix_p)
        # except np.linalg.LinAlgError:
        #     singular = True  # I have never seen this happen
        pPp = a_eigenvals[-1]
        optimal_quat = aa_eigenvects[:, -1]  # pull out the largest magnitude eigenvector
        # normalize the vector
        # (It should be normalized already, but just in case it is not, do it again)
        # optimal_quat /= np.linalg.norm(optimal_quat)

    # Calculate the rotation matrix corresponding to "optimal_quat" which is in scipy quaternion format
    """
    rotation_matrix = np.empty((3, 3))
    rotation_matrix[0][0] = (optimal_quat[0]*optimal_quat[0])-(optimal_quat[1]*optimal_quat[1])
                     -(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[1][1] = -(optimal_quat[0]*optimal_quat[0])+(optimal_quat[1]*optimal_quat[1])
                      -(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[2][2] = -(optimal_quat[0]*optimal_quat[0])-(optimal_quat[1]*optimal_quat[1])
                      +(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[0][1] = 2*(optimal_quat[0]*optimal_quat[1] - optimal_quat[2]*optimal_quat[3])
    rotation_matrix[1][0] = 2*(optimal_quat[0]*optimal_quat[1] + optimal_quat[2]*optimal_quat[3])
    rotation_matrix[1][2] = 2*(optimal_quat[1]*optimal_quat[2] - optimal_quat[0]*optimal_quat[3])
    rotation_matrix[2][1] = 2*(optimal_quat[1]*optimal_quat[2] + optimal_quat[0]*optimal_quat[3])
    rotation_matrix[0][2] = 2*(optimal_quat[0]*optimal_quat[2] + optimal_quat[1]*optimal_quat[3])
    rotation_matrix[2][0] = 2*(optimal_quat[0]*optimal_quat[2] - optimal_quat[1]*optimal_quat[3])
    """
    # Alternatively, in modern python versions, this code also works:
    rotation_matrix = Rotation.from_quat(optimal_quat).as_matrix()

    # Finally compute the RMSD between the two coordinate sets:
    # First compute E0 from equation 24 of the paper
    # e0 = np.sum((fixed_coords_at_origin - moving_coords_at_origin) ** 2)
    # sum_sqr_dist = max(0, ((fixed_coords_at_origin-moving_coords_at_origin) ** 2).sum() - 2.*pPp)

    # if sum_weights != 0.:
    try:
        rmsd = np.sqrt(max(0, ((fixed_coords_at_origin-moving_coords_at_origin) ** 2).sum() - 2.*pPp) / number_of_points)
    except ZeroDivisionError:
        rmsd = 0.  # the weights are a total of zero which is allowed algorithmically, but not possible

    # Lastly, calculate the translational offset:
    # Recall that:
    # RMSD=sqrt((Σ_i  w_i * |X_i - (Σ_j c*R_ij*x_j + T_i))|^2) / (Σ_j w_j))
    #    =sqrt((Σ_i  w_i * |X_i - x_i'|^2) / (Σ_j w_j))
    #  where
    # x_i' = Σ_j c*R_ij*x_j + T_i
    #      = Xcm_i + c*R_ij*(x_j - xcm_j)
    #  and Xcm and xcm = center_of_mass for the frozen and mobile point clouds
    #                  = center_of_mass_fixed[]       and       center_of_mass_moving[],  respectively
    # Hence:
    #  T_i = Xcm_i - Σ_j c*R_ij*xcm_j  =  a_translate[i]

    # a_translate = center_of_mass_fixed - np.matmul(c * rotation_matrix, center_of_mass_moving).T.reshape(3,)

    # Calculate the translation
    translation = center_of_mass_fixed - np.matmul(rotation_matrix, center_of_mass_moving)
    if quaternion:  # does the caller want the quaternion?
        # The p array is a quaternion that uses this convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        # However it seems that the following convention is much more popular:
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        # https://mathworld.wolfram.com/Quaternion.html
        # So I return "q" (a version of "p" using the more popular convention).
        # rotation_matrix = np.array([p[3], p[0], p[1], p[2]])
        # KM: Disregard above, I am using the scipy version for python continuity which returns X, Y, Z, W
        return rmsd, optimal_quat, translation
    else:
        return rmsd, rotation_matrix, translation


def superposition3d_weighted(fixed_coords: np.ndarray, moving_coords: np.ndarray, a_weights: np.ndarray = None,
                             quaternion: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """Takes two xyz coordinate sets (same length), and attempts to superimpose them using rotations, translations,
    and (optionally) rescale operations to minimize the root mean squared distance (RMSD) between them. The found
    transformation operations should be applied to the "moving_coords" to place them in the setting of the fixed_coords

    This function implements a more general variant of the method from:
    R. Diamond, (1988) "A Note on the Rotational Superposition Problem", Acta Cryst. A44, pp. 211-216
    This version has been augmented slightly. The version in the original paper only considers rotation and translation
    and does not allow the coordinates of either object to be rescaled (multiplication by a scalar).
    (Additional documentation can be found at https://pypi.org/project/superpose3d/ )

    The quaternion_matrix has the last entry storing cos(θ/2) (where θ is the rotation angle). The first 3 entries
    form a vector (of length sin(θ/2)), pointing along the axis of rotation.
    Details: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    MIT License. Copyright (c) 2016, Andrew Jewett
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
    Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
    OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Args:
        fixed_coords: The coordinates for the 'frozen' object
        moving_coords: The coordinates for the 'mobile' object
        a_weights: Weights for the calculation of RMSD
        quaternion: Whether to report the rotation angle and axis in Scipy.Rotation quaternion format
    Raises:
        AssertionError: If coordinates are not the same length
    Returns:
        rmsd, rotation/quaternion_matrix, translation_vector
    """
    number_of_points = fixed_coords.shape[0]
    if number_of_points != moving_coords.shape[0]:
        raise ValueError(f'{superposition3d.__name__}: Inputs should have the same size. '
                         f'Input 1={number_of_points}, 2={moving_coords.shape[0]}')

    # convert weights into array
    if a_weights is None or len(a_weights) == 0:
        a_weights = np.full((number_of_points, 1), 1.)
        sum_weights = float(number_of_points)
    else:  # reshape a_eights so multiplications are done column-wise
        a_weights = np.array(a_weights).reshape(number_of_points, 1)
        sum_weights = np.sum(a_weights, axis=0)

    # Find the center of mass of each object:
    center_of_mass_fixed = np.sum(fixed_coords * a_weights, axis=0)
    center_of_mass_moving = np.sum(moving_coords * a_weights, axis=0)

    # Subtract the centers-of-mass from the original coordinates for each object
    # if sum_weights != 0:
    try:
        center_of_mass_fixed /= sum_weights
        center_of_mass_moving /= sum_weights
    except ZeroDivisionError:
        pass  # the weights are a total of zero which is allowed algorithmically, but not possible

    aa_xf = fixed_coords - center_of_mass_fixed
    aa_xm = moving_coords - center_of_mass_moving

    # Calculate the "m" array from the Diamond paper (equation 16)
    m = np.matmul(aa_xm.T, (aa_xf * a_weights))

    # Calculate "v" (equation 18)
    v = np.empty(3)
    v[0] = m[1][2] - m[2][1]
    v[1] = m[2][0] - m[0][2]
    v[2] = m[0][1] - m[1][0]

    # Calculate "P" (equation 22)
    matrix_p = np.zeros((4, 4))
    # Calculate "q" (equation 17)
    # q = m + m.T - 2*utils.symmetry.identity_matrix*np.trace(m)
    matrix_p[:3, :3] = m + m.T - 2*identity_matrix*np.trace(m)
    # matrix_p[:3, :3] = m + m.T - 2*utils.symmetry.identity_matrix*np.trace(m)
    matrix_p[3, :3] = v
    matrix_p[:3, 3] = v
    # [[ q[0][0] q[0][1] q[0][2] v[0] ]
    #  [ q[1][0] q[1][1] q[1][2] v[1] ]
    #  [ q[2][0] q[2][1] q[2][2] v[2] ]
    #  [ v[0]    v[1]    v[2]    0    ]]

    # Calculate "p" - optimal_quat
    # "p" contains the optimal rotation (in backwards-quaternion format)
    # (Note: A discussion of various quaternion conventions is included below)
    if number_of_points < 2:
        # Specify the default values for p, pPp
        optimal_quat = np.array([0., 0., 0., 1.])  # p = [0,0,0,1]    default value
        pPp = 0.  # = p^T * P * p    (zero by default)
    else:
        # try:
        a_eigenvals, aa_eigenvects = np.linalg.eigh(matrix_p)
        # except np.linalg.LinAlgError:
        #     singular = True  # I have never seen this happen
        pPp = np.max(a_eigenvals)
        optimal_quat = aa_eigenvects[:, np.argmax(a_eigenvals)]  # pull out the largest magnitude eigenvector
        # normalize the vector
        # (It should be normalized already, but just in case it is not, do it again)
        optimal_quat /= np.linalg.norm(optimal_quat)

    # Calculate the rotation matrix corresponding to "optimal_quat" which is in scipy quaternion format
    """
    rotation_matrix = np.empty((3, 3))
    rotation_matrix[0][0] = (optimal_quat[0]*optimal_quat[0])-(optimal_quat[1]*optimal_quat[1])
                     -(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[1][1] = -(optimal_quat[0]*optimal_quat[0])+(optimal_quat[1]*optimal_quat[1])
                      -(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[2][2] = -(optimal_quat[0]*optimal_quat[0])-(optimal_quat[1]*optimal_quat[1])
                      +(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[0][1] = 2*(optimal_quat[0]*optimal_quat[1] - optimal_quat[2]*optimal_quat[3])
    rotation_matrix[1][0] = 2*(optimal_quat[0]*optimal_quat[1] + optimal_quat[2]*optimal_quat[3])
    rotation_matrix[1][2] = 2*(optimal_quat[1]*optimal_quat[2] - optimal_quat[0]*optimal_quat[3])
    rotation_matrix[2][1] = 2*(optimal_quat[1]*optimal_quat[2] + optimal_quat[0]*optimal_quat[3])
    rotation_matrix[0][2] = 2*(optimal_quat[0]*optimal_quat[2] + optimal_quat[1]*optimal_quat[3])
    rotation_matrix[2][0] = 2*(optimal_quat[0]*optimal_quat[2] - optimal_quat[1]*optimal_quat[3])
    """
    # Alternatively, in modern python versions, this code also works:
    rotation_matrix = Rotation.from_quat(optimal_quat).as_matrix()

    # Finally compute the RMSD between the two coordinate sets:
    # First compute E0 from equation 24 of the paper
    # e0 = np.sum((aa_xf - aa_xm) ** 2)
    # sum_sqr_dist = max(0, ((aa_xf-aa_xm) ** 2).sum() - 2.*pPp)

    # if sum_weights != 0.:
    try:
        rmsd = np.sqrt(max(0, ((aa_xf-aa_xm) ** 2).sum() - 2.*pPp) / sum_weights)
    except ZeroDivisionError:
        rmsd = 0.  # the weights are a total of zero which is allowed algorithmically, but not possible

    # Lastly, calculate the translational offset:
    # Recall that:
    # RMSD=sqrt((Σ_i  w_i * |X_i - (Σ_j c*R_ij*x_j + T_i))|^2) / (Σ_j w_j))
    #    =sqrt((Σ_i  w_i * |X_i - x_i'|^2) / (Σ_j w_j))
    #  where
    # x_i' = Σ_j c*R_ij*x_j + T_i
    #      = Xcm_i + c*R_ij*(x_j - xcm_j)
    #  and Xcm and xcm = center_of_mass for the frozen and mobile point clouds
    #                  = center_of_mass_fixed[]       and       center_of_mass_moving[],  respectively
    # Hence:
    #  T_i = Xcm_i - Σ_j c*R_ij*xcm_j  =  a_translate[i]

    # a_translate = center_of_mass_fixed - np.matmul(c * aa_rotate, center_of_mass_moving).T.reshape(3,)

    # return rmsd, aa_rotate, center_of_mass_fixed - np.matmul(aa_rotate, center_of_mass_moving).T.reshape(3,)
    # Calculate the translation
    translation = center_of_mass_fixed - np.matmul(rotation_matrix, center_of_mass_moving)
    if quaternion:  # does the caller want the quaternion?
        # The p array is a quaternion that uses this convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        # However it seems that the following convention is much more popular:
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        # https://mathworld.wolfram.com/Quaternion.html
        # So I return "q" (a version of "p" using the more popular convention).
        # rotation_matrix = np.array([p[3], p[0], p[1], p[2]])
        # KM: Disregard above, I am using the scipy version for python continuity
        return rmsd, optimal_quat, translation
    else:
        return rmsd, rotation_matrix, translation


# @njit
def transform_coordinates(coords: np.ndarray | Iterable, rotation: np.ndarray | Iterable = None,
                          translation: np.ndarray | Iterable | int | float = None,
                          rotation2: np.ndarray | Iterable = None,
                          translation2: np.ndarray | Iterable | int | float = None) -> np.ndarray:
    """Take a set of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the order of
    operations as: rotation, translation, rotation2, translation2

    Args:
        coords: The coordinates to transform, can be shape (number of coordinates, 3)
        rotation: The first rotation to apply, expected general rotation matrix shape (3, 3)
        translation: The first translation to apply, expected shape (3)
        rotation2: The second rotation to apply, expected general rotation matrix shape (3, 3)
        translation2: The second translation to apply, expected shape (3)
    Returns:
        The transformed coordinate set with the same shape as the original
    """
    new_coords = coords.copy()

    if rotation is not None:
        np.matmul(new_coords, np.transpose(rotation), out=new_coords)

    if translation is not None:
        new_coords += translation  # No array allocation, sets in place

    if rotation2 is not None:
        np.matmul(new_coords, np.transpose(rotation2), out=new_coords)

    if translation2 is not None:
        new_coords += translation2

    return coords


# @njit
def transform_coordinate_sets_with_broadcast(coord_sets: np.ndarray,
                                             rotation: np.ndarray = None,
                                             translation: np.ndarray | Iterable | int | float = None,
                                             rotation2: np.ndarray = None,
                                             translation2: np.ndarray | Iterable | int | float = None) \
        -> np.ndarray:
    """Take stacked sets of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the
    order of operations as: rotation, translation, rotation2, translation2. Non-efficient memory use

    Args:
        coord_sets: The coordinates to transform, can be shape (number of sets, number of coordinates, 3)
        rotation: The first rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation: The first translation to apply, expected shape (number of sets, 3)
        rotation2: The second rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation2: The second translation to apply, expected shape (number of sets, 3)
    Returns:
        The transformed coordinate set with the same shape as the original
    """
    # in general, the np.tensordot module accomplishes this coordinate set multiplication without stacking
    # np.tensordot(a, b, axes=1)  <-- axes=1 performs the correct multiplication with a 3d (3,3,N) by 2d (3,3) matrix
    # np.matmul solves as well due to broadcasting
    set_shape = getattr(coord_sets, 'shape', None)
    if set_shape is None or set_shape[0] < 1:
        return coord_sets
    # else:  # Create a new array for the result
    #     new_coord_sets = coord_sets.copy()

    if rotation is not None:
        coord_sets = np.matmul(coord_sets, rotation.swapaxes(-2, -1))

    if translation is not None:
        coord_sets += translation  # No array allocation, sets in place

    if rotation2 is not None:
        coord_sets = np.matmul(coord_sets, rotation2.swapaxes(-2, -1))

    if translation2 is not None:
        coord_sets += translation2

    return coord_sets


# @njit
def transform_coordinate_sets(coord_sets: np.ndarray,
                              rotation: np.ndarray = None, translation: np.ndarray | Iterable | int | float = None,
                              rotation2: np.ndarray = None, translation2: np.ndarray | Iterable | int | float = None) \
        -> np.ndarray:
    """Take stacked sets of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the
    order of operations as: rotation, translation, rotation2, translation2. If transformation uses broadcasting, for
    efficient memory use, the returned array will be the size of the coord_sets multiplied by rotation. Additional
    broadcasting is not allowed. If that behavior is desired, use "transform_coordinate_sets_with_broadcast()" instead

    Args:
        coord_sets: The coordinates to transform, can be shape (number of sets, number of coordinates, 3)
        rotation: The first rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation: The first translation to apply, expected shape (number of sets, 3)
        rotation2: The second rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation2: The second translation to apply, expected shape (number of sets, 3)
    Returns:
        The transformed coordinate set with the same shape as the original
    """
    # in general, the np.tensordot module accomplishes this coordinate set multiplication without stacking
    # np.tensordot(a, b, axes=1)  <-- axes=1 performs the correct multiplication with a 3d (3,3,N) by 2d (3,3) matrix
    # np.matmul solves as well due to broadcasting
    set_shape = getattr(coord_sets, 'shape', None)
    if set_shape is None or set_shape[0] < 1:
        return coord_sets

    if rotation is not None:
        new_coord_sets = np.matmul(coord_sets, rotation.swapaxes(-2, -1))
    else:  # Create a new array for the result
        new_coord_sets = coord_sets.copy()

    if translation is not None:
        new_coord_sets += translation  # No array allocation, sets in place

    if rotation2 is not None:
        np.matmul(new_coord_sets, rotation2.swapaxes(-2, -1), out=new_coord_sets)
        # new_coord_sets[:] = np.matmul(new_coord_sets, rotation2.swapaxes(-2, -1))
        # new_coord_sets = np.matmul(new_coord_sets, rotation2.swapaxes(-2, -1))

    if translation2 is not None:
        new_coord_sets += translation2

    return new_coord_sets
