import numpy as np
from numpy import linalg as LA


class Coords:
    def __init__(self, coords=None):
        if coords:
            self.coords = np.array(coords)
        else:
            self.coords = np.array([])

    @property
    def coords(self):  # , transformation_operator=None):
        """This holds the atomic coords which is a view from the Structure that created them"""
        # if transformation_operator:
        #     return np.matmul([self.x, self.y, self.z], transformation_operator)
        # else:
        return self.coords  # [self.x, self.y, self.z]

    @coords.setter
    def coords(self, coords):
        self.coords = coords

    def __len__(self):
        return self.coords.shape[0]

    # @property
    # def x(self):
    #     return self.coords[0]  # x
    #
    # @x.setter
    # def x(self, x):
    #     self.coords[0] = x
    #
    # @property
    # def y(self):
    #     return self.coords[1]  # y
    #
    # @y.setter
    # def y(self, y):
    #     self.coords[1] = y
    #
    # @property
    # def z(self):
    #     return self.coords[2]  # z
    #
    # @z.setter
    # def z(self, z):
    #     self.coords[2] = z


def superposition3d(aa_xf_orig, aa_xm_orig, a_weights=None, allow_rescale=False, report_quaternion=False):
    """
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

    Superpose3D() takes two lists of xyz coordinates (same length), and attempts to superimpose them using rotations,
     translations, and (optionally) rescale operations in order to minimize the root-mean-squared-distance (RMSD)
     between them. These operations should be applied to the "aa_xf_orig" argument.

    This function implements a more general variant of the method from:
    R. Diamond, (1988) "A Note on the Rotational Superposition Problem", Acta Cryst. A44, pp. 211-216
    This version has been augmented slightly. The version in the original paper only considers rotation and translation
    and does not allow the coordinates of either object to be rescaled (multiplication by a scalar).
    (Additional documentation can be found at https://pypi.org/project/superpose3d/ )

    Args:
        aa_xf_orig (numpy.array): The coordinates for the "frozen" object
        aa_xm_orig (numpy.array): The coordinates for the "mobile" object
    Keyword Args:
        aWeights=None (numpy.array): The optional weights for the calculation of RMSD
        allow_rescale=False (bool): Attempt to rescale the mobile point cloud in addition to translation/rotation?
        report_quaternion=False (bool): Whether to report the rotation angle and axis in typical quaternion fashion
    Returns:
        (float, numpy.array, numpy.array, float): Corresponding to the rmsd, optimal rotation_matrix or
        quaternion_matrix (if report_quaternion=True), optimal_translation_vector, and optimal_scale_factor.
        The quaternion_matrix has the first row storing cos(θ/2) (where θ is the rotation angle). The following 3 rows
        form a vector (of length sin(θ/2)), pointing along the axis of rotation.
        Details here: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    # convert input lists as to numpy arrays

    aa_xf_orig = np.array(aa_xf_orig)
    aa_xm_orig = np.array(aa_xm_orig)

    if aa_xf_orig.shape[0] != aa_xm_orig.shape[0]:
        raise ValueError("Inputs should have the same size.")

    number_of_points = aa_xf_orig.shape[0]
    # Find the center of mass of each object:
    """ # old code (using for-loops)
    if (aWeights == None) or (len(aWeights) == 0):
        aWeights = np.full(number_of_points, 1.0)
    a_center_f = np.zeros(3)
    a_center_m = np.zeros(3)
    sum_weights = 0.0
    for n in range(0, number_of_points):
        for d in range(0, 3):
            a_center_f[d] += aaXf_orig[n][d]*aWeights[n]
            a_center_m[d] += aaXm_orig[n][d]*aWeights[n]
        sum_weights += aWeights[n]
    """
    # new code (avoiding for-loops)
    # convert weights into array
    if not a_weights or (len(a_weights) == 0):
        a_weights = np.full((number_of_points, 1), 1.0)
    else:
        # reshape aWeights so multiplications are done column-wise
        a_weights = np.array(a_weights).reshape(number_of_points, 1)

    a_center_f = np.sum(aa_xf_orig * a_weights, axis=0)
    a_center_m = np.sum(aa_xm_orig * a_weights, axis=0)
    sum_weights = np.sum(a_weights, axis=0)

    # Subtract the centers-of-mass from the original coordinates for each object
    """ # old code (using for-loops)
    if sum_weights != 0:
        for d in range(0, 3):
            a_center_f[d] /= sum_weights
            a_center_m[d] /= sum_weights
    for n in range(0, number_of_points):
        for d in range(0, 3):
            aa_xf[n][d] = aaXf_orig[n][d] - a_center_f[d]
            aa_xm[n][d] = aaXm_orig[n][d] - a_center_m[d]
    """
    # new code (avoiding for-loops)
    if sum_weights != 0:
        a_center_f /= sum_weights
        a_center_m /= sum_weights
    aa_xf = aa_xf_orig - a_center_f
    aa_xm = aa_xm_orig - a_center_m

    # Calculate the "M" array from the Diamond paper (equation 16)
    """ # old code (using for-loops)
    M = np.zeros((3,3))
    for n in range(0, number_of_points):
        for i in range(0, 3):
            for j in range(0, 3):
                M[i][j] += aWeights[n] * aa_xm[n][i] * aa_xf[n][j]
    """
    M = np.matmul(aa_xm.T, (aa_xf * a_weights))

    # Calculate Q (equation 17)

    """ # old code (using for-loops)
    traceM = 0.0
    for i in range(0, 3):
        traceM += M[i][i]
    Q = np.empty((3,3))
    for i in range(0, 3):
        for j in range(0, 3):
            Q[i][j] = M[i][j] + M[j][i]
            if i==j:
                Q[i][j] -= 2.0 * traceM
    """
    Q = M + M.T - 2 * np.eye(3) * np.trace(M)

    # Calculate v (equation 18)
    v = np.empty(3)
    v[0] = M[1][2] - M[2][1]
    v[1] = M[2][0] - M[0][2]
    v[2] = M[0][1] - M[1][0]

    # Calculate "P" (equation 22)
    """ # old code (using for-loops)
    P = np.empty((4,4))
    for i in range(0,3):
        for j in range(0,3):
            P[i][j] = Q[i][j]
    P[0][3] = v[0]
    P[3][0] = v[0]
    P[1][3] = v[1]
    P[3][1] = v[1]
    P[2][3] = v[2]
    P[3][2] = v[2]
    P[3][3] = 0.0
    """
    P = np.zeros((4, 4))
    P[:3, :3] = Q
    P[3, :3] = v
    P[:3, 3] = v

    # Calculate "p".
    # "p" contains the optimal rotation (in backwards-quaternion format)
    # (Note: A discussion of various quaternion conventions is included below.)
    # First, specify the default value for p:
    p = np.zeros(4)
    p[3] = 1.0           # p = [0,0,0,1]    default value
    pPp = 0.0            # = p^T * P * p    (zero by default)
    singular = (number_of_points < 2)   # (it doesn't make sense to rotate a single point)

    try:
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html
        a_eigenvals, aa_eigenvects = LA.eigh(P)
    except LinAlgError:
        singular = True  # (I have never seen this happen.)

    if not singular:  # (don't crash if the caller supplies nonsensical input)
        """ # old code (using for-loops)
        eval_max = a_eigenvals[0]
        i_eval_max = 0
        for i in range(1, 4):
            if a_eigenvals[i] > eval_max:
                eval_max = a_eigenvals[i]
                i_eval_max = i
        p[0] = aa_eigenvects[0][i_eval_max]
        p[1] = aa_eigenvects[1][i_eval_max]
        p[2] = aa_eigenvects[2][i_eval_max]
        p[3] = aa_eigenvects[3][i_eval_max]
        pPp = eval_max
        """
        # new code (avoiding for-loops)
        i_eval_max = np.argmax(a_eigenvals)
        pPp = np.max(a_eigenvals)
        p[:] = aa_eigenvects[:, i_eval_max]

    # normalize the vector
    # (It should be normalized already, but just in case it is not, do it again)
    p /= np.linalg.norm(p)

    # Finally, calculate the rotation matrix corresponding to "p"
    # (p is in backwards-quaternion format)

    aa_rotate = np.empty((3, 3))
    aa_rotate[0][0] = (p[0]*p[0])-(p[1]*p[1])-(p[2]*p[2])+(p[3]*p[3])
    aa_rotate[1][1] = -(p[0]*p[0])+(p[1]*p[1])-(p[2]*p[2])+(p[3]*p[3])
    aa_rotate[2][2] = -(p[0]*p[0])-(p[1]*p[1])+(p[2]*p[2])+(p[3]*p[3])
    aa_rotate[0][1] = 2*(p[0]*p[1] - p[2]*p[3])
    aa_rotate[1][0] = 2*(p[0]*p[1] + p[2]*p[3])
    aa_rotate[1][2] = 2*(p[1]*p[2] - p[0]*p[3])
    aa_rotate[2][1] = 2*(p[1]*p[2] + p[0]*p[3])
    aa_rotate[0][2] = 2*(p[0]*p[2] + p[1]*p[3])
    aa_rotate[2][0] = 2*(p[0]*p[2] - p[1]*p[3])

    # Alternatively, in modern python versions, this code also works:
    """
    from scipy.spatial.transform import Rotation as R
    the_rotation = R.from_quat(p)
    aa_rotate = the_rotation.as_matrix()
    """

    # Optional: Decide the scale factor, c
    c = 1.0   # by default, don't rescale the coordinates
    if allow_rescale and (not singular):
        """ # old code (using for-loops)
        Waxaixai = 0.0
        WaxaiXai = 0.0
        for a in range(0, number_of_points):
            for i in range(0, 3):
                Waxaixai += aWeights[a] * aa_xm[a][i] * aa_xm[a][i]
                WaxaiXai += aWeights[a] * aa_xm[a][i] * aa_xf[a][i]
        """
        # new code (avoiding for-loops)
        Waxaixai = np.sum(a_weights * aa_xm ** 2)
        WaxaiXai = np.sum(a_weights * aa_xf ** 2)

        c = (WaxaiXai + pPp) / Waxaixai

    # Finally compute the RMSD between the two coordinate sets:
    # First compute E0 from equation 24 of the paper

    """ # old code (using for-loops)
    E0 = 0.0
    for n in range(0, number_of_points):
        for d in range(0, 3):
            # (remember to include the scale factor "c" that we inserted)
            E0 += aWeights[n] * ((aa_xf[n][d] - c*aa_xm[n][d])**2)
    sum_sqr_dist = E0 - c*2.0*pPp
    if sum_sqr_dist < 0.0: #(edge case due to rounding error)
        sum_sqr_dist = 0.0
    """
    # new code (avoiding for-loops)
    E0 = np.sum((aa_xf - c * aa_xm) ** 2)
    sum_sqr_dist = max(0, E0 - c * 2.0 * pPp)

    rmsd = 0.0
    if sum_weights != 0.0:
        rmsd = np.sqrt(sum_sqr_dist/sum_weights)

    # Lastly, calculate the translational offset:
    # Recall that:
    #RMSD=sqrt((Σ_i  w_i * |X_i - (Σ_j c*R_ij*x_j + T_i))|^2) / (Σ_j w_j))
    #    =sqrt((Σ_i  w_i * |X_i - x_i'|^2) / (Σ_j w_j))
    #  where
    # x_i' = Σ_j c*R_ij*x_j + T_i
    #      = Xcm_i + c*R_ij*(x_j - xcm_j)
    #  and Xcm and xcm = center_of_mass for the frozen and mobile point clouds
    #                  = a_center_f[]       and       a_center_m[],  respectively
    # Hence:
    #  T_i = Xcm_i - Σ_j c*R_ij*xcm_j  =  a_translate[i]

    """ # old code (using for-loops)
    a_translate = np.empty(3)
    for i in range(0,3):
        a_translate[i] = a_center_f[i]
        for j in range(0,3):
            a_translate[i] -= c*aa_rotate[i][j]*a_center_m[j]
    """
    # new code (avoiding for-loops)
    a_translate = a_center_f - np.matmul(c * aa_rotate, a_center_m).T.reshape(3,)

    if report_quaternion:  # does the caller want the quaternion?
        # The p array is a quaternion that uses this convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        # However it seems that the following convention is much more popular:
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        # https://mathworld.wolfram.com/Quaternion.html
        # So I return "q" (a version of "p" using the more popular convention).
        q = np.empty(4)
        q[0] = p[3]
        q[1] = p[0]
        q[2] = p[1]
        q[3] = p[2]
        return rmsd, q, a_translate, c
    else:
        return rmsd, aa_rotate, a_translate, c
