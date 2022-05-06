import numpy as np

from SymDesignUtils import start_log

logger = start_log(name=__name__)


class OptimalTx:
    def __init__(self, dof_ext=None, zshift1=None, zshift2=None, max_z_value=1., number_of_coordinates=3):
        self.max_z_value = max_z_value
        self.number_of_coordinates = number_of_coordinates
        self.dof_ext = np.array(dof_ext)  # External translational DOF (number DOF external x 3)
        self.dof = self.dof_ext.copy()
        # print('self.dof', self.dof)
        self.zshift1 = zshift1  # internal translational DOF1
        self.zshift2 = zshift2  # internal translational DOF2
        self.dof9 = None
        self.dof9_t = None
        self.dof9t_dof9 = None

        # add internal z-shift degrees of freedom to 9-dim arrays if they exist
        self.n_dof_internal = 0
        if self.zshift1 is not None:
            # print('self.zshift1', self.zshift1)
            self.dof = np.append(self.dof, -self.zshift1, axis=0)
            self.n_dof_internal += 1
        if self.zshift2 is not None:
            self.dof = np.append(self.dof, self.zshift2, axis=0)
            self.n_dof_internal += 1
        # print('self.dof', self.dof)

        self.n_dof_external = self.dof_ext.shape[0]  # get the length of the numpy array
        self.n_dof = self.dof.shape[0]
        if self.n_dof > 0:
            self.dof_convert9()
        else:
            raise ValueError('n_dof is not set! Can\'t get the OptimalTx without passing dof_ext, zshift1, or zshift2')

    @classmethod
    def from_dof(cls, dof_ext=None, zshift1=None, zshift2=None, max_z_value=1.):  # setting1, setting2,
        return cls(dof_ext=dof_ext, zshift1=zshift1, zshift2=zshift2, max_z_value=max_z_value)
        # setting1=setting1 setting2=setting2

    # @classmethod
    # def from_tx_params(cls, optimal_tx_params, error_zvalue):
    #     return cls(tx_params=optimal_tx_params, error=error_zvalue)

    def dof_convert9(self):
        """Convert input degrees of freedom to 9-dim arrays. Repeat DOF ext for each set of 3 coordinates (3 sets)"""
        self.dof9_t = np.zeros((self.n_dof, 9))
        for i in range(self.n_dof):
            # self.dof9_t[i] = np.array(self.number_of_coordinates * [self.dof[i]]).flatten()
            self.dof9_t[i] = np.tile(self.dof[i], self.number_of_coordinates)
            # dof[i] = (np.array(3 * [self.dof_ext[i]])).flatten()
        # print('self.dof9_t', self.dof9_t)
        self.dof9 = np.transpose(self.dof9_t)
        # print('self.dof9', self.dof9)
        self.dof9t_dof9 = np.matmul(self.dof9_t, self.dof9)

    def solve_optimal_shift(self, coords1, coords2, coords_rmsd_reference):
        """This routine solves the optimal shift problem for overlapping a pair of coordinates and comparing to a
        reference RMSD to compute an error

        Args:
            coords1 (np.ndarray): A 3 x 3 array with cartesian coordinates
            coords2 (np.ndarray): A 3 x 3 array with cartesian coordinates
            coords_rmsd_reference (float): The reference deviation to compare to the coords1 and coords2 error
        Returns:
            (Union[numpy.ndarray, None]): Returns the optimal translation or None if error is too large.
                Optimal translation has external dof first, followed by internal tx dof
        """
        # form the guide coords into a matrix (column vectors)
        # guide_target_10 = np.transpose(coords1)
        # guide_query_10 = np.transpose(coords2)

        # calculate the initial difference between query and target (9 dim vector)
        # With the transpose and the flatten, it could be accomplished by normal flatten!
        guide_delta = np.transpose([coords1.flatten() - coords2.flatten()])
        # flatten column vector matrix above [[x, y, z], [x, y, z], [x, y, z]] -> [x, y, z, x, y, z, x, y, z], then T

        # # isotropic case based on simple rmsd
        # | var_tot_inv = np.zeros([9, 9])
        # | for i in range(9):
        # |     # fill in var_tot_inv with 1/ 3x the mean squared deviation (deviation sum)
        # |     var_tot_inv[i, i] = 1. / (float(self.number_of_coordinates) * coords_rmsd_reference ** 2)
        # can be simplified to just use the scalar
        var_tot = float(self.number_of_coordinates) * coords_rmsd_reference ** 2

        # solve the problem using 9-dim degrees of freedom arrays
        # self.dof9 is column major (9 x n_dof_ext) degree of freedom matrix
        # self.dof9_t transpose (row major: n_dof_ext x 9)
        # below is degrees_of_freedom / variance
        # | dinvv = np.matmul(var_tot_inv, self.dof9)  # 1/variance (9 x 9) x degree of freedom (9 x n_dof) = (9 x n_dof)
        # dinvv = self.dof9 / var_tot
        # below, each i, i is the (individual_dof^2) * 3 / variance. i, j is the (covariance of i and jdof * 3) / variance
        # | vtdinvv = np.matmul(self.dof9_t, dinvv)  # transpose of degrees of freedom (n_dof x 9) x (9 x n_dof) = (n_dof x n_dof)
        # above could be simplifed to vtdinvv = np.matmul(self.dof9_t, self.dof9) / var_tot_inv  # first part same for each guide coord
        # now done below
        # vtdinvv = np.matmul(self.dof9_t, self.dof9) / var_tot  # transpose of degrees of freedom (n_dof x 9) x (9 x n_dof) = (n_dof x n_dof)
        vtdinvv = self.dof9t_dof9 / var_tot  # transpose of degrees of freedom (n_dof x 9) x (9 x n_dof) = (n_dof x n_dof)
        vtdinvvinv = np.linalg.inv(vtdinvv)  # Inverse of above - (n_dof x n_dof)
        # below is guide atom difference / variance
        # | dinvdelta = np.matmul(var_tot_inv, guide_delta)  # 1/variance (9 x 9) x guide atom diff (9 x 1) = (9 x 1)
        # dinvdelta = guide_delta / var_tot
        # below is essentially (SUM(dof basis * guide atom basis difference) for each guide atom) /variance by each DOF
        # | vtdinvdelta = np.matmul(self.dof9_t, dinvdelta)  # transpose of degrees of freedom (n_dof x 9) x (9 x 1) = (n_dof x 1)
        vtdinvdelta = np.matmul(self.dof9_t, guide_delta) / var_tot # transpose of degrees of freedom (n_dof x 9) x (9 x 1) = (n_dof x 1)

        # below is inverse dof covariance matrix/variance * dof guide_atom_delta sum / variance
        # | shift = np.matmul(vtdinvvinv, vtdinvdelta)  # (n_dof x n_dof) x (n_dof x 1) = (n_dof x 1)
        shift = np.matmul(vtdinvvinv, vtdinvdelta)  # (n_dof x n_dof) x (n_dof x 1) = (n_dof x 1)

        # get error value from the ideal translation and the delta
        resid = np.matmul(self.dof9, shift) - guide_delta  # (9 x n_dof) x (n_dof x 1) - (9 x 1) = (9 x 1)
        error = \
            np.sqrt(np.matmul(np.transpose(resid), resid) / float(self.number_of_coordinates)) / coords_rmsd_reference
        # NEW. Is float(3.0) a scale?
        # OLD. sqrt(variance / 3) / cluster_rmsd

        if error <= self.max_z_value:
            return shift[:, 0]  # .tolist()  # , error
        else:
            return

    def solve_optimal_shifts(self, coords1, coords2, coords_rmsd_reference):
        """This routine solves the optimal shift problem for overlapping a pair of coordinates and comparing to a
        reference RMSD to compute an error

        Args:
            coords1 (np.ndarray): A N x 3 x 3 array with cartesian coordinates
            coords2 (np.ndarray): A N x 3 x 3 array with cartesian coordinates
            coords_rmsd_reference (np.ndarray): Array with length N with reference deviation to compare to the coords1
                and coords2 error
        Returns:
            (numpy.ndarray): Returns the optimal translation or None if error is too large.
                Optimal translation has external dof first, followed by internal tx dof
        """
        # calculate the initial difference between each query and target (9 dim vector by coords.shape[0])
        guide_delta = (coords1 - coords2).reshape(-1, 1, 9).swapaxes(-2, -1)
        # flatten column vector matrix above [[x, y, z], [x, y, z], [x, y, z]] -> [x, y, z, x, y, z, x, y, z], then T
        # # isotropic case based on simple rmsd
        # | var_tot_inv = np.zeros([9, 9])
        # | for i in range(9):
        # |     # fill in var_tot_inv with 1/ 3x the mean squared deviation (deviation sum)
        # |     var_tot_inv[i, i] = 1. / (float(self.number_of_coordinates) * coords_rmsd_reference ** 2)
        # can be simplified to just use the scalar
        var_tot = (float(self.number_of_coordinates) * coords_rmsd_reference ** 2).reshape(-1, 1, 1)

        # solve the problem using 9-dim degrees of freedom arrays
        # self.dof9 is column major (9 x n_dof_ext) degree of freedom matrix
        # self.dof9_t transpose (row major: n_dof_ext x 9)
        # below is degrees_of_freedom / variance
        # | dinvv = np.matmul(var_tot_inv, self.dof9)  # 1/variance (9 x 9) x degree of freedom (9 x n_dof) = (9 x n_dof)
        # dinvv = self.dof9 / var_tot
        # below, each i, i is the (individual_dof^2) * 3 / variance. i, j is the (covariance of i and jdof * 3) / variance
        # | vtdinvv = np.matmul(self.dof9_t, dinvv)  # transpose of degrees of freedom (n_dof x 9) x (9 x n_dof) = (n_dof x n_dof)
        # above could be simplifed to vtdinvv = np.matmul(self.dof9_t, self.dof9) / var_tot_inv  # first part same for each guide coord
        # now done below
        # vtdinvv = np.matmul(self.dof9_t, self.dof9) / var_tot  # transpose of degrees of freedom (n_dof x 9) x (9 x n_dof) = (n_dof x n_dof)
        # vtdinvv = np.tile(self.dof9t_dof9, (coords1.shape[0], 1, 1)) / var_tot  # transpose of degrees of freedom (n_dof x 9) x (9 x n_dof) = (n_dof x n_dof)

        # vtdinvvinv = np.linalg.inv(vtdinvv)  # Inverse of above - (n_dof x n_dof)
        # below is guide atom difference / variance
        # | dinvdelta = np.matmul(var_tot_inv, guide_delta)  # 1/variance (9 x 9) x guide atom diff (9 x 1) = (9 x 1)
        # dinvdelta = guide_delta / var_tot
        # below is essentially (SUM(dof basis * guide atom basis difference) for each guide atom) /variance by each DOF
        # | vtdinvdelta = np.matmul(self.dof9_t, dinvdelta)  # transpose of degrees of freedom (n_dof x 9) x (9 x 1) = (n_dof x 1)
        # vtdinvdelta = np.matmul(np.tile(self.dof9_t, (coords1.shape[0], 1, 1)), guide_delta) / var_tot  # transpose of degrees of freedom (n_dof x 9) x (9 x 1) = (n_dof x 1)

        # below is inverse dof covariance matrix/variance * dof guide_atom_delta sum / variance
        # shift = np.matmul(vtdinvvinv, vtdinvdelta)  # (n_dof x n_dof) x (n_dof x 1) = (n_dof x 1)
        # print('self.dof9t_dof9', self.dof9t_dof9)
        # print('tiled_array', np.tile(self.dof9t_dof9, (coords1.shape[0], 1, 1)))
        shift = np.matmul(np.linalg.inv(np.tile(self.dof9t_dof9, (coords1.shape[0], 1, 1)) / var_tot),
                          np.matmul(np.tile(self.dof9_t, (coords1.shape[0], 1, 1)), guide_delta) / var_tot)  # (n_dof x n_dof) x (n_dof x 1) = (n_dof x 1)

        # get error value from the ideal translation and the delta
        resid = np.matmul(np.tile(self.dof9, (coords1.shape[0], 1, 1)), shift) - guide_delta  # (9 x n_dof) x (n_dof x 1) - (9 x 1) = (9 x 1)
        error = np.sqrt(np.matmul(resid.swapaxes(-2, -1), resid) / float(self.number_of_coordinates)).flatten() \
            / coords_rmsd_reference

        return shift[np.nonzero(error <= self.max_z_value)].squeeze()
