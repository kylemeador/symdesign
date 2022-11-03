import logging
from math import sqrt

import numpy as np

logger = logging.getLogger(__name__)


class OptimalTx:
    def __init__(self, dof_ext: np.ndarray = None, zshift1: np.ndarray = None, zshift2: np.ndarray = None,
                 max_z_value: float = 1., number_of_coordinates: int = 3):
        self.max_z_value = max_z_value
        self.number_of_coordinates = number_of_coordinates
        if dof_ext is None:  # Todo include np.zeros((1, 3)) like in SymEntry
            raise ValueError(f"Can't initialize {type(self.__name__)} without passing dof_ext")
        else:
            self.dof_ext = dof_ext  # External translational DOF with shape (number DOF external, 3)
        self.dof = self.dof_ext.copy()
        # logger.debug('self.dof', self.dof)
        self.zshift1 = zshift1  # internal translational DOF1
        self.zshift2 = zshift2  # internal translational DOF2
        self.dof9 = None
        self.dof9_t = None
        self.dof9t_dof9 = None

        # Add internal z-shift degrees of freedom to 9-dim arrays if they exist
        self.n_dof_internal = 0
        if self.zshift1 is not None:
            # logger.debug('self.zshift1', self.zshift1)
            self.dof = np.append(self.dof, -self.zshift1, axis=0)
            self.n_dof_internal += 1
        if self.zshift2 is not None:
            self.dof = np.append(self.dof, self.zshift2, axis=0)
            self.n_dof_internal += 1
        # logger.debug('self.dof', self.dof)

        self.n_dof_external = self.dof_ext.shape[0]  # Get the length of the array
        self.n_dof = self.dof.shape[0]
        if self.n_dof > 0:
            self.dof_convert9()
        else:
            raise ValueError(f"n_dof is not set! Can't get the {type(self).__name__}"
                             f" without passing dof_ext, zshift1, or zshift2")

    @classmethod
    def from_dof(cls, dof_ext: np.ndarray, zshift1: np.ndarray = None, zshift2: np.ndarray = None,
                 max_z_value: float = 1.):
        return cls(dof_ext=dof_ext, zshift1=zshift1, zshift2=zshift2, max_z_value=max_z_value)

    # @classmethod
    # def from_tx_params(cls, optimal_tx_params, error_zvalue):
    #     return cls(tx_params=optimal_tx_params, error=error_zvalue)

    def dof_convert9(self):
        """Convert input degrees of freedom to 9-dim arrays. Repeat DOF ext for each set of 3 coordinates (3 sets)"""
        self.dof9_t = np.zeros((self.n_dof, 9))
        for dof_idx in range(self.n_dof):
            # self.dof9_t[dof_idx] = np.array(self.number_of_coordinates * [self.dof[dof_idx]]).flatten()
            self.dof9_t[dof_idx] = np.tile(self.dof[dof_idx], self.number_of_coordinates)
            # dof[dof_idx] = (np.array(3 * [self.dof_ext[dof_idx]])).flatten()
        # logger.debug('self.dof9_t', self.dof9_t)
        self.dof9 = np.transpose(self.dof9_t)
        # logger.debug('self.dof9', self.dof9)
        self.dof9t_dof9 = np.matmul(self.dof9_t, self.dof9)

    def solve_optimal_shift(self, coords1: np.ndarray, coords2: np.ndarray, coords_rmsd_reference: float) -> \
            np.ndarray | None:
        """This routine solves the optimal shift problem for overlapping a pair of coordinates and comparing to a
        reference RMSD to compute an error

        Args:
            coords1: A 3 x 3 array with cartesian coordinates
            coords2: A 3 x 3 array with cartesian coordinates
            coords_rmsd_reference: The reference deviation to compare to the coords1 and coords2 error
        Returns:
            Returns the optimal translation or None if error is too large.
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
        vtdinvdelta = np.matmul(self.dof9_t, guide_delta) / var_tot  # transpose of degrees of freedom (n_dof x 9) x (9 x 1) = (n_dof x 1)

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
            return None

    def solve_optimal_shifts(self, coords1: np.ndarray, coords2: np.ndarray, coords_rmsd_reference: np.ndarray) -> \
            np.ndarray:
        """This routine solves the optimal shift problem for overlapping a pair of coordinates and comparing to a
        reference RMSD to compute an error

        Args:
            coords1: A N x 3 x 3 array with cartesian coordinates
            coords2: A N x 3 x 3 array with cartesian coordinates
            coords_rmsd_reference: Array with length N with reference deviation to compare to the coords1 and coords2
                error
        Returns:
            Returns the optimal translations with shape (N, number_degrees_of_freedom) if the translation is less than
                the calculated error. Axis 1 has degrees of freedom with external first, then internal dof
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

        return shift[np.nonzero(error <= self.max_z_value)].reshape(-1, self.n_dof)


class OptimalTxOLD:
    def __init__(self, setting1, setting2, is_zshift1, is_zshift2, cluster_rmsd, guide_atom_coods1, guide_atom_coods2,
                 dof_ext):
        self.setting1 = np.array(setting1)
        self.setting2 = np.array(setting2)
        self.is_zshift1 = is_zshift1
        self.is_zshift2 = is_zshift2
        self.dof_ext = np.array(dof_ext)
        self.n_dof_external = len(self.dof_ext)
        self.cluster_rmsd = cluster_rmsd
        self.guide_atom_coords1 = guide_atom_coods1
        self.guide_atom_coords2 = guide_atom_coods2

        self.n_dof_internal = [self.is_zshift1, self.is_zshift2].count(True)
        self.optimal_tx = (np.array([]), float('inf'))  # (shift, error_zvalue)
        # self.guide_atom_coords1_set = []
        # self.guide_atom_coords2_set = []

    def dof_convert9(self):
        # convert input degrees of freedom to 9-dim arrays
        ndof = len(self.dof_ext)
        dof = np.zeros((ndof, 9))
        for i in range(ndof):
            dof[i] = (np.array(3 * [self.dof_ext[i]])).flatten()
        return np.transpose(dof)

    def solve_optimal_shift(self):
        # This routine does the work to solve the optimal shift problem

        # form the guide atoms into a matrix (column vectors)
        # guide_target_10 = np.transpose(self.guide_atom_coords1_set)
        # guide_query_10 = np.transpose(self.guide_atom_coords2_set)

        # calculate the initial difference between query and target (9 dim vector)
        guide_delta = np.transpose([self.guide_atom_coords1.flatten() - self.guide_atom_coords2.flatten()])

        # isotropic case based on simple rmsd
        diagval = 1. / (3. * self.cluster_rmsd ** 2)
        var_tot_inv = np.zeros([9, 9])
        for i in range(9):
            var_tot_inv[i, i] = diagval

        # add internal z-shift degrees of freedom to 9-dim arrays if they exist
        if self.is_zshift1:
            self.dof_ext = np.append(self.dof_ext, -self.setting1[:, None, 2].T, axis=0)
        if self.is_zshift2:
            self.dof_ext = np.append(self.dof_ext, self.setting2[:, None, 2].T, axis=0)

        # convert degrees of freedom to 9-dim array
        dof = self.dof_convert9()

        # solve the problem
        dofT = np.transpose(dof)
        dinvv = np.matmul(var_tot_inv, dof)
        vtdinvv = np.matmul(dofT, dinvv)
        vtdinvvinv = np.linalg.inv(vtdinvv)

        dinvdelta = np.matmul(var_tot_inv, guide_delta)
        vtdinvdelta = np.matmul(dofT, dinvdelta)

        shift = np.matmul(vtdinvvinv, vtdinvdelta)

        # get error value
        resid = np.matmul(dof, shift) - guide_delta
        residT = np.transpose(resid)

        error = sqrt(np.matmul(residT, resid) / float(3.0)) / self.cluster_rmsd  # sqrt(variance / 3) / cluster_rmsd # NEW ERROR

        # etmp = np.matmul(var_tot_inv, resid)  # need to comment out, old error
        # error = np.matmul(residT, etmp)  # need to comment out, old error
        # self.optimal_tx = (shift[:, 0], error[0, 0])  # (shift, error_zvalue) # need to comment out, old error

        self.optimal_tx = (shift[:, 0], error)  # (shift, error_zvalue)

    @staticmethod
    def mat_vec_mul3(a, b):
        c = [0. for i in range(3)]

        for i in range(3):
            c[i] = 0.
            for j in range(3):
                c[i] += a[i][j] * b[j]

        return c

    def set_guide_atoms(self, rot_mat, coords):
        rotated_coords = []

        for coord in coords:
            x, y, z = self.mat_vec_mul3(rot_mat, [coord[0], coord[1], coord[2]])
            rotated_coords.append([x, y, z])

        return rotated_coords

    # def apply(self):
    #     # Apply Setting Matrix to Guide Atoms
    #     # self.guide_atom_coords1_set = self.set_guide_atoms(self.setting1, self.guide_atom_coords1)
    #     # self.guide_atom_coords2_set = self.set_guide_atoms(self.setting2, self.guide_atom_coords2)
    #     # self.guide_atom_coords1_set = np.matmul(self.guide_atom_coords1, np.transpose(self.setting1))
    #     # self.guide_atom_coords2_set = np.matmul(self.guide_atom_coords2, np.transpose(self.setting2))
    #
    #     # solve for shifts and resulting error
    #     self.solve_optimal_shift()

    def get_optimal_tx_dof_int(self):
        tx_dof_int = []

        shift, error_zvalue = self.optimal_tx
        index = self.n_dof_external

        if self.is_zshift1:
            tx_dof_int.append(shift[index:index + 1][0])
            index += 1

        if self.is_zshift2:
            tx_dof_int.append(shift[index:index + 1][0])

        return tx_dof_int

    def get_optimal_tx_dof_ext(self):
        shift, error_zvalue = self.optimal_tx
        return shift[0:self.n_dof_external].tolist()

    def get_all_optimal_shifts(self):
        shift, error_zvalue = self.optimal_tx
        return shift.tolist()

    def get_n_dof_external(self):
        return self.n_dof_external

    def get_n_dof_internal(self):
        return self.n_dof_internal

    def get_zvalue(self):
        shift, error_zvalue = self.optimal_tx
        return error_zvalue
