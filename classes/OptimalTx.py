from math import sqrt

import numpy as np


class OptimalTx:
    def __init__(self, dof_ext=None, zshift1=None, zshift2=None, max_z_value=1.):
        self.max_z_value = max_z_value
        self.dof_ext = np.array(dof_ext)  # External translational DOF (number DOF external x 3)
        self.dof = self.dof_ext.copy()
        self.zshift1 = zshift1  # internal translational DOF1
        self.zshift2 = zshift2  # internal translational DOF2
        self.dof9 = None
        self.dof9_t = None

        # add internal z-shift degrees of freedom to 9-dim arrays if they exist
        self.n_dof_internal = 0
        if self.zshift1 is not None:
            self.dof = np.append(self.dof, -self.zshift1, axis=0)
            self.n_dof_internal += 1
        if self.zshift2 is not None:
            self.dof = np.append(self.dof, self.zshift2, axis=0)
            self.n_dof_internal += 1

        self.n_dof_external = self.dof_ext.shape[0]  # get the length of the numpy array
        self.n_dof = self.dof.shape[0]
        if self.n_dof > 0:
            self.dof_convert9()
        else:
            raise ValueError('n_dof is not set! Can\'t get the OptimalTx without passing DOF')

    @classmethod
    def from_dof(cls, dof_ext=None, zshift1=None, zshift2=None, max_z_value=1.):  # setting1, setting2,
        return cls(dof_ext=dof_ext, zshift1=zshift1, zshift2=zshift2, max_z_value=max_z_value)
        # setting1=setting1 setting2=setting2

    # @classmethod
    # def from_tx_params(cls, optimal_tx_params, error_zvalue):
    #     return cls(tx_params=optimal_tx_params, error=error_zvalue)

    def dof_convert9(self, number_of_coordinates=3):
        """convert input degrees of freedom to 9-dim arrays, repeat DOF ext for each set of 3 coordinates (3 sets)"""
        self.dof9 = np.zeros((self.n_dof, 9))
        for i in range(self.n_dof):
            self.dof9[i] = (np.array(number_of_coordinates * [self.dof[i]])).flatten()
            # dof[i] = (np.array(3 * [self.dof_ext[i]])).flatten()
        self.dof9_t = self.dof9
        self.dof9 = np.transpose(self.dof9)

    def solve_optimal_shift(self, coords1, coords2, coords_rmsd_reference):
        """This routine does the work to solve the optimal shift problem

        Args:
            coords1 (np.ndarray): A 3 x 3 array with cartesian coordinates
            coords2 (np.ndarray): A 3 x 3 array with cartesian coordinates
            coords_rmsd_reference (float): The reference deviation to compare to the coords1 and coords2 error
        Keyword Args:
            max_z_value=1 (float): The maximum initial error tolerated
        Returns:
            (list(list)), (float): Returns the optimal translation for the set of coordinates
        """

        # form the guide coords into a matrix (column vectors)
        guide_target_10 = np.transpose(coords1)
        guide_query_10 = np.transpose(coords2)

        # calculate the initial difference between query and target (9 dim vector)
        guide_delta = np.transpose([guide_target_10.flatten('F') - guide_query_10.flatten('F')])
        # flatten column vector matrix above [[x, x, x], [y, y, y], [z, z, z]] -> [x, y, z, x, y, z, x, y, z], then T

        # isotropic case based on simple rmsd
        var_tot_inv = np.zeros([9, 9])
        for i in range(9):
            # fill in var_tot_inv with 1/ 3x the mean squared deviation (deviation sum)
            var_tot_inv[i, i] = 1. / (3. * coords_rmsd_reference ** 2)

        # solve the problem using 9-dim degrees of freedom arrays
        # self.dof9 is column major (9 x n_dof_ext) degree of freedom matrix
        # self.dof9_t transpose (row major: n_dof_ext x 9)
        dinvv = np.matmul(var_tot_inv, self.dof9)  # 1/variance x degree of freedom = (9 x n_dof)
        vtdinvv = np.matmul(self.dof9_t, dinvv)  # transpose of degrees of freedom (n_dof x 9) x (9 x n_dof) = (n_dof x n_dof)
        vtdinvvinv = np.linalg.inv(vtdinvv)  # Inverse of above - (n_dof x n_dof)

        dinvdelta = np.matmul(var_tot_inv, guide_delta)  # 1/variance (9 x 9) x guide atom diff (9 x 1) = (9 x 1)
        vtdinvdelta = np.matmul(self.dof9_t, dinvdelta)  # transpose of degrees of freedom (n_dof x 9) x (9 x 1) = (n_dof x 1)

        shift = np.matmul(vtdinvvinv, vtdinvdelta)  # (n_dof x n_dof) x (n_dof x 1) = (n_dof x 1)

        # get error value
        resid = np.matmul(self.dof9, shift) - guide_delta
        resid_t = np.transpose(resid)

        error = sqrt(np.matmul(resid_t, resid) / float(3.0)) / coords_rmsd_reference  # NEW. Is float(3.0) a scale?
        # sqrt(variance / 3) / cluster_rmsd # old error

        if error <= self.max_z_value:
            # print('Found match (shift, rmsd ref)', shift, coords_rmsd_reference)
            return shift[:, 0].tolist()  # , error
        else:
            return None

    def apply(self, coords1=None, coords2=None, coords_rmsd_reference=None):  # UNUSED
        """Apply Setting Matrix to provided Coords and solve for the translational shifts to overlap them"""
        # coords1_set = self.set_coords(self.setting1, coords1)
        # coords2_set = self.set_coords(self.setting2, coords2)
        coords1_set = np.matmul(coords1, np.transpose(self.setting1))
        coords2_set = np.matmul(coords2, np.transpose(self.setting2))

        # solve for shifts and resulting error
        return self.solve_optimal_shift(coords1_set, coords2_set, coords_rmsd_reference)
