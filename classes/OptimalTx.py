from math import sqrt

import numpy as np


class OptimalTx:
    def __init__(self, setting1=None, setting2=None, is_zshift1=None, is_zshift2=None, dof_ext=None, tx_params=None,
                 error=None):  # this was before dor_ext -> cluster_rmsd, guide_atom_coords1, guide_atom_coords2
        if setting1:
            self.setting1 = np.array(setting1)
        else:
            self.setting1 = np.array([])  # Todo make np.matmul handle an empty matrix

        if setting2:
            self.setting2 = np.array(setting2)
        else:
            self.setting2 = np.array([])  # Todo make np.matmul handle an empty matrix

        self.is_zshift1 = is_zshift1  # Whether or not the space has internal translational DOF
        self.is_zshift2 = is_zshift2  # Whether or not the space has internal translational DOF
        self.dof_ext = np.array(dof_ext)  # External translational DOF (number DOF external x 3)
        self.dof = self.dof_ext.copy()
        self.dof9 = None

        # add internal z-shift degrees of freedom to 9-dim arrays if they exist
        if self.is_zshift1:
            self.dof = np.append(self.dof, -self.setting1[:, 2:3].T, axis=0)
            # self.dof_ext = np.append(self.dof_ext, -self.setting1[:, 2:3].T, axis=0)
        if self.is_zshift2:
            self.dof = np.append(self.dof, self.setting2[:, 2:3].T, axis=0)
            # self.dof_ext = np.append(self.dof_ext, self.setting2[:, 2:3].T, axis=0)
        self.n_dof_internal = [self.is_zshift1, self.is_zshift2].count(True)

        if setting1:  # ensures that a setting matrix is passed in order to get the 9 dimensional dof
            self.n_dof_external = self.dof_ext.shape[0]  # get the length of the numpy array
            self.n_dof = self.dof.shape[0]
            self.dof_convert9()
        else:
            self.n_dof_external = 0

        # self.cluster_rmsd = cluster_rmsd
        # self.guide_atom_coords1 = guide_atom_coords1
        # self.guide_atom_coords1 = np.array(guide_atom_coords1)  for current instantiation, use std if passing np.array
        # self.guide_atom_coords2 = guide_atom_coords2
        # self.guide_atom_coords2 = np.array(guide_atom_coords2)
        # self.guide_atom_coords1_set = []
        # self.guide_atom_coords2_set = []
        if tx_params:
            self.optimal_tx = np.array(tx_params)
        else:
            self.optimal_tx = np.array([])  # , float('inf'))  # (shift, error_zvalue)
        if error:
            self.error_zvalue = error
        else:
            self.error_zvalue = float('inf')

    @classmethod
    def from_dof(cls, setting1, setting2, is_zshift1, is_zshift2, dof_ext):
        return cls(setting1=setting1, setting2=setting2, is_zshift1=is_zshift1, is_zshift2=is_zshift2, dof_ext=dof_ext)

    @classmethod
    def from_tx_params(cls, optimal_tx_params, error_zvalue):
        return cls(tx_params=optimal_tx_params, error=error_zvalue)

    def dof_convert9(self, number_of_coordinates=3):
        """convert input degrees of freedom to 9-dim arrays, repeat DOF ext for each set of 3 coordinates (3 sets)"""
        self.dof9 = np.zeros((self.n_dof, 9))
        for i in range(self.n_dof):
            self.dof9[i] = (np.array(number_of_coordinates * [self.dof[i]])).flatten()
            # dof[i] = (np.array(3 * [self.dof_ext[i]])).flatten()
        self.dof9 = np.transpose(self.dof9)

    def solve_optimal_shift(self, coords1, coords2, coords_rmsd_reference):
        """This routine does the work to solve the optimal shift problem

        Returns:
            (list(list)), (float): Returns the optimal translation for the set of coordinates and their error value
        """

        # form the guide coords into a matrix (column vectors)
        guide_target_10 = np.transpose(np.array(coords1))
        # guide_target_10 = np.transpose(np.array(self.guide_atom_coords1_set))
        guide_query_10 = np.transpose(np.array(coords2))
        # guide_query_10 = np.transpose(np.array(self.guide_atom_coords2_set))

        # calculate the initial difference between query and target (9 dim vector)
        guide_delta = np.transpose([guide_target_10.flatten('F') - guide_query_10.flatten('F')])
        # flatten column vector matrix above [[x, x, x], [y, y, y], [z, z, z]] -> [x, y, z, x, y, z, x, y, z], then T

        # isotropic case based on simple rmsd fill in var_tot_inv with 1/ 3x the mean squared deviation (deviation sum)
        coords_rmsd_reference = max(coords_rmsd_reference, 0.01)
        diagval = 1. / (3. * coords_rmsd_reference ** 2)
        var_tot_inv = np.zeros([9, 9])
        for i in range(9):
            var_tot_inv[i, i] = diagval

        # Use degrees of freedom 9-dim array
        # self.dof9 is column major (9 x n_dof_ext) degree of freedom matrix
        dof_t = np.transpose(self.dof9)  # degree of freedom transpose (row major: n_dof_ext x 9)

        # solve the problem
        # print('dof: %s' % self.dof)
        # print('9: %s' % self.dof9)
        # print('var_inv_tot: %s' % var_tot_inv)
        dinvv = np.matmul(var_tot_inv, self.dof9)  # 1/variance x degree of freedom = (9 x n_dof)
        vtdinvv = np.matmul(dof_t, dinvv)  # transpose of degrees of freedom (n_dof x 9) x (9 x n_dof) = (n_dof x n_dof)
        vtdinvvinv = np.linalg.inv(vtdinvv)  # Inverse of above - (n_dof x n_dof)

        dinvdelta = np.matmul(var_tot_inv, guide_delta)  # 1/variance x guide atom diff = (9 x 1)
        vtdinvdelta = np.matmul(dof_t, dinvdelta)  # transpose of degrees of freedom (n_dof x 9) x (9 x 1) = (n_dof x 1)

        shift = np.matmul(vtdinvvinv, vtdinvdelta)  # (n_dof x n_dof) x (n_dof x 1) = (n_dof x 1)

        # get error value
        resid = np.matmul(self.dof9, shift) - guide_delta
        resid_t = np.transpose(resid)

        error = sqrt(np.matmul(resid_t, resid) / float(3.0)) / coords_rmsd_reference  # NEW. Is float(3.0) a scale?
        # sqrt(variance / 3) / cluster_rmsd # old error

        # etmp = np.matmul(var_tot_inv, resid)  # need to comment out, old error
        # error = np.matmul(residT, etmp)  # need to comment out, old error

        return shift[:, 0], error  # (shift, error_zvalue)
        # self.optimal_tx = shift[:, 0]  # (shift, error_zvalue)
        # self.error_zvalue = error

    @staticmethod
    def mat_vec_mul3(a, b):
        c = [0. for i in range(3)]
        for i in range(3):
            # c[i] = 0.
            for j in range(3):
                c[i] += a[i][j] * b[j]

        return c

    # @staticmethod
    def set_coords(self, mat, coords):
        """Apply a matrix to a set of 3 sets of x, y, z coordinates"""
        # return np.matmul(mat, coords)  # Doesn't work because no transpose!
        # return np.matmul(coords, np.transpose(mat))  # Todo, check. This should work!
        return [self.mat_vec_mul3(mat, [coord[0], coord[1], coord[2]]) for coord in coords]

    def apply(self, coords1=None, coords2=None, coords_rmsd_reference=None):
        """Apply Setting Matrix to provided Coords and solve for the translational shifts to overlap them"""
        coords1_set = self.set_coords(self.setting1, coords1)
        coords2_set = self.set_coords(self.setting2, coords2)

        # solve for shifts and resulting error
        return self.solve_optimal_shift(coords1_set, coords2_set, coords_rmsd_reference)

    def get_optimal_tx_dof_int(self):
        index = self.n_dof_external

        tx_dof_int = []
        if self.is_zshift1:
            tx_dof_int.append(self.optimal_tx[index:index + 1][0])
            index += 1

        if self.is_zshift2:
            tx_dof_int.append(self.optimal_tx[index:index + 1][0])

        return tx_dof_int

    def get_optimal_tx_dof_ext(self):
        # shift, error_zvalue = self.optimal_tx
        return self.optimal_tx[:self.n_dof_external].tolist()

    def get_all_optimal_shifts(self):
        # shift, error_zvalue = self.optimal_tx
        return self.optimal_tx.tolist()

    def get_n_dof_external(self):
        return self.n_dof_external

    def get_n_dof_internal(self):
        return self.n_dof_internal

    def get_zvalue(self):
        # shift, error_zvalue = self.optimal_tx
        return self.error_zvalue
