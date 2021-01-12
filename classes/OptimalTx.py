from math import sqrt

import numpy as np


class OptimalTx:
    def __init__(self, setting1, setting2, is_zshift1, is_zshift2, cluster_rmsd, guide_atom_coords1, guide_atom_coords2,
                 dof_ext):
        self.setting1 = np.array(setting1)
        self.setting2 = np.array(setting2)
        self.is_zshift1 = is_zshift1  # Whether or not the space has internal translational DOF
        self.is_zshift2 = is_zshift2  # Whether or not the space has internal translational DOF
        self.dof_ext = np.array(dof_ext)  # External translational DOF (number DOF external x 3)
        self.n_dof_external = self.dof_ext.__len__
        self.cluster_rmsd = cluster_rmsd
        self.guide_atom_coords1 = guide_atom_coords1
        self.guide_atom_coords2 = guide_atom_coords2

        self.n_dof_internal = [self.is_zshift1, self.is_zshift2].count(True)
        self.optimal_tx = (np.array([]), float('inf'))  # (shift, error_zvalue)
        self.guide_atom_coords1_set = []
        self.guide_atom_coords2_set = []

    def dof_convert9(self):
        # convert input degrees of freedom to 9-dim arrays, repeat DOF ext for each guide coordinate (3 sets of x, y, z)
        dof = np.zeros((self.get_n_dof_external(), 9))
        for i in range(self.get_n_dof_external()):
            dof[i] = (np.array(3 * [self.dof_ext[i]])).flatten()
        return np.transpose(dof)

    def solve_optimal_shift(self):
        # This routine does the work to solve the optimal shift problem

        # form the guide atoms into a matrix (column vectors)
        guide_target_10 = np.transpose(np.array(self.guide_atom_coords1_set))
        guide_query_10 = np.transpose(np.array(self.guide_atom_coords2_set))

        # calculate the initial difference between query and target (9 dim vector)
        guide_delta = np.transpose([guide_target_10.flatten('F') - guide_query_10.flatten('F')])
        # flatten column vector matrix above [[x, x, x], [y, y, y], [z, z, z]] -> [x, y, z, x, y, z, x, y, z], then T

        # isotropic case based on simple rmsd
        self.cluster_rmsd = max(self.cluster_rmsd, 0.01)
        diagval = 1. / (3. * self.cluster_rmsd ** 2)  # fill in values with 3x the mean squared deviation (deviation sum)
        var_tot_inv = np.zeros([9, 9])
        for i in range(9):
            var_tot_inv[i, i] = diagval

        # add internal z-shift degrees of freedom to 9-dim arrays if they exist
        if self.is_zshift1:
            self.dof_ext = np.append(self.dof_ext, -self.setting1[:, 2:3].T, axis=0)
        if self.is_zshift2:
            self.dof_ext = np.append(self.dof_ext, self.setting2[:, 2:3].T, axis=0)

        # convert degrees of freedom to 9-dim array
        dof = self.dof_convert9()  # column major (9 x n_dof_ext) degree of freedom matrix
        dofT = np.transpose(dof)  # degree of freedom transpose (row major - n_dof_ext x 9)

        # solve the problem
        dinvv = np.matmul(var_tot_inv, dof)  # 1/variance x degree of freedom = (9 x n_dof)
        vtdinvv = np.matmul(dofT, dinvv)  # degree of freedom transpose x (9 x n_dof) = (n_dof x n_dof)
        vtdinvvinv = np.linalg.inv(vtdinvv)  # Inverse of above - (n_dof x n_dof)

        dinvdelta = np.matmul(var_tot_inv, guide_delta)  # 1/variance x guide atom diff = (9 x 1)
        vtdinvdelta = np.matmul(dofT, dinvdelta)  # transpose of degrees of freedom x (9 x 1) = (n_dof x 1)

        shift = np.matmul(vtdinvvinv, vtdinvdelta)  # (n_dof x n_dof) x (n_dof x 1) = (n_dof x 1)

        # get error value
        resid = np.matmul(dof, shift) - guide_delta
        residT = np.transpose(resid)

        error = sqrt(np.matmul(residT, resid) / float(3.0)) / self.cluster_rmsd  # NEW ERROR. Is float(3.0) the scale?
        # sqrt(variance / 3) / cluster_rmsd # old error

        # etmp = np.matmul(var_tot_inv, resid)  # need to comment out, old error
        # error = np.matmul(residT, etmp)  # need to comment out, old error
        # self.optimal_tx = (shift[:, 0], error[0, 0])  # (shift, error_zvalue) # need to comment out, old error

        self.optimal_tx = (shift[:, 0], error)  # (shift, error_zvalue)

    @staticmethod
    def mat_vec_mul3(a, b):
        c = [0. for i in range(3)]
        for i in range(3):
            # c[i] = 0.
            for j in range(3):
                c[i] += a[i][j] * b[j]

        return c

    def set_guide_atoms(self, rot_mat, coords):
        return [list(self.mat_vec_mul3(rot_mat, [coord[0], coord[1], coord[2]])) for coord in coords]

    def apply(self):
        # Apply Setting Matrix to Guide Atoms
        self.guide_atom_coords1_set = self.set_guide_atoms(self.setting1, self.guide_atom_coords1)
        self.guide_atom_coords2_set = self.set_guide_atoms(self.setting2, self.guide_atom_coords2)

        # solve for shifts and resulting error
        self.solve_optimal_shift()

    def get_optimal_tx_dof_int(self):
        shift, error_zvalue = self.optimal_tx
        index = self.get_n_dof_external()

        tx_dof_int = []
        if self.is_zshift1:
            tx_dof_int.append(shift[index:index + 1][0])
            index += 1

        if self.is_zshift2:
            tx_dof_int.append(shift[index:index + 1][0])

        return tx_dof_int

    def get_optimal_tx_dof_ext(self):
        shift, error_zvalue = self.optimal_tx
        return shift[0:self.get_n_dof_external()].tolist()

    def get_all_optimal_shifts(self):
        shift, error_zvalue = self.optimal_tx
        return shift.tolist()

    def get_n_dof_external(self):
        return self.n_dof_external()

    def get_n_dof_internal(self):
        return self.n_dof_internal

    def get_zvalue(self):
        shift, error_zvalue = self.optimal_tx
        return error_zvalue
