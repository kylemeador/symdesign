from math import sqrt

import numpy as np


class OptimalTx:
    def __init__(self, setting1, setting2, is_zshift1, is_zshift2, cluster_rmsd, guide_atom_coods1, guide_atom_coods2, dof_ext):
        self.setting1 = np.array(setting1)
        self.setting2 = np.array(setting2)
        self.is_zshift1 = is_zshift1
        self.is_zshift2 = is_zshift2
        self.dof_ext = np.array(dof_ext)
        self.n_dof_external = len(self.dof_ext)
        self.cluster_rmsd = cluster_rmsd
        self.guide_atom_coods1 = guide_atom_coods1
        self.guide_atom_coods2 = guide_atom_coods2

        self.n_dof_internal = [self.is_zshift1, self.is_zshift2].count(True)
        self.optimal_tx = (np.array([]), float('inf'))  # (shift, error_zvalue)
        self.guide_atom_coods1_set = []
        self.guide_atom_coods2_set = []

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
        guide_target_10 = np.transpose(np.array(self.guide_atom_coods1_set))
        guide_query_10 = np.transpose(np.array(self.guide_atom_coods2_set))

        # calculate the initial difference between query and target (9 dim vector)
        guide_delta = np.transpose([guide_target_10.flatten('F') - guide_query_10.flatten('F')])

        # isotropic case based on simple rmsd
        self.cluster_rmsd = max(self.cluster_rmsd, 0.01)
        diagval = 1. / (3. * self.cluster_rmsd ** 2)
        var_tot_inv = np.zeros([9, 9])
        for i in range(9):
            var_tot_inv[i, i] = diagval

        # add internal z-shift degrees of freedom to 9-dim arrays if they exist
        if self.is_zshift1:
            self.dof_ext = np.append(self.dof_ext, -self.setting1[:, 2:3].T, axis=0)
        if self.is_zshift2:
            self.dof_ext = np.append(self.dof_ext, self.setting2[:, 2:3].T, axis=0)

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

    def apply(self):
        # Apply Setting Matrix to Guide Atoms
        self.guide_atom_coods1_set = self.set_guide_atoms(self.setting1, self.guide_atom_coods1)
        self.guide_atom_coods2_set = self.set_guide_atoms(self.setting2, self.guide_atom_coods2)

        # solve for shifts and resulting error
        self.solve_optimal_shift()

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

