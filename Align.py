import argparse
import math
import os

from Pose import Model
from Structure import superposition3d


class ListFile:
    def __init__(self):
        self.path = None
        self.list_file = []

    def read(self, path):
        self.path = path
        f = open(self.path, "r")
        file_lines = f.readlines()
        f.close()

        for line in file_lines:
            if line != '\n':
                line = line.rstrip()
                self.list_file.append(line)


class FetchPDBBA:
    def __init__(self, pdb_listfile_path):
        lf = ListFile()
        lf.read(pdb_listfile_path)
        self.pdblist = lf.list_file

    def fetch(self):
        print('FETCHING PDB FILES')
        for pdb in self.pdblist:
            os.system('wget https://files.rcsb.org/download/%s.pdb1 >> fetch_pdb.out 2>&1' % pdb.rstrip())
        print('DONE FETCHING PDB FILES')


class AtomPair:
    def __init__(self, atom1, atom2):
        self.atom1 = atom1
        self.atom2 = atom2
        self.distance = atom1.distance(atom2)

    def __str__(self):
        return "%s - %s: %f" % (self.atom1.residue_number, self.atom2.residue_number, self.distance)


class PDBOverlap:
    def __init__(self, coords_fixed, coords_moving):
        self.coords_fixed = coords_fixed
        self.coords_moving = coords_moving

    def vdot3(self, a, b):
        dot = 0.
        for i in range(3):
            dot += a[i] * b[i]
        return dot

    def vnorm3(self, a):
        b = [0., 0., 0.]
        dot = 0.
        for i in range(3):
            dot += a[i] * a[i]
        for i in range(3):
            b[i] = a[i] / math.sqrt(dot)
        return b

    def vcross(self, a, b):
        c = [0., 0., 0.]
        for i in range(3):
            c[i] = a[(i + 1) % 3] * b[(i + 2) % 3] - a[(i + 2) % 3] * b[(i + 1) % 3]
        return c

    def inv3(self, a):
        ainv = [[0. for j in range(3)] for i in range(3)]
        det = 0.
        for i in range(3):
            det += a[(i + 0) % 3][0] * a[(i + 1) % 3][1] * a[(i + 2) % 3][2]
            det -= a[(i + 0) % 3][2] * a[(i + 1) % 3][1] * a[(i + 2) % 3][0]

        for i in range(3):
            for j in range(3):
                ainv[j][i] = (a[(i + 1) % 3][(j + 1) % 3] * a[(i + 2) % 3][(j + 2) % 3] - a[(i + 2) % 3][(j + 1) % 3] *
                              a[(i + 1) % 3][(j + 2) % 3]) / det
        return ainv

    def mat_vec_mul3(self, a, b):
        c = [0. for i in range(3)]
        for i in range(3):
            c[i] = 0.
            for j in range(3):
                c[i] += a[i][j] * b[j]
        return c

    def apply(self, rot, tx, moving):
        moved = []
        for coord in moving:
            coord_moved = self.mat_vec_mul3(rot, coord)
            for j in range(3):
                coord_moved[j] += tx[j]
            moved.append(coord_moved)
        return moved

    def get_rmsd(self, moving, fixed):
        n = len(moving)
        rmsd = 0.
        for i in range(n):
            for j in range(3):
                rmsd += (moving[i][j] - fixed[i][j]) ** 2
        rmsd = math.sqrt(rmsd / n)
        return rmsd

    def overlap(self):
        n = len(self.coords_moving)
        m = len(self.coords_fixed)
        if (n != m):
            print("Length of matching coordinates must match!")
        #print("# of atoms for overlapping = ", n)

        # calculate centers of mass
        else:
            cm_fixed = [0. for j in range(3)]
            cm_moving = [0. for j in range(3)]
            n = len(self.coords_fixed)
            for i in range(n):
                # print(coords_fixed[i][0], coords_fixed[i][1], coords_fixed[i][2])
                for j in range(3):
                    cm_fixed[j] = cm_fixed[j] + self.coords_fixed[i][j]
            for j in range(3):
                cm_fixed[j] = cm_fixed[j] / n
                # print(cm_fixed)
            n = len(self.coords_moving)
            for i in range(n):
                # print(coords_moving[i][0], coords_moving[i][1], coords_moving[i][2])
                for j in range(3):
                    cm_moving[j] = cm_moving[j] + self.coords_moving[i][j]
            for j in range(3):
                cm_moving[j] = cm_moving[j] / n
                # print(cm_moving)

            # form 3x3 matrices as the sums of outer products
            f_m_mat = [[0. for j in range(3)] for i in range(3)]
            m_m_mat = [[0. for j in range(3)] for i in range(3)]
            for k in range(n):
                for i in range(3):
                    for j in range(3):
                        f_m_mat[i][j] += (self.coords_fixed[k][i] - cm_fixed[i]) * (self.coords_moving[k][j] - cm_moving[j])
                        m_m_mat[i][j] += (self.coords_moving[k][i] - cm_moving[i]) * (self.coords_moving[k][j] - cm_moving[j])

            # solve for best transformation matrix (which could include a stretch)
            m_m_inv = self.inv3(m_m_mat)
            rot_mat = [[0. for j in range(3)] for i in range(3)]
            rot = [[0. for j in range(3)] for i in range(3)]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        rot_mat[i][j] = rot_mat[i][j] + f_m_mat[i][k] * m_m_inv[k][j]

            # extract rotation part from transformation matrix
            rot[0] = self.vnorm3(rot_mat[0])
            dotp = self.vdot3(rot_mat[1], rot[0])
            for i in range(3):
                rot[1][i] = rot_mat[1][i] - dotp * rot[0][i]
            rot[1] = self.vnorm3(rot[1])
            rot[2] = self.vcross(rot[0], rot[1])

            # for translational part of transformation from rot mat and centers of mass
            tx = self.mat_vec_mul3(rot, cm_moving)
            for i in range(3):
                tx[i] = cm_fixed[i] - tx[i]

            # apply transformation to moving coordinates
            coords_moved = self.apply(rot, tx, self.coords_moving)

            # calculate rmsd between moved coordinates and fixed coordinates
            rmsd = self.get_rmsd(coords_moved, self.coords_fixed)

            return rot, tx, rmsd, coords_moved


class AngleDistance:
    def __init__(self, axis1, axis2):
        self.axis1 = axis1
        self.axis2 = axis2
        self.vec1 = [axis1[2][0] - axis1[0][0], axis1[2][1] - axis1[0][1], axis1[2][2] - axis1[0][2]]
        self.vec2 = [axis2[2][0] - axis2[0][0], axis2[2][1] - axis2[0][1], axis2[2][2] - axis2[0][2]]
        self.length_1 = self.length(self.vec1)
        self.length_2 = self.length(self.vec2)

    def length(self, vec):
        length = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
        return length

    def cos_angle(self):
        if self.length_1 != 0 and self.length_2 !=0:
            cosangle = (self.vec1[0] / self.length_1) * (self.vec2[0] / self.length_2) + (self.vec1[1] / self.length_1) * (self.vec2[1] / self.length_2) +(self.vec1[2] / self.length_1) * (self.vec2[2] / self.length_2)
            return cosangle
        else:
            return 0

    def angle(self):
        angle = (math.acos(abs(self.cos_angle()))*180)/math.pi
        return angle

    def distance(self):
        crossproduct = [self.vec1[1] * self.vec2[2] - self.vec1[2] * self.vec2[1], self.vec1[2] * self.vec2[0] - self.vec1[0] * self.vec2[2], self.vec1[0] * self.vec2[1] - self.vec1[1] * self.vec2[0]]
        crossproduct_length = math.sqrt((crossproduct[0] * crossproduct[0]) + (crossproduct[1] * crossproduct[1]) + (crossproduct[2] * crossproduct[2]))
        connect_vec1_vec2 = [self.axis1[0][0] - self.axis2[0][0], self.axis1[0][1] - self.axis2[0][1], self.axis1[0][2] - self.axis2[0][2]]
        distance = abs(crossproduct[0] * connect_vec1_vec2[0] + crossproduct[1] * connect_vec1_vec2[1] + crossproduct[2] * connect_vec1_vec2[2]) / float(crossproduct_length)
        return distance

    def is_parallel(self, err=5):
        if (self.angle() >= 180 - err and self.angle() <= 180) or (self.angle() >= 0 and self.angle() <= 0 + err):
            return True
        else:
            return False

    def is_90(self, err=10):
        if (self.angle() >= 90 - err and self.angle() <= 90) or (self.angle() >= 90 and self.angle() <= 90 + err):
            return True
        else:
            return False
    def is_35(self, err=10):
        if (self.angle() >= 35 - err and self.angle() <= 35) or (self.angle() >= 35 and self.angle() <= 35 + err):
            return True
        else:
            return False

    def is_55(self, err=10):
        if (self.angle() >= 55 - err and self.angle() <= 55) or (self.angle() >= 55 and self.angle() <= 55 + err):
            return True
        else:
            return False


class Orient:
    def __init__(self, pdblist):
        self.pdblist = pdblist

    def run(self, symm):
        for pdb in self.pdblist:
            if os.path.exists(pdb):
                os.system('cp %s input.pdb' %pdb)
                os.system('./helix_fusion_tool/orient_oligomer >> orient.out 2>&1 << eof\n./helix_fusion_tool/%s_symm.txt\neof' %symm)
                os.system('mv output.pdb %s_orient.pdb' %os.path.splitext(pdb)[0])
            os.system('find -type f -size 0 -delete')
            os.system('rm input.pdb')


class HelixFusion:
    def __init__(self, target_protein_path,  targetprotein_term, targetprotein_symm, orient_target, add_target_helix, oligomer_list_path, oligomer_term, oligomer_symm, work_dir):
        self.target_protein_path = target_protein_path
        self.targetprotein_term = targetprotein_term
        self.targetprotein_symm = targetprotein_symm
        self.orient_target = orient_target
        self.add_target_helix = add_target_helix  # bool?, termini, chain id
        self.oligomer_list_path = oligomer_list_path
        self.oligomer_term = oligomer_term
        self.oligomer_symm = oligomer_symm
        self.work_dir = work_dir

    def run(self):
        # Make Directory for Design Candidates if it Doesn't Exist Already
        design_directory = os.path.join(self.work_dir, 'DESIGN_CANDIDATES')
        # if not os.path.exists(design_directory):
        os.makedirs(design_directory, exist_ok=True)
        # Orient Target Protein if desired
        if self.orient_target:
            if os.path.exists(self.target_protein_path):
                print('Orienting Target Molecule')
                orient_target = Orient([self.target_protein_path])
                orient_target.run(self.targetprotein_symm)
            else:
                print('Could Not Find Target PDB File')
                return -1

        # Read in Fixed PDB file or Oriented PDB File
        if self.orient_target:
            orient_target_path = os.path.splitext(self.target_protein_path)[0] + '_orient.pdb'
            if os.path.exists(orient_target_path):
                print('Done Orienting Target Molecule')
                target_protein = Model.from_file(orient_target_path)
            else:
                print('Could Not Orient Target Molecule')
                return -1
        else:
            target_protein = Model.from_file(self.target_protein_path)

        # Add Ideal 10 Ala Helix to Target if desired
        if self.add_target_helix[0]:
            print('Adding Ideal Ala Helix to Target Molecule')
            target_protein.add_ideal_helix(self.add_target_helix[1], self.add_target_helix[2])
            if self.add_target_helix[1] == 'N':
                target_term_resi = target_protein.chain(target_protein.chain_ids[self.add_target_helix[2]])[0].residue_number
            elif self.add_target_helix[1] == 'C':
                target_term_resi = target_protein.chain(target_protein.chain_ids[self.add_target_helix[2]])[-1].residue_number - 9
            else:
                print('Select N or C Terminus for Target Molecule')
                return -1
            print("Done Adding Ideal Ala Helix to Target Molecule")

        # Run Stride On Target Protein
        # else:
        #     if self.targetprotein_term == "N" or self.targetprotein_term == "C":
        #         stride_target = Stride(self.target_protein_path)
        #         stride_target.run()
        #         if len(stride_target.ss_asg) > 0:
        #             if self.targetprotein_term == "N":
        #                 target_term_resi = stride_target.is_n_term_helical()[1]
        #                 print("Done Running Stride On Target Molecule")
        #             else:
        #                 target_term_resi = stride_target.is_c_term_helical()[1]
        #                 print("Done Running Stride On Target Molecule")
        #         else:
        #             print("Error Running Stride On Target Molecule")
        #             return -1
        #     else:
        #         print("Select N or C Terminus for Target Molecule")
        #         return -1

        else:
            target_term_resi = self.add_target_helix[1]

        # Add Axis / Axes to Target Molecule
        if self.targetprotein_symm[0:1] == 'C':
            target_protein.AddCyclicAxisZ()
        elif self.targetprotein_symm == 'D2':
            target_protein.AddD2Axes()
        else:
            print('Target Protein Symmetry Not Supported')
            return -1

        # Fetch Oligomer PDB files
        fetch_oligomers = FetchPDBBA(self.oligomer_list_path)
        fetch_oligomers.fetch()

        # Try To Correct State issues
        print('Trying To Correct State Issues')
        oligomer_id_listfile = ListFile()
        oligomer_id_listfile.read(self.oligomer_list_path)
        oligomer_id_list = oligomer_id_listfile.list_file
        for oligomer_id in oligomer_id_list:
            oligomer_filepath = os.path.join(self.work_dir, '%s.pdb1' % oligomer_id)
            correct_oligomer_state = Model.from_file(oligomer_filepath)
            correct_sate_out_path = os.path.splitext(oligomer_filepath)[0] + '.pdb'
            correct_sate_out = open(correct_sate_out_path, 'w')
            for atom in correct_oligomer_state.all_atoms:
                correct_sate_out.write(str(atom))
            correct_sate_out.close()

        # Orient Oligomers
        correct_state_oligomer_filepath_list = []
        for oligomer_id in oligomer_id_list:
            correct_state_oligomer_filepath = os.path.join(self.work_dir, '%s.pdb' % oligomer_id)
            correct_state_oligomer_filepath_list.append(correct_state_oligomer_filepath)
        print('Orienting Oligomers')
        orient_oligomers = Orient(correct_state_oligomer_filepath_list)
        orient_oligomers.run(self.oligomer_symm)
        print('Done Orienting Oligomers')

        print('Fusing Target To Oligomers')
        for oligomer_id in oligomer_id_list:
            oriented_oligomer_filepath = os.path.join(self.work_dir, '%s_orient.pdb' % oligomer_id)
            if os.path.isfile(oriented_oligomer_filepath):
                for i in range(6):
                    # Read in Moving PDB
                    pdb_oligomer = Model.from_file(oriented_oligomer_filepath)

                    # Run Stride On Oligomer
                    if self.oligomer_term in ['N', 'C']:
                        raise NotImplementedError('Need to rework Stride execution here')
                        # stride_oligomer = Stride(oriented_oligomer_filepath)
                        # stride_oligomer.run()
                        if self.oligomer_term == 'N':
                            oligomer_term_resi = stride_oligomer.is_n_term_helical()[1]
                        elif self.oligomer_term == 'C':
                            oligomer_term_resi = stride_oligomer.is_c_term_helical()[1]
                    else:
                        print('Select N or C Terminus For Oligomer')
                        return -1

                    if type(oligomer_term_resi) is int:
                        # Add Axis / Axes to Oligomers
                        if self.oligomer_symm[:1] == 'C':
                            pdb_oligomer.AddCyclicAxisZ()
                        elif self.targetprotein_symm == 'D2':
                            pdb_oligomer.AddD2Axes()
                        else:
                            print('Oligomer Symmetry Not Supported')
                            return -1

                        # Extract coordinates of segment to be overlapped from PDB Fixed
                        pdb_fixed_coords = target_protein.chain(self.add_target_helix[2]).get_coords_subset(target_term_resi + i, target_term_resi + 4 + i)
                        # Extract coordinates of segment to be overlapped from PDB Moving
                        pdb_moble_coords = pdb_oligomer.chains[0].get_coords_subset(oligomer_term_resi, oligomer_term_resi + 4)

                        # Create PDBOverlap instance
                        # pdb_overlap = PDBOverlap(pdb_fixed_coords, pdb_moble_coords)
                        rmsd, rot, tx = superposition3d(pdb_fixed_coords, pdb_moble_coords)

                        # if pdb_overlap.overlap() != 'lengths mismatch':
                        #     # Calculate Optimal (rot, tx, rmsd, coords_moved)
                        #     rot, tx, rmsd, coords_moved = pdb_overlap.overlap()

                        # Apply optimal rot and tx to PDB moving axis (does NOT change axis coordinates in instance)
                        pdb_moving_axes = PDB()  # Todo this is outdated

                        if self.oligomer_symm == 'D2':
                            pdb_moving_axes.AddD2Axes()
                            pdb_moving_axes.transform(rot, tx)
                            moving_axis_x = pdb_moving_axes.axisX()
                            moving_axis_y = pdb_moving_axes.axisY()
                            moving_axis_z = pdb_moving_axes.axisZ()
                        elif self.oligomer_symm[0:1] == 'C':
                            pdb_moving_axes.AddCyclicAxisZ()
                            pdb_moving_axes.transform(rot, tx)
                            moving_axis_z = pdb_moving_axes.axisZ()
                        else:
                            print('Oligomer Symmetry Not Supported')
                            return -1

                        # # Check Angle Between Fixed and Moved Axes

                        # D2_D2 3D Crystal Check
                        #angle_check_1 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                        # is_parallel_1 = angle_check_1.is_parallel()
                        #
                        # if is_parallel_1:
                        #     pdb_oligomer.apply(rot, tx)
                        #     pdb_oligomer.rename_chains(target_protein.chain_ids)
                        #
                        #     PDB_OUT = PDB()
                        #     PDB_OUT.read_atom_list(target_protein.all_atoms + pdb_oligomer.all_atoms)
                        #
                        #     out_path = design_directory + "/" + os.path.basename(self.target_protein_path)[0:4] + "_" + oligomer_id + "_" + str(i) + ".pdb"
                        #     outfile = open(out_path, "w")
                        #     for atom in PDB_OUT.all_atoms:
                        #         outfile.write(str(atom))
                        #     outfile.close()



                        # D2_C3 3D Crystal I4132 Check
                        # angle_check_1 = AngleDistance(target_protein.axisX(), moving_axis_z)
                        # is_90_1 = angle_check_1.is_90()
                        # angle_check_2 = AngleDistance(target_protein.axisY(), moving_axis_z)
                        # is_90_2 = angle_check_2.is_90()
                        # angle_check_3 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                        # is_90_3 = angle_check_3.is_90()
                        #
                        # angle_check_4 = AngleDistance(target_protein.axisX(), moving_axis_z)
                        # is_35_1 = angle_check_4.is_35()
                        # angle_check_5 = AngleDistance(target_protein.axisY(), moving_axis_z)
                        # is_35_2 = angle_check_5.is_35()
                        # angle_check_6 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                        # is_35_3 = angle_check_6.is_35()
                        #
                        # angle_check_7 = AngleDistance(target_protein.axisX(), moving_axis_z)
                        # is_55_1 = angle_check_7.is_55()
                        # angle_check_8 = AngleDistance(target_protein.axisY(), moving_axis_z)
                        # is_55_2 = angle_check_8.is_55()
                        # angle_check_9 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                        # is_55_3 = angle_check_9.is_55()
                        #
                        # check_90 = [is_90_1, is_90_2, is_90_3]
                        # check_35 = [is_35_1, is_35_2, is_35_3]
                        # check_55= [is_55_1, is_55_2, is_55_3]
                        #
                        # count_90 = 0
                        # for test in check_90:
                        #     if test is True:
                        #         count_90 = count_90 + 1
                        #
                        # count_35 = 0
                        # for test in check_35:
                        #     if test is True:
                        #         count_35 = count_35 + 1
                        #
                        # count_55 = 0
                        # for test in check_55:
                        #     if test is True:
                        #         count_55 = count_55 + 1
                        #
                        # if count_90 > 0 and count_35 > 0 and count_55 > 0:
                        #     for k in [0, 1, 2]:
                        #         if check_90[k] is True:
                        #             check_90_index = k
                        #
                        #     if check_90_index == 0:
                        #         axis_90 = target_protein.axisX()
                        #     elif check_90_index == 1:
                        #         axis_90 = target_protein.axisY()
                        #     else:
                        #         axis_90 = target_protein.axisZ()
                        #
                        #     distance_check_1 = AngleDistance(axis_90, moving_axis_z)
                        #
                        #     if distance_check_1.distance() <= 5:
                        #
                        #             pdb_oligomer.apply(rot, tx)
                        #             pdb_oligomer.rename_chains(target_protein.chain_ids)
                        #
                        #             PDB_OUT = PDB()
                        #             PDB_OUT.read_atom_list(target_protein.all_atoms + pdb_oligomer.all_atoms)
                        #
                        #             out_path = design_directory + "/" + os.path.basename(self.target_protein_path)[0:4] + "_" + oligomer_id + "_" + str(i) + ".pdb"
                        #             outfile = open(out_path, "w")
                        #             for atom in PDB_OUT.all_atoms:
                        #                 outfile.write(str(atom))
                        #             outfile.close()

                        # D2_C3 2D Layer Check p622 Check
                        angle_check_1 = AngleDistance(target_protein.axisX(), moving_axis_z)
                        is_parallel_1 = angle_check_1.is_parallel()
                        angle_check_2 = AngleDistance(target_protein.axisY(), moving_axis_z)
                        is_parallel_2 = angle_check_2.is_parallel()
                        angle_check_3 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                        is_parallel_3 = angle_check_3.is_parallel()

                        check_parallel = [is_parallel_1, is_parallel_2, is_parallel_3]
                        count_parallel = 0
                        for test in check_parallel:
                            if test is True:
                                count_parallel = count_parallel + 1

                        angle_check_4 = AngleDistance(target_protein.axisX(), moving_axis_z)
                        is_90_1 = angle_check_4.is_90()
                        angle_check_5 = AngleDistance(target_protein.axisY(), moving_axis_z)
                        is_90_2 = angle_check_5.is_90()
                        angle_check_6 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                        is_90_3 = angle_check_6.is_90()

                        check_90 = [is_90_1, is_90_2, is_90_3]
                        count_90 = 0
                        for test in check_90:
                            if test is True:
                                count_90 = count_90 + 1

                        if count_parallel > 0 and count_90 > 0:
                            for k in [0, 1, 2]:
                                if check_90[k] is True:
                                    check_90_index = k

                            if check_90_index == 0:
                                axis_90 = target_protein.axisX()
                            elif check_90_index == 1:
                                axis_90 = target_protein.axisY()
                            else:
                                axis_90 = target_protein.axisZ()

                            distance_check_1 = AngleDistance(axis_90, moving_axis_z)

                            if distance_check_1.distance() <= 3:
                                pdb_oligomer.apply(rot, tx)
                                pdb_oligomer.rename_chains(exclude_chains=target_protein.chain_ids)

                                out_pdb = Model.from_atoms(target_protein.atoms + pdb_oligomer.atoms)

                                out_path = os.path.join(design_directory,
                                                        '%s_%s_%d.pdb' % (os.path.basename(self.target_protein_path)[0:4], oligomer_id, i))
                                out_pdb.write(out_path=out_path)

        print('Done')


def align(pdb1_path, start_1, end_1, chain_1, pdb2_path, start_2, end_2, chain_2, extend_helix=False):
        pdb1 = Model.from_file(pdb1_path)
        pdb2 = Model.from_file(pdb2_path)

        if extend_helix:
            n_terminus = pdb1.chain(chain_1).n_terminal_residue.number
            if n_terminus in range(start_1, end_1) or n_terminus < start_1:
                term = 'N'
            else:
                term = 'C'
            print('Adding ideal helix to %s-terminus of reference molecule' % term)
            pdb1.add_ideal_helix(term, chain_1)  # terminus, chain number
        coords1 = pdb1.chain(chain_1).get_coords_subset(start_1, end_1)
        coords2 = pdb2.chain(chain_2).get_coords_subset(start_2, end_2)

        rmsd, rot, tx = superposition3d(coords1, coords2)
        pdb2.transform(rot, tx)

        return pdb2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=os.path.basename(__file__) +
                                     '\nTool for aligning terminal helices of two proteins')
    parser.add_argument('-r', '--reference_pdb', type=str, help='The disk location of pdb file to serve as reference')
    parser.add_argument('-s', '--ref_start_res', type=int, help='First residue in a range to serve as reference for '
                                                                'alignment')
    parser.add_argument('-e', '--ref_end_res', type=int, help='Last residue to serve as reference for alignment')
    parser.add_argument('-c', '--ref_chain', help='Chain ID of the reference moleulce, Default=A', default='A')
    parser.add_argument('-a', '--aligned_pdb', type=str, help='The disk location of pdb file to be aligned to the '
                                                              'reference')
    parser.add_argument('-as', '--align_start_res', type=int, help='First residue to align to reference')
    parser.add_argument('-ae', '--align_end_res', type=int, help='Last residue to align to reference')
    parser.add_argument('-ac', '--aligned_chain', help='Chain Id of the moving molecule')
    parser.add_argument('-x', '--extend_helical_termini', action='store_true',
                        help='Whether to extend the termini in question with an ideal 10 residue alpha helix. All '
                             'residue ranges will be modified accordingly. '
                             'Ex. --extend_helical_termini --ref_start_res 1 --ref_end_res 9 '
                             'will insert a 10 residue alpha helix to the reference range and perform alignment from '
                             'residue 1-9 of the extended alpha helix. Default=False')
    parser.add_argument('-o', '--out_file_path', type=str, help='The disk location of file containing a pdb to be moved')
    args = parser.parse_args()

    aligned_pdb = align(args.reference_pdb, args.ref_start_res, args.ref_end_res, args.ref_chain,
                        args.aligned_pdb, args.align_start_res, args.align_end_res, args.aligned_chain,
                        extend_helix=args.extend_helical_termini)
    aligned_pdb.write(args.out_file_path)
