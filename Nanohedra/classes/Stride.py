import os
import subprocess


class Stride:
    """
    H	    Alpha helix
    G	    3-10 helix
    I	    PI-helix
    E	    Extended conformation
    B or	b   Isolated bridge
    T	    Turn
    C	    Coil (none of the above)
    """
    def __init__(self, pdbfilepath, chain="A", stride_exe_path='./stride/stride'):
        self.pdbfilepath = pdbfilepath
        self.ss_asg = []
        self.chain = chain
        self.stride_exe_path = stride_exe_path

    def run(self):
        try:
            with open(os.devnull, 'w') as devnull:
                stride_out = subprocess.check_output([self.stride_exe_path, '%s' %self.pdbfilepath, '-c%s' %self.chain], stderr=devnull)

        except:
            stride_out = None

        if stride_out is not None:
            lines = stride_out.split('\n')
            for line in lines:
                if line[0:3] == "ASG":
                    self.ss_asg.append((int(filter(str.isdigit, line[10:15].strip())), line[24:25]))

    def is_N_Helical(self):
        if len(self.ss_asg) >= 10:
            for i in range(5):
                temp_window = ''.join([self.ss_asg[0+i:5+i][j][1] for j in range(5)])
                res_number = self.ss_asg[0+i:5+i][0][0]
                if "HHHHH" in temp_window:
                    return True, res_number
        return False, None

    def is_C_Helical(self):
        if len(self.ss_asg) >= 10:
            for i in range(5):
                reverse_ss_asg = self.ss_asg[::-1]
                temp_window = ''.join([reverse_ss_asg[0+i:5+i][j][1] for j in range(5)])
                res_number = reverse_ss_asg[0+i:5+i][4][0]
                if "HHHHH" in temp_window:
                    return True, res_number
        return False, None
