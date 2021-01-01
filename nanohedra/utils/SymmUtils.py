def parse_uc_str_to_tuples(uc_string):
    """Acquire unit cell parameters from specified external degrees of freedom string"""
    def s_to_l(string):
        s1 = string.replace('(', '')
        s2 = s1.replace(')', '')
        l1 = s2.split(',')
        l2 = [x.replace(' ', '') for x in l1]
        return l2

    if '),' in uc_string:
        l = uc_string.split('),')
    else:
        l = [uc_string]

    return [s_to_l(s) for s in l]


def get_uc_var_vec(string_vec, var):
    """From the length specification return the unit vector"""
    return_vec = [0.0, 0.0, 0.0]
    for i in range(len(string_vec)):
        if var in string_vec[i] and '*' in string_vec[i]:
            return_vec[i] = (float(string_vec[i].split('*')[0]))
        elif var == string_vec[i]:
            return_vec.append(1.0)
    return return_vec


def get_uc_dimensions(uc_string, e=1, f=0, g=0):
    """Return an array with the three unit cell lengths and three angles [20, 20, 20, 90, 90, 90] by combining UC
    basis vectors with component translation degrees of freedom"""
    uc_string_vec = parse_uc_str_to_tuples(uc_string)

    lengths = [0.0, 0.0, 0.0]
    string_vec_lens = uc_string_vec[0]
    e_vec = get_uc_var_vec(string_vec_lens, 'e')
    f_vec = get_uc_var_vec(string_vec_lens, 'f')
    g_vec = get_uc_var_vec(string_vec_lens, 'g')
    e1 = [e_vec_val * e for e_vec_val in e_vec]
    f1 = [f_vec_val * f for f_vec_val in f_vec]
    g1 = [g_vec_val * g for g_vec_val in g_vec]
    for i in range(len(string_vec_lens)):
        lengths[i] = abs((e1[i] + f1[i] + g1[i]))
    if len(string_vec_lens) == 2:
        lengths[2] = 1.0

    string_vec_angles = uc_string_vec[1]
    if len(string_vec_angles) == 1:
        angles = [90.0, 90.0, float(string_vec_angles[0])]
    else:
        angles = [0.0, 0.0, 0.0]
        for i in range(len(string_vec_angles)):
            angles[i] = float(string_vec_angles[i])

    uc_dimensions = lengths + angles

    return uc_dimensions
