from Structure import Atom


def x_axis(self):
    axis_a = Atom(1, "CA", " ", "GLY", "9", 1, " ", 80.000, 0.000, 0.000, 1.00, 20.00, "C", "")
    axis_b = Atom(2, "CA", " ", "GLY", "9", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
    axis_c = Atom(3, "CA", " ", "GLY", "9", 3, " ", -80.000, 0.000, 0.000, 1.00, 20.00, "C", "")

    return [axis_a, axis_b, axis_c]
    # self.atoms.extend(axis)


def y_axis(self):
    axis_a = Atom(1, "CA", " ", "GLY", "8", 1, " ", 0.000, 80.000, 0.000, 1.00, 20.00, "C", "")
    axis_b = Atom(2, "CA", " ", "GLY", "8", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
    axis_c = Atom(3, "CA", " ", "GLY", "8", 3, " ", 0.000, -80.000, 0.000, 1.00, 20.00, "C", "")

    return [axis_a, axis_b, axis_c]
    # self.atoms.extend(axis)


def z_axis(self):
    axis_a = Atom(1, "CA", " ", "GLY", "7", 1, " ", 0.000, 0.000, 80.000, 1.00, 20.00, "C", "")
    axis_b = Atom(2, "CA", " ", "GLY", "7", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
    axis_c = Atom(3, "CA", " ", "GLY", "7", 3, " ", 0.000, 0.000, -80.000, 1.00, 20.00, "C", "")

    return [axis_a, axis_b, axis_c]
    # self.atoms.extend(axis)


def create_axes(self):
    x = z_axis()
    y = z_axis()
    z = z_axis()

    return x + y + z
    # self.atoms.extend(axes)
