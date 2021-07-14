#include <armadillo>
#include <carma>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


//void bind_load_couplings_to_numpy(py::module &m);
//void bind_load_matrix_to_numpy(py::module &m);
//arma::field<arma::Mat<double>> load_couplings_to_numpy(std::string coupling_file, int size, int aa);
py::array_t<double> load_couplings(std::string coupling_file, int size, int aa, py::args args, py::kwargs kwargs);
arma::Mat<double> load_fields(std::string fields_file, py::args args, py::kwargs kwargs);
