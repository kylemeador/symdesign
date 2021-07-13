#include "load_armadillo_to_numpy.h"
#include <carma>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


// Create binding, see pybind11 documentation for details
//void bind_load_couplings_to_numpy(py::module &m) {
//    m.def(
//        "load_couplings_to_numpy",
//        &load_couplings_to_numpy,
//        R"pbdoc(
//            Automatic conversion of armadillo binary J couplings to numpy.array
//            Parameters
//            ----------
//            coupling_file : str
//                The name of the coupling file
//            size : int
//                The expected size of the matrix
//            Returns
//            -------
//            result : np.array
//                output array with F order memory
//        )pbdoc",
//        py::arg("coupling_file")
//        py::arg("size")
//
//    );
//}

// Create binding, see pybind11 documentation for details
//void bind_load_matrix_to_numpy(py::module &m) {
//    m.def(
//        "load_matrix_to_numpy",
//        &load_matrix_to_numpy,
//        R"pbdoc(
//            Automatic conversion of armadillo matrix binary to numpy.ndarray
//            Parameters
//            ----------
//            bin_file : str
//                The name of the binary file
//            Returns
//            -------
//            result : np.ndarray
//                output array with F order memory
//        )pbdoc",
//        py::arg("bin_file")
//    );
//}

//arma::Mat<double> load_to_numpy(arma::Mat<double> & mat) {
//    // normally you do something useful here with mat ...
//    arma::Mat<double> rand = arma::Mat<double>(mat.n_rows, mat.n_cols, arma::fill::randu);
//
//    arma::Mat<double> result = mat + rand;
//    // type caster will take care of casting `result` to a Numpy array.
//    return result;
//}
//void
//convertParametersToAscii(std::string h_file, std::string J_file){

arma::field<arma::Mat<double>> load_couplings_to_numpy(std::string coupling_file, int size){
    // Check file extensions and parse out file names.
    int idx = coupling_file.find_last_of(".");
    //std::string coupling_name = coupling_file.substr(0, idx);
    std::string coupling_ext = coupling_file.substr(idx + 1);

    if (coupling_ext != "bin") {
        std::cerr << "ERROR: input coupling parameters " << coupling_file << " do not have 'bin' extension."
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    arma::field<arma::Mat<double>> J(size, size);
    J.load(coupling_file, arma::arma_binary);
    // type caster will take care of casting `result` to a Numpy array.
    return J;
}

arma::Mat<double> load_matrix_to_numpy(std::string bin_file) {
    int idx = bin_file.find_last_of(".");
    //std::string bin_name = bin_file.substr(0, idx);
    std::string bin_ext = bin_file.substr(idx + 1);

    if (bin_ext != "bin"){
        std::cerr << "ERROR: input fields parameters " << bin_file << " do not have 'bin' extension."
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    arma::Mat<double> m;

    m.load(bin_file, arma::arma_binary);
    //int N = h.n_cols; // length of sequence
    //int Q = h.n_rows; // length of alphabet
    // type caster will take care of casting `result` to a Numpy array.
    return m;
}

PYBIND11_MODULE(load_armadillo_to_numpy, m) {
    m.doc() = "convert armadillo binaries to numpy"; // module docstring

    m.def("load_couplings_to_numpy", &load_couplings_to_numpy, "A function which adds two numbers");
    m.def("load_matrix_to_numpy", &load_matrix_to_numpy, "A function which loads a binary matrix to a numpy array");
}