#include "bmdca.h"
#include <carma>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


// Create binding, see pybind11 documentation for details
//void bind_load_couplings(py::module &m) {
//    m.def(
//        "load_couplings",
//        &load_couplings,
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
//void bind_load_fields(py::module &m) {
//    m.def(
//        "load_fields",
//        &load_fields,
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

//arma::field<arma::Mat<double>> load_couplings(std::string coupling_file, int size){
//    // Check file extensions and parse out file names.
//    int idx = coupling_file.find_last_of(".");
//    //std::string coupling_name = coupling_file.substr(0, idx);
//    std::string coupling_ext = coupling_file.substr(idx + 1);
//
//    if (coupling_ext != "bin") {
//        std::cerr << "ERROR: input coupling parameters " << coupling_file << " do not have 'bin' extension."
//                  << std::endl;
//        std::exit(EXIT_FAILURE);
//    }
//
//    arma::field<arma::Mat<double>> J(size, size);
//    J.load(coupling_file, arma::arma_binary);
//    // type caster will take care of casting `result` to a Numpy array.
//    return J;
//}

// This version takes the field, breaks each matrix to an individual array then returns full array
py::array_t<double> load_couplings(std::string coupling_file, int size, int aa, py::args args, py::kwargs kwargs){
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
    // Write J to array
    py::array_t<double> a;
    size_t N;
    bool refcheck;
    //a.flags(true);
    a.resize({size, size, aa, aa}, refcheck);
    auto r = a.mutable_unchecked<4>();
    //std::cout << a.shape(0) << a.shape(1) << a.shape(2) << a.shape(3) << std::endl;
    for (py::ssize_t i = 0; i < a.shape(0); i++) {
        for (py::ssize_t j = i + 1; j < a.shape(1); j++) {
            for (py::ssize_t aa1 = 0; aa1 < a.shape(2); aa1++) {
                for (py::ssize_t aa2 = 0; aa2 < a.shape(3); aa2++) {
                    //r(i, j, k, l) = 1.0;
                    //std::cout << i << ' ' << j << ' ' << aa1 << ' ' << aa2 << std::endl;
                    r(i, j, aa1, aa2) = J(i, j)(aa1, aa2);
                }
            }
        }
    }
    //for (int i = 0; i < size; i++) {
    //    for (int j = i + 1; j < size; j++) {
            //for (int aa1 = 0; aa1 < Q; aa1++) {
            //    for (int aa2 = 0; aa2 < Q; aa2++) {
            //      output_stream << "J " << i << " " << j << " " << aa1 << " " << aa2
            //                    << " " << J(i, j)(aa1, aa2) << std::endl;
            //    }
            //}
    //        mat_to_arr(arma::Mat<T>& J(i,j))  // returns py::array_t<T>, bool copy=false)

    //    }
    //}
    //return J;
    return a;
}

//void f(py::array_t<double, py::array::c_style | py::array::forcecast> array);
arma::Mat<double> load_fields(std::string bin_file, py::args args, py::kwargs kwargs) {
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
    // make the array c_contiguous for python currently handled in load
    //py::array_t<double> a = carma::mat_to_array(m, false)
    // //to_numpy(armaT<eT>& src, bool copy=false) // template <typename armaT> py::array_t<eT>
    // if (is_f_contiguous(const & a); //bool is_f_contiguous(const py::array_t<T> & arr)

    // type caster will take care of casting `result` to a Numpy array.
    return m;
}

PYBIND11_MODULE(bmdca, m) {
    m.doc() = "Convert armadillo binaries from bmDCA to numpy.ndarray objects"; // module docstring
    m.def("load_couplings", &load_couplings, "A function which loads an armadillo binary coupling field with 4 dimensions");
    m.def("load_fields", &load_fields, "A function which loads a binary fields matrix to a numpy array");
}