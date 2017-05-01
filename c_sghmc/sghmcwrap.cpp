<%
cfg['compiler_args'] = ['-std=c++11']
cfg['include_dirs'] = ['./eigen']
setup_pybind11(cfg)
%>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <chrono>
#include <random>
#include <Eigen/Cholesky> 
#include <Eigen/LU>
#include <pybind11/functional.h>

namespace py = pybind11;
 
    
// sghmc function
float sghmc(const std::function<float(float)> &U, const std::function<float(float)> &gradU, float M, float epsilon, int m, float theta, float C, float V) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution (0, 1);
    float r;
    r=distribution(generator)*pow(M,0.5);
    float Ax;
    Ax=pow(2*(C-0.5*V*epsilon)*epsilon,0.5);
    for (int i=0; i<m-1; ++i){
        r=r-gradU(theta)*epsilon-r*C*epsilon+distribution(generator)*Ax;
        theta=theta+(r/M)*epsilon;
        }
    return theta;
}

PYBIND11_PLUGIN(sghmcwrap) {
    pybind11::module m("sghmcwrap", "auto-compiled c++ extension of sghmc");
    m.def("sghmc", &sghmc);
    return m.ptr();
}