#using Eigen library for C++ la
! git clone https://github.com/RLovelett/eigen.git

    
#pybind11 wrap of the SGHMC C++ function
%%file sghmcwrap.cpp
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



import cppimport
sghwrap=cppimport.imp("sghmcwrap")

#test it on a simple example from
#"Stochastic Gradient Hamiltonian Monte Carlo" by Tianqi Chen, Emily B. Fox, Carlos Guestrin
import sympy as sp
x=sp.symbols('x')
U = sp.symbols('U', cls=sp.Function)
U=sp.Matrix([-2* x**2 + x**4]) #define your potential energy here
x = sp.Matrix([x])
gradientU = sp.simplify(U.jacobian(x))

#cover sympy function object into a callable function
U=sp.lambdify(x,U)
gradU=sp.lambdify(x,gradientU)

#Parameters for analysis (to replicate the paper)
nsample=80000 #number of iterations for the sample
xstep=0.01 #step size for true distribution
M=1 #mass
C=3 #constant for sghmc
epsilon=0.1 #dt stepsize term
m=50 #number of steps for Monte-Carlo
V=4 #estimate of Fisher Info for Bhat approximation in sghmc
numpy.random.seed(2017)

def sghmc(U,gradU,M,epsilon,m,theta,C,V,nsample):
    samplessghmc=np.zeros(shape=(nsample,1))
    theta=0
    for i in range(1,nsample+1):
        theta=sghwrap.sghmc(U,gradU,M,epsilon,m,theta,C,V)
        samplessghmc[i-1]=theta
    return samplesshmc
