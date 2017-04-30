! git clone https://github.com/RLovelett/eigen.git

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

// randnorm
Eigen::VectorXd randnorm(float mu, float Sigma, int n) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution (mu, Sigma);
    Eigen::VectorXd y(n,1);
    for (int i=0; i<n; ++i){
        y(i)=distribution(generator);
        }
    return y;
}
    
// randnorm one draw
float randnorm1(float mu, float Sigma) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution (mu, Sigma);
    return distribution(generator);
}    
    
// sghmc function
float sghmc(const std::function<float(float)> &U, const std::function<float(float)> &gradU, float m, float dt, int nstep, float x, float C, float V) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution (0, 1);
    float r;
    r=distribution(generator)*pow(m,0.5);
    float D;
    D=pow(2*(C-0.5*V*dt)*dt,0.5);
    for (int i=0; i<nstep-1; ++i){
        r=r-gradU(x)*dt-r*C*dt+distribution(generator)*D;
        x=x+(r/m)*dt;
        }
    return x;
}

PYBIND11_PLUGIN(sghmcwrap) {
    pybind11::module m("sghmcwrap", "auto-compiled c++ extension of sghmc");
    m.def("randnorm", &randnorm);
    m.def("randnorm1", &randnorm1);
    m.def("sghmc", &sghmc);
    return m.ptr();
}



import cppimport
sghwrap=cppimport.imp("sghmcwrap")



import sympy as sp
x=sp.symbols('x')
U = sp.symbols('U', cls=sp.Function)
U=sp.Matrix([-2* x**2 + x**4])
x = sp.Matrix([x])
gradientU = sp.simplify(U.jacobian(x))

U=sp.lambdify(x,U)
gradU=sp.lambdify(x,gradientU)



import numba
from numba import jit
from numba import float64

@jit(float64[:](float64, float64, float64, float64, float64, float64))
def sampling(nsample,m,dt,nstep,C,V):
    x=0
    for i in range(1,nsample+1):
        x=sghwrap.sghmc(U,gradU,m,dt,nstep,x,C,V)
        samples[i-1]=x
    return samples



    %%time

import numpy as np

nsample=80000
xstep=0.1
m=1
C=3
dt=0.1
nstep=50
V=4

samples=np.zeros(shape=(nsample,1))

samples=sampling(nsample,m,dt,nstep,C,V)