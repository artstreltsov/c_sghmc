C-SGHMC
=======================

This is a repository created as part of a statistical computation project at Duke University.

The code to be found here is a C++ implementation of SGHMC algorithm wrapped for Python using pybind11. It is further used to verify the results in the original paper on SGHMC: "Stochastic Gradient Hamiltonian Monte Carlo" by Tianqi Chen, Emily B. Fox and Carlos Guestrin.

As seen from our project this is a **15-time-efficiency-improvement** in terms of runtime over the pure python version.

Packages required to run the code
=======================

- The implementation makes use of **Eigen** library for C++ and hence the line:
! git clone https://github.com/RLovelett/eigen.git

Make sure you have it cloned in the same directory as your code.

- **pybind11** and **cppimport** are the two standard packages required for wrapping C++ codes for Python and importing them:
! pip3 install pybind11

! pip3 install cppimport

The example makes use of **sympy** to specify functions (such as potential and kinetic energies as well as the gradient of the former), **numpy** and **numpy.random**.
