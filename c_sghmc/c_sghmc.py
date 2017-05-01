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
