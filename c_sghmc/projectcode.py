import numpy as np
import numpy.random
import sympy as sp
import seaborn as sns
import matplotlib.pyplot as plt

def hmc(U, gradU, M, epsilon, m, theta, mhtest=1):
    """Hamiltonian Monte-Carlo algorithm with an optional Metropolis-Hastings test
    U is potential energy as a callable function
    gradU is its gradient as a callable function
    M is a mass matrix for kinetic energy
    epsilon is the step size dt
    m is the number of iterations
    theta is the parameter of interest
    mhters=1 is to include MH test by default - yes
    """
    #draw momentum
    r=numpy.random.normal(size=(np.size(theta),1))*np.sqrt(M)
    theta0=theta
    E0=r.T * M * r/2+U(theta)
    
    #do leapfrog
    for i in range(1,m+1):
        r=r-gradU(theta)*epsilon/2
        theta=theta+epsilon*r/M
        r=r-gradU(theta)*epsilon/2
    r=-r
    
    #carry out MH test
    if mhtest != 0:
        Enew=r.T * M * r/2+U(theta)

        if np.exp(E0-Enew)<numpy.random.uniform(0,1,(1,1)):
            theta=theta0

    newtheta=theta
    return newtheta


#Parameters for analysis (to replicate the paper)
nsample=80000 #number of iterations for the sample
xstep=0.01 #step size for true distribution
M=1 #mass
C=3 #constant for sghmc
epsilon=0.1 #dt stepsize term
m=50 #number of steps for Monte-Carlo
V=4 #estimate of Fisher Info for Bhat approximation in sghmc
numpy.random.seed(2017)


x=sp.symbols('x')
U = sp.symbols('U', cls=sp.Function)
U=sp.Matrix([-2* x**2 + x**4]) #define your potential energy here
x = sp.Matrix([x])
gradientU = sp.simplify(U.jacobian(x))

#cover sympy function object into a callable function
U=sp.lambdify(x,U)
gradU=sp.lambdify(x,gradientU)


#True distribution
plt.figure(1)
plt.subplot(211)
gridx=np.linspace(-3,3,6/xstep)
y=np.exp(-U(gridx))
plt.plot(gridx, np.reshape(y/np.sum(y)/xstep, (int(6/xstep), 1)) , 'bo')
pass



#hmc sampling alhorithm
sampleshmc=np.zeros(shape=(nsample,1))
theta=0
for i in range(1,nsample+1):
	theta=hmc(U,gradU,M,epsilon,m,theta)
	sampleshmc[i-1]=theta

#function to access the precision of approximation
def comparison(y,samples):
    """Returns a euclidean distance as precision proxy
    y is the true ditribution
    samples are drawn using an MCMC algorithm
    """
    
    y=np.reshape(y/np.sum(y)/xstep, (int(6/xstep), 1))
    yh, xh= numpy.histogram(samples, bins=gridx) #compute a histogram for samples
    yh=yh/np.sum(yh)/xstep
    return np.sqrt(np.sum((yh[:,None]-y[1:])**2)) #euc distance between the two

#hmc precision
comparison(y,sampleshmc)

#normalized histogram of hmc drawn samples
sns.distplot(sampleshmc)
pass

def sghmc(U,gradU,M,epsilon,m,theta,C,V):
    """Stochastic Gradient Hamiltonian Monte-Carlo algorithm
    U is potential energy as a callable function
    gradU is its gradient as a callable function (noisy)
    M is a mass matrix for kinetic energy
    epsilon is the step size dt
    m is the number of iterations
    theta is the parameter of interest
    C is a user defined constant
    V is a Fisher info approximation
    """
    
    #draw a momentum and compute Bhat
    r=numpy.random.standard_normal(size=(np.size(theta),1))*np.sqrt(M)
    Bhat=0.5*V*epsilon
    Ax=np.sqrt(2*(C-Bhat)*epsilon)
    #do leapfrog
    for i in range(1,m+1):
        r=r-gradU(theta)*epsilon-r*C*epsilon+numpy.random.standard_normal(size=(1,1))*Ax
        theta=theta+(r/M)*epsilon
    newtheta=theta
    return newtheta




#sghmc sampling alhorithm (Pure python)
samplessghmc=np.zeros(shape=(nsample,1))
theta=0
for i in range(1,nsample+1):
    theta=sghmc(U,gradU,M,epsilon,m,theta,C,V)
    samplessghmc[i-1]=theta

#pure sghmc precision
comparison(y,samplessghmc)

#import a wrapped in pybind11 c++ implementation of sghmc algorithm
import cppimport
sghwrap=cppimport.imp("sghmcwrap")



#sghmc sampling alhorithm (compilation in C++)
samplessghmc_c=np.zeros(shape=(nsample,1))
theta=0
for i in range(1,nsample+1):
    theta=sghwrap.sghmc(U,gradU,M,epsilon,m,theta,C,V)
    samplessghmc_c[i-1]=theta


#c++ sghmc precision
comparison(y,samplessghmc_c)


import numba
from numba import jit
from numba import float64

#prepare a just-in-time compiled function calling C++ sghmc algorithm
@jit(float64[:](float64, float64, float64, float64, float64, float64))
def sampling(nsample,M,epsilon,m,C,V):
    theta=0
    for i in range(1,nsample+1):
        theta=sghwrap.sghmc(U,gradU,M,epsilon,m,theta,C,V)
        samplessghmc_numba[i-1]=theta
    return samplessghmc_numba


#sghmc sampling alhorithm (compilation in C++ of a jitted function)
samplessghmc_numba=np.zeros(shape=(nsample,1))
samplessghmc_numba=sampling(nsample,M,epsilon,m,C,V)

#jitted c++ sghmc precision
comparison(y,samplessghmc_numba)

#normalized histogram of sghmc drawn samples
import seaborn as sns
sns.distplot(samplessghmc_numba)
pass


%load_ext Cython


import scipy.io
import scipy
import scipy.linalg as la
import scipy.sparse
import urllib.request

#call "Australian credit" dataset for a Bayesian Linear Regression analysis
#Bache, K. and Lichman, M. UCI machine learning repository,2013. URL http://archive.ics.uci. edu/ml.
filename = 'australian'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat'
urllib.request.urlretrieve(url, filename)
data = np.loadtxt(filename)

#Parameters for BLR
alpha=0.01 #sigma of prior normal
nstepsunscaled=1000 #unscaled number of steps for Monte-Carlo
scaleHCM=2 #ratio of size of steps for integration to their number
niters=6000 #number of iterations
scale_StepSize=0.5 #default 0.5 for sigma=0.01
m = np.round(nstepsunscaled/scaleHCM) #scaled number of steps for Monte-Carlo
BurnIn = 1000 #number of iteration to use for burn in
StepSize = 0.1 #unscaled dt/epsilon step size for dynamics
StepSize = scale_StepSize*StepSize*scaleHCM; #scaled dt/epsilon step size for dynamics
Poly_Order = 1 #order of polynomial to fit
numpy.random.seed(2017)


Xraw=data
Y=Xraw[:,-1] #to test on
Xraw = np.delete(Xraw, -1, 1) #leave only the data for training

# Normalize Data
N,D=Xraw.shape
Xraw=(Xraw-np.mean(Xraw,0))/np.std(Xraw,0)

# Create Polynomial Basis
X = np.ones(shape=(N,1))
for i in range(Poly_Order):
    X = np.concatenate((X,Xraw**(i+1)),1)

N,D = X.shape
Mass = np.eye(D)

InvMass = scipy.sparse.csr_matrix(la.inv(Mass)) #find inverse of Mass

# Set initial values of w
w = np.zeros(shape=(D,1))
ws = np.zeros(shape=(niters-BurnIn,D))

def LogNormPDF(xs,mu,sigma):
    """LogPrior calculcation as a LogNormal distribution
    xs are the values (Dx1)
    mu are the means (Dx1)
    sigma is the cov matrix (Dx1 as diag)
    """
 
    if xs.shape[1] > 1:
        xs = xs.T

    if mu.shape[1] > 1:
           mu = mu.T

    D = max(xs.shape)
    return sum( -np.ones(shape=(D,1))*(0.5*np.log(2*np.pi*sigma)) - ((xs-mu)**2)/(2*(np.ones(shape=(D,1))*sigma)) )


#Compute energy and joint loglikelihood for current w
LogPrior      = LogNormPDF(np.zeros(shape=(1,D)),w,alpha)
f             = X@w
LogLikelihood = f.T@Y - np.sum(np.log(1+np.exp(f)))
CurrentLJL    = LogLikelihood + LogPrior

Proposed = 0
Accepted = 0



#Pure Python version of HMC BLR

for iteration in range(niters):
        
    #draw momentum and stepsize
    r = (numpy.random.standard_normal(size=(1,D))@Mass).T
    r0 = r
        
    wnew = w
    Proposed = Proposed + 1
      
    RandomStep = np.round(np.random.rand(1)*(m-1))+1
        
    #do leapfrog
    mark = 0
    f = X@wnew
    r = r + (StepSize/2)*( X.T@( Y[:,None] - (np.exp(f)/(1+np.exp(f))) ) - np.eye(D)*(1/alpha)@wnew)
    for step in range(int(RandomStep)-1):
        #make sure everything is well-behaved
        if (np.isnan(np.sum(r)) or np.isnan(np.sum(wnew)) or np.isinf(np.sum(r)) or np.isinf(np.sum(wnew))):
            mark = 1
            break
        wnew = wnew + StepSize*(r)                
        f = X@wnew
        r = r + StepSize*( X.T@( Y[:,None] - (1./(1+np.exp(-f))) ) - np.eye(D)*(1/alpha)@wnew )
        r = np.real(r)
        f = np.real(f)
    
    if (mark == 0):
        wnew = wnew + StepSize*(r)              
        f = X@wnew
        r = r + (StepSize/2)*( X.T@( Y[:,None] - (np.exp(f)/(1+np.exp(f))) ) - np.eye(D)*(1/alpha)@wnew )
    else:
        r = r - (StepSize/2)*( X.T@( Y[:,None] - (np.exp(f)/(1+np.exp(f))) ) - np.eye(D)*(1/alpha)@wnew )
    
    #find proposed energy H and train likelihood   
    LogPrior      = LogNormPDF(np.zeros(shape=(1,D)),wnew,alpha)
    f             = X@wnew
    LogLikelihood = f.T@Y - np.sum(np.log(1+np.exp(f)))
    ProposedLJL   = LogLikelihood + LogPrior
    
    ProposedH = -ProposedLJL + (r.T@InvMass@r)/2
        
    #compute current H value
    CurrentH  = -CurrentLJL + (r0.T@InvMass@r0)/2
       
    #Accept according to Metropolis-Hastings ratio
    MH = -ProposedH + CurrentH
          
    if (MH > 0) or (MH > np.log(numpy.random.rand(1))):
        CurrentLJL = ProposedLJL
        w = wnew
        Accepted = Accepted + 1
    

    #Now save samples after burn in
    if iteration > BurnIn:
    	ws[[iteration-BurnIn-1],:] = w.T
    elif np.mod(iteration,50) == 0:
        Accepted = 0
        Proposed = 0

#Fit the model and find R squared
bhat=np.mean(ws,0)
Yhat=X@bhat[:,None]
SSR=np.sqrt(np.sum((Y[:,None]-Yhat)**2))
TSS=np.sum((Y-np.mean(Y,0))**2)
Rsq=1-SSR/TSS
Rsq


Proposed=0
Accepted=0


%%cython -a

import cython
import numpy as np
cimport numpy as np
import numpy.random

cdef inline int int_max(int a, int b): return a if a >= b else b #a quicker version of max

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True) 
cdef LogNormPDF_cython(np.ndarray[np.float64_t, ndim=2] O, np.ndarray[np.float64_t, ndim=2] xs, np.ndarray[np.float64_t, ndim=2] mu, double sigma):
    """LogPrior calculcation as a LogNormal distribution
    xs are the values (Dx1)
    mu are the means (Dx1)
    sigma is the cov matrix (Dx1 as diag)
    """
    if xs.shape[1] > 1:
        xs = xs.T

    if mu.shape[1] > 1:
        mu = mu.T

    cdef int D = int_max(xs.shape[0],xs.shape[1])
    return sum( -O*(0.5*np.log(2*np.pi*sigma)) - ((xs-mu)**2)/(2*(O)*sigma)) 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True) 
cdef momentupdate(np.ndarray[np.float64_t, ndim=2] E, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] Y, np.ndarray[np.float64_t, ndim=2] f, int D, double alpha, np.ndarray[np.float64_t, ndim=2] wnew):
    """Update momentum given current data
    """
    cdef np.ndarray[np.float64_t, ndim=2] g=np.exp(f)
    return ( np.dot(X.T,( Y[:,None] - (g/(1+g)) )) - E*(1/alpha)@wnew)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True) 
cdef lfrogupdate(np.ndarray[np.float64_t, ndim=2] E, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] Y, np.ndarray[np.float64_t, ndim=2] f, int D, double alpha, np.ndarray[np.float64_t, ndim=2] wnew):
    """Update momentum given current data in leapfrog iterations
    """
    return ( np.dot(X.T,( Y[:,None] - (1./(1+np.exp(-f))) )) - E*(1/alpha)@wnew)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def BLR_hmc_cython(int D, np.ndarray[np.float64_t, ndim=2] Mass, np.ndarray[np.float64_t, ndim=2] w, double m, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] Y, np.ndarray[np.float64_t, ndim=2] f, double alpha, double StepSize, int BurnIn, int niters, double CurrentLJL):
    """Bayesian Linear Regression using HMC algorithm implemented using Cython
    D is shape of data
    Mass is the mass matrix of kinetic energy
    w is a vector of coefficients to estimate
    m is number of iterations for Monte-Carlo
    X is the explanatory data matrix
    Y is the explained vector
    f fit given initial coefficients (0s)
    alpha is variance of prior
    StepSize dt for dynamics
    BurnIn number of iteration to use for burn in
    niters number of iteration for Monte-Carlo
    CurrentLJL initial state of energy
    """
    cdef int Proposed=0
    cdef int Accepted=0
    cdef int iteration, mark, step
    cdef np.ndarray[np.float64_t, ndim=2] ws = np.zeros(shape=(niters-BurnIn,D)) #coefficients to save
    cdef np.ndarray[np.float64_t, ndim=2] wnew
    cdef np.ndarray[np.float64_t, ndim=2] r, r0 
    cdef np.ndarray[np.float64_t, ndim=1] LogPrior, LogLikelihood, ProposedLJL, RandomStep
    cdef np.ndarray[np.float64_t, ndim=2] MH, ProposedH, CurrentH
    cdef np.ndarray[np.float64_t, ndim=2] Z=np.zeros(shape=(1,D))
    cdef np.ndarray[np.float64_t, ndim=2] O=np.ones(shape=(D,1))
    cdef np.ndarray[np.float64_t, ndim=2] E=np.eye(D)
    
    for iteration in range(niters):

        #draw momentum
        r = (np.dot(numpy.random.standard_normal(size=(1,D)),Mass)).T
        r0 = r

        wnew = w
        Proposed = Proposed + 1

        RandomStep = np.round(np.random.rand(1)*(m-1))+1

        #do leapfrog
        mark = 0
        f = np.dot(X,wnew)
        r = r + (StepSize/2)*momentupdate(E,X,Y,f,D,alpha,wnew)
        for step in range(np.int(RandomStep)-1):
            #make sure everything is well-behaved
            if (np.isnan(np.sum(r)) or np.isnan(np.sum(wnew)) or np.isinf(np.sum(r)) or np.isinf(np.sum(wnew))):
                mark = 1
                break
            wnew = wnew + StepSize*(r)                
            f = np.dot(X,wnew)
            r = r + StepSize*lfrogupdate(E,X,Y,f,D,alpha,wnew)
            r = np.real(r)
            f = np.real(f)

        if (mark == 0):
            wnew = wnew + StepSize*(r)              
            f = np.dot(X,wnew)
            r = r + (StepSize/2)*momentupdate(E,X,Y,f,D,alpha,wnew)
        else:
            r = r - (StepSize/2)*momentupdate(E,X,Y,f,D,alpha,wnew)

        #find proposed energy H and train likelihood  
        LogPrior      = LogNormPDF_cython(O,Z,wnew,alpha)
        f             = np.dot(X,wnew)
        LogLikelihood = np.dot(f.T,Y) - np.sum(np.log(1+np.exp(f)))
        ProposedLJL   = LogLikelihood + LogPrior

        ProposedH = -ProposedLJL + (np.dot(np.dot(r.T,Mass),r))/2

        #compute current H value
        CurrentH  = -CurrentLJL + (np.dot(np.dot(r0.T,Mass),r0))/2

        #Accept according to Metropolis-Hastings ratio
        MH = -ProposedH + CurrentH

        if (MH > 0) or (MH > np.log(numpy.random.rand(1))):
            CurrentLJL = ProposedLJL
            w = wnew
            Accepted = Accepted + 1
        

        #Now save samples after burn in
        if iteration > BurnIn:
            ws[iteration-BurnIn-1,:] = np.ravel(w)
        elif np.mod(iteration,50) == 0:
            Accepted = 0
            Proposed = 0
        
    return ws



BRLHMCcoeffs=BLR_hmc_cython(D, Mass, w, m, X, Y, f, alpha, StepSize, BurnIn, niters, CurrentLJL)



#Fit the model and find R squared
bhat=np.mean(BRLHMCcoeffs,0)
Yhat=X@bhat[:,None]
SSR=np.sqrt(np.sum((Y[:,None]-Yhat)**2))
TSS=np.sum((Y-np.mean(Y,0))**2)
Rsq=1-SSR/TSS
Rsq



#Pure Python version of SGHMC BLR
C=3 #user-chosen const s.t. C>=B
Bhat=0 #for simplicity, but ideally Bhat=0.5*Vhat*dt with Vhat estimated via empirical Fisher Info

for iteration in range(niters):
        
    #draw momentum
    r = (numpy.random.standard_normal(size=(1,D))@Mass).T
    r0 = r
        
    wnew = w
    RandomStep = np.round(np.random.rand(1)*(m-1))+1
        
    #do leapfrog
    mark = 0
    f = X@wnew
    J = np.sqrt( 2 * (C-Bhat) * StepSize)
    for step in range(int(RandomStep)-1):
        #make sure everything is well-behaved
        if (np.isnan(np.sum(r)) or np.isnan(np.sum(wnew)) or np.isinf(np.sum(r)) or np.isinf(np.sum(wnew))):
            mark = 1
            break
        wnew = wnew + StepSize*(r)                
        f = X@wnew
        r = (r + StepSize*( X.T@( Y[:,None] - (1./(1+np.exp(-f))) )
                - np.eye(D)*(1/alpha)@wnew )-StepSize*C*(r)+numpy.random.standard_normal(size=(D,1))*J)
        r = np.real(r)
        f = np.real(f)
    
    if (mark == 0):
        wnew = wnew + StepSize*(r)              
        f = X@wnew
        
    #find proposed total energy H and train likelihood   
    LogPrior      = LogNormPDF(np.zeros(shape=(1,D)),wnew,alpha)
    f             = X@wnew
    LogLikelihood = f.T@Y - np.sum(np.log(1+np.exp(f))) #training likelihood
    ProposedLJL   = LogLikelihood + LogPrior

    w=wnew

    #Now save samples after burn in
    if iteration > BurnIn:
    	ws[iteration-BurnIn-1,:] = w.ravel()

bhat=np.mean(ws,0)
Yhat=X@bhat[:,None]
SSR=np.sqrt(np.sum((Y[:,None]-Yhat)**2))
TSS=np.sum((Y-np.mean(Y,0))**2)
Rsq=1-SSR/TSS
Rsq


C=3 #user-chosen const s.t. C>=B
Bhat=0 #for simplicity, but ideally Bhat=0.5*Vhat*dt with Vhat estimated via empirical Fisher Info



%%cython -a

import cython
import numpy as np
cimport numpy as np
import numpy.random

cdef inline int int_max(int a, int b): return a if a >= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True) 
cdef LogNormPDF_cython(np.ndarray[np.float64_t, ndim=2] O, np.ndarray[np.float64_t, ndim=2] xs, np.ndarray[np.float64_t, ndim=2] mu, double sigma):
    """LogPrior calculcation as a LogNormal distribution
    xs are the values (Dx1)
    mu are the means (Dx1)
    sigma is the cov matrix (Dx1 as diag)
    """
    if xs.shape[1] > 1:
        xs = xs.T

    if mu.shape[1] > 1:
        mu = mu.T

    cdef int D = int_max(xs.shape[0],xs.shape[1])
    return sum( -O*(0.5*np.log(2*np.pi*sigma)) - ((xs-mu)**2)/(2*(O)*sigma)) 
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True) 
cdef lfrogupdate(np.ndarray[np.float64_t, ndim=2] E, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] Y, np.ndarray[np.float64_t, ndim=2] f, int D, double alpha, np.ndarray[np.float64_t, ndim=2] wnew):
    """Update momentum given current data in leapfrog iterations
    """
    return ( np.dot(X.T,( Y[:,None] - (1./(1+np.exp(-f))) )) - E*(1/alpha)@wnew)

cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def BLR_sghmc_cython(int C, int Bhat, int D, np.ndarray[np.float64_t, ndim=2] Mass, np.ndarray[np.float64_t, ndim=2] w, double m, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] Y, np.ndarray[np.float64_t, ndim=2] f, double alpha, double StepSize, int BurnIn, int niters, double CurrentLJL):
    """Bayesian Linear Regression using HMC algorithm implemented using Cython
    C is a user specified constant
    Bhat is an approximate set to 0 here (it should converge to 0)
    D is shape of data
    Mass is the mass matrix of kinetic energy
    w is a vector of coefficients to estimate
   m is number of iterations for Monte-Carlo
    X is the explanatory data matrix
    Y is the explained vector
    f fit given initial coefficients (0s)
    alpha is variance of prior
    StepSize dt for dynamics
    BurnIn number of iteration to use for burn in
    niters number of iteration for Monte-Carlo
    CurrentLJL initial state of energy
    """
   
    cdef int iteration, mark, step
    cdef np.ndarray[np.float64_t, ndim=2] ws = np.zeros(shape=(niters-BurnIn,D)) #coefficients to save
    cdef np.ndarray[np.float64_t, ndim=2] wnew
    cdef np.ndarray[np.float64_t, ndim=2] r, r0 
    cdef np.ndarray[np.float64_t, ndim=1] LogPrior, LogLikelihood, ProposedLJL, RandomStep
    cdef np.ndarray[np.float64_t, ndim=2] Z=np.zeros(shape=(1,D))
    cdef np.ndarray[np.float64_t, ndim=2] O=np.ones(shape=(D,1))
    cdef np.ndarray[np.float64_t, ndim=2] E=np.eye(D)
    cdef double J = np.sqrt( 2 * (C-Bhat) * StepSize) #sd for friction
   
    for iteration in range(niters):
        
        #draw momentum
        r = (np.dot(numpy.random.standard_normal(size=(1,D)),Mass)).T
        r0 = r

        wnew = w
        RandomStep = np.round(np.random.rand(1)*(m-1))+1

       #do leapfrog
        mark = 0
        f = np.dot(X,wnew)

        for step in range(int(RandomStep)-1):
            #make sure everything is well-behaved
            if (np.isnan(np.sum(r)) or np.isnan(np.sum(wnew)) or np.isinf(np.sum(r)) or np.isinf(np.sum(wnew))):
                mark = 1
                break
            wnew = wnew + StepSize*(r)                
           f = np.dot(X,wnew)
            r = (r + StepSize*lfrogupdate(E,X,Y,f,D,alpha,wnew)-StepSize*C*(r)+numpy.random.standard_normal(size=(D,1))*J)
            r = np.real(r)
            f = np.real(f)

        if (mark == 0):
            wnew = wnew + StepSize*(r)              
            f = np.dot(X,wnew)

        #find proposed total energy H and train likelihood   
       LogPrior      = LogNormPDF_cython(O,Z,wnew,alpha)
        f             = np.dot(X,wnew)
        LogLikelihood = np.dot(f.T,Y) - np.sum(np.log(1+np.exp(f))) #training likelihood
        ProposedLJL   = LogLikelihood + LogPrior

        w=wnew

        #Now save samples after burn in
        if iteration > BurnIn:
            ws[iteration-BurnIn-1,:] = w.ravel()
        
    return ws



BRLSGHMCcoeffs=BLR_sghmc_cython(C, Bhat, D, Mass, w, m, X, Y, f, alpha, StepSize, BurnIn, niters, CurrentLJL)

bhat=np.mean(BRLSGHMCcoeffsBRLSGHMCcoeffsBRLSGHMCcoeffs## 663 Final Project Second Report
,0)
Yhat=X@bhat[:,None]
SSR=np.sqrt(np.sum((Y[:,None]-Yhat)**2))
TSS=np.sum((Y-np.mean(Y,0))**2)
Rsq=1-SSR/TSS
Rsq

