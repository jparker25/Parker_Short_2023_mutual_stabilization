# Script to run various Rk4 integrations on a given dynamical system
# Author: John Parker

# Import Python modules
import numpy as np

def rk4(x,dt,f):
    '''
    Reads in a state X, time step dt, and dynamics function f. Performs autonomous
    RK4 method and returns next iteration
    '''
    k1 = dt*f(x)
    k2 = dt*f(x+k1/2)
    k3 = dt*f(x+k2/2)
    k4 = dt*f(x+k3)
    return x+(k1+2*k2+2*k3+k4)/6

def rk4_N(x0,t0,dt,f,N):
    '''
    Reads in initial state x0, time t0, time step dt, function f, and iterations N.
    Performs RK4 method for N steps and returns matrix of all time and spatial steps
    '''
    sol = np.empty((N,x0.size+1))
    sol[0,0] = t0; sol[0,1:] = x0
    for i in range(1,N):
        sol[i,1:] = rk4(sol[i-1,1:],dt,f)
        sol[i,0] = sol[i-1,0]+dt
    return sol

def rk4_Nj(x0,t0,dt,f,N):
    '''
    Reads in initial state x0, time t0, time step dt, function f, and iterations N.
    Assumes function f depends on current time iteration index.
    Performs RK4 method for N steps and returns matrix of all time and spatial steps
    '''
    sol = np.empty((N,x0.size+1))
    sol[0,0] = t0; sol[0,1:] = x0
    for i in range(1,N):
        sol[i,1:] = rk4j(sol[i-1,1:],dt,f,i-1)
        sol[i,0] = sol[i-1,0]+dt
    return sol

def rk4_henon(xi,ti,dd,f):
    '''
    Perform Henon's trick with RK4 method. xi is the initial state after varying
    array to insert independent variable before the trick. ti is the state of Henon's
    independent variable. dd is the distance to travel and f is the dynamics for Henon's trick.
    '''
    k1 = dd*f(xi,ti)
    k2 = dd*f(xi+k1/2,ti+dd/2)
    k3 = dd*f(xi+k2/2,ti+dd/2)
    k4 = dd*f(xi+k3,ti+dd)
    update = xi+(k1+2*k2+2*k3+k4)/6
    return update[0],update[1],update[2], ti+dd

def rk4j(x,dt,f,j):
    '''
    Reads in a state X, time step dt, and dynamics function f, and current time iteration j. Performs autonomous
    RK4 method and returns next iteration
    '''
    k1 = dt*f(x,j)
    k2 = dt*f(x+k1/2,j)
    k3 = dt*f(x+k2/2,j)
    k4 = dt*f(x+k3,j)
    return x+(k1+2*k2+2*k3+k4)/6
