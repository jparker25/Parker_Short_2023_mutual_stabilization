# System model of Hindmarsh-Rose Neuron
# Author: John E. Parker

# Import Python modules
import numpy as np
import pickle

class hindmarsh_rose:
    def __init__(self,a = 1, b=3, c=1, d=5, s=4, xr=-8/5, r=0.006, I=3.25):
        '''
        Initialize a Hindmarsh-Rose neuron with default parameters unless specified.
        '''
        self.a = a # Default parameter value in Hindmarsh & Rose 1984
        self.b = b # Default parameter value in Hindmarsh & Rose 1984
        self.c = c # Default parameter value in Hindmarsh & Rose 1984
        self.d = d # Default parameter value in Hindmarsh & Rose 1984
        self.s = s # Default parameter value in Hindmarsh & Rose 1984
        self.xr = xr  # Default parameter  valuein Hindmarsh & Rose 1984
        self.r = r # Bifurcation parameter, value chosen to be in chaotic regime with default I
        self.I = I # Represents incoming current, value chosen to be in chaotic regime with default r


    def hr_dynamics(self, state):
        '''
        Generates the Hindmarsh-Rose dynamical state based off state (vector of x y z values)
        '''
        x,y,z = state # Redefine values from state for readability
        return np.array([
            y-self.a*x**3+self.b*x**2-z+self.I, # xdot equation
            self.c-self.d*x**2-y, # ydot equation
            self.r*(self.s*(x-self.xr)-z) # zdot equation
            ])

    def hr_dy_dynamics(self, state, y):
        '''
        Uses Henon's trick to return dynamics based off of d/dy. Assumes
        x is t,x,z state and y is value of y variable. Returns d/dy.
        '''
        t,x,z = state # Redefine values from state for readability
        dy = (self.c-self.d*x**2-y);
        return np.array([
            1/dy,
            (y-self.a*x**3+self.b*x**2-z+self.I)/dy,
             self.r*(self.s*(x-self.xr)-z)/dy,
        ])

    def hr_dx_dynamics(self, state, x):
        '''
        Uses Henon's trick to return dynamics based off of d/dx. Assumes
        x is t,y,z state and y is value of x variable. Returns d/dx.
        '''
        t,y,z = state # Redefine values from state for readability
        dx = y-self.a*x**3+self.b*x**2-z+self.I;
        return np.array([
            1/dx,
            (self.c-self.d*x**2-y)/dx,
            self.r*(self.s*(x-self.xr)-z)/dx
        ])

    def hr_jacobian(self,t,state):
        '''
        Returns the jacobian of the HR system at a given state space.
        '''
        x,y,z = state; # Redefine values from state for readability
        return np.array([ # Jacobian
            [-3*self.a*x*x+2*self.b*x, 1, -1],
            [-2*self.d*x,-1,0],
            [self.r*self.s,0,-self.r]
        ])

    def save_neuron(self,direc):
        '''
        Saves the neuron attributes to neuron.obj in directory direc.
        '''
        pickle.dump(self,open("{0}/neuron.obj".format(direc),'wb'))
