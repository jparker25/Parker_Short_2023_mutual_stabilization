import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import time
import os, sys
from datetime import timedelta, datetime
from multiprocessing import Pool

from hindmarsh_rose import hindmarsh_rose as hr
import rk4

def f(Icurr):
    direc = 'tmp'
    tt = 0
    t0 = 0;
    trans=2000;
    tf = 5000;
    dt=1/128;
    st = time.time()
    neuron = hr(I=Icurr,r=0.006)
    x0 = np.random.rand(3);
    transient = rk4.rk4_N(x0,t0,dt,neuron.hr_dynamics,int(trans/dt))
    sol = rk4.rk4_N(transient[-1,1:],t0,dt,neuron.hr_dynamics,int(tf/dt))
    pks,_ = find_peaks(sol[:,1],height=1)
    np.savetxt(f'tmp/peaks_I_{Icurr:.3f}.txt',sol[pks],newline='\n',delimiter='\t')

if __name__ == '__main__':
    N = 100;
    Ii = 1.75;
    If = 4;
    Ispace = np.linspace(Ii,If,N)
    pool = Pool(os.cpu_count())
    pool.map(f,Ispace)
    pool.close()
    pool.join()
