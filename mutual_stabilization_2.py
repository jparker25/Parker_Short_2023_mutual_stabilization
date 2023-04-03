import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import bisect
import sys,os
import time
from datetime import datetime, timedelta
from scipy.signal import find_peaks

import rk4
import control_planes as gc
import keep_data as kd
import cupolet_search as cupg
import signal_analysis as sa
from helpers import *

def integrateAndFire(vs,subbits,threshold):
    if sum(vs) >= threshold:
        return 1
    else:
        return 0

def mutual_stabilization(direc,dt,t0,tf,control1,control2,ifm,ifn):
    bin_rn_direc = '{0}/bins_1600_rN_16/'.format(direc)
    neuron = pickle.load(open("{0}/neuron.obj".format(direc),'rb'))
    N = int(tf/dt); trans = np.zeros((N,7));
    for neu in range(0,2):
        trans[:,3*neu+1:3*neu+4] = rk4.rk4_N(np.random.rand(3),t0,dt,neuron.hr_dynamics,N)[:,1:]

    ps0inits = np.loadtxt('{0}/coding_fcn/ps0_inits.txt'.format(bin_rn_direc))
    ps1inits = np.loadtxt('{0}/coding_fcn/ps1_inits.txt'.format(bin_rn_direc))

    ps0endpts = np.loadtxt('{0}/coding_fcn/ps0_endpoints.txt'.format(bin_rn_direc))
    ps1endpts = np.loadtxt('{0}/coding_fcn/ps1_endpoints.txt'.format(bin_rn_direc))

    ps0x, ps0y, ps0z = np.loadtxt('{0}/control_planes/ps0_vertices.txt'.format(direc),unpack=True)
    ps1x, ps1y, ps1z = np.loadtxt('{0}/control_planes/ps1_vertices.txt'.format(direc),unpack=True)

    ctrl1,_ = np.loadtxt('{0}/macrocontrol/ps1_macrocontrol.txt'.format(bin_rn_direc),unpack=True)
    ctrl0,_ = np.loadtxt('{0}/macrocontrol/ps0_macrocontrol.txt'.format(bin_rn_direc),unpack=True)

    sol = np.zeros((4*N,7))
    sol[0,0] = 0; sol[0,1:] = trans[-1,1:];
    ctrl1i = 0;
    ctrl2i = 0;

    #vs1 = []; vs2 = [];
    #vs1ms = []; vs2ms = [];
    #control1ms = []; control2ms = [];
    vs = [[],[]]; vsms = [[],[]]; control = [[],[]];

    # The following for loop creates the two cupolets based on control1 and control2
    for i in range(0,sol.shape[0]-1):
        for neu in range(0,2):
            sol[i+1,3*neu+1:3*neu+4] = rk4.rk4(sol[i,3*neu+1:3*neu+4],dt,neuron.hr_dynamics)
            sol[i+1,0] = sol[i,0]+dt

        if gc.crossed(sol[i,1],sol[i+1,1],sol[i,2],sol[i+1,2],1,np.array([[ps1x[0],ps1y[0]],[ps1x[1],ps1y[1]]])):
            tp,xp,zp,yp = rk4.rk4_henon(np.array([sol[i,0],sol[i,1],sol[i,3]]),sol[i,2],-(sol[i,2]-ps1y[0]),neuron.hr_dy_dynamics)
            ii = bisect.bisect(ps1endpts,xp)-1;
            newval = None
            if len(vs[1]) >= ifm and i > 2*N:
                newval = ps1inits[ii] if integrateAndFire(vs[1][-ifm:],ifm,ifn) == 0 else ps1inits[int(ctrl1[ii])];
                control[0].append(integrateAndFire(vs[1][-ifm:],ifm,ifn))
                vsms[0].append(1)
            else:
                newval = ps1inits[ii] if control1[ctrl1i] == 0 else ps1inits[int(ctrl1[ii])];
            newdt = dt-(tp-sol[i,0]);
            next = rk4.rk4(newval,newdt,neuron.hr_dynamics)
            sol[i+1,1:4] = next;
            sol[i+1,0] = sol[i,0]+dt;
            ctrl1i += 1
            vs[0].append(1)

        elif gc.crossed(sol[i,1],sol[i+1,1],sol[i,2],sol[i+1,2],0,np.array([[ps0x[0],ps0y[0]],[ps0x[1],ps0y[1]]])):
            tp,yp,zp,xp = rk4.rk4_henon(np.array([sol[i,0],sol[i,2],sol[i,3]]),sol[i,1],-(sol[i,1]-ps0x[0]),neuron.hr_dx_dynamics)
            ii = bisect.bisect(ps0endpts,yp)-1;
            newval = None
            if len(vs[1]) >= ifm and i > 2*N:
                newval = ps0inits[ii] if integrateAndFire(vs[1][-ifm:],ifm,ifn) == 0 else ps0inits[int(ctrl0[ii])];
                control[0].append(integrateAndFire(vs[1][-ifm:],ifm,ifn))
                vsms[0].append(0)
            else:
                newval = ps0inits[ii] if control1[ctrl1i] == 0 else ps0inits[int(ctrl0[ii])];
            newdt = dt-(tp-sol[i,0]);
            next = rk4.rk4(newval,newdt,neuron.hr_dynamics)
            sol[i+1,1:4] = next;
            sol[i+1,0] = sol[i,0]+dt;
            ctrl1i += 1
            vs[0].append(0)

        if ctrl1i == len(control1):
            ctrl1i = 0

        if gc.crossed(sol[i,4],sol[i+1,4],sol[i,5],sol[i+1,5],1,np.array([[ps1x[0],ps1y[0]],[ps1x[1],ps1y[1]]])):
            tp,xp,zp,yp = rk4.rk4_henon(np.array([sol[i,0],sol[i,4],sol[i,6]]),sol[i,5],-(sol[i,5]-ps1y[0]),neuron.hr_dy_dynamics)
            ii = bisect.bisect(ps1endpts,xp)-1;
            newval = None
            if len(vs[0]) >= ifm and i > 2*N:
                newval = ps1inits[ii] if integrateAndFire(vs[0][-ifm:],ifm,ifn) == 0 else ps1inits[int(ctrl1[ii])];
                control[1].append(integrateAndFire(vs[0][-ifm:],ifm,ifn))
                vsms[1].append(1)
            else:
                newval = ps1inits[ii] if control2[ctrl2i] == 0 else ps1inits[int(ctrl1[ii])];
            newdt = dt-(tp-sol[i,0]);
            next = rk4.rk4(newval,newdt,neuron.hr_dynamics)
            sol[i+1,4:] = next;
            sol[i+1,0] = sol[i,0]+dt;
            ctrl2i += 1
            vs[1].append(1)

        elif gc.crossed(sol[i,4],sol[i+1,4],sol[i,5],sol[i+1,5],0,np.array([[ps0x[0],ps0y[0]],[ps0x[1],ps0y[1]]])):
            tp,yp,zp,xp = rk4.rk4_henon(np.array([sol[i,0],sol[i,5],sol[i,6]]),sol[i,4],-(sol[i,4]-ps0x[0]),neuron.hr_dx_dynamics)
            ii = bisect.bisect(ps0endpts,yp)-1;
            newval = None
            if len(vs[0]) >= ifm and i > 2*N:
                newval = ps0inits[ii] if integrateAndFire(vs[0][-ifm:],ifm,ifn) == 0 else ps0inits[int(ctrl0[ii])];
                control[1].append(integrateAndFire(vs[0][-ifm:],ifm,ifn))
                vsms[1].append(0)
            else:
                newval = ps0inits[ii] if control2[ctrl2i] == 0 else ps0inits[int(ctrl0[ii])];
            newdt = dt-(tp-sol[i,0]);
            next = rk4.rk4(newval,newdt,neuron.hr_dynamics)
            sol[i+1,4:] = next;
            sol[i+1,0] = sol[i,0]+dt;
            ctrl2i += 1
            vs[1].append(0)

        if ctrl2i == len(control2):
            ctrl2i = 0

    return sol, vs, vsms, control

def plot_mutual_stabilization(sol,save_direc):
    N = int(sol.shape[0]/4)
    fig, axs = plt.subplots(2,2,figsize=(10,8),subplot_kw=dict(projection='3d'))
    plt.setp(axs, xlim=(np.min(sol[:,1]),np.max(sol[:,1])), ylim=(np.min(sol[:,2]),np.max(sol[:,2])),zlim=(np.min(sol[:,3]),np.max(sol[:,3])))
    for i in range(2):
        for j in range(2):
            axs[i,j].plot(sol[(2*i+1)*N:(2*i+2)*N,3*j+1],sol[(2*i+1)*N:(2*i+2)*N,3*j+2],sol[(2*i+1)*N:(2*i+2)*N,3*j+3],linewidth=0.5)
            axs[i,j].set_xlabel('$x$'); axs[i,j].set_ylabel('$y$'); axs[i,j].set_zlabel('$z$');
            axs[i,j].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs[i,j].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs[i,j].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs[i,j].xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            axs[i,j].yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            axs[i,j].zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    axs[0,0].set_title('(A)',loc='right', y=1.0, pad=-14)
    axs[0,1].set_title('(B)',loc='right', y=1.0, pad=-14)
    axs[1,0].set_title('(C)',loc='right', y=1.0, pad=-14)
    axs[1,1].set_title('(D)',loc='right', y=1.0, pad=-14)
    plt.savefig('{0}/mutual_stabilization_3d.eps'.format(save_direc),dpi=300)
    plt.close()

def plot_raster(sol,save_direc,dt):
    N = int(sol.shape[0]/4)
    _,index_period,_ = sa.get_period(sol[3*N:4*N,1],dt)
    fig, axs = plt.subplots(2,1,figsize=(8,10))
    for i in range(2):
        ed = int((2*i+2)*N);
        st = ed-4*index_period;
        t = sol[st:ed,0]
        pks1,_ = find_peaks(sol[st:ed,1],height=1)
        pks2,_ = find_peaks(sol[st:ed,4],height=1)
        axs[i].eventplot([t[pks2],t[pks1]],linewidth=0.5)
        axs[i].set_yticks([0,1]); axs[i].set_yticklabels(['Neuron 2','Neuron 1'])
    plt.savefig('{0}/raster.eps'.format(save_direc),dpi=300)
    plt.close()

def save_data(sol,vs,vsms,control1, control2,control,save_direc):
    np.savetxt('{0}/sol.txt'.format(save_direc),sol,delimiter='\t',newline='\n')
    np.savetxt('{0}/control1.txt'.format(save_direc),np.asarray(control1),delimiter='\t',newline='\n')
    np.savetxt('{0}/control2.txt'.format(save_direc),np.asarray(control2),delimiter='\t',newline='\n')
    for i in range(1,3):
        np.savetxt('{0}/vs{1}.txt'.format(save_direc,i),np.asarray(vs[i-1]),delimiter='\t',newline='\n')
        np.savetxt('{0}/vsms{1}.txt'.format(save_direc,i),np.asarray(vsms[i-1]),delimiter='\t',newline='\n')
        np.savetxt('{0}/controlms{1}.txt'.format(save_direc,i),np.asarray(control[i-1]),delimiter='\t',newline='\n')

#direc = '/Volumes/JEP_IAM/NeuralNetwork/HindmarshRose/cupolets/standard_naming_tests'
#save_direc = '/Volumes/JEP_IAM/NeuralNetwork/HindmarshRose/cupolets/standard_naming_tests/bins_1600_rN_16/mutual_stabilization_new'


direc = "/Users/johnparker/paper_repos/Parker_Short_2023_mutual_stabilization/paper_data/"
save_direc = f"{direc}/mutual_stabilization_test"
kd.check_direc(save_direc)
dt = 1/128; tf = 10000; t0 = 0;
control1 = [0,0,1]; control2 = [0,1,0,1,0,0,1,0];
ifm = 4; ifn = 3;
save_direc = '{0}/c1_{1}_c2_{2}_q_{3}_k_{4}_none'.format(save_direc,cupg.ctrl_to_string(control1),cupg.ctrl_to_string(control2),ifm,ifn)
kd.check_direc(save_direc)

sol, vs, vsms, control = mutual_stabilization(direc,dt,t0,tf,control1,control2,ifm,ifn)
save_data(sol, vs, vsms, control1, control2, control, save_direc)

#data_direc = '/Volumes/JEP_IAM/NeuralNetwork/HindmarshRose/cupolets/standard_naming_tests/bins_1600_rN_16/mutual_stabilization_new/c1_C001_c2_C01_q_4_k_3'
#sol = np.loadtxt('{0}/sol.txt'.format(data_direc))
plot_raster(sol,save_direc,dt)
plot_mutual_stabilization(sol,save_direc)
