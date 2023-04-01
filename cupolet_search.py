# Searches for cupolet sequences through the macro and micro control mappings.
# Author: John E. Parker

# Import Python modules
import numpy as np
import bisect, sys
from matplotlib import pyplot as plt
from itertools import permutations
import itertools

# Import user modules
import keep_data as kd
import rk4
import control_planes as cp

def find_cupolets(ctrl,bin_rn_direc):
    ps0micro = np.loadtxt('{0}/microcontrol/ps0_microcontrol.txt'.format(bin_rn_direc)) # Read in ps0micro control map
    ps1micro = np.loadtxt('{0}/microcontrol/ps1_microcontrol.txt'.format(bin_rn_direc)) # Read in ps1micro control map
    ps0macro = np.loadtxt('{0}/macrocontrol/ps0_macrocontrol.txt'.format(bin_rn_direc)) # Read in ps0macro control map
    ps1macro = np.loadtxt('{0}/macrocontrol/ps1_macrocontrol.txt'.format(bin_rn_direc)) # Read in ps1macro control map
    found = 0
    found_cupolets = []
    for i in range(ps0micro.shape[0]): # Iterate through all possible initial conditions on PS1 to find a cupolet
        cupolet = False # Boolean that determines if cupolet has been found
        ci = 0; # Index of control
        cbins = [[i,1]] # Initial starting position
        checks = 0
        while not cupolet and checks < 600: # While cupolet is not found, map the bin connections
            if ctrl[ci] == 0: # If the current control is a 0, apply a microcontrol
                if cbins[-1][1] == 1: # If the current bin is on PS1 apply PS1 micrcontrol
                    cbins.append([int(ps1micro[cbins[-1][0]][0]),int(ps1micro[cbins[-1][0]][1])])
                    checks += 1
                else: # If the current bin is on PS0 apply PS0 micrcontrol
                    cbins.append([int(ps0micro[cbins[-1][0]][0]),int(ps0micro[cbins[-1][0]][1])])
            else: # If the curret control is a 1, apply a macrocontrol
                if cbins[-1][1] == 1: # If the current bin is on PS1 apply PS1 macrcontrol
                    cbins.append([int(ps1macro[cbins[-1][0]][0]),1])
                    cbins.append([int(ps1micro[cbins[-1][0]][0]),int(ps1micro[cbins[-1][0]][1])])
                    checks += 1
                else: # If the current bin is on PS0 apply PS0 macrcontrol
                    cbins.append([int(ps0macro[cbins[-1][0]][0]),0])
                    cbins.append([int(ps0micro[cbins[-1][0]][0]),int(ps0micro[cbins[-1][0]][1])])
            if cbins[0] == cbins[-1] and ci == len(ctrl)-1:
                cupolet = True
                found += 1
                found_cupolets.append(cbins)
            ci = 0 if ci == len(ctrl)-1 else ci + 1
    return found_cupolets

def find_all_cupolets(bits,neuron,Nfinal,bin_rn_direc,direc,generate,dt=1/128,bins=1600,coding_fcn_N=16,cupolet_percent=75):
    print(f'Finding all possible cupolet sequences of 2 to {bits} total bits...')
    all_sequences = []
    for bits in range(2,bits+1):
        sequences = itertools.product("01",repeat=bits)
        for seq in sequences:
            all_sequences.append([eval(x) for x in seq])
    print(f'{len(all_sequences)} total cupolet sequences to use...')
    cups = 0
    for seq in all_sequences:
        cbins = find_cupolets(seq,bin_rn_direc) # Find all cupolets that match with ctrl
        if generate:
            generate_found_cupolets(cbins,neuron,seq,Nfinal,bin_rn_direc,direc,dt,bins,coding_fcn_N,cupolet_percent) # Generate and save cupolets that are found
        if len(cbins) > 0:
            cups += 1
            print(f'\t Found cupolet with sequence {seq}')
    print(f'Finished cupolet search. Found {cups}/{len(all_sequences)} cupolets.')


def generate_found_cupolets(found_cupolets,neuron,control,Nfinal,bin_rn_direc,direc,dt=1/128,bins=1600,coding_fcn_N=16,cupolet_percent=75):
    store_direc = f"{bin_rn_direc}/cupolets"
    kd.check_direc(store_direc)
    store_direc = f"{store_direc}/{ctrl_to_string(control)}"
    kd.check_direc(store_direc)
    for cup in found_cupolets:
        bin_init = cup[0][0]
        cup_direc = f"{store_direc}/bin_ps1_{bin_init}"
        kd.check_direc(cup_direc)
        cup = cupolet_test(neuron,dt,bin_init,control,Nfinal,bin_rn_direc,direc)
        np.savetxt(f"{cup_direc}/cupolet_time_series.txt",cup[int((1-cupolet_percent/100)*cup.shape[0]):],delimiter="\t",newline="\n",header="T X Y Z cupolet values")
        plot_cupolet(cup,control,cup_direc,bins,coding_fcn_N)

def cupolet_test(neuron,dt,bin_init,control,Nfinal,bin_rn_direc,direc):
    '''
    Generates cupolet based off of control. Reads in neuron state, dt, number of bins and
    crossings, control sequence, number of iterations, directory to pull info, and ctrl0,ctrl1 arrays
    that tell how to implement macrocontrol. Returns time series of cupolet.
    '''
    ps0inits = np.loadtxt('{0}/coding_fcn/ps0_inits.txt'.format(bin_rn_direc))
    ps1inits = np.loadtxt('{0}/coding_fcn/ps1_inits.txt'.format(bin_rn_direc))

    ps0endpts = np.loadtxt('{0}/coding_fcn/ps0_endpoints.txt'.format(bin_rn_direc))
    ps1endpts = np.loadtxt('{0}/coding_fcn/ps1_endpoints.txt'.format(bin_rn_direc))

    ps0x, ps0y, ps0z = np.loadtxt('{0}/control_planes/ps0_vertices.txt'.format(direc),unpack=True)
    ps1x, ps1y, ps1z = np.loadtxt('{0}/control_planes/ps1_vertices.txt'.format(direc),unpack=True)

    ctrl1 = np.loadtxt('{0}/macrocontrol/ps1_macrocontrol.txt'.format(bin_rn_direc))
    ctrl0 = np.loadtxt('{0}/macrocontrol/ps0_macrocontrol.txt'.format(bin_rn_direc))

    backup = np.empty((5,4))
    backup[0,1:] = ps1inits[bin_init];
    backup[0,0] = 0;
    for i in range(backup.shape[0]-1):
        backup[i+1,1:] = rk4.rk4(backup[i,1:],-dt/2,neuron.hr_dynamics)
    cupolet = np.empty((Nfinal,4))
    cupolet[0,1:] = backup[-1,1:]
    cupolet[0,0] = 0
    ctrli = 0
    #print(control)
    for i in range(0,Nfinal-1):
        curr = cupolet[i,1:]
        next = rk4.rk4(curr,dt,neuron.hr_dynamics)
        if cp.crossed(curr[0],next[0],curr[1],next[1],1,np.array([[ps1x[0],ps1y[0]],[ps1x[1],ps1y[1]]])):
            tp,xp,zp,yp = rk4.rk4_henon(np.array([cupolet[i,0],curr[0],curr[2]]),curr[1],-(curr[1]-ps1y[0]),neuron.hr_dy_dynamics)
            ii = bisect.bisect(ps1endpts,xp)-1;
            #print(ctrl1[ii])
            newval = ps1inits[ii] if control[ctrli] == 0 else  ps1inits[int(ctrl1[ii][0])];
            newdt = dt-(tp-cupolet[i,0]);
            next = rk4.rk4(newval,newdt,neuron.hr_dynamics)
            cupolet[i+1,1:] = next;
            cupolet[i+1,0] = cupolet[i,0]+dt;
            ctrli += 1

        elif cp.crossed(curr[0],next[0],curr[1],next[1],0,np.array([[ps0x[0],ps0y[0]],[ps0x[1],ps0y[1]]])):
            tp,yp,zp,xp = rk4.rk4_henon(np.array([cupolet[i,0],curr[1],curr[2]]),curr[0],-(curr[0]-ps0x[0]),neuron.hr_dx_dynamics)
            ii = bisect.bisect(ps0endpts,yp)-1;
            newval = ps0inits[ii] if control[ctrli] == 0 else  ps0inits[int(ctrl0[ii][0])];
            newdt = dt-(tp-cupolet[i,0]);
            next = rk4.rk4(newval,newdt,neuron.hr_dynamics)
            cupolet[i+1,1:] = next;
            cupolet[i+1,0] = cupolet[i,0]+dt;
            ctrli += 1

        else:
            cupolet[i+1,1:] = next;
            cupolet[i+1,0] = cupolet[i,0]+dt;

        if ctrli == len(control):
            ctrli = 0;
    return cupolet

def plot_cupolet(cupolet,control,direc,bins,coding_fcn_N):
    '''
    Plots and stores a figure of the cupolet and respective time series for each variable.
    '''
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot3D(cupolet[:,1],cupolet[:,2],cupolet[:,3],linewidth=0.5)
    plt.title("HR {0} Time Series Data".format(ctrl_to_string(control)))
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$'); ax.set_zlabel(r'$z$')
    plt.savefig('{0}/cupolet_3d_plot.pdf'.format(direc),dpi=300)
    plt.close()

    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,8))
    ax1.plot(cupolet[:,0],cupolet[:,1],linewidth=0.5)
    ax1.set_ylabel(r"$x$")
    ax2.plot(cupolet[:,0],cupolet[:,2],linewidth=0.5)
    ax2.set_ylabel(r"$y$")
    ax3.plot(cupolet[:,0],cupolet[:,3],linewidth=0.5)
    ax3.set_xlabel(r"$t$"); ax3.set_ylabel(r"$z$");
    plt.suptitle("HR {0} Time Series Data".format(ctrl_to_string(control)))
    plt.savefig('{0}/cupolet_time_series_plot.pdf'.format(direc),dpi=300)
    plt.close()

def ctrl_to_string(control):
    '''
    Reads in array of 0 and 1s and returns a string of them with a C at the beginning
    '''
    str = "C"
    for i in range(0,len(control)):
        str = "{0}{1}".format(str,control[i])
    return str
