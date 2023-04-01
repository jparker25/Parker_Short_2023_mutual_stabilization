# Generates microcontrol mapping for all bins
# Author: John E. Parker

# Import Python modules
import numpy as np
import bisect, sys

# Import user modules
import keep_data as kd
import rk4 as rk4
import control_planes as cp

def establish_microcontrol(neuron,direc,bins_rn_direc,bins=1600,dt=1/128):
    '''
    Reads in neuron object, direc directory, number of bins bins, and time step dt.
    Generates ps0_micro and ps1_micro that represents mapping for each bin and save to direc.
    '''
    print("Creating microcontrol sequence based on cupolet control...")
    store_direc = "{0}/microcontrol".format(bins_rn_direc) # Directory where to store microcontrol info
    kd.check_direc(store_direc) # Create the directory if it does not exist

    # Create the microcontrol map for each control plane
    ps1_micro = micro_control_map(neuron,dt,bins,1,direc,bins_rn_direc)
    ps0_micro = micro_control_map(neuron,dt,bins,0,direc,bins_rn_direc)

    # Save the microcontrol map to the store_direc directory
    np.savetxt(f"{store_direc}/ps1_microcontrol.txt",ps1_micro,delimiter="\t",newline="\n",header="Microcontrol map for PS1")
    np.savetxt(f"{store_direc}/ps0_microcontrol.txt",ps0_micro,delimiter="\t",newline="\n",header="Microcontrol map for PS0")

    print(f"Saved microcontrol map to {store_direc}")

def micro_control_map(neuron,dt,bins,ps,direc,bins_rn_direc):
    '''
    Reads in neuron object, dt time step, number of bins bins, ps control plane, and direc directory. Returns corresponding PS control plane microcontrol mapping.
    '''
    # Read in the control plane initial conditions
    ps0inits = np.loadtxt('{0}/coding_fcn/ps0_bin_inits.txt'.format(bins_rn_direc))
    ps1inits = np.loadtxt('{0}/coding_fcn/ps1_bin_inits.txt'.format(bins_rn_direc))

    # Read in the control plane endpoints for the polynomial fits
    ps0endpts = np.loadtxt('{0}/coding_fcn/ps0_bin_endpoints.txt'.format(bins_rn_direc))
    ps1endpts = np.loadtxt('{0}/coding_fcn/ps1_bin_endpoints.txt'.format(bins_rn_direc))

    # Read in the vertices of each control plane
    ps0x, ps0y, ps0z = np.loadtxt('{0}/control_planes/ps0_vertices.txt'.format(direc),unpack=True)
    ps1x, ps1y, ps1z = np.loadtxt('{0}/control_planes/ps1_vertices.txt'.format(direc),unpack=True)

    # Create empty array to store microcontrol map
    psmicro_map = np.empty((bins,2))
    for i in range(0,bins): # Iterate through each bin and find the microcontrol map
        curr = ps1inits[i] if ps == 1 else ps0inits[i]; # Initial point dependent on which plane to start
        t = 0; # Start with t = 0
        crossed = False; # Contiue to integrate system until control plane is crossed
        while not crossed:
            next = rk4.rk4(curr,dt,neuron.hr_dynamics) # Integrate the system one time step
            if cp.crossed(curr[0],next[0],curr[1],next[1],1,np.array([[ps1x[0],ps1y[0]],[ps1x[1],ps1y[1]]])): # True if crossed PS1
                tp,xp,zp,yp = rk4.rk4_henon(np.array([t,curr[0],curr[2]]),curr[1],-(curr[1]-ps1y[0]),neuron.hr_dy_dynamics) # Find where trajectory intersects control plane
                ii = bisect.bisect(ps1endpts,xp)-1; # Find the left side of the intersecting bin endpoint
                psmicro_map[i,0] = ii; psmicro_map[i,1] = 1; # Store the values
                crossed = True # Mark as crossed to move to next bin
            elif cp.crossed(curr[0],next[0],curr[1],next[1],0,np.array([[ps0x[0],ps0y[0]],[ps0x[1],ps0y[1]]])): # True if crossed PS0
                tp,yp,zp,xp = rk4.rk4_henon(np.array([t,curr[1],curr[2]]),curr[0],-(curr[0]-ps0x[0]),neuron.hr_dx_dynamics) # Find where trajectory intersects control plane
                ii = bisect.bisect(ps0endpts,yp)-1; # Find the left side of the intersecting bin endpoint
                psmicro_map[i,0] = ii; psmicro_map[i,1] = 0; # Store the values
                crossed = True # Mark as crossed to move to next bin
            curr = next # Set new values to old values
            t += dt # Move time forward
    return psmicro_map # Return the microcontrol map
