# Generates the coding function for HR system
# Author: John E. Parker

# Import Python Modules
import numpy as np
from matplotlib import pyplot as plt
import sys

# Import user modules
import keep_data as kd
import control_planes as cp
import rk4

def coding_fcn(neuron,direc,bin_rn_direc,dt=1/128,coding_fcn_N=16,bins=1600):
    '''
    Read in HR neuron, dt, direc directory to store files, the number of iterations for the coding
    function to store and the number of bins to parse each poincare surface
    '''
    print("Generating coding functions...")
    store_direc = "{0}/coding_fcn".format(bin_rn_direc) # Directory to store all coding function data
    kd.check_direc(store_direc) # Create directory if it does not exist

    # Read in PS1 and PS0 vertices
    ps1x, ps1y, ps1z = np.loadtxt('{0}/control_planes/ps1_vertices.txt'.format(direc),unpack=True)
    ps0x, ps0y, ps0z = np.loadtxt('{0}/control_planes/ps0_vertices.txt'.format(direc),unpack=True)

    # Read in PS1 and PS0 poincare points
    ps1pts = np.loadtxt('{0}/control_planes/ps1_pts.txt'.format(direc))
    ps0pts = np.loadtxt('{0}/control_planes/ps0_pts.txt'.format(direc))

    # Create polynomial fits for PS1 and PS0 pts
    ps1_p, resid1_sq, poly1 = find_fit(ps1pts[:,1],ps1pts[:,3],3)
    ps0_p, resid0_sq, poly0 = find_fit(ps0pts[:,2],ps0pts[:,3],2)

    # Save the polynomial coefficients for each plane
    np.savetxt(f"{store_direc}/ps1_poly_coeffs.txt",ps1_p,delimiter="\t",newline="\n",header="Coefficients of approximating polynomial for PS1 (first is coefficient of highest degree).")
    np.savetxt(f"{store_direc}/ps0_poly_coeffs.txt",ps0_p,delimiter="\t",newline="\n",header="Coefficients of approximating polynomial for PS0 (first is coefficient of highest degree).")

    # Generate the end points, middle, and correspoding x y z values of each midpoint for both planes
    bin0, bin0_mids, bin0_inits = gen_bin_data(ps0x,ps0y,poly0,bins,0)
    bin1, bin1_mids, bin1_inits = gen_bin_data(ps1x,ps1y,poly1,bins,1)

    # Save the end points of the bins for PS0 and PS1
    np.savetxt(f"{store_direc}/ps0_bin_endpoints.txt",bin0,delimiter="\t",newline="\n",header="End points of each bin for PS0 (y values).")
    np.savetxt(f"{store_direc}/ps1_bin_endpoints.txt",bin1,delimiter="\t",newline="\n",header="End points of each bin for PS1 (x values).")

    # Save the mid points of each of the bins for PS0 and PS1
    np.savetxt(f"{store_direc}/ps0_bin_midpoints.txt",bin0_mids,delimiter="\t",newline="\n",header="Mid points of each bin for PS0 (y values).")
    np.savetxt(f"{store_direc}/ps1_bin_midpoints.txt",bin1_mids,delimiter="\t",newline="\n",header="Mid points of each bin for PS1 (x values).")

    # Save the initial points of the bins for PS0 and PS1 (x y z values corresponding to the middle of the bin on the planes)
    np.savetxt(f"{store_direc}/ps0_bin_inits.txt",bin0_inits,delimiter="\t",newline="\n",header="Initial conditions of each bin for PS0 (x y z values).")
    np.savetxt(f"{store_direc}/ps1_bin_inits.txt",bin1_inits,delimiter="\t",newline="\n",header="Initial conditions of each bin for PS1 (x y z values).")

    # Determine the coding sequence and corresponding binary representation of middle point on each bin
    rn1, map1 = gen_crossing_sequence(bin1_inits,neuron,ps0x,ps0y,ps1x,ps1y,N=coding_fcn_N,bins=bins,dt=dt)
    rn0, map0 = gen_crossing_sequence(bin0_inits,neuron,ps0x,ps0y,ps1x,ps1y,N=coding_fcn_N,bins=bins,dt=dt)

    # Save binary decmial for each coding map for each control plane
    np.savetxt(f"{store_direc}/ps1_rn_fcn.txt",rn1,delimiter="\t",newline="\n",header="Rn function for PS1 (binary decimal).")
    np.savetxt(f"{store_direc}/ps0_rn_fcn.txt",rn0,delimiter="\t",newline="\n",header="Rn function for PS0 (binary decimal).")

    # Save each coding map for each control plane
    np.savetxt(f"{store_direc}/ps1_rn_map_fcn.txt",map1,delimiter="\t",newline="\n",header="Symbolic future crossings of PS1 for each bin.")
    np.savetxt(f"{store_direc}/ps0_rn_map_fcn.txt",map0,delimiter="\t",newline="\n",header="Symbolic future crossings of PS1 for each bin.")

    plot_poly_fits(bins,ps1pts,ps0pts,bin1,bin0,poly1(bin1),poly0(bin0),resid1_sq,resid0_sq,store_direc) # Plot the polynomial fits
    plot_rn(rn1,rn0,store_direc,bins=bins,coding_fcn_N=coding_fcn_N) # Plto each coding function
    print(f"Saved coding function for each control plane to {store_direc}")

def gen_crossing_sequence(x,neuron,ps0x,ps0y,ps1x,ps1y,dt=1/128,N=16,bins=1600):
    '''
    Reads in list of initial bins x, crossings N, number of bins, dt, vertex data of PS0 and PS1
    (ps0x, ps0y, ps1x, ps1y) returns decimal value of binary visitation sequence and corresponding symbolic dynamics
    for each bin
    '''
    map = np.zeros((bins,N)) # Will contain the binary sequence of the next N control planes
    rmap = np.zeros(bins) # Will contain the binary decimal that map produces
    for i in range(0,bins): # Loop through each initial condition of eaach bin
        cross = 0; curr = x[i]; # Start with 0 crossings and the initial condition is the middle of the bin
        while cross < N: # Continue collecting crossings until N are found
            next = rk4.rk4(curr,dt,neuron.hr_dynamics) # Integrate one step forward
            if cp.crossed(curr[0],next[0],curr[1],next[1],0,np.array([[ps0x[0],ps0y[0]],[ps0x[1],ps0y[1]]])): # Check if PS0 has been crossed
                map[i,cross] = 0 # Add a 0 to the symbolic dynamics
                cross += 1 # Increase the number of crossings
            elif cp.crossed(curr[0],next[0],curr[1],next[1],1,np.array([[ps1x[0],ps1y[0]],[ps1x[1],ps1y[1]]])): # Check if PS1 has been crossed
                map[i,cross] = 1 # Add a 0 to the symbolic dynamics
                cross += 1 # Increase the number of crossings
            curr = next # New data because old data
        rmap[i] = np.sum([map[i,j]/(2**(j+1)) for j in range(N)])
    return rmap, map #  Return the r values and the corresponding maps

def rn(x):
    '''
    Reads in binary sequnce x and returns corresponding decimal value
    '''
    val = 0;
    for i in range(0,len(x)):
        val += x[i]/(2**(i+1))
    return val

def find_fit(ind,dep,degree):
    '''
    Returns polynomial, coefficients, and degree fit resid squared for a given set
    of independent and dependent values for given degree
    '''
    psp, resid, _,_,_ = np.polyfit(ind,dep,degree,full=True) # Call polyfit function to find coefficients of polynomial degree
    if degree == 3: # Return coefficients, squared residual fit and polynomical function for degree 3 polynomial
        return psp, np.dot(resid,resid), np.poly1d([psp[0],psp[1],psp[2],psp[3]])
    if degree == 2: # Return coefficients, squared residual fit and polynomical function for degree 2 polynomial
        return psp, np.dot(resid,resid), np.poly1d([psp[0],psp[1],psp[2]])

def gen_bin_data(psx,psy,poly,bins,ps):
    '''
    For surface PS reads in x vertex values psx, yvertex values psy and polynmoial fit poly,
    and bins to generate the bin spacing, midpoints, and initial data for evaluation
    '''
    if ps == 0: # Create all the bin values for PS0
        bin_endpts = np.linspace(psy[0],psy[1],bins+1) # The edges of all the bins
        mid_db = (bin_endpts[1]-bin_endpts[0])/2; # Half the distance between two bins
        bin_mids = np.linspace(bin_endpts[0]+mid_db,bin_endpts[-1]-mid_db,bins) # All the middles of the bins
        bin_inits = np.transpose(np.array([psx[0]*np.ones(bins),bin_mids,poly(bin_mids)])) # Initial conditions (x y z) corresponding to each bin
        return bin_endpts, bin_mids, bin_inits # return the endpoints, middles, and initial conditions
    if ps == 1: # Create all the bin values for PS1
        bin_endpts = np.linspace(psx[0],psx[1],bins+1)  # The edges of all the bins
        mid_db = (bin_endpts[1]-bin_endpts[0])/2;  # Half the distance between two bins
        bin_mids = np.linspace(bin_endpts[0]+mid_db,bin_endpts[-1]-mid_db,bins) # All the middles of the bins
        bin_inits = np.transpose(np.array([bin_mids,psy[0]*np.ones(bins),poly(bin_mids)])) # Initial conditions (x y z) corresponding to each bin
        return bin_endpts, bin_mids, bin_inits # return the endpoints, middles, and initial conditions

def plot_poly_fits(bins,ps1pts,ps0pts,bin1,bin0,polyfit1,polyfit0,resid1,resid0,direc):
    '''
    Saves a plot of the polynomials and their fits with the scatter poitns of the PS0 and PS1
    crossings.
    '''
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6),dpi=300) # Create figure to plot polynomial fits
    ax1.scatter(ps0pts[:,2],ps0pts[:,3],s=2,color='y') # Plot points on PS0 plane
    ax1.plot(bin0,polyfit0,linewidth=0.5) # Plot the polynomial fit of PS0
    ax1.set_xlabel(r"$y$"); ax1.set_ylabel(r"$z$") # Label the axes
    ax2.scatter(ps1pts[:,1],ps1pts[:,3],s=2,color='purple') # Plot points on PS1 plane
    ax2.plot(bin1,polyfit1,linewidth=0.5) # Plot the polynomial fit of PS1
    ax2.set_xlabel(r"$x$"); ax2.set_ylabel(r"$z$") # Label the axes
    plt.suptitle(f"PS0 (top) Resid Sq: {resid0:.2e}, PS1 (bottom) Resid Sq: {resid1:.2e}") # Title the plot
    plt.savefig(f"{direc}/ps1_ps0_poly_fit_w_pts_{bins}.pdf") # Save the figure
    plt.close() # Close the figure

def plot_rn(rn1,rn0,direc,bins=1600,coding_fcn_N=16):
    '''
    Saves a plot in direc of the coding function for each PS0 and PS1
    '''
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6), dpi =300) # Create the figure to plot the coding functions
    ax1.plot(list(range(bins)),rn0,lw=0.5); ax1.set_xlabel("Bin"); ax1.set_ylabel(r"$r_N(x)$"); # Plot the coding function and label the axes for PS0
    ax2.plot(list(range(bins)),rn1,lw=0.5); ax2.set_xlabel("Bin"); ax2.set_ylabel(r"$r_N(x)$"); # Plot the coding function and label the axes for PS1
    plt.suptitle(f"PS0 (top), PS1 (bottom) r(x) - {coding_fcn_N} Crossings, {bins} Bins") # Title the figure
    plt.savefig(f"{direc}/ps1_ps0_coding_fcn.pdf") # Save the figure
    plt.close() # Close the figure
