# Generates macrocontrol map for a control plane
# Author: John E. Parker

# Import Python modules
import numpy as np
from matplotlib import pyplot as plt

# Import User modules
import keep_data as kd


def establish_macrocontrol(direc,coding_fcn=16):
    '''
    Reads in the number of bins, crossings, and top level directory name. Generates the macrocontrol
    sequence that corresponds to the number of bins and crossings and saves in directory /direc/macrocontrol.
    '''
    print("Creating macrocontrol sequence based on cupolet control...")
    store_direc = "{0}/macrocontrol".format(direc) # Directory where to store macrocontrol info
    kd.check_direc(store_direc) # Create the directory if it does not exist

    # Read in the coding function for each control plane
    rn0 = np.loadtxt('{0}/coding_fcn/ps0_rn_fcn.txt'.format(direc))
    rn1 = np.loadtxt('{0}/coding_fcn/ps1_rn_fcn.txt'.format(direc))

    # Find the macrocontrol map for each control plane, and the difference map
    ctrl1, diff1 = macro_control_map(rn1)
    ctrl0, diff0 = macro_control_map(rn0)

    # plot the macrocontrol and difference maps of each plane
    macro_control_graphics(ctrl0,diff0,0,store_direc)
    macro_control_graphics(ctrl1,diff1,1,store_direc)

    # Save the macrocontrol map data
    np.savetxt(f"{store_direc}/ps0_macrocontrol.txt",ctrl0,delimiter="\t",newline="\n",header="PS0 macrocontrol map.") # Save the macrocontrol map for PS0
    np.savetxt(f"{store_direc}/ps1_macrocontrol.txt",ctrl1,delimiter="\t",newline="\n",header="PS1 macrocontrol map.") # Save the macrocontrol map for PS1
    np.savetxt(f"{store_direc}/ps0_diff_macrocontrol.txt",diff0,delimiter="\t",newline="\n",header="PS0 macrocontrol difference map.") # Save the macrocontrol difference map for PS0
    np.savetxt(f"{store_direc}/ps1_diff_macrocontrol.txt",diff1,delimiter="\t",newline="\n",header="PS1 macrocontrol difference map.") # Save the macrocontrol difference map for PS1

    print(f"Saved macrocontrol map to {store_direc}")


def macro_control_map(rn):
    '''
    Reads in a coding sequence rn and number of bins. Returns an array that maps bin B to
    the nearest smallest difference of the coding sequence and an array that says the size of
    that difference.
    '''
    bins = len(rn) # Define the number of bins
    ctrl = np.zeros(bins); diff = np.zeros(bins); # Create empty arrays for the control map and differences
    for i in range(0,bins): # Iterate through each bin and find the macrocontrol destination
        diffs = abs(rn-rn[i]) # Find all the differences from the ith bin
        #sorted_diffs = sorted(range(len(diffs)), key=lambda k: diffs[k])
        min_sort = np.where(diffs == min(diffs[np.where(diffs >0)]))[0]; # Find the bins that are equal to the smallest difference
        bin_dist = abs(min_sort-i) # Calculate the bin distances
        diff[i] = min(diffs[min_sort]) # Find the smallest bin difference
        min_diff_sort = min(bin_dist) # Find the value of the smallest bin difference
        reps = np.count_nonzero(bin_dist == min_diff_sort) # Count the number of smallest bin differences
        if reps == 1: # If only 1 then store that as the nearest value
            ctrl[i] = min_sort[np.argmin(bin_dist)]
        else: # If there are mulitple find the one that is closest by bin order
            new = np.where(abs(min_sort-i) == min(bin_dist))[0] # Grab the first bin that is the closest distance
            ctrl[i] = min_sort[new[1]] # Store the bin
    return ctrl, diff # Return the ctrl, diff arrays

def macro_control_graphics(ctrl,diff,ps,direc):
    '''
    Saves a figure in direc that shows the macrocontrol bin shift and the distance
    of the coding function rn between that shift. Reads in ctrl and diff arrays that
    have this data.
    '''
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,8),dpi=300) # Create the figure
    ax1.plot(list(range(0,len(ctrl))),ctrl,lw=0.5) # Plot the macrocontrol map
    ax1.set_ylabel("Macrocontrol Bin Shift") # Set the ylabel of the macrocontrol map
    ax2.plot(list(range(0,len(ctrl))),diff,lw=0.5) # # Plot the differences of the control map
    ax2.set_xlabel("Bin"); ax2.set_ylabel(r"Macrocontrol $r_N$ Difference") # Set tye x and y label of the difference map
    plt.suptitle(r"PS{0} Macrocontrol Bin Displacement (Top) and $r_N$ Difference (Bottom)".format(ps)) # Title the figure
    plt.savefig(f"{direc}/ps{ps}_macrocontrol_visual.pdf") # Save the figure
    plt.close() # Close the figure
