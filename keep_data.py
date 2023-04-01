# keep_data.py script. Reads in a data set of txyz columns and saves the raw data
# and a 3d plot of the data in a specified directory
# Author: John E. Parker
import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

def check_direc(direc):
    '''
    Simple script to create directory if it does not exist.
    '''
    if not os.path.exists(direc):
        os.mkdir(direc)

def percent_keep(data,percent,direc):
    '''
    Reads in data assumed to be columns of t,x,y,z, variable percent that will be the
    percent of iteratiosn to keep from data, and directory direc to store all information
    Calls functions to write data to a time series file and save 3d figure
    '''
    print("Evaluating initial HR system...")
    direc = f"{direc}/hr_system" # directory name to save HR integration
    check_direc(direc) # Check if directory exists, if not create it, for HR integration
    np.savetxt(f"{direc}/hr_time_series.txt",data,delimiter="\t",newline="\n",header="t x y z data from HR integration") # Save all the data
    plot_3d_xyz(data[int((1-percent/100)*(data.shape[0])):,:],direc) # Plot percent of data in 3D
    print(f"Saved initial simulation of HR system to {direc}")

def plot_3d_xyz(data,direc):
    '''
    Reads in a set of data called data that is assumed to have columns t,x,y,z
    Plot data 3D figure (pdf) and save as hr_3d_orig.pdf in direc
    '''
    fig = plt.figure(figsize=(8,6)) # Create figure to plot on
    ax = fig.add_subplot(1,1,1,projection='3d') # Make figure 3d
    ax.plot3D(data[:,1],data[:,2],data[:,3],linewidth=0.5) # Plot 3d data
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$'); ax.set_zlabel(r'$z$') # Set x y z labels
    plt.title("HR System") # Set title
    plt.savefig(f"{direc}/hr_3d_orig.pdf",dpi=300) # Save figure
    plt.close() # Close figure
