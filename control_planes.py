# control_planes.py: Script that generates and analyzes the control planes
# Author: John E. Parker

# Import Python modules
import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

# Import user modules
import keep_data as kd
import rk4

def find_surfaces(neuron,dt=1/128,percent=75,direc='save_direc'):
    '''
    Reads in neuron, dt, and directory direc and generates all related Poincare surface information through
    various functions and plots that are stored in direc.
    '''
    print("Generating Poincare surfaces and corresponding intersections...")
    store_direc = "{0}/control_planes".format(direc) # Define directory name for storing control plane data
    kd.check_direc(store_direc) # Create control plane directory if it does not exist

    t,x,y,z = np.loadtxt('{0}/hr_system/hr_time_series.txt'.format(direc),unpack=True) # Read in data
    start = int((1-percent/100)*(len(t))) # Find index for percent data remaining
    t = t[start:]; x = x[start:]; y = y[start:]; z = z[start:]; # Redefine data for percent data
    peaks,_=find_peaks(x) # Find indicies of peaks for x time series - used for PS1
    mid_peaks,_ = find_peaks(-x,height=(0,1)) # Finds indices of -x time series b/w 0 and 1 - used for PS0

    verts0 = ps_verts(x,y,z,mid_peaks,0) # Find the vertices of the surface for PS0
    verts1 = ps_verts(x,y,z,peaks,1) # Find the vertices of the surface for PS1

    pts = ps_points(t,x,y,z,verts0,verts1,neuron) # Find all the points on the control planes

    # These lines are used to shrink PS0 to a better fit for the points on the plane
    yp0 = [p[2] for p in pts if p[4]==0] # Grab all the y values for the PS0 points on plane
    ypmax = max(yp0); ypmin = min(yp0); # Find the min and max of the y values
    ypinc = abs(ypmax-ypmin)*0.50; # Slightly extend the y range by 50%
    verts0[0,1] = ypmin-ypinc; verts0[1,1] = ypmax+ypinc; # Replace y values of PS0 with augmented ones

    np.savetxt(f"{store_direc}/ps0_vertices.txt",verts0,delimiter="\t",newline="\n",header="X Y Z limits of PS0 plane") # save PS0 data
    np.savetxt(f"{store_direc}/ps1_vertices.txt",verts1,delimiter="\t",newline="\n",header="X Y Z limits of PS1 plane") # Save PS1 data

    np.savetxt(f"{store_direc}/ps0_pts.txt",np.asarray([pt for pt in pts if pt[4] == 0]),delimiter="\t",newline="\n",header="PS0 points on plane.") # Save all the PS0 points
    np.savetxt(f"{store_direc}/ps1_pts.txt",np.asarray([pt for pt in pts if pt[4] == 1]),delimiter="\t",newline="\n",header="PS1 points on plane.") # Save all the PS1 points
    np.savetxt(f"{store_direc}/ps_all_pts.txt",pts,delimiter="\t",newline="\n",header="All points on control planes.") # Save all the PS0 and PS1 points

    generate_figure_ps_pts(t,x,y,z,store_direc,pts,verts0,verts1,True) # Create figure of HR neuron with Poincare surfaces and points
    generate_figure_ps_pts(t,x,y,z,store_direc,pts,verts0,verts1,False) # Create figure of HR neuron with Poincare surfaces and no points

    print(f"Completed integration of Poincare surfaces and established control planes. Saved to {direc}/control_planes")

def ps_verts(x,y,z,peaks,ps):
    '''
    Function that reads in x y z values, a peak array, and plane indicator ps. Returns a matrix containing
    the x y z values of either PS1 values (ps = 1) or PS0 values (ps = 0)
    '''
    verts = np.zeros((2,3))
    xpmin = min(x[peaks]); xpmax = max(x[peaks]); # Min and max of x peaks
    x_ext = abs(xpmax-xpmin)*0.05 # Increase x endpoints
    ypmin = min(y); ypmax = max(y); # Min and max of y values
    y_ext = abs(ypmax-ypmin)*0.05 # Increment y endpoints by percent
    zpmin = min(z); zpmax = max(z); # Min and max of z values
    z_ext = abs(zpmax-zpmin)*0.01 # Increase z endpoints
    if ps == 0:
        verts[:,0] = np.min(x[peaks])*np.ones(2) # Establish x PS0 values
        verts[:,1] = np.array([ypmin-y_ext,ypmax+y_ext]) # Establish y PS0 values
        verts[:,2] = np.array([zpmin-z_ext,zpmax+z_ext]) # Establish z PS0 vaues
    elif ps == 1:
        verts[:,0] = np.array([xpmin-x_ext,xpmax+x_ext]) # Establish x PS1 values
        verts[:,1] = np.mean(y)*np.ones(2) # Establish y PS1 values
        verts[:,2] = np.array([zpmin-z_ext,zpmax+z_ext]) # Establish z PS1 vaues
    return verts

def ps_points(t,x,y,z,verts0,verts1,neuron):
    '''
    Reads in time series t,x,y,z, vertex data for surfaces 0 and 1, as well as neural system neuron.
    Returns list of point of all crossings of PS0 and PS1 as t,x,y,z,PS where PS = 1 for PS1 or 0 for
    PS0. Uses Henon's trick to integrate exactly onto surface.
    '''
    pts = [] # Array to be filled with pts on contro plane
    for i in range(0,len(x)-1): # Iterate through all points
        xn = x[i]; xn1 = x[i+1]; # Grab consecutive x points
        yn = y[i]; yn1 = y[i+1]; # Grab consecutive y points
        zn = z[i]; zn1 = z[i+1]; # Grab consecutive z points
        if crossed(xn,xn1,yn,yn1,1,verts1): # Check to see PS1 has been crossed
            tp,xp,zp,yp = rk4.rk4_henon(np.array([t[i],xn,zn]),yn,-(yn-verts1[0,1]),neuron.hr_dy_dynamics) # If crossed, use Henon's trick to integrate onto surface
            pts.append([tp,xp,yp,zp,1]) # Add the surface points to the point array
        if crossed(xn,xn1,yn,yn1,0,verts0):  # Check to see PS1 has been crossed
            tp,yp,zp,xp = rk4.rk4_henon(np.array([t[i],yn,zn]),xn,-(xn-verts0[0,0]),neuron.hr_dx_dynamics) # If crossed, use Henon's trick to integrate onto surface
            pts.append([tp,xp,yp,zp,0]) # Add the surface points to the point array
    return np.asarray(pts) # Return all the points on the surfaces

def crossed(x_n,x_n1,y_n,y_n1,ps,verts):
    '''
    Reads in points x_n, x_n1, y_n, y_n1 and using surface ps (indicates which plane, 0 or 1) with
    vertices verts, checks to see if pts are on opposite sides of the surface
    '''
    if ps == 1: # Check for crossing PS1
        sp = y_n - verts[0,1]; sp1 = y_n1 - verts[0,1]; # Change of signs in y direction indicates crossing PS1
        if x_n <= verts[1,0] and x_n1 >= verts[0,0] and np.sign(sp1) != np.sign(sp) and np.sign(sp) == 1:
            return True # PS1 crossed if yn and yn1 signs differ, and the xn and xn1 are within the PS1 boundary
    if ps == 0:  # Check for crossing PS1
        sp = x_n - verts[0,0]; sp1 = x_n1 - verts[0,0]; # Change of signs in x direction indicates crossing PS1
        if y_n <= verts[1,1] and y_n1 >= verts[0,1] and np.sign(sp1) != np.sign(sp) and np.sign(sp) == 1:
            return True # PS0 crossed if xn and xn1 signs differ, and the yn and yn1 are within the PS1 boundary
    return False # False if neither plane crossed

def generate_figure_ps_pts(t,x,y,z,direc,pts,verts0,verts1,plot_pts):
    '''
    Reads in t,x,y,z HR data, direc directory to store plots, pts of poincare points,
    verts0 and verts 1 of the vertices of PS0 and PS1, and boolean plot_pts. Saves 3D figure of HR system with
    PS0 and PS1 plotted with points (plot_pts = True) or without (false)
    '''
    fig = plt.figure(figsize=(8,6),dpi=300) # Create figure
    ax = fig.add_subplot(1,1,1,projection='3d') # Make figure 3D projection
    ax.plot3D(x,y,z,linewidth=0.5) # Plot the data

    if plot_pts: # If true, then plot control plane points (PS1 purple, PS0 yellow)
        for p in pts:
            ax.scatter(p[1],p[2],p[3],s=2,color='purple' if p[4] == 1 else 'y')

    # Draw a black rectangle that is PS1
    ax.add_line(mplot3d.art3d.Line3D([verts1[0,0],verts1[1,0]],[verts1[0,1],verts1[0,1]],[verts1[0,2],verts1[0,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts1[0,0],verts1[1,0]],[verts1[0,1],verts1[0,1]],[verts1[1,2],verts1[1,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts1[0,0],verts1[0,0]],[verts1[0,1],verts1[0,1]],[verts1[0,2],verts1[1,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts1[1,0],verts1[1,0]],[verts1[0,1],verts1[0,1]],[verts1[0,2],verts1[1,2]],color='k'))

    # Draw a black rectangle that is PS0
    ax.add_line(mplot3d.art3d.Line3D([verts0[0,0],verts0[1,0]],[verts0[0,1],verts0[1,1]],[verts0[0,2],verts0[0,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts0[0,0],verts0[1,0]],[verts0[0,1],verts0[1,1]],[verts0[1,2],verts0[1,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts0[0,0],verts0[0,0]],[verts0[0,1],verts0[0,1]],[verts0[0,2],verts0[1,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts0[1,0],verts0[1,0]],[verts0[1,1],verts0[1,1]],[verts0[0,2],verts0[1,2]],color='k'))

    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$'); ax.set_zlabel(r'$z$') # Set figure labels
    ax.set_xlim([min(x),max(x)]); ax.set_ylim([min(y),max(y)]); ax.set_zlim([min(z),max(z)]) # Set axes limits
    plt.title('PS0 (Yellow), PS1 (Purple) Control Planes and Crossings' if plot_pts else 'Control Planes on HR Neuron') # Write title
    plt.savefig(f'{direc}/ps0_ps1_surfaces_crossings_pts_{plot_pts}.pdf') # Save the figure
    plt.close() # Close the figure
