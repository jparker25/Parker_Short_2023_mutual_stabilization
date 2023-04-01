# File to run cupolet generation program with various options
# Author: John Parker

# Import Python modules
import argparse, os, pickle, sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
import itertools

# Import user modules
from hindmarsh_rose import hindmarsh_rose as hr
import rk4
import keep_data as kd
import control_planes as cp
import coding_fcn as cf
import macrocontrol as mac
import microcontrol as mic
import cupolet_search as cs

parser = argparse.ArgumentParser(description='Runs cupolet generation program. Various options described below.')

parser.add_argument('-I',nargs='?',default=3.25,type=float,help="Corresponding parameter in HR system. Default: 3.25.")
parser.add_argument('-r',nargs='?',default=0.006,type=float,help="Corresponding parameter in HR system. Default: 0.006")
parser.add_argument('-a',nargs='?',default=1.0,type=float,help="Corresponding parameter in HR system. Default: 1.0")
parser.add_argument('-b',nargs='?',default=3.0,type=float,help="Corresponding parameter in HR system. Default: 3.0")
parser.add_argument('-c',nargs='?',default=1.0,type=float,help="Corresponding parameter in HR system. Default: 1.0")
parser.add_argument('-d',nargs='?',default=5.0,type=float,help="Corresponding parameter in HR system. Default: 5.0")
parser.add_argument('-s',nargs='?',default=4.0,type=float,help="Corresponding parameter in HR system. Default: 4.0")
parser.add_argument('-xr',nargs='?',default=-1.6,type=float,help="Corresponding parameter in HR system. Default: -1.6")

parser.add_argument('-direc',nargs='?',default='save_direc',type=str,help="Directory to store all simulation data including subdirectories. Assumes new data is being used. Default: ./save_direc")
parser.add_argument('-neuron',default='',type=str,help="Designated as path directory that contains neuron object. Assumes old data is being used. Default: ''")
parser.add_argument('-dt',nargs='?',default=1/128.0,type=float,help="Discrete time step for numerical integration. Default: 1/128")
parser.add_argument('-secs',nargs='?',default=10000.0,type=float,help="Number of seconds to simulate. Iterations will be N = secs/dt. Default: 10000.0")
parser.add_argument('-pplot',nargs='?',default=75.0,type=float,help="Percent of iterations to plot for initial simulation of HR system. Default: 75.0")
parser.add_argument('-generate',action='store_true',help="If given, generate the found cupolets.")

parser.add_argument('-bins',nargs='?',default=1600,type=int,help="Number of bins to partition control planes into. Default: 1600")
parser.add_argument('-crossings',nargs='?',default=16,type=int,help="Number of total crossings of control planes for coding function. Default: 16")

parser.add_argument('-csecs',nargs='?',default=2000.0,type=float,help="Number of seconds to simulate cupolet. Iterations will be N = csecs/dt. Default: 2000.0")
parser.add_argument('-cplot',nargs='?',default=75.0,type=float,help="Percent of iterations to store for simulation of cupolet. Default: 75.0")
parser.add_argument('-ctrl',nargs='+',default=[0,0],type=int,help="Control code for generation of cupolet. Default: 0 0")
parser.add_argument('-bits',nargs='?',default=0,type=int,help="Flag to search for all possible cupolets up to given integer number of bits.")

args = parser.parse_args() # read in passed arguments

if args.neuron == '' and args.direc != '': # If no path to neuron object provided, create a new neuron
    kd.check_direc(args.direc) # Create the directory to save data if it does not already exist
    neuron = hr(I=args.I,r=args.r,a=args.a,b=args.b,c=args.c,d=args.d,s=args.s,xr=args.xr) # Create the HR enruon
    neuron.save_neuron(args.direc) # Save the HR neuron object
    sol = rk4.rk4_N(np.random.rand(3),0,args.dt,neuron.hr_dynamics,int(args.secs/args.dt)) # Integrate HR neuron
    kd.percent_keep(sol,args.pplot,args.direc) # Save HR integration
    cp.find_surfaces(neuron,dt=args.dt,percent=args.pplot,direc=args.direc) # Creates and saves data for control planes based on args.pplot percent of initial simulation (default 75 ignores transient)

    bin_rn_direc = f"{args.direc}/bins_{args.bins}_rN_{args.crossings}"; # Define the directory to store cupolet data
    kd.check_direc(bin_rn_direc) # Create directory to store cupolet data if it does not exist

    cf.coding_fcn(neuron,args.direc,bin_rn_direc,dt=args.dt,coding_fcn_N=args.crossings,bins=args.bins) # generate the coding function

    mac.establish_macrocontrol(bin_rn_direc,args.crossings) # Generate macrocontrol map for each control plane
    mic.establish_microcontrol(neuron,args.direc,bin_rn_direc,args.bins,args.dt) # Generate microcontrol map for each plane

    if args.bits != 0: # If bits is given, iterate through all possible control sequences to search for a cupolet
        cs.find_all_cupolets(args.bits,neuron,int(args.csecs/args.dt),bin_rn_direc,args.direc,args.generate,args.dt,args.bins,args.crossings,args.cplot)
    else:
        cbins = cs.find_cupolets(args.ctrl,bin_rn_direc) # Find all cupolets that match with ctrl
        if args.generate: # If true, then plot the cupolet that was located
            cs.generate_found_cupolets(cbins,neuron,args.ctrl,int(args.csecs/args.dt),bin_rn_direc,args.direc,args.dt,args.bins,args.crossings,args.cplot) # Generate and save cupolets that are found

elif args.neuron != '': # If path to neuron path provided, read in neuron object in neuron path
    neuron = pickle.load(open(f"{args.neuron}/neuron.obj",'rb'))

    bin_rn_direc = f"{args.neuron}/bins_{args.bins}_rN_{args.crossings}"; # Define the directory to store cupolet data
    kd.check_direc(bin_rn_direc) # Create directory to store cupolet data if it does not exist

    if args.bits != 0: # If bits is given, iterate through all possible control sequences to search for a cupolet
        cs.find_all_cupolets(args.bits,neuron,int(args.csecs/args.dt),bin_rn_direc,args.neuron,args.generate,args.dt,args.bins,args.crossings,args.cplot)
    else:
        cbins = cs.find_cupolets(args.ctrl,bin_rn_direc) # Find all cupolets that match with ctrl
        if args.generate: # If true, then plot the cupolet that was located
            cs.generate_found_cupolets(cbins,neuron,args.ctrl,int(args.csecs/args.dt),bin_rn_direc,args.neuron,args.dt,args.bins,args.crossings,args.cplot) # Generate and save cupolets that are found
