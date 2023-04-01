# generate_graphics.py
# Creates figures for HR Single Cupolet Paper.
# Author: John E. Parker
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import os, sys, argparse
import matplotlib
from scipy.signal import find_peaks
import string
import seaborn as sns

import signal_analysis as sa
plt.rcParams.update({'font.size': 18})

def figure1(direc,save_direc):
    '''
    Bifurcation diagram.
    '''
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    for file in os.listdir(direc):
        if file.endswith('.txt'):
            i = float(file[8:13])
            t = np.loadtxt('{0}/{1}'.format(direc,file),unpack=True,usecols=0)
            t_isi = t[1:] - t[0:-1];
            ax.scatter(i*np.ones(len(t_isi)),t_isi,color='b',s=2)
    for i in ['left','right','top','bottom']:
        ax.spines[i].set_linewidth(2)
    ax.tick_params('both', width=2)
    sns.despine()
    ax.set_ylabel(r'$ISI$',fontsize=18); ax.set_xlabel(r'$I$',fontsize=18);
    plt.savefig("{0}/figure_1.eps".format(save_direc),dpi=300)
    plt.close()

def figure2(data_direc,dt,plot_percent,save_direc):
    '''
    Figure 2 - Chaotic dynamics of HR system. Phase space plot.
    '''
    plt.rcParams.update({'font.size': 14})
    t,x,y,z = np.loadtxt('{0}/hr_system/hr_time_series.txt'.format(data_direc),unpack=True) # Read in data
    st = int(len(x)*(1-plot_percent)) # Only graph plot_percent of data
    t=t[st:]; x=x[st:]; y=y[st:]; z=z[st:];

    fig = plt.figure(figsize=(10,8)); # Create figure for plotting
    ax1 = fig.add_subplot(221,projection='3d'); # subplot 1 - 3D phase space
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$"); ax1.set_zlabel("$z$"); # axes label
    ax1.set_title('(A)',loc='right', y=1.0, pad=-14)
    ax1.plot(x,y,z,linewidth=0.5,color="b")

    ax2=fig.add_subplot(222); # Subplot 2 - x y projection
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$"); # axes label
    ax2.set_title('(B)',loc='right', y=1.0, pad=-14)
    ax2.plot(x,y,linewidth=0.5,color="b")

    ax3=fig.add_subplot(223); # Subplot 3 - x z projection
    ax3.set_xlabel("$x$"); ax3.set_ylabel("$z$");  # axes label
    ax3.set_title('(C)',loc='right', y=1.0, pad=-14)
    ax3.plot(x,z,linewidth=0.5,color="b")

    ax4=fig.add_subplot(224); # Subplot 4 - y z projection
    ax4.set_xlabel("$y$"); ax4.set_ylabel("$z$"); # axes label
    ax4.set_title('(D)',loc='right', y=1.0, pad=-14)
    ax4.plot(y,z,linewidth=0.5,color="b")

    for i in ['left','right','top','bottom']:
        ax1.spines[i].set_linewidth(2)
        ax2.spines[i].set_linewidth(2)
        ax3.spines[i].set_linewidth(2)
        ax4.spines[i].set_linewidth(2)
        ax1.tick_params('both', width=2)
        ax2.tick_params('both', width=2)
        ax3.tick_params('both', width=2)
        ax4.tick_params('both', width=2)
    sns.despine()

    plt.tight_layout()
    plt.savefig("{0}/figure_2.eps".format(save_direc),dpi=300)
    plt.close()

def figure3(data_direc,dt,plot_percent,save_direc):
    '''
    Figure 3 - Chaotic dynamics of HR system. Time series plot.
    '''
    plt.rcParams.update({'font.size': 14})
    t,x,y,z = np.loadtxt('{0}/hr_system/hr_time_series.txt'.format(data_direc),unpack=True) # Read in data
    st = int(len(x)*(1-plot_percent)) # Only graph plot_percent of data
    t=t[st:]; x=x[st:]; y=y[st:]; z=z[st:];

    fig = plt.figure(figsize=(8,12)); # Create figure for plotting
    ax1 = fig.add_subplot(311); # subplot 1 - 3D phase space
    ax1.set_xlabel("$t$"); ax1.set_ylabel("$x$");  # axes label
    ax1.set_title('(A)',loc='right', y=1.0, pad=-14)
    ax1.plot(t,x,linewidth=1,color="b")

    ax2=fig.add_subplot(312); # Subplot 2 - x y projection
    ax2.set_xlabel("$t$"); ax2.set_ylabel("$y$"); # axes label
    ax2.set_title('(B)',loc='right', y=1.0, pad=-14)
    ax2.plot(t,y,linewidth=1,color="b")

    ax3=fig.add_subplot(313); # Subplot 3 - x z projection
    ax3.set_xlabel("$t$"); ax3.set_ylabel("$z$");  # axes label
    ax3.set_title('(C)',loc='right', y=1.0, pad=-14)
    ax3.plot(t,z,linewidth=1,color="b")

    for i in ['left','right','top','bottom']:
        ax1.spines[i].set_linewidth(2)
        ax2.spines[i].set_linewidth(2)
        ax3.spines[i].set_linewidth(2)
    ax1.tick_params('both', width=2)
    ax2.tick_params('both', width=2)
    ax3.tick_params('both', width=2)
    sns.despine()

    plt.tight_layout()
    plt.savefig("{0}/figure_3.eps".format(save_direc),dpi=300)
    plt.close()

def figure4(data_direc,plot_percent,save_direc):
    '''
    Figure 4 - control planes on 3D HR attractor with Henon's trick points.
    '''
    plt.rcParams.update({'font.size': 14})
    t,x,y,z = np.loadtxt('{0}/hr_system/hr_time_series.txt'.format(data_direc),unpack=True) # Read in all HR system chaotic data
    st = int(len(x)*(1-plot_percent)) # Only graph plot_percent of data
    t=t[st:]; x=x[st:]; y=y[st:]; z=z[st:];

    pts = np.loadtxt('{0}/control_planes/ps_all_pts.txt'.format(data_direc)) #Poincare points
    verts0 = np.loadtxt('{0}/control_planes/ps0_vertices.txt'.format(data_direc)) #vertices of PS0
    verts1 = np.loadtxt('{0}/control_planes/ps1_vertices.txt'.format(data_direc)) #vertices of PS1

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot3D(x,y,z,linewidth=0.5,color="b")

    for p in pts:
        if p[4] == 1:
            ax.scatter(p[1],p[2],p[3],color='purple')
        else:
            ax.scatter(p[1],p[2],p[3],color='y')

    ax.add_line(mplot3d.art3d.Line3D([verts1[0,0],verts1[1,0]],[verts1[0,1],verts1[0,1]],[verts1[0,2],verts1[0,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts1[0,0],verts1[1,0]],[verts1[0,1],verts1[0,1]],[verts1[1,2],verts1[1,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts1[0,0],verts1[0,0]],[verts1[0,1],verts1[0,1]],[verts1[0,2],verts1[1,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts1[1,0],verts1[1,0]],[verts1[0,1],verts1[0,1]],[verts1[0,2],verts1[1,2]],color='k'))

    ax.add_line(mplot3d.art3d.Line3D([verts0[0,0],verts0[1,0]],[verts0[0,1],verts0[1,1]],[verts0[0,2],verts0[0,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts0[0,0],verts0[1,0]],[verts0[0,1],verts0[1,1]],[verts0[1,2],verts0[1,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts0[0,0],verts0[0,0]],[verts0[0,1],verts0[0,1]],[verts0[0,2],verts0[1,2]],color='k'))
    ax.add_line(mplot3d.art3d.Line3D([verts0[1,0],verts0[1,0]],[verts0[1,1],verts0[1,1]],[verts0[0,2],verts0[1,2]],color='k'))

    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$'); ax.set_zlabel(r'$z$')
    ax.set_xlim([min(x),max(x)]); ax.set_ylim([min(y),max(y)]); ax.set_zlim([min(z),max(z)])
    plt.tight_layout()
    plt.savefig('{0}/figure_4.pdf'.format(save_direc),dpi=300)

    plt.close()

def figure5(data_direc,save_direc):
    '''
    Figure 5 - polynomial fit for Poincare points and residuals.
    '''
    plt.rcParams.update({'font.size': 14})
    ps0pts = np.loadtxt('{0}/control_planes/ps0_pts.txt'.format(data_direc))
    ps1pts = np.loadtxt('{0}/control_planes/ps1_pts.txt'.format(data_direc))

    psp0, resid0, _, _, _ = np.polyfit(ps0pts[:,2],ps0pts[:,3],2,full=True)
    psp1, resid1, _, _, _ = np.polyfit(ps1pts[:,1],ps1pts[:,3],3,full=True)

    resid_sq0 = np.dot(resid0,resid0)
    resid_sq1 = np.dot(resid1,resid1)

    poly0 = np.poly1d(psp0)
    poly1 = np.poly1d(psp1)

    bin0 = np.loadtxt('{0}/bins_1600_rN_16/coding_fcn/ps0_endpoints.txt'.format(data_direc))
    bin1 = np.loadtxt('{0}/bins_1600_rN_16/coding_fcn/ps1_endpoints.txt'.format(data_direc))

    polyfit0 = poly0(bin0); polyfit1 = poly1(bin1);

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,8))
    ax1.scatter(ps0pts[:,2],ps0pts[:,3],color='y')
    ax1.plot(bin0,polyfit0,linewidth=1,color="b")
    ax1.set_xlabel(r"$y$"); ax1.set_ylabel(r"$z$")
    ax1.set_title('(A)',loc='right', y=1.0, pad=-14)
    ax2.scatter(ps1pts[:,1],ps1pts[:,3],color='purple')
    ax2.plot(bin1,polyfit1,linewidth=1,color="b")
    ax2.set_xlabel(r"$x$"); ax2.set_ylabel(r"$z$")
    ax2.set_title('(B)',loc='right', y=1.0, pad=-14)

    for i in ['left','right','top','bottom']:
        ax1.spines[i].set_linewidth(2)
        ax2.spines[i].set_linewidth(2)

    ax1.tick_params('both', width=2)
    ax2.tick_params('both', width=2)
    sns.despine()

    ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))

    plt.tight_layout()
    plt.savefig("{0}/figure_5.eps".format(save_direc),dpi=300)
    plt.close()

def figure6(data_direc, bins, save_direc):
    '''
    Figure 6 - coding function r_n
    '''
    plt.rcParams.update({'font.size': 14})
    rn0 = np.loadtxt('{0}/bins_1600_rN_16/coding_fcn/ps0_rn_fcn.txt'.format(data_direc),usecols=16)
    rn1 = np.loadtxt('{0}/bins_1600_rN_16/coding_fcn/ps1_rn_fcn.txt'.format(data_direc),usecols=16)

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,8))
    ax1.plot(list(range(bins)),rn0,lw=1,color="b"); ax1.set_ylabel(r"$r_N(x)$");
    ax2.plot(list(range(bins)),rn1,lw=1,color="b"); ax2.set_xlabel("Bin"); ax2.set_ylabel(r"$r_N(x)$");
    for i in ['left','right','top','bottom']:
        ax1.spines[i].set_linewidth(2)
        ax2.spines[i].set_linewidth(2)
        ax1.tick_params('both', width=2)
        ax2.tick_params('both', width=2)
    sns.despine()
    ax1.set_title('(A)',loc='right', y=1.0, pad=-14)
    ax2.set_title('(B)',loc='right', y=1.0, pad=-14);
    plt.tight_layout()
    plt.savefig("{0}/figure_6.eps".format(save_direc),dpi=300)
    plt.close()


def figure7(data_direc,plot_percent,save_direc):
    '''
    Figure 7
    '''
    plt.rcParams.update({'font.size': 10})
    x,y,z = np.loadtxt(f'{data_direc}/bins_1600_rN_16/cupolets/C01011/bin_ps1_1109/cupolet_time_series.txt',unpack=True,usecols=(1,2,3))
    st = int(len(x)*(1-plot_percent)) # Only graph plot_percent of data
    x=x[st:]; y=y[st:]; z=z[st:];

    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot3D(x,y,z,linewidth=1,color="b")
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$'); ax.set_zlabel(r'$z$')
    ax.set_xlim([min(x),max(x)]); ax.set_ylim([min(y),max(y)]); ax.set_zlim([min(z),max(z)])
    plt.tight_layout()
    plt.savefig('{0}/figure_7.eps'.format(save_direc),dpi=300)
    plt.close()

def figure8(data_direc,save_direc,plot_percent):
    '''
    Figure 8 - several cupolets.
    '''
    plt.rcParams.update({'font.size': 10})
    _,x,y,z = np.loadtxt(f'{data_direc}/bins_1600_rN_16/cupolets/C11/bin_ps1_535/cupolet_time_series.txt',unpack=True)
    st = int(len(x)*(1-plot_percent))
    fig = plt.figure(figsize=(9,8)); # Create figure for plotting
    ax1 = fig.add_subplot(221,projection='3d'); # subplot 1 - 3D phase space
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$"); ax1.set_zlabel("$z$"); # axes label
    ax1.plot(x[st:],y[st:],z[st:],linewidth=1,color="b")
    ax1.set_title('(A)',loc='right', y=1.0, pad=-14)

    _,x,y,z = np.loadtxt(f'{data_direc}/bins_1600_rN_16/cupolets/C0110/bin_ps1_948/cupolet_time_series.txt',unpack=True)
    st = int(len(x)*(1-plot_percent))
    ax2 = fig.add_subplot(222,projection='3d'); # subplot 1 - 3D phase space
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$"); ax2.set_zlabel("$z$"); # axes label
    ax2.plot(x[st:],y[st:],z[st:],linewidth=1,color="b")
    ax2.set_title('(B)',loc='right', y=1.0, pad=-14)

    _,x,y,z = np.loadtxt(f'{data_direc}/bins_1600_rN_16/cupolets/C1010010/bin_ps1_685/cupolet_time_series.txt',unpack=True)
    st = int(len(x)*(1-plot_percent))
    ax3 = fig.add_subplot(223,projection='3d'); # subplot 1 - 3D phase space
    ax3.set_xlabel("$x$"); ax3.set_ylabel("$y$"); ax3.set_zlabel("$z$"); # axes label
    ax3.plot(x[st:],y[st:],z[st:],linewidth=1,color="b")
    ax3.set_title('(C)',loc='right', y=1.0, pad=-14)

    _,x,y,z = np.loadtxt(f'{data_direc}/bins_1600_rN_16/cupolets/C01010010/bin_ps1_542/cupolet_time_series.txt',unpack=True)
    st = int(len(x)*(1-plot_percent))
    ax4 = fig.add_subplot(224,projection='3d'); # subplot 1 - 3D phase space
    ax4.set_xlabel("$x$"); ax4.set_ylabel("$y$"); ax4.set_zlabel("$z$"); # axes label
    ax4.plot(x[st:],y[st:],z[st:],linewidth=1,color="b")
    ax4.set_title('(D)',loc='right', y=1.0, pad=-14)

    axes = [ax1,ax2,ax3,ax4]
    xlimits = [np.min([xax.get_xlim() for xax in axes]),np.max([xax.get_xlim() for xax in axes])]
    ylimits = [np.min([xax.get_ylim() for xax in axes]),np.max([xax.get_ylim() for xax in axes])]
    zlimits = [np.min([xax.get_zlim() for xax in axes]),np.max([xax.get_zlim() for xax in axes])]
    for i in range(len(axes)):
        axes[i].set_xlim(xlimits); axes[i].set_ylim(ylimits); axes[i].set_zlim(zlimits)
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(5))
        axes[i].zaxis.set_major_locator(plt.MaxNLocator(5))

    plt.tight_layout()
    plt.savefig(f"{save_direc}/figure_8.eps",dpi=300)
    plt.close()

def figure9(data_direc,save_direc,plot_percent,dt):
    '''
    Beginning of period may be slightly different than figure in paper.
    '''
    fig, axs = plt.subplots(2,2,figsize=(10,8))
    matplotlib.rcParams.update({'font.size': 12})
    c = 0
    axes = [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]
    labels = ["(A)","(B)","(C)","(D)"]
    cupolets = ["C11","C0110","C1010010","C01010010"]
    bins = ["bin_ps1_535","bin_ps1_948","bin_ps1_685","bin_ps1_542"]
    for i in range(0,4):
        t,x,y,z = np.loadtxt(f'{data_direc}/bins_1600_rN_16/cupolets/{cupolets[i]}/{bins[i]}/cupolet_time_series.txt',unpack=True)
        x = x[int(len(x)*(1-plot_percent)):]
        _,index_period,_ = sa.get_period(x,dt)
        t = t[-index_period:]; x = x[-index_period:]; y = y[-index_period:]; z = z[-index_period:];
        axes[i].plot(t,x,linewidth=1,color='b')
        axes[i].set_title(labels[i],loc='right', y=1.0, pad=-14);
    xlimits = [np.min([xax.get_xlim() for xax in axes]),np.max([xax.get_xlim() for xax in axes])]
    ylimits = [np.min([xax.get_ylim() for xax in axes]),np.max([xax.get_ylim() for xax in axes])]
    for i in range(len(axes)):
        axes[i].set_ylim(ylimits);
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(4))
        axes[i].set_xlim(right=xlimits[1])

    for i in ['left','right','top','bottom']:
        for k in axes:
            k.spines[i].set_linewidth(2)
            k.tick_params('both', width=2)

    sns.despine()
    fig.text(0.5, 0.04, '$t$', ha='center', va='center',fontsize=16)
    fig.text(0.06, 0.5, f'$x$', ha='center', va='center', rotation='vertical',fontsize=16)
    plt.savefig(f"{save_direc}/figure_9.eps",dpi=300)
    plt.close()

def figure11(data_direc,save_direc,dt):
    alphabet_string = string.ascii_uppercase
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(4,2,figsize=(9,16),subplot_kw=dict(projection='3d'))
    cupolets = ['C001','C10010','C11010011','C1101110','C11100010','C10000','C0111110','C01100011']
    bins = ["bin_ps1_566","bin_ps1_272","bin_ps1_1328","bin_ps1_539","bin_ps1_138","bin_ps1_115","bin_ps1_397","bin_ps1_74"]
    for i in range(len(cupolets)):
        t,x,y,z = np.loadtxt(f'{data_direc}/bins_1600_rN_16/cupolets/{cupolets[i]}/{bins[i]}/cupolet_time_series.txt',unpack=True)
        _,ip,_ = sa.get_period(x[int(len(x)*0.25):],dt);
        t = t[-ip:]; x = x[-ip:]; y = y[-ip:]; z = z[-ip:];
        pks,_ = find_peaks(x,height=1)
        print(cupolets[i],len(pks),len(pks)/(t[-1]-t[-ip]),ip,t[-1]-t[-ip])
        if i < len(cupolets)/2:
            ax[i,0].plot(x,y,z,linewidth=1,color="b")
            ax[i,0].set_xlabel("$x$"); ax[i,0].set_ylabel("$y$"); ax[i,0].set_zlabel("$z$"); # axes label
            ax[i,0].set_title('({0})'.format(alphabet_string[i]),loc='right', y=1.0, pad=-14);
        else:
            shift=int(len(cupolets)/2)
            ax[i-shift,1].plot(x,y,z,linewidth=1,color="b")
            ax[i-shift,1].set_xlabel("$x$"); ax[i-shift,1].set_ylabel("$y$"); ax[i-shift,1].set_zlabel("$z$"); # axes label
            ax[i-shift,1].set_title('({0})'.format(alphabet_string[i]),loc='right', y=1.0, pad=-14);
    axes = [ax[i,j] for j in range(2) for i in range(4)]
    xlimits = [np.min([xax.get_xlim() for xax in axes]),np.max([xax.get_xlim() for xax in axes])]
    ylimits = [np.min([xax.get_ylim() for xax in axes]),np.max([xax.get_ylim() for xax in axes])]
    zlimits = [np.min([xax.get_zlim() for xax in axes]),np.max([xax.get_zlim() for xax in axes])]

    for k in range(len(axes)):
        axes[k].set_ylim(ylimits);
        axes[k].set_xlim(xlimits)
        axes[k].set_zlim(zlimits)

    plt.tight_layout()
    fig.savefig(f"{save_direc}/figure_11.eps",dpi=300)
    plt.close()

def figure10(data_direc,save_direc):
    plt.rcParams.update({'font.size': 10})
    types = 2; bins = 1600; dt = 1/128;
    cupolet1 = np.loadtxt(f'{data_direc}/type_0/cupolet_time_series_bins_1600_N_16.txt')
    _,ip1,_ = sa.get_period(cupolet1[int(cupolet1.shape[0]*(0.25)):,1],dt)
    pks1,_ = find_peaks(cupolet1[-ip1:,1],height=1)
    t1 = cupolet1[-ip1:,0]; x1 = cupolet1[-ip1:,1];

    cupolet2 = np.loadtxt(f'{data_direc}/type_1/cupolet_time_series_bins_1600_N_16.txt')
    _,ip2,_ = sa.get_period(cupolet2[int(cupolet2.shape[0]*(0.25)):,1],dt)
    pks2,_ = find_peaks(cupolet2[-ip2:,1],height=1)
    t2 = cupolet2[-ip2:,0]; x2 = cupolet2[-ip2:,1];

    type = np.zeros((bins))
    for t in range(0,2):
        for line in open("{0}/type_{1}/corresponding_init_bins".format(data_direc,t),'r').readlines():
            type[eval(line.split('_')[1])] = t


    fig = plt.figure(figsize=(8,9),dpi=300)
    gs = fig.add_gridspec(5,2)
    ax1 = fig.add_subplot(gs[0:2,0],projection='3d')
    ax1.plot(cupolet1[-ip1:,1],cupolet1[-ip1:,2],cupolet1[-ip1:,3],linewidth=1,color='b')
    ax1.set_title("(A)",loc='right', y=1.0, pad=-14);
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$"); ax1.set_zlabel("$z$")

    ax2 = fig.add_subplot(gs[0:2,1],projection='3d')
    ax2.plot(cupolet2[-ip2:,1],cupolet2[-ip2:,2],cupolet2[-ip2:,3],linewidth=1,color='b')
    ax2.set_title("(B)",loc='right', y=1.0, pad=-14);
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$"); ax2.set_zlabel("$z$")

    ax3 = fig.add_subplot(gs[2:4,0])
    ax3.plot(t1,x1,linewidth=1,color='b')
    ax3.set_title("(C)",loc='right', y=1.0, pad=-14);
    ax3.set_xlabel("$t$"); ax3.set_ylabel("$x$");

    ax4 = fig.add_subplot(gs[2:4,1])
    ax4.plot(t2,x2,linewidth=1,color='b')
    ax4.set_title("(D)",loc='right', y=1.0, pad=-14);
    ax4.set_xlabel("$t$");

    ax5 = fig.add_subplot(gs[4,0:])
    for i in range(0,bins):
        if type[i] == 0:
            ax5.bar(i,0.25,color='b',width=1.0)
        if type[i] == 1:
            ax5.bar(i,0.25,color='r',width=1.0)
        if type[i] == 2:
            ax5.bar(i,0.25,color='g',width=1.0)
    ax5.set_title("(E)");
    ax5.set_yticks([])
    ax5.set_xlabel("Bin on $PS1$")


    for i in ['left','right','top','bottom']:
        for k in [ax1,ax2,ax3,ax4]:
            k.spines[i].set_linewidth(2)
            k.tick_params('both', width=2)

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    ax5.set_xlim([0,bins]); ax5.set_ylim([0,0.25]);

    axes = [ax1,ax2]
    xlimits = [np.min([xax.get_xlim() for xax in axes]),np.max([xax.get_xlim() for xax in axes])]
    ylimits = [np.min([xax.get_ylim() for xax in axes]),np.max([xax.get_ylim() for xax in axes])]
    zlimits = [np.min([xax.get_zlim() for xax in axes]),np.max([xax.get_zlim() for xax in axes])]

    for k in range(len(axes)):
        axes[k].set_ylim(ylimits);
        axes[k].set_zlim(zlimits)
        axes[k].set_xlim(xlimits)

    axes = [ax3,ax4]
    ylimits = [np.min([xax.get_ylim() for xax in axes]),np.max([xax.get_ylim() for xax in axes])]

    for k in range(len(axes)):
        axes[k].set_ylim(ylimits);

    plt.tight_layout()
    plt.savefig(f"{save_direc}/figure_homolog.eps")
    plt.close()


def epstopdf(figures,save_direc):
    for f in figures:
        os.system(f'epstopdf {save_direc}/figure_{f}.eps {save_direc}/figure_{f}.pdf')

parser = parser = argparse.ArgumentParser(description='Generates figures based on input.')

parser.add_argument('-sd',nargs='?',default='paper_figures',type=str,help='Path to directory to store generated figures. Default: paper_figures')
parser.add_argument('-d',nargs='?',default='paper_data',type=str,help='Path to directory containing data. Default: paper_data')

parser.add_argument('-f',nargs='+',default=[1,2,3,4,5,6,7,8,9,10,11],type=int,help='Array of figure numbers to generate. Default: 1 2 3 4 5 6 7 8 9 10 11')
parser.add_argument('-b',nargs='?',default=1600,type=int,help="Number of bins in data. Default: 1600")
parser.add_argument('-dt',nargs='?',default=1/128.0,type=float,help="dt used in data. Default: 1/128")
parser.add_argument('-pp',nargs='?',default=0.75,type=float,help="Proportion of data to plot. Default: 0.75")
parser.add_argument('-pps',nargs='?',default=0.25,type=float,help="Proportion of data to plot, for the full chaotic attractor. Default: 0.25")

args = parser.parse_args()

data_direc = args.d;
save_direc = args.sd;
bins = args.b;
dt = args.dt;
plot_percent = args.pp;
plot_percent_short = args.pps;

for fig in args.f:
    if fig == 1:
        figure1(data_direc,save_direc)
    if fig == 2:
        figure2(data_direc, dt,plot_percent,save_direc)
    if fig == 3:
        figure3(data_direc, dt,plot_percent_short,save_direc)
    if fig == 4:
        figure4(data_direc,plot_percent,save_direc)
    if fig == 5:
        figure5(data_direc,save_direc)
    if fig == 6:
        figure6(data_direc,bins,save_direc)
    if fig == 7:
        figure7(data_direc, plot_percent,save_direc)
    if fig == 8:
        figure8(data_direc,save_direc, 0.75)
    if fig == 9:
        figure9(data_direc,save_direc,plot_percent,dt)
    if fig == 10:
        figure10(f"{data_direc}/bins_1600_rN_16/cupolets/c11/unique",save_direc)
    if fig == 11:
        figure11(data_direc,save_direc,dt)

epstopdf(args.f,save_direc)
