# generate_graphics.py
# Creates figures for HR Single Cupolet Paper.
# Author: John E. Parker
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import argparse

from helpers import *
import mutual_stabilization as ms
import loop_mutual_stabilization as loop
import chain_mutual_stabilization as chain

plt.rcParams.update({"font.size": 18})


def generate_figure_3(data_direc, plot_percent, save_direc):
    """
    Figure 3 - control planes on 3D HR attractor with Henon's trick points.
    """
    plt.rcParams.update({"font.size": 14})
    t, x, y, z = np.loadtxt(
        f"{data_direc}/hr_system/hr_time_series.txt", unpack=True
    )  # Read in all HR system chaotic data
    st = int(len(x) * (1 - plot_percent))  # Only graph plot_percent of data
    t = t[st:]
    x = x[st:]
    y = y[st:]
    z = z[st:]

    pts = np.loadtxt(f"{data_direc}/control_planes/ps_all_pts.txt")  # Poincare points
    verts0 = np.loadtxt(
        f"{data_direc}/control_planes/ps0_vertices.txt"
    )  # vertices of PS0
    verts1 = np.loadtxt(
        f"{data_direc}/control_planes/ps1_vertices.txt"
    )  # vertices of PS1

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot3D(x, y, z, linewidth=0.5, color="b")

    for p in pts:
        if p[4] == 1:
            ax.scatter(p[1], p[2], p[3], color="purple")
        else:
            ax.scatter(p[1], p[2], p[3], color="y")

    ax.add_line(
        mplot3d.art3d.Line3D(
            [verts1[0, 0], verts1[1, 0]],
            [verts1[0, 1], verts1[0, 1]],
            [verts1[0, 2], verts1[0, 2]],
            color="k",
        )
    )
    ax.add_line(
        mplot3d.art3d.Line3D(
            [verts1[0, 0], verts1[1, 0]],
            [verts1[0, 1], verts1[0, 1]],
            [verts1[1, 2], verts1[1, 2]],
            color="k",
        )
    )
    ax.add_line(
        mplot3d.art3d.Line3D(
            [verts1[0, 0], verts1[0, 0]],
            [verts1[0, 1], verts1[0, 1]],
            [verts1[0, 2], verts1[1, 2]],
            color="k",
        )
    )
    ax.add_line(
        mplot3d.art3d.Line3D(
            [verts1[1, 0], verts1[1, 0]],
            [verts1[0, 1], verts1[0, 1]],
            [verts1[0, 2], verts1[1, 2]],
            color="k",
        )
    )

    ax.add_line(
        mplot3d.art3d.Line3D(
            [verts0[0, 0], verts0[1, 0]],
            [verts0[0, 1], verts0[1, 1]],
            [verts0[0, 2], verts0[0, 2]],
            color="k",
        )
    )
    ax.add_line(
        mplot3d.art3d.Line3D(
            [verts0[0, 0], verts0[1, 0]],
            [verts0[0, 1], verts0[1, 1]],
            [verts0[1, 2], verts0[1, 2]],
            color="k",
        )
    )
    ax.add_line(
        mplot3d.art3d.Line3D(
            [verts0[0, 0], verts0[0, 0]],
            [verts0[0, 1], verts0[0, 1]],
            [verts0[0, 2], verts0[1, 2]],
            color="k",
        )
    )
    ax.add_line(
        mplot3d.art3d.Line3D(
            [verts0[1, 0], verts0[1, 0]],
            [verts0[1, 1], verts0[1, 1]],
            [verts0[0, 2], verts0[1, 2]],
            color="k",
        )
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])
    ax.set_zlim([min(z), max(z)])
    plt.tight_layout()
    plt.savefig(f"{save_direc}/figure_3.eps", dpi=300)
    plt.close()


def epstopdf(figures, save_direc):
    for f in figures:
        run_cmd(f"epstopdf {save_direc}/figure_{f}.eps {save_direc}/figure_{f}.pdf")
        run_cmd(f"open {save_direc}/figure_{f}.pdf")


parser = parser = argparse.ArgumentParser(
    description="Generates figures based on input."
)

parser.add_argument(
    "-sd",
    nargs="?",
    default="paper_figures",
    type=str,
    help="Path to directory to store generated figures. Default: paper_figures",
)
parser.add_argument(
    "-d",
    nargs="?",
    default="paper_data",
    type=str,
    help="Path to directory containing data. Default: paper_data",
)

parser.add_argument(
    "-f",
    nargs="+",
    default=[3, 4, 5, 6, 7],
    type=int,
    help="Array of figure numbers to generate. Default: 3 4 5 6 7",
)
parser.add_argument(
    "-b",
    nargs="?",
    default=1600,
    type=int,
    help="Number of bins in data. Default: 1600",
)
parser.add_argument(
    "-dt",
    nargs="?",
    default=1 / 128.0,
    type=float,
    help="dt used in data. Default: 1/128",
)
parser.add_argument(
    "-pp",
    nargs="?",
    default=0.75,
    type=float,
    help="Proportion of data to plot. Default: 0.75",
)

args = parser.parse_args()

data_direc = args.d
save_direc = args.sd
bins = args.b
dt = args.dt
plot_percent = args.pp

for fig in args.f:
    if fig == 3:
        generate_figure_3(data_direc, plot_percent, save_direc)
    if fig == 4:
        ms.generate_figure_4(
            data_direc,
            save_direc,
            dt=dt,
            t0=0,
            tf=10000,
            control1=[0, 0, 1],
            control2=[0, 1],
            ifm=5,
            ifn=3,
        )
    if fig == 5:
        ms.generate_figure_5(
            data_direc,
            save_direc,
            dt=dt,
            t0=0,
            tf=10000,
            control1=[0, 0, 1],
            control2=[0, 1],
            ifm=4,
            ifn=4,
        )
    if fig == 6:
        chain.generate_figure_6(
            data_direc,
            save_direc,
            dt=dt,
            tf=10000,
            control1=[1, 1],
            ifm=5,
            ifn=3,
            neurons=4,
        )
    if fig == 7:
        loop.generate_figure_7(
            data_direc,
            save_direc,
            dt=dt,
            tf=10000,
            control1=[1, 1],
            ifm=5,
            ifn=3,
            neurons=4,
        )


epstopdf(args.f, save_direc)
