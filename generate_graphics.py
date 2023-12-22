# generate_graphics.py
# Creates figures for HR Single Cupolet Paper.
# Author: John E. Parker
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import argparse

# import user modules
from helpers import *
import mutual_stabilization as ms
import loop_mutual_stabilization as loop
import chain_mutual_stabilization as chain


plt.rcParams.update({"font.size": 18})


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
    default=[4, 5, 6, 7],
    type=int,
    help="Array of figure numbers to generate. Default: 4 5 6 7",
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

# Loop to generate all provided figures from -f flag.
for fig in args.f:
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

# Generate PDF figure from EPS.
epstopdf(args.f, save_direc)
