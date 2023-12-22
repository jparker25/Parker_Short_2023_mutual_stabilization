# Helper script with several auxiliary functions.
# Author: John Parker

# Import Python modules
import os, string


def check_direc(str):
    run_cmd(f"mkdir -p {str}")


def run_cmd(str):
    print(str)
    os.system(str)


def makeNice3D(axes):
    if type(axes) == list:
        for ax in axes:
            for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
                axis.line.set_linewidth(3)
                for line in axis.get_ticklines():
                    line.set_visible(False)
    else:
        for axis in [axes.w_xaxis, axes.w_yaxis, axes.w_zaxis]:
            axis.line.set_linewidth(3)


def makeNice(axes):
    if type(axes) == list:
        for ax in axes:
            for i in ["left", "right", "top", "bottom"]:
                if i != "left" and i != "bottom":
                    ax.spines[i].set_visible(False)
                    ax.tick_params("both", width=0, labelsize=8)
                else:
                    ax.spines[i].set_linewidth(3)
                    ax.tick_params("both", width=0, labelsize=8)
    else:
        for i in ["left", "right", "top", "bottom"]:
            if i != "left" and i != "bottom":
                axes.spines[i].set_visible(False)
                axes.tick_params("both", width=0, labelsize=8)
            else:
                axes.spines[i].set_linewidth(3)
                axes.tick_params("both", width=0, labelsize=8)


def add_fig_labels(axes):
    labels = string.ascii_uppercase
    for i in range(len(axes)):
        axes[i].text(
            0.03,
            0.98,
            labels[i],
            fontsize=16,
            transform=axes[i].transAxes,
            fontweight="bold",
            color="gray",
        )
