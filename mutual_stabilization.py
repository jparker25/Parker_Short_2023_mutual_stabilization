# Script to generate mutual stabilization between two HR neurons.
# Author: John Parker

# Import Python modules
import numpy as np
from matplotlib import pyplot as plt
import pickle
import bisect

# import user modules
import rk4
import control_planes as gc
import keep_data as kd
import cupolet_search as cupg
from helpers import *


def integrateAndFire(vs, subbits, threshold):
    """
    Reads in a visitation sequence VS, integer subbits, and integer thrshold.
    Performs integrateAndFire function on the visitation sequence where
    if the subbits sum to the threshold a 1 is returned otherwise a 0 is returned.
    """
    if sum(vs) >= threshold:
        return 1
    else:
        return 0


def mutual_stabilization(direc, dt, t0, tf, control1, control2, ifm, ifn):
    """
    Performs mutual stabilization between two neurons.
    """
    bin_rn_direc = f"{direc}/bins_1600_rN_16/"
    neuron = pickle.load(open(f"{direc}/neuron.obj", "rb"))
    N = int(tf / dt)
    trans = np.zeros((N, 7))
    for neu in range(0, 2):
        trans[:, 3 * neu + 1 : 3 * neu + 4] = rk4.rk4_N(
            np.random.rand(3), t0, dt, neuron.hr_dynamics, N
        )[:, 1:]

    ps0inits = np.loadtxt(f"{bin_rn_direc}/coding_fcn/ps0_inits.txt")
    ps1inits = np.loadtxt(f"{bin_rn_direc}/coding_fcn/ps1_inits.txt")

    ps0endpts = np.loadtxt(f"{bin_rn_direc}/coding_fcn/ps0_endpoints.txt")
    ps1endpts = np.loadtxt(f"{bin_rn_direc}/coding_fcn/ps1_endpoints.txt")

    ps0x, ps0y, ps0z = np.loadtxt(
        f"{direc}/control_planes/ps0_vertices.txt", unpack=True
    )
    ps1x, ps1y, ps1z = np.loadtxt(
        f"{direc}/control_planes/ps1_vertices.txt", unpack=True
    )

    ctrl1, _ = np.loadtxt(
        f"{bin_rn_direc}/macrocontrol/ps1_macrocontrol.txt", unpack=True
    )
    ctrl0, _ = np.loadtxt(
        f"{bin_rn_direc}/macrocontrol/ps0_macrocontrol.txt", unpack=True
    )

    sol = np.zeros((4 * N, 7))
    sol[0, 0] = 0
    sol[0, 1:] = trans[-1, 1:]
    ctrl1i = 0
    ctrl2i = 0
    vs = [[], []]
    vsms = [[], []]
    control = [[], []]

    # The following for loop creates the two cupolets based on control1 and control2
    for i in range(0, sol.shape[0] - 1):
        for neu in range(0, 2):
            sol[i + 1, 3 * neu + 1 : 3 * neu + 4] = rk4.rk4(
                sol[i, 3 * neu + 1 : 3 * neu + 4], dt, neuron.hr_dynamics
            )
            sol[i + 1, 0] = sol[i, 0] + dt

        if gc.crossed(
            sol[i, 1],
            sol[i + 1, 1],
            sol[i, 2],
            sol[i + 1, 2],
            1,
            np.array([[ps1x[0], ps1y[0]], [ps1x[1], ps1y[1]]]),
        ):
            tp, xp, zp, yp = rk4.rk4_henon(
                np.array([sol[i, 0], sol[i, 1], sol[i, 3]]),
                sol[i, 2],
                -(sol[i, 2] - ps1y[0]),
                neuron.hr_dy_dynamics,
            )
            ii = bisect.bisect(ps1endpts, xp) - 1
            newval = None
            if len(vs[1]) >= ifm and i > 2 * N:
                newval = (
                    ps1inits[ii]
                    if integrateAndFire(vs[1][-ifm:], ifm, ifn) == 0
                    else ps1inits[int(ctrl1[ii])]
                )
                control[0].append(integrateAndFire(vs[1][-ifm:], ifm, ifn))
                vsms[0].append(1)
            else:
                newval = (
                    ps1inits[ii] if control1[ctrl1i] == 0 else ps1inits[int(ctrl1[ii])]
                )
            newdt = dt - (tp - sol[i, 0])
            next = rk4.rk4(newval, newdt, neuron.hr_dynamics)
            sol[i + 1, 1:4] = next
            sol[i + 1, 0] = sol[i, 0] + dt
            ctrl1i += 1
            vs[0].append(1)

        elif gc.crossed(
            sol[i, 1],
            sol[i + 1, 1],
            sol[i, 2],
            sol[i + 1, 2],
            0,
            np.array([[ps0x[0], ps0y[0]], [ps0x[1], ps0y[1]]]),
        ):
            tp, yp, zp, xp = rk4.rk4_henon(
                np.array([sol[i, 0], sol[i, 2], sol[i, 3]]),
                sol[i, 1],
                -(sol[i, 1] - ps0x[0]),
                neuron.hr_dx_dynamics,
            )
            ii = bisect.bisect(ps0endpts, yp) - 1
            newval = None
            if len(vs[1]) >= ifm and i > 2 * N:
                newval = (
                    ps0inits[ii]
                    if integrateAndFire(vs[1][-ifm:], ifm, ifn) == 0
                    else ps0inits[int(ctrl0[ii])]
                )
                control[0].append(integrateAndFire(vs[1][-ifm:], ifm, ifn))
                vsms[0].append(0)
            else:
                newval = (
                    ps0inits[ii] if control1[ctrl1i] == 0 else ps0inits[int(ctrl0[ii])]
                )
            newdt = dt - (tp - sol[i, 0])
            next = rk4.rk4(newval, newdt, neuron.hr_dynamics)
            sol[i + 1, 1:4] = next
            sol[i + 1, 0] = sol[i, 0] + dt
            ctrl1i += 1
            vs[0].append(0)

        if ctrl1i == len(control1):
            ctrl1i = 0

        if gc.crossed(
            sol[i, 4],
            sol[i + 1, 4],
            sol[i, 5],
            sol[i + 1, 5],
            1,
            np.array([[ps1x[0], ps1y[0]], [ps1x[1], ps1y[1]]]),
        ):
            tp, xp, zp, yp = rk4.rk4_henon(
                np.array([sol[i, 0], sol[i, 4], sol[i, 6]]),
                sol[i, 5],
                -(sol[i, 5] - ps1y[0]),
                neuron.hr_dy_dynamics,
            )
            ii = bisect.bisect(ps1endpts, xp) - 1
            newval = None
            if len(vs[0]) >= ifm and i > 2 * N:
                newval = (
                    ps1inits[ii]
                    if integrateAndFire(vs[0][-ifm:], ifm, ifn) == 0
                    else ps1inits[int(ctrl1[ii])]
                )
                control[1].append(integrateAndFire(vs[0][-ifm:], ifm, ifn))
                vsms[1].append(1)
            else:
                newval = (
                    ps1inits[ii] if control2[ctrl2i] == 0 else ps1inits[int(ctrl1[ii])]
                )
            newdt = dt - (tp - sol[i, 0])
            next = rk4.rk4(newval, newdt, neuron.hr_dynamics)
            sol[i + 1, 4:] = next
            sol[i + 1, 0] = sol[i, 0] + dt
            ctrl2i += 1
            vs[1].append(1)

        elif gc.crossed(
            sol[i, 4],
            sol[i + 1, 4],
            sol[i, 5],
            sol[i + 1, 5],
            0,
            np.array([[ps0x[0], ps0y[0]], [ps0x[1], ps0y[1]]]),
        ):
            tp, yp, zp, xp = rk4.rk4_henon(
                np.array([sol[i, 0], sol[i, 5], sol[i, 6]]),
                sol[i, 4],
                -(sol[i, 4] - ps0x[0]),
                neuron.hr_dx_dynamics,
            )
            ii = bisect.bisect(ps0endpts, yp) - 1
            newval = None
            if len(vs[0]) >= ifm and i > 2 * N:
                newval = (
                    ps0inits[ii]
                    if integrateAndFire(vs[0][-ifm:], ifm, ifn) == 0
                    else ps0inits[int(ctrl0[ii])]
                )
                control[1].append(integrateAndFire(vs[0][-ifm:], ifm, ifn))
                vsms[1].append(0)
            else:
                newval = (
                    ps0inits[ii] if control2[ctrl2i] == 0 else ps0inits[int(ctrl0[ii])]
                )
            newdt = dt - (tp - sol[i, 0])
            next = rk4.rk4(newval, newdt, neuron.hr_dynamics)
            sol[i + 1, 4:] = next
            sol[i + 1, 0] = sol[i, 0] + dt
            ctrl2i += 1
            vs[1].append(0)

        if ctrl2i == len(control2):
            ctrl2i = 0

    return sol, vs, vsms, control


def plot_mutual_stabilization(sol, save_direc):
    """
    Plots mutual stabilization result.
    """
    N = int(sol.shape[0] / 4)
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        subplot_kw=dict(projection="3d"),
        dpi=300,
        tight_layout=True,
    )
    plt.setp(
        axs,
        xlim=(np.min(sol[:, 1]), np.max(sol[:, 1])),
        ylim=(np.min(sol[:, 2]), np.max(sol[:, 2])),
        zlim=(np.min(sol[:, 3]), np.max(sol[:, 3])),
    )
    for i in range(2):
        for j in range(2):
            axs[i, j].plot(
                sol[(2 * i + 1) * N : (2 * i + 2) * N, 3 * j + 1],
                sol[(2 * i + 1) * N : (2 * i + 2) * N, 3 * j + 2],
                sol[(2 * i + 1) * N : (2 * i + 2) * N, 3 * j + 3],
                linewidth=1.5,
                color="blue",
            )
            axs[i, j].set_xlabel("$x$")
            axs[i, j].set_ylabel("$y$")
            axs[i, j].set_zlabel("$z$")
            axs[i, j].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs[i, j].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs[i, j].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs[i, j].xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            axs[i, j].yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            axs[i, j].zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    axs[0, 0].set_title("(A)", loc="left", y=1, pad=-14)
    axs[0, 1].set_title("(B)", loc="left", y=1, pad=-14)
    axs[1, 0].set_title("(C)", loc="left", y=1, pad=-14)
    axs[1, 1].set_title("(D)", loc="left", y=1, pad=-14)
    makeNice3D([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]])
    plt.savefig(f"{save_direc}/mutual_stabilization_3d.eps")
    plt.close()


def save_data(sol, vs, vsms, control1, control2, control, save_direc):
    np.savetxt(f"{save_direc}/sol.txt", sol, delimiter="\t", newline="\n")
    np.savetxt(
        f"{save_direc}/control1.txt", np.asarray(control1), delimiter="\t", newline="\n"
    )
    np.savetxt(
        f"{save_direc}/control2.txt", np.asarray(control2), delimiter="\t", newline="\n"
    )
    for i in range(1, 3):
        np.savetxt(
            f"{save_direc}/vs{i}.txt",
            np.asarray(vs[i - 1]),
            delimiter="\t",
            newline="\n",
        )
        np.savetxt(
            f"{save_direc}/vsms{i}.txt",
            np.asarray(vsms[i - 1]),
            delimiter="\t",
            newline="\n",
        )
        np.savetxt(
            f"{save_direc}/controlms{i}.txt",
            np.asarray(control[i - 1]),
            delimiter="\t",
            newline="\n",
        )


def generate_figure_4(
    direc,
    save_direc,
    dt=1 / 128,
    t0=0,
    tf=10000,
    control1=[0, 0, 1],
    control2=[0, 1],
    ifm=5,
    ifn=3,
):
    """
    Simulates and plots mutual stabilization result.
    """
    np.random.seed(24)
    save_direc = f"{save_direc}/c1_{cupg.ctrl_to_string(control1)}_c2_{cupg.ctrl_to_string(control2)}_q_{ifm}_k_{ifn}_none"
    kd.check_direc(save_direc)

    sol, vs, vsms, control = mutual_stabilization(
        direc, dt, t0, tf, control1, control2, ifm, ifn
    )
    save_data(sol, vs, vsms, control1, control2, control, save_direc)

    sol = np.loadtxt(f"{save_direc}/sol.txt")
    plot_mutual_stabilization(sol, save_direc)
    run_cmd(f"cp {save_direc}/mutual_stabilization_3d.eps paper_figures/figure_4.eps")


def generate_figure_5(
    direc,
    save_direc,
    dt=1 / 128,
    t0=0,
    tf=10000,
    control1=[0, 0, 1],
    control2=[0, 1],
    ifm=4,
    ifn=4,
):
    np.random.seed(24)
    save_direc = f"{save_direc}/c1_{cupg.ctrl_to_string(control1)}_c2_{cupg.ctrl_to_string(control2)}_q_{ifm}_k_{ifn}_none"
    kd.check_direc(save_direc)

    sol, vs, vsms, control = mutual_stabilization(
        direc, dt, t0, tf, control1, control2, ifm, ifn
    )
    save_data(sol, vs, vsms, control1, control2, control, save_direc)

    sol = np.loadtxt(f"{save_direc}/sol.txt")
    plot_mutual_stabilization(sol, save_direc)
    run_cmd(f"cp {save_direc}/mutual_stabilization_3d.eps paper_figures/figure_5.eps")
