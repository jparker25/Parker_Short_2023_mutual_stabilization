import numpy as np
from matplotlib import pyplot as plt
import pickle
import bisect
from scipy.signal import find_peaks

import rk4
import control_planes as gc
import keep_data as kd
import cupolet_search as cupg
import signal_analysis as sa
from helpers import *


def integrateAndFire(vs, subbits, threshold):
    if sum(vs) >= threshold:
        return 1
    else:
        return 0


def unidirectional_loop(direc, dt, tf, neurons, control1, ifm, ifn):
    bin_rn_direc = f"{direc}/bins_1600_rN_16/"
    neuron = pickle.load(open(f"{direc}/neuron.obj", "rb"))

    t0 = 0
    x0 = np.random.rand(neurons * 3)
    N = int(tf / dt)
    trans = np.zeros((N, neurons * 3 + 1))
    for neu in range(0, neurons):
        trans[:, 3 * neu + 1 : 3 * neu + 4] = rk4.rk4_N(
            x0[3 * neu : 3 * neu + 3], t0, dt, neuron.hr_dynamics, N
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

    sol = np.zeros((2 * (neurons + 1) * N, neurons * 3 + 1))
    sol[0, 0] = 0
    sol[0, 1:] = trans[-1, 1:]
    ctrl1i = 0
    vs = []
    vsms = []
    control = []
    for n in range(0, neurons):
        vs.append([])
        vsms.append([])
        control.append([])
    control[0] = control1

    # The following for loop creates the two cupolets based on control1 and control2
    for i in range(0, sol.shape[0] - 1):
        for neu in range(0, neurons):
            sol[i + 1, 3 * neu + 1 : 3 * neu + 4] = rk4.rk4(
                sol[i, 3 * neu + 1 : 3 * neu + 4], dt, neuron.hr_dynamics
            )
            sol[i + 1, 0] = sol[i, 0] + dt

        if (
            gc.crossed(
                sol[i, 1],
                sol[i + 1, 1],
                sol[i, 2],
                sol[i + 1, 2],
                1,
                np.array([[ps1x[0], ps1y[0]], [ps1x[1], ps1y[1]]]),
            )
            and i < 2 * neurons * N
        ):
            tp, xp, zp, yp = rk4.rk4_henon(
                np.array([sol[i, 0], sol[i, 1], sol[i, 3]]),
                sol[i, 2],
                -(sol[i, 2] - ps1y[0]),
                neuron.hr_dy_dynamics,
            )
            ii = bisect.bisect(ps1endpts, xp) - 1
            if i > 2 * N:
                vsms[0].append(1)
            newval = (
                ps1inits[ii] if control[0][ctrl1i] == 0 else ps1inits[int(ctrl1[ii])]
            )
            newdt = dt - (tp - sol[i, 0])
            next = rk4.rk4(newval, newdt, neuron.hr_dynamics)
            sol[i + 1, 1:4] = next
            sol[i + 1, 0] = sol[i, 0] + dt
            ctrl1i += 1
            vs[0].append(1)

        elif (
            gc.crossed(
                sol[i, 1],
                sol[i + 1, 1],
                sol[i, 2],
                sol[i + 1, 2],
                0,
                np.array([[ps0x[0], ps0y[0]], [ps0x[1], ps0y[1]]]),
            )
            and i < 2 * neurons * N
        ):
            tp, yp, zp, xp = rk4.rk4_henon(
                np.array([sol[i, 0], sol[i, 2], sol[i, 3]]),
                sol[i, 1],
                -(sol[i, 1] - ps0x[0]),
                neuron.hr_dx_dynamics,
            )
            ii = bisect.bisect(ps0endpts, yp) - 1
            if i > 2 * N:
                vsms[0].append(0)
            newval = (
                ps0inits[ii] if control[0][ctrl1i] == 0 else ps0inits[int(ctrl0[ii])]
            )
            newdt = dt - (tp - sol[i, 0])
            next = rk4.rk4(newval, newdt, neuron.hr_dynamics)
            sol[i + 1, 1:4] = next
            sol[i + 1, 0] = sol[i, 0] + dt
            ctrl1i += 1
            vs[0].append(0)

        elif gc.crossed(
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
            newval = (
                ps1inits[ii]
                if integrateAndFire(vsms[-1][-ifm:], ifm, ifn) == 0
                else ps1inits[int(ctrl1[ii])]
            )
            control[0].append(integrateAndFire(vsms[-1][-ifm:], ifm, ifn))
            newdt = dt - (tp - sol[i, 0])
            next = rk4.rk4(newval, newdt, neuron.hr_dynamics)
            sol[i + 1, 1:4] = next
            sol[i + 1, 0] = sol[i, 0] + dt
            vsms[0].append(1)
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
            newval = (
                ps0inits[ii]
                if integrateAndFire(vsms[-1][-ifm:], ifm, ifn) == 0
                else ps0inits[int(ctrl0[ii])]
            )
            control[0].append(integrateAndFire(vsms[-1][-ifm:], ifm, ifn))
            newdt = dt - (tp - sol[i, 0])
            next = rk4.rk4(newval, newdt, neuron.hr_dynamics)
            sol[i + 1, 1:4] = next
            sol[i + 1, 0] = sol[i, 0] + dt
            vsms[0].append(0)
            vs[0].append(0)

        if ctrl1i == len(control[0]):
            ctrl1i = 0

        for neural_i in range(1, neurons):
            if gc.crossed(
                sol[i, 3 * neural_i + 1],
                sol[i + 1, 3 * neural_i + 1],
                sol[i, 3 * neural_i + 2],
                sol[i + 1, 3 * neural_i + 2],
                1,
                np.array([[ps1x[0], ps1y[0]], [ps1x[1], ps1y[1]]]),
            ):
                tp, xp, zp, yp = rk4.rk4_henon(
                    np.array(
                        [sol[i, 0], sol[i, 3 * neural_i + 1], sol[i, 3 * neural_i + 3]]
                    ),
                    sol[i, 3 * neural_i + 2],
                    -(sol[i, 3 * neural_i + 2] - ps1y[0]),
                    neuron.hr_dy_dynamics,
                )
                ii = bisect.bisect(ps1endpts, xp) - 1
                if len(vsms[neural_i - 1]) >= ifm and i > 2 * neural_i * N:
                    newval = (
                        ps1inits[ii]
                        if integrateAndFire(vsms[neural_i - 1][-ifm:], ifm, ifn) == 0
                        else ps1inits[int(ctrl1[ii])]
                    )
                    control[neural_i].append(
                        integrateAndFire(vsms[neural_i - 1][-ifm:], ifm, ifn)
                    )
                    newdt = dt - (tp - sol[i, 0])
                    next = rk4.rk4(newval, newdt, neuron.hr_dynamics)
                    sol[i + 1, 3 * neural_i + 1 : 3 * neural_i + 4] = next
                    sol[i + 1, 0] = sol[i, 0] + dt
                    vsms[neural_i].append(1)
                    vs[neural_i].append(1)

            elif gc.crossed(
                sol[i, 3 * neural_i + 1],
                sol[i + 1, 3 * neural_i + 1],
                sol[i, 3 * neural_i + 2],
                sol[i + 1, 3 * neural_i + 2],
                0,
                np.array([[ps0x[0], ps0y[0]], [ps0x[1], ps0y[1]]]),
            ):
                tp, yp, zp, xp = rk4.rk4_henon(
                    np.array(
                        [sol[i, 0], sol[i, 3 * neural_i + 2], sol[i, 3 * neural_i + 3]]
                    ),
                    sol[i, 3 * neural_i + 1],
                    -(sol[i, 3 * neural_i + 1] - ps0x[0]),
                    neuron.hr_dx_dynamics,
                )
                ii = bisect.bisect(ps0endpts, yp) - 1
                if len(vsms[neural_i - 1]) >= ifm and i > 2 * neural_i * N:
                    newval = (
                        ps0inits[ii]
                        if integrateAndFire(vsms[neural_i - 1][-ifm:], ifm, ifn) == 0
                        else ps0inits[int(ctrl0[ii])]
                    )
                    control[neural_i].append(
                        integrateAndFire(vsms[neural_i - 1][-ifm:], ifm, ifn)
                    )
                    newdt = dt - (tp - sol[i, 0])
                    next = rk4.rk4(newval, newdt, neuron.hr_dynamics)
                    sol[i + 1, 3 * neural_i + 1 : 3 * neural_i + 4] = next
                    sol[i + 1, 0] = sol[i, 0] + dt
                    vsms[neural_i].append(0)
                    vs[neural_i].append(0)

    return sol, vs, vsms, control


def plot_loop_mutual_stabilization(sol, neurons, save_direc):
    N = int(sol.shape[0] / (2 * (neurons + 1)))
    fig, axs = plt.subplots(
        neurons + 1,
        neurons,
        figsize=(16, 12),
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
    for i in range(neurons + 1):
        for j in range(neurons):
            axs[i, j].plot(
                sol[(2 * i + 1) * N : (2 * i + 2) * N, 3 * j + 1],
                sol[(2 * i + 1) * N : (2 * i + 2) * N, 3 * j + 2],
                sol[(2 * i + 1) * N : (2 * i + 2) * N, 3 * j + 3],
                linewidth=0.5,
                color="blue",
            )
            # axs[i,j].set_xlabel('$x$'); axs[i,j].set_ylabel('$y$'); axs[i,j].set_zlabel('$z$');
            axs[i, j].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs[i, j].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs[i, j].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs[i, j].xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            axs[i, j].yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            axs[i, j].zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    axes = []
    for i in range(neurons + 1):
        for j in range(neurons):
            axes.append(axs[i, j])
    makeNice3D(axes)
    for i in range(len(axes)):
        axes[i].axis("off")
    plt.savefig(f"{save_direc}/neural_states_3d.eps")
    plt.close()
    run_cmd(f"open {save_direc}/neural_states_3d.eps")


def plot_raster(sol, neurons, save_direc, dt):
    N = int(sol.shape[0] / (2 * neurons + 2))
    _, index_period, _ = sa.get_period(sol[N : 2 * N, 1], dt)
    fig, axs = plt.subplots(neurons + 1, 1, figsize=(8, 10))
    for i in range(neurons + 1):
        ed = int((2 * i + 2) * N)
        st = ed - int(N / 3)
        t = sol[st:ed, 0]
        pks1, _ = find_peaks(sol[st:ed, 1], height=1)
        pks2, _ = find_peaks(sol[st:ed, 4], height=1)
        pks3, _ = find_peaks(sol[st:ed, 7], height=1)
        pks4, _ = find_peaks(sol[st:ed, 10], height=1)
        axs[i].eventplot(
            [t[pks4], t[pks3], t[pks2], t[pks1]], linewidth=1, color="blue"
        )
        axs[i].set_yticks(list(range(neurons)))
        axs[i].set_yticklabels(["Neuron 4", "Neuron 3", "Neuron 2", "Neuron 1"])
    plt.savefig(f"{save_direc}/raster.eps", dpi=300)
    plt.close()


def plot_raster_scatter(sol, neurons, save_direc, dt):
    N = int(sol.shape[0] / (2 * neurons + 2))
    _, axs = plt.subplots(neurons + 1, 1, figsize=(8, 10), dpi=300, tight_layout=300)
    for i in range(neurons + 1):
        ed = int((2 * i + 2) * N)
        st = ed - int(N / 3)
        t = sol[st:ed, 0]
        pks1, _ = find_peaks(sol[st:ed, 1], height=1)
        pks2, _ = find_peaks(sol[st:ed, 4], height=1)
        pks3, _ = find_peaks(sol[st:ed, 7], height=1)
        pks4, _ = find_peaks(sol[st:ed, 10], height=1)
        axs[i].eventplot(
            [t[pks4], t[pks3], t[pks2], t[pks1]], linewidth=1, color="blue"
        )
        # axs[i].scatter(t[pks4],np.ones(len(pks4))*4,marker="|",s=50,color="blue")
        # axs[i].scatter(t[pks3],np.ones(len(pks3))*3,marker="|",s=50,color="blue")
        # axs[i].scatter(t[pks2],np.ones(len(pks2))*2,marker="|",s=50,color="blue")
        # axs[i].scatter(t[pks1],np.ones(len(pks1))*1,marker="|",s=50,color="blue")
        # axs[i].set_yticks(list(range(1,neurons+1))); axs[i].set_yticklabels(['Neuron 1','Neuron 2','Neuron 3','Neuron 4'])
    makeNice([axs[i] for i in range(neurons + 1)])
    plt.savefig(f"{save_direc}/raster.eps")
    plt.close()
    run_cmd(f"open {save_direc}/raster.eps")


def save_data(sol, vs, vsms, control, neurons, save_direc):
    np.savetxt(f"{save_direc}/sol.txt", sol, delimiter="\t", newline="\n")
    for i in range(1, neurons + 1):
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
            f"{save_direc}/control{i}.txt",
            np.asarray(control[i - 1]),
            delimiter="\t",
            newline="\n",
        )


def generate_figure_7(
    direc,
    save_direc,
    dt=1 / 128,
    tf=10000,
    control1=[1, 1],
    ifm=5,
    ifn=3,
    neurons=4,
):
    np.random.seed(24)
    # direc = "/Users/johnparker/paper_repos/Parker_Short_2023_mutual_stabilization/paper_data/"
    save_direc = f"{save_direc}/loop_stabilization"

    kd.check_direc(save_direc)
    save_direc = f"{save_direc}/{cupg.ctrl_to_string(control1)}"
    kd.check_direc(save_direc)
    save_direc = f"{save_direc}/neurons_{neurons}_q_{ifm}_k_{ifn}"
    kd.check_direc(save_direc)

    sol, vs, vsms, control = unidirectional_loop(
        direc, dt, tf, neurons, control1, ifm, ifn
    )
    save_data(sol, vs, vsms, control, neurons, save_direc)

    sol = np.loadtxt(f"{save_direc}/sol.txt")
    plot_loop_mutual_stabilization(sol, neurons, save_direc)
    run_cmd(f"cp {save_direc}/mutual_stabilization_3d.eps paper_figures/figure_7.eps")

    """
    np.random.seed(24)
    dt = 1 / 128
    tf = 10000
    neurons = 4
    control1 = [1, 1]
    ifm = 5
    ifn = 3

    direc = "/Users/johnparker/paper_repos/Parker_Short_2023_mutual_stabilization/paper_data/"
    save_direc = f"{direc}/loop_stabilization_test"
    kd.check_direc(save_direc)
    save_direc = f"{save_direc}/{cupg.ctrl_to_string(control1)}"
    kd.check_direc(save_direc)
    save_direc = f"{save_direc}/neurons_{neurons}_q_{ifm}_k_{ifn}"
    kd.check_direc(save_direc)
    sol, vs, vsms, control = unidirectional_loop(
        direc, dt, tf, neurons, control1, ifm, ifn
    )
    save_data(sol, vs, vsms, control, neurons, save_direc)
    sol = np.loadtxt(f"{save_direc}/sol.txt")
    plot_loop_mutual_stabilization(sol, neurons, save_direc)
    run_cmd(f"cp {save_direc}/mutual_stabilization_3d.eps paper_figures/figure_7.eps")
    """
