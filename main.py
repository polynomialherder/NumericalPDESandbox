import math

from math import pi

import numpy as np

from solver.stokes import StokesSolver

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def test_factory(L, H, N):
    if isinstance(N, float):
        N = int(N)
    a = 2
    b = 2
    p = lambda x, y: -0*x/L - 0
    p_x = lambda x, y: -2/L
    p_y = lambda x, y: 0

    u_actual = lambda x, y: L*np.sin(a*pi*x/L)*np.cos(b*pi*y/H)
    uxx = lambda x, y: -(a**2 * pi**2 * np.cos(b*pi*y/H)*np.sin(a*pi*x/L))/L
    uyy = lambda x, y: -(b**2 * L * pi**2* np.cos(b*pi*y/H)*np.sin(a*pi*x/L))/H**2

    v_actual = lambda x, y: -H*np.cos(a*pi*x/L)*np.sin(b*pi*y/H)
    vyy = lambda x, y: (b**2 * pi**2 * np.cos(a*pi*x/L)*np.sin(b*pi*y/H))/H
    vxx = lambda x, y: (a**2 * H * pi**2 * np.cos(a*pi*x/L)*np.sin(b*pi*y/H))/L**2
    f = lambda x, y: uxx(x, y) + uyy(x, y) - p_x(x, y)
    g = lambda x, y: vxx(x, y) + vyy(x, y) - p_y(x, y)
    return StokesSolver(
        f=f,
        g=g,
        u_actual=u_actual,
        v_actual=v_actual,
        p_actual=p,
        rows_x=N,
        rows_y=N,
        x_lower=0,
        x_upper=L,
        y_lower=0,
        y_upper=H
    )


if __name__ == '__main__':
    s1 = test_factory(1, 1, 1000)
    sqrt_resolutions = np.array([
        32, 45, 71, 100, 141, 224
    ])
    errors_u_p1 = []
    errors_u_p2 = []
    errors_u_pinf = []

    errors_v_p1 = []
    errors_v_p2 = []
    errors_v_pinf = []

    errors_p_p1 = []
    errors_p_p2 = []
    errors_p_pinf = []

    for sqrt_n in sqrt_resolutions:
        print(f"Calculating errors on grid resolution N={sqrt_n**2:,G}")
        s = test_factory(1, 1, sqrt_n)

        errors_u_p1.append(
            s.error_u(p=1)
        )
        errors_u_p2.append(
            s.error_u(p=2)
        )
        errors_u_pinf.append(
            s.error_u(p=np.Inf)
        )

        errors_v_p1.append(
            s.error_v(p=1)
        )
        errors_v_p2.append(
            s.error_v(p=2)
        )
        errors_v_pinf.append(
            s.error_p(p=np.Inf)
        )

        errors_p_p1.append(
            s.error_p(p=1)
        )
        errors_p_p2.append(
            s.error_p(p=2)
        )
        errors_p_pinf.append(
            s.error_p(p=np.Inf)
        )

    p1_color = "forestgreen"
    p2_color = "slateblue"
    pinf_color = "maroon"

    fig, ax = plt.subplots()

    resolutions = sqrt_resolutions**2
    ax.set_title("$\\lVert \\hat{u} - u \\rVert$")
    ax.plot(resolutions, errors_u_p1, "--o", color=p1_color, label="$1$-norm")
    ax.plot(resolutions, errors_u_p2, "--o", color=p2_color, label="$2$-norm")
    ax.plot(resolutions, errors_u_pinf, "--o", color=pinf_color, label=r"$\infty$-norm")
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=2)
    ax.grid()
    fig.legend()
    fig.show()

    fig, ax = plt.subplots()
    ax.set_title("$\\lVert \\hat{v} - v \\rVert$")
    ax.plot(resolutions, errors_v_p1, "--o", color=p1_color, label="$1$-norm")
    ax.plot(resolutions, errors_v_p2, "--o", color=p2_color, label="$2$-norm")
    ax.plot(resolutions, errors_v_pinf, "--o", color=pinf_color, label=r"$\infty$-norm")
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=2)
    ax.grid()
    fig.legend()
    fig.show()

    fig, ax = plt.subplots()
    ax.set_title("$\\lVert \\hat{p} - p \\rVert$")
    ax.plot(resolutions, errors_p_p1, "--o", color=p1_color, label="$1$-norm")
    ax.plot(resolutions, errors_p_p2, "--o", color=p2_color, label="$2$-norm")
    ax.plot(resolutions, errors_p_pinf, "--o", color=pinf_color, label=r"$\infty$-norm")
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=2)
    ax.grid()
    fig.legend()
    fig.show()
