from functools import partial

import numpy as np
import pandas as pd

from math import cos

def derivative_plus(f):
    h = 0.01
    def derivative_(x):
        return (f(x + h) - f(x))/h
    return derivative_


def derivative_minus(f):
    h = 0.0001
    def derivative_(x):
        return (f(x) - f(x - h))/h
    return derivative_

def derivative_0(f):
    def derivative_(x):
        return 0.5 * (derivative_plus(f)(x) + derivative_minus(f)(x))
    return derivative_

def derivative_centered(f, h=0.1):
    def derivative_(x):
        return (1/h**2)*()


def derivative_df(fn, differentiator_fn):
    fnprime = differentiator_fn(fn)
    fnprimeprime = differentiator_fn(fnprime)
    fnprimeprimeprime = differentiator_fn(fnprimeprime)
    x = np.array([1, 2, 3, 4, 5])
    return pd.DataFrame(
        {
         "x": x,
         "g(x)": np.apply_along_axis(fn, 0, x),
         "g'(x)": np.apply_along_axis(fnprime, 0, x),
         "g''(x)": np.apply_along_axis(fnprimeprime, 0, x),
         "g'''(x)": np.apply_along_axis(fnprimeprimeprime, 0, x)
        }
    )

def solve_diffusion_equation(f, h=0.1):
    coef = 1/(h**2)
    alpha = 0
    beta =  50
    F = np.array([
        f(0.1) - alpha*coef,
        f(0.2),
        f(0.3),
        f(0.4),
        f(0.5),
        f(0.6),
        f(0.7),
        f(0.8),
        f(0.9) - beta*coef,
    ])
    A = np.diag(np.full(8, 1), k=-1) + \
        np.diag(np.full(9, -2), k=0) + \
        np.diag(np.full(8, 1), k=1)

    print(f"{coef*A} * U = {np.transpose(F)}")
    return np.linalg.solve(coef*A, F)




if __name__ == '__main__':
    np.set_printoptions(precision=30, suppress=True)
    f = lambda x: 6*x + 100
    solution = solve_diffusion_equation(f)
    actual = lambda x: x*(x**2 + 50*x -1)
    actual_ = np.array([
        actual(0.1),
        actual(0.2),
        actual(0.3),
        actual(0.4),
        actual(0.5),
        actual(0.6),
        actual(0.7),
        actual(0.8),
        actual(0.9)
    ])
    print(solution - actual_)
