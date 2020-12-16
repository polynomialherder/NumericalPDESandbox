import numpy as np

from simple_second_order import _2norm

if __name__ == '__main__':
    # Comparison of differential operators A and A_
    # A corresponds to a modified equation 2.57 in LeVeque (modified to support 2 Dirichlet boundary conditions)
    # A_ corresponds to equation 2.10
    # This is to test the generalizability of 2.57
    u = lambda x: (1/20)*x*(x ** 4 + 399)
    f = lambda x: x ** 3
    h = 0.1
    grid = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    F = np.apply_along_axis(f, 0, grid)

    A = (1/h**2)*np.array([
        [h**2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, h**2]
    ])
    F[0] = 0
    F[-1] = 20
    solution = np.linalg.solve(A, F)
    residuals = A @ solution - F
    uhat = np.apply_along_axis(u, 0, grid)
    lte = A @ uhat - F
    lte_2norm = _2norm(h, lte)
    # lte_2norm evaluates to 0.002669269562960097

    grid_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    A_ = (1/h**2)*np.array([
        [-2, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, -2, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, -2, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, -2, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, -2, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, -2, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, -2],
    ])
    F_ = np.apply_along_axis(f, 0, grid_)
    F_[0] = F_[0]
    F_[-1] = F_[-1] - 20/h**2
    solution_ = np.linalg.solve(A_, F_)
    residuals_ = A_ @ solution_ - F_
    uhat_ = np.apply_along_axis(u, 0, grid_)
    lte_ = A_ @ uhat_ - F_
    lte__2norm = _2norm(h, lte_)
    # lte__2norm evaluates to 0.0026692695628849137
    # so we can simplify the implementation by using an operator
    # based on A
