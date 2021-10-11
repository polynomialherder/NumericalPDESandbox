import math

import numpy as np

from scipy.fft import fft, ifft

if __name__ == '__main__':
    f = lambda x: x**2
    mesh = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    F = f(mesh)
    transformed = fft(F)
    midpoint = len(F) // 2
    k = np.array([i if i <= midpoint else len(mesh)-i for i in range(len(F))])
    L = 1
    coefficients = 1j*2*math.pi*k/L
    solution = ifft(coefficients*transformed)
