import numpy as np

def _compute_fourier_coeffs(func, N, L, M_fine=None):

    if M_fine is None:
        M_fine = 2*N + 1
    M = max(M_fine, 2*N + 1)
    x = np.linspace(0, L, M, endpoint=False)
    f_vals = func(x)
    fft = np.fft.fft(f_vals)

    coeffs = np.zeros(2*N + 1, dtype=complex)
    sqrtL = np.sqrt(L)
    for i, k in enumerate(range(-N, N + 1)):
        idx = k % M
        coeffs[i] = (sqrtL/M) * fft[idx]
    return coeffs