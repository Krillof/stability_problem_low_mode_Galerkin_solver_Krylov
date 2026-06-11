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

def identity_matrix(N): # Матрица оператора тождества
    size = 2*N + 1
    return np.eye(size)


def diff_matrix(N, L, order=1):
    modes = np.arange(-N, N + 1)
    k = modes/L

    if order == 1:
        diag = 1j*k
    elif order == 2:
        diag = -k**2
    else:
        diag = (1j*k)**order
    return np.diag(diag)


def mult_matrix(N, L, func, M_fine=None):
    coeffs_wide = _compute_fourier_coeffs(func, 2*N, L, M_fine)
    sqrtL = np.sqrt(L)
    size = 2*N + 1
    mat = np.zeros((size, size), dtype=complex)
    modes = np.arange(-N, N + 1)

    for i, m in enumerate(modes):
        for j, n in enumerate(modes):
            p = m - n               
            idx = p + 2*N         
            mat[i, j] = coeffs_wide[idx]/sqrtL
    return mat