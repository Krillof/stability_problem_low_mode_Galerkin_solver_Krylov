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


def _diff_block(k, order):
    J_powers = {
        0: np.array([[1.0, 0.0], [0.0, 1.0]]),
        1: np.array([[0.0, 1.0], [-1.0, 0.0]]),
        2: np.array([[-1.0, 0.0], [0.0, -1.0]]),
        3: np.array([[0.0, -1.0], [1.0, 0.0]]),
    }
    return (k ** order) * J_powers[order % 4]

def diff_matrix(N, L, order=1):
    size = 2 * N + 1
    mat = np.zeros((size, size))
    for n in range(1, N + 1):
        k = 2 * np.pi * n / L
        i = 2 * n - 1
        mat[i:i + 2, i:i + 2] = _diff_block(k, order)
    return mat


def mult_matrix(N, L, func, M_fine=None):
    size = 2 * N + 1
    if M_fine is None:
        M_fine = max(8 * N + 1, 256)
    M = M_fine
    x = np.linspace(0, L, M, endpoint=False)
    dx = L / M

    Phi = np.empty((M, size))
    Phi[:, 0] = 1.0 / np.sqrt(L)
    norm = np.sqrt(2.0 / L)
    for n in range(1, N + 1):
        k = 2 * np.pi * n / L
        Phi[:, 2 * n - 1] = norm * np.cos(k * x)
        Phi[:, 2 * n] = norm * np.sin(k * x)

    g = func(x)
    return (Phi.T * (g * dx)) @ Phi