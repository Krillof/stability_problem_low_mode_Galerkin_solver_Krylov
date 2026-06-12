import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

class FourierGalerkinSolver:
    def __init__(self, N, L=2 * np.pi):
        self.N = N
        self.L = L
        self.size = 2 * N + 1
        self.modes = np.arange(-N, N + 1)

        # Включаем интерактивный режим matplotlib,
        # чтобы plt.show() не блокировал выполнение.
        self._ion_was_on = plt.isinteractive()
        if not self._ion_was_on:
            plt.ion()
            
    def __del__(self):
        """Восстанавливаем прежний интерактивный режим."""
        if hasattr(self, '_ion_was_on') and not self._ion_was_on:
            try:
                plt.ioff()
            except Exception:
                pass
    
    def assemble_operator(self, term_list):
        L_mat = np.zeros((self.size, self.size), dtype=complex)
        for coef, mat in term_list:
            L_mat += coef * mat
        return L_mat

    def project_rhs(self, func, M_fine=None):
        if M_fine is None:
            M_fine = max(self.size, 64)
        M = M_fine
        x = np.linspace(0, self.L, M, endpoint=False)
        f_vals = func(x)
        fft = np.fft.fft(f_vals)

        coeffs = np.zeros(self.size, dtype=complex)
        sqrtL = np.sqrt(self.L)
        for i, k in enumerate(self.modes):
            idx = k % M
            coeffs[i] = (sqrtL / M) * fft[idx]
        return coeffs

    def solve(self, L_mat, rhs_coeffs):
        return solve(L_mat, rhs_coeffs)
    
    def reconstruct(self, coeffs, x_grid):
        sqrtL = np.sqrt(self.L)
        u = np.zeros_like(x_grid, dtype=complex)
        for i, n in enumerate(self.modes):
            u += coeffs[i] * np.exp(1j * 2 * np.pi * n * x_grid / self.L) / sqrtL
        return u
    
    def eigenvalues(self, term_list, k=None, return_eigenvectors=False, sort_by='real'):
        L_mat = self.assemble_operator(term_list)
        w, vr = eig(L_mat, left=False, right=True)

        if sort_by == 'real':
            idx = np.argsort(w.real)
        elif sort_by == 'magnitude':
            idx = np.argsort(np.abs(w))
        else:
            idx = np.arange(len(w))

        w_sorted = w[idx]
        if k is not None:
            w_sorted = w_sorted[:k]
            idx = idx[:k]

        if return_eigenvectors:
            v_sorted = vr[:, idx]
            return w_sorted, v_sorted
        else:
            return w_sorted

    def eigenfunction(self, coeffs, x_grid):
        return self.reconstruct(coeffs, x_grid)