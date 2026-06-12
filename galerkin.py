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