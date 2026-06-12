import numpy as np
from scipy.linalg import solve, eig
import matplotlib.pyplot as plt
from scipy.io import savemat

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
    
    def plot_solution(self, coeffs, x_grid=None, num_points=200,
                      plot_real=True, plot_imag=False, title=None,
                      label=None, show=True, **plot_kwargs):
        if x_grid is None:
            x_grid = np.linspace(0, self.L, num_points)
        u = self.reconstruct(coeffs, x_grid)

        fig, ax = plt.subplots(figsize=(8, 4))
        if plot_real:
            lbl = f'{label} (real)' if label else 'Re(u)'
            ax.plot(x_grid, u.real, label=lbl, **plot_kwargs)
        if plot_imag:
            lbl = f'{label} (imag)' if label else 'Im(u)'
            ax.plot(x_grid, u.imag, '--', label=lbl, **plot_kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        if title:
            ax.set_title(title)
        if plot_real and plot_imag:
            ax.legend()
        ax.grid(True)
        if show:
            plt.show()
        return fig, ax

    def plot_eigenfunction(self, eigvec, eigenvalue, x_grid=None,
                           num_points=200, title=None, show=True, **plot_kwargs):
        if title is None:
            title = f'Eigenfunction, λ = {eigenvalue.real:.4f}'
        return self.plot_solution(eigvec, x_grid=x_grid, num_points=num_points,
                                  plot_real=True, plot_imag=False,
                                  title=title, show=show, **plot_kwargs)

    def plot_eigenvalues(self, eigvals, title='Eigenvalues',
                     show=True, mark_real_axis=True, ylim=None,
                     **scatter_kwargs):
        
        fig, ax = plt.subplots(figsize=(6, 6))
        if 's' not in scatter_kwargs:
            scatter_kwargs['s'] = 80
        ax.scatter(eigvals.real, eigvals.imag, **scatter_kwargs)
        ax.set_xlabel('Re(λ)')
        ax.set_ylabel('Im(λ)')
        ax.set_title(title)
        if mark_real_axis:
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            im_min, im_max = np.min(eigvals.imag), np.max(eigvals.imag)
            if np.allclose(im_min, im_max):
                ax.set_ylim(-1, 1)  

        if show:
            plt.show()
        return fig, ax
    
    def save_solution(self, coeffs, filename, format='npy'):
        if format == 'mat':
            filename = filename if filename.endswith('.mat') else filename + '.mat'
            mdic = {
                'coeffs': coeffs,
                'N': self.N,
                'L': self.L,
                'modes': self.modes
            }
            savemat(filename, mdic)
        elif format == 'npy':
            filename = filename if filename.endswith('.npy') else filename + '.npy'
            np.save(filename, coeffs)
        elif format == 'npz':
            filename = filename if filename.endswith('.npz') else filename + '.npz'
            np.savez(filename, coeffs=coeffs)
        else:
            raise ValueError("format must be 'npy', 'npz' or 'mat'")

    def save_eigenvalues(self, eigvals, eigvecs=None, filename='eigenvalues',
                         format='npz'):
        if format == 'mat':
            filename = filename if filename.endswith('.mat') else filename + '.mat'
            mdic = {'eigvals': eigvals}
            if eigvecs is not None:
                mdic['eigvecs'] = eigvecs
            mdic['N'] = self.N
            mdic['L'] = self.L
            mdic['modes'] = self.modes
            savemat(filename, mdic)
        elif format == 'npz':
            filename = filename if filename.endswith('.npz') else filename + '.npz'
            if eigvecs is not None:
                np.savez(filename, eigvals=eigvals, eigvecs=eigvecs)
            else:
                np.savez(filename, eigvals=eigvals)
        elif format == 'npy':
            np.save(filename + '_eigvals.npy', eigvals)
            if eigvecs is not None:
                np.save(filename + '_eigvecs.npy', eigvecs)
        else:
            raise ValueError("format must be 'npz', 'npy' or 'mat'")

    def save_to_mat(self, filename, **variables):
       
        filename = filename if filename.endswith('.mat') else filename + '.mat'
        savemat(filename, variables)


    def hold_plots(self):
        was_interactive = plt.isinteractive()
        plt.ioff()
        plt.show()
        if was_interactive:
            plt.ion()