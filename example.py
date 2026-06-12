# example.py
import numpy as np
import matplotlib.pyplot as plt
from galerkin import FourierGalerkinSolver
from operator_matrices import diff_matrix, identity_matrix
import os
from scipy.io import loadmat


N = 8
L = 2*np.pi

solver = FourierGalerkinSolver(N, L)

terms = [(-1.0, diff_matrix(N, L, order=2)),
         ( 1.0, identity_matrix(N))]
L_mat = solver.assemble_operator(terms)
print("1. Operator matrix assembled.")

f = lambda x: np.sin(x)
f_coeffs = solver.project_rhs(f)
print("2. RHS projected.")

c = solver.solve(L_mat, f_coeffs)
print("3. System solved, coefficients obtained.")

x_grid = np.linspace(0, L, 200)
u_vals = solver.reconstruct(c, x_grid)
print("4. Solution reconstructed on a grid.")

fig1, ax1 = solver.plot_solution(c, title="Solution of -u'' + u = sin(x)",
                                 color='blue', linewidth=2, show=False)
print("5. Solution plot created (not shown yet).")

solver.save_solution(c, 'solution_coeffs', format='npy')
solver.save_solution(c, 'solution_coeffs', format='npz')
solver.save_solution(c, 'solution_coeffs', format='mat')
print("6. Solution saved in .npy, .npz, .mat.")

loaded_c = np.load('solution_coeffs.npy')
assert np.allclose(loaded_c, c), "Error loading .npy"
print("   .npy check: OK")

data_npz = np.load('solution_coeffs.npz')
assert np.allclose(data_npz['coeffs'], c)
print("   .npz check: OK")

data_mat = loadmat('solution_coeffs.mat')
loaded_c_mat = data_mat['coeffs'].flatten()
assert np.allclose(loaded_c_mat, c)
print("   .mat check: OK")

eigvals, eigvecs = solver.eigenvalues(terms, k=6,
                                      return_eigenvectors=True,
                                      sort_by='real')
print("\n7. Eigenvalues:")
for i, lam in enumerate(eigvals):
    print(f"   lambda_{i} = {lam.real:.6f}")

print("8. Plotting eigenfunctions...")
for i in range(4):
    solver.plot_eigenfunction(eigvecs[:, i], eigvals[i], show=False)

fig2, ax2 = solver.plot_eigenvalues(eigvals, title="Spectrum of -u'' + u",
                                    s=80, color='red', edgecolors='k',
                                    linewidth=0.5, show=False)
print("9. Eigenvalues plot created.")

solver.save_eigenvalues(eigvals, eigvecs, 'eigenvalues', format='npz')
solver.save_eigenvalues(eigvals, eigvecs, 'eigenvalues', format='npy')
solver.save_eigenvalues(eigvals, eigvecs, 'eigenvalues', format='mat')
print("10. Eigenvalues and eigenvectors saved in .npz, .npy, .mat.")

ev_npz = np.load('eigenvalues.npz')
assert np.allclose(ev_npz['eigvals'], eigvals)
assert np.allclose(ev_npz['eigvecs'], eigvecs)
print("   .npz check: OK")

ev_val = np.load('eigenvalues_eigvals.npy')
ev_vec = np.load('eigenvalues_eigvecs.npy')
assert np.allclose(ev_val, eigvals) and np.allclose(ev_vec, eigvecs)
print("   .npy check: OK")

ev_mat = loadmat('eigenvalues.mat')
assert np.allclose(ev_mat['eigvals'].flatten(), eigvals)
assert np.allclose(ev_mat['eigvecs'], eigvecs)
print("   .mat check: OK")

solver.save_to_mat('all_data.mat',
                   solution_coeffs=c,
                   eigenvalues=eigvals,
                   eigenvectors=eigvecs)
all_data = loadmat('all_data.mat')
assert np.allclose(all_data['solution_coeffs'].flatten(), c)
assert np.allclose(all_data['eigenvalues'].flatten(), eigvals)
assert np.allclose(all_data['eigenvectors'], eigvecs)
print("11. Universal save to .mat: OK")

print("\nAll windows are open. Close them to finish the test.")
for fig_num in plt.get_fignums():
    plt.figure(fig_num).canvas.draw()
solver.hold_plots()

file_list = ['solution_coeffs.npy', 'solution_coeffs.npz', 'solution_coeffs.mat',
             'eigenvalues.npz', 'eigenvalues_eigvals.npy', 'eigenvalues_eigvecs.npy',
             'eigenvalues.mat', 'all_data.mat']
for f in file_list:
    try:
        os.remove(f)
        print(f"Removed file: {f}")
    except:
        pass

print("\nTest completed successfully!")