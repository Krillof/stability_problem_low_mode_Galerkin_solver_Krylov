import numpy as np
import matplotlib.pyplot as plt

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