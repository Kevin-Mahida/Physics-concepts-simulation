import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import matplotlib.animation as animation

#physics constant
N = 500          
x_max = 5.0       
x = np.linspace(-x_max, x_max, N)
dx = x[1] - x[0]

omega = 1.0
V = 0.5 * omega**2 * x**2 

kinetic_coeff = -1.0 / (2.0 * dx**2)
main_diag = -2.0 * kinetic_coeff * np.ones(N) + V
off_diag = kinetic_coeff * np.ones(N - 1)
H = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

energies, wavefunctions = eigh(H)

psi_0 = wavefunctions[:, 0] / np.sqrt(np.trapezoid(wavefunctions[:, 0]**2, x))
psi_1 = wavefunctions[:, 1] / np.sqrt(np.trapezoid(wavefunctions[:, 1]**2, x))


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-1, 3)
ax.plot(x, V, 'k--', label='Potential V(x)')
ax.set_title('1D Quantum Harmonic Oscillator Phase Evolution')
ax.set_xlabel('Position (x)')
ax.set_ylabel('Wavefunction Amplitude (Real Part)')
ax.grid(True)

line_psi0, = ax.plot([], [], 'b-', lw=2, label=f'State n=0 (E={energies[0]:.2f})')
line_psi1, = ax.plot([], [], 'r-', lw=2, label=f'State n=1 (E={energies[1]:.2f})')
ax.legend(loc='upper right')

def init():
    line_psi0.set_data([], [])
    line_psi1.set_data([], [])
    return line_psi0, line_psi1


def update(frame):
    t = frame * 0.1
    # Psi(x,t) = psi(x) * e^(-i * E * t) 
    Psi0_t = psi_0 * np.exp(-1j * energies[0] * t)
    Psi1_t = psi_1 * np.exp(-1j * energies[1] * t)
    
    line_psi0.set_data(x, energies[0] + np.real(Psi0_t))
    line_psi1.set_data(x, energies[1] + np.real(Psi1_t))
    
    return line_psi0, line_psi1


ani = animation.FuncAnimation(fig, update, frames=150, init_func=init, blit=False)
ani.save('qmo.gif', writer='pillow', fps=30)
print("Saved 'qmo.gif'")
