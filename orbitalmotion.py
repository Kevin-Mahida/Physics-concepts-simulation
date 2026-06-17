import matplotlib
matplotlib.use('Agg') 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

#physics constant
GM = 4 * np.pi**2 

def orbit_deriv(t, y):
    rx, ry, vx, vy = y
    r = np.sqrt(rx**2 + ry**2)
    ax = -GM * rx / r**3
    ay = -GM * ry / r**3
    return [vx, vy, ax, ay]

y0 = [1.0, 0.0, 0.0, 5.0] 
t_eval = np.linspace(0, 2, 400) 
solution = solve_ivp(orbit_deriv, [0, 2], y0, t_eval=t_eval, method='RK45')

x_data, y_data = solution.y[0], solution.y[1]
vx_data, vy_data = solution.y[2], solution.y[3]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Orbital Dynamics & Phase Space', fontsize=16, fontweight='bold')

# Panel 1
ax1.set_aspect('equal')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlabel('X Position (AU)')
ax1.set_ylabel('Y Position (AU)')
ax1.set_title('Orbital Plane')
ax1.grid(True)
ax1.plot(0, 0, 'ro', markersize=10, label='Central Star')
ax1.legend(loc='upper right')

# Panel 2
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-8.0, 8.0)
ax2.set_xlabel('X Position (AU)')
ax2.set_ylabel('X Velocity (AU/yr)')
ax2.set_title('Phase Space ($v_x$ vs $x$)')
ax2.grid(True)


trail, = ax1.plot([], [], 'b-', lw=2, alpha=0.5)
planet, = ax1.plot([], [], 'bo', markersize=6)

phase_line, = ax2.plot([], [], 'g-', lw=2, alpha=0.5)
phase_dot, = ax2.plot([], [], 'go', markersize=6)

hud = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, va='top', ha='left',
               fontfamily='monospace', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

def init():
    trail.set_data([], [])
    planet.set_data([], [])
    phase_line.set_data([], [])
    phase_dot.set_data([], [])
    hud.set_text('')
    return trail, planet, phase_line, phase_dot, hud


def update(frame):
    trail.set_data(x_data[:frame], y_data[:frame])
    planet.set_data([x_data[frame]], [y_data[frame]])
    
    phase_line.set_data(x_data[:frame], vx_data[:frame])
    phase_dot.set_data([x_data[frame]], [vx_data[frame]])
    
    r = np.sqrt(x_data[frame]**2 + y_data[frame]**2)
    v = np.sqrt(vx_data[frame]**2 + vy_data[frame]**2)
    hud.set_text(f"Time: {t_eval[frame]:.2f} yr\nR:    {r:.2f} AU\nVel:  {v:.2f} AU/yr")
    
    return trail, planet, phase_line, phase_dot, hud


ani = animation.FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=False)
ani.save('orbital.gif', writer='pillow', fps=30)
print("Saved 'orbital.gif'")
