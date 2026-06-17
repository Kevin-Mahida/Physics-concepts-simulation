import matplotlib
matplotlib.use('Agg') # Force headless rendering for stable GIF generation

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

#physics constant
m = 1.0   # Mass
k = 10.0  # Spring constant
c = 0.5   # Damping coefficient

def oscillator_derivatives(t, y):
    x, v = y
    dxdt = v
    dvdt = -(k/m)*x - (c/m)*v
    return [dxdt, dvdt]

# Initial conditions: x=2.0, v=0.0
y0 = [2.0, 0.0] 
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 500) # 500 frames 


solution = solve_ivp(oscillator_derivatives, t_span, y0, t_eval=t_eval, method='RK45')

# arrays
t_data = solution.t
x_data = solution.y[0]
v_data = solution.y[1]
energy_data = 0.5 * m * v_data**2 + 0.5 * k * x_data**2


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Damped Harmonic Oscillator Dynamics', fontsize=18, fontweight='bold')

# Panel 1
ax1.set_xlim(0, 20)
ax1.set_ylim(-2.5, 2.5)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Position (m)', fontsize=12)
ax1.set_title('Time Evolution')
ax1.grid(True)

line_x, = ax1.plot([], [], 'b-', lw=2, alpha=0.6)
glow_x, = ax1.plot([], [], 'bo', alpha=0.2, markersize=18) 
core_x, = ax1.plot([], [], 'bo', markersize=5)          


# Panel 2
ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-6.0, 6.0)
ax2.set_xlabel('Position (m)', fontsize=12)
ax2.set_ylabel('Velocity (m/s)', fontsize=12)
ax2.set_title('Phase Space Trajectory')
ax2.grid(True)

line_phase, = ax2.plot([], [], 'r-', lw=1.5, alpha=0.6)
glow_phase, = ax2.plot([], [], 'ro', alpha=0.2, markersize=18)
core_phase, = ax2.plot([], [], 'ro', markersize=5)


telemetry = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, va='top', ha='left',
                     fontfamily='monospace', color='black', fontsize=11,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

def init():
    """Initializes the baseline state of the animation."""
    line_x.set_data([], [])
    glow_x.set_data([], [])
    core_x.set_data([], [])
    line_phase.set_data([], [])
    glow_phase.set_data([], [])
    core_phase.set_data([], [])
    telemetry.set_text('')
    return line_x, glow_x, core_x, line_phase, glow_phase, core_phase, telemetry


def update(frame):
    
    line_x.set_data(t_data[:frame], x_data[:frame])
    glow_x.set_data([t_data[frame]], [x_data[frame]])
    core_x.set_data([t_data[frame]], [x_data[frame]])

    
    line_phase.set_data(x_data[:frame], v_data[:frame])
    glow_phase.set_data([x_data[frame]], [v_data[frame]])
    core_phase.set_data([x_data[frame]], [v_data[frame]])

    
    t_current = t_data[frame]
    x_current = x_data[frame]
    v_current = v_data[frame]
    e_current = energy_data[frame]
    
    hud_text = (f"Time:   {t_current:>5.2f} s\n"
                f"Pos:    {x_current:>5.2f} m\n"
                f"Vel:    {v_current:>5.2f} m/s\n"
                f"Energy: {e_current:>5.2f} J")
    telemetry.set_text(hud_text)

    return line_x, glow_x, core_x, line_phase, glow_phase, core_phase, telemetry


ani = animation.FuncAnimation(fig, update, frames=len(t_eval), 
                              init_func=init, blit=False)

# Save directly to GIF 
ani.save('ho.gif', writer='pillow', fps=30)
print("Success! Saved as 'ho.gif'.")
