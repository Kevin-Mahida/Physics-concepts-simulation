import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#arrays
t = np.linspace(0, 1, 1000, endpoint=False)
freqs = np.fft.fftfreq(len(t), t[1] - t[0])
pos_freqs = freqs[:500]

theta = np.linspace(-0.05, 0.05, 1000)
wavelength = 500e-9


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Dynamic Spectral Analysis & Diffraction', fontsize=16, fontweight='bold')

# Panel 1
ax1.set_xlim(0, 150)
ax1.set_ylim(0, 600)
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Live Fast Fourier Transform (FFT)')
ax1.grid(True)

# Panel 2
ax2.set_xlim(-0.05, 0.05)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('Angle (radians)')
ax2.set_ylabel('Relative Intensity')
ax2.set_title('Single Slit Diffraction ($a$ changing)')
ax2.grid(True)


line_fft, = ax1.plot([], [], 'b-', lw=2)
line_diff, = ax2.plot([], [], 'r-', lw=2)

hud = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, va='top', ha='left',
               fontfamily='monospace', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

def init():
    line_fft.set_data([], [])
    line_diff.set_data([], [])
    hud.set_text('')
    return line_fft, line_diff, hud


def update(frame):
    f2 = 50 + 50 * np.sin(frame * 0.05)
    signal = np.sin(50 * 2 * np.pi * t) + 0.5 * np.sin(f2 * 2 * np.pi * t)
    fft_vals = np.fft.fft(signal)
    line_fft.set_data(pos_freqs, np.abs(fft_vals)[:500])


    a = 50e-6 + 30e-6 * np.sin(frame * 0.05)
    beta = (np.pi * a * np.sin(theta)) / wavelength
    intensity = np.sinc(beta / np.pi)**2
    line_diff.set_data(theta, intensity)

    hud.set_text(f"Signal 2 Freq: {f2:.1f} Hz\nSlit Width:    {a*1e6:.1f} um")
    return line_fft, line_diff, hud


ani = animation.FuncAnimation(fig, update, frames=125, init_func=init, blit=False)
ani.save('ft.gif', writer='pillow', fps=30)
print("Saved 'ft.gif'")
