import numpy as np
import matplotlib.pyplot as plt

# Generate a sample signal
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t)  # Example signal (5 Hz sine wave)

# Apply Hamming window to the signal
hamming_window = np.hamming(len(signal))
signal_windowed = signal * hamming_window

# Compute FFT of the windowed signal
fft_signal = np.fft.fft(signal_windowed)

# Compute FFT of the Hamming window
fft_hamming = np.fft.fft(hamming_window)

# Convolve FFT of the signal with FFT of the Hamming window
convolved_fft = fft_signal * fft_hamming

# Plotting
plt.figure(figsize=(10, 6))
plt.xlim(0,10)

# Plot FFT spectrum convolved with Hamming window
freq = np.fft.fftfreq(len(t), d=t[1]-t[0])  # Frequency axis
plt.plot(freq, np.abs(convolved_fft), label='Convolved FFT Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Spectrum Convolved with Hamming Window')
plt.grid(True)
plt.legend()
plt.show()
