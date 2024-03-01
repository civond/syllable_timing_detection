import librosa as lr
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
# Read the audio file
audio_file = 'Audio_Filtered/budgie_filtered_0.flac'
audio, sr = lr.load(audio_file, sr=None)  # sr=None to preserve the original sampling rate


# Compute the short-time Fourier transform (STFT)

n_fft = 2048
ft = np.abs(lr.stft(audio, n_fft=n_fft,  hop_length=256))

print(ft)
print(len(ft))
ft_dB = lr.amplitude_to_db(ft, ref=np.max)


"""window_size = 101

# Apply the median filter
filtered_audio = signal.medfilt(audio, kernel_size=window_size)
#filtered_audio = signal.medfilt(filtered_audio, kernel_size=window_size)
#filtered_audio = signal.medfilt(filtered_audio, kernel_size=window_size)

# Save or play the filtered audio
wavfile.write('median_audio.wav', sr, filtered_audio)"""

plt.figure(1);
lr.display.specshow(ft_dB, sr=sr, x_axis='time', y_axis='linear');    
plt.show()


"""plt.figure(1);
fig, ax = plt.subplots()
img = lr.display.specshow(lr.amplitude_to_db(S, ref=np.max), y_axis='linear', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.xlim(25,30)
plt.show();"""