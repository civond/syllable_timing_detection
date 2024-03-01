import librosa as lr
import matplotlib.pyplot as plt
import numpy as np

audio_path = "Data\c50_brian_filtered_song.wav"
[audio, fs] = lr.load(audio_path, sr=None)

rms = lr.feature.rms(y=audio, hop_length=256)
zero_crossings = lr.feature.zero_crossing_rate(y=audio, hop_length=256)

min=227*fs
max=231*fs

print(len(audio))
print(len(zero_crossings[0]))
print(len(rms[0]))

plt.figure(1)
plt.subplot(2,1,1)
n_fft = 2048
ft = np.abs(lr.stft(audio, n_fft=n_fft,  hop_length=256))
ft_dB = lr.amplitude_to_db(ft, ref=np.max)
lr.display.specshow(ft_dB, sr=fs, x_axis='time', y_axis='linear');    


"""plt.subplot(3,1,2)
plt.plot(zero_crossings[0],color='b')
plt.title("Zero Crossings")
plt.xlim(0,len(zero_crossings[0])-1)
plt.grid(True)"""

plt.subplot(2,1,2)
plt.plot(rms[0],color='r')
plt.title("RMS")
plt.grid(True)
plt.xlim(0,len(zero_crossings[0])-1)
plt.show()