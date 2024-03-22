import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

data = "Audio_Filtered/Ti81_master_0.flac"
[audio, fs] = lr.load(data, sr=None)
audio = audio*32767

audio_rms = lr.feature.rms(y=audio, hop_length=256)[0]


threshold = 80  

interpolated_rms = np.interp(np.arange(len(audio)), np.arange(len(audio_rms))*256, audio_rms)

# Apply the threshold to the audio signal
thresholded_audio = np.where(interpolated_rms >= threshold, audio, np.nan)


"""clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0

print(f"Writing")
sf.write("rms_thresholded.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')
input('stop: ')"""

copy = thresholded_audio.copy()

start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))

for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]
    
    if len(section) < ((60 / 1000)*fs):
        thresholded_audio[start:end] = np.nan
        print(thresholded_audio[start:end])

start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]

    if np.max(np.abs(section)) > 4000 or np.max(np.abs(section)) < 1000:
        thresholded_audio[start:end] = np.nan
    else:
        fft = np.fft.fft(section)
        power = np.abs(fft) ** 2
        power_dB = 10* np.log10(np.abs(power))
        # Compute the corresponding frequencies
        freqs = np.fft.fftfreq(len(section), 1/fs)

        target_bin = [1000,4000]
        noise_bin = [4000,7000]

        index_1000Hz = np.argmax(freqs >= 1000)
        index_2000Hz = np.argmax(freqs >= 2000) 

        index_4000Hz = np.argmax(freqs >= 2000)
        index_7000Hz = np.argmax(freqs >= 5000) 

        # Step 5: Calculate the integral (sum) of magnitudes within the frequency range
        integral_garbage = np.sum(np.abs(power[index_1000Hz:index_2000Hz+1]))
        integral_target = np.sum(np.abs(power[index_4000Hz:index_7000Hz+1]))
        print(f"Integral: {integral_target}, Garbage: {integral_garbage}")

        if integral_target > 3*integral_garbage:
            print("section passes\n")
            pass

            plt.figure(1, figsize=(10, 5))
            plt.subplot(2,2,1)
            plt.plot(freqs, abs(fft), color='b')
            plt.title(f"Onset: {round(start/fs,2)}, Ratio:{round(integral_target/integral_garbage,2)}, Len: {round(len(section)/fs,2)}")
            plt.grid(True)
            plt.xlim(0,7000)

            plt.subplot(2,2,2)
            plt.plot(freqs, power, color='r')
            plt.title(f"Pwr Mag [0,7000]")
            plt.grid(True)
            plt.xlim(0,7000)

            plt.subplot(2,2,3)
            n_fft = 2048
            ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
            ft_dB = lr.amplitude_to_db(ft, ref=np.max)
            lr.display.specshow(ft_dB, sr=fs, x_axis='time', y_axis='linear')
            plt.title("STFT")
            plt.tight_layout()

            plt.subplot(2,2,4)
            plt.title(f"Waveform")
            plt.plot(section, color='g')
            plt.xlim(0, len(section)-1)
            plt.xlabel("Samples")
            plt.grid(True)
            #plt.show()

        else:
            thresholded_audio[start:end] = np.nan

start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
merge_time = 500
merge_samples = (2000/1000)*fs

# Merge sections within merge_samples
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]

    if nan_count < merge_samples:
        thresholded_audio[start:start+length] = audio[start:start+length]
    else:
        pass


"""plt.figure(2,figsize=(10, 5))
plt.subplot(2,1,1)
plt.plot(audio_rms)
plt.xlim(0,len(audio_rms)-1)
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(audio, color='b')
plt.plot(copy, color='r')
plt.plot(thresholded_audio, color='k')
plt.xlim(0,len(audio)-1)
plt.grid(True)
plt.show()"""


clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0


sf.write("rms_master.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')

t = np.arange(0,len(audio)/fs, 1/fs)


"""# Plot the original and thresholded audio signals
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(t, interpolated_rms, color='b')
plt.title('RMS')
plt.xlabel('Sample')
plt.axhline(y = threshold, xmin = 0, xmax = t[-1],
                        color = 'r', linestyle = '--', linewidth=1) 
#plt.xlim(0,t[-1])
plt.xlim(53,59)
plt.ylabel('Energy')

plt.subplot(2, 1, 2)
plt.plot(t, audio, color='b')
plt.plot(t, thresholded_audio, color='r')
plt.title('Thresholded Audio Signal')
plt.xlabel('Sample')
plt.xlim(53,59)
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()"""


"""import librosa as lr
import matplotlib.pyplot as plt
import numpy as np

data = "Audio_Filtered/Ti81_filtered_0.flac"
[audio, fs] = lr.load(data, sr=None)

audio = audio * 32767
rms = lr.feature.rms(y=audio, hop_length=256)
# Calculate the duration of each frame in seconds
frame_duration = 256 / fs

# Create a time axis for the RMS values
time_axis = np.arange(0, len(audio) / fs, frame_duration)[:len(rms[0])]

plt.plot(time_axis, rms[0])
plt.xlabel('Time (s)')
plt.ylabel('RMS')
plt.title('RMS of Audio Signal')
plt.show()"""