import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal


data = "temp/short.flac"
data = "Audio_Filtered/Ti81_end_filtered_0.flac"
[audio, fs] = lr.load(data, sr=None)
audio = audio*32767

audio_rms = lr.feature.rms(y=audio, hop_length=256, frame_length=512)[0] #original = 1024

interpolated_rms = np.interp(np.arange(len(audio)), 
                             np.arange(len(audio_rms))*256, 
                             audio_rms)

audio_zc = lr.feature.zero_crossing_rate(audio, hop_length=256, frame_length=512)
interpolated_zc = np.interp(np.arange(len(audio)), 
                             np.arange(len(audio_zc[0]))*256, 
                             audio_zc[0])

duration = len(audio)/fs
dt = 1/fs
t = np.arange(0,duration,dt)

"""plt.figure(1, figsize=(8,6))
plt.subplot(3,1,1)
n_fft = 1024
ft = np.abs(lr.stft(audio, n_fft=n_fft,  hop_length=256))
ft_dB = lr.amplitude_to_db(ft, ref=np.max)
lr.display.specshow(ft_dB, sr=fs*2, x_axis='time', y_axis='linear')
#plt.title(f"Start: {round(start/fs, 2)}, Avg RMS: {np.round(np.mean(interpolated_rms[start:end]),2)}")
plt.title("Spectrogram View")
plt.xlim(43.5,46)

plt.subplot(3,1,2)
plt.title(f"Zero Crossings")
plt.plot(t, interpolated_zc, color='b')
plt.axhline(y = 0.25, xmin = 0, xmax = 100,
                                color = 'r', linestyle = '--', linewidth=1)
plt.xlim(43.5,46)
plt.grid(True)
plt.tight_layout()

plt.subplot(3,1,3)
plt.title(f"RMS")
plt.plot(t, interpolated_rms, color='r')
plt.axhline(y = 0.25, xmin = 0, xmax = 100,
                                color = 'r', linestyle = '--', linewidth=1)
plt.xlim(43.5,46)
plt.grid(True)
plt.tight_layout()
plt.show()"""

threshold = 5 #30 original -> 10

temp = np.where(interpolated_rms >= threshold, interpolated_rms, np.nan)
#thresholded_audio = np.where((interpolated_rms >= threshold) & (interpolated_rms < 2000), audio, np.nan)
thresholded_audio = np.where(interpolated_rms >= threshold, audio, np.nan)
#thresholded_audio = np.where(interpolated_rms < 800, thresholded_audio, np.nan)

clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0

print('Writing RMS thresholded')
sf.write("temp/low_RMS.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')
input('RMS ')
# Drop short sections
start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]
    drop_time = (20/1000)*fs # ms

    if len(section) < drop_time:
        thresholded_audio[start:end] = np.nan
    if np.max(np.abs(section)) > 6000:
        thresholded_audio[start:end] = np.nan
    
clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0

print('writing')
sf.write("temp/low_RMS_noshort.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')

start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]
    merge_short = (30/1000)*fs # ms

    duration = len(section)/fs
    dt = 1/fs
    t = np.arange(0,duration,dt)

    """plt.figure(1,figsize=(8,5))
    plt.subplot(2,1,1)
    n_fft = 1024
    ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
    ft_dB = lr.amplitude_to_db(ft, ref=np.max)
    lr.display.specshow(ft_dB, sr=fs, x_axis='time', y_axis='linear')
    plt.title(f"Start: {round(start/fs, 2)}, Avg RMS: {np.round(np.mean(interpolated_rms[start:end]),2)}")
    plt.tight_layout()

    plt.subplot(2,1,2)
    plt.title(f"LEN: {np.round(len(section)/fs,2)} s")
    plt.plot(section)
    plt.tight_layout()
    plt.show()"""

    if nan_count < merge_short:
        thresholded_audio[start:start+length] = audio[start:start+length]

# Drop short sections
start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]
    drop_time = (50/1000)*fs # ms

    if len(section) < drop_time:
        thresholded_audio[start:end] = np.nan
clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0

print('writing')
sf.write("temp/low_RMS_100ms.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')

# Zero Crossing Thresholding
start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]

    zc_section = interpolated_zc[start:end]

    #fft = np.fft.fft(section)
    #power = np.abs(fft) ** 2
    #power_dB = 20* np.log10(np.abs(power))
    #freqs = np.fft.fftfreq(len(section), 1/fs)

    """local_max, _ = signal.find_peaks(abs(section), distance=30)

    duration = len(section)/fs
    dt = 1/fs
    t = np.arange(0,duration,dt)
    
    plt.figure(1,figsize=(8,5))
    plt.subplot(2,2,1)
    n_fft = 1024
    ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
    ft_dB = lr.amplitude_to_db(ft, ref=np.max)
    lr.display.specshow(ft_dB, sr=fs, x_axis='time', y_axis='linear')
    plt.title(f"Start: {round(start/fs, 2)}, Avg RMS: {np.round(np.mean(interpolated_rms[start:end]),2)}")
    plt.tight_layout()

    plt.subplot(2,2,2)
    plt.title(f"Avg. ZC: {np.round(np.mean(zc_section),2)}")
    plt.plot(zc_section, color='b')

    plt.subplot(2,2,3)
    plt.title(f"LEN: {np.round(len(section)/fs,2)} s")
    plt.plot(section)
    plt.tight_layout()
    plt.show()
    """
    if np.mean(zc_section) > 0.3:
        thresholded_audio[start:end] = np.nan

clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0
print(f"Writing")
sf.write("temp/low_zc.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')


# Merge near sections
start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]
    merge_short = (100/1000)*fs # ms

    if nan_count < merge_short:
        thresholded_audio[start:start+length] = audio[start:start+length]


clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0
print(f"Writing")
sf.write("temp/low_zc_merge.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')

# Histogram
start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]
    zc_section = interpolated_zc[start:end]

    fft = np.fft.fft(section)
    num_samples = len(section)
    frequencies = np.fft.fftfreq(num_samples, 1 / fs)

    # Calculate the histogram
    hist, bins = np.histogram(abs(section**2), bins=200)
    hist_db= 20 * np.log10(hist)

    inf_indices = np.isinf(hist_db)
    hist_db[inf_indices] = 0
    hist[inf_indices] = 0

    sum_low = np.round(np.sum(hist_db[:25]),1)
    sum_high = np.round(np.sum(hist_db[25:]),1)

    """plt.figure(figsize=(8,5))
    plt.subplot(2,2,1)
    n_fft = 1024
    ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
    ft_dB = lr.amplitude_to_db(ft, ref=np.max)
    lr.display.specshow(ft_dB, sr=fs, x_axis='time', y_axis='linear')
    plt.title(f"Start: {round(start/fs, 2)}, Avg RMS: {np.round(np.mean(interpolated_rms[start:end]),2)}")
    plt.tight_layout()

    plt.subplot(2,2,2)
    plt.bar(bins[:-1], hist_db, width=np.diff(bins))
    plt.scatter(bins[:25], hist_db[:25], color='b', s=2)
    plt.scatter(bins[25:-1], hist_db[25:], color='r', s=2)
    plt.title(f"Low: {sum_low}, High: {sum_high}, Ratio: {round(sum_high/sum_low,2)}")
    plt.xlim(0,bins[-2])
    plt.tight_layout()

    plt.subplot(2,2,3)
    plt.plot(section,color='b')

    plt.subplot(2,2,4)
    plt.title(f"Avg. ZC: {np.round(np.mean(zc_section),2)}")
    plt.plot(zc_section, color='b')

    plt.show()"""

    #if sum_low > 1.5*sum_high:
    if (sum_high/sum_low) < 0.3:
        thresholded_audio[start:end] = np.nan
    else:
        #print("Pass")
        pass

clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0
print(f"Writing")
sf.write("temp/zc_hist.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')
input('zc hist')

############
"""start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]
    drop_time = (30/1000)*fs # ms
    if len(section) < drop_time:
        thresholded_audio[start:end] = np.nan

start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]

    section_rms = interpolated_rms[start:end]
    

    hamming_window = np.hamming(len(section))
    section_windowed = section * hamming_window
    rms_window = lr.feature.rms(y=section_windowed, hop_length=256, frame_length=1024)[0]
    interpolated_section_rms = np.interp(np.arange(len(section)), 
                             np.arange(len(rms_window))*256, 
                             rms_window)
    
    print(len(interpolated_section_rms))
    #section_rms_hamming = interpolated_section_rms[start:end]
    

    local_max_rms, _ = signal.find_peaks(section_rms, 
                                         distance=30,
                                         height=50)
    peak_values = interpolated_rms[start:end][local_max_rms]  # Extract peak values
    average_peak_value = np.mean(peak_values)

    if len(peak_values) == 1:
        thresholded_audio[start:end] = np.nan
    else:

        #hamming_window = np.hamming(len(section))
        #signal_windowed = section * hamming_window

        duration = len(section)/fs
        dt = 1/fs
        t = np.arange(0,duration,dt)

        plt.figure(2)
        plt.subplot(2,1,1)
        plt.plot(t, section,color= 'b')
        plt.plot(t, signal_windowed, color='r')
        plt.subplot(2,1,2)
        plt.plot(t,hamming_window)

        plt.figure(1)
        plt.subplot(2,2,1)
        n_fft = 1024
        ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
        ft_dB = lr.amplitude_to_db(ft, ref=np.max)
        lr.display.specshow(ft_dB, sr=fs, x_axis='time', y_axis='linear')
        plt.title(f"Start: {round(start/fs, 2)}, Avg RMS: {np.round(np.mean(interpolated_rms[start:end]),2)}")
        plt.tight_layout()

        plt.subplot(2,2,2)
        plt.title(f"len: {len(section)}")
        plt.plot(t, section, color='b')
        plt.plot(t,section_windowed, color='g')
        plt.axhline(y = 4000, xmin = 0, xmax = 10,
                                color = 'r', linestyle = '--', linewidth=1)
        plt.grid(True)
        plt.xlim(0,t[-1])

        plt.subplot(2,2,3)
        plt.plot(t,interpolated_rms[start:end], color='b')
        plt.plot(t, interpolated_section_rms, color='g')
        plt.scatter(t[local_max_rms], section_rms[local_max_rms], color='r', s=10)

        plt.title(f"Avg. peak: {np.round(average_peak_value,2)}, Overall avg: {np.round(np.mean(section_rms),2)}")
        plt.xlim(0,t[-1])
        plt.grid(True)
        plt.tight_layout()

        plt.subplot(2,2,4)
        plt.plot(t, interpolated_zc[start:end], color='g')
        plt.title(f"Avg zc:{np.round(np.mean(interpolated_zc[start:end]),2)}")
        plt.xlim(0,t[-1])
        plt.ylim(0,0.7)
        plt.grid(True)
        plt.show()

        if average_peak_value < 200:
            if average_peak_value > 100 and len(peak_values) >= 3:
                pass
            else:
                thresholded_audio[start:end] = np.nan"""

"""start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]

    time_max = 400 # ms
    time_min = 30 # ms

    if len(section)/fs > (time_max/1000) or len(section)/fs < (time_min/1000):
        thresholded_audio[start:end] = np.nan"""
#######
        
"""#Derivative
start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]

    local_max, _ = signal.find_peaks(abs(section), distance=30)
    duration = len(section)/fs
    dt = 1/fs
    t = np.arange(0,duration,dt)
    try:
        grad = np.gradient(np.abs(section[local_max]))
        grad2 = np.gradient(grad)
    except ValueError:
        grad = 1000*np.ones(len(section))
        grad2 = 1000*np.ones(len(section))

    if np.max(np.abs(grad)) > 250:
        
        plt.figure(1, figsize=(8,5))
        plt.subplot(3,2,1)
        n_fft = 1024
        ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
        ft_dB = lr.amplitude_to_db(ft, ref=np.max)
        lr.display.specshow(ft_dB, sr=fs, x_axis='time', y_axis='linear')
        plt.title(f"Start: {round(start/fs, 2)}")

        plt.subplot(3,2,2)
        plt.title(f"Len: {len(section)}")
        #corr = signal.correlate(section, noise, mode='same', method='fft') / (np.linalg.norm(section) * np.linalg.norm(noise))
        #corr = signal.correlate(section, noise,mode='same', method='fft')
        plt.plot(t, section, color='b')
        #plt.plot(amplitude_envelope, color='r')
        plt.plot(t[local_max], abs(section[local_max]), color='r')


        plt.subplot(3,2,3)

        plt.bar(bins[:-1], hist_db, width=np.diff(bins))
        plt.scatter(bins[:25], hist_db[:25], color='b', s=2)
        plt.scatter(bins[25:-1], hist_db[25:], color='r', s=2)
        plt.grid(True)
        plt.title(f"Low: {sum_low}, High: {sum_high}, Ratio: {round(sum_high/sum_low,2)}")
        plt.xlim(0,bins[-2])
        plt.tight_layout()

        plt.subplot(3,2,4)
        #grad = np.gradient(np.abs(section[local_max]))
        plt.title(f"d1: {np.round(np.max(np.abs(grad)),2)}, d2: {np.round(np.max(np.abs(grad2)),2)}")
        plt.plot(grad, color='r')
        plt.plot(grad2, color='g', linestyle = '--')

        plt.subplot(3,2,5)
        plt.plot(freqs, power_dB, color='b')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('FFT of Signal')
        plt.xlim(0,7000)
        plt.grid(True)
        #plt.show()
        
        thresholded_audio[start:end] = np.nan
    #thresholded_audio[start:end] = np.nan"""

    #else:
        #print("Pass")
        #pass

clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0
print(f"Writing")
sf.write("temp/low_zc_merge.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')
input('')

merge_time = 1000
merge_samples = int(round(merge_time/1000*fs,0))

start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]

    if nan_count < merge_samples:
        thresholded_audio[start:start+length] = audio[start:start+length]
    else:
        pass

clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0
print(f"Writing")
sf.write("temp/Ti81end_low_merge.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')