import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
import scipy.fftpack as fftpack

data = "temp/short.flac"
data = "Audio_Filtered/Ti81_end_filtered_0.flac"
[audio, fs] = lr.load(data, sr=None)
audio = audio*32767

audio_rms = lr.feature.rms(y=audio, hop_length=256, frame_length=256)[0] #original = 1024

interpolated_rms = np.interp(np.arange(len(audio)), 
                             np.arange(len(audio_rms))*256, 
                             audio_rms)

audio_zc = lr.feature.zero_crossing_rate(audio, hop_length=256, frame_length=256)
interpolated_zc = np.interp(np.arange(len(audio)), 
                             np.arange(len(audio_zc[0]))*256, 
                             audio_zc[0])

duration = len(audio)/fs
dt = 1/fs
t = np.arange(0,duration,dt)

threshold = 5 #30 original -> 10

temp = np.where(interpolated_rms >= threshold, interpolated_rms, np.nan)
thresholded_audio = np.where(interpolated_rms >= threshold, audio, np.nan)


"""plt.figure(1, figsize=(8,6))
plt.subplot(3,1,1)
n_fft = 1024
ft = np.abs(lr.stft(audio, n_fft=n_fft,  hop_length=256))
ft_dB = lr.amplitude_to_db(ft, ref=np.max)
lr.display.specshow(ft_dB, sr=fs*2, x_axis='time', y_axis='linear')
#plt.title(f"Start: {round(start/fs, 2)}, Avg RMS: {np.round(np.mean(interpolated_rms[start:end]),2)}")
plt.title("Spectrogram View")
plt.xlim(444,447.5)

plt.subplot(3,1,2)
plt.title(f"Zero Crossings")
plt.plot(t, interpolated_zc, color='b')
plt.axhline(y = 0.25, xmin = 0, xmax = 100,
                                color = 'r', linestyle = '--', linewidth=1)
plt.xlim(444,447.5)
plt.grid(True)
plt.tight_layout()

plt.subplot(3,1,3)
plt.title(f"RMS")
plt.plot(t, interpolated_rms, color='g')
plt.axhline(y = 5, xmin = 0, xmax = t[-1],
                                color = 'r', linestyle = '--', linewidth=1)
plt.xlim(444,447.5)
plt.ylim(0,1500)
plt.grid(True)
plt.tight_layout()
plt.show()"""

clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0

print('Writing RMS thresholded')
"""sf.write("temp/low_RMS.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')"""

# No short
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
"""sf.write("temp/low_RMS_noshort.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')
"""
# Merge near sections
start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]
    merge_short = (10/1000)*fs # ms


    if nan_count < merge_short:
        thresholded_audio[start:start+length] = audio[start:start+length]
print('writing')
clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0
"""sf.write("temp/merge_near.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')"""

# Drop spike noise
start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]
    
    peaks, _ = signal.find_peaks(section, 
                              height=0.2*np.max(section), 
                              distance = int(np.round(0.2*len(section),0))
                              )
    
    x = np.arange(0,len(section))
    """if start/fs > 456:
        plt.subplot(2,1,1)
        n_fft = 512
        ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
        ft_dB = lr.amplitude_to_db(ft, ref=np.max)
        lr.display.specshow(ft_dB, sr=fs*2, x_axis='time', y_axis='linear')
        plt.title(f"Start: {round(start/fs, 2)}, Avg RMS: {np.round(np.mean(interpolated_rms[start:end]),2)}")
        

        plt.subplot(2,1,2)
        plt.plot(x, section, color='b')
        plt.scatter(x[peaks], section[peaks], color='r')
        plt.show()"""
    
    if len(peaks) < 2:
        if len(section) > 3000:
            pass
        else:
            thresholded_audio[start:end] = np.nan

clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0
print(f"Writing")
"""sf.write("temp/no_spike.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')"""

# Bandpassing
start_freq = 300
end_freq = 7000
filter_spacing=500
num_filters = int((end_freq - start_freq) / filter_spacing)



start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]
    
    filters = []
    freq_resp = []
    filtered_output = []
    filtered_if = []
    
    if start/fs > 440: 
        fig, axes = plt.subplots(num_filters, figsize=(5, 8), sharex=True)
        for i in range(num_filters):
            low_freq = start_freq + i * filter_spacing
            high_freq = low_freq + filter_spacing

            # Calculate filter order (using a standard rule of thumb)
            filter_order = 4
            # Design the filter
            b, a = signal.butter(filter_order, 
                                [low_freq, high_freq], 
                                fs=fs,
                                btype='band')
            
            y_filtered = signal.filtfilt(b,a,section)
            filters.append((b, a))
            w, h = signal.freqz(b, a)
            freq_resp.append((w,h))
            filtered_output.append(y_filtered)
            
            analytic_signal = signal.hilbert(y_filtered)
            amplitude_envelope = np.abs(analytic_signal)
            #instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_phase = np.angle(analytic_signal)
            instantaneous_frequency = (np.diff(instantaneous_phase) /
                                    (2.0*np.pi) * fs)
            
          
            """window = signal.windows.hann(len(section))
            fft_w = fftpack.fft(amplitude_envelope*window)
            freqs = fftpack.fftfreq(len(amplitude_envelope), 1/fs)
            
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft_w = fft_w[:len(freqs)//2]
            
            power_w = np.abs(positive_fft_w) ** 2
            power_dB_w = 20* np.log10(np.abs(power_w))"""
            
            """index_500 = np.argmax(positive_freqs >= 500)
            index_6300 = np.argmax(positive_freqs >= 6300)
            average_power = np.mean(power_dB_w[index_500:index_6300])"""

            

            axes[i].plot(y_filtered, color='b', label=f'{low_freq} - {high_freq} Hz')
            axes[i].plot(amplitude_envelope, color='r')
            axes[i].set_xlim(0,len(section))
            axes[i].set_title(f'{low_freq} - {high_freq} Hz')
            axes[i].grid(True)
            
            
            #axes[i,1].plot(instantaneous_frequency, label=f'Inst Freq.')
            #plt.plot(freqs, power_dB, color='g')
            #indices_below_100Hz = np.where(positive_freqs < 50)[0]
            
            """axes2[i].scatter(positive_freqs[indices_below_100Hz[1:-1]], power_w[indices_below_100Hz[1:-1]], color='g')
            axes2[i].grid(True)"""
            #axes2[i].set_xlim(0,positive_freqs[100])
            #axes[i,1].set_ylim(-10,np.max(power_dB_w)+10)

        plt.figure(1)
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title(start/fs)
        plt.tight_layout()
        
        
        plt.figure(3)
        n_fft = 512
        ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
        ft_dB = lr.amplitude_to_db(ft, ref=np.max)
        lr.display.specshow(ft_dB, sr=fs, x_axis='time', y_axis='linear')
        plt.title(f"Start: {round(start/fs, 2)}, Avg RMS: {np.round(np.mean(interpolated_rms[start:end]),2)}")
        
        plt.show()
"""plt.figure(2, figsize=(7, 5))
for w, h in freq_resp:
    freq_hz = w * (fs / (2 * np.pi))
    plt.plot(freq_hz, 20*np.log10(abs(h)), linewidth=1)
plt.grid(True)
plt.xlim(0,end_freq+100)
plt.ylim(-20,10)
plt.title("Butterworth Filter Bank")
plt.ylabel("Attenuation (dB)")
plt.xlabel("Hz")
plt.tight_layout()
plt.show()"""


start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]
    section_zc = interpolated_zc[start:end]
    
    """if start/fs > 8*60+3:
        plt.figure(1, figsize=(8,6))
        plt.subplot(2,2,1)
        n_fft = 512
        ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
        ft_dB = lr.amplitude_to_db(ft, ref=np.max)
        lr.display.specshow(ft_dB, sr=fs*2, x_axis='time', y_axis='linear')
        plt.title(f"Start: {round(start/fs, 2)}")
        
        plt.subplot(2,2,2)
        analytic_signal = signal.hilbert(section)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) /
                                (2.0*np.pi) * fs)
        plt.plot(section, color='b')
        plt.plot(amplitude_envelope, color='r')
        plt.xlim(0,len(section)-1)
        plt.grid(True)
        plt.title(f"Length: {np.round(len(section)/fs,2)}s ({len(section)})")
        
        plt.subplot(2,2,3)
        window = signal.windows.hamming(len(section))
        
        fft = fftpack.fft(section)
        fft_w = fftpack.fft(section*window)
        freqs = fftpack.fftfreq(len(section), 1/fs)
        
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft[:len(freqs)//2]
        positive_fft_w = fft_w[:len(freqs)//2]
        
        power = np.abs(positive_fft) ** 2
        power_w = np.abs(positive_fft_w) ** 2
        power_dB = 20* np.log10(np.abs(power))
        power_dB_w = 20* np.log10(np.abs(power_w))
        
        index_500 = np.argmax(positive_freqs >= 500)
        index_6300 = np.argmax(positive_freqs >= 6300)
        average_power = np.mean(power_dB_w[index_500:index_6300])

        #plt.plot(freqs, power_dB, color='g')
        plt.plot(positive_freqs, power_dB_w, color='b')
        plt.title(f"Windowed Pwr. (avg_pwr = {np.round(average_power,2)})")

        plt.xlabel('Hz')
        plt.ylabel('Mag. (dB)')
        plt.grid(True)
        plt.xlim(0,7000)
        plt.axvline(x=500, color='r', linestyle='--', label='500 Hz')
        plt.axvline(x=6300, color='r', linestyle='--', label='6000 Hz')
        plt.axhline(y=average_power, color='g', linestyle='--')
        
        plt.subplot(2,2,4)
        plt.title(f"ZC rate. (avg: {np.round(np.mean(section_zc),2)})")
        plt.plot(section_zc, color='b')
        plt.grid(True)
        plt.xlim(0,len(section_zc)-1)
        plt.ylim(0,0.5)
        plt.axhline(y=0.3, color='r', linestyle='--')
        
        plt.tight_layout()
        

        plt.show()"""
        
"""start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
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
    if start/fs > 6*60:
        plt.figure(figsize=(8,5))
        plt.subplot(2,2,1)
        n_fft = 1024
        ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
        ft_dB = lr.amplitude_to_db(ft, ref=np.max)
        lr.display.specshow(ft_dB, sr=fs, x_axis='time', y_axis='linear')
        plt.title(f"Start: {round(start/fs, 2)}, Avg RMS: {np.round(np.mean(interpolated_rms[start:end]),2)}")

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
        plt.tight_layout()
        
        plt.show()
    
    if sum_low > 1.5*sum_high:
    #if (sum_high/sum_low) < 0.3:
        thresholded_audio[start:end] = np.nan
    else:
        #print("Pass")
        pass"""
    
clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0
print(f"Writing")
sf.write("temp/low_histogram.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')