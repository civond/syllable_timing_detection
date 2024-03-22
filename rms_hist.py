import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal


data = "Audio_Filtered/Ti81end_filtered_0.flac"
[audio, fs] = lr.load(data, sr=None)
audio = audio*32767

audio_rms = lr.feature.rms(y=audio, hop_length=256)[0]
threshold = 80  
interpolated_rms = np.interp(np.arange(len(audio)), 
                             np.arange(len(audio_rms))*256, 
                             audio_rms)
temp = np.where(interpolated_rms >= threshold, interpolated_rms, np.nan)
thresholded_audio = np.where(interpolated_rms >= threshold, audio, np.nan)

"""duration = len(audio)/fs
dt = 1/fs
t = np.arange(0,duration,dt)

n_fft = 2048
plt.figure(1, figsize=(10,7))
plt.subplot(3,1,1)
ft = np.abs(lr.stft(audio, n_fft=n_fft,  hop_length=256))
ft_dB = lr.amplitude_to_db(ft, ref=np.max)
lr.display.specshow(ft_dB, sr=fs, x_axis='time', y_axis='linear')
plt.title(f"Ti81end_filtered_0.flac")
plt.xlim(250,300)

plt.subplot(3,1,2)
plt.plot(t,audio, color='b')
plt.plot(t,thresholded_audio, color='r')
plt.grid(True)
plt.title("RMS-Thresholded Audio")
plt.xlim(250,300)

plt.subplot(3,1,3)
plt.plot(t, interpolated_rms, color='b')
plt.plot(t, temp, color='r')
plt.grid(True)
plt.title("RMS")
plt.xlim(250,300)

plt.tight_layout()
plt.show()"""

clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0

"""print(f"Writing")
sf.write("temp/Ti831end_RMS_thresholded.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')
"""
start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]

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

    plt.figure(figsize=(8,3))
    """plt.subplot(2,1,1)
    plt.title(f"Start: {round(start/fs,2)}")
    plt.hist(np.abs(section**2), bins=200, log=True)"""
    #print(section)
    #plt.subplot(2,2,2)
    #plt.hist(np.abs(section), bins=200, log=False)

    #plt.subplot(2,1,2)
    plt.bar(bins[:-1], hist_db, width=np.diff(bins))
    plt.scatter(bins[:25], hist_db[:25], color='b', s=2)
    plt.scatter(bins[25:-1], hist_db[25:], color='r', s=2)
    plt.title(f"Low: {sum_low}, High: {sum_high}, Ratio: {round(sum_high/sum_low,2)}")
    plt.xlim(0,bins[-2])
    plt.tight_layout()
    plt.show()

    if sum_low > 1.5*sum_high:
        thresholded_audio[start:end] = np.nan
    else:
        print("Pass")
        pass

clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0
print(f"Writing")
sf.write("temp/Ti831end_histogram.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')

input('stop')

merge_time = 2000
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
sf.write("temp/Ti81_master_merge.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')