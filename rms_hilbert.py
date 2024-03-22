import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
from scipy.interpolate import UnivariateSpline
import emd


data = "Audio_Filtered/Ti81end_filtered_0.flac"
[audio, fs] = lr.load(data, sr=None)
audio = audio*32767

audio_rms = lr.feature.rms(y=audio, hop_length=256)[0]


threshold = 80  

interpolated_rms = np.interp(np.arange(len(audio)), np.arange(len(audio_rms))*256, audio_rms)

# Apply the threshold to the audio signal
thresholded_audio = np.where(interpolated_rms >= threshold, audio, np.nan)

start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
merge_samples = (800/1000)*fs

"""for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]

    if nan_count < merge_samples:
        thresholded_audio[start:start+length] = audio[start:start+length]
    else:
        pass"""

"""clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0

print(f"Writing")
sf.write("IMF_corr.flac", 
                    clone/32767, 
                    fs, 
                    format='FLAC')
input('stop:')"""

for start, length in zip(start_indices, chunk_lengths):
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]

    start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0];
    chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
    for start, length in zip(start_indices, chunk_lengths):
        nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
        end = start+length-nan_count
        section = thresholded_audio[start:end]

        analytic_signal = signal.hilbert(section)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) /
                                (2.0*np.pi) * fs)

        plt.subplot(1,2,1)
        plt.plot(section)
        plt.subplot(1,2,2)
        plt.plot(section, label='Original Signal')
        plt.plot(amplitude_envelope, label='Envelope')
        #plt.show()
        plt.clf()
        try:
            imf = emd.sift.mask_sift(section, max_imfs=5)
            num_imfs = imf.shape[1]
            imf1 = imf[:,1]
            imf2 = imf[:,2]
            imf3 = imf[:,3]

        #imf1_rms = lr.feature.rms(y=imf1, hop_length=256)[0]
        #imf1_zc = lr.feature.zero_crossing_rate(y=imf1, hop_length=256)[0]

        except IndexError:
            imf1=0
            imf2=0
            imf3=0
            imf1_rms = 0
            imf_zc = 0

        plt.figure(1, figsize=(10,10))
        plt.subplot(2,2,1)
        n_fft = 2048
        ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
        ft_dB = lr.amplitude_to_db(ft, ref=np.max)
        lr.display.specshow(ft_dB, sr=fs, x_axis='time', y_axis='linear')
        plt.title(f"Start: {round(start/fs, 2)}")

        plt.subplot(2,2,2)
        plt.plot(imf1+100, color='b')
        plt.plot(imf2, color='g')
        plt.plot(imf3-100, color = 'r')

        try:
            #cross_corr = np.correlate(imf1, imf2)
            #norm_cross_corr = np.correlate(imf1, imf2, mode='same') / (np.linalg.norm(imf1) * np.linalg.norm(imf2))
            norm_cross_corr = signal.correlate(imf1, imf2, mode='same') / (np.linalg.norm(imf1) * np.linalg.norm(imf2))
            norm_cross_corr2 = signal.correlate(imf1, imf3, mode='same') / (np.linalg.norm(imf1) * np.linalg.norm(imf3))
        except ValueError:
            cross_corr = 0
            norm_cross_corr = 0
        #print(norm_cross_corr)
        plt.subplot(2,2,3)
        plt.title(f"IMF2 vs. IMF3 {round(np.max(norm_cross_corr2),2)}")
        plt.plot(norm_cross_corr2)

        plt.subplot(2,2,4)
        plt.plot(norm_cross_corr)
        plt.title(f"IMF1 vs. IMF3 {round(np.max(norm_cross_corr),2)}")
        plt.tight_layout()
        
        plt.show()

        if np.max(norm_cross_corr) == 0 or np.max(norm_cross_corr) > 0.20:
            thresholded_audio[start:end] = np.nan
        
    clone = thresholded_audio.copy()
    clone[np.isnan(clone)] = 0

    print(f"Writing")
    sf.write("IMF_corr.flac", 
                        clone/32767, 
                        fs, 
                        format='FLAC')
    input('stop: ')


    """ plt.subplot(2,3,6)
    try:
        cross_corr = np.correlate(imf2, imf3)
        norm_cross_corr = np.correlate(imf2, imf3, mode='same') / (np.linalg.norm(imf2) * np.linalg.norm(imf3))
    except ValueError:
        cross_corr = 0
        norm_cross_corr = 0
    plt.plot(norm_cross_corr)
    plt.title(f"IMF2 vs. IMF3 {round(np.max(norm_cross_corr),2)}")"""
    #plt.axhline(y=avg_zc, color='r', linestyle='--', label='avg_zc')
    #plt.plot(imf1_zc, color='b')
    #plt.ylim(0,0.2)
    #plt.title("IMF1 ZC")
    #plt.show()

    #plt.clf()