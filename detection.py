import numpy as np
import librosa as lr
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import soundfile as sf
import os

plt.close()
class Detect_syllable:
    def __init__(self):
        self.audio_path = "Data\c50_brian_filtered_song.wav"
        #self.audio_path = "Data\cut.wav"
        #self.audio_dir_contents = os.listdir(self.audio_dir)
        self.time_long = 330
        self.time_short = 20
        [self.y, self.fs] = lr.load(self.audio_path, sr=None)
        self.threshold_minimum_time = 40
    
    def convolve_fast(self, y, time):
        fs = self.fs
        y = np.abs(y)
        window_size = int((time/1000) * fs)
        weights = np.array((1/window_size) * np.ones(window_size))
        signal_smooth = fftconvolve(y, weights, "same")
        return signal_smooth
        
    def generate_mask(self):
        y = (self.y)*32767
        fs = self.fs
        time_short = self.time_short
        time_long = self.time_long
        threshold_minimum_time = self.threshold_minimum_time
        threshold_minimum_samples = (threshold_minimum_time/1000) * fs
        
        dt = 1/fs
        x = np.arange(0,len(y)/self.fs,dt)
        
        
        #rms = lr.feature.rms(y=y)[0]
        #print(len(rms))
        
        n_fft = 2048
        ft = np.abs(lr.stft(y, n_fft=n_fft,  hop_length=512))
        ft_dB = lr.amplitude_to_db(ft, ref=np.max)
        
        
        
        
        
        
        #min = 55
        #max = 65
        
        min=227
        max=231.5
        
        maxZeros = int(round(fs*0.04,0))
        threshold_amplitude = 10_000
        threshold_maximum_length = 1*fs # seconds
        
        
        extension = 0.02
        threshold_mask_val = -400
        
    
        short_smooth = self.convolve_fast(y, 
                                          time_short)
        long_smooth = self.convolve_fast(y, 
                                         time_long)
        
        
        
        ## Test
        mask = short_smooth > (long_smooth + threshold_mask_val)
        mask = mask.astype(float)
        mask[mask == False] = np.nan
        masked_short = short_smooth * mask.astype(np.float32)
        
        # drop if doesnt meet amplitude threshold
        start_indices = np.where(~np.isnan(masked_short) & ~np.roll(~np.isnan(masked_short), 1))[0]
        chunk_lengths = np.diff(np.append(start_indices, len(masked_short)))
        for start, length in zip(start_indices, chunk_lengths):
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            end = start+length-nan_count;
            section = y[start:end];
            #print(f"Start: {start/30000}, Length: {length}, NaNcount: {nan_count}")
            if len(section) > threshold_maximum_length:
                masked_short[start:start+length] = np.nan
            if np.max(section) < threshold_amplitude:
                #y[start:start + length] = np.nan;
                masked_short[start:start+length] = np.nan
                
        # Merge near sections
        start_indices = np.where(~np.isnan(masked_short) & ~np.roll(~np.isnan(masked_short), 1))[0]
        chunk_lengths = np.diff(np.append(start_indices, len(masked_short)))
        for start, length in zip(start_indices, chunk_lengths):
            # Define Section
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            end = start+length-nan_count;
            section = y[start:end];
            
            print(f"Start: {start/30000}, Length: {length}, NaNcount: {nan_count}")
            if nan_count < maxZeros:
                #plt.plot(masked_short[start:start+length])
                masked_short[start:start+length] = short_smooth[start:start+length];
                #plt.plot(masked_short[start:start+length],alpha=0.1)
                #plt.plot(masked_short[start:start+length]);
                
        # Drop when doesnt exceed length
        """start_indices = np.where(~np.isnan(masked_short) & ~np.roll(~np.isnan(masked_short), 1))[0]
        chunk_lengths = np.diff(np.append(start_indices, len(masked_short)))
        for start, length in zip(start_indices, chunk_lengths):
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            end = start+length-nan_count;
            section = y[start:end];
            #print(f"Start: {start/30000}, Length: {length}, NaNcount: {nan_count}")
            if len(section) < threshold_minimum_samples:
                masked_short[start:start + length] = np.nan;"""
        #else:
        #    masked_short[start:end-nan_count] = short_smooth[start:end-nan_count];"""
            
        # This section works
        """for start, length in zip(start_indices, chunk_lengths):
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            end = start+length-nan_count;
            section = y[start:end];
            print(f"Start: {start/30000}, Length: {length}, NaNcount: {nan_count}")
            if len(section) < threshold_minimum_samples:
                masked_short[start:start + length] = np.nan;"""
                
                
                
                
                
        # This does not        
        """print(f"Threshold minimum samples : {threshold_minimum_samples}")
        print(f"maxZeros: {maxZeros}")
        print(f"Sections before merge: {len(start_indices)}")
        for start, length in zip(start_indices, chunk_lengths):
            # Define Section
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            end = start+length-nan_count;
            #print(start, length)
            #print(f"Start: {start/30000}, Length: {length}, NaNcount: {nan_count}")
            section = y[start:end];
            
            if nan_count < maxZeros:
                masked_short[start:end] = short_smooth[start:end];
            else:
                masked_short[start:end-nan_count] = y[start:end-nan_count];
                
                
        start_indices = np.where(~np.isnan(masked_short) & ~np.roll(~np.isnan(masked_short), 1))[0]
        chunk_lengths = np.diff(np.append(start_indices, len(masked_short)))
        print(f"Sections after merge: {len(start_indices)}")"""
        
        
        mask = ~np.isnan(masked_short)
        mask = np.where(mask, True, np.nan)
        test_output = y * mask#.astype(np.float32)
        
        start_indices = np.where(~np.isnan(test_output) & ~np.roll(~np.isnan(test_output), 1))[0]
        chunk_lengths = np.diff(np.append(start_indices, len(test_output)))
        
        
        beg_extension_samples = int(extension*fs)
        end_extension_samples = int((extension+0.01)*fs)
        write_dir = "Audio_Cut/"
        
        for start, length in zip(start_indices, chunk_lengths):
            write_path = os.path.join(write_dir,str(start)+".wav")
            
            # Define Section
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            #end = extension+(start+length-nan_count)
            #start = ((start)/fs)-extension
            beg = start-beg_extension_samples
            end = start+length-nan_count+end_extension_samples
            
            print(write_path)
            section = y[beg:end]/32767
            sf.write(write_path, section, fs, format='WAV')
            #print(f"{write_path}")
            
            
            
            
            
            
            #plt.axvline(x=start, color = 'y')
            #plt.axvline(x=end, color = 'c')
            
        print(mask)
        print(masked_short)
        print(test_output)
        
        """# plotting
        plt.figure(1)
        plt.subplot(3,1,1)
        lr.display.specshow(ft_dB, sr=self.fs, x_axis='time', y_axis='linear');    
        plt.xlim(min,max)
        for start, length in zip(start_indices, chunk_lengths):
            # Define Section
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            
            end = extension+(start+length-nan_count)/fs
            start = ((start)/fs)-extension
            plt.axvline(x=start, color = 'y')
            plt.axvline(x=end, color = 'c')
        
        plt.subplot(3,1,2)
        plt.plot(x,short_smooth, color='b')
        plt.plot(x,long_smooth, color='g')
        plt.plot(x,masked_short, color='r')
        plt.xlim(min,max)
        #plt.grid(True)
        
        for start, length in zip(start_indices, chunk_lengths):
            # Define Section
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            
            end = extension+(start+length-nan_count)/fs
            start = ((start)/fs)-extension
            plt.axvline(x=start, color = 'y')
            plt.axvline(x=end, color = 'c')
        
        plt.subplot(3,1,3)
        plt.plot(x, y, color='r')
        plt.plot(x,test_output,color='b')
        plt.axhline(y=threshold_amplitude, color='k',linestyle='--')
        plt.xlim(min,max)
        #plt.grid(True)
        
        for start, length in zip(start_indices, chunk_lengths):
            # Define Section
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            
            end = extension+(start+length-nan_count)/fs
            start = ((start)/fs)-extension
            plt.axvline(x=start, color = 'y',alpha=0.5)
            plt.axvline(x=end, color = 'c',alpha=0.5)
            
            #plt.axhline(y= 5000, xmin=start, xmax=end, color='c')
        
        plt.figure(2)
        plt.title("Cut Signal")
        plt.plot(x,test_output,color='b')
        plt.xlim(min,max)
        #plt.xlim(55,65)
        
        
        plt.show()"""
        
Detect_syllable().generate_mask()