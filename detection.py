import numpy as np
import librosa as lr
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import soundfile as sf
import os
import toml

with open('settings_piezo_budgie.toml', 'r') as f:
    config = toml.load(f);
    
plt.close()
class Detect_syllable:
    def __init__(self, config):
        self.audio_path = config["Main"]["audio_path"]
        self.time_short = int(config['Convolve']['time_short'])
        self.time_long = int(config['Convolve']['time_long'])
        [self.y, self.fs] = lr.load(self.audio_path, sr=None)
        
        self.threshold_minimum_time = int(config['Mask']['threshold_minimum_time'])
        self.maxZeros = int(config['Mask']['maxZeros'])
        self.threshold_amplitude_min = int(config['Mask']['threshold_amplitude_min'])
        self.threshold_amplitude_max = int(config['Mask']['threshold_amplitude_max'])
        
        self.threshold_mask_val = int(config['Mask']['threshold_mask_val'])
        self.threshold_maximum_length = int(config['Mask']['threshold_maximum_length'])
        self.extension_length = int(config['Mask']['extension_length'])
        
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
        
        min=0
        max=10.5
        
        maxZeros = int(round((self.maxZeros/1000)*fs))
        threshold_amplitude_min = self.threshold_amplitude_min
        threshold_amplitude_max = self.threshold_amplitude_max
        threshold_maximum_length = self.threshold_maximum_length*fs # seconds
        
        
        extension = self.extension_length
        threshold_mask_val = self.threshold_mask_val
        
    
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
            if np.max(section) < threshold_amplitude_min or np.max(section) > threshold_amplitude_max:
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
                
        
        
        mask = ~np.isnan(masked_short)
        mask = np.where(mask, True, np.nan)
        test_output = y * mask#.astype(np.float32)
        
        start_indices = np.where(~np.isnan(test_output) & ~np.roll(~np.isnan(test_output), 1))[0]
        chunk_lengths = np.diff(np.append(start_indices, len(test_output)))
        
        
        beg_extension_samples = int(extension*fs)
        end_extension_samples = int((extension+0)*fs) # +0.01 for condenser mic
        write_dir = "Audio_Cut/"
        
        for start, length in zip(start_indices, chunk_lengths):
            write_path = os.path.join(write_dir,str(start)+".wav")
            
            # Define Section
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            #end = extension+(start+length-nan_count)
            #start = ((start)/fs)-extension
            beg = int(start-(beg_extension_samples/1000))
            end = int(start+(length-nan_count+end_extension_samples/1000))
            
            print(write_path)
            section = y[beg:end]/32767
            sf.write(write_path, section, fs, format='WAV')
            
        """print(mask)
        print(masked_short)
        print(test_output)"""
        
        # Plotting
        plt.figure(1)
        plt.subplot(3,1,1)
        lr.display.specshow(ft_dB, sr=self.fs, x_axis='time', y_axis='linear');    
        plt.xlim(min,max)
        for start, length in zip(start_indices, chunk_lengths):
            # Define Section
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            
            end = (extension/1000) +(start+length-nan_count)/fs
            start = ((start)/fs)- (extension/1000)
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
            
            end = (extension/1000) +(start+length-nan_count)/fs
            start = ((start)/fs)- (extension/1000)
            plt.axvline(x=start, color = 'y')
            plt.axvline(x=end, color = 'c')
        
        plt.subplot(3,1,3)
        plt.plot(x, y, color='r')
        plt.plot(x,test_output,color='b')
        plt.axhline(y=threshold_amplitude_min, color='k',linestyle='--')
        plt.axhline(y=threshold_amplitude_max, color='k',linestyle='--')
        plt.xlim(min,max)
        #plt.grid(True)
        
        for start, length in zip(start_indices, chunk_lengths):
            # Define Section
            nan_count = np.sum(np.isnan(masked_short[start:start+length]))
            
            end = (extension/1000) +(start+length-nan_count)/fs
            start = ((start)/fs)- (extension/1000)
            plt.axvline(x=start, color = 'y',alpha=0.5)
            plt.axvline(x=end, color = 'c',alpha=0.5)
            
            #plt.axhline(y= 5000, xmin=start, xmax=end, color='c')
        
        """plt.figure(2)
        plt.title("Cut Signal")
        plt.plot(x,test_output,color='b')
        plt.xlim(min,max)
        #plt.xlim(55,65)"""
        
        
        plt.show()
        
Detect_syllable(config).generate_mask()