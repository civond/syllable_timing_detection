import librosa as lr
import scipy.signal as signal
import toml
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

class Filter_signal:
    def __init__(self, config, show_freq_resp = False):
        # Main parameters
        self.audio_path = config["Main"]["audio_path"]
        self.write_dir = config["Main"]["filtered_audio_directory"]
        self.audio_name = self.audio_path.split('/')[-1].split('.')[0] + '_filtered.flac'
        self.max_duration = config["Filter"]["max_length"]
        self.show_freq_resp = show_freq_resp
        
        # Filter parameters
        self.cutoff_low = config["Filter"]["cutoff_low"]
        self.cutoff_high = config["Filter"]["cutoff_high"]
        self.minimum_attenuation = config["Filter"]["minimum_attenuation"]
        self.order = config["Filter"]["order"]
        
        # Load audio
        [self.audio, self.fs] = lr.load(self.audio_path, sr=None)
        print(f"Filter fs: {self.fs}")
        self.max_duration_samples = int(self.max_duration*3600*self.fs)
    
    # Split the array into managable chunks
    def create_array(self, audio):
        max_duration_samples = self.max_duration_samples

        if len(audio) < max_duration_samples:
            return [audio]
        else:
            temp_list = []
            [iter, rem] = np.divmod(len(audio), max_duration_samples)
            for i in range(iter):
                temp_list.append(audio[(i*max_duration_samples):(i*max_duration_samples+max_duration_samples)])
            if rem != 0:
                temp_list.append(audio[iter*max_duration_samples:-1])
            
            return temp_list
    
    # Filter each element in array
    def filter_array(self):
        audio = self.audio;
        audio_path = self.audio_path;
        write_dir = self.write_dir;
        temp = self.create_array(audio);
        
        # Checks if write directory exists
        if os.path.isdir(write_dir) == False:
            print(f"Write directory not found, creating: {write_dir}");
            os.mkdir(write_dir);
        else: 
            pass
        
        # Define filter
        [b, a] = signal.cheby2(self.order, 
                               self.minimum_attenuation, 
                               [self.cutoff_low, self.cutoff_high], 
                               fs=self.fs, 
                               btype="bandpass")
            
        # Loop over each array in temp
        for index, array in enumerate(temp):
            audio_name = audio_path.split('/')[1].split('.')[0] + '_filtered_' + str(index*self.max_duration_samples) + '.flac'
            destination_path = os.path.join(write_dir,audio_name)
            
            if os.path.exists(destination_path):
                print(f"\t{destination_path} already exists. Skipping.")
                pass
            else:
                # Filter and scale signal
                y_filtered = signal.filtfilt(b,a,array)
                #scaled = np.int16(y_filtered / np.max(np.abs(y_filtered)) * 32767);
                
                # Write audio
                print(f"\t\tWriting: {destination_path}")
                sf.write(destination_path, 
                         y_filtered, 
                         self.fs, 
                         format='FLAC')        

        if self.show_freq_resp == True:
            [w, h] = signal.freqz(b,a)
            freq_hz = w * (self.fs / (2*np.pi))
            
            plt.figure(1)
            plt.plot(freq_hz, 
                     20*np.log10(abs(h)),
                     color='b',
                     linewidth=1)
            plt.axhline(y = -self.minimum_attenuation, xmin = 0, xmax = freq_hz[-1],
                        color = 'r', linestyle = '--', linewidth=1) 
            
            plt.axvline(x = self.cutoff_low, 
                        ymin = -self.minimum_attenuation - 20, ymax = 0+10,
                        color = 'g', linestyle = '--', linewidth=1)
            
            plt.axvline(x = self.cutoff_high, 
                        ymin = -self.minimum_attenuation - 20, ymax = 0+10,
                        color = 'm', linestyle = '--', linewidth=1)
            plt.legend(['Freq. Response', 
                        'Min. Atten.',
                        'Wn_low',
                        'Wn_high'])
            
            plt.title("Cheby2 Bandpass Filter")
            plt.xlabel("Hz")
            plt.ylabel("Attenuation (dB)")
            plt.xlim(0, self.cutoff_high+1000)
            plt.ylim(-self.minimum_attenuation - 30, 10)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

class Generate_mask:
    def __init__(self, filtered_audio_path, config, show_mask = False):
        self.show_mask = show_mask
        
        self.filtered_audio_path = filtered_audio_path
        self.time_short = int(config['Convolve']['time_short'])
        self.time_long = int(config['Convolve']['time_long'])
        self.time_smooth = int(config['Convolve']['time_smooth'])
        
        self.threshold_mask_val = config['Mask']['threshold_mask_val']
        self.threshold_amplitude_max = config['Mask']['threshold_amplitude_max']
        
        [self.y, self.fs] = lr.load(self.filtered_audio_path, sr=None)
        self.y = self.y*32767
        self.y_abs = abs(self.y)
        
        self.threshold_minimum_time = config['Mask']['threshold_minimum_time']
        self.threshold_samples = (self.threshold_minimum_time / 1000) * self.fs
        
        self.threshold_overlap_min = config['Mask']['threshold_overlap_min']
        self.threshold_overlap_min_samples = (self.threshold_overlap_min / 1000) * self.fs
        
    def convolve_fast(self, time):
        print(f"\tConvolving audio with window of length: {time} ms.")
        y_abs = self.y_abs
        samples = int((time/(10**3)) * self.fs); # Time must be in ms
        weights = (1/samples) * np.ones(samples)
        smooth = signal.fftconvolve(y_abs,weights,'same')
        return smooth
    
    def filter_overlap(self):
        print(f"\nGenerating Mask for {self.filtered_audio_path}")
        short_smooth = self.convolve_fast(self.time_short)
        long_smooth = self.convolve_fast(self.time_long)
        temp = short_smooth - long_smooth
        temp_mask = temp < self.threshold_mask_val
        temp[temp_mask] = np.nan
        
        
        smooth = self.convolve_fast(self.time_smooth)
        copyy = smooth.copy()
        thresh = 30
        temp_mask = smooth <= thresh  # Create a mask for values under or equal to the threshold
        smooth[temp_mask] = np.nan
        
        """thresh = 300
        temp_mask = smooth > thresh
        print(temp_mask)
        smooth[temp_mask] = np.nan"""

        duration = len(short_smooth)/self.fs
        dt = 1/self.fs
        t = np.arange(0,duration,dt)
        
        """plt.figure(2)
        plt.plot(t, copyy, color='b')
        plt.plot(t, smooth, color='r')
        plt.grid(True)
        #plt.xlim(0,t[-1])
        plt.xlim(35,55)"""

        #self.show_mask = True
        if self.show_mask == True:
            duration = len(short_smooth)/self.fs
            dt = 1/self.fs
            t = np.arange(0,duration,dt)
        
            plt.figure(1)
            plt.plot(t,short_smooth, color='g')
            plt.plot(t,long_smooth, color='b')
            plt.plot(t,temp+long_smooth, color='r')
            plt.grid("True")
            #plt.xlim(0,t[-1])
            plt.xlim(35,55)
            plt.ylabel('Moving Average Magnitude')
            plt.xlabel('Time (s)')
            plt.legend(['Short (50ms)', 'Long (250ms)', 'Overlap'],loc='best')

            plt.title('Smoothing Filter Overlaps')
            plt.tight_layout()
            plt.show()
    
        return temp, smooth

    def remove_short_overlaps(self, overlap):
        start_indices = np.where(~np.isnan(overlap) & ~np.roll(~np.isnan(overlap), 1))[0]
        chunk_lengths = np.diff(np.append(start_indices, len(overlap)))
        for start, length in zip(start_indices, chunk_lengths):
            nan_count = np.sum(np.isnan(overlap[start:start+length]))
            if (length-nan_count) < self.threshold_overlap_min_samples:
                overlap[start:start + length] = np.nan
    
        return overlap
    
    def apply_mask(self):
        overlap = self.filter_overlap()[0]
        overlap = self.remove_short_overlaps(overlap)
        
        start_indices = np.where(~np.isnan(overlap) & ~np.roll(~np.isnan(overlap), 1))[0];
        chunk_lengths = np.diff(np.append(start_indices, len(overlap)))
        for start, length in zip(start_indices, chunk_lengths):
            end = start+length
            nan_count = np.sum(np.isnan(overlap[start:start+length]))
            
            if nan_count < self.threshold_samples:
                overlap[start:end] = self.y[start:end]
            else:
                overlap[start:end-nan_count] = self.y[start:end-nan_count]
                
            if np.max(self.y[start:end]) > self.threshold_amplitude_max:
                overlap[start:end] = np.nan
        #test        
        overlap2 = self.filter_overlap()[1]
        overlap2 = self.remove_short_overlaps(overlap2)
        start_indices = np.where(~np.isnan(overlap2) & ~np.roll(~np.isnan(overlap2), 1))[0];
        chunk_lengths = np.diff(np.append(start_indices, len(overlap2)))
        for start, length in zip(start_indices, chunk_lengths):
            end = start+length
            nan_count = np.sum(np.isnan(overlap2[start:start+length]))
            
            if nan_count < self.threshold_samples:
                overlap2[start:end] = self.y[start:end]
            else:
                overlap2[start:end-nan_count] = self.y[start:end-nan_count]
            if np.max(self.y[start:end]) > self.threshold_amplitude_max:
                overlap2[start:end] = np.nan
        #  
        """plt.figure(2)
        duration = len(overlap)/self.fs
        dt = 1/self.fs
        t = np.arange(0,duration,dt)
        plt.subplot(2,1,1)
        plt.plot(t, self.y, color='b')
        plt.plot(t, overlap, color='r')
        plt.xlim(170, 230)
        plt.grid(True)
        
        plt.subplot(2,1,2)
        plt.plot(t, self.y, color='b')
        plt.plot(t, overlap2, color='k')
        plt.grid(True)
        plt.xlim(170, 230)
    
        plt.show()
        """
        return overlap2
        
class Get_valid_regions:
    def __init__(self, cut_signal, filtered_audio_path, config):
        self.filtered_audio_path = filtered_audio_path
        self.cut_signal = cut_signal
        [self.y, self.fs] = lr.load(self.filtered_audio_path, sr=None)
        self.y = 32767*self.y
        print(self.fs)
        
    def section_fft(self):
        cut_signal = self.cut_signal
        start_indices = np.where(~np.isnan(cut_signal) & ~np.roll(~np.isnan(cut_signal), 1))[0];
        chunk_lengths = np.diff(np.append(start_indices, len(cut_signal)))
        
        for start, length in zip(start_indices, chunk_lengths):
            nan_count = np.sum(np.isnan(cut_signal[start:start+length]))
            end = start+length-nan_count
            section = cut_signal[start:end]
            
            n_fft = 2048
            ft = np.abs(lr.stft(section, n_fft=n_fft,  hop_length=256))
            ft_dB = lr.amplitude_to_db(ft, ref=np.max)
            
            plt.figure(1)
            plt.subplot(2,1,1)
            [s, rem] = np.divmod(start, self.fs)
            plt.title(f"Time: {s}.{np.round(rem,2)} s, Sample:{start}, len={len(section)/self.fs}")
            lr.display.specshow(ft_dB, sr=self.fs, x_axis='time', y_axis='linear')
            
            plt.subplot(2,1,2)
            duration = len(section)/self.fs
            dt = 1/self.fs
            t = np.arange(0,duration,dt)
            plt.plot(t, section, color='b')
            plt.xlim(0,t[-1])   
            plt.show()

    
    """def fourier_spec(self, cut_audio)
    start_indices = np.where(~np.isnan(overlap) & ~np.roll(~np.isnan(overlap), 1))[0];
        chunk_lengths = np.diff(np.append(start_indices, len(overlap)))
        for start, length in zip(start_indices, chunk_lengths):
            end = start+length
            nan_count = np.sum(np.isnan(overlap[start:start+length]))"""

if __name__ == "__main__":
    with open('settings.toml', 'r') as f:
        config = toml.load(f)
        
    Filter_signal(config, show_freq_resp= True).filter_array()
    filtered_audio_directory = config["Main"]["filtered_audio_directory"]
    
    for audio_file in os.listdir(filtered_audio_directory):
        filtered_audio_path = os.path.join(filtered_audio_directory, audio_file)
        cut_signal = Generate_mask(filtered_audio_path, config).apply_mask()
        cut_signal = Get_valid_regions(cut_signal, filtered_audio_path, config).section_fft()
        #cut_signal = Generate_mask(filtered_audio_path, config).filter_overlap()