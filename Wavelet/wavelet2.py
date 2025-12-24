import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from scipy import signal
import pycwt as wavelet
from pycwt import cwt, significance
import warnings
warnings.filterwarnings('ignore')

class EEGWaveletAnalyzer:
    """
    EEG Wavelet Transform Analyzer using PyCWT
    
    This class handles the wavelet analysis of EEG data including:
    - Continuous Wavelet Transform (CWT) using Morlet wavelet
    - Scale to frequency conversion
    - Cone of Influence (COI) identification and exclusion
    - Frequency band analysis
    """
    
    def __init__(self, sampling_rate=250, mother='morlet'):
        """
        Initialize the wavelet analyzer
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate of EEG data (default: 250 Hz)
        mother : str
            Mother wavelet type (default: 'morlet')
        """
        self.fs = sampling_rate
        self.dt = 1.0 / sampling_rate  # dt = 1/fs as suggested by teammate
        self.mother = mother
        
        # Standard EEG frequency bands
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Wavelet parameters as suggested by teammate
        self.dj = 0.25  # Scale resolution
        self.s0 = self.dt  # Smallest scale (dt instead of 2*dt)
        self.J = int(7 / self.dj)  # J = 7/dj as per teammate's method
        
    def load_subject_data(self, subject_file):
        """
        Load EEG data for a single subject
        
        Parameters:
        -----------
        subject_file : str
            Path to the subject's pickle file
            
        Returns:
        --------
        data : dict or np.array
            Loaded EEG data
        """
        try:
            with open(subject_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Successfully loaded data from {subject_file}")
            
            # Print data structure information
            if isinstance(data, dict):
                print(f"Data type: Dictionary with keys: {list(data.keys())}")
                for key, value in data.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {type(value)} - Shape: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
            elif hasattr(data, 'shape'):
                print(f"Data type: Array - Shape: {data.shape}, dtype: {data.dtype}")
            else:
                print(f"Data type: {type(data)}")
                
            return data
        except Exception as e:
            print(f"Error loading {subject_file}: {e}")
            return None
    
    def extract_eeg_signal(self, data, video_end_time=None):
        """
        Extract EEG signal from loaded data using [0,0,:] indexing
        Takes first video, first channel for 30 seconds as suggested by teammate
        
        Parameters:
        -----------
        data : dict or np.array
            Loaded EEG data
        video_end_time : float or None
            Time when video ends (if None, use entire signal or create synthetic)
            
        Returns:
        --------
        signal_data : np.array
            Extracted EEG signal (first video, first channel)
        """
        try:
            if isinstance(data, dict):
                # Common possible keys for EEG data
                possible_keys = ['eeg', 'data', 'signal', 'EEG', 'channels', 'raw_data']
                
                eeg_data = None
                for key in possible_keys:
                    if key in data:
                        eeg_data = data[key]
                        print(f"Found EEG data under key: '{key}'")
                        break
                
                if eeg_data is None:
                    # If no standard key found, look for array-like values
                    for key, value in data.items():
                        if hasattr(value, 'shape') and len(value.shape) >= 1:
                            eeg_data = value
                            print(f"Using data from key: '{key}' as EEG signal")
                            break
                
                if eeg_data is None:
                    print("Could not find EEG data in dictionary")
                    return self._create_synthetic_signal()
                    
            elif hasattr(data, 'shape'):
                eeg_data = data
                print("Using array data as EEG signal")
            else:
                print("Data format not recognized")
                return self._create_synthetic_signal()
            
            # Convert to numpy array if needed
            eeg_array = np.array(eeg_data)
            print(f"EEG array shape: {eeg_array.shape}")
            
            # Use teammate's indexing method [0,0,:] for first video, first channel
            if len(eeg_array.shape) >= 3:
                # 3D array: [video, channel, time] - use [0,0,:]
                signal_data = eeg_array[0, 0, :]
                print(f"Using [0,0,:] indexing - first video, first channel")
                print(f"Extracted signal shape: {signal_data.shape}")
            elif len(eeg_array.shape) == 2:
                # 2D array - assume first dimension is what we want
                signal_data = eeg_array[0, :]
                print(f"2D array - using first row [0,:]")
            elif len(eeg_array.shape) == 1:
                # 1D array - use as is
                signal_data = eeg_array
                print(f"1D array - using as is")
            else:
                print(f"Unexpected array shape: {eeg_array.shape}")
                signal_data = eeg_array.flatten()
                print(f"Flattened to 1D")
            
            # Extract 30 seconds (7500 samples at 250Hz) from the beginning
            samples_30s = int(30 * self.fs)  # 30 seconds * 250 Hz = 7500 samples
            if len(signal_data) > samples_30s:
                signal_data = signal_data[:samples_30s]
                print(f"Extracted first 30 seconds: {samples_30s} samples")
            else:
                print(f"Signal is shorter than 30 seconds: {len(signal_data)} samples ({len(signal_data)/self.fs:.2f}s)")
            
            print(f"Final signal shape: {signal_data.shape}")
            return signal_data
            
        except Exception as e:
            print(f"Error extracting EEG signal: {e}")
            print(f"Falling back to synthetic signal")
            return self._create_synthetic_signal()
    
    def _create_synthetic_signal(self, duration=30):
        """
        Create synthetic EEG signal for testing purposes
        
        Parameters:
        -----------
        duration : int
            Duration in seconds
            
        Returns:
        --------
        signal_data : np.array
            Synthetic EEG signal
        """
        print(f"Creating synthetic EEG signal ({duration} seconds)")
        time = np.linspace(0, duration, int(duration * self.fs))
        
        # Synthetic EEG with multiple frequency components
        np.random.seed(42)
        signal_data = (
            2 * np.sin(2 * np.pi * 10 * time) +  # Alpha (10 Hz)
            1.5 * np.sin(2 * np.pi * 6 * time) +  # Theta (6 Hz)
            1 * np.sin(2 * np.pi * 20 * time) +   # Beta (20 Hz)
            0.8 * np.sin(2 * np.pi * 2 * time) +  # Delta (2 Hz)
            0.6 * np.sin(2 * np.pi * 35 * time) + # Gamma (35 Hz)
            0.5 * np.random.randn(len(time))       # Noise
        )
        
        return signal_data
    
    def extract_post_video_segment(self, eeg_signal, video_end_time, duration=30):
        """
        Extract 30-second segment after video ends
        
        Parameters:
        -----------
        eeg_signal : np.array
            EEG signal data
        video_end_time : float
            Time when video ends (in seconds)
        duration : int
            Duration to extract (default: 30 seconds)
            
        Returns:
        --------
        segment : np.array
            Extracted EEG segment
        """
        start_sample = int(video_end_time * self.fs)
        end_sample = int((video_end_time + duration) * self.fs)
        
        if end_sample > len(eeg_signal):
            print(f"Warning: Requested segment extends beyond signal length")
            end_sample = len(eeg_signal)
            
        return eeg_signal[start_sample:end_sample]
    
    def compute_cwt(self, signal_data):
        """
        Compute Continuous Wavelet Transform using PyCWT
        
        Parameters:
        -----------
        signal_data : np.array
            1D EEG signal data
            
        Returns:
        --------
        wave : np.array
            Wavelet transform coefficients
        scales : np.array
            Scales used in the transform
        freqs : np.array
            Corresponding frequencies
        coi : np.array
            Cone of influence
        """
        # Normalize the signal
        signal_normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
        
        # J is already set as 7/dj in __init__
        # No need to calculate dynamically
        
        # Compute CWT
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
            signal_normalized, 
            self.dt, 
            dj=self.dj, 
            s0=self.s0, 
            J=self.J, 
            wavelet=self.mother
        )
        
        print(f"CWT computed successfully:")
        print(f"  Signal length: {len(signal_normalized)}")
        print(f"  Number of scales: {len(scales)}")
        print(f"  Frequency range: {freqs.min():.2f} - {freqs.max():.2f} Hz")
        print(f"  COI shape: {coi.shape}")
        
        return wave, scales, freqs, coi
    
    def scale_to_frequency_conversion(self, scales):
        """
        Convert scales to frequencies for Morlet wavelet
        
        Parameters:
        -----------
        scales : np.array
            Wavelet scales
            
        Returns:
        --------
        frequencies : np.array
            Corresponding frequencies in Hz
        """
        # For Morlet wavelet, the relationship is: f = f0 / (scale * dt)
        # where f0 is the central frequency of the Morlet wavelet (typically 6)
        f0 = 6.0  # Central frequency for Morlet wavelet
        frequencies = f0 / (scales * self.dt)
        
        print(f"Scale to frequency conversion:")
        print(f"  Scale range: {scales.min():.2f} - {scales.max():.2f}")
        print(f"  Frequency range: {frequencies.min():.2f} - {frequencies.max():.2f} Hz")
        
        return frequencies
    
    def identify_coi_matrix(self, wave, coi, freqs):
        """
        Identify which parts of the wavelet matrix correspond to COI
        
        Parameters:
        -----------
        wave : np.array
            Wavelet transform coefficients
        coi : np.array
            Cone of influence
        freqs : np.array
            Frequencies corresponding to scales
            
        Returns:
        --------
        coi_mask : np.array
            Boolean mask indicating COI regions (True = inside COI, False = outside)
        valid_region : tuple
            (freq_start_idx, freq_end_idx, time_start_idx, time_end_idx) for valid data
        """
        # Create time array
        time = np.arange(wave.shape[1]) * self.dt
        
        # Create frequency-time mesh
        freq_mesh, time_mesh = np.meshgrid(freqs, time, indexing='ij')
        
        # COI mask: True where frequency > COI frequency (invalid region)
        coi_freq = 1.0 / coi  # Convert COI to frequency
        coi_mask = freq_mesh > coi_freq[np.newaxis, :]
        
        # Find valid region (excluding COI)
        valid_mask = ~coi_mask
        valid_freq_indices = np.where(np.any(valid_mask, axis=1))[0]
        valid_time_indices = np.where(np.any(valid_mask, axis=0))[0]
        
        if len(valid_freq_indices) > 0 and len(valid_time_indices) > 0:
            valid_region = (
                valid_freq_indices[0], valid_freq_indices[-1],
                valid_time_indices[0], valid_time_indices[-1]
            )
        else:
            valid_region = (0, len(freqs)-1, 0, wave.shape[1]-1)
        
        print(f"COI Analysis:")
        print(f"  Total matrix size: {wave.shape}")
        print(f"  COI excluded region: {np.sum(coi_mask)} points ({np.sum(coi_mask)/wave.size*100:.1f}%)")
        print(f"  Valid region - Freq indices: {valid_region[0]}-{valid_region[1]}")
        print(f"  Valid region - Time indices: {valid_region[2]}-{valid_region[3]}")
        print(f"  Valid frequency range: {freqs[valid_region[0]]:.2f} - {freqs[valid_region[1]]:.2f} Hz")
        
        return coi_mask, valid_region
    
    def compute_frequency_band_power(self, wave, freqs, coi_mask, exclude_coi=True):
        """
        Compute power in different frequency bands
        
        Parameters:
        -----------
        wave : np.array
            Wavelet transform coefficients
        freqs : np.array
            Frequencies corresponding to scales
        coi_mask : np.array
            Boolean mask for COI regions
        exclude_coi : bool
            Whether to exclude COI regions from calculation
            
        Returns:
        --------
        band_powers : dict
            Power in each frequency band
        """
        # Compute power (squared magnitude)
        power = np.abs(wave) ** 2
        
        # Apply COI mask if requested
        if exclude_coi:
            power_masked = np.where(coi_mask, np.nan, power)
        else:
            power_masked = power
        
        band_powers = {}
        
        for band_name, (f_low, f_high) in self.freq_bands.items():
            # Find frequency indices for this band
            freq_mask = (freqs >= f_low) & (freqs <= f_high)
            
            if np.any(freq_mask):
                # Sum power across frequencies in this band
                band_power = np.nansum(power_masked[freq_mask, :], axis=0)
                
                # Calculate mean power across time
                mean_power = np.nanmean(band_power)
                
                band_powers[band_name] = {
                    'time_series': band_power,
                    'mean_power': mean_power,
                    'freq_range': (f_low, f_high),
                    'n_freq_points': np.sum(freq_mask)
                }
                
                print(f"{band_name.capitalize()} band ({f_low}-{f_high} Hz):")
                print(f"  Frequency points: {np.sum(freq_mask)}")
                print(f"  Mean power: {mean_power:.2e}")
            else:
                print(f"Warning: No frequency points found for {band_name} band ({f_low}-{f_high} Hz)")
                band_powers[band_name] = None
        
        return band_powers
    
    def plot_wavelet_spectrogram(self, wave, freqs, coi, time_duration=30, 
                                save_path=None, subject_id="unknown"):
        """
        Plot wavelet spectrogram with COI
        
        Parameters:
        -----------
        wave : np.array
            Wavelet transform coefficients
        freqs : np.array
            Frequencies
        coi : np.array
            Cone of influence
        time_duration : float
            Duration of the signal in seconds
        save_path : str
            Path to save the plot
        subject_id : str
            Subject identifier for the plot title
        """
        # Create time array
        time = np.linspace(0, time_duration, wave.shape[1])
        
        # Compute power
        power = np.abs(wave) ** 2
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot spectrogram
        im = ax.contourf(time, freqs, np.log10(power), levels=50, cmap='jet', extend='both')
        
        # Plot COI
        coi_freq = 1.0 / coi
        ax.plot(time, coi_freq, 'k--', linewidth=2, label='COI')
        ax.fill_between(time, coi_freq, freqs.max(), alpha=0.3, color='white', 
                       hatch='///', label='COI excluded region')
        
        # Formatting
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title(f'Wavelet Spectrogram - Subject {subject_id}\n30-second Post-Video Analysis', 
                    fontsize=14)
        ax.set_yscale('log')
        ax.set_ylim([freqs.min(), freqs.max()])
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Log10(Power)', fontsize=12)
        
        # Add frequency band lines
        for band_name, (f_low, f_high) in self.freq_bands.items():
            if f_low >= freqs.min() and f_high <= freqs.max():
                ax.axhline(y=f_low, color='white', linestyle=':', alpha=0.7, linewidth=1)
                ax.axhline(y=f_high, color='white', linestyle=':', alpha=0.7, linewidth=1)
                ax.text(time_duration*0.02, (f_low + f_high)/2, band_name.capitalize(), 
                       color='white', fontsize=10, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Spectrogram saved to: {save_path}")
        
        plt.show()
    
    def analyze_single_subject(self, subject_file, video_end_time=None, 
                             plot_results=True, save_results=True):
        """
        Complete analysis pipeline for a single subject
        
        Parameters:
        -----------
        subject_file : str
            Path to subject's data file
        video_end_time : float
            Time when video ends (if None, analyze entire signal)
        plot_results : bool
            Whether to generate plots
        save_results : bool
            Whether to save results
            
        Returns:
        --------
        results : dict
            Analysis results including band powers and wavelet data
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING SUBJECT: {os.path.basename(subject_file)}")
        print(f"{'='*60}")
        
        # Load data
        data = self.load_subject_data(subject_file)
        if data is None:
            return None
        
        # Extract subject ID
        subject_id = os.path.basename(subject_file).replace('.pkl', '')
        
        # Extract EEG signal from data
        signal_data = self.extract_eeg_signal(data, video_end_time)
        
        if signal_data is None:
            print("Could not extract EEG signal from data")
            return None
            
        duration = len(signal_data) / self.fs
        print(f"Signal extracted: {len(signal_data)} samples, {duration:.2f} seconds")
        
        # Compute CWT
        print(f"\nComputing Continuous Wavelet Transform...")
        wave, scales, freqs, coi = self.compute_cwt(signal_data)
        
        # Scale to frequency conversion verification
        freqs_converted = self.scale_to_frequency_conversion(scales)
        
        # COI analysis
        print(f"\nAnalyzing Cone of Influence...")
        coi_mask, valid_region = self.identify_coi_matrix(wave, coi, freqs)
        
        # Frequency band analysis
        print(f"\nComputing frequency band powers...")
        band_powers = self.compute_frequency_band_power(wave, freqs, coi_mask, exclude_coi=True)
        
        # Plot results
        if plot_results:
            save_path = f"{subject_id}_spectrogram.png" if save_results else None
            self.plot_wavelet_spectrogram(wave, freqs, coi, duration, save_path, subject_id)
        
        # Compile results
        results = {
            'subject_id': subject_id,
            'signal_data': signal_data,
            'wavelet_coefficients': wave,
            'scales': scales,
            'frequencies': freqs,
            'coi': coi,
            'coi_mask': coi_mask,
            'valid_region': valid_region,
            'band_powers': band_powers,
            'analysis_parameters': {
                'sampling_rate': self.fs,
                'dt': self.dt,
                'dj': self.dj,
                's0': self.s0,
                'J': self.J,
                'mother_wavelet': self.mother
            }
        }
        
        # Save results
        if save_results:
            results_path = f"{subject_id}_wavelet_analysis.pkl"
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to: {results_path}")
        
        return results

def main():
    """
    Main analysis function - Directly analyze sub000.pkl using relative paths
    """
    print("PyCWT EEG Wavelet Analysis - Subject 000")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = EEGWaveletAnalyzer(sampling_rate=250)
    
    # Use relative path to sub000.pkl (both py file and Processed_data are in wavelet folder)
    target_subject = "Processed_data/sub000.pkl"
    
    # Check if sub000.pkl exists
    if not os.path.exists(target_subject):
        print(f"Error: sub000.pkl not found at {target_subject}")
        print("Available files:")
        data_dir = "Processed_data"
        if os.path.exists(data_dir):
            available_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
            for file in sorted(available_files):
                print(f"  - {file}")
        return
    
    print(f"Analyzing subject: sub000.pkl")
    
    # Run analysis
    results = analyzer.analyze_single_subject(
        target_subject, 
        video_end_time=None,  # Analyze entire signal for now
        plot_results=True,
        save_results=True
    )
    
    if results:
        print(f"\nAnalysis completed successfully for sub000!")
        print(f"Results summary:")
        for band_name, band_data in results['band_powers'].items():
            if band_data:
                print(f"  {band_name.capitalize()}: {band_data['mean_power']:.2e} (mean power)")
    
    print(f"\nNext steps:")
    print(f"1. Adapt data loading section to match your actual data format")
    print(f"2. Implement cross-validation with existing methods")
    print(f"3. Compare results with current analysis pipeline")
    print(f"4. Fine-tune wavelet parameters (dj, s0, J) if needed")

if __name__ == "__main__":
    main()
