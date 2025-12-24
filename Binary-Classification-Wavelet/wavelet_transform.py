import numpy as np
import os
import pickle
import argparse
import pycwt as wavelet
from matplotlib import pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Wavelet transform for EEG data')
parser.add_argument('--data_path', default='../Processed_data', type=str, help='Path to the EEG data')
parser.add_argument('--output_path', default='../Wavelet', type=str, help='Path to save the wavelet features')
args = parser.parse_args()

class WaveletTransform:
    def __init__(self, params, freqs_band, mother='Morlet'):
        """
        Initialize the WaveletTransform

        Parameters:
        - params: dict, parameters for the processed data
        - freqs_band: dict, frequency bands for analysis
        - mother: str, mother wavelet to use
        """
        self.params = params
        self.freqs_band = freqs_band
        self.cwt_params = {
            'mother': mother,
            'w0': 6,
            'dt': 1/self.params['fs'],  # time step
            'dj': 1/12,                 # scale resolution (changed from unstable value)
            's0': 2 * (1/self.params['fs']),  # smallest scale
            'J': 9 / (1/12)  # number of scales
        }
        self.results = {}

    def load_subject_data(self, data_path):
        """
        Load EEG data for a subject

        Parameters:
        - data_path: str, path to the subject's data file

        Returns:
        - data: np.ndarray, loaded EEG data
        """
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Successfully loaded data from {data_path}", flush=True)

            # print structure of data
            print(f"Data shape: {data.shape}", flush=True)  # Assuming data is a numpy array

        except Exception as e:
            print(f"Error loading data from {data_path}: {e}", flush=True)
            data = None
        return data
    


    def cwt_transform(self, data):
        """
        Perform Continuous Wavelet Transform (CWT) on the data

        Parameters:
        - data: np.ndarray, input EEG data (1D array)
        """
        # create mother wavelet object
        wavelet_func = getattr(wavelet, self.cwt_params['mother'])
        w = wavelet_func(self.cwt_params['w0'])

        # Normalize data
        data_norm = (data - np.mean(data)) / np.std(data)

        try:
            wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
                data_norm, 
                self.cwt_params['dt'], 
                self.cwt_params['dj'], 
                self.cwt_params['s0'], 
                self.cwt_params['J'],
                wavelet=w
            )

            # power calculation
            power = np.abs(wave) ** 2

            # COI calculation
            coi_freq = 1 / coi  # Convert COI periods to frequencies

            self.results = {
                'wave': wave,
                'scales': scales,
                'freqs': freqs,
                'coi': coi,
                'coi_freq': coi_freq,
                'power': power.copy()  # store a copy to avoid modifying original
            }

            return True
        except Exception as e:
            print(f"Error in wavelet transform: {e}", flush=True)
            return False

    def remove_coi(self):
        """
        Remove the cone of influence (COI) from the results.
        """
        if not self.results:
            print("No wavelet results to process", flush=True)
            return False

        # Make a copy to avoid modifying the original
        power = self.results['power'].copy()
        freqs = self.results['freqs']
        coi_freq = self.results['coi_freq']

        for i in range(power.shape[1]):
            bad_freq_idx = np.where(freqs < coi_freq[i])[0]
            power[bad_freq_idx, i] = np.nan  # Set power values outside COI to NaN

        self.results['power_no_coi'] = power
        
        return True

    def average_band_power(self):
        """
        Average power in each frequency band.

        Returns:
        - band_power: np.ndarray, averaged power in each frequency band
        """

        power = self.results['power_no_coi']
        freqs = self.results['freqs']
        n_bands = len(self.freqs_band)
        n_times = power.shape[1]
        band_power = np.zeros((n_times, n_bands))

        for i, (band_name, (f_low, f_high)) in enumerate(self.freqs_band.items()):
            freq_indices = np.where((freqs >= f_low) & (freqs <= f_high))[0]
            
            if freq_indices.size > 0:
                # For each time point, check if we have non-NaN values
                for t in range(n_times):
                    values = power[freq_indices, t]
                    valid_values = values[~np.isnan(values)]
                    
                    if len(valid_values) > 0:
                        band_power[t, i] = np.mean(valid_values)
                    else:
                        band_power[t, i] = np.nan
            else:
                print(f"Warning: No frequencies found in band {band_name} ({f_low}-{f_high} Hz)", flush=True)
                band_power[:, i] = np.nan

        return band_power


# Define plot_freq_spectrum as a standalone function, not a class method
def plot_freq_spectrum(power, freqs, coi_freq, n_time, freq_bands):
    """
    Plot the wavelet power spectrum.

    Parameters:
    - power: np.ndarray, power spectrum
    - freqs: np.ndarray, frequency values
    - coi_freq: np.ndarray, cone of influence frequencies
    - n_time: np.ndarray, time points
    - freq_bands: dict, frequency bands to mark on the plot
    """
    # plot the power spectrum
    fig, ax = plt.subplots(figsize=(10, 6))

    # create power colormap
    im = ax.contourf(
        n_time, 
        freqs, 
        np.log10(power), 
        levels=50, 
        cmap='jet', 
        extend='both'
    )

    ax.set_yscale('log')
    ax.set_ylim([freqs[-1], freqs[0]])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Wavelet Power Spectrum')
    cbar = fig.colorbar(im)
    cbar.set_label('log(Power)')

    # coi range
    ax.fill_between(
        n_time,
        freqs.min(), 
        coi_freq,
        color='white',
        alpha=0.5,
        hatch='//',
        edgecolor='black'
    )

    # mark standard EEG bands
    for band_name, (f_low, f_high) in freq_bands.items():
        if f_low >= freqs.min() and f_high <= freqs.max():
            ax.axhline(y=f_low, color='white', linestyle=':', alpha=0.7, linewidth=1)
            ax.axhline(y=f_high, color='white', linestyle=':', alpha=0.7, linewidth=1)
            ax.text(30*0.02, (f_low + f_high)/2, band_name.capitalize(), 
                   color='white', fontsize=10, weight='bold')
    
    plt.tight_layout()
    print("Frequency spectrum plotted successfully", flush=True)
    return fig

def main(params, freqs_band, mother='Morlet'):
    """
    Main function to perform wavelet transform on all subjects.
    
    Parameters:
    - params: dict, parameters for processing
    - freqs_band: dict, frequency bands for analysis
    - mother: str, mother wavelet to use
    """
    # create analyzer instance
    analyzer = WaveletTransform(params, freqs_band, mother)

    # Ensure output directory exists
    os.makedirs(params['output_path'], exist_ok=True)
    
    # Process each subject
    for subject_id in range(params['subjects']):
        subject_file = os.path.join(params['data_path'], f'sub{str(subject_id).zfill(3)}.pkl')
        
        if not os.path.exists(subject_file):
            print(f"Subject file not found: {subject_file}, skipping.", flush=True)
            continue
            
        data_sub = analyzer.load_subject_data(subject_file)
        if data_sub is None:
            print(f"Failed to load data for subject {subject_id}, skipping.", flush=True)
            continue
        
        # Initialize output array
        output_data = np.zeros((params['n_vids'], params['chn'], params['fs'] * params['sec'], len(freqs_band)))

        # Process each video and channel
        for vid in range(params['n_vids']):
            for chn in range(params['chn']):
                # Extract data for this video and channel
                data = data_sub[vid, chn, :]
                N = params['fs'] * params['sec']
                dt = 1 / params['fs']
                time = np.arange(0, N) * dt

                # wavelet parameters setup
                analyzer.cwt_params['mother'] = mother
                analyzer.cwt_params['w0'] = 6
                analyzer.cwt_params['dt'] = dt
                analyzer.cwt_params['s0'] = 2 * dt
                analyzer.cwt_params['dj'] = params['dj']
                analyzer.cwt_params['J'] = 9 / analyzer.cwt_params['dj']

                # Perform wavelet transform
                if not analyzer.cwt_transform(data):
                    print(f"Wavelet transform failed for subject {subject_id}, video {vid}, channel {chn}", flush=True)
                    continue
                print(f"Wavelet transform successful for subject {subject_id}, video {vid}, channel {chn}", flush=True)

                # Remove cone of influence
                if not analyzer.remove_coi():
                    print(f"COI removal failed for subject {subject_id}, video {vid}, channel {chn}", flush=True)
                    continue
                print(f"COI removal successful for subject {subject_id}, video {vid}, channel {chn}", flush=True)

                # Average power in each frequency band
                band_power = analyzer.average_band_power()
                print(f"Averaged band power calculated for subject {subject_id}, video {vid}, channel {chn}", flush=True)
                
                # Store the results
                output_data[vid, chn, :, :] = band_power
                
                # Optional: save plots for verification
                # if vid == 0 and chn == 0:  # Only for first video and channel as example
                #     fig = plot_freq_spectrum(
                #         analyzer.results['power'], 
                #         analyzer.results['freqs'], 
                #         analyzer.results['coi_freq'], 
                #         time, 
                #         freqs_band
                #     )
                #     fig.savefig(os.path.join(params['output_path'], f'sub{str(subject_id).zfill(3)}_wavelet_plot.png'))
                #     print(f"Plot saved for subject {subject_id}", flush=True)
                #     plt.close(fig)

        # Save output data for this subject
        output_file = os.path.join(params['output_path'], f'sub{str(subject_id).zfill(3)}_wavelet.pkl')
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(output_data, f)
            print(f"Output data saved to {output_file}", flush=True)
        except Exception as e:
            print(f"Error saving output for subject {subject_id}: {e}", flush=True)

if __name__ == "__main__":
    freqs_band = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50)
    }

    # parameters setting
    params = {
        'data_path': args.data_path,
        'output_path': args.output_path,  # Use the direct path, not os.listdir()
        'subjects': 123,
        'n_vids': 28,
        'chn': 32,
        'fs': 250,
        'sec': 30,
        'dj': 0.25  # Added proper scale resolution parameter
    }

    # Check if data path exists
    if not os.path.exists(params['data_path']):
        print(f"Data path does not exist: {params['data_path']}", flush=True)
        exit(1)

    # Call main with correct parameter order
    main(params, freqs_band)