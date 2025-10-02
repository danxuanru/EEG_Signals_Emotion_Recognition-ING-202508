import numpy as np
import pickle
import mne
import os
import argparse
from scipy.signal.windows import hann

# Parse command-line arguments
parser = argparse.ArgumentParser(description='PSD feature extraction for EEG data')
parser.add_argument('--data_path', default='../Processed_data', type=str, help='Path to the EEG data')
parser.add_argument('--output_path', default='../Features/PSD_features_sum', type=str, help='Path to save the PSD features')
args = parser.parse_args()

# calculate the power spectral density using improved method
def extract_psd_feature(data, fs=250, stft_n=256, freq_bands=None):
    """
    Calculate PSD features using windowing approach and relative power.
    """
    if freq_bands is None:
        freq_bands = freqs
        
    # Ensure data is 2D (channels, samples)
    if len(data.shape) > 2:
        data = np.squeeze(data)
        
    n_channels, n_samples = data.shape
    
    # Define 1-second windows
    window_size = 1  # 1 second window
    point_per_window = int(fs * window_size)
    window_num = int(n_samples // point_per_window)
    
    psd_feature = np.zeros((window_num, len(freq_bands), n_channels))
    
    for window_index in range(window_num):
        start_index, end_index = point_per_window * window_index, point_per_window * (window_index + 1)
        window_data = data[:, start_index:end_index]
        
        # Apply Hann window
        hdata = window_data * hann(point_per_window)
        
        # Compute FFT
        fft_data = np.fft.fft(hdata, n=stft_n)
        energy_graph = np.abs(fft_data[:, 0: int(stft_n / 2)])
        
        # Normalize for relative power
        relative_energy_graph = energy_graph / np.sum(energy_graph, axis=1, keepdims=True)
        
        # Extract power for each frequency band
        for band_index, band in enumerate(freq_bands):
            start_bin = int(np.floor(band[0] / fs * stft_n))
            end_bin = int(np.floor(band[1] / fs * stft_n))
            
            # Ensure valid indices
            start_bin = max(1, start_bin)  # Avoid DC component
            end_bin = min(int(stft_n / 2), end_bin)
            
            band_power = np.sum(relative_energy_graph[:, start_bin-1:end_bin] ** 2, axis=1)
            psd_feature[window_index, band_index, :] = band_power
    
    # Reshape to match expected output format (channels, windows, bands)
    return np.transpose(psd_feature, (2, 0, 1))

# Create output directory if it doesn't exist
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created directory: {output_path}")

# Load the data
data_path = args.data_path
data_paths = os.listdir(data_path)
data_paths.sort()
n_vids = 28
chn = 32
fs = 250
sec = 30
freqs = [[1,4], [4,8], [8,14], [14,30], [30,47]]

print(f"Processing data from {len(data_paths)} subjects...")

# Process each subject
for idx, path in enumerate(data_paths):
    print(f"Processing subject {idx+1}/{len(data_paths)}: {path}")
    
    # Load subject data
    with open(os.path.join(data_path, path), 'rb') as f:
        data_sub = pickle.load(f)
    
    
    # Initialize PSD features array
    # Shape: (videos, channels, seconds, frequency_bands)
    psd_features = np.zeros((n_vids, chn, sec, len(freqs)))
    
    for j in range(n_vids):
        print(f"Processing video {j+1}/{n_vids}")
        # Get data for current video
        data_video = data_sub[j, :, :]  # Shape: (channels, samples)
        
        # Calculate PSD features
        psd_result = extract_psd_feature(data_video, fs=fs, freq_bands=freqs)
        
        # Reshape and store PSD features
        for s in range(sec):
            for band_idx in range(len(freqs)):
                psd_features[j, :, s, band_idx] = psd_result[:, s, band_idx]
    
    # Save the PSD features for this subject
    subject_id = path.split('.')[0]  # Extract subject ID from filename
    output_file = os.path.join(output_path, f"{subject_id}_psd.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump(psd_features, f)
    
    print(f"Saved PSD features for subject {subject_id} to {output_file}")

print(f"Completed PSD feature extraction for all subjects. Files saved to {output_path}")
print(f"Feature shape: (videos={n_vids}, channels={chn}, seconds={sec}, frequency_bands={len(freqs)})")
