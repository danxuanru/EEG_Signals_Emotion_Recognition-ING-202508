import numpy as np
import pickle
import pywt
import mne
import os
import argparse
from scipy import signal
import h5py

# Parse command-line arguments
parser = argparse.ArgumentParser(description='DE feature extraction for EEG data')
parser.add_argument('--data_path', default='../Processed_data', type=str, help='Path to the EEG data')
parser.add_argument('--output_path', default='../DE_features', type=str, help='Path to save the DE features')
args = parser.parse_args()

# Create output directory if it doesn't exist
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created directory: {output_path}")

# Wavelet-based differential entropy extraction function
def extract_wavelet_de(data, fs=250, freq_bands=None):
    """
    Extract differential entropy features using wavelet decomposition.
    
    Parameters:
    -----------
    data : ndarray
        EEG data with shape (channels, samples)
    fs : int
        Sampling frequency
    freq_bands : list
        List of frequency bands, each defined as [low_freq, high_freq]
        
    Returns:
    --------
    de_features : ndarray
        Differential entropy features with shape (channels, seconds, bands)
    """
    if freq_bands is None:
        freq_bands = [[1,4], [4,8], [8,14], [14,30], [30,47]]
    
    n_channels, n_samples = data.shape
    seconds = n_samples // fs
    n_bands = len(freq_bands)
    
    # Reshape to separate into seconds
    data_seconds = data.reshape(n_channels, seconds, fs)
    
    # Initialize output array
    de_features = np.zeros((n_channels, seconds, n_bands))
    
    # Define wavelet
    wavelet = 'db4'  # Daubechies 4 wavelet
    
    for ch in range(n_channels):
        for s in range(seconds):
            # Get 1-second data segment
            segment = data_seconds[ch, s, :]
            
            # Perform wavelet decomposition (5 levels for 250Hz is appropriate)
            coeffs = pywt.wavedec(segment, wavelet, level=5)
            
            # Extract coefficients for each frequency band
            # For fs=250Hz with 5 levels:
            # Level 1 (d1): ~62.5-125 Hz
            # Level 2 (d2): ~31.25-62.5 Hz (covers gamma)
            # Level 3 (d3): ~15.625-31.25 Hz (covers beta)
            # Level 4 (d4): ~7.8125-15.625 Hz (covers alpha)
            # Level 5 (d5): ~3.90625-7.8125 Hz (covers theta)
            # Approximation (a5): ~0-3.90625 Hz (covers delta)
            
            # Extract appropriate coefficients for each band
            band_data = []
            
            # Delta: 1-4 Hz (approximation coefficients)
            band_data.append(coeffs[-1])
            
            # Theta: 4-8 Hz (level 5 detail coefficients)
            band_data.append(coeffs[1])
            
            # Alpha: 8-14 Hz (level 4 detail coefficients)
            band_data.append(coeffs[2])
            
            # Beta: 14-30 Hz (level 3 detail coefficients)
            band_data.append(coeffs[3])
            
            # Gamma: 30-47 Hz (part of level 2 detail coefficients)
            band_data.append(coeffs[4])
            
            # Calculate differential entropy for each band
            for i, band_coeffs in enumerate(band_data):
                # Add small epsilon to prevent log(0)
                epsilon = 1e-10
                # Calculate variance of wavelet coefficients
                variance = np.var(band_coeffs) + epsilon
                # Calculate differential entropy
                de = 0.5 * np.log(2 * np.pi * np.exp(1) * variance)
                de_features[ch, s, i] = de
    
    return de_features

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
    
    # Initialize DE features array
    # Shape: (videos, channels, seconds, frequency_bands)
    de_features = np.zeros((n_vids, chn, sec, len(freqs)))
    
    for j in range(n_vids):
        print(f"Processing video {j+1}/{n_vids}")
        # Get data for current video
        data_video = data_sub[j, :, :]  # Shape: (channels, samples)
        
        # Extract differential entropy features using wavelet transform
        de_one = extract_wavelet_de(data_video, fs=fs, freq_bands=freqs)
        
        # Store features
        de_features[j, :, :, :] = de_one
    
    # Save the DE features for this subject
    subject_id = path.split('.')[0]  # Extract subject ID from filename
    output_file = os.path.join(output_path, f"{subject_id}_de.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump(de_features, f)
    
    print(f"Saved DE features for subject {subject_id} to {output_file}")

print(f"Completed DE feature extraction for all subjects. Files saved to {output_path}")
print(f"Feature shape: ", de_features.shape)
