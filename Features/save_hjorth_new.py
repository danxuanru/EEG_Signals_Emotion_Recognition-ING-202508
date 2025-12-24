import numpy as np
import pickle
import mne
import os
import argparse
from scipy import signal
import h5py

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Hjorth feature extraction for EEG data')
parser.add_argument('--data_path', default='../Processed_data', type=str, help='Path to the EEG data')
parser.add_argument('--output_path', default='../Hjorth_features_re', type=str, help='Path to save the Hjorth features')
args = parser.parse_args()

# Calculate the Hjorth parameters
def extract_hjorth(data):
    """Hjorth Parameters: Activity, Mobility, Complexity"""
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    activity = np.var(data, axis=-1) + epsilon
    diff1 = np.diff(data, axis=-1)
    mobility = np.sqrt((np.var(diff1, axis=-1) + epsilon) / activity)
    diff2 = np.diff(diff1, axis=-1)
    complexity = np.sqrt((np.var(diff2, axis=-1) + epsilon) / (np.var(diff1, axis=-1) + epsilon)) / mobility
    return np.stack([mobility, complexity], axis=-1)

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
    
    # Initialize Hjorth features array
    # Shape: (videos, channels, seconds, frequency_bands, hjorth_parameters)
    hjorth_features = np.zeros((n_vids, chn, sec, len(freqs), 2))
    
    # Extract Hjorth features for each frequency band
    for i, freq_band in enumerate(freqs):
        print(f"Processing frequency band {i+1}/{len(freqs)}: {freq_band} Hz")
        
        for j in range(n_vids):
            # Get data for current video
            data_video = data_sub[j, :, :]  # Shape: (channels, samples)
            
            # Filter data to the current frequency band
            low_freq = freq_band[0]
            high_freq = freq_band[1]
            data_video_filt = mne.filter.filter_data(data_video, fs, l_freq=low_freq, h_freq=high_freq)
            
            # Reshape to separate into seconds
            data_video_filt = data_video_filt.reshape(chn, sec, fs)
            
            # Calculate Hjorth parameters for each second
            for s in range(sec):
                hjorth_features[j, :, s, i, :] = extract_hjorth(data_video_filt[:, s, :])
    
    # Save the Hjorth features for this subject
    subject_id = path.split('.')[0]  # Extract subject ID from filename
    output_file = os.path.join(output_path, f"{subject_id}_hjorth.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump(hjorth_features, f)
    
    print(f"Saved Hjorth features for subject {subject_id} to {output_file}")

print(f"Completed Hjorth feature extraction for all subjects. Files saved to {output_path}")
print(f"Feature shape: ", hjorth_features.shape)
