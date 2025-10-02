import numpy as np
import pickle
import mne
import os
import argparse
from scipy import signal

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
    # Shape: (videos, channels, seconds, frequency_bands, DE_parameters)
    de_features = np.zeros((n_vids, chn, sec, len(freqs)))
    
    # Extract DE features for each frequency band
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
            
            # Add small epsilon to prevent log(0)
            epsilon = 1e-10
            variance = np.var(data_video_filt, 2) + epsilon
            de_one = 0.5*np.log(2*np.pi*np.exp(1)*variance)
            # (n_subs, 30, 28*30, freqs)
            de_features[j, :, :, i] = de_one
    
    # Save the DE features for this subject
    subject_id = path.split('.')[0]  # Extract subject ID from filename
    output_file = os.path.join(output_path, f"{subject_id}_de.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump(de_features, f)
    
    print(f"Saved DE features for subject {subject_id} to {output_file}")

print(f"Completed DE feature extraction for all subjects. Files saved to {output_path}")
print(f"Feature shape: ", de_features.shape)
