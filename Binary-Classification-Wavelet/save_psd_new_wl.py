import numpy as np
import pickle
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='PSD feature extraction for Wavelet EEG data')
parser.add_argument('--data_path', default='../Wavelet', type=str, help='Path to the Wavelet EEG data')
parser.add_argument('--output_path', default='../Features/PSD_features_sum_wl', type=str, help='Path to save the PSD features')
args = parser.parse_args()

def calculate_psd_from_power(power_values):
    """
    Calculate PSD from power values.
    
    Parameters:
    -----------
    power_values : ndarray
        Power values for a specific segment
        
    Returns:
    --------
    psd : float
        PSD value
    """
    epsilon = 1e-10  # Small constant to avoid log(0)

    # Calculate the sum of power values: E[x^2]
    psd = np.sum(power_values ** 2) + epsilon
    return psd

# Create output directory if it doesn't exist
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created directory: {output_path}")

# Load the data
data_path = args.data_path
data_paths = os.listdir(data_path)
data_paths.sort()
fs = 250
sec = 30
sample_per_sec = fs
# freqs = [[0.5,4], [4,8], [8,14], [14,30], [30,50]]

print(f"Processing data from {len(data_paths)} subjects...")

# Process each subject
for idx, path in enumerate(data_paths):
    print(f"Processing subject {idx+1}/{len(data_paths)}: {path}")
    
    # Load subject data
    with open(os.path.join(data_path, path), 'rb') as f:
        data_sub = pickle.load(f)
    
    
    # Initialize PSD features array
    # Shape: (videos, channels, seconds, frequency_bands)
    n_vids, n_chn, n_samples, n_bands = data_sub.shape
    psd_features = np.zeros((n_vids, n_chn, sec, n_bands))
    
    # Reshape data to (videos, channels, seconds, samples_per_second, frequency_bands)
    data_reshaped = data_sub.reshape(n_vids, n_chn, sec, sample_per_sec, n_bands)
    
    for v in range(n_vids):
        for c in range(n_chn):
            for s in range(sec):
                for b in range(n_bands):
                    segment_power = data_reshaped[v, c, s, :, b]  # extract the per second power values
                    psd_features[v, c, s, b] = calculate_psd_from_power(segment_power)

    # Save the PSD features for this subject
    subject_id = path.split('.')[0]  # Extract subject ID from filename
    output_file = os.path.join(output_path, f"{subject_id}_psd.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump(psd_features, f)
    
    print(f"Saved PSD features for subject {subject_id} to {output_file}")

print(f"Completed PSD feature extraction for all subjects. Files saved to {output_path}")
print(f"PSD Feature shape: ", psd_features.shape)