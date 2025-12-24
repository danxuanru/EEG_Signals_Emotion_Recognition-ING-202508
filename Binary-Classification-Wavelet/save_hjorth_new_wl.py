import numpy as np
import pickle
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Hjorth feature extraction for Wavelet EEG data')
parser.add_argument('--data_path', default='../Wavelet', type=str, help='Path to the Wavelet EEG data')
parser.add_argument('--output_path', default='../Features/Hjorth_features2_wl', type=str, help='Path to save the Hjorth features')
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
    
    # Initialize Hjorth features array
    # Shape: (videos, channels, seconds, frequency_bands, hjorth_params)
    n_vids, n_chn, n_samples, n_bands = data_sub.shape
    hjorth_features = np.zeros((n_vids, n_chn, sec, n_bands, 2))

    # Reshape data to (videos, channels, seconds, samples_per_second, frequency_bands)
    data_reshaped = data_sub.reshape(n_vids, n_chn, sec, sample_per_sec, n_bands)

    for v in range(n_vids):
        for c in range(n_chn):
            for s in range(sec):
                for b in range(n_bands):
                    segment_power = data_reshaped[v, c, s, :, b]  # extract the per second power values
                    hjorth_features[v, c, s, b, :] = extract_hjorth(segment_power)

    # Save the Hjorth features for this subject
    subject_id = path.split('.')[0]  # Extract subject ID from filename
    output_file = os.path.join(output_path, f"{subject_id}_hjorth.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump(hjorth_features, f)
    
    print(f"Saved Hjorth features for subject {subject_id} to {output_file}")

print(f"Completed Hjorth feature extraction for all subjects. Files saved to {output_path}")
print(f"Hjorth Feature shape: ", hjorth_features.shape)
