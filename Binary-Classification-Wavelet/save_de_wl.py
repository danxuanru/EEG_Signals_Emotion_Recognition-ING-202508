import numpy as np
import pickle
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='DE feature extraction for Wavelet EEG data')
parser.add_argument('--data_path', default='../Wavelet', type=str, help='Path to the Wavelet data')
parser.add_argument('--output_path', default='../Features/DE_features_wl', type=str, help='Path to save the DE features')
args = parser.parse_args()

# Create output directory if it doesn't exist
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created directory: {output_path}")

def calculate_de_from_power(power_values):
    """
    Calculate differential entropy from power values.
    
    Parameters:
    -----------
    power_values : ndarray
        Power values for a specific segment
        
    Returns:
    --------
    de : float
        Differential entropy value
    """
    epsilon = 1e-10  # Small constant to avoid log(0)
    
    variance = np.var(power_values)
    de = 0.5 * np.log(2 * np.pi * np.e * (variance + epsilon))
    return de

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
    
    # Initialize DE features array
    # Shape: (videos, channels, seconds, frequency_bands)
    n_vids, n_chn, n_samples, n_bands = data_sub.shape
    de_features = np.zeros((n_vids, n_chn, sec, n_bands))
    
    # Reshape data to (videos, channels, seconds, samples_per_second, frequency_bands)
    data_reshaped = data_sub.reshape(n_vids, n_chn, sec, sample_per_sec, n_bands)

    for v in range(n_vids):
        for c in range(n_chn):
            for s in range(sec):
                for b in range(n_bands):
                    segment_power = data_reshaped[v, c, s, :, b]  # extract the per second power values
                    de_features[v, c, s, b] = calculate_de_from_power(segment_power)

    # Save the DE features for this subject
    subject_id = path.split('.')[0]  # Extract subject ID from filename
    output_file = os.path.join(output_path, f"{subject_id}_de.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump(de_features, f)
    
    print(f"Saved DE features for subject {subject_id} to {output_file}")

print(f"Completed DE feature extraction for all subjects. Files saved to {output_path}")
print(f"DE Feature shape: ", de_features.shape)
