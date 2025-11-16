import numpy as np
import pickle
from asi_filter import extract_filtered_data
import mne
import os
import argparse
import scipy.io as sio
from scipy.signal import welch

# Parse command-line arguments
parser = argparse.ArgumentParser(description='DE feature extraction for EEG data after AsI filtering')
parser.add_argument('--data_path', default='../Processed_Data_after_AsI', type=str, help='Path to the filtered EEG data')
parser.add_argument("--feature_path", default="../Features", type=str, help="Path to save the extracted features")
parser.add_argument('--output_path', default='./', type=str, help='Path to save the DE features')
parser.add_argument('--feature', default='DE_PSD_H', choices=['DE', 'DE_PSD', 'DE_PSD_H'], help='Feature combination: DE, DE_PSD, or DE_PSD_H')
args = parser.parse_args()

# Create output directory if it doesn't exist
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created directory: {output_path}")

# Load the data
data_path = args.data_path
data_paths = os.listdir(data_path)
# Filter only .pkl files
data_paths = [p for p in data_paths if p.endswith('.pkl')]
data_paths.sort()
subject = 123
n_vids = 28
chn = 32
fs = 250
valid_windows = 96185
freqs = [[1,4], [4,8], [8,14], [14,30], [30,47]]

feature_path = args.feature_path
features = args.feature

def load_data(file_path):
    """Load EEG data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_de(data, output_path):
    '''
    Calculate and save Differential Entropy (DE) features.
    
    input:
    - data: EEG data in shape (n_windows, n_channels, window_samples)

    output:
    - Saves DE features to the specified feature path. (n_windows, n_channels, n_bands)
    '''

    n_windows = data.shape[0]
    n_channels = data.shape[1]
    window_samples = data.shape[2]

    de_features = np.zeros((n_windows, n_channels, len(freqs)))

    # Extract DE features for each frequency band
    for window in range(n_windows):
        for band_idx, freq_band in enumerate(freqs):
            
            data_window = data[window, :, :]  # Shape: (n_channels, window_samples)

            # Filter data to the current frequency band
            low_freq = freq_band[0]
            high_freq = freq_band[1]
            data_filt = mne.filter.filter_data(data_window, fs, l_freq=low_freq, h_freq=high_freq)
            
            # Add small epsilon to prevent log(0)
            epsilon = 1e-10
            variance = np.var(data_filt, axis=-1) + epsilon
            de_one = 0.5 * np.log(2 * np.pi * np.exp(1) * variance)
            de_features[window, :, band_idx] = de_one
        
    with open(output_path, 'wb') as f:
        pickle.dump(de_features, f)

    return

def save_psd(data, output_path):
    '''
    Calculate and save Power Spectral Density (PSD) features.

    input:
    - data: EEG data in shape (n_windows, n_channels, window_samples)

    output:
    - Saves PSD features to the specified feature path. (n_windows, n_channels, n_bands)
    '''
    n_windows = data.shape[0]
    n_channels = data.shape[1]
    window_samples = data.shape[2]

    psd_features = np.zeros((n_windows, n_channels, len(freqs)))

    # Extract PSD features for each frequency band
    for window in range(n_windows):
        data_window = data[window, :, :]  # Shape: (n_channels, window_samples)
        
        for ch in range(n_channels):
            # Compute PSD using Welch's method
            freqs_psd, psd = welch(data_window[ch, :], fs, nperseg=min(256, window_samples))
            
            for band_idx, freq_band in enumerate(freqs):
                low_freq = freq_band[0]
                high_freq = freq_band[1]
                
                # Find frequency indices for the current band
                freq_mask = (freqs_psd >= low_freq) & (freqs_psd <= high_freq)
                
                # Calculate average power in the frequency band
                if np.any(freq_mask):
                    band_power = np.sum(psd[freq_mask])
                else:
                    band_power = 0
                
                psd_features[window, ch, band_idx] = band_power
        
    with open(output_path, 'wb') as f:
        pickle.dump(psd_features, f)
    
    return

def save_hjorth(data, output_path):
    '''
    Calculate and save Hjorth parameters features.

    input:
    - data: EEG data in shape (n_windows, n_channels, window_samples)

    output:
    - saves Hjorth features to the specified feature path. (n_windows, n_channels, n_bands, 2)   
    '''
    n_windows = data.shape[0]
    n_channels = data.shape[1]
    window_samples = data.shape[2]

    hjorth_features = np.zeros((n_windows, n_channels, len(freqs), 2))

    # Extract PSD features for each frequency band
    for window in range(n_windows):
        for band_idx, freq_band in enumerate(freqs):
            
            data_window = data[window, :, :]  # Shape: (n_channels, window_samples)

            # Filter data to the current frequency band
            low_freq = freq_band[0]
            high_freq = freq_band[1]
            data_filt = mne.filter.filter_data(data_window, fs, l_freq=low_freq, h_freq=high_freq)
            
            # Add small epsilon to prevent log(0)
            epsilon = 1e-10
            activity = np.var(data_filt, axis=-1) + epsilon
            diff1 = np.diff(data_filt, axis=-1)
            mobility = np.sqrt((np.var(diff1, axis=-1) + epsilon) / activity)
            diff2 = np.diff(diff1, axis=-1)
            complexity = np.sqrt((np.var(diff2, axis=-1) + epsilon) / (np.var(diff1, axis=-1) + epsilon)) / mobility
            hjorth_features[window, :, band_idx, :] = np.stack([mobility, complexity], axis=-1)
        
    with open(output_path, 'wb') as f:
        pickle.dump(hjorth_features, f)
    
    return

def save_features():
    """Extract and save features for each subject."""
    
    for sub_idx in range(subject):

        # load data
        data_path_sub = os.path.join(data_path, f'sub{sub_idx:03d}.pkl')
        sub_data, labels = extract_filtered_data(data_path_sub, output_format='windows')

        # Create feature directories
        de_dir = os.path.join(feature_path, 'DE_Feature_AsI')
        psd_dir = os.path.join(feature_path, 'PSD_Feature_AsI')
        hjorth_dir = os.path.join(feature_path, 'Hjorth_Feature_AsI')
        
        for dir_path in [de_dir, psd_dir, hjorth_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        de_features_path = os.path.join(de_dir, f'sub{sub_idx:03d}.pkl')
        psd_features_path = os.path.join(psd_dir, f'sub{sub_idx:03d}.pkl')
        hjorth_features_path = os.path.join(hjorth_dir, f'sub{sub_idx:03d}.pkl')

        # feature extraction for each subject
        if not os.path.exists(de_features_path):
            save_de(sub_data, de_features_path)

        if not os.path.exists(psd_features_path) and features in ['DE_PSD', 'DE_PSD_H']:
            save_psd(sub_data, psd_features_path)

        if not os.path.exists(hjorth_features_path) and features == 'DE_PSD_H':
            save_hjorth(sub_data, hjorth_features_path)

        print(f"Features extracted and saved for subject {sub_idx+1}/{subject}")
    print("All features extracted and saved.")


def create_pool():
    '''
    Create a pool of feature vectors from all subjects.
    '''
    
    all_features = []
    all_labels = []

    for sub_idx in range(subject):
        # load labels
        data_path_sub = os.path.join(data_path, f'sub{sub_idx:03d}.pkl')
        sub_data, labels = extract_filtered_data(data_path_sub, output_format='windows')

        # load features data
        de_features_path = os.path.join(feature_path, 'DE_Feature_AsI', f'sub{sub_idx:03d}.pkl')
        
        # load DE features data
        with open(de_features_path, 'rb') as f:
            de_data = pickle.load(f) 

        # exclude last 2 channels (keep first 30 channels)
        window_features = de_data[:, :-2, :]  # (n_windows, 30, n_bands)

        if features in ['DE_PSD', 'DE_PSD_H']:
            psd_features_path = os.path.join(feature_path, 'PSD_Feature_AsI', f'sub{sub_idx:03d}.pkl')
            with open(psd_features_path, 'rb') as f:
                psd_data = pickle.load(f)
            # Concatenate along the feature dimension
            window_features = np.concatenate([window_features, psd_data[:, :-2, :]], axis=-1)

        if features == 'DE_PSD_H':
            hjorth_features_path = os.path.join(feature_path, 'Hjorth_Feature_AsI', f'sub{sub_idx:03d}.pkl')
            with open(hjorth_features_path, 'rb') as f:
                hjorth_data = pickle.load(f)
            # Reshape hjorth data to match concatenation: (n_windows, 30, n_bands, 2) -> (n_windows, 30, n_bands*2)
            hjorth_reshaped = hjorth_data[:, :-2, :, :].reshape(hjorth_data.shape[0], hjorth_data.shape[1]-2, -1)
            window_features = np.concatenate([window_features, hjorth_reshaped], axis=-1)

        # Flatten features for each window: (n_windows, channels*bands*features)
        flattened_features = window_features.reshape(window_features.shape[0], -1)
        
        # Add subject and video index as first two columns
        n_windows = flattened_features.shape[0]
        subject_column = np.full((n_windows, 1), sub_idx)  # Subject index
        video_column = np.array(labels).reshape(-1, 1)     # Video index (labels)
        
        # Concatenate: [subject, video, flattened_features]
        features_with_metadata = np.concatenate([subject_column, video_column, flattened_features], axis=1)
        
        all_features.append(features_with_metadata)
        all_labels.extend(labels)

    # Concatenate all subjects' data
    pool_X = np.vstack(all_features)
    pool_Y = np.array(all_labels)

    # Convert labels to binary classification: >16 -> 1, <=16 -> -1
    pool_Y = np.where(pool_Y > 16, 1, -1)

    print(f"Pool data shape: X={pool_X.shape}, Y={pool_Y.shape}")
    print(f"Features format: [subject_id, video_id, flattened_features...]")

    # save pool data
    output_file = os.path.join(output_path, f'pooled_data_{features}.mat')
    sio.savemat(output_file, {'X': pool_X, 'Y': pool_Y})
    
    print(f"Pooled data saved to: {output_file}")
    return pool_X, pool_Y


if __name__ == "__main__":
    # Extract features for all subjects
    save_features()
    
    # Create pooled dataset
    create_pool()

