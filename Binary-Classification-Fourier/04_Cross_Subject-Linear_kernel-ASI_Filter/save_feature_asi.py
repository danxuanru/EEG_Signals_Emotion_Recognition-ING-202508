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
parser.add_argument('--data-path', default='../Processed_Data_after_AsI', type=str, help='Path to the filtered EEG data')
parser.add_argument("--feature-path", default="../Features_Data", type=str, help="Path to save the extracted features")
parser.add_argument('--output-path', default='./', type=str, help='Path to save the DE features')
parser.add_argument('--window-sec', default=1, type=int, help='Window size in seconds for feature extraction')
parser.add_argument('--overlap-ratio', default=0.5, type=float, help='Overlap ratio for feature extraction')
parser.add_argument('--feature', default='DE_PSD_H', choices=['DE', 'DE_PSD', 'DE_PSD_H'], help='Feature combination: DE, DE_PSD, or DE_PSD_H')
parser.add_argument('--log-interval', type=int, default=10, help='Log progress every N subjects')
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
freqs = [[1,4], [4,8], [8,14], [14,30], [30,47]]

feature_path = args.feature_path
features = args.feature
window_sec = args.window_sec
overlap_ratio = args.overlap_ratio

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

def save_features(window_sec=1, overlap_ratio=0.5, log_interval=10):
    """Extract and save features for each subject."""

    # Create feature directories
    if overlap_ratio == 0:
        suffix = f'_{window_sec}s'
    else:
        suffix = f'_{window_sec}s_overlap{int(overlap_ratio * 100)}'
    de_dir = os.path.join(feature_path, f'DE_Features_after_AsI{suffix}')
    psd_dir = os.path.join(feature_path, f'PSD_Features_after_AsI{suffix}')
    hjorth_dir = os.path.join(feature_path, f'Hjorth_Features_after_AsI{suffix}')

    created_dirs = 0
    for dir_path in [de_dir, psd_dir, hjorth_dir]:  # 若是第一次非執行DE_PSD_H
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            created_dirs += 1
            print(f"Created directory: {dir_path}")
    # all path have existed or created -> finish the feature directory setup
    if created_dirs == 0:
        print("Feature directories already exist. Skipping feature extraction.")
        return -1  # Indicate that feature extraction was skipped
    
    total_windows = 0
    
    for sub_idx in range(subject):

        # load data
        data_path_sub = os.path.join(data_path, f'sub{sub_idx:03d}.pkl')
        sub_data, labels = extract_filtered_data(data_path_sub, output_format='windows')
        total_windows += sub_data.shape[0]

        de_features_path = os.path.join(de_dir, f'sub{sub_idx:03d}.pkl')
        psd_features_path = os.path.join(psd_dir, f'sub{sub_idx:03d}.pkl')
        hjorth_features_path = os.path.join(hjorth_dir, f'sub{sub_idx:03d}.pkl')

        # feature extraction for each subject
        if not os.path.exists(de_features_path):
            save_de(sub_data, de_features_path)

        if not os.path.exists(psd_features_path):
            save_psd(sub_data, psd_features_path)

        if not os.path.exists(hjorth_features_path):
            save_hjorth(sub_data, hjorth_features_path)

        if ((sub_idx + 1) % log_interval == 0) or (sub_idx == subject - 1):
            print(f"Features extracted for subject {sub_idx + 1}/{subject}")
    print("All features extracted and saved.")
    return total_windows


def create_pool(window_sec=1, overlap_ratio=0.5):
    '''
    Create a pool of feature vectors from all subjects.
    Binary classification: Negative (-1) vs Positive (1) valence.
    Neutral videos (13-16) are excluded.
    '''
    if overlap_ratio == 0:
        suffix = f'_{window_sec}s'
    else:
        suffix = f'_{window_sec}s_overlap{int(overlap_ratio * 100)}'
    de_dir = os.path.join(feature_path, f'DE_Features_after_AsI{suffix}')
    psd_dir = os.path.join(feature_path, f'PSD_Features_after_AsI{suffix}')
    hjorth_dir = os.path.join(feature_path, f'Hjorth_Features_after_AsI{suffix}')
    
    all_features = []
    all_labels = []

    for sub_idx in range(subject):
        # load labels
        data_path_sub = os.path.join(data_path, f'sub{sub_idx:03d}.pkl')
        sub_data, labels = extract_filtered_data(data_path_sub, output_format='windows')

        # load features data
        de_features_path = os.path.join(de_dir, f'sub{sub_idx:03d}.pkl')
        
        if not os.path.exists(de_features_path):
            raise FileNotFoundError(f"Missing DE features: {de_features_path}. Run save_features with matching window settings.")
        
        # load DE features data
        with open(de_features_path, 'rb') as f:
            de_data = pickle.load(f) 

        # exclude last 2 channels (keep first 30 channels)
        window_features = de_data[:, :-2, :]  # (n_windows, 30, n_bands)

        if features in ['DE_PSD', 'DE_PSD_H']:
            psd_features_path = os.path.join(psd_dir, f'sub{sub_idx:03d}.pkl')
            if not os.path.exists(psd_features_path):
                raise FileNotFoundError(f"Missing PSD features: {psd_features_path}. Run save_features with matching window settings.")
            with open(psd_features_path, 'rb') as f:
                psd_data = pickle.load(f)
            window_features = np.concatenate([window_features, psd_data[:, :-2, :]], axis=-1)

        if features == 'DE_PSD_H':
            hjorth_features_path = os.path.join(hjorth_dir, f'sub{sub_idx:03d}.pkl')
            if not os.path.exists(hjorth_features_path):
                raise FileNotFoundError(f"Missing Hjorth features: {hjorth_features_path}. Run save_features with matching window settings.")
            with open(hjorth_features_path, 'rb') as f:
                hjorth_data = pickle.load(f)
            hjorth_reshaped = hjorth_data[:, :-2, :, :].reshape(hjorth_data.shape[0], hjorth_data.shape[1]-2, -1)
            window_features = np.concatenate([window_features, hjorth_reshaped], axis=-1)

        # Flatten features
        flattened_features = window_features.reshape(window_features.shape[0], -1)

        if len(labels) != flattened_features.shape[0]:
            raise ValueError(f"Label count ({len(labels)}) mismatches feature windows ({flattened_features.shape[0]}). Verify window length/overlap settings.")
        
        # Convert video indices to valence labels
        valence_map = {
            1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1, 11: -1, 12: -1,
            17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1
        }
        valence_labels = np.array([valence_map[int(vid)] for vid in labels])
        
        # Filter out neutral samples
        non_neutral_mask = valence_labels != 0
        
        if not np.any(non_neutral_mask):
            print(f"Warning: Subject {sub_idx} has no non-neutral samples")
            continue
        
        flattened_features = flattened_features[non_neutral_mask]
        valence_labels = valence_labels[non_neutral_mask]
        video_labels = np.array(labels)[non_neutral_mask]
        
        # Add metadata
        n_windows = flattened_features.shape[0]
        subject_column = np.full((n_windows, 1), sub_idx)
        video_column = video_labels.reshape(-1, 1)
        
        features_with_metadata = np.concatenate([subject_column, video_column, flattened_features], axis=1)
        
        all_features.append(features_with_metadata)
        all_labels.extend(valence_labels)

    # Concatenate all subjects' data
    pool_X = np.vstack(all_features)
    pool_Y = np.array(all_labels)

    print(f"Pool data shape: X={pool_X.shape}, Y={pool_Y.shape}")
    print(f"Positive samples: {np.sum(pool_Y == 1)}, Negative samples: {np.sum(pool_Y == -1)}")
    print(f"Class balance: {np.sum(pool_Y == 1) / len(pool_Y) * 100:.2f}% positive")
    print(f"Features format: [subject_id, video_id, flattened_features...]")

    # save pool data
    output_file = os.path.join(output_path, f'pooled_data_{features}.mat')
    sio.savemat(output_file, {'X': pool_X, 'Y': pool_Y})
    
    print(f"Pooled data saved to: {output_file}")
    return pool_X, pool_Y


def main():
    # 動態計算有效 window 數
    print("Calculating valid windows from filtered data...")
    valid_windows = 0
    
    data_paths = [p for p in os.listdir(args.data_path) if p.endswith('.pkl')]
    for data_file in data_paths:
        file_path = os.path.join(args.data_path, data_file)
        try:
            sub_data, _ = extract_filtered_data(file_path, output_format='windows')
            valid_windows += sub_data.shape[0]
        except Exception as e:
            print(f"Warning: {data_file} - {e}")
    
    print(f"Total valid windows: {valid_windows}")
    
    # Extract features for all subjects
    total_windows = save_features(window_sec=args.window_sec,
                                  overlap_ratio=args.overlap_ratio,
                                  log_interval=args.log_interval)
    
    # Verify consistency
    if total_windows == -1:
        print("Feature extraction already completed previously. Skipping verification.")
    elif total_windows != valid_windows:
        print(f"Warning: Mismatch in window count!")
        print(f"  Calculated: {valid_windows}")
        print(f"  Actual: {total_windows}")
    
    # Create pooled dataset
    create_pool(args.window_sec, args.overlap_ratio)

if __name__ == "__main__":
    main()

