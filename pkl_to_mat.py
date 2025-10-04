import os
import numpy as np
import pickle
import scipy.io as sio
import argparse

# Add command line argument for session length
parser = argparse.ArgumentParser(description='Convert pickle files to mat format')
parser.add_argument('--session-length', default=30, type=int, help='Length of each video session in seconds')
parser.add_argument('--remove-band', default=0, type=int, help='Remove specific frequency band (1-5) or None to include all bands')
parser.add_argument('--feature', default='DE', choices=['DE', 'DE_PSD', 'DE_PSD_H'], help='Feature combination: DE, DE_PSD, or DE_PSD_H')
args = parser.parse_args()

data_H_path = '../Features/Hjorth_features'  # (28,32,30,5,2)
data_DE_path = '../Features/DE_features'  # (28,32,30,5)
data_PSD_path = '../Features/PSD_features_sum'  # (28,32,30,5)

# Constants for dimensions
n_subs = 123
n_vids = 28
chn = 30
fs = 250
sec = args.session_length  # Use session length from args
remove_band = args.remove_band - 1 if args.remove_band in [1,2,3,4,5] else None  # Adjust for 0-based index
freq = 4 if remove_band is not None else 5  # Adjust frequency count if a band is removed

print(f"Using session length: {sec} seconds")
print(f"Using feature combination: {args.feature}")

# Process Hjorth features (only if needed)
if args.feature == 'DE_PSD_H':
    H_data_paths = os.listdir(data_H_path)
    H_data_paths.sort()
    H_data = np.zeros((n_subs, n_vids, chn, sec, freq, 2)) 
    for idx, path in enumerate(H_data_paths):
        if idx >= n_subs:
            break
        f = open(os.path.join(data_H_path, path), 'rb')
        data_sub = pickle.load(f)
        print(f"Hjorth subject {idx+1}, shape: {data_sub.shape}")
        if remove_band is not None:
            H_data[idx, :, :, :, :, :] = np.delete(data_sub[:, :-2, :sec, :, :], remove_band, axis=-2)  # Remove the specified band from the second last dimension
        else:
            H_data[idx, :, :, :, :, :] = data_sub[:, :-2, :sec, :, :]  # Keep all bands if none is removed

# Process DE features
DE_data_paths = os.listdir(data_DE_path)
DE_data_paths.sort()
DE_data = np.zeros((n_subs, n_vids, chn, sec, freq))
for idx, path in enumerate(DE_data_paths):
    if idx >= n_subs:
        break
    f = open(os.path.join(data_DE_path, path), 'rb')
    data_sub = pickle.load(f)
    print(f"DE subject {idx+1}, shape: {data_sub.shape}")
    if remove_band is not None:
        DE_data[idx, :, :, :, :] = np.delete(data_sub[:, :-2, :sec, :], remove_band, axis=-1)  # Remove the specified band from the last dimension
    else:
        DE_data[idx, :, :, :, :] = data_sub[:, :-2, :sec, :]  # Keep all bands if none is removed

# Process PSD features (only if needed)
if args.feature in ['DE_PSD', 'DE_PSD_H']:
    PSD_data_paths = os.listdir(data_PSD_path)
    PSD_data_paths.sort()
    PSD_data = np.zeros((n_subs, n_vids, chn, sec, freq))
    for idx, path in enumerate(PSD_data_paths):
        if idx >= n_subs:
            break
        f = open(os.path.join(data_PSD_path, path), 'rb')
        data_sub = pickle.load(f)
        print(f"PSD subject {idx+1}, shape: {data_sub.shape}")
        if remove_band is not None:
            PSD_data[idx, :, :, :, :] = np.delete(data_sub[:, :-2, :sec, :], remove_band, axis=-1)  # Remove the specified band from the last dimension
        else:
            PSD_data[idx, :, :, :, :] = data_sub[:, :-2, :sec, :]  # Keep all bands if none is removed

# Reshape features based on selection
if args.feature == 'DE_PSD_H':
    # Reshape Hjorth features
    H_data = H_data.transpose([0, 2, 4, 5, 1, 3]).reshape(n_subs, chn, freq, 2, n_vids*sec)
    H_data = H_data.transpose([0, 1, 4, 2, 3])
    print('Hjorth data shape:', H_data.shape)  # Should be (n_subs, chn, n_vids*sec, freq, 2)

# Reshape DE features
DE_data = DE_data.transpose([0, 2, 4, 1, 3]).reshape(n_subs, chn, freq, n_vids*sec)
DE_data = DE_data.transpose([0, 1, 3, 2])
print('DE data shape:', DE_data.shape)  # Should be (n_subs, chn, n_vids*sec, freq)

if args.feature in ['DE_PSD', 'DE_PSD_H']:
    # Reshape PSD features
    PSD_data = PSD_data.transpose([0, 2, 4, 1, 3]).reshape(n_subs, chn, freq, n_vids*sec)
    PSD_data = PSD_data.transpose([0, 1, 3, 2])
    print('PSD data shape:', PSD_data.shape)  # Should be (n_subs, chn, n_vids*sec, freq)

# Combine features based on selection
if args.feature == 'DE':
    combined_data = DE_data
    print('Combined data shape (DE only):', combined_data.shape)
elif args.feature == 'DE_PSD':
    combined_data = np.concatenate([DE_data, PSD_data], axis=-1)
    print('Combined data shape (DE + PSD):', combined_data.shape)
elif args.feature == 'DE_PSD_H':
    # Reshape Hjorth data to combine the last two dimensions
    H_data_reshaped = H_data.reshape(n_subs, chn, n_vids*sec, freq*2)
    
    # Concatenate all features along the last dimension
    combined_data = np.concatenate([
        H_data_reshaped,     # freq*2 features
        DE_data,             # freq features
        PSD_data,            # freq features
    ], axis=-1)
    print('Combined data shape (DE + PSD + H):', combined_data.shape)

# Save to .mat file
total_data = {'data': combined_data}
sio.savemat('./data.mat', total_data)
print('data.mat saved')
