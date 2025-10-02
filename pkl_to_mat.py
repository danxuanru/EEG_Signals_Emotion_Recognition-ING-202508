import os
import numpy as np
import pickle
import scipy.io as sio
import argparse

# Add command line argument for session length
parser = argparse.ArgumentParser(description='Convert pickle files to mat format')
parser.add_argument('--session-length', default=30, type=int, help='Length of each video session in seconds')
args = parser.parse_args()

data_DE_path = '../Features/DE_features'  # (28,32,30,5)

# Constants for dimensions
n_subs = 123
n_vids = 28
chn = 30
fs = 250
sec = args.session_length  # Use session length from args
freq = 5

print(f"Using session length: {sec} seconds")

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
    DE_data[idx, :, :, :, :] = data_sub[:, :-2, :sec, :]

# Reshape DE features
DE_data = DE_data.transpose([0, 2, 4, 1, 3]).reshape(n_subs, chn, freq, n_vids*sec)
DE_data = DE_data.transpose([0, 1, 3, 2])
print('DE data shape:', DE_data.shape)  # Should be (n_subs, chn, n_vids*sec, freq)

# Save to .mat file
total_data = {'data': DE_data}
sio.savemat('./data.mat', total_data)
print('data.mat saved')
