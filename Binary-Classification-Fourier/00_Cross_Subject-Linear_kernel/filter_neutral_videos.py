import os
import scipy.io as sio
import numpy as np
from reorder_vids import video_order_load
import argparse
import time

parser = argparse.ArgumentParser(description='Filter out neutral emotion videos for binary classification')
parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')
parser.add_argument('--use-data', default='de', type=str,
                    help='what data to use')
parser.add_argument('--dataset', default='both', type=str,
                    help='first_batch or second_batch')
parser.add_argument('--input-file', default='data.mat', type=str,
                    help='input data file')
parser.add_argument('--output-file', default='data_binary.mat', type=str,
                    help='output data file without neutral videos')

args = parser.parse_args()

# Set random seeds for reproducibility
np.random.seed(args.randSeed)

# Constants
sec = 15  # seconds per video
n_subs = 123  # number of subjects
chn = 30

# Load the video order for all subjects
print("Loading video order...")
vid_orders = video_order_load(args.dataset, 28)
print(f"Video order loaded with shape: {vid_orders.shape}")

# Load the data file
print(f"Loading data from {args.input_file}...")
data = sio.loadmat(os.path.join('./', args.input_file))['data']
print(f"Original data shape: {data.shape}")

# Process each subject to remove neutral videos (13-16)
filtered_data = []
filtered_vid_orders = np.zeros((n_subs, 24))  # To store filtered video orders

for sub_idx in range(n_subs):
    # Get this subject's video order
    sub_order = vid_orders[sub_idx, :]
    
    # Find indices of neutral videos (13-16) in this subject's order
    neutral_indices = []
    for i, vid_num in enumerate(sub_order):
        if 13 <= vid_num <= 16:
            neutral_indices.append(i)
    
    print(f"Subject {sub_idx+1}: Neutral videos at positions {neutral_indices}")
    
    # Extract the subject's data
    sub_data = data[sub_idx]  # (chn, n_vids*sec, features)
    print(f"Subject {sub_idx+1} data shape before filtering: {sub_data.shape}")

    # Reshape to separate videos
    sub_data_by_vid = sub_data.reshape(chn, 28, sec, -1)
    print(f"Subject {sub_idx+1} data shape after reshaping: {sub_data_by_vid.shape}")
    
    # Remove the neutral videos
    mask = np.ones(28, dtype=bool)
    mask[neutral_indices] = False
    filtered_sub_data = sub_data_by_vid[:, mask]
    print(f"Subject {sub_idx+1} data shape after filtering: {filtered_sub_data.shape}")
    
    # Create filtered video order by removing neutral videos and adjusting the numbering
    filtered_order = np.delete(sub_order, neutral_indices)
    
    # Adjust video numbers: videos 1-12 stay the same, videos 17-28 become 13-24
    for i in range(len(filtered_order)):
        if filtered_order[i] >= 17:
            filtered_order[i] -= 4
    
    filtered_vid_orders[sub_idx] = filtered_order
    
    # Reshape back to original format but with fewer videos
    filtered_sub_data = filtered_sub_data.reshape(chn, 24*sec, -1)
    print(f"Subject {sub_idx+1} filtered data shape after reshaping: {filtered_sub_data.shape}")
    
    # Add to our list of filtered data
    filtered_data.append(filtered_sub_data)

# Stack all subjects' data back together
filtered_data = np.stack(filtered_data)
print(f"Filtered data shape: {filtered_data.shape}")

# Save the filtered data
output_dict = {'data': filtered_data}
output_path = os.path.join('./', args.output_file)
print(f"Saving filtered data to {output_path}...")
sio.savemat(output_path, output_dict)

# Save the filtered video orders
vid_order_path = os.path.join('./', 'vid_order_binary.mat')
print(f"Saving filtered video orders to {vid_order_path}...")
sio.savemat(vid_order_path, {'vid_order': filtered_vid_orders})

print("Done!")
