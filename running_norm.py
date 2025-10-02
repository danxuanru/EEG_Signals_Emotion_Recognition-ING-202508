import os
import scipy.io as sio
import numpy as np
from reorder_vids import video_order_load, reorder_vids, reorder_vids_back
import random
import argparse
import time

parser = argparse.ArgumentParser(description='Multi-scale running norm for EEG data with different feature types')
parser.add_argument('--timeLen', default=5, type=int,
                    help='time length in seconds')
parser.add_argument('--use-data', default='de', type=str,
                    help='what data to use')
parser.add_argument('--normTrain', default='yes', type=str,
                    help='whether normTrain')
parser.add_argument('--n-vids', default=28, type=int,
                    help='use how many videos')
parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')
parser.add_argument('--dataset', default='both', type=str,
                    help='first_batch or second_batch')
parser.add_argument('--session-length', default=30, type=int,
                    help='length of each video session in seconds')
parser.add_argument('--feature', default='DE', choices=['DE', 'DE_PSD', 'DE_PSD_H'], help='Feature combination: DE, DE_PSD, or DE_PSD_H')
args = parser.parse_args()

random.seed(args.randSeed)
np.random.seed(args.randSeed)

# Feature configuration
use_features = args.use_data
normTrain = args.normTrain
n_vids = args.n_vids
isCar = True
sec = args.session_length

root_dir = './'
save_dir = os.path.join(root_dir, 'running_norm_'+ str(n_vids))

bn_val = 1
n_total = sec*n_vids
n_counters = int(np.ceil(n_total / bn_val))

n_subs = 123
n_folds = 10
n_per = round(n_subs / n_folds)

chn = 30
fs = 250
freqs_bn = 5
feature = args.feature

# Feature dimensions: Hjorth(10) + DE(5) + PSD(5) = 20 features per channel
hjorth_features = freqs_bn * 2  # 10 features (activity, mobility for 5 freq bands)
de_features = freqs_bn         # 5 features
psd_features = freqs_bn        # 5 features

if feature == 'DE':
    total_features_per_channel = de_features
elif feature == 'DE_PSD':
    total_features_per_channel = de_features + psd_features
elif feature == 'DE_PSD_H':
    total_features_per_channel = de_features + psd_features + hjorth_features

def apply_multiscale_running_norm(data, train_sub, decay_rate=0.990):
    """
    Apply running normalization separately for each feature type
    
    Args:
        data: (n_subs, n_timepoints, n_total_features)
        train_sub: indices of training subjects
        decay_rate: exponential decay rate
    
    Returns:
        normalized data with same shape as input
    """
    n_subs, n_timepoints, n_total_features = data.shape
    
    # Calculate features per channel
    n_features_per_channel = n_total_features // chn
    
    # Feature type boundaries (per channel)
    hjorth_end = hjorth_features
    de_end = hjorth_end + de_features
    psd_end = de_end + psd_features
    
    print(f"Feature boundaries per channel: Hjorth(0-{hjorth_end}), DE({hjorth_end}-{de_end}), PSD({de_end}-{psd_end})")
    
    data_norm = np.zeros_like(data)
    
    # Process each channel separately
    for ch in range(chn):
        print(f"Processing channel {ch+1}/{chn}")
        
        # Extract channel data
        ch_start = ch * n_features_per_channel
        ch_end = (ch + 1) * n_features_per_channel
        ch_data = data[:, :, ch_start:ch_end]
        
        # Split into feature types
        hjorth_data = ch_data[:, :, :hjorth_features]
        de_data = ch_data[:, :, hjorth_features:hjorth_end + de_features] 
        psd_data = ch_data[:, :, hjorth_end + de_features:]
        
        # Calculate baseline statistics for each feature type (training subjects only)
        hjorth_mean = np.mean(np.mean(hjorth_data[train_sub, :, :], axis=1), axis=0)
        hjorth_var = np.mean(np.var(hjorth_data[train_sub, :, :], axis=1), axis=0)
        
        de_mean = np.mean(np.mean(de_data[train_sub, :, :], axis=1), axis=0)
        de_var = np.mean(np.var(de_data[train_sub, :, :], axis=1), axis=0)
        
        psd_mean = np.mean(np.mean(psd_data[train_sub, :, :], axis=1), axis=0)
        psd_var = np.mean(np.var(psd_data[train_sub, :, :], axis=1), axis=0)
        
        print(f"Channel {ch+1} - Hjorth range: [{np.min(hjorth_data):.2e}, {np.max(hjorth_data):.2e}]")
        print(f"Channel {ch+1} - DE range: [{np.min(de_data):.2e}, {np.max(de_data):.2e}]")
        print(f"Channel {ch+1} - PSD range: [{np.min(psd_data):.2e}, {np.max(psd_data):.2e}]")
        
        # Apply running normalization to each subject
        for sub in range(n_subs):
            # Initialize running statistics for each feature type
            hjorth_running_sum = np.zeros(hjorth_features)
            hjorth_running_square = np.zeros(hjorth_features)
            
            de_running_sum = np.zeros(de_features)
            de_running_square = np.zeros(de_features)
            
            psd_running_sum = np.zeros(psd_features)
            psd_running_square = np.zeros(psd_features)
            
            decay_factor = 1.0
            
            for counter in range(n_counters):
                # Extract current time step data for each feature type
                hjorth_one = hjorth_data[sub, counter*bn_val:(counter+1)*bn_val, :]
                de_one = de_data[sub, counter*bn_val:(counter+1)*bn_val, :]
                psd_one = psd_data[sub, counter*bn_val:(counter+1)*bn_val, :]
                
                # Squeeze the time dimension to match running stats arrays
                hjorth_one_squeezed = np.squeeze(hjorth_one, axis=0)
                de_one_squeezed = np.squeeze(de_one, axis=0)
                psd_one_squeezed = np.squeeze(psd_one, axis=0)

                # Update running statistics for Hjorth features
                hjorth_running_sum += hjorth_one_squeezed
                hjorth_running_mean = hjorth_running_sum / (counter + 1)
                hjorth_running_square += hjorth_one_squeezed**2
                hjorth_running_var = (hjorth_running_square - 2 * hjorth_running_mean * hjorth_running_sum) / (counter + 1) + hjorth_running_mean**2
                
                # Update running statistics for DE features
                de_running_sum += de_one_squeezed
                de_running_mean = de_running_sum / (counter + 1)
                de_running_square += de_one_squeezed**2
                de_running_var = (de_running_square - 2 * de_running_mean * de_running_sum) / (counter + 1) + de_running_mean**2
                
                # Update running statistics for PSD features
                psd_running_sum += psd_one_squeezed
                psd_running_mean = psd_running_sum / (counter + 1)
                psd_running_square += psd_one_squeezed**2
                psd_running_var = (psd_running_square - 2 * psd_running_mean * psd_running_sum) / (counter + 1) + psd_running_mean**2
                
                # Calculate current normalization parameters for each feature type
                hjorth_curr_mean = decay_factor*hjorth_mean + (1-decay_factor)*hjorth_running_mean
                hjorth_curr_var = decay_factor*hjorth_var + (1-decay_factor)*hjorth_running_var
                
                de_curr_mean = decay_factor*de_mean + (1-decay_factor)*de_running_mean
                de_curr_var = decay_factor*de_var + (1-decay_factor)*de_running_var
                
                psd_curr_mean = decay_factor*psd_mean + (1-decay_factor)*psd_running_mean
                psd_curr_var = decay_factor*psd_var + (1-decay_factor)*psd_running_var
                
                # Apply feature-specific normalization
                hjorth_norm = (hjorth_one - hjorth_curr_mean) / np.sqrt(hjorth_curr_var + 1e-10)
                de_norm = (de_one - de_curr_mean) / np.sqrt(de_curr_var + 1e-8)
                psd_norm = (psd_one - psd_curr_mean) / np.sqrt(psd_curr_var + 1e-8)
                
                # Combine normalized features back
                combined_norm = np.concatenate([hjorth_norm, de_norm, psd_norm], axis=-1)
                data_norm[sub, counter*bn_val:(counter+1)*bn_val, ch_start:ch_end] = combined_norm
                
                # Update decay factor
                decay_factor *= decay_rate
    
    return data_norm

# Main processing loop
for decay_rate in [0.990]:
    print(f"Processing with decay rate: {decay_rate}")
    for fold in range(n_folds):
        print(f"Processing fold {fold}")
        
        if use_features == 'de':
            data_name = 'data_binary.mat' if n_vids == 24 else 'data.mat'
            data = sio.loadmat(os.path.join(root_dir, data_name))['data']
            print(f"Original data shape: {data.shape}")
            
            # Reshape: (subjects, channels, time_points, features) -> (subjects, time_points, all_features)
            data = data.transpose([0,2,3,1]).reshape(n_subs, n_vids*sec, chn*total_features_per_channel)
            print(f"Reshaped data shape: {data.shape}")
        else:
            # Handle other feature types if needed
            continue
            
        # Define validation and training subjects
        if fold < n_folds-1:
            val_sub = np.arange(n_per*fold, n_per*(fold+1))
        else:
            val_sub = np.arange(n_per*fold, n_per*(fold+1)-1)
        train_sub = list(set(np.arange(n_subs)) - set(val_sub))

        # Load video order
        if n_vids == 24:
            vid_order_path = os.path.join(root_dir, 'vid_order_binary.mat')
            if os.path.exists(vid_order_path):
                vid_order = sio.loadmat(vid_order_path)['vid_order']
            else:
                vid_order = np.tile(np.arange(1, 25), (n_subs, 1))
        else:
            vid_order = video_order_load(args.dataset, 28)
        
        # Reorder videos
        data, vid_play_order_new = reorder_vids(data, vid_order, session_length=sec)
        
        # Handle NaN values
        data[np.isnan(data)] = -30

        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Apply multi-scale running normalization
        print("Applying multi-scale running normalization...")
        data_norm = apply_multiscale_running_norm(data, train_sub, decay_rate)
        
        # Reorder back to original video order
        data_norm_back = reorder_vids_back(data_norm, vid_play_order_new, session_length=sec)
        
        # Save results
        de = {'de': data_norm_back}
        print(f"Final normalized data shape: {data_norm_back.shape}")
        
        # Create save directory
        if isCar:
            save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre_%dvideo_car' % (decay_rate, n_vids))
        else:
            save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre_%dvideo' % (decay_rate, n_vids))
        
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        
        save_file = os.path.join(save_name, 'de_fold%d.mat' % fold)
        print(f"Saving to: {save_file}")
        sio.savemat(save_file, de)

print("Multi-scale running normalization completed!")
