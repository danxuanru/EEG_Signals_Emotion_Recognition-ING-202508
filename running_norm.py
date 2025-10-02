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

# Feature dimensions: Hjorth(15) + DE(5) + PSD(5) = 25 features per channel
hjorth_features = freqs_bn * 3  # 15 features (activity, mobility, complexity for 5 freq bands)
de_features = freqs_bn         # 5 features
psd_features = freqs_bn        # 5 features

if feature == 'DE':
    total_features_per_channel = de_features
elif feature == 'DE_PSD':
    total_features_per_channel = de_features + psd_features
elif feature == 'DE_PSD_H':
    total_features_per_channel = de_features + psd_features + hjorth_features

def running_normalization(data, decay_rate):
    data_mean = np.mean(np.mean(data[train_sub, :, :], axis=1), axis=0)
    data_var = np.mean(np.var(data[train_sub, :, :], axis=1), axis=0)
    
    data_norm = np.zeros_like(data)
    for sub in range(data.shape[0]):
        running_sum = np.zeros(data.shape[-1])
        running_square = np.zeros(data.shape[-1])
        decay_factor = 1.
        # start_time = time.time()
        for counter in range(n_counters):
            data_one = data[sub, counter*bn_val: (counter+1)*bn_val, :]
            running_sum = running_sum + data_one
            running_mean = running_sum / (counter+1)
            # running_mean = counter / (counter+1) * running_mean + 1/(counter+1) * data_one
            running_square = running_square + data_one**2
            running_var = (running_square - 2 * running_mean * running_sum) / (counter+1) + running_mean**2

            # print(decay_factor)
            curr_mean = decay_factor*data_mean + (1-decay_factor)*running_mean
            curr_var = decay_factor*data_var + (1-decay_factor)*running_var
            decay_factor = decay_factor*decay_rate

            # print(running_var[:3])
            # if counter >= 2:
            data_one = (data_one - curr_mean) / np.sqrt(curr_var + 1e-5)
            data_norm[sub, counter*bn_val: (counter+1)*bn_val, :] = data_one
        # end_time = time.time()
        # print('time consumed: %.3f, counter: %d' % (end_time-start_time, counter+1))
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
        # data[data<=-30] = -30
        
        # Calculate running normalization
        data_norm = running_normalization(data, decay_rate)

        # Reorder videos back to original order
        data_norm = reorder_vids_back(data_norm, vid_play_order_new)
        de = {'de': data_norm}
        print(data_norm.shape)
        if (use_features == 'de') or (use_features == 'CoCA'):
            if n_vids == 28:
                if isCar:
                    save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre_%dvideo_car' % (decay_rate, n_vids))
                else:
                    save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre_%dvideo' % (decay_rate, n_vids))
                if not os.path.exists(save_name):
                    os.makedirs(save_name)
                save_file = os.path.join(save_name, 'de_fold%d.mat' % fold)
            elif n_vids == 24:
                if isCar:
                    save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre_%dvideo_car' % (decay_rate, n_vids))
                else:
                    save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre_%dvideo' % (decay_rate, n_vids))
                if not os.path.exists(save_name):
                    os.makedirs(save_name)
                save_file = os.path.join(save_name, 'de_fold%d.mat' % fold)
            print(save_file)
            sio.savemat(save_file, de)

            

