import os
import scipy.io as sio
import numpy as np
from reorder_vids import video_order_load, reorder_vids, reorder_vids_back
import random
import argparse
import time

parser = argparse.ArgumentParser(description='Running norm the EEG data of baseline model')
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

args = parser.parse_args()

random.seed(args.randSeed)
np.random.seed(args.randSeed)


displace = False
use_features = args.use_data
normTrain = args.normTrain
n_vids = args.n_vids
isCar = True
randomInit = False

root_dir = './'
save_dir = os.path.join(root_dir, 'running_norm_'+ str(n_vids))

bn_val = 1
# rn_momentum = 0.995
# print(rn_momentum)
# momentum = 0.9

sec = 15
n_total = sec*n_vids
n_counters = int(np.ceil(n_total / bn_val))

n_subs = 123
n_folds = 10
n_per = round(n_subs / n_folds)
feas = 5
chn = 30

for decay_rate in [0.990]:
    print(decay_rate)
    for fold in range(n_folds):
    # for fold in range(n_folds-1, n_folds):
        print(fold)
        if use_features == 'de':
            # data = sio.loadmat(os.path.join(save_dir, 'deFeature_all.mat'))['deFeature_all']
            data_name = 'data.mat'
            data = sio.loadmat(os.path.join(root_dir, data_name))['data']
            print(data.shape)
            data = data.transpose([0,2,3,1]).reshape(n_subs, 28*sec,  30*feas)
            if n_vids == 24:
                data = np.concatenate((data[:, :12*30, :], data[:, 16*30:, :]), 1)
        elif use_features == 'CoCA':
            if n_vids == 28:
                data = sio.loadmat(os.path.join(save_dir, 'de_CoCA_fold%d.mat' % fold))['de_all']
            elif n_vids == 24:
                data = sio.loadmat(os.path.join(save_dir, 'de_CoCA_fold%d.mat' % fold))['de_all']
        elif use_features == 'SA':
            if n_vids == 28:
                data = sio.loadmat(os.path.join(save_dir, 'de_%d.mat' % fold))['de_all']
            elif n_vids == 24:
                data = sio.loadmat(os.path.join(save_dir, 'de_%d.mat' % fold))['de_all']
        elif (use_features == 'pretrained') or (use_features == 'simseqclr'):
            if normTrain == 'yes':
                data = sio.loadmat(os.path.join(save_dir, str(fold), 'features1_de_1s_normTrain.mat'))['de']
            else:
                data = sio.loadmat(os.path.join(save_dir, str(fold), 'features1_de_1s.mat'))['de']
        print(data.shape)

        if fold < n_folds-1:
            val_sub = np.arange(n_per*fold, n_per*(fold+1))
        else:
            val_sub = np.arange(n_per*fold, n_per*(fold+1)-1)
        train_sub = list(set(np.arange(n_subs)) - set(val_sub))

        vid_order = video_order_load(args.dataset, n_vids)

        data, vid_play_order_new = reorder_vids(data, vid_order)
        print(vid_play_order_new)

        data[np.isnan(data)] = -30
        # data[data<=-30] = -30
        
        data_mean = np.mean(np.mean(data[train_sub, :, :], axis=1), axis=0)
        data_var = np.mean(np.var(data[train_sub, :, :], axis=1), axis=0)
        
        data_norm = np.zeros_like(data)
        for sub in range(data.shape[0]):
            running_sum = np.zeros(data.shape[-1])
            running_square = np.zeros(data.shape[-1])
            decay_factor = 1.
            start_time = time.time()
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
            end_time = time.time()
            # print('time consumed: %.3f, counter: %d' % (end_time-start_time, counter+1))

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

            

