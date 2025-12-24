import argparse
import numpy as np
import torch
import os
import scipy.io as sio
import pickle
import random

def LDS(sequence):
    # print(sequence.shape) # (30, 256)

    # sequence_new = np.zeros_like(sequence) # (30, 256)
    ave = np.mean(sequence, axis=0)  # [256,]
    u0 = ave
    X = sequence.transpose((1, 0))  # [256, 30]

    V0 = 0.01
    A = 1
    T = 0.0001
    C = 1
    sigma = 1

    [m, n] = X.shape  # (1, 30)
    P = np.zeros((m, n))  # (1, 1, 30) dia
    u = np.zeros((m, n))  # (1, 30)
    V = np.zeros((m, n))  # (1, 1, 30) dia
    K = np.zeros((m, n))  # (1, 1, 30)

    K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * np.ones((m,))
    u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
    V[:, 0] = (np.ones((m,)) - K[:, 0] * C) * V0

    for i in range(1, n):
        P[:, i - 1] = A * V[:, i - 1] * A + T
        K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
        u[:, i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
        V[:, i] = (np.ones((m,)) - K[:, i] * C) * P[:, i - 1]

    X = u

    return X.transpose((1, 0))


parser = argparse.ArgumentParser(description='Smooth the EEG data of baseline model')

parser.add_argument('--use-data', default='de', type=str,
                    help='what data to use')
parser.add_argument('--n-vids', default=28, type=int,
                    help='use how many videos')
parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')
parser.add_argument('--smooth-length', default=30, type=int,
                    help='the length for lds smooth')
parser.add_argument('--dataset', default='both', type=str,
                    help='first_batch or second_batch')
parser.add_argument('--session-length', default=30, type=int,
                    help='length of each video session in seconds')
args = parser.parse_args()


random.seed(args.randSeed)
np.random.seed(args.randSeed)
n_vids = args.n_vids
sec = args.session_length  # Use the session length from args

root_dir = './running_norm_%s/normTrain_rnPreWeighted0.990_newPre_%svideo_car' % (n_vids, n_vids)
save_dir = './smooth_' + str(n_vids)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

n_subs = 123

n_folds = 10
n_per = round(n_subs/n_folds)
n_length = args.smooth_length

for fold in range(n_folds):
    data_dir = os.path.join(root_dir, 'de_fold' + str(fold) + '.mat')
    feature_de_norm = sio.loadmat(data_dir)['de']
    subs_feature_lds = np.ones_like(feature_de_norm)
    for sub in range(n_subs):
        subs_feature_lds[sub, : ,:] = LDS(feature_de_norm[sub,:,:])
    de_lds = {'de_lds': subs_feature_lds}
    save_file = os.path.join(save_dir, 'de_lds_fold' + str(fold) + '.mat')
    print(save_file)
    print(de_lds['de_lds'].shape)
    sio.savemat(save_file, de_lds)



