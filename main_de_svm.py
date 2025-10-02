import argparse
import numpy as np
import pandas as pd
import os
import scipy.io as sio
from load_data import load_srt_de
import random
import time
from sklearn.svm import LinearSVC
import joblib

parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')
parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')
parser.add_argument('--n-iters', default=2000, type=int,
                    help='number of trees in GBT')
parser.add_argument('--training-fold', default='all', type=str,
                    help='the number of training fold, 0~9,and 9 for the subs leaf')
parser.add_argument('--subjects-type', default='cross', type=str,
                    help='cross or intra subject')
parser.add_argument('--valid-method', default='10-folds', type=str, help='the valid method, 10-folds or leave one out')
parser.add_argument('--n-vids', default=24, type=int, help='number of video')
parser.add_argument('--train-or-test',default='train',type=str,help='Using for strategy')
parser.add_argument('--session-length', default=30, type=int, help='length of each video session in seconds')

args = parser.parse_args()
train_or_test = args.train_or_test

random.seed(args.randSeed)
np.random.seed(args.randSeed)
print('n_iters:', args.n_iters)

C_cands = 10.**np.arange(-5,1,0.5)
print('C: ', C_cands)

pretrained = False
use_features = 'de'
channel_norm = True
subjects_type = args.subjects_type
valid_method = args.valid_method
n_vids = args.n_vids
sec = args.session_length  # Use the session length from args


if n_vids == 24:
    label_type = 'cls2'
elif n_vids == 28:
    label_type = 'cls9'

n_subs = 123
if valid_method == '10-folds':
    n_folds = 10
elif valid_method == 'loo':
    n_folds = n_subs

n_per = round(n_subs / n_folds)

timeLen = 1
timeStep = 1
isFilt = False
filtLen = 1

# create save directory
save_dir = os.path.join('./', 'svm_weights')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# It works only when valid method is 10-folds
if args.training_fold == 'all':
    folds_list = np.arange(0, n_folds)
else:
    # training_fold = 0~9
    folds_list = [int(args.training_fold)]


root_dir = './smooth_' + str(n_vids)
val_acc_folds = np.zeros(n_folds)
best_C_folds = np.zeros(n_folds)
if train_or_test == 'test':
    subjects_score = np.zeros((n_subs))
    if subjects_type == 'intra':
        subjects_results_ = np.zeros((n_subs,sec * n_vids))
        label_val_ = np.zeros((n_subs,sec * n_vids))

# when val method is loo ,folds_list is the range of n_subs
for fold in folds_list:
    print('fold', fold)
    data_dir = os.path.join(root_dir,'de_lds_fold%d.mat' % (fold))
    data = sio.loadmat(data_dir)['de_lds']
    data, label_repeat, n_samples = load_srt_de(data, channel_norm, isFilt, filtLen, label_type, session_length=sec)
    print('data_dir:', data_dir)
    print(len(label_repeat)) # label shape: dependent on session length
    print('original data shape:', data.shape)

    # Reshape if needed
    if data.shape[1] == n_vids and data.shape[2] == 30:
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2], -1)
        print('reshaped data shape:', data.shape)

    if subjects_type == 'cross':
        if fold < n_folds - 1:
            val_sub = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_sub = np.arange(n_per * fold, n_subs)
        # print('val', val_sub)
        train_sub = np.array(list(set(np.arange(n_subs)) - set(val_sub)))
        # print('train', train_sub)

        data_train = data[list(train_sub), :, :].reshape(-1, data.shape[-1])
        data_val = data[list(val_sub), :, :].reshape(-1, data.shape[-1])

        label_train = np.tile(label_repeat, len(train_sub))
        label_val = np.tile(label_repeat, len(val_sub))
        print('train', data_train.shape, label_train.shape)
        print('val', data_val.shape, label_val.shape)

    elif subjects_type == 'intra':
        # 1. Calculate validation and training data time lengths
        val_seconds = sec / n_folds  # validation seconds per fold
        train_seconds = sec - val_seconds  # remaining seconds for training
        data_list = np.arange(0,len(label_repeat))
        
        # 2. Select validation data indices
        # For each 'sec' length video, select specific seconds for validation
        # If n_folds=10 and sec=15, each fold takes 1.5 seconds
        val_list_start = np.arange(0,len(label_repeat),sec) + int(val_seconds * fold)
        val_list = val_list_start.copy()
        
        # 3. Create complete validation data index list
        # Add all consecutive seconds in validation interval
        for sec_idx in range(1,int(val_seconds)):
            val_list = np.concatenate((val_list, val_list_start + sec_idx)).astype(int)
        
        # Handle fractional seconds if val_seconds is not an integer
        if val_seconds % 1 > 0:
            # Add partial second at the end of each validation segment
            fractional_idx = int(val_seconds)
            fractional_count = int((val_seconds % 1) * len(val_list_start))
            if fractional_count > 0:
                additional_indices = val_list_start[:fractional_count] + fractional_idx
                val_list = np.concatenate((val_list, additional_indices)).astype(int)
        
        # 4. Find training data indices by set difference
        train_list = np.array(list(set(data_list) - set(val_list))).astype(int)
        print('length of train list:', train_list.shape)
        print('length of val list:', val_list.shape)
        
        # 5. Extract training and validation data
        data_train = data[:,list(train_list),:].reshape(-1, data.shape[-1])
        data_val = data[:,list(val_list),:].reshape(-1, data.shape[-1])
        
        # 6. Create corresponding labels for training and validation data
        label_train = np.tile(np.array(label_repeat)[train_list], n_subs)
        label_val = np.tile(np.array(label_repeat)[val_list], n_subs)
        print('train', data_train.shape, label_train.shape)
        print('val', data_val.shape, label_val.shape)

    start_time = time.time()
    val_acc_best = 0

    if train_or_test == 'train':
        for C in C_cands:
            clf = LinearSVC(random_state=args.randSeed, C=C).fit(data_train, label_train)
            preds_train = clf.predict(data_train)
            preds_val = clf.predict(data_val)
            end_time = time.time()
            print('time consumed:', end_time - start_time)

            train_acc = np.sum(preds_train==label_train) / len(label_train)
            val_acc = np.sum(preds_val==label_val) / len(label_val)

            if val_acc > val_acc_best:
                val_acc_best = val_acc
                best_C = C
                model_save_path = os.path.join(save_dir, 'subject_%s_vids_%s_fold_%s_valid_%s.joblib' %(subjects_type,str(n_vids),str(fold),valid_method))
                joblib.dump(clf, model_save_path)
            print('C', C, 'train acc:', train_acc, 'val acc:', val_acc)

        val_acc_folds[fold] = val_acc_best
        best_C_folds[fold] = best_C

        print('best C', best_C, 'best val acc:', val_acc_best)

    if train_or_test == 'test':
        model_path = os.path.join(save_dir,'subject_%s_vids_%s_fold_%s_valid_%s.joblib' %(subjects_type,str(n_vids),str(fold),valid_method))
        clf2 = joblib.load(model_path)
        subjects_results = clf2.predict(data_val)
        print("subjects_results.shape: ", subjects_results.shape)
        if subjects_type == 'cross':
            # 確保 subjects_results 和 label_val 形狀一致
            subjects_results = subjects_results.reshape(len(val_sub), -1)
            label_val_reshaped = np.array(label_val).reshape(len(val_sub), -1)
            print("after reshape - subjects_results.shape: ", subjects_results.shape)
            # 計算每個受試者的準確率
            val_result = [np.sum(subjects_results[i, :] == label_val_reshaped[i, :]) / subjects_results.shape[1] 
                          for i in range(len(val_sub))]
            subjects_score[val_sub] = val_result
        elif subjects_type == 'intra':
            # 同樣需要重塑
            subjects_results = subjects_results.reshape(n_subs, -1)
            label_val_reshaped = np.array(label_val).reshape(n_subs, -1)
            subjects_results_[:, val_list] = subjects_results
            label_val_[:, val_list] = label_val_reshaped


if train_or_test == 'train':
    print('acc mean: %.3f, std: %.3f' % (np.mean(val_acc_folds), np.std(val_acc_folds)))

elif train_or_test == 'test':
    if subjects_type == 'intra':
        subjects_score = [np.sum(subjects_results_[i, :] == label_val_[i, :]) / subjects_results_.shape[1] for i in range(0, n_subs)]
        subjects_score = np.array(subjects_score).reshape(n_subs,-1)
