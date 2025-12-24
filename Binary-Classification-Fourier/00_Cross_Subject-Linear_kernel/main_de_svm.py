import argparse
import numpy as np
import pandas as pd
import torch
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
# parser.add_argument('--dataset', default='both', type=str, help= 'first_batch or second_batch')
parser.add_argument('--valid-method', default='10-folds', type=str, help='the valid method, 10-folds or leave one out')
parser.add_argument('--n-vids', default=28, type=int, help='number of video')
# parser.add_argument('--label-type', default ='cls2', type=str, help='2 classifier or 9 classifier')
parser.add_argument('--train-or-test',default='train',type=str,help='Using for strategy')

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

if n_vids == 24:
    label_type = 'cls2'
elif n_vids == 28:
    label_type = 'cls9'

n_subs = 123
if valid_method == '10-folds':
    n_folds = 10
elif valid_method == 'loo':
    n_folds = n_subs

n_per = round(n_subs / n_folds) # # of sub in each fold 
sec = 30

timeLen = 1
timeStep = 1
isFilt = False
filtLen = 1

# create the weight directory
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
save_dir = './'
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
    data, label_repeat, n_samples = load_srt_de(data, channel_norm, isFilt, filtLen, label_type)
    print('data_dir:', data_dir)
    print(len(label_repeat)) # label shape: 720
    print('data shape:', data.shape) # data shape: (123, 720, 120)

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
        val_seconds = 30 / n_folds ; train_seconds = 30 - val_seconds
        data_list = np.arange(0,len(label_repeat))
        # pick out the val sec
        val_list_start = np.arange(0,len(label_repeat),30) + int(val_seconds * fold)
        val_list = val_list_start.copy()
        for sec in range(1,int(val_seconds)):
            val_list = np.concatenate((val_list, val_list_start + sec)).astype(int)
        train_list = np.array(list(set(data_list) - set(val_list))).astype(int)
        print('length of train list:', train_list.shape)
        print('length of val list:', val_list.shape)
        data_train = data[:,list(train_list),:].reshape(-1, data.shape[-1])
        data_val = data[:,list(val_list),:].reshape(-1, data.shape[-1])
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
                model_save_path = os.path.join('./svm_weights', 'subject_%s_vids_%s_fold_%s_valid_%s.joblib' %(subjects_type,str(n_vids),str(fold),valid_method))
                joblib.dump(clf, model_save_path)
            print('C', C, 'train acc:', train_acc, 'val acc:', val_acc)

        val_acc_folds[fold] = val_acc_best
        best_C_folds[fold] = best_C

        print('best C', best_C, 'best val acc:', val_acc_best)

    if train_or_test == 'test':
        model_path = os.path.join('./svm_weights','subject_%s_vids_%s_fold_%s_valid_%s.joblib' %(subjects_type,str(n_vids),str(fold),valid_method))
        clf2 = joblib.load(model_path)
        subjects_results = clf2.predict(data_val)
        if subjects_type == 'cross':
            subjects_results = subjects_results.reshape(val_sub.shape[0],-1)
            label_val = np.array(label_val).reshape(val_sub.shape[0], -1)
            val_result = [np.sum(subjects_results[i, :] == label_val[i, :]) / subjects_results.shape[1] for i in
                          range(0, val_sub.shape[0])]
            subjects_score[val_sub] = val_result
        elif subjects_type == 'intra':
            subjects_results = subjects_results.reshape(n_subs,-1)
            label_val = np.array(label_val).reshape(n_subs,-1)
            subjects_results_[:,val_list] = subjects_results
            label_val_[:,val_list] = label_val


if train_or_test == 'train':
    print('acc mean: %.3f, std: %.3f' % (np.mean(val_acc_folds), np.std(val_acc_folds)))

elif train_or_test == 'test':
    if subjects_type == 'intra':
        subjects_score = [np.sum(subjects_results_[i, :] == label_val_[i, :]) / subjects_results_.shape[1] for i in range(0, n_subs)]
        subjects_score = np.array(subjects_score).reshape(n_subs,-1)
    pd.DataFrame(subjects_score).to_csv('./subject_%s_vids_%s_valid_%s.csv' %(subjects_type,str(n_vids),valid_method))
