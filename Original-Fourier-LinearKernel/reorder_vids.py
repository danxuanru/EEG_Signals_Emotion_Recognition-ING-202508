import scipy.io as sio
import numpy as np
import os
import copy
# import hdf5storage

sec = 15

def video_order_load(dataset,n_vids):
    datapath = '../../After_remarks'
    filesPath = os.listdir(datapath)
    filesPath.sort()
    vid_orders = np.zeros((len(filesPath), n_vids))

    for idx, file in enumerate(filesPath):
        # Here don't forget to arange the subjects arrangement
        # print(file)
        remark_file = os.path.join(datapath,file,'After_remarks.mat')
        #print("file: ", remark_file)
        try:
            # 使用 scipy.io.loadmat 替代 hdf5storage.loadmat
            subject_remark = sio.loadmat(remark_file)['After_remark']
            #print("sub", idx, pd.DataFrame(subject_remark).head())
            vid_orders[idx, :] = [float(subject_remark[vid][0][2]) for vid in range(0, n_vids)]
        except Exception as e:
            print(f"Error loading {remark_file}: {str(e)}")
            continue
    #print('vid_order shape: ', vid_orders.shape)
    #print(pd.DataFrame(vid_orders).head())
    return vid_orders


def reorder_vids(data, vid_play_order):
    # data: (n_subs, n_points, n_feas)
    n_vids = int(data.shape[1] / sec)
    n_subs = data.shape[0]
    # Deep copy
    vid_play_order_copy = vid_play_order.copy()

    if n_vids == 24:
        vid_play_order_new = np.zeros((n_subs, n_vids)).astype(np.int32)
        data_reorder = np.zeros_like(data)
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:]
            # tmp = tmp[(tmp<13)|(tmp>16)]
            tmp[tmp>=17] = tmp[tmp>=17] - 4
            tmp = tmp - 1
            vid_play_order_new[sub, :] = tmp

            print('video arrange:', tmp)

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, sec, data_sub.shape[-1])
            # Error occurs saying that the elements of tmp is not int
            tmp = [int(i) for i in tmp]
            data_sub = data_sub[tmp, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(n_vids*sec, data_sub.shape[-1])

    elif n_vids == 28:
        vid_play_order_new = np.zeros((n_subs, n_vids)).astype(np.int32)
        data_reorder = np.zeros_like(data)

        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:]
            tmp = tmp - 1
            vid_play_order_new[sub, :] = tmp

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, sec, data_sub.shape[-1])
            # Error occurs saying that the elements of tmp is not int
            tmp = [int(i) for i in tmp]
            data_sub = data_sub[tmp, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(n_vids*sec, data_sub.shape[-1])

    return data_reorder, vid_play_order_new



def reorder_vids_back(data, vid_play_order_new):
    # data: (n_subs, n_points, n_feas)
    n_vids = int(data.shape[1] / sec)
    n_subs = data.shape[0]

    data_back = np.zeros((n_subs, n_vids, sec, data.shape[-1]))

    for sub in range(n_subs):
        data_sub = data[sub, :, :].reshape(n_vids, sec, data.shape[-1])
        data_back[sub, vid_play_order_new[sub, :], :, :] = data_sub
    data_back = data_back.reshape(n_subs, n_vids*sec, data.shape[-1])
    return data_back
