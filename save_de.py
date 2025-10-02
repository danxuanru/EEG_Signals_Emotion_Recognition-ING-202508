import numpy as np
import scipy.io as sio
import pickle
import mne
import os


# Load the data
data_path = '../Processed_data'
data_paths = os.listdir(data_path)
data_paths.sort()
n_vids = 28;
chn = 30;
fs = 250;
sec = 30;
data = np.zeros((len(data_paths), n_vids, chn, fs * sec))
for idx, path in enumerate(data_paths):
    f = open(os.path.join(data_path, path), 'rb')
    data_sub = pickle.load(f)
    data[idx, :, :, :] = data_sub[:, :-2, :]

# data shape :(sub, vid, chn, fs * sec)
print('data loaded:', data.shape)


n_subs = data.shape[0]

fs = 250
freqs = [[1,4], [4,8], [8,14], [14,30], [30,47]]

de = np.zeros((n_subs, 30, 28*30, len(freqs)))
for i in range(len(freqs)):
    print('Current freq band: ', freqs[i])
    for sub in range(n_subs):
        for j in range(28):
            data_video = data[sub, j, :, :]
            print(data_video.shape)
            low_freq = freqs[i][0]
            high_freq = freqs[i][1]
            data_video_filt = mne.filter.filter_data(data_video, fs, l_freq=low_freq, h_freq=high_freq)
            data_video_filt = data_video_filt.reshape(30, -1, fs)
            de_one = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data_video_filt, 2)))
            # n_subs, 30, 28*30, freqs
            de[sub, :, 30*j: 30*(j+1), i] = de_one

    
print(de.shape)
de = {'de': de}
sio.savemat('./data.mat', de)
