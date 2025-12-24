import numpy as np
import os
import pickle
import argparse
import pycwt as wavelet
from matplotlib import pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='PSD feature extraction for EEG data')
parser.add_argument('--data_path', default='../Processed_data', type=str, help='Path to the EEG data')
parser.add_argument('--output_path', default='../Wavelet', type=str, help='Path to save the PSD features')
args = parser.parse_args()

freq_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 47)
}

def plot_freq_spectrum(power, freqs, coi_freq, time, freq_bands):
    
    # plot the power spectrum
    fig, ax = plt.subplots(figsize=(10, 6))

    # create power colormap
    im = ax.contourf(
        time, 
        freqs, 
        np.log10(power), 
        levels=50, 
        cmap='jet', 
        extend='both'
    )

    ax.set_yscale('log') # 由於頻率通常呈對數分佈，設為對數坐標
    ax.set_ylim([freqs[-1], freqs[0]]) # 調整y軸範圍
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Wavelet Power Spectrum')
    cbar = fig.colorbar(im)
    cbar.set_label('log(Power)')

    # coi range
    ax.fill_between(
        time,
        freqs.min(), 
        coi_freq,  # Fill area between min frequency and COI frequency
        color='white',
        alpha=0.5,
        hatch='//',
        edgecolor='black'
    )

    # mark standard EEG bands
    for band_name, (f_low, f_high) in freq_bands.items():
        if f_low >= freqs.min() and f_high <= freqs.max():
            ax.axhline(y=f_low, color='white', linestyle=':', alpha=0.7, linewidth=1)
            ax.axhline(y=f_high, color='white', linestyle=':', alpha=0.7, linewidth=1)
            ax.text(30*0.02, (f_low + f_high)/2, band_name.capitalize(), 
                   color='white', fontsize=10, weight='bold')
    
    ax.legend(loc='upper right', fontsize='small', framealpha=0.5)
    plt.show()

    # save figure as png
    fig.savefig(f"wavelet_power_spectrum.png", dpi=300, bbox_inches='tight')

def plot_period_spectrum(power, period, coi):
    # plot the power spectrum
    fig, ax = plt.subplots(figsize=(10, 6))

    # create power colormap
    im = ax.pcolormesh(
        np.arange(N),
        period,
        power,
        shading='auto',
        cmap='jet'
    )

    ax.set_yscale('log') # 由於週期通常呈對數分佈，設為對數坐標
    ax.set_ylim([period.min(), period.max()]) # 調整y軸範圍：週期從小到大
    ax.set_xlabel('Time')
    ax.set_ylabel('Period (s)')
    ax.set_title('Wavelet Power Spectrum (Period)')
    cbar = fig.colorbar(im)
    cbar.set_label('Power')

    # coi range - 在週期圖中，COI以上的區域（大週期值）是不可信的
    ax.fill_between(
        np.arange(N),
        coi,  
        period.max(),   # Fill area between COI to max period (上方區域)
        color='white',
        alpha=0.5,
        hatch='//',
        edgecolor='black'
    )

    plt.show()

# load the data
data_path = args.data_path
data_paths = os.listdir(data_path)
data_paths.sort()
n_vids = 28
chn = 32
fs = 250
sec = 30

# load data
print(f"Processing Subject 1: {data_paths[0]}")  # test only the first subject
with open(os.path.join(data_path, data_paths[0]), 'rb') as f:
    data_sub = pickle.load(f)

data = data_sub[0, 0, :]  # test only the first video and first channel
print("Data shape:", data.shape)  # (samples,)
N = data.shape[0]

# normalization
data = (data - np.mean(data)) / np.std(data)

# Wavelet parameters
mother_wavelet = wavelet.Morlet(6)  # Morlet wavelet with wavenumber 6
dt = 1 / fs  # Time step
dj = 0.25  # Scale resolution (Spacing between discrete scales)
s0 = 2 * dt  # smallest scale
j1 = 7 / dj  # Number of scales (9 octaves with dj sub-octaves)

# Wavelet Transform
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data, dt, dj, s0, j1)

# print("Wavelet shape:", wave.shape)  # (scales, time)
# print("Scales:", scales)  
# print("Scale size:", scales.size)
# print("Frequencies:", freqs)
# print("Frequencies size:", freqs.size)
# print("COI:", coi)  # Cone of Influence: 可信結果的最小週期(最大頻率)
# print("COI shape:", coi.shape)
# print("FFT Frequencies:", fftfreqs)

coi_freq = 1 / coi  # Convert COI periods to frequencies

# power = np.abs(wave) ** 2  # Compute Power Spectrum
power = np.abs(wave) ** 2  # Log-transform Power Spectrum


# 去除COI區域
coi_freq = 1 / coi  # Convert COI periods to frequencies
for i in range(power.shape[1]):
    bad_freq_idx = np.where(freqs < coi_freq[i])[0]  # 
    power[bad_freq_idx, i] = np.nan  # Set power values outside COI to NaN

print("Power after removing COI shape:", power.shape)

# get frequency range
freq_range = (freqs[0], freqs[-1])
print("Frequency range:", freq_range)

# Computing frequency band powers
band_powers = {}
for band, (fmin, fmax) in freq_bands.items():
    # Get frequency indices
    freq_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    band_powers[band] = np.nanmean(power[freq_idx, :], axis=0)

print("Delta band (0.5-4 Hz):")
print("\tFrequency points: ", len(np.where((freqs >= 0.5) & (freqs <= 4))[0]))
print("\tMean power: ", np.nanmean(band_powers['Delta']))
print("Theta band (4-8 Hz):")
print("\tFrequency points: ", len(np.where((freqs >= 4) & (freqs <= 8))[0]))
print("\tMean power: ", np.nanmean(band_powers['Theta']))
print("Alpha band (8-13 Hz):")
print("\tFrequency points: ", len(np.where((freqs >= 8) & (freqs <= 13))[0]))
print("\tMean power: ", np.nanmean(band_powers['Alpha']))
print("Beta band (13-30 Hz):")
print("\tFrequency points: ", len(np.where((freqs >= 13) & (freqs <= 30))[0]))
print("\tMean power: ", np.nanmean(band_powers['Beta']))
print("Gamma band (30-47 Hz):")
print("\tFrequency points: ", len(np.where((freqs >= 30) & (freqs <= 47))[0]))
print("\tMean power: ", np.nanmean(band_powers['Gamma']))

# plot the frequency spectrum
time = np.arange(N)*dt
plot_freq_spectrum(power, freqs, coi_freq, time, freq_bands)

