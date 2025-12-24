# Wavelet Pipeline

This folder reproduces the DE+SVM workflow using wavelet-based features. Unlike the Fourier-based folders, wavelet processing introduces its own extraction scripts, so everything needed to run the method (transform → feature export → classification) lives here.

## Workflow overview

1. **Continuous Wavelet Transform** – `wavelet_transform.py`
   - Loads raw EEG pickles (default `../Processed_data`).
   - Applies Morlet CWT per subject/video/channel, removes the cone of influence, and averages power within standard EEG bands.
   - Saves a per-subject file `subXXX_wavelet.pkl` under `Wavelet/` (or a custom `--output_path`).

2. **Feature export** – `save_de_wl.py`, `save_psd_new_wl.py`, `save_hjorth_new_wl.py`
   - Read the wavelet pickles and repack them into `.mat` structures compatible with the classifiers.
   - Mimic the naming convention of the Fourier pipelines (`de_features.mat`, `psd_features.mat`, etc.).

3. **Classification** – `run_all.py`
   - Mirrors the `Binary-Classification-Fourier` steps: `pkl_to_mat.py → filter_neutral_videos.py → running_norm.py → smooth_lds.py → main_de_svm.py → plot_result.py`.
   - Supports `--feature {DE,DE_PSD,DE_PSD_H}`, `--session-length`, `--remove-band`, and `--n-vids`.

If you want to sweep multiple removals or session lengths, `run_all_loop.py` works the same way as in folder `03`.

## Helper scripts

- `wavelet.py`, `wavelet2.py`: Prototypes for different wavelet parameterisations; useful when experimenting with other mother wavelets or scale schedules.
- `io_utils.py`, `load_data.py`, `reorder_vids.py`: Copied from the baseline folders to keep the data interfaces consistent.
- `run_all_loop.py`: Automates “all bands vs. band-removed” experiments.

## Typical command sequence

```bash
# 1. Generate wavelet band-power cubes
python wavelet_transform.py --data_path ../Processed_data --output_path ./wavelet_outputs

# 2. Convert to .mat features (pick which ones you need)
python save_de_wl.py --input ./wavelet_outputs
python save_psd_new_wl.py --input ./wavelet_outputs
python save_hjorth_new_wl.py --input ./wavelet_outputs

# 3. Run the classification pipeline (example: 24-video binary task)
python run_all.py --n-vids 24 --session-length 5 --feature DE_PSD
```

Notes:
- The `save_*_wl.py` scripts live here (instead of `Features/`) because they depend on the intermediate wavelet pickles; keeping them nearby reduces path juggling.
- All downstream folders (Binary-Classification-Fourier) can reuse the `.mat` outputs if you prefer to compare Fourier vs. Wavelet features under identical SVM settings.

## 中文說明

`Wavelet/` 提供以小波特徵為核心的完整流程：從連續小波轉換 (CWT)、特徵輸出到 SVM 分類皆集中於此。

### 流程總覽

1. **CWT 轉換 (`wavelet_transform.py`)**
   - 讀取原始 EEG pickle (`../Processed_data`)。
   - 以 Morlet 小波對每位受試者 / 每支影片 / 每個頻道進行 CWT，移除影響錐 (COI)，再於標準頻帶取平均功率。
   - 於 `Wavelet/` 內輸出 `subXXX_wavelet.pkl`。

2. **特徵輸出 (`save_de_wl.py`, `save_psd_new_wl.py`, `save_hjorth_new_wl.py`)**
   - 讀取上一步的 wavelet pickle，轉存為與其他資料夾相容的 `.mat` 格式。
   - 檔名與 Fourier 管線一致 (`de_features.mat` 等)。

3. **分類 (`run_all.py`)**
   - 依序呼叫 `pkl_to_mat.py → filter_neutral_videos.py → running_norm.py → smooth_lds.py → main_de_svm.py → plot_result.py`。
   - 支援 `--feature {DE,DE_PSD,DE_PSD_H}`、`--session-length`、`--remove-band`、`--n-vids`。

`run_all_loop.py` 可一次跑完「保留所有頻帶」與「移除單一頻帶」的比較實驗。

### 指令範例

```bash
python wavelet_transform.py --data_path ../Processed_data --output_path ./wavelet_outputs
python save_de_wl.py --input ./wavelet_outputs
python save_psd_new_wl.py --input ./wavelet_outputs
python save_hjorth_new_wl.py --input ./wavelet_outputs
python run_all.py --n-vids 24 --session-length 5 --feature DE_PSD
```

### 為何把 `save_*_wl.py` 放在此處？

這些腳本仰賴 `wavelet_transform.py` 產生的中介 pickle，與 `Features/` 一般化的 Fourier 特徵路徑不同。放在 `Wavelet/` 便於維護、減少路徑設定，也強調「Wavelet 方法」為獨立模組。如果需要比較 Fourier 與 Wavelet，只要將這裡輸出的 `.mat` 檔交給同一套 SVM 管線即可。
