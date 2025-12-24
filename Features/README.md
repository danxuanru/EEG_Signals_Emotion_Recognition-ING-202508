# Features Toolkit

Utility scripts in this folder convert the raw pickled EEG cubes stored under `Processed_data/` (or feature-specific export directories) into `.mat` files that the classification pipelines expect. Each script focuses on a specific feature family so you can mix and match them in later experiments.

## Script overview

| Script | Description | Primary outputs |
| --- | --- | --- |
| `save_de.py` | Recomputes classical Differential Entropy (DE) statistics across the five canonical EEG bands (delta→gamma) and writes `de_features.mat`. | `./de_features.mat` |
| `save_psd.py` / `save_psd_new.py` | Extracts Power Spectral Density windows; the `*_new` variant follows the updated directory layout used by folders `01–04`. | `./psd_features.mat` or `./psd_new/*.mat` |
| `save_hjorth.py` / `save_hjorth_new.py` | Computes Hjorth parameters (Activity, Mobility, Complexity); the `*_new` script mirrors the newer saving convention. | `./hjorth_features.mat` |

### Common usage pattern

1. Point the script to the source directory if you changed where processed EEG pickles are stored (default: `../../Processed_data`).
2. Adjust any window length, channel selection, or fold arguments inside the script if you only need a subset of the data.
3. Run the script in this folder:
   ```bash
   python save_de.py
   python save_psd_new.py
   python save_hjorth_new.py
   ```
4. Copy or reference the resulting `.mat` files from the downstream classification folder (e.g., `Binary-Classification-Fourier/00_*/`).

Because these scripts are self-contained, you can re-run them whenever the upstream preprocessing changes without touching the classification code.

## 中文說明

`Features/` 內的腳本負責把預先處理好的 EEG pickle 檔轉成各分類流程所需的 `.mat` 特徵。每支腳本專注在一種特徵：

| 腳本 | 功能 | 主要輸出 |
| --- | --- | --- |
| `save_de.py` | 重新計算五個 EEG 頻帶的微分熵 (DE)，輸出 `de_features.mat`。 | `./de_features.mat` |
| `save_psd.py` / `save_psd_new.py` | 擷取功率譜密度 (PSD) 視窗；`*_new` 版本符合新版資料夾架構。 | `./psd_features.mat` 或 `./psd_new/*.mat` |
| `save_hjorth.py` / `save_hjorth_new.py` | 計算 Hjorth 參數 (Activity/Mobility/Complexity)；`*_new` 版本採用更新的命名方式。 | `./hjorth_features.mat` |

### 建議流程

1. 如有需要，先修改腳本中的原始資料路徑 (預設 `../../Processed_data`)。
2. 若只想處理部分秒數或頻道，直接在腳本中調整相關設定。
3. 在此資料夾依序執行：
   ```bash
   python save_de.py
   python save_psd_new.py
   python save_hjorth_new.py
   ```
4. 將產生的 `.mat` 檔複製或直接引用到分類管線 (例如 `Binary-Classification-Fourier/00_*`)。

所有腳本彼此獨立，當上游預處理改變時可重新執行而不影響分類程式。
