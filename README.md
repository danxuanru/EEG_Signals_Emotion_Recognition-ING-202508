# EEG_Signals_Emotion_Recognition-ING-202508

End-to-end pipelines for EEG-based emotion recognition. The repository contains Differential Entropy (DE)-centric Fourier pipelines, a Wavelet alternative, reusable feature extractors, and statistical analysis utilities. Most experiments target binary (positive/negative) and nine-class (9 emotions) setups using Support Vector Machines (SVMs).

## Quick Start

1. **Environment**
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2. **Place processed EEG pickles** under `Processed_data/` (path referenced by the feature scripts). Adjust script arguments if your data lives elsewhere. Dataset download: [FACED EEG Repository](https://www.synapse.org/Synapse:syn50614194/files/).
3. **Extract features** (DE / PSD / Hjorth, or wavelet-specific features) using the scripts inside `Features/` or `Wavelet/`.
4. **Run a pipeline** (e.g., `Binary-Classification-Fourier/00_Cross_Subject-Linear_kernel/run_all.py`) to perform normalization, smoothing, SVM training/testing, and plotting.
5. **Validate results** with the statistical tools in `Statistics/` when comparing multiple experiment settings.

## Repository Layout

| Path | Description |
| --- | --- |
| `Binary-Classification-Fourier/` | Main Fourier-based pipelines for binary emotion tasks. Subfolders `00`–`04` capture iterative improvements (baseline, intra-subject, RBF kernels, frequency-band studies, and AsI-filter enhancements). Each has its own README with usage details. |
| `Nine-Classification-Fourier/` | Equivalent workflows for nine-class emotion recognition. Mirrors the scripts from the binary folder but configured for all 28 videos. |
| `Original-Fourier-LinearKernel/` | Legacy linear-kernel implementation retained for reproducibility and comparisons against the refactored folders. |
| `Wavelet/` | Complete wavelet-based alternative: `wavelet_transform.py` → `save_*_wl.py` → `run_all.py`. Supports DE / PSD / Hjorth combinations derived from wavelet band power. |
| `Features/` | Standalone feature exporters (DE, PSD, Hjorth). They ingest pickled EEG cubes and output `.mat` files that every pipeline can reuse. |
| `Statistics/` | Post-processing toolbox with ANOVA, Friedman, Wilcoxon, paired t-tests, summary generators, and result-merging helpers. |
| `Code_Rearchiture/` | Placeholder for future refactoring work. Currently empty. |
| `requirements.txt` | Python dependency list shared across all pipelines. |

## Workflow Highlights

- **Feature extraction**: choose Fourier (`Features/`) or Wavelet (`Wavelet/`) scripts depending on the experiment.
- **Normalization & smoothing**: `running_norm*.py` and `smooth_lds.py` handle per-fold scaling and LDS smoothing across folders for reproducibility.
- **Classification**: `main_de_svm.py` (linear) and `main_svm_asi_voting.py` / RBF variants provide exhaustive SVM grid search, model saving, and fold-wise evaluation.
- **Visualization**: `plot_result.py` scripts summarize accuracy curves per experiment.
- **Statistics**: merge fold outputs and run hypothesis tests before publishing comparisons.

## 中文說明

本專案提供完整的 EEG 情緒辨識流程，涵蓋以微分熵 (DE) 為核心的傅立葉方法與小波 (Wavelet) 方法，並附上特徵擷取與統計分析工具。主要任務為二元 (正/負) 與九分類情緒，分類器多以 SVM 為主。

### 快速開始

1. **建立環境**
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2. **放置處理後的 EEG pickle** 於 `Processed_data/` (如路徑不同，需在腳本中調整)。 資料集下載: [FACED EEG Repository](https://www.synapse.org/Synapse:syn50614194/files/)
3. **執行特徵擷取**：使用 `Features/` (傅立葉) 或 `Wavelet/` (小波) 內的腳本產生 `.mat` 特徵檔。
4. **跑分類流程**：例如 `Binary-Classification-Fourier/00_Cross_Subject-Linear_kernel/run_all.py`，可自動完成正規化、LDS 平滑、SVM 訓練/測試與繪圖。
5. **結果驗證**：若需比較多個設定，使用 `Statistics/` 進行 ANOVA、Friedman、Wilcoxon 等統計檢定。

### 主要資料夾

| 路徑 | 說明 |
| --- | --- |
| `Binary-Classification-Fourier/` | 二元情緒任務的傅立葉管線。子資料夾 `00`–`04` 代表不同版本 (基準、受試者內、RBF、頻帶分析、AsI 濾波)。各資料夾皆附 README。 |
| `Nine-Classification-Fourier/` | 九分類情緒流程，結構類似於二元版本但涵蓋 28 支影片。 |
| `Original-Fourier-LinearKernel/` | 最初的線性核實作，保留以利回溯與比較。 |
| `Wavelet/` | 小波方法：`wavelet_transform.py` → `save_*_wl.py` → `run_all.py`，支援 DE / PSD / Hjorth 組合。 |
| `Features/` | 通用特徵匯出腳本 (DE、PSD、Hjorth)。輸出 `.mat` 檔供各管線共用。 |
| `Statistics/` | 統計分析工具，包含合併結果、產生摘要、ANOVA / Friedman / Wilcoxon / t-test 等檢定。 |
| `Code_Rearchiture/` | 預留的重構空間，目前尚未放置程式。 |
| `requirements.txt` | 所需 Python 套件列表。 |

### 流程重點

- **特徵擷取**：可依需求選擇傅立葉或小波腳本。
- **正規化與平滑**：`running_norm*.py`、`smooth_lds.py` 在各資料夾中確保折數一致與 LDS 平滑。
- **分類訓練**：`main_de_svm.py`、`main_svm_asi_voting.py` 提供 C/gamma 搜尋、模型儲存與折別評估。
- **視覺化**：`plot_result.py` 產出每個實驗的準確率曲線與摘要。
- **統計檢定**：使用 `Statistics/` scripts 整合並檢驗多組實驗結果。
