# 01 · Intra-Subject Linear Kernel

This stage extends the baseline pipeline to intra-subject evaluation and variable-length sessions. Features are converted from pre-extracted pickle files (`Features/DE_features`) and trimmed to any `--sec` window before training per-subject SVMs.

---

### Classification Validation: SVM Analysis

1. **Intra-subject folds** split each 30 s video into 10 folds (default) per subject. `main_de_svm.py` supports both `--subjects-type intra` and `--subjects-type cross`, enabling consistency checks against folder `00`.
2. **Binary vs. nine-class** decisions still depend on `--n-vids` (`24` for positive/negative, `28` for nine emotions). Labels are re-used via `load_data.py` helpers.
3. **Window control** – `pkl_to_mat.py --sec {seconds}` truncates the per-video timeline before running normalization or SVMs, allowing comparisons such as 5 s vs. 15 s segments.

### Usage

1. **Convert pickle features to MAT**
   ```bash
   python pkl_to_mat.py --sec 10
   python filter_neutral_videos.py --sec 10   # only for 24-video runs
   ```

2. **Normalize, smooth, and train**
   ```bash
   python run_all.py --sec 10 --subject-type intra
   ```
   `run_all.py` orchestrates:
   * `running_norm.py --n-vids 24 --sec 10`
   * `smooth_lds.py --n-vids 24 --sec 10`
   * `main_de_svm.py --subjects-type intra --sec 10`
   * `plot_result.py --n-vids 24`

3. **Inspect / reuse model weights**
   * `svm_weights/subject_intra_vids_{n}_fold_{k}_valid_10-folds.joblib` – per-fold models.
   * `load_svm_weights.py` prints historical C (and gamma for RBF) to warm-start future tuning.
   * `run_intra_new.py` provides an alternative entry point for sweeping subjects or seconds in batch mode.

### Key Arguments

`run_all.py` forwards the most used toggles:

| Argument | Purpose |
| --- | --- |
| `--subject-type` | `cross`, `first_batch`, or `intra` mode. |
| `--sec` | Seconds kept per video (<=30). Propagates to all downstream scripts. |
| `--n-vids` | 24 (binary) / 28 (nine-class). |
| `--kernel` | `linear` (default) or `rbf` when calling `main_de_svm.py`. |
| `--valid-method` | `10-folds` (default) or `loo`. |

### Modifications

1. Added `pkl_to_mat.py --sec` to slice features at load time instead of materialising full 30 s sequences.
2. Updated `running_norm.py`/`smooth_lds.py`/`main_de_svm.py` to accept the `--sec` flag, ensuring folds align with truncated sessions.
3. Introduced `load_svm_weights.py` for analysing previously trained models (extracting best `C`/`gamma`).
4. Added `plot_result.py` parameters to tag different second-length experiments in the output figures.

---

### 分類驗證：SVM 分析

1. **受試者內驗證**：`main_de_svm.py --subjects-type intra` 會依影片長度 (`sec`) 將 30 秒切成等份折數，並針對每位受試者進行 10 折訓練/測試。
2. **二元 / 九分類**：透過 `--n-vids` 控制，仍沿用前一資料夾的標籤定義。
3. **可變視窗控制**：`pkl_to_mat.py --sec` 直接限制匯入的秒數，後續正規化、LDS 以及 SVM 皆自動同步。

### 使用方法

1. **轉換特徵檔**
   ```bash
   python pkl_to_mat.py --sec 15
   python filter_neutral_videos.py --sec 15   # 僅限 24 影片
   ```

2. **執行完整流程**
   ```bash
   python run_all.py --sec 15 --subject-type intra
   ```
   此指令會串接 `running_norm.py`、`smooth_lds.py`、`main_de_svm.py` 與 `plot_result.py` 並自動檢查輸出檔案是否存在。

3. **檢視模型與結果**
   * `svm_weights/`：儲存每折 Joblib 模型，可交由 `load_svm_weights.py` 查看最佳 C / gamma。
   * `plot_result.py`：繪製不同秒數設定的折線圖便於比較。

### 修改內容

1. 新增 `--sec` 參數以支援短視窗分析，所有腳本皆會依此截取資料。
2. `filter_neutral_videos.py`、`running_norm.py`、`smooth_lds.py` 皆加入秒數與影片數檢查，避免資料錯位。
3. 增設 `load_svm_weights.py`、`run_intra_new.py` 等工具，以便快速重複利用舊權重或批次實驗。
