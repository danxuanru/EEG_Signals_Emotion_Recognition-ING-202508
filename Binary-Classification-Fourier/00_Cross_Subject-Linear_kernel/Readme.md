# 00 · Cross-Subject Linear Kernel (Baseline)

This baseline reproduces the classical cross-subject EEG emotion recognition pipeline with Differential Entropy (DE) features and linear SVMs. It reads raw EEG recordings, extracts five-band DE features, applies temporal normalization (running norm + LDS), and trains/testing SVMs for binary (24 videos) or nine-class (28 videos) classification.

---

### Classification Validation: SVM Analysis

Classification follows the original paper settings:
1. **Binary positive/negative vs. nine-category** decisions at a 1 s time scale.
2. **Cross-subject 10-fold protocol** (9 folds × 12 subjects, final fold × 15 subjects) implemented in `main_de_svm.py` with `--subjects-type cross`.

`save_de.py`, `running_norm.py`, `smooth_lds.py`, `main_de_svm.py`, and the helper utilities (`io_utils.py`, `load_data.py`, `reorder_vids.py`) form the ordered steps. `main_de_svm.py` performs exhaustive C-search (`10**np.arange(-5,1,0.5)`) and stores the best model per fold into `svm_weights/` for later testing.

### Usage

1. **Feature calculation**
    ```bash
    python save_de.py                      # build de_features.mat from Processed_data
    python filter_neutral_videos.py        # only when --n-vids 24 (drop neutral clips)
    ```

2. **Run the full pipeline**
    ```bash
    python run_all.py --n-vids 24 --subjects-type cross \
      --kernel linear --valid-method 10-folds --search-method grid \
      --feature-selection none --n-features 50 --n-jobs 1 --early-stopping 3
    ```

    Key arguments exposed by `run_all.py`:
    * `--n-vids`: 24 (binary) or 28 (nine-class).
    * `--kernel`: any scikit-learn SVM kernel (`linear`, `rbf`, `poly`, `sigmoid`).
    * `--valid-method`: `10-folds`, `loo`, `time-based`, `stratified`.
    * `--search-method`: `grid`, `random`, `bayesian` search over C.
    * `--feature-selection`: `none`, `selectk`, `pca` (with `--n-features`).
    * `--skip-data-prep`: reuse existing `.mat`/`smooth_*` outputs and only retrain the classifier.

    Output artifacts:
    * `running_norm_{n_vids}/normTrain_rnPreWeighted0.990_*` – decay-based running norm splits.
    * `smooth_{n_vids}/de_lds_fold*.mat` – LDS-smoothed features consumed by the SVM.
    * `svm_weights/subject_cross_vids_{n_vids}_fold_{k}_valid_*.joblib` – trained fold models.

3. **Plot results**
    `plot_result.py` visualizes fold accuracies and writes CSV summaries for reporting.

### Modifications

1. Fully scripted pipeline (`run_all.py`) with argument hooks for kernels, validation, search method, and optional feature selection.
2. `save_de.py` extracts five canonical bands (delta–gamma) directly from `Processed_data/` with `mne` filtering to guarantee consistent preprocessing for later folders.
3. Running normalization (`running_norm.py`) supports 24-video filtering, random folds, and deterministic seeding, mirroring later experiments.
4. LDS smoothing (`smooth_lds.py`) produces per-fold `.mat` files that upstream SVM scripts can reuse, keeping the data/label ordering traceable via `reorder_vids.py`.
5. `plot_result.py` standardizes accuracy reporting for cross-folder comparison.

---

### 分類驗證：SVM 分析

此資料夾沿用經典設定：
1. **二元 (24 支影片) 與九分類 (28 支影片)**，以 1 秒視窗計算 DE 特徵。
2. **跨受試者 10 等分驗證**，同樣由 `main_de_svm.py` 依照受試者折數切分並搜尋最佳 C 值，模型存入 `svm_weights/` 供測試階段讀取。

整體流程由 `save_de.py → filter_neutral_videos.py → running_norm.py → smooth_lds.py → main_de_svm.py → plot_result.py` 串接，並依序使用 `io_utils.py`、`load_data.py`、`reorder_vids.py` 協助資料重排。

### 使用方法

1. **計算 DE 特徵**
    ```bash
    python save_de.py
    python filter_neutral_videos.py   # 僅限 24 影片 (移除中性片段)
    ```

2. **執行整體流程**
    ```bash
    python run_all.py --n-vids 28 --subjects-type cross \
      --kernel linear --valid-method 10-folds --search-method grid \
      --feature-selection none --n-features 50 --n-jobs 1 --early-stopping 3
    ```

    主要參數：
    * `--n-vids`：24 (二元) / 28 (九分類)。
    * `--subjects-type`：`cross`、`intra`。
    * `--kernel`：`linear`、`rbf`...。
    * `--search-method`：`grid`、`random`、`bayesian`；可透過 `--skip-data-prep` 直接重跑 SVM。
    * `--feature-selection`：`none`、`selectk`、`pca` + `--n-features`。

3. **結果檢視**
    `plot_result.py` 會繪製每折準確率與總結檔，可直接比對後續資料夾的演進差異。

### 修改內容

1. `save_de.py` 使用 `mne.filter.filter_data` 建立五頻帶 DE 特徵，作為所有後續資料夾的共同起點。
2. `filter_neutral_videos.py` 針對二元任務移除中性片段，以配合 `running_norm.py` 的影片重排流程。
3. `run_all.py` 集中管理參數搜尋、核心選擇、折數設定與資料備妥檢查，確保每個步驟都有檔案存在驗證。
4. `running_norm.py` 導入遞減因子 (0.990) 的 running statistics，在 10 折與隨機種子下可重複生成相同結果。
5. `smooth_lds.py` 與 `plot_result.py` 與後續資料夾共用，方便跨版本比較。