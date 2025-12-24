# 04 · Cross-Subject Linear Kernel + AsI Filter

This folder adds an Adaptive Sliding Interval (AsI) preprocessing stage and ensemble voting on overlapping windows. It builds on folder `03` by letting you re-segment EEG signals, apply AsI filtering, and train SVMs on DE / DE+PSD / DE+PSD+Hjorth features extracted from the new windows.

---

### Classification Validation: SVM Analysis

1. **AsI filtering** (`asi_filter.py`) re-aligns EEG segments by adaptive intervals before any feature extraction. You can control window length (`--window-sec`) and overlap ratio (`--overlap-ratio`).
2. **Window-level classifiers** – `main_svm_asi_voting.py` trains per-fold SVMs (linear or RBF) on every AsI window. During testing the script aggregates predictions to video level using weighted, majority, or average-probability voting.
3. **Cross-subject folds** remain identical (10 folds) so results stay comparable to folders `00–03`.

### Usage

1. **Run the complete pipeline**
   ```bash
   python run_all_asi_windows.py --feature DE_PSD --window-sec 2 --overlap-ratio 0.5 \
     --voting-method weighted --kernel linear
   ```
   Steps performed:
   * (Optional) `asi_filter.py` – uncomment inside the script if you need to regenerate filtered signals.
   * `save_feature_asi.py` – exports pooled `pooled_data_{feature}.mat` with subject/video metadata.
   * `running_norm_asi.py --feature {feature}` – standardises irregular-length windows and creates `running_norm_{feature}/{feature}_fold*.mat`.
   * `main_svm_asi_voting.py --train-or-test train|test` – trains fold models then evaluates them with the chosen voting scheme.

2. **Key arguments**
   * `--feature`: `DE`, `DE_PSD`, `DE_PSD_H`.
   * `--window-sec`: length per AsI window.
   * `--overlap-ratio`: 0 (no overlap) to 0.9. Affects both data directory name and voting weights.
   * `--voting-method`: `weighted`, `majority`, `average_prob`.
   * `--kernel`: `linear` or `rbf` for both training and testing.

3. **Outputs**
   * `pooled_data_{feature}.mat` – feature pool with subject/video metadata.
   * `running_norm_{feature}/{feature}_fold{fold}.mat` – normalised folds ready for SVMs.
   * `result_asi_{feature}/svm_weights/svm_asi_{feature}_fold{fold}.joblib` – trained models.
   * `result_asi_{feature}/video_results_{feature}_{voting}.csv` – video-level accuracies per subject/fold.
   * `result_asi_{feature}/voting_stats_fold{fold}_{feature}_{voting}.csv` – detailed voting diagnostics.

### Modifications

1. Added AsI-specific preprocessing (`asi_filter.py`, `save_feature_asi.py`) supporting arbitrary window/overlap settings and feature bundles.
2. Implemented `running_norm_asi.py` to handle irregular numbers of windows per subject and store scaler statistics for reproducibility.
3. `main_svm_asi_voting.py` provides multiple voting schemes, PCA + StandardScaler, and fold-wise statistics/plots.
4. `run_all_asi_windows.py` ties everything together with file-existence checks so downstream steps do not run if inputs are missing.

---

### 分類驗證：SVM 分析

1. **AsI 濾波**：`asi_filter.py` 會依 `--window-sec`、`--overlap-ratio` 重新切割訊號，再輸出至 `Processed_Data_after_AsI_*` 供後續使用。
2. **視窗級分類 + 投票**：`main_svm_asi_voting.py` 先對每個 AsI 視窗訓練 SVM，測試時依 `--voting-method` (加權 / 多數決 / 平均) 匯總成影片層級的結果。
3. **跨受試者 10 折**：仍沿用既有折數切法，方便與前面資料夾直接比較。

### 使用方法

1. **一鍵執行**
   ```bash
   python run_all_asi_windows.py --feature DE_PSD_H --window-sec 1 \
     --overlap-ratio 0.25 --voting-method majority --kernel rbf
   ```
   (如需重新產生 AsI 過濾後之原始資料，可解除腳本內 `asi_filter.py` 的註解。)

2. **主要參數**
   * `--feature`：`DE` / `DE_PSD` / `DE_PSD_H`。
   * `--window-sec`：AsI 視窗長度 (秒)。
   * `--overlap-ratio`：視窗重疊比例 (0~0.9)。
   * `--voting-method`：`weighted`、`majority`、`average_prob`。
   * `--kernel`：`linear` 或 `rbf`。

3. **輸出檔案**
   * `pooled_data_{feature}.mat`：包含主體 / 影片索引與特徵矩陣。
   * `running_norm_{feature}/`：儲存每折標準化後的資料。
   * `result_asi_{feature}/svm_weights/`：保存 Joblib 模型。
   * `result_asi_{feature}/video_results_*.csv`、`voting_stats_*.csv`：影片層級準確率與投票統計。

### 修改內容

1. 導入 `asi_filter.py`、`save_feature_asi.py` 以支援 AsI 視窗的資料重抽樣與多特徵輸出。
2. `running_norm_asi.py` 能處理各受試者不同的視窗數量，並保留 `scaler_mean`、`scaler_scale` 以利重現。
3. `main_svm_asi_voting.py` 新增重疊視窗加權、多種投票策略、PCA + StandardScaler pipeline 及詳細統計輸出。
4. `run_all_asi_windows.py` 提供單一指令即可完成濾波→特徵→正規化→訓練→測試全流程，且含必要檔案檢查。
