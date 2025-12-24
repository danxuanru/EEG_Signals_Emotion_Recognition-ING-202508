# 02 · Cross-Subject RBF Kernel

This folder upgrades the baseline to non-linear SVMs (default `rbf`) and adds automated hyper-parameter search, feature selection, and parallel execution. It reuses the DE features exported from `Features/DE_features` and keeps the same 10-fold cross-subject protocol.

---

### Classification Validation: SVM Analysis

1. **Cross-subject folds** identical to folder `00`, but evaluated with configurable kernels (`--kernel linear|rbf|poly|sigmoid`).
2. **Flexible search** over `C` and `gamma` via `--search-method {grid,random,bayesian}` or fixed values with `--fixed-c` / `--fixed-gamma` to replicate reported settings quickly.
3. **Optional feature selection** – before training, `main_de_svm.py` can run `SelectKBest` or PCA (`--feature-selection selectk|pca`, `--n-features`), helping high-dimensional variants (e.g., PSD/Hjorth) when they are plugged in later.

### Usage

1. **Prepare DE features**
   ```bash
   python pkl_to_mat.py
   python filter_neutral_videos.py       # only for 24-video binary runs
   ```

2. **Execute the pipeline**
   ```bash
   python run_all.py --n-vids 24 --kernel rbf --subjects-type cross \
     --search-method grid --feature-selection none --n-jobs 4 --early-stopping 5
   ```

   Parameters forwarded by `run_all.py`:
   * `--n-vids`: 24 / 28.
   * `--kernel`: passed to both training and testing commands.
   * `--valid-method`: `10-folds` (default) or `loo`.
   * `--search-method`: `grid`, `random`, `bayesian`.
   * `--n-jobs`: `-1` to use every CPU thread during hyper-parameter sweeps.
   * `--feature-selection`: `none`, `selectk`, `pca`.
   * `--early-stopping`: stop nested loops after N non-improving trials.
   * `--skip-data-prep`: reuse cached `smooth_{n_vids}` outputs when only hyper-parameters change.

3. **Inspect outputs**
   * `running_norm_{n_vids}/` and `smooth_{n_vids}/` – identical semantics to previous folders.
   * `svm_weights/subject_cross_vids_{n}_fold_{k}_valid_{method}_kernel_{kernel}.joblib` – best model per fold (contains the tuned kernel, `C`, and `gamma`).
   * `subject_cross_vids_{n}_valid_{method}_kernel_{kernel}.csv` – fold-wise test accuracy for later statistical analysis.

### Modifications

1. Switched `main_de_svm.py` to `sklearn.svm.SVC`, enabling kernel choices, gamma sweeps, and early stopping.
2. Added CLI hooks for parameter search strategy, number of parallel jobs, and optional fixed hyper-parameters.
3. Provided feature-selection scaffolding (SelectKBest/PCA) to stabilise high-dimensional settings.
4. Created `get_model_path.py` (utility) to resolve best-model locations for evaluation scripts.
5. Extended `plot_result.py` to annotate kernel names and search strategies on accuracy plots.

---

### 分類驗證：SVM 分析

1. **跨受試者 10 折**：與資料夾 00 相同，但可透過 `--kernel` 迅速切換線性、RBF、Polynomial 或 Sigmoid。
2. **彈性搜尋策略**：`main_de_svm.py` 會依 `--search-method` 掃描 `C` / `gamma`，也可使用 `--fixed-c`、`--fixed-gamma` 直接鎖定參數。
3. **特徵選擇**：`--feature-selection selectk|pca` 可在訓練前降低維度，`--n-features` 控制輸出維度。

### 使用方法

1. **匯入特徵**
   ```bash
   python pkl_to_mat.py
   python filter_neutral_videos.py   # 僅限 24 影片
   ```

2. **執行整體流程**
   ```bash
   python run_all.py --n-vids 28 --kernel rbf --subjects-type cross \
     --search-method random --n-jobs -1 --feature-selection pca --n-features 128
   ```
   常用參數：
   * `--n-vids`：24 (binary) / 28 (nine-class)。
   * `--valid-method`：`10-folds`、`loo`。
   * `--feature-selection`：`none`、`selectk`、`pca`。
   * `--early-stopping`：連續 N 次未改善即跳出迴圈。

3. **檢視結果**
   * `svm_weights/`：儲存每折模型 (含 kernel、C、gamma)。
   * `subject_cross_vids_*_kernel_*.csv`：輸出各折測試準確率。
   * `plot_result.py`：繪製不同 kernel / 搜尋策略的比較圖。

### 修改內容

1. `main_de_svm.py` 改用 `SVC`，支援 kernel、gamma、early-stopping 及多核心搜尋。
2. `run_all.py` 新增 `--n-jobs`、`--search-method`、`--feature-selection` 等參數，並加上 `--skip-data-prep` 方便重複實驗。
3. 加入 `load_svm_weights.py` / `get_model_path.py`，方便讀取既有模型並轉換為 `.joblib`、`.csv` 紀錄。
4. `plot_result.py` 可顯示 kernel 與搜尋策略資訊，以利和其他資料夾比對。
