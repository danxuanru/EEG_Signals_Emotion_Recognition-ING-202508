# 03 · Frequency Band Analysis (Cross-Subject Linear Kernel)

This stage investigates how individual frequency bands (delta–gamma) and feature bundles (DE / DE+PSD / DE+PSD+Hjorth) affect accuracy. It extends folder `02` by adding band-removal switches and automated experiment loops.

---

### Classification Validation: SVM Analysis

1. **Cross-subject setup** mirrors earlier folders. Linear SVMs are kept to isolate the impact of spectral content and feature unions.
2. **Band ablation** – `pkl_to_mat.py --remove-band {1-5}` excludes one of the five canonical bands before merging features. Band indices map to Delta, Theta, Alpha, Beta, Gamma respectively.
3. **Multi-feature experiments** – `--feature {DE,DE_PSD,DE_PSD_H}` concatenates PSD and Hjorth descriptors generated in `Features/` and feeds them through the same normalization + SVM pipeline.

### Usage

1. **Single experiment**
    ```bash
    python run_all.py --session-length 10 --feature DE_PSD_H --remove-band 0
    ```
    Parameters:
    * `--session-length`: seconds per video retained from the 30 s recordings.
    * `--feature`: choose the feature bundle (DE only, DE+PSD, or DE+PSD+Hjorth).
    * `--remove-band`: `0` keeps all bands; `1-5` removes Delta/Theta/Alpha/Beta/Gamma respectively.
    * `--n-vids`: default 24 (binary) but can be switched to 28.

2. **Batch all bands**
    ```bash
    python run_all_loop.py --session-length 10 --feature DE
    ```
    `run_all_loop.py` sequentially runs `run_all.py` for: all bands, -Delta, -Theta, -Alpha, -Beta, -Gamma, saving each result.

3. **Outputs**
    * `running_norm_{n_vids}/` and `smooth_{n_vids}/` keep per-feature subfolders.
    * `result_feature_{name}_remove_{band}.csv/png` (from `plot_result.py`) summarize comparisons.

### Modifications

1. `pkl_to_mat.py` now slices along time (`--session-length`) and can drop a selected frequency band before packing into `.mat`.
2. `run_all.py` and `running_norm.py` understand the `--feature` switch to merge DE/PSD/Hjorth combinations and normalise each block independently.
3. Added `run_all_loop.py` to automate six experiments (baseline + five removals) for fair comparisons.
4. `plot_result.py` annotates figures with the missing band name so they can be directly included in reports.

---

### 分類驗證：SVM 分析

1. **跨受試者設定**：維持線性 SVM 與 10 折，專注檢驗不同頻帶、特徵組合對準確度的影響。
2. **頻帶移除**：`--remove-band 1~5` 依序移除 Delta、Theta、Alpha、Beta、Gamma，`0` 代表保留全部頻帶。
3. **多特徵組合**：`--feature DE`、`DE_PSD`、`DE_PSD_H` 會自動載入對應的 PSD、Hjorth pickle 檔並串接於 DE 後。

### 使用方法

1. **單次實驗**
    ```bash
    python run_all.py --session-length 5 --feature DE_PSD --remove-band 3
    ```
    常用參數：`--session-length` (視窗秒數)、`--n-vids` (24/28)、`--subject-type` (預設 `cross`)。

2. **六種頻帶一次跑完**
    ```bash
    python run_all_loop.py --session-length 5 --feature DE_PSD_H
    ```
    程式會依照「全部→刪除 Delta→…→刪除 Gamma」循序呼叫 `run_all.py`，結果皆保存在各自的輸出資料夾。

3. **結果檢視**
    * `plot_result.py` 會在標題中標示 `remove-band` 參數，方便整理報表。
    * `.csv` 檔紀錄每個頻帶配置的折線平均與標準差，可供統計模組 (Statistics/) 後續分析。

### 修改內容

1. `pkl_to_mat.py` 增加 `--session-length` 與 `--remove-band`，可對時間與頻帶同時裁切。
2. `run_all.py` / `running_norm.py` 支援多特徵串接並確保三種特徵各自正規化，再合併餵入 SVM。
3. `run_all_loop.py` 自動化六種頻帶實驗流程，減少手動重複操作。
4. `plot_result.py` 與結果檔名皆加註頻帶資訊，以符合報告需求。