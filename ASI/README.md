# 基於 AsI 濾波 EEG 的情緒辨識（SVM Pipeline）

本專案實作一個端到端的 **EEG 情緒辨識流程**，使用 **AsI 濾波後的 EEG 資料** 搭配 **SVM 分類器**，主要包含：

- 從 AsI 濾波後的 EEG 抽取特徵（DE / PSD / Hjorth）
- 以受試者為單位的 10 折交叉驗證
- 標準化與 PCA 降維
- 線性 SVM 訓練與測試
- 視窗層級與影片層級（投票）的分類準確率

---

## 1. 專案架構與主要腳本

主要 Python 腳本：

- `save_feature_asi.py`  
  從 AsI 濾波後的 EEG 資料中抽取特徵，並建立 pooled dataset。
- `running_norm_asi.py`  
  執行 running normalization，並產生以受試者為單位的 10 折資料。
- `main_svm_asi_voting.py`  
  訓練與測試 Linear SVM，並進行影片層級投票。
- `run_all_asi.py`  
  串接上述所有步驟，從 AsI 濾波資料一路執行到最終評估。

一般情況下，只需要執行 `run_all_asi.py` 即可。

---

## 2. 整體流程概觀

完整實驗流程如下：

1. **特徵抽取與 pooled dataset 建立**  
   （`save_feature_asi.py`）
2. **Running normalization + 以受試者為單位的 10 折切分**  
   （`running_norm_asi.py`）
3. **SVM 訓練（逐折）**  
   （`main_svm_asi_voting.py --train-or-test train`）
4. **SVM 測試 + 影片層級投票**  
   （`main_svm_asi_voting.py --train-or-test test`）

可直接使用下列指令一鍵執行：

```bash
python run_all_asi.py --feature DE
# 或指定不同特徵組合：
python run_all_asi.py --feature DE_PSD
python run_all_asi.py --feature DE_PSD_H
```

---

## 3. 資料需求

### 3.1 AsI 濾波後 EEG 資料

`save_feature_asi.py` 預期輸入資料位於：

```text
../Processed_Data_after_AsI/
    sub000.pkl
    sub001.pkl
    ...
```

每個 `subXXX.pkl` 需包含：

- EEG 資料：已作 AsI 濾波並切成視窗  
  形狀約為 `(n_windows, n_channels, window_samples)`
- 對應標籤（影片編號或分數）  
  透過 `extract_filtered_data(..., output_format='windows')` 讀取。

### 3.2 特徵輸出資料夾

特徵會被儲存在：

```text
../Features/
    DE_Feature_AsI/
    PSD_Feature_AsI/
    Hjorth_Feature_AsI/
```

若路徑不存在，程式會自動建立。

---

## 4. 各步驟詳細說明

### Step 1：特徵抽取與 pooled dataset 建立  
**腳本：** `save_feature_asi.py`

一般由 `run_all_asi.py` 呼叫：

```bash
python save_feature_asi.py --feature DE
# 或 DE_PSD / DE_PSD_H
```

主要動作：

1. **對每位受試者 `subXXX.pkl`：**
   - 讀取 AsI 濾波後的 EEG 視窗與標籤：
     - `sub_data`: `(n_windows, n_channels, window_samples)`
     - `labels`: 與每個視窗對應之影片資訊
   - 計算特徵：
     - **DE**（Differential Entropy）頻帶特徵
     - **PSD**（Power Spectral Density，若選 DE_PSD 或 DE_PSD_H）
     - **Hjorth 參數**（mobility、complexity，若選 DE_PSD_H）
   - 將各受試者特徵儲存於：
     - `../Features/DE_Feature_AsI/subXXX.pkl`
     - `../Features/PSD_Feature_AsI/subXXX.pkl`
     - `../Features/Hjorth_Feature_AsI/subXXX.pkl`

2. **建立全體 pooled dataset：**
   - 讀取所有受試者的 DE（及 PSD/Hjorth，如有）
   - 僅使用前 30 個 channel（去除最後 2 個）
   - 在特徵維度上將 DE（+ PSD + Hjorth）串接
   - 將每一個視窗的特徵攤平成一維向量 `flattened_features`
   - 在前面加上 metadata 欄位：
     - `[subject_id, video_id, flattened_features...]`
   - 合併所有受試者資料：
     - `X`: 形狀 `(total_windows, 2 + feature_dim)`
     - `Y`: 標籤
   - 將標籤轉成**二元情緒**：
     - `Y > 16` → `1`（高情緒）
     - `Y <= 16` → `-1`（低情緒）
   - 存成 MAT 檔：
     ```text
     ./pooled_data_{feature}.mat
     # 內容包含：X, Y
     ```

**目的：**  
將 AsI 濾波後的 EEG 視窗轉換成頻帶相關的特徵向量，並整合所有受試者成一個含有 subject/video metadata 以及二元情緒標籤的 pooled dataset，方便後續交叉驗證與 SVM 訓練。

---

### Step 2：Running normalization 與 10 折（受試者切分）  
**腳本：** `running_norm_asi.py`

執行：

```bash
python running_norm_asi.py --feature DE
# 或 DE_PSD / DE_PSD_H
```

輸入：

- `./pooled_data_{feature}.mat`
  - `X`: `[subject_id, video_id, features...]`
  - `Y`: 二元標籤（`1` / `-1`）

主要動作：

1. 拆出 metadata 與特徵：
   - `subject_ids = X[:, 0]`
   - `video_ids   = X[:, 1]`
   - `features    = X[:, 2:]`
2. 取得所有不同 subject，並設定 **10 折 subject-wise cross-validation**：
   - 將 subject 隨機打亂
   - 每折使用一組不重複的 subject 當作測試集
3. 對每一折 fold：
   - 定義 `train_subjects` 與 `test_subjects`
   - 透過 `subject_ids` 產生 train / test mask
   - 切分：
     - `X_train`, `y_train`, `X_test`, `y_test`
   - 套用 `StandardScaler`：
     - 只在 `X_train` 上 `fit`
     - 用同一 scaler 去 `transform` `X_train` & `X_test`
   - 建立合併後的 `X_normalized`：
     - `[subject_id, video_id, normalized_features...]`
   - 存成：
     ```text
     ./running_norm_{feature}/
         de_fold0.mat       （DE）
         de_fold1.mat
         ...
         de_psd_fold0.mat   （DE_PSD）
         de_psd_h_fold0.mat （DE_PSD_H）
     ```

每個 fold 的 MAT 檔包含：

- `data`: `[subject_id, video_id, normalized_features...]`
- `labels`: 全部標籤 `Y`
- `train_subjects`, `test_subjects`
- `scaler_mean`, `scaler_scale`

**目的：**  
以 **受試者為單位** 做 10 折交叉驗證切分，並在每折中以訓練資料進行標準化（避免資料洩漏），準備好後續 SVM 所需的輸入。

---

### Step 3：SVM 訓練與測試（含投票）  
**腳本：** `main_svm_asi_voting.py`

#### 3.1 共用設定

- 讀取來源：`./running_norm_{feature}/...`
- 共有 10 折：`fold = 0..9`
- SVM 超參數：
  ```python
  C_cands = 10.**np.arange(-5, 1, 0.5)  # 1e-5 ~ 1e0
  max_iter = 10000
  dual = False
  tol  = 1e-4
  class_weight = 'balanced'
  ```

每折流程：

- 讀取：
  - `data` → 解析出 `subject_ids`, `video_ids`, `features`
  - `labels`, `train_subjects`, `test_subjects`
- 依 subject 切出 train / test
- 套用 **PCA**：
  - `PCA(n_components=0.95)`（保留 95% 變異）
  - 在 `X_train` 上 `fit`，並 `transform` `X_train`、`X_test`

---

#### 3.2 訓練模式

執行：

```bash
python main_svm_asi_voting.py --train-or-test train --feature DE
```

每一折：

1. 對每個 `C` in `C_cands`：
   - 使用 `X_train`, `y_train` 訓練 `LinearSVC`
   - 計算訓練準確率（作為簡化的 validation）
   - 若準確率較佳，更新 `best_C`、`best_val_acc`，並保存模型
2. 儲存最佳模型：
   ```text
   ./result_asi_{feature}/svm_weights/
       svm_asi_{feature}_fold0.joblib
       svm_asi_{feature}_fold1.joblib
       ...
   ```
3. 收集每折的 `best_C_folds`、`val_acc_folds`，最後輸出平均與標準差。

**目的：**  
在每折的訓練資料上，經 PCA 降維後訓練 Linear SVM，搜尋適當的正則化係數 `C`，並保存每折對應的最佳模型。

---

#### 3.3 測試模式 + 影片層級投票

執行：

```bash
python main_svm_asi_voting.py --train-or-test test --feature DE
```

每一折：

1. 讀取該折訓練好的 SVM 模型。
2. 在 `X_test` 上做視窗層級預測：
   - `window_predictions = clf.predict(X_test)`
   - `window_acc = mean(window_predictions == y_test)`
3. 對每一位測試 subject：
   - 取出該 subject 的所有視窗預測與對應 `video_ids`
   - 依 `video_id` 將視窗預測分組
   - 做 **影片層級投票**（預設 `majority`）：
     - 對每支影片，計數 `-1` 與 `1` 的票數，以多數決得到影片預測標籤
   - 與實際該影片的真實標籤比較，計算此 subject 在該折的 **影片層級準確率**
   - 將結果紀錄到 `video_results[subject].append(video_acc)`

所有折完成後：

- 針對每個 subject：
  - 計算在多個折中的平均影片準確率
- 計算所有 subject 的平均與標準差：
  - overall mean ± std
- 將結果存成 CSV：
  ```text
  ./result_asi_{feature}/video_results_{feature}_{voting_method}.csv
  ```

**目的：**  
評估訓練好的 SVM 在未出現過的受試者上的表現，並透過視窗→影片的投票機制，得到更符合實際應用的 **影片層級情緒辨識準確率**。

---

## 5. 一鍵執行 (`run_all_asi.py`)

`run_all_asi.py` 將上述流程串接起來：

```bash
python run_all_asi.py --feature DE
# 參數：
#   --feature {DE, DE_PSD, DE_PSD_H}
#   --remove-band （目前未實際使用）
#   --use-asi-filtered （預設 True）
```

依序執行：

1. **Step 1**：特徵抽取與 pooled dataset 建立  
   `python save_feature_asi.py --feature {feature}`  
   → `pooled_data_{feature}.mat`
2. **Step 2**：Running normalization + 10 折切分  
   `python running_norm_asi.py --feature {feature}`  
   → `running_norm_{feature}/..._fold{0..9}.mat`
3. **Step 4**：SVM 訓練  
   `python main_svm_asi_voting.py --train-or-test train --feature {feature}`  
   → 每折的 SVM 模型
4. **Step 5**：SVM 測試 + 影片層級評估  
   `python main_svm_asi_voting.py --train-or-test test --feature {feature}`  
   → 每位受試者的影片層級準確率與 CSV 結果

最後會在終端機顯示整體訓練與測試完成訊息。

---

## 6. 輸出結果與意義

主要輸出包含：

- **各受試者特徵檔案**
  - `../Features/DE_Feature_AsI/subXXX.pkl`
  - `../Features/PSD_Feature_AsI/subXXX.pkl`
  - `../Features/Hjorth_Feature_AsI/subXXX.pkl`

- **Pooled dataset**
  - `./pooled_data_{feature}.mat`
    - `X`: `[subject_id, video_id, flattened_features...]`
    - `Y`: 二元情緒標籤（1 / -1）

- **標準化後的各折資料**
  - `./running_norm_{feature}/de_fold{0..9}.mat`（或 `de_psd`、`de_psd_h`）

- **訓練好的 SVM 模型**
  - `./result_asi_{feature}/svm_weights/svm_asi_{feature}_fold{0..9}.joblib`

- **評估結果**
  - 終端機輸出：
    - 各折視窗層級 accuracy
    - 各受試者及整體的影片層級 accuracy
  - CSV：
    - `./result_asi_{feature}/video_results_{feature}_{voting_method}.csv`

可用來分析：

- 系統對高 / 低情緒的辨識能力
- 不同受試者 / 影片的表現差異
- 不同特徵組合（`DE`、`DE_PSD`、`DE_PSD_H`）對結果的影響

---

## 7. 注意事項與待優化項目

- `run_all_asi.py` 目前檢查檔名為 `pooled_data.mat`，  
  但 `save_feature_asi.py` 實際輸出為 `pooled_data_{feature}.mat`，建議統一檔名或修正檢查邏輯。
- 部分超參數（例如：
  - 受試者總數、
  - 頻帶設定、
  - SVM 搜尋的 C 範圍  
  ）可依實驗需求在程式中調整。

---

## 8. 快速開始（Quick Start）

1. 準備 AsI 濾波後的 EEG 資料，放在 `../Processed_Data_after_AsI/subXXX.pkl`。
2. 在本資料夾（`00_Code`）中執行：

   ```bash
   # 範例：僅使用 DE 特徵
   python run_all_asi.py --feature DE

   # 範例：使用 DE + PSD 特徵
   python run_all_asi.py --feature DE_PSD

   # 範例：使用 DE + PSD + Hjorth 特徵
   python run_all_asi.py --feature DE_PSD_H
   ```

3. 查看輸出結果：

   - `./result_asi_{feature}/`
   - `./result_asi_{feature}/video_results_{feature}_{voting_method}.csv`