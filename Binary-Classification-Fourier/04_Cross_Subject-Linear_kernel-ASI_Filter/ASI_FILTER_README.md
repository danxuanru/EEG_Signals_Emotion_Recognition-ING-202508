# AsI (Asymmetry Index) 濾波器使用說明

## 📋 移除條件說明

### 窗口級別過濾
- **保留條件**: AsI[Fp1-Fp2] >= 0.5
- **移除條件**: AsI[Fp1-Fp2] < 0.5

AsI 值越高，表示情緒激發程度越強。當 AsI < 0.5 時，表示受試者在該窗口的情緒激發不足。

### Trial 級別過濾
- **移除條件**: 如果單個 trial（影片）中移除的窗口數量 **> 15 個**
- **理由**: 該受試者在觀看該影片時未能有效激發情緒

### 中性影片
- Video 13-16 被用作 **baseline（基線訊號）**
- 這些中性影片不會被過濾，用於計算情緒激發狀態的參考

---

## 🚀 使用方法

### 基本使用
```bash
cd 00_Code
python asi_filter.py
```

### 自訂參數
```bash
python asi_filter.py \
    --input_dir ../Processed_data \
    --output_dir ../Processed_data_after_AsI \
    --dataset both \
    --session_sec 30 \
    --window_sec 1 \
    --fs 250
```

### 參數說明
- `--input_dir`: 輸入資料夾路徑（預設: `../Processed_data`）
- `--output_dir`: 輸出資料夾路徑（預設: `../Processed_data_after_AsI`）
- `--dataset`: 資料集類型（預設: `both`，可選: `first_batch`, `second_batch`）
- `--session_sec`: 每個影片長度（秒，預設: 30）
- `--window_sec`: 窗口長度（秒，預設: 1）
- `--fs`: 採樣頻率（Hz，預設: 250）
- `--A`: MDI 歷史時間點數量（預設: 5）
- `--B`: MDI 未來時間點偏移（預設: 5）

---

## 📊 輸出報告

程式執行完成後會自動生成三個報告文件：

### 1. `AsI_Filter_Report.txt`
詳細的文字報告，包含：
- ✅ 配置參數和過濾規則
- ✅ 總體統計（保留/移除比例）
- ✅ 每個受試者的詳細資訊
- ✅ AsI 值統計（平均值、標準差、範圍等）
- ✅ 移除的 trials 和窗口清單

**適合**：人工閱讀和檢查

### 2. `AsI_Filter_Report.json`
結構化的 JSON 報告，包含所有統計資訊

**適合**：程式讀取和進一步分析

### 3. `AsI_Filter_Summary.csv`
CSV 格式的摘要表格，包含每個受試者的統計資訊

**適合**：Excel 打開、製作圖表、統計分析

| Subject | Total_Trials | Neutral_Trials | Emotion_Trials | Kept_Trials | Removed_Trials | ... |
|---------|--------------|----------------|----------------|-------------|----------------|-----|
| 0       | 28           | 4              | 24             | 20          | 4              | ... |
| 1       | 28           | 4              | 24             | 22          | 2              | ... |

---

## 💾 輸出資料格式

每個受試者的過濾後資料儲存為 `.pkl` 文件，包含：

```python
{
    'filtered_trials': [  # 列表，每個元素是一個字典
        {
            'trial_idx': int,        # trial 索引
            'video_num': int,        # 影片編號
            'data': ndarray,         # 原始資料 (n_channels, n_samples)
            'kept_windows': list,    # 保留的窗口索引列表
            'removed': bool          # 是否整個 trial 被移除
        },
        ...
    ],
    'filtered_info': {  # 過濾統計資訊
        'subject_idx': int,
        'total_trials': int,
        'neutral_trials': int,
        'emotion_trials': int,
        'removed_windows': list,
        'removed_trials': list,
        'asi_values': list
    },
    'metadata': {  # 元資料
        'n_trials': int,
        'n_channels': int,
        'n_samples': int,
        'window_samples': int,
        'n_windows_per_trial': int,
        'session_sec': int,
        'window_sec': int,
        'fs': int
    }
}
```

---

## 🔍 載入和使用過濾後的資料

### 方法 1: 使用輔助函數（推薦）

```python
from asi_filter import extract_filtered_data

# 拼接每個 trial 保留的窗口（推薦用於特徵提取）
trials_data = extract_filtered_data(
    '../Processed_data_after_AsI/sub000.pkl', 
    output_format='concatenate'
)

# 提取所有窗口作為獨立樣本（推薦用於窗口級別分類）
windows_data, labels = extract_filtered_data(
    '../Processed_data_after_AsI/sub000.pkl', 
    output_format='windows'
)

# 保持原始 trial 結構，移除的窗口用 NaN 填充
trials_data = extract_filtered_data(
    '../Processed_data_after_AsI/sub000.pkl', 
    output_format='trials'
)
```

### 方法 2: 直接載入完整資料結構

```python
import pickle

with open('../Processed_data_after_AsI/sub000.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# 獲取所有過濾後的 trials
filtered_trials = data_dict['filtered_trials']

# 處理每個 trial
for trial in filtered_trials:
    if not trial['removed']:  # 只處理保留的 trials
        trial_data = trial['data']
        kept_windows = trial['kept_windows']
        video_num = trial['video_num']
        
        # 提取保留的窗口
        for win_idx in kept_windows:
            start = win_idx * window_samples
            end = start + window_samples
            window_data = trial_data[:, start:end]
            # 進行特徵提取或其他處理...
```

---

## 📈 報告解讀

### 總體統計示例
```
總體統計:
  處理受試者數: 123
  總 trials 數: 3444
    - 中性 trials: 492 (用作 baseline)
    - 情緒 trials: 2952
  總窗口數: 103,320

  保留情況:
    - 保留 trials: 2650/2952 (89.77%)
    - 保留窗口: 78,450/103,320 (75.93%)

  移除情況:
    - 移除 trials: 302/2952 (10.23%)
    - 移除窗口: 24,870/103,320 (24.07%)
```

### 解讀說明
- **Trial 保留率 89.77%**: 大部分受試者能有效被情緒刺激激發
- **窗口保留率 75.93%**: 約 3/4 的時間窗口顯示出足夠的情緒激發
- **移除 trials 10.23%**: 這些影片對受試者的情緒刺激效果不佳

### AsI 值統計示例
```
AsI 值統計:
  平均值: 0.5823
  標準差: 0.2156
  最小值: 0.0234
  最大值: 0.9876
  中位數: 0.5912
  >= 0.5 的窗口比例: 76.34%
```

---

## ⚠️ 注意事項

1. **確保資料夾存在**:
   - `../Processed_data`: 輸入資料
   - `../After_remarks`: Video order 資訊

2. **電極索引**:
   - 目前設定: Fp1=0, Fp2=2, Fz=9
   - 如果您的資料集電極順序不同，請修改 `get_channel_indices()` 函數

3. **記憶體使用**:
   - 程式會保留完整的原始資料
   - 如果處理大量受試者，可能需要較大記憶體

4. **處理時間**:
   - MDI 計算較為耗時
   - 預計每個受試者需要 1-5 分鐘（取決於硬體）

---

## 🐛 常見問題

### Q: 報告顯示某些受試者移除率很高？
A: 這可能表示：
- 受試者對情緒刺激不敏感
- 該批次影片的情緒刺激效果較弱
- 可以考慮調整 AsI 閾值（修改 `apply_asi_filter_rule` 函數）

### Q: 如何調整過濾閾值？
A: 修改 `asi_filter.py` 中的兩個參數：
```python
# 窗口級別閾值（第 174 行）
if asi_fp >= 0.5:  # 可以改為 0.4 或 0.6

# Trial 級別閾值（第 320 行）
if removed_windows_count > 15:  # 可以改為 10 或 20
```

### Q: 如何只處理特定受試者？
A: 在 `process_all_subjects` 函數中添加過濾條件：
```python
for sub_idx, subject_file in enumerate(subject_files):
    if sub_idx not in [0, 1, 2]:  # 只處理 subject 0, 1, 2
        continue
    # ... 處理代碼
```

---

## 📝 引用

如果您在研究中使用此 AsI 濾波器，請引用原始 AsI 論文。

---

## 📧 聯絡

如有問題或建議，請聯繫開發者。
