# EEG Emotion Recognition using SVM

This project performs emotion recognition from EEG signals using Differential Entropy (DE) features and Support Vector Machine (SVM) classifiers. It includes analyses for binary (positive/negative) and nine-category emotion classification.

---

### Classification Validation: SVM Analysis

Classification analysis was conducted to validate the utility of the data records in two parts:
1.  A binary classification of positive and negative emotional states to make a direct comparison with previous studies.
2.  A classification of the nine-category emotional states ('Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness') to test whether the present dataset supports a more fine-grained emotion recognition.

The classification of emotional states was conducted on a 1-second time scale.

The classical method combining differential entropy (DE) features with the support vector machine (SVM) was used for both intra-subject and cross-subject emotion recognition.

*   **Intra-subject emotion recognition:** For all positive/negative video clips, 90% of EEG data in each video clip was used as the training set, and the remaining 10% was used as the testing set for each subject.
*   **Cross-subject emotion recognition:** The subjects were divided into 10 folds (12 subjects for the first nine folds, and 15 subjects for the 10th fold). Nine folds of subjects were used as the training set, and the remaining subjects were used as the testing set. The procedure was repeated 10 times, and the classification performances were obtained by averaging accuracies across the 10 folds.

The `io_utils.py`, `reorder_vids.py`, and `load_data.py` include relevant classes and functions needed for the program.

### Usage

1.  **Feature Calculation**
    First, run the following three scripts to extract all DE, PSD, and Hjorth features from the raw data.
    ```bash
    python save_de.py
    python save_psd_sum.py
    python save_hjorth_re.py
    ```

2.  **Run the full pipeline**
    Next, use `run_all.py` to execute the entire pipeline. This script merges the selected features, performs normalization, and handles classification and result plotting.

    **Parameters:**
    *   `--n-vids`: Number of videos. Use `24` for binary classification (positive/negative) or `28` for nine-category classification.
    *   `--subject-type`: Type of analysis. Use `cross` for cross-subject or `intra` for intra-subject analysis.
    *   `--session-length`: Specifies the number of seconds to use from the beginning of the 30-second feature data provided for each video.
    *   `--feature`: The combination of features to use. Options are `DE` (default), `DE_PSD` (DE + PSD), or `DE_PSD_H` (DE + PSD + Hjorth).

    **Example (Nine-category, cross-subject, using all features for the first second):**
    ```bash
    python run_all.py --n-vids 28 --subject-type cross --session-length 1 --feature DE_PSD_H
    ```

### Modifications

1.  Added `filter_neutral_videos.py` to support binary classification by removing neutral videos, which is required for the reordering step in `running_norm.py`.
2.  Rewrote `save_de.py` to save feature results into separate directories. Features are now merged using `pkl_to_mat.py`, allowing for adjustable time window analysis.
3.  Parameterized `run_all.py` by adding a `session-length` parameter to specify the duration of features to use.
4.  Added `plot_result.py` for simple result visualization.
5.  Introduced `PSD` and `Hjorth` as new features, selectable via the `--feature` argument, with dedicated extraction scripts (`save_psd_sum.py`, `save_hjorth_re.py`).
6.  Implemented multi-scale normalization in `running_norm.py` to process each feature type (DE, PSD, Hjorth) independently.

---

### 分類驗證：SVM 分析

本研究進行了分類分析，以驗證數據記錄的有效性，分為兩部分：
1.  **二元情緒分類**：對正面和負面情緒狀態進行分類，以便與先前的研究進行直接比較。
2.  **九類情緒分類**：對九種情緒狀態（'憤怒', '厭惡', '恐懼', '悲傷', '中性', '愉悅', '啟發', '喜悅', '溫柔'）進行分類，以測試當前數據集是否支持更細緻的情緒識別。

情緒狀態的分類以1秒的時間尺度進行。

研究採用結合微分熵（DE）特徵與支持向量機（SVM）的經典方法，進行了受試者內（intra-subject）和跨受試者（cross-subject）的情緒識別。

*   **受試者內情緒識別**：對於所有正面/負面影片片段，每個受試者在每個影片片段中的90%的EEG數據被用作訓練集，其餘10%用作測試集。
*   **跨受試者情緒識別**：將受試者分為10組（前九組各12名受試者，第十組15名受試者）。使用九組受試者作為訓練集，其餘受試者作為測試集。此過程重複10次，最終的分類性能是10次交叉驗證準確率的平均值。

`io_utils.py`、`reorder_vids.py` 和 `load_data.py` 包含了程式所需的相關類別和函式。

### 使用方法

1.  **特徵計算**
    首先，執行以下三個腳本，從原始數據中提取所有的 DE、PSD 和 Hjorth 特徵。
    ```bash
    python save_de.py
    python save_psd_sum.py
    python save_hjorth_re.py
    ```

2.  **執行完整流程**
    接著，使用 `run_all.py` 執行整個流程。此腳本會整合選擇的特徵、執行正規化、處理分類及繪製結果。

    **參數說明：**
    *   `--n-vids`：影片數量。二元分類（正面／負面）使用 `24`，九分類使用 `28`。
    *   `--subject-type`：分析類型。`cross` 為跨受試者分析，`intra` 為受試者內分析。
    *   `--session-length`：指定使用每個影片所提供之 30 秒特徵數據的前幾秒。
    *   `--feature`：要使用的特徵組合。可選 `DE`（預設）、`DE_PSD`（DE + PSD）或 `DE_PSD_H`（DE + PSD + Hjorth）。

    **範例（九分類、跨受試者、使用所有特徵的第一秒數據）：**
    ```bash
    python run_all.py --n-vids 28 --subject-type cross --session-length 1 --feature DE_PSD_H
    ```

### 修改內容

1.  新增 `filter_neutral_videos.py` 以支援二元分類，此步驟會移除中性影片，以適應 `running_norm.py` 中的數據重新排序（reorder）需求。
2.  改寫 `save_de.py`，將特徵提取結果以資料夾形式儲存，並使用 `pkl_to_mat.py` 合併特徵，方便依不同時間窗口進行分析。
3.  參數化 `run_all.py`，加入 `session-length` 參數，可用於設定數據切割的時間窗口長度。
4.  加入 `plot_result.py` 以實現簡易的結果可視化。
5.  引入 `PSD` 和 `Hjorth` 兩種新特徵，可透過 `--feature` 參數進行選擇，並提供專門的提取腳本（`save_psd_sum.py`, `save_hjorth_re.py`）。
6.  在 `running_norm.py` 中實現多尺度正規化，對每種特徵類型（DE, PSD, Hjorth）進行獨立處理。