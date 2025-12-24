# Statistics Toolbox

This directory collects standalone scripts for validating classifier outputs with classical statistical tests and for merging multiple folds’ results into a single summary. They assume that each experiment leaves CSV files (accuracy per subject, per second, per band, etc.) inside the experiment folder.

## Key scripts

- `merge_result.py` / `merge_result_time.py`: Concatenate accuracy CSV files across folds, kernels, or time windows before running any tests.
- `generate_summary.py`: Builds a report (mean ± std, max, min) from the merged CSVs.
- `anova_analysis_10s_15s.py`: Example of repeated-measure ANOVA comparing 10 s vs 15 s session lengths.
- `friedman_analysis.py`: Non-parametric Friedman test for related samples (e.g., comparing several feature combinations on the same subjects).
- `post_hoc_pairwise_test.py`: Performs pairwise comparisons with correction after ANOVA/Friedman.
- `run_paired_t_test.py`: Quick paired t-test between two experiment settings.
- `shapiro_wilk_test.py`: Normality check helper before deciding whether to use parametric or non-parametric tests.
- `wilcoxon_15s.py`, `wilcoxon_analysis.py`, `wilcoxon_comprehensive_analysis.py`: Variants of Wilcoxon signed-rank workflows for different subsets of the data.

## Recommended workflow

1. **Prepare data**: Ensure each experiment writes a CSV where rows represent subjects (or videos) and columns represent metrics.
2. **Merge**: Use the merge scripts to align column names and stack multiple CSVs if necessary.
3. **Check assumptions**: Run `shapiro_wilk_test.py` to gauge normality, then choose ANOVA/t-test (parametric) or Wilcoxon/Friedman (non-parametric).
4. **Run the main test**: Execute the corresponding script with the merged CSV path (most scripts accept arguments via `argparse`).
5. **Post-hoc**: If the omnibus test is significant, rely on `post_hoc_pairwise_test.py` or the Wilcoxon comprehensive script for detailed comparisons.
6. **Summarize**: Optionally regenerate a human-readable summary with `generate_summary.py` for inclusion in reports.

Each script prints the test statistics and p-values to stdout, making it easy to copy results into tables or notebooks.

## 中文說明

`Statistics/` 匯集各種統計檢定與結果整併工具，適合在完成分類實驗後驗證不同設定之間的差異是否顯著。

### 主要腳本

- `merge_result.py` / `merge_result_time.py`：先把多個折數或不同視窗長度的 CSV 檔整理合併。
- `generate_summary.py`：輸出平均值、標準差、最大/最小等摘要資訊。
- `anova_analysis_10s_15s.py`：對 10 秒 vs 15 秒條件做重複量數 ANOVA。
- `friedman_analysis.py`：多條件的非母數 Friedman 檢定。
- `post_hoc_pairwise_test.py`：在 ANOVA / Friedman 顯著後進行成對比較並做校正。
- `run_paired_t_test.py`：兩個條件的配對 t 檢定。
- `shapiro_wilk_test.py`：正態性檢定，協助決定使用何種統計方法。
- `wilcoxon_15s.py`、`wilcoxon_analysis.py`、`wilcoxon_comprehensive_analysis.py`：Wilcoxon signed-rank 系列腳本。

### 建議流程

1. **資料準備**：確保每個實驗輸出的 CSV 以受試者或影片為列，欄位為指標。
2. **先合併**：視需求跑 `merge_result*.py` 整合不同來源資料。
3. **檢查假設**：使用 `shapiro_wilk_test.py` 檢查正態性，再決定 ANOVA/t-test 或 Wilcoxon/Friedman。
4. **執行主要檢定**：透過對應腳本並輸入合併後的 CSV 路徑。
5. **事後檢定**：若整體檢定顯著，再用 `post_hoc_pairwise_test.py` 或 Wilcoxon 綜合腳本分析配對差異。
6. **摘要整理**：可再次執行 `generate_summary.py` 產出報告用統計表。

所有腳本會將統計量與 p 值顯示在終端機，方便直接貼到報告或筆記中。
