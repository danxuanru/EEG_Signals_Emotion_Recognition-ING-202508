import os
import math
import pandas as pd
from scipy import stats

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'merged_by_time')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), 'wilcoxon_15s_results.csv')
ALPHA = 0.05


def find_value_columns(df):
    cols = [c for c in df.columns if c.lower() != 'subject']
    if len(cols) != 2:
        raise ValueError(f'Expected exactly 2 value columns besides Subject, got {len(cols)}: {cols}')
    return cols


def compute_effect_sizes(w_stat, n):
    # Total sum of ranks = n(n+1)/2
    total_ranks = n * (n + 1) / 2
    # Rank-biserial correlation (requires W+ = w_stat)
    r_rank_biserial = (2 * w_stat - total_ranks) / total_ranks if total_ranks > 0 else float('nan')
    # Approximate z for Wilcoxon (continuity correction not applied)
    mean_w = n * (n + 1) / 4
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sd_w == 0:
        z = float('nan')
        r = float('nan')
    else:
        z = (w_stat - mean_w) / sd_w
        r = abs(z) / math.sqrt(n)
    return r_rank_biserial, r, z


def analyze_file(path):
    df = pd.read_csv(path)
    value_cols = find_value_columns(df)
    # Drop rows with NA in either value col
    sub_df = df[['Subject'] + value_cols].dropna()
    # Align by Subject just in case
    sub_df = sub_df.sort_values('Subject')
    x = sub_df[value_cols[0]].values
    y = sub_df[value_cols[1]].values

    # Differences
    diff = x - y
    # Exclude zero differences for W computation per standard (scipy handles depending on zero_method)
    res = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided', mode='auto')
    w_stat = res.statistic
    p_value = res.pvalue

    # Number of non-zero diffs used
    n_nonzero = (diff != 0).sum()
    r_rb, r, z = compute_effect_sizes(w_stat, n_nonzero)

    median_x = float(pd.Series(x).median())
    median_y = float(pd.Series(y).median())
    median_diff = float(pd.Series(diff).median())
    mean_diff = float(pd.Series(diff).mean())

    significance = 'Yes' if p_value < ALPHA else 'No'

    return {
        'file': os.path.basename(path),
        'col_1': value_cols[0],
        'col_2': value_cols[1],
        'n_pairs': int(len(x)),
        'n_nonzero': int(n_nonzero),
        'wilcoxon_W': float(w_stat),
        'p_value': float(p_value),
        'significant_0.05': significance,
        'median_col_1': median_x,
        'median_col_2': median_y,
        'median_diff': median_diff,
        'mean_diff': mean_diff,
        'rank_biserial_r': r_rb,
        'effect_size_r': r,
        'z_approx': z,
    }


def main():
    print(f"Scanning directory: {DATA_DIR}", flush=True)
    if not os.path.isdir(DATA_DIR):
        print("Data directory not found.")
        return
    all_files = os.listdir(DATA_DIR)
    print(f"Found files: {all_files}", flush=True)
    results = []
    for fname in all_files:
        if not fname.endswith('.csv'):
            continue
        if '15s' not in fname:
            continue
        print(f"Processing {fname} ...", flush=True)
        fpath = os.path.join(DATA_DIR, fname)
        try:
            res = analyze_file(fpath)
            results.append(res)
        except Exception as e:
            print(f'Error processing {fname}: {e}')

    if not results:
        print('No 15s files processed. Check directory or filename patterns.')
        return

    df_out = pd.DataFrame(results)
    # Multiple comparison adjustment (Bonferroni)
    m = len(df_out)
    df_out['p_bonferroni'] = df_out['p_value'] * m
    df_out['p_bonferroni'] = df_out['p_bonferroni'].clip(upper=1.0)
    df_out['significant_bonferroni'] = df_out['p_bonferroni'] < ALPHA

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f'Saved Wilcoxon results to {OUTPUT_CSV}', flush=True)
    # Simple textual summary
    sig = df_out[df_out['significant_0.05'] == 'Yes']
    print('\nSummary:')
    print(df_out[['file', 'p_value', 'significant_0.05', 'p_bonferroni', 'significant_bonferroni']])
    if not sig.empty:
        print(f'Files with p < 0.05 (uncorrected): {sig.file.tolist()}')
    else:
        print('No uncorrected significant differences.')


if __name__ == '__main__':
    main()
