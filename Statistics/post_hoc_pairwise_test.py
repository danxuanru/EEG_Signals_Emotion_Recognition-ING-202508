import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests
import os
import glob
from pathlib import Path
import itertools
from collections import defaultdict

def load_merged_data(data_folder):
    """
    Load all CSV files from merged_by_time folder
    """
    csv_files = glob.glob(os.path.join(data_folder, "*_10s.csv"))
    
    data_dict = {}
    for file_path in csv_files:
        filename = Path(file_path).stem
        df = pd.read_csv(file_path)
        # Extract all columns except the first one
        data_dict[filename] = df.iloc[:, 1:]
        print(f"Loaded {filename}: {df.shape}")
    
    return data_dict

def check_normality_of_differences(data1, data2, alpha=0.05):
    """
    Check normality of paired differences using Shapiro-Wilk test
    """
    differences = data1 - data2
    # Remove NaN values
    differences = differences.dropna()
    
    if len(differences) < 3:
        return False, np.nan
    
    stat, p_value = shapiro(differences)
    is_normal = p_value > alpha
    
    return is_normal, p_value

def perform_paired_ttest(data1, data2):
    """
    Perform paired t-test and calculate Cohen's d_z
    """
    differences = data1 - data2
    differences = differences.dropna()
    
    if len(differences) < 2:
        return np.nan, np.nan, np.nan, np.nan
    
    # Paired t-test
    t_stat, p_value = ttest_rel(data1, data2, nan_policy='omit')
    
    # Cohen's d_z = mean difference / std of differences
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    if std_diff == 0:
        cohens_dz = 0
    else:
        cohens_dz = mean_diff / std_diff
    
    # Direction
    direction = "positive" if mean_diff > 0 else "negative" if mean_diff < 0 else "no difference"
    
    return t_stat, p_value, cohens_dz, direction

def perform_wilcoxon_test(data1, data2):
    """
    Perform Wilcoxon signed-rank test and calculate effect size r
    """
    # Remove NaN values
    valid_mask = ~(pd.isna(data1) | pd.isna(data2))
    data1_clean = data1[valid_mask]
    data2_clean = data2[valid_mask]
    
    if len(data1_clean) < 2:
        return np.nan, np.nan, np.nan, np.nan
    
    # Wilcoxon signed-rank test
    try:
        stat, p_value = wilcoxon(data1_clean, data2_clean, alternative='two-sided')
        
        # Effect size r = Z / sqrt(N)
        n = len(data1_clean)
        z_score = stats.norm.ppf(1 - p_value/2)  # Convert p-value to z-score
        if p_value == 0:
            z_score = stats.norm.ppf(0.9999)  # Handle p=0 case
        
        effect_size_r = z_score / np.sqrt(n)
        
        # Direction
        median_diff = np.median(data1_clean - data2_clean)
        direction = "positive" if median_diff > 0 else "negative" if median_diff < 0 else "no difference"
        
        return stat, p_value, effect_size_r, direction
        
    except Exception as e:
        print(f"Error in Wilcoxon test: {e}")
        return np.nan, np.nan, np.nan, np.nan

def calculate_kendalls_w(data_matrix):
    """
    Calculate Kendall's W (effect size for Friedman test)
    """
    n, k = data_matrix.shape  # n = subjects, k = conditions
    
    # Rank each row
    ranked_data = np.apply_along_axis(stats.rankdata, 1, data_matrix)
    
    # Calculate rank sums for each condition
    rank_sums = np.sum(ranked_data, axis=0)
    
    # Calculate Kendall's W
    mean_rank_sum = np.mean(rank_sums)
    ss_rank_sums = np.sum((rank_sums - mean_rank_sum) ** 2)
    
    max_possible_ss = (n**2 * (k**3 - k)) / 12
    
    if max_possible_ss == 0:
        return 0
    
    kendalls_w = ss_rank_sums / max_possible_ss
    
    return kendalls_w

def perform_pairwise_analysis(data_dict):
    """
    Perform pairwise comparisons for all feature combinations
    """
    all_results = []
    
    for feature_name, df in data_dict.items():
        print(f"\nAnalyzing {feature_name}...")
        
        # Get numeric columns (time segments)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            print(f"Skipping {feature_name}: insufficient time segments")
            continue
        
        # Calculate Kendall's W for overall effect size
        data_matrix = df[numeric_columns].dropna().values
        kendalls_w = calculate_kendalls_w(data_matrix)
        
        # Perform pairwise comparisons
        pairs = list(itertools.combinations(numeric_columns, 2))
        
        for col1, col2 in pairs:
            data1 = df[col1]
            data2 = df[col2]
            
            # Check normality of differences
            is_normal, normality_p = check_normality_of_differences(data1, data2)
            
            if is_normal:
                # Use paired t-test
                test_stat, p_value, effect_size, direction = perform_paired_ttest(data1, data2)
                test_type = "Paired t-test"
                effect_type = "Cohen's d_z"
            else:
                # Use Wilcoxon signed-rank test
                test_stat, p_value, effect_size, direction = perform_wilcoxon_test(data1, data2)
                test_type = "Wilcoxon signed-rank"
                effect_type = "Effect size r"
            
            result = {
                'Feature_Combination': feature_name,
                'Comparison': f"{col1} vs {col2}",
                'Test_Type': test_type,
                'Normality_Check': 'Normal' if is_normal else 'Non-normal',
                'Normality_P_Value': normality_p,
                'Test_Statistic': test_stat,
                'P_Value': p_value,
                'Effect_Size': effect_size,
                'Effect_Type': effect_type,
                'Direction': direction,
                'Kendalls_W': kendalls_w
            }
            
            all_results.append(result)
    
    return all_results

def apply_fdr_correction(results):
    """
    Apply FDR correction to p-values
    """
    p_values = [r['P_Value'] for r in results if not pd.isna(r['P_Value'])]
    
    if not p_values:
        return results
    
    # Apply FDR correction
    rejected, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Update results with corrected p-values
    p_idx = 0
    for i, result in enumerate(results):
        if not pd.isna(result['P_Value']):
            results[i]['P_Value_FDR_Corrected'] = p_adjusted[p_idx]
            results[i]['Significant_After_FDR'] = 'Yes' if rejected[p_idx] else 'No'
            results[i]['Significant_Before_FDR'] = 'Yes' if p_values[p_idx] < 0.05 else 'No'
            p_idx += 1
        else:
            results[i]['P_Value_FDR_Corrected'] = np.nan
            results[i]['Significant_After_FDR'] = 'N/A'
            results[i]['Significant_Before_FDR'] = 'N/A'
    
    return results

def save_results(results, output_path):
    """
    Save post-hoc analysis results to CSV
    """
    results_df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = [
        'Feature_Combination', 'Comparison', 'Test_Type', 'Normality_Check', 
        'Normality_P_Value', 'Test_Statistic', 'P_Value', 'P_Value_FDR_Corrected',
        'Significant_Before_FDR', 'Significant_After_FDR', 'Effect_Size', 
        'Effect_Type', 'Direction', 'Kendalls_W'
    ]
    
    results_df = results_df[column_order]
    results_df.to_csv(output_path, index=False)
    print(f"\nPost-hoc results saved to: {output_path}")
    return results_df

def print_summary(results_df):
    """
    Print summary of post-hoc analysis
    """
    print("\n" + "="*80)
    print("POST-HOC PAIRWISE ANALYSIS SUMMARY")
    print("="*80)
    
    # Test type distribution
    test_counts = results_df['Test_Type'].value_counts()
    print(f"\nTest types used:")
    for test, count in test_counts.items():
        print(f"  {test}: {count}")
    
    # Significance before and after FDR
    sig_before = results_df['Significant_Before_FDR'].value_counts().get('Yes', 0)
    sig_after = results_df['Significant_After_FDR'].value_counts().get('Yes', 0)
    total_tests = len(results_df)
    
    print(f"\nSignificance results:")
    print(f"  Before FDR correction: {sig_before}/{total_tests} significant")
    print(f"  After FDR correction: {sig_after}/{total_tests} significant")
    
    # Effect sizes
    print(f"\nEffect size ranges:")
    for effect_type in results_df['Effect_Type'].unique():
        if pd.notna(effect_type):
            subset = results_df[results_df['Effect_Type'] == effect_type]['Effect_Size']
            subset = subset.dropna()
            if len(subset) > 0:
                print(f"  {effect_type}: {subset.min():.4f} to {subset.max():.4f}")

def main():
    # Set paths
    data_folder = os.path.join(os.path.dirname(__file__), 'merged_by_time')
    output_path = os.path.join(os.path.dirname(__file__), "post_hoc_pairwise_results.csv")
    
    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found: {data_folder}")
        return
    
    print("Starting post-hoc pairwise analysis...")
    print(f"Data folder: {data_folder}")
    
    # Load data
    data_dict = load_merged_data(data_folder)
    
    if not data_dict:
        print("No data files found!")
        return
    
    # Perform pairwise analysis
    results = perform_pairwise_analysis(data_dict)
    
    if not results:
        print("No valid results obtained!")
        return
    
    # Apply FDR correction
    results = apply_fdr_correction(results)
    
    # Save results
    results_df = save_results(results, output_path)
    
    # Print summary
    print_summary(results_df)

if __name__ == "__main__":
    main()
