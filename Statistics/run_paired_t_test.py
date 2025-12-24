import os
import pandas as pd
from scipy import stats
from itertools import combinations
import re

def perform_paired_t_test(input_dir, output_file):
    """
    Performs paired samples t-tests on CSV files in the input directory,
    aggregates the results, and saves them to a single CSV file.

    Args:
        input_dir (str): The directory containing the merged CSV files.
        output_file (str): The path to save the aggregated t-test results.
    """
    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    all_results = []

    # Iterate over each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv') and '30s' not in filename:
            print(f"Processing file: {filename}...")
            file_path = os.path.join(input_dir, filename)
            
            # Extract feature_set and time_group from filename
            match = re.match(r'(.+)_(\d+_\d+s)\.csv', filename)
            if not match:
                print(f"  Skipping file with unexpected name format: {filename}")
                continue
            
            feature_set, time_group = match.groups()

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"  Could not read file {file_path}: {e}")
                continue

            # Get the time segment columns to compare
            time_columns = [col for col in df.columns if col != 'Subject']

            # Generate all unique pairs of time columns for comparison
            for col1, col2 in combinations(time_columns, 2):
                # Prepare the data for the t-test
                group1_data = df[col1].dropna()
                group2_data = df[col2].dropna()
                
                # Ensure the data is properly paired by index
                common_index = group1_data.index.intersection(group2_data.index)
                if len(common_index) < 2: # Need at least 2 pairs for a t-test
                    print(f"  Not enough paired data for {col1} and {col2} in {filename}")
                    continue

                paired_group1 = group1_data.loc[common_index]
                paired_group2 = group2_data.loc[common_index]

                # Perform the paired samples t-test
                t_statistic, p_value = stats.ttest_rel(paired_group1, paired_group2)

                # Store the result
                all_results.append({
                    'feature_set': feature_set,
                    'time_group': time_group,
                    'comparison_1': col1,
                    'comparison_2': col2,
                    't_statistic': t_statistic,
                    'p_value': p_value
                })

    if not all_results:
        print("No t-test results were generated.")
        return

    # Create a DataFrame from the results
    results_df = pd.DataFrame(all_results)

    # Save the results to the output CSV file
    results_df.to_csv(output_file, index=False)
    print(f"\nPaired t-test results have been successfully saved to '{output_file}'")

if __name__ == '__main__':
    input_directory = 'merged_by_time'
    output_csv_file = 't_test_results.csv'
    
    perform_paired_t_test(input_directory, output_csv_file)
