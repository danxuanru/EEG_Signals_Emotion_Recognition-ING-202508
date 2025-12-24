import os
import pandas as pd
from collections import defaultdict
import re

def merge_results_by_time(root_dir, output_dir):
    """
    Merges CSV files based on time groups within each feature directory.

    The directory structure is expected to be:
    root_dir/{feature_set}/{time_group}_{sub_index}/subject_{id}.csv

    It groups by {time_group} (e.g., '01_15s', '02_10s') and creates a pivoted CSV
    for each group within each feature set.

    Args:
        root_dir (str): The root directory containing the results.
        output_dir (str): The directory to save the merged CSV files.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Check if the root directory exists
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory not found at '{root_dir}'")
        return

    # Walk through the feature directories
    for feature_set in os.listdir(root_dir):
        feature_path = os.path.join(root_dir, feature_set)
        if not os.path.isdir(feature_path):
            continue

        print(f"Processing feature set: {feature_set}...")
        
        time_segment_groups = defaultdict(list)
        # Group time segment directories by prefix (e.g., '01_15s')
        for time_segment in os.listdir(feature_path):
            match = re.match(r'(\d+_\d+s)', time_segment)
            if match:
                group_key = match.group(1)
                time_segment_groups[group_key].append(time_segment)

        # Process each group
        for group_key, time_segments in time_segment_groups.items():
            all_data_for_group = []
            for time_segment in time_segments:
                time_path = os.path.join(feature_path, time_segment)
                if os.path.isdir(time_path):
                    for filename in os.listdir(time_path):
                        if filename.endswith('.csv'):
                            file_path = os.path.join(time_path, filename)
                            try:
                                df = pd.read_csv(file_path, header=None, names=['Subject', 'Accuracy'])
                                df['Time'] = time_segment
                                df = df[1:]
                                all_data_for_group.append(df)
                            except Exception as e:
                                print(f"Could not read or process file {file_path}: {e}")
            
            if not all_data_for_group:
                print(f"  No data found for group {group_key} in {feature_set}.")
                continue

            # Concatenate and pivot the data
            group_df = pd.concat(all_data_for_group, ignore_index=True)
            
            if 'Subject' not in group_df.columns or 'Accuracy' not in group_df.columns:
                print(f"  Skipping group {group_key} in {feature_set} due to missing 'Subject' or 'Accuracy' columns.")
                continue

            # Pivot the table
            pivoted_df = group_df.pivot(index='Subject', columns='Time', values='Accuracy')
            
            # Sort columns to maintain order (e.g., 01_15s_01, 01_15s_02)
            pivoted_df = pivoted_df.reindex(sorted(pivoted_df.columns), axis=1)

            # Reset index to make 'Subject' a column again
            pivoted_df.reset_index(inplace=True)

            # Save the pivoted dataframe to a new CSV file
            output_filename = f"{feature_set}_{group_key}.csv"
            output_file_path = os.path.join(output_dir, output_filename)
            pivoted_df.to_csv(output_file_path, index=False)
            print(f"  Successfully merged and saved '{output_file_path}'")

if __name__ == '__main__':
    results_root_dir = os.path.join('..\\', '00_result', '01_remove_activity_psd_sum')
    output_csv_dir = 'merged_by_time'
    
    merge_results_by_time(results_root_dir, output_csv_dir)
