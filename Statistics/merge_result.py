import os
import pandas as pd

def merge_results(root_dir, output_file):
    """
    Merges all CSV files from a nested directory structure into a single CSV file.

    The directory structure is expected to be:
    root_dir/{feature_set}/{time_segment}/subject_{id}.csv

    Each CSV file should contain 'Subject' and 'Accuracy' columns.

    Args:
        root_dir (str): The root directory containing the results.
        output_file (str): The path to the output merged CSV file.
    """
    all_data = []
    
    # Check if the root directory exists
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory not found at '{root_dir}'")
        return

    # Walk through the directory
    for feature_set in os.listdir(root_dir):
        feature_path = os.path.join(root_dir, feature_set)
        if os.path.isdir(feature_path):
            for time_segment in os.listdir(feature_path):
                time_path = os.path.join(feature_path, time_segment)
                if os.path.isdir(time_path):
                    for filename in os.listdir(time_path):
                        if filename.endswith('.csv'):
                            file_path = os.path.join(time_path, filename)
                            try:
                                # Read the CSV file
                                df = pd.read_csv(file_path)
                                
                                # Add feature and time columns
                                df['Feature'] = feature_set
                                df['Time'] = time_segment
                                
                                all_data.append(df)
                            except Exception as e:
                                print(f"Could not read or process file {file_path}: {e}")

    if not all_data:
        print("No data found to merge.")
        return

    # Concatenate all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)

    # Reorder columns to be more intuitive
    if 'Subject' in merged_df.columns and 'Accuracy' in merged_df.columns:
        merged_df = merged_df[['Subject', 'Feature', 'Time', 'Accuracy']]
    else:
        print("Warning: 'Subject' or 'Accuracy' columns not found in all files.")


    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully merged all results into '{output_file}'")

if __name__ == '__main__':
    # Define the root directory for the results and the output file name
    results_root_dir = os.path.join('..\\', '00_result', '01_remove_activity_psd_sum')
    output_csv_file = 'merged_results.csv'
    
    merge_results(results_root_dir, output_csv_file)
