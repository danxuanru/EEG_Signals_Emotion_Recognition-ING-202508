import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import os
import glob
from pathlib import Path

def load_merged_data(data_folder):
    """
    Load all CSV files from merged_by_time folder
    """
    csv_files = glob.glob(os.path.join(data_folder, "*_10s.csv"))
    
    if len(csv_files) != 3:
        print(f"Warning: Expected 3 CSV files, found {len(csv_files)}")
    
    data_dict = {}
    for file_path in csv_files:
        filename = Path(file_path).stem
        df = pd.read_csv(file_path)
        data_dict[filename] = df.iloc[:, 1:]  # Extract all columns except the first one
        print(f"Loaded {filename}: {df.shape}")
    
    return data_dict

def perform_friedman_tests(data_dict):
    """
    Perform Friedman tests for each feature combination
    """
    results = []
    
    for feature_name, df in data_dict.items():
        print(f"\nProcessing {feature_name}...")
        
        # Get all numeric columns (assuming they represent different 10s segments)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 3:
            print(f"Warning: {feature_name} has less than 3 numeric columns, skipping Friedman test")
            continue
        
        # Prepare data for Friedman test
        # Each column represents a different time condition
        time_segments_data = []
        for col in numeric_columns:
            # Remove NaN values
            clean_data = df[col].dropna()
            time_segments_data.append(clean_data)
        
        # Ensure all segments have the same length for Friedman test
        min_length = min(len(segment) for segment in time_segments_data)
        time_segments_data = [segment[:min_length] for segment in time_segments_data]
        
        # Perform Friedman test
        try:
            statistic, p_value = friedmanchisquare(*time_segments_data)
            
            result = {
                'Feature_Combination': feature_name,
                'Friedman_Statistic': statistic,
                'P_Value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No',
                'Num_Time_Segments': len(numeric_columns),
                'Sample_Size': min_length
            }
            
            results.append(result)
            print(f"Friedman test completed: χ² = {statistic:.4f}, p = {p_value:.4f}")
            
        except Exception as e:
            print(f"Error performing Friedman test for {feature_name}: {e}")
    
    return results

def save_results(results, output_path):
    """
    Save Friedman test results to CSV
    """
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    return results_df

def main():
    # Set paths
    data_folder = os.path.join(os.path.dirname(__file__), 'merged_by_time')
    output_path = os.path.join(os.path.dirname(__file__), "friedman_test_results.csv")

    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found: {data_folder}")
        return
    
    print("Starting Friedman test analysis...")
    print(f"Data folder: {data_folder}")
    
    # Load data
    data_dict = load_merged_data(data_folder)
    
    if not data_dict:
        print("No data files found!")
        return
    
    # Perform Friedman tests
    results = perform_friedman_tests(data_dict)
    
    if not results:
        print("No valid results obtained!")
        return
    
    # Save results
    results_df = save_results(results, output_path)
    
    # Display summary
    print("\n" + "="*50)
    print("FRIEDMAN TEST RESULTS SUMMARY")
    print("="*50)
    print(results_df.to_string(index=False))
    
    significant_count = sum(1 for r in results if r['Significant'] == 'Yes')
    print(f"\nSignificant results (p < 0.05): {significant_count}/{len(results)}")

if __name__ == "__main__":
    main()
