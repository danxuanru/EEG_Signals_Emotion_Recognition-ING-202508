import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def load_and_analyze_mat(file_path, num_samples=10):
    """Load .mat file and analyze the first few samples"""
    try:
        # Load the .mat file
        data_dict = sio.loadmat(file_path)
        
        # Extract the 'de' data
        if 'de' in data_dict:
            data = data_dict['de']
        else:
            print(f"Available keys in {file_path}: {list(data_dict.keys())}")
            return None
        
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        
        # Display first few samples
        print(f"\nFirst {num_samples} time points for first subject:")
        print("Shape: (time_points, features)")
        print("-" * 40)
        
        if len(data.shape) == 3:  # (subjects, time_points, features)
            first_subject_data = data[0, :num_samples, :]
            print(f"Showing data[0, :{num_samples}, :] - shape: {first_subject_data.shape}")
            
            # Show first few features for each time point
            max_features_to_show = min(10, first_subject_data.shape[1])
            print(f"First {max_features_to_show} features:")
            
            for t in range(num_samples):
                print(f"t={t:2d}: {first_subject_data[t, :max_features_to_show]}")
                
        else:
            print(f"Unexpected data shape: {data.shape}")
            
        return data
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def export_statistics_to_csv(stats_data, output_dir="./analysis_results"):
    """Export analysis results to CSV files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Overall statistics CSV
    overall_stats = []
    for file_stats in stats_data:
        overall_stats.append({
            'filename': file_stats['filename'],
            'data_shape': str(file_stats['shape']),
            'min_value': file_stats['min'],
            'max_value': file_stats['max'],
            'mean': file_stats['mean'],
            'std': file_stats['std'],
            'median': file_stats['median'],
            'nan_count': file_stats['nan_count'],
            'inf_count': file_stats['inf_count'],
            'zero_count': file_stats['zero_count'],
            'percentile_1': file_stats['percentiles'][1],
            'percentile_5': file_stats['percentiles'][5],
            'percentile_25': file_stats['percentiles'][25],
            'percentile_75': file_stats['percentiles'][75],
            'percentile_95': file_stats['percentiles'][95],
            'percentile_99': file_stats['percentiles'][99]
        })
    
    overall_df = pd.DataFrame(overall_stats)
    overall_csv_path = os.path.join(output_dir, f"overall_statistics_{timestamp}.csv")
    overall_df.to_csv(overall_csv_path, index=False)
    print(f"Overall statistics saved to: {overall_csv_path}")
    
    # Feature-wise statistics CSV
    feature_stats = []
    for file_stats in stats_data:
        if 'feature_means' in file_stats and 'feature_stds' in file_stats:
            for i, (mean, std) in enumerate(zip(file_stats['feature_means'], file_stats['feature_stds'])):
                feature_stats.append({
                    'filename': file_stats['filename'],
                    'feature_index': i,
                    'feature_mean': mean,
                    'feature_std': std
                })
    
    if feature_stats:
        feature_df = pd.DataFrame(feature_stats)
        feature_csv_path = os.path.join(output_dir, f"feature_statistics_{timestamp}.csv")
        feature_df.to_csv(feature_csv_path, index=False)
        print(f"Feature-wise statistics saved to: {feature_csv_path}")
    
    return overall_csv_path, feature_csv_path if feature_stats else None

def load_original_data(base_dir="./"):
    """Load original data before normalization - now uses adjusted_data"""
    # First try to load the adjusted data that matches preprocessing
    adjusted_data_path = os.path.join(base_dir, "adjusted_data.mat")
    if os.path.exists(adjusted_data_path):
        try:
            data_dict = sio.loadmat(adjusted_data_path)
            if 'adjusted_data' in data_dict:
                print(f"Found adjusted original data in: adjusted_data.mat")
                return data_dict['adjusted_data'], "adjusted_data.mat"
            elif 'data' in data_dict:
                print(f"Found data in adjusted_data.mat")
                return data_dict['data'], "adjusted_data.mat"
        except Exception as e:
            print(f"Error loading adjusted_data.mat: {e}")
    
    # Fallback to original data files if adjusted data not found
    possible_files = ['data.mat', 'data_binary.mat']
    
    for filename in possible_files:
        file_path = os.path.join(base_dir, filename)
        if os.path.exists(file_path):
            try:
                data_dict = sio.loadmat(file_path)
                if 'data' in data_dict:
                    print(f"Found original data in: {filename} (fallback - may not match preprocessing)")
                    return data_dict['data'], filename
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print("No original data file found (neither adjusted_data.mat nor data.mat)")
    return None, None

def analyze_data_statistics(data, file_name, collect_stats=False):
    """Analyze statistical properties of the data"""
    print(f"\n{'='*60}")
    print(f"Statistical Analysis for {file_name}")
    print(f"{'='*60}")
    
    if data is None:
        print("No data to analyze")
        return None
    
    # Calculate statistics
    data_flat = data.flatten()
    
    stats = {
        'filename': file_name,
        'shape': data.shape,
        'min': float(np.min(data_flat)),
        'max': float(np.max(data_flat)),
        'mean': float(np.mean(data_flat)),
        'std': float(np.std(data_flat)),
        'median': float(np.median(data_flat)),
        'nan_count': int(np.sum(np.isnan(data_flat))),
        'inf_count': int(np.sum(np.isinf(data_flat))),
        'zero_count': int(np.sum(data_flat == 0))
    }
    
    print(f"Overall Statistics:")
    print(f"  Min value: {stats['min']:.6f}")
    print(f"  Max value: {stats['max']:.6f}")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std: {stats['std']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    
    # Percentiles
    percentiles = [1, 5, 25, 75, 95, 99]
    stats['percentiles'] = {}
    print(f"\nPercentiles:")
    for p in percentiles:
        pct_val = float(np.percentile(data_flat, p))
        stats['percentiles'][p] = pct_val
        print(f"  {p:2d}%: {pct_val:.6f}")
    
    print(f"\nData Quality:")
    print(f"  NaN count: {stats['nan_count']}")
    print(f"  Inf count: {stats['inf_count']}")
    print(f"  Zero count: {stats['zero_count']}")
    
    # Feature-wise statistics (if 3D data)
    if len(data.shape) == 3:
        n_subs, n_time, n_features = data.shape
        print(f"\nPer-feature statistics (averaged across subjects and time):")
        
        # Calculate mean and std for each feature
        feature_means = np.mean(data, axis=(0, 1))
        feature_stds = np.std(data, axis=(0, 1))
        
        stats['feature_means'] = feature_means.tolist()
        stats['feature_stds'] = feature_stds.tolist()
        
        print(f"  Feature means range: [{np.min(feature_means):.6f}, {np.max(feature_means):.6f}]")
        print(f"  Feature stds range: [{np.min(feature_stds):.6f}, {np.max(feature_stds):.6f}]")
        
        # Show first 10 features
        print(f"\nFirst 10 features - Mean ± Std:")
        for i in range(min(10, n_features)):
            print(f"  Feature {i:2d}: {feature_means[i]:8.6f} ± {feature_stds[i]:8.6f}")
    
    return stats if collect_stats else None

def compare_normalization_effects(original_path, normalized_path):
    """Compare original and normalized data"""
    print(f"\n{'='*80}")
    print("NORMALIZATION EFFECT COMPARISON")
    print(f"{'='*80}")
    
    # Load both datasets
    orig_data = None
    norm_data = None
    
    if os.path.exists(original_path):
        orig_dict = sio.loadmat(original_path)
        orig_data = orig_dict.get('data', orig_dict.get('de'))
        
    if os.path.exists(normalized_path):
        norm_dict = sio.loadmat(normalized_path)
        norm_data = norm_dict.get('de')
    
    if orig_data is not None and norm_data is not None:
        print("Before normalization:")
        print(f"  Shape: {orig_data.shape}")
        print(f"  Range: [{np.min(orig_data):.6f}, {np.max(orig_data):.6f}]")
        print(f"  Mean ± Std: {np.mean(orig_data):.6f} ± {np.std(orig_data):.6f}")
        
        print("\nAfter normalization:")
        print(f"  Shape: {norm_data.shape}")
        print(f"  Range: [{np.min(norm_data):.6f}, {np.max(norm_data):.6f}]")
        print(f"  Mean ± Std: {np.mean(norm_data):.6f} ± {np.std(norm_data):.6f}")
        
        # Calculate normalization ratio
        orig_std = np.std(orig_data)
        norm_std = np.std(norm_data)
        print(f"\nNormalization effect:")
        print(f"  Std reduction ratio: {orig_std/norm_std:.2f}x")
        print(f"  Mean shift: {np.mean(orig_data)} -> {np.mean(norm_data):.6f}")
        
    else:
        print("Could not load both original and normalized data for comparison")

def create_comparison_csv(original_stats, normalized_stats_list, output_dir="./analysis_results"):
    """Create detailed comparison CSV between original and normalized data"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    comparison_data = []
    
    if original_stats:
        # Add original data stats
        comparison_data.append({
            'data_type': 'Original',
            'filename': original_stats['filename'],
            'shape': str(original_stats['shape']),
            'min_value': original_stats['min'],
            'max_value': original_stats['max'],
            'range': original_stats['max'] - original_stats['min'],
            'mean': original_stats['mean'],
            'std': original_stats['std'],
            'median': original_stats['median'],
            'nan_count': original_stats['nan_count'],
            'inf_count': original_stats['inf_count'],
            'zero_count': original_stats['zero_count'],
            'percentile_1': original_stats['percentiles'][1],
            'percentile_99': original_stats['percentiles'][99],
            'normalization_effect': 'baseline'
        })
    
    # Add normalized data stats with comparison metrics
    for norm_stats in normalized_stats_list:
        if original_stats:
            # Calculate normalization effects
            range_reduction = (original_stats['max'] - original_stats['min']) / (norm_stats['max'] - norm_stats['min']) if (norm_stats['max'] - norm_stats['min']) != 0 else np.inf
            std_reduction = original_stats['std'] / norm_stats['std'] if norm_stats['std'] != 0 else np.inf
            mean_shift = abs(original_stats['mean'] - norm_stats['mean'])
            
            normalization_effect = f"Range:{range_reduction:.2f}x, Std:{std_reduction:.2f}x, MeanShift:{mean_shift:.6f}"
        else:
            normalization_effect = 'no_original_data'
        
        comparison_data.append({
            'data_type': 'Normalized',
            'filename': norm_stats['filename'],
            'shape': str(norm_stats['shape']),
            'min_value': norm_stats['min'],
            'max_value': norm_stats['max'],
            'range': norm_stats['max'] - norm_stats['min'],
            'mean': norm_stats['mean'],
            'std': norm_stats['std'],
            'median': norm_stats['median'],
            'nan_count': norm_stats['nan_count'],
            'inf_count': norm_stats['inf_count'],
            'zero_count': norm_stats['zero_count'],
            'percentile_1': norm_stats['percentiles'][1],
            'percentile_99': norm_stats['percentiles'][99],
            'normalization_effect': normalization_effect
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv_path = os.path.join(output_dir, f"normalization_comparison_{timestamp}.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Normalization comparison saved to: {comparison_csv_path}")
    
    return comparison_csv_path

def main():
    # Set paths
    base_dir = "./"
    
    # Load original data for comparison
    print("Loading adjusted original data for comparison...")
    original_data, original_filename = load_original_data(base_dir)
    original_stats = None
    
    if original_data is not None:
        print(f"Original data shape: {original_data.shape}")
        print(f"Data source: {original_filename}")
        
        # Check if this is already preprocessed/adjusted data
        if "adjusted_data" in original_filename:
            print("Using preprocessed adjusted data - should match normalization input")
            # Assume adjusted data is already in the correct format for comparison
            if len(original_data.shape) == 3:  # Already reshaped (subjects, time_points, features)
                original_stats = analyze_data_statistics(original_data, f"Adjusted_{original_filename}", collect_stats=True)
            elif len(original_data.shape) == 4:  # Still needs reshaping
                n_subs, chn, n_timepoints, n_features = original_data.shape
                original_data_reshaped = original_data.transpose([0,2,3,1]).reshape(n_subs, n_timepoints, chn*n_features)
                print(f"Reshaped adjusted data to: {original_data_reshaped.shape}")
                original_stats = analyze_data_statistics(original_data_reshaped, f"Adjusted_{original_filename}", collect_stats=True)
        else:
            print("Using raw data - may not exactly match normalization input due to preprocessing")
            # Reshape original data to match normalized format if needed
            if len(original_data.shape) == 4:  # (subjects, channels, time_points, features)
                # Match the reshaping logic from running_norm_multiscale.py
                n_subs, chn, n_timepoints, n_features = original_data.shape
                original_data_reshaped = original_data.transpose([0,2,3,1]).reshape(n_subs, n_timepoints, chn*n_features)
                print(f"Reshaped original data to: {original_data_reshaped.shape}")
                original_stats = analyze_data_statistics(original_data_reshaped, f"Raw_{original_filename}", collect_stats=True)
            else:
                original_stats = analyze_data_statistics(original_data, f"Raw_{original_filename}", collect_stats=True)
    
    # Look for normalized data files
    norm_dirs = ["running_norm_multiscale_28", "running_norm_28", "running_norm_24"]
    
    mat_files = []
    found_norm_dir = None
    
    for norm_dir in norm_dirs:
        norm_path = os.path.join(base_dir, norm_dir)
        if os.path.exists(norm_path):
            found_norm_dir = norm_path
            # Find all .mat files in subdirectories
            for root, dirs, files in os.walk(norm_path):
                for file in files:
                    if file.endswith('.mat'):
                        mat_files.append(os.path.join(root, file))
            break
    
    if mat_files:
        print(f"Found {len(mat_files)} .mat files in {found_norm_dir}")
        
        # Collect statistics for CSV export
        all_stats = []
        
        # Analyze first few files
        for i, mat_file in enumerate(mat_files[:]):  # Analyze first 3 files
            print(f"\n{'*'*80}")
            print(f"ANALYZING FILE {i+1}/{min(3, len(mat_files))}")
            print(f"{'*'*80}")
            
            # Load and display samples
            data = load_and_analyze_mat(mat_file, num_samples=10)
            
            # Analyze statistics and collect for CSV
            stats = analyze_data_statistics(data, os.path.basename(mat_file), collect_stats=True)
            if stats:
                all_stats.append(stats)
        
        # Export statistics to CSV
        if all_stats:
            print(f"\n{'='*80}")
            print("EXPORTING ANALYSIS RESULTS TO CSV")
            print(f"{'='*80}")
            
            # Export individual file statistics
            overall_csv, feature_csv = export_statistics_to_csv(all_stats)
            
            # Create comparison CSV with original data
            comparison_csv = create_comparison_csv(original_stats, all_stats)
            
            print(f"\nCSV files generated:")
            print(f"  1. Overall statistics: {overall_csv}")
            if feature_csv:
                print(f"  2. Feature statistics: {feature_csv}")
            print(f"  3. Normalization comparison: {comparison_csv}")
            
    else:
        print("No .mat files found in normalized data directories")
        print(f"Searched directories: {norm_dirs}")

if __name__ == "__main__":
    main()
