import os
import sys
import scipy.io as sio
import argparse
import numpy as np

def check_file_exists(filepath, step_name):
    if not os.path.exists(filepath):
        print(f"Error: Required file {filepath} not found for {step_name}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run the full EEG classification pipeline')
    parser.add_argument('--n-vids', default=24, type=int,
                        help='Number of videos (24 for binary, 28 for 9-class)')
    parser.add_argument('--kernel', default='rbf', type=str,
                        help='Kernel type for SVM (linear, rbf, poly, sigmoid)')
    parser.add_argument('--valid-method', default='10-folds', type=str,
                        help='Validation method (10-folds, loo)')
    parser.add_argument('--search-method', default='grid', type=str,
                        help='Parameter search method (grid, random, bayesian)')
    parser.add_argument('--n-jobs', default=1, type=int,
                        help='Number of parallel jobs (-1 for all CPUs)')
    parser.add_argument('--feature-selection', default='none', type=str,
                        help='Feature selection method (none, selectk, pca)')
    parser.add_argument('--n-features', default=50, type=int,
                        help='Number of features to select')
    parser.add_argument('--early-stopping', default=0, type=int,
                        help='Early stopping after N iterations without improvement (0 to disable)')
    parser.add_argument('--subjects-type', default='cross', type=str,
                        help='Subject validation type (cross, intra)')
    parser.add_argument('--skip-data-prep', action='store_true',
                        help='Skip data preparation steps and only run SVM')
    parser.add_argument('--c-value', type=float, default=0.01, help='Fixed C parameter')
    parser.add_argument('--gamma-value', type=float, default=0.1, help='Fixed gamma parameter')
    
    args = parser.parse_args()
    
    # Define the number of videos for classification
    n_vids = args.n_vids
    kernel_type = args.kernel
    
    if not args.skip_data_prep:
        # Step 1: DE Feature Calculation
        print("Step 1: Calculating ALL features...")
        os.system("python pkl_to_mat.py")
        check_file_exists("data.mat", "Step 1")

        if n_vids == 24:
            print("Using 24 videos for binary classification.")
            os.system("python filter_neutral_videos.py")
            check_file_exists("data_binary.mat", "Step 1-1")
            check_file_exists("vid_order_binary.mat", "Step 1-1 (Video Order)")


        # Step 2: Running_norm Calculation
        print(f"Step 2: Performing running normalization with {n_vids} videos...")
        os.system(f"python running_norm.py --n-vids {n_vids}")

        # Step 3: Using LDS to Smooth the Data
        print(f"Step 3: Smoothing data with LDS for {n_vids} videos...")
        os.system(f"python smooth_lds.py --n-vids {n_vids}")
        check_file_exists(f"./smooth_{n_vids}", "Step 3")
    else:
        print("Skipping data preparation steps...")
    
    # Step 4: Using SVM for Classification
    print(f"Step 4: Training SVM classifier with {n_vids} videos and {kernel_type} kernel...")
    train_cmd = (
        f"python main_de_svm.py --subjects-type {args.subjects_type} "
        f"--valid-method {args.valid_method} --n-vids {n_vids} "
        f"--kernel {kernel_type} --search-method {args.search_method} "
        f"--n-jobs {args.n_jobs} --feature-selection {args.feature_selection} "
        f"--n-features {args.n_features} --early-stopping {args.early_stopping}"
    )
    print(f"Running command: {train_cmd}")
    os.system(train_cmd)

    # Step 5: Evaluating Accuracy
    print("Step 5: Testing SVM classifier...")
    test_cmd = (
        f"python main_de_svm.py --train-or-test test "
        f"--subjects-type {args.subjects_type} --valid-method {args.valid_method} "
        f"--n-vids {n_vids} --kernel {kernel_type} "
        f"--feature-selection {args.feature_selection} --n-features {args.n_features}"
    )
    print(f"Running command: {test_cmd}")
    os.system(test_cmd)

    # Step 6: Plotting Results
    print("Step 6: Plotting results...")
    os.system("python plot_result.py")

    print("Complete!")

if __name__ == "__main__":
    main()
