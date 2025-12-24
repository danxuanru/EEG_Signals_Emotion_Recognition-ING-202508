import os
import sys
import scipy.io as sio
import argparse

def check_file_exists(filepath, step_name):
    if not os.path.exists(filepath):
        print(f"Error: Required file {filepath} not found for {step_name}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run intra-subject binary classification with SVM')
    parser.add_argument('--kernel', default='linear', type=str,
                        help='Kernel type for SVM (linear, rbf, poly, sigmoid)')
    parser.add_argument('--n-jobs', default=-1, type=int,
                        help='Number of parallel jobs (-1 for all CPUs)')
    parser.add_argument('--feature-selection', default='pca', type=str,
                        help='Feature selection method (none, selectk, pca)')
    parser.add_argument('--n-features', default=30, type=int,
                        help='Number of features to select')
    parser.add_argument('--skip-data-prep', action='store_true',
                        help='Skip data preparation steps')
    parser.add_argument('--sec', default=10, type=int,
                        help='Time length in seconds for each video')
    
    args = parser.parse_args()
    
    # Define the number of videos for binary classification
    n_vids = 24  # Binary classification (positive/negative)
    kernel_type = args.kernel
    sec = args.sec  # Use sec from command line arguments
    
    if not args.skip_data_prep:
        # Step 1: DE Feature Calculation
        print("Step 1: Calculating ALL features...")
        #os.system(f"python pkl_to_mat.py --sec {sec}")
        check_file_exists("data.mat", "Step 1")

        # Step 1.1: Filter neutral videos for binary classification
        print("Step 1.1: Filtering neutral videos for binary classification...")
        #os.system(f"python filter_neutral_videos.py --sec {sec}")
        check_file_exists("data_binary.mat", "Step 1.1")
        check_file_exists("vid_order_binary.mat", "Step 1.1 (Video Order)")

        # Step 2: Running_norm Calculation
        print(f"Step 2: Performing running normalization with {n_vids} videos...")
        #os.system(f"python running_norm.py --n-vids {n_vids} --sec {sec}")

        # Step 3: Using LDS to Smooth the Data
        print(f"Step 3: Smoothing data with LDS for {n_vids} videos...")
        #os.system(f"python smooth_lds.py --n-vids {n_vids} --sec {sec}")
        check_file_exists(f"./smooth_{n_vids}", "Step 3")
    else:
        print("Skipping data preparation steps...")
    
    # Step 4: Using SVM for Classification with intra-subject validation
    print(f"Step 4: Training intra-subject SVM classifier with {n_vids} videos...")
    train_cmd = (
        f"python main_de_svm.py --subjects-type intra --valid-method 10-folds "
        f"--n-vids {n_vids} --kernel {kernel_type} --search-method grid --sec {sec}"
    )
    print(f"Running command: {train_cmd}")
    os.system(train_cmd)

    # Step 5: Evaluating Accuracy
    print("Step 5: Testing SVM classifier...")
    test_cmd = (
        f"python main_de_svm.py --train-or-test test --subjects-type intra "
        f"--valid-method 10-folds --n-vids {n_vids} --kernel {kernel_type} --sec {sec}"
    )
    print(f"Running command: {test_cmd}")
    os.system(test_cmd)

    # Step 6: Plotting Results
    print("Step 6: Plotting results...")
    os.system(f"python plot_result.py --subject-type intra --n-vids {n_vids} --kernel {kernel_type}")

    print("Complete!")

if __name__ == "__main__":
    main()
