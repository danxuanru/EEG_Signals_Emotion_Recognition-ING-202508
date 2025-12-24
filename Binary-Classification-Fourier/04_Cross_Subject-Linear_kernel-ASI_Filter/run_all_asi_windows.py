import argparse
import os
import sys
import scipy.io as sio

def check_file_exists(filepath, step_name):
    if not os.path.exists(filepath):
        print(f"Error: Required file {filepath} not found for {step_name}")
        sys.exit(1)

parser = argparse.ArgumentParser(description='Run all steps for EEG feature extraction and classification with AsI filtered data')
parser.add_argument('--feature', default='DE', choices=['DE', 'DE_PSD', 'DE_PSD_H'], help='Feature combination: DE, DE_PSD, or DE_PSD_H')
parser.add_argument('--window-sec', default=1, type=int, help='Window size in seconds for feature extraction')
parser.add_argument('--overlap-ratio', default=0, type=float, help='Overlap ratio for feature extraction')
parser.add_argument('--voting-method', default='weighted', choices=['weighted', 'majority', 'average_prob'], help='Voting method for ensemble classification')
parser.add_argument('--kernel', default='linear', choices=['linear', 'rbf'],
                    help='SVM kernel to use during training and testing')
args = parser.parse_args()

feature = args.feature
window_sec = args.window_sec
overlap_ratio = args.overlap_ratio
voting_method = args.voting_method
kernel = args.kernel

if overlap_ratio == 0:
    data_dir = f"../Processed_Data_after_AsI_{window_sec}s"
else:
    data_dir = f"../Processed_Data_after_AsI_{window_sec}s_overlap{int(overlap_ratio*100)}"

print("="*80)
print("EEG Emotion Recognition Pipeline with AsI Filtered Data")
print("="*80)
print(f"Configuration:")
print(f"  - Window size (sec): {window_sec}")
print(f"  - Overlap ratio: {overlap_ratio}")
print(f"  - Feature type: {feature}")
print("="*80)


# Step 1: ASI filter 
print(f"\nStep 1: Applying AsI filter to EEG data...", flush=True)
#os.system(f"python asi_filter.py --window-sec {window_sec} --overlap-ratio {overlap_ratio} --output-dir {data_dir}")


# Step 2: Convert AsI filtered features to MAT format
print(f"\nStep 2: Converting AsI filtered {feature} features to MAT format...", flush=True)
os.system(f"python save_feature_asi.py --feature {feature} --data-path {data_dir} --window-sec {window_sec} --overlap-ratio {overlap_ratio}")
# check_file_exists(f"pool_data_{feature}.mat", "Step 1")
check_file_exists(f"pooled_data_{feature}.mat", "Step 2")

# Step 3: Running_norm Calculation with irregular data
print(f"\nStep 3: Performing running normalization with AsI filtered data)...", flush=True)
os.system(f"python running_norm_asi.py --feature {feature}")
#check_file_exists(f"./running_norm_{feature}/{feature}_fold9.mat", "Step 2")


# Step 4: Using SVM for Classification
print(f"\nStep 4: Training SVM classifier with AsI filtered data...", flush=True)
print(f"Note: Using {feature} features may require longer convergence time for high-dimensional data")
os.system(f"python main_svm_asi_voting.py --train-or-test train --feature {feature} --kernel {kernel}")

# Check if training completed successfully
if not os.path.exists(f"result_asi_{feature}/svm_weights"):
    print("Warning: SVM training may not have completed successfully!")
    print("Check the training logs above for convergence issues.")
else:
    print("SVM training completed successfully!")

# Step 5: Evaluating Accuracy
print("\nStep 5: Testing SVM classifier...", flush=True)
os.system(f"python main_svm_asi_voting.py --train-or-test test --feature {feature} --voting-method {voting_method} --overlap-ratio {overlap_ratio} --kernel {kernel}")
# # Step 6: Plotting Results
# print("\nStep 6: Plotting results...", flush=True)
# os.system(f"python plot_result_asi.py --subject-type {subject_type} --n-vids {n_vids} --session-length {session_length} --remove-band {remove_band}")

print("\n" + "="*80)
print("Complete! AsI filtered data processing and classification finished.")
print("="*80)
