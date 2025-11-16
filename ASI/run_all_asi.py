import argparse
import os
import sys
import scipy.io as sio

def check_file_exists(filepath, step_name):
    if not os.path.exists(filepath):
        print(f"Error: Required file {filepath} not found for {step_name}")
        sys.exit(1)

parser = argparse.ArgumentParser(description='Run all steps for EEG feature extraction and classification with AsI filtered data')
parser.add_argument('--remove-band', default=0, type=int, help='Remove specific frequency band (1-5) or None to include all bands')
parser.add_argument('--feature', default='DE', choices=['DE', 'DE_PSD', 'DE_PSD_H'], help='Feature combination: DE, DE_PSD, or DE_PSD_H')
parser.add_argument('--use-asi-filtered', action='store_true', default=True, help='Use AsI filtered data')
args = parser.parse_args()


remove_band = args.remove_band
feature = args.feature
use_asi_filtered = args.use_asi_filtered

print("="*80)
print("EEG Emotion Recognition Pipeline with AsI Filtered Data")
print("="*80)
print(f"Configuration:")
print(f"  - Remove band: {remove_band}")
print(f"  - Feature type: {feature}")
print(f"  - Use AsI filtered data: {use_asi_filtered}")
print("="*80)

# Step 1: Convert AsI filtered features to MAT format
print(f"\nStep 1: Converting AsI filtered {feature} features to MAT format...", flush=True)
os.system(f"python save_feature_asi.py --feature {feature}")
# check_file_exists(f"pool_data_{feature}.mat", "Step 1")
check_file_exists("pooled_data.mat", "Step 1")

# Step 2: Running_norm Calculation with irregular data
print(f"\nStep 2: Performing running normalization with AsI filtered data)...", flush=True)
os.system(f"python running_norm_asi.py --feature {feature}")
#check_file_exists(f"./running_norm_{feature}/{feature}_fold9.mat", "Step 2")

# # Step 3: Using LDS to Smooth the Data
# print(f"\nStep 3: Smoothing AsI filtered data with LDS...", flush=True)
# os.system(f"python smooth_lds_asi.py --n-vids {n_vids} --session-length {session_length}")
# check_file_exists(f"./smooth_asi_{n_vids}", "Step 3")

# Step 4: Using SVM for Classification
print(f"\nStep 4: Training SVM classifier with AsI filtered data...", flush=True)
print(f"Note: Using {feature} features may require longer convergence time for high-dimensional data")
os.system(f"python main_svm_asi_voting.py --train-or-test train --feature {feature}")

# Check if training completed successfully
if not os.path.exists(f"result_asi_{feature}/svm_weights"):
    print("Warning: SVM training may not have completed successfully!")
    print("Check the training logs above for convergence issues.")
else:
    print("SVM training completed successfully!")

# Step 5: Evaluating Accuracy
print("\nStep 5: Testing SVM classifier...", flush=True)
os.system(f"python main_svm_asi_voting.py --train-or-test test --feature {feature}")

# # Step 6: Plotting Results
# print("\nStep 6: Plotting results...", flush=True)
# os.system(f"python plot_result_asi.py --subject-type {subject_type} --n-vids {n_vids} --session-length {session_length} --remove-band {remove_band}")

print("\n" + "="*80)
print("Complete! AsI filtered data processing and classification finished.")
print("="*80)
