import argparse
import os
import sys
import scipy.io as sio

def check_file_exists(filepath, step_name):
    if not os.path.exists(filepath):
        print(f"Error: Required file {filepath} not found for {step_name}")
        sys.exit(1)

parser = argparse.ArgumentParser(description='Run all steps for EEG feature extraction and classification')
parser.add_argument('--subject-type', default='cross', type=str, help='Type of subjects (cross or first_batch)')
parser.add_argument('--session-length', default=30, type=int, help='Length of each video session in seconds')
parser.add_argument('--n-vids', default=24, type=int, help='Number of videos for classification')
parser.add_argument('--feature', default='DE', choices=['DE', 'DE_PSD', 'DE_PSD_H'], help='Feature combination: DE, DE_PSD, or DE_PSD_H')

args = parser.parse_args()
subject_type = args.subject_type
session_length = args.session_length
n_vids = args.n_vids
feature = args.feature

# # Step 1: DE Feature Calculation
print(f"Step 1: Calculating ALL features with session length {session_length}s...")
os.system(f"python pkl_to_mat.py --session-length {session_length} --feature {feature}")
check_file_exists("data.mat", "Step 1")

if n_vids == 24:
    print("Using 24 videos for binary classification.")
    os.system(f"python filter_neutral_videos.py --session-length {session_length}")
    check_file_exists("data_binary.mat", "Step 1-1")
    check_file_exists("vid_order_binary.mat", "Step 1-1 (Video Order)")

# Step 2: Running_norm Calculation
print(f"Step 2: Performing running normalization with {n_vids} videos, session length {session_length}s...")
os.system(f"python running_norm.py --n-vids {n_vids} --session-length {session_length} --feature {feature}")
check_file_exists(f"./running_norm_{n_vids}/normTrain_rnPreWeighted0.990_newPre_{n_vids}video_car", "Step 2")

# Step 3: Using LDS to Smooth the Data
print(f"Step 3: Smoothing data with LDS for {n_vids} videos, session length {session_length}s...")
os.system(f"python smooth_lds.py --n-vids {n_vids}")
check_file_exists(f"./smooth_{n_vids}", "Step 3")

# Step 4: Using SVM for Classification
print(f"Step 4: Training SVM classifier with {n_vids} videos, session length {session_length}s...")
os.system(f"python main_de_svm.py --subjects-type {subject_type} --valid-method 10-folds --n-vids {n_vids} --session-length {session_length}")

# Step 5: Evaluating Accuracy
print("Step 5: Testing SVM classifier...")
os.system(f"python main_de_svm.py --train-or-test test --subjects-type {subject_type} --n-vids {n_vids} --session-length {session_length}")

# Step 6: Plotting Results
print("Step 6: Plotting results...")
os.system(f"python plot_result.py --subject-type {subject_type} --n-vids {n_vids} --session-length {session_length}")

print("Complete!")
