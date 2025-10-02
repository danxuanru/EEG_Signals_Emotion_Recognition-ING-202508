import os
import sys
import scipy.io as sio

def check_file_exists(filepath, step_name):
    if not os.path.exists(filepath):
        print(f"Error: Required file {filepath} not found for {step_name}")
        sys.exit(1)

# Define the number of videos for classification
n_vids = 24  # Use 24 for binary classification, 28 for 9-class classification

# # Step 1: DE Feature Calculation
print("Step 1: Calculating DE features...")
os.system("python save_de.py")
check_file_exists("data.mat", "Step 1")

# Step 2: Running_norm Calculation
print(f"Step 2: Performing running normalization with {n_vids} videos...")
os.system(f"python running_norm.py --n-vids {n_vids}")
# check_file_exists(f"./running_norm_{n_vids}/normTrain_rnPreWeighted0.990_newPre_{n_vids}video_car", "Step 2")

# Step 3: Using LDS to Smooth the Data
print(f"Step 3: Smoothing data with LDS for {n_vids} videos...")
os.system(f"python smooth_lds.py --n-vids {n_vids}")
check_file_exists(f"./smooth_{n_vids}", "Step 3")

# Step 4: Using SVM for Classification
print(f"Step 4: Training SVM classifier with {n_vids} videos...")
os.system(f"python main_de_svm.py --subjects-type cross --valid-method 10-folds --n-vids {n_vids}") # <-- original
# os.system("python main_de_svm.py --subjects-type cross --valid-method 10-folds --n-vids 28 --n-jobs 8 --early-stopping 3")
# os.system("python main_de_svm.py --subjects-type cross --valid-method 10-folds --grid-search fast --n-jobs 4 --early-stopping 3")
# os.system("python main_de_svm.py --n-jobs -1 --grid-search fast")

# Step 5: Evaluating Accuracy
print("Step 5: Testing SVM classifier...")
os.system(f"python main_de_svm.py --train-or-test test --n-vids {n_vids}")

# Step 6: Plotting Results
print("Step 6: Plotting results...")
os.system("python plot_result.py")

print("Complete!")
