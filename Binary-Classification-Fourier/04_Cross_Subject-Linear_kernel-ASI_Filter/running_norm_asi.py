"""
Running normalization for AsI filtered irregular data
Handles different numbers of windows per subject
Generates separate files for each fold (similar to original running_norm.py)
"""
import numpy as np
import scipy.io as sio
import os
import argparse
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='Running normalization for AsI filtered data')
parser.add_argument('--feature', default='DE', type=str, help='Feature type')
parser.add_argument('--remove-band', default=0, type=int, help='Frequency band to remove')
parser.add_argument('--random-state', default=42, type=int, help='Random state for reproducibility')
args = parser.parse_args()


feature_type = args.feature
remove_band = args.remove_band
random_state = args.random_state

# Load data
mat_file = f'./pooled_data_{feature_type}.mat'
print(f"Loading {mat_file}...")
if not os.path.exists(mat_file):
    print(f"Error: {mat_file} not found!")
    exit(1)

mat_data = sio.loadmat(mat_file)
X = mat_data['X']
Y = mat_data['Y'].flatten()

print(f"Data shape: X={X.shape}, Y={Y.shape}")

# Extract metadata (subject and video indices are in first two columns)
subject_ids = X[:, 0].astype(int)
video_ids = X[:, 1].astype(int)
features = X[:, 2:]  # Actual feature data

print(f"Features shape after extracting metadata: {features.shape}")
print(f"Unique subjects: {len(np.unique(subject_ids))}")
print(f"Unique videos: {len(np.unique(video_ids))}")

# Get unique subjects and setup 10-fold cross validation
unique_subjects = np.unique(subject_ids)
n_subs = len(unique_subjects)
n_folds = 10
n_per = round(n_subs / n_folds)

print(f"Total subjects: {n_subs}")
print(f"Subjects per fold: {n_per}")

# Set random seed and shuffle subjects
np.random.seed(random_state)
shuffled_subjects = np.random.permutation(unique_subjects)

# Create output directory
output_dir = f'running_norm_{feature_type}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each fold
for fold in range(n_folds):
    print(f"\nProcessing fold {fold}")
    
    # Define validation (test) subjects for this fold
    if fold < n_folds - 1:
        val_subjects = shuffled_subjects[n_per * fold:n_per * (fold + 1)]
    else:
        val_subjects = shuffled_subjects[n_per * fold:]
    
    # Training subjects are all others
    train_subjects = np.setdiff1d(shuffled_subjects, val_subjects)
    
    print(f"Train subjects: {len(train_subjects)}, Test subjects: {len(val_subjects)}")
    
    # Create masks for train/test data based on subject IDs
    train_mask = np.isin(subject_ids, train_subjects)
    test_mask = np.isin(subject_ids, val_subjects)
    
    # Split features and labels
    X_train = features[train_mask]
    y_train = Y[train_mask]
    X_test = features[test_mask]
    y_test = Y[test_mask]
    
    # Get corresponding metadata
    subject_train = subject_ids[train_mask]
    video_train = video_ids[train_mask]
    subject_test = subject_ids[test_mask]
    video_test = video_ids[test_mask]
    
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Apply StandardScaler (fit on train, transform both train and test)
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    
    # Fit scaler only on training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using the fitted scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Combine all data (train + test) for this fold
    # Reconstruct the full dataset with normalized features
    all_features_normalized = np.zeros_like(features)
    all_features_normalized[train_mask] = X_train_scaled
    all_features_normalized[test_mask] = X_test_scaled
    
    # Combine with metadata to maintain original structure
    X_normalized = np.column_stack([subject_ids, video_ids, all_features_normalized])
    
    # Save in format compatible with SVM training
    # Save as 'data' to match expected format from running_norm.py
    normalized_data = {
        'data': X_normalized,
        'labels': Y,
        'train_subjects': train_subjects,
        'test_subjects': val_subjects,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }
    
    # Save file with same naming convention as running_norm.py
    feature_name = 'de'
    if feature_type == 'DE_PSD':
        feature_name = 'de_psd'
    elif feature_type == 'DE_PSD_H':
        feature_name = 'de_psd_h'

    save_file = os.path.join(output_dir, f'{feature_name}_fold{fold}.mat')
    print(f"Saving fold {fold} to: {save_file}")
    sio.savemat(save_file, normalized_data)

print(f"\n10-fold cross validation normalization completed!")
print(f"All files saved in: {output_dir}")
print("Files can be directly used for SVM training.")

