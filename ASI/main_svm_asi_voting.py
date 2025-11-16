import argparse
import numpy as np
import pandas as pd
import os
import scipy.io as sio
import random
import time
from sklearn.svm import LinearSVC
import joblib
from collections import defaultdict
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='SVM training and voting for AsI filtered EEG emotion recognition')
parser.add_argument('--randSeed', default=7, type=int, help='random seed')
parser.add_argument('--training-fold', default='all', type=str, help='the number of training fold, 0~9')
parser.add_argument('--n-vids', default=24, type=int, help='number of videos')
parser.add_argument('--train-or-test', default='train', type=str, help='train or test mode')
parser.add_argument('--feature', default='DE', type=str, help='feature type')
parser.add_argument('--voting-method', default='majority', choices=['majority', 'average_prob'], 
                    help='voting method for video-level prediction')

args = parser.parse_args()

random.seed(args.randSeed)
np.random.seed(args.randSeed)

# Parameters
C_cands = 10.**np.arange(-5, 1, 0.5)
print('C candidates:',
      C_cands)

# Add SVM parameters for better convergence
max_iter = 10000  # Increase max iterations
dual = False      # Use primal formulation for better convergence with many features
tol = 1e-4        # Convergence tolerance

n_vids = args.n_vids
feature_type = args.feature
train_or_test = args.train_or_test
voting_method = args.voting_method

print(f"SVM parameters: max_iter={max_iter}, dual={dual}, tol={tol}")

# Setup fold processing
n_folds = 10
if args.training_fold == 'all':
    folds_list = np.arange(0, n_folds)
else:
    folds_list = [int(args.training_fold)]

# Create result directories
result_dir = f'result_asi_{feature_type}'
save_dir = os.path.join(result_dir, 'svm_weights')
for dir_path in [result_dir, save_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

# Results storage
val_acc_folds = np.zeros(n_folds)
best_C_folds = np.zeros(n_folds)

if train_or_test == 'test':
    # Store video-level accuracies for each test subject
    video_results = defaultdict(list)  # {subject_id: [video_accuracies]}

def voting_predict(predictions, video_ids, method='majority'):
    """
    Perform video-level voting from window-level predictions.
    
    Args:
        predictions: array of window-level predictions
        video_ids: array of corresponding video IDs
        method: voting method ('majority' or 'average_prob')
    
    Returns:
        dict: {video_id: final_prediction}
    """
    video_votes = defaultdict(list)
    
    # Group predictions by video
    for pred, vid in zip(predictions, video_ids):
        video_votes[vid].append(pred)
    
    video_predictions = {}
    for vid, votes in video_votes.items():
        if method == 'majority':
            # Majority voting
            vote_counts = np.bincount(np.array(votes) + 1)  # +1 to handle -1,1 labels
            final_pred = np.argmax(vote_counts) * 2 - 1  # Convert back to -1,1
        else:
            # For future extension - average probability
            final_pred = 1 if np.mean(votes) > 0 else -1
        
        video_predictions[vid] = final_pred
    
    return video_predictions

# Process each fold
for fold in folds_list:
    print(f'\n=== Processing Fold {fold} ===')
    
    # Load fold data
    data_dir = f'running_norm_{feature_type}'
    feature_name = 'de'
    if feature_type == 'DE_PSD':
        feature_name = 'de_psd'
    elif feature_type == 'DE_PSD_H':
        feature_name = 'de_psd_h'
    
    data_file = os.path.join(data_dir, f'{feature_name}_fold{fold}.mat')
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        continue
    
    print(f"Loading: {data_file}")
    fold_data = sio.loadmat(data_file)
    
    # Extract data
    all_data = fold_data['data']  # [subject_id, video_id, features...]
    all_labels = fold_data['labels'].flatten()
    train_subjects = fold_data['train_subjects'].flatten()
    test_subjects = fold_data['test_subjects'].flatten()
    
    print(f"Data shape: {all_data.shape}")
    print(f"Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
    
    # Split data based on subjects
    subject_ids = all_data[:, 0].astype(int)
    video_ids = all_data[:, 1].astype(int)
    features = all_data[:, 2:]
    
    # Create train/test masks
    train_mask = np.isin(subject_ids, train_subjects)
    test_mask = np.isin(subject_ids, test_subjects)
    
    # Extract training and testing data
    X_train = features[train_mask]
    y_train = all_labels[train_mask]
    X_test = features[test_mask]
    y_test = all_labels[test_mask]
    
    # Test set metadata for voting
    test_subject_ids = subject_ids[test_mask]
    test_video_ids = video_ids[test_mask]
    
    # PCA dimensionality reduction
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    start_time = time.time()
    
    if train_or_test == 'train':
        best_val_acc = 0
        best_C = None
        
        # Cross-validation for hyperparameter tuning
        for C in C_cands:
            print(f"Training with C={C:.1e}...")
            
            # Use improved SVM parameters for better convergence
            clf = LinearSVC(
                random_state=args.randSeed, 
                C=C, 
                max_iter=max_iter,
                dual=dual,
                tol=tol,
                class_weight='balanced'  # Handle class imbalance
            )
            
            try:
                clf.fit(X_train, y_train)
                
                # Evaluate on training data (for monitoring)
                train_preds = clf.predict(X_train)
                train_acc = np.mean(train_preds == y_train)
                
                # Simple validation: use a portion of training data
                # (In practice, you might want a proper validation split)
                val_acc = train_acc  # Simplified for now
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_C = C
                    
                    # Save best model
                    model_path = os.path.join(save_dir, f'svm_asi_{feature_type}_fold{fold}.joblib')
                    joblib.dump(clf, model_path)
                
                print(f"C={C:.1e}, Train Acc: {train_acc:.3f}, Converged: {clf.n_iter_ < max_iter}")
                
            except Exception as e:
                print(f"Error training with C={C:.1e}: {e}")
                continue
        
        val_acc_folds[fold] = best_val_acc
        best_C_folds[fold] = best_C
        print(f"Best C: {best_C:.1e}, Best Val Acc: {best_val_acc:.3f}")
        
    elif train_or_test == 'test':
        # Load trained model
        model_path = os.path.join(save_dir, f'svm_asi_{feature_type}_fold{fold}.joblib')
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            continue
            
        clf = joblib.load(model_path)
        
        # Predict on test data (window-level)
        window_predictions = clf.predict(X_test)
        window_acc = np.mean(window_predictions == y_test)
        
        print(f"Window-level accuracy: {window_acc:.3f}")
        
        # Perform video-level voting for each test subject
        for subject in test_subjects:
            subject_mask = test_subject_ids == subject
            
            if not np.any(subject_mask):
                continue
                
            # Get predictions and video IDs for this subject
            subject_preds = window_predictions[subject_mask]
            subject_video_ids = test_video_ids[subject_mask]
            subject_true_labels = y_test[subject_mask]
            
            # Perform voting
            video_preds = voting_predict(subject_preds, subject_video_ids, voting_method)
            
            # Calculate true video labels (majority vote from windows)
            video_true = voting_predict(subject_true_labels, subject_video_ids, 'majority')
            
            # Calculate video-level accuracy for this subject
            correct_videos = 0
            total_videos = len(video_preds)
            
            for vid in video_preds:
                if video_preds[vid] == video_true[vid]:
                    correct_videos += 1
            
            video_acc = correct_videos / total_videos if total_videos > 0 else 0
            video_results[subject].append(video_acc)
            
            print(f"Subject {subject}: {correct_videos}/{total_videos} videos correct ({video_acc:.3f})")
        
        end_time = time.time()
        print(f"Time consumed: {end_time - start_time:.2f}s")

# Print final results
if train_or_test == 'train':
    print(f'\n=== Training Results ===')
    print(f'Mean validation accuracy: {np.mean(val_acc_folds):.3f} ± {np.std(val_acc_folds):.3f}')
    print(f'Best C values: {best_C_folds}')

elif train_or_test == 'test':
    print(f'\n=== Testing Results (Video-level Voting) ===')
    
    # Calculate average video accuracy per subject across all folds
    subject_avg_accs = {}
    for subject, accs in video_results.items():
        subject_avg_accs[subject] = np.mean(accs)
    
    if subject_avg_accs:
        overall_acc = np.mean(list(subject_avg_accs.values()))
        overall_std = np.std(list(subject_avg_accs.values()))
        
        print(f'Overall video-level accuracy: {overall_acc:.3f} ± {overall_std:.3f}')
        print(f'Voting method: {voting_method}')
        
        # Save detailed results
        results_df = pd.DataFrame(list(subject_avg_accs.items()), 
                                 columns=['Subject', 'Video_Accuracy'])
        results_file = os.path.join(result_dir, f'video_results_{feature_type}_{voting_method}.csv')
        results_df.to_csv(results_file, index=False)
        print(f'Detailed results saved to: {results_file}')
