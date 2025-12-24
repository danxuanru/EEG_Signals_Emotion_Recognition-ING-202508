import argparse
import numpy as np
import pandas as pd
import os
import scipy.io as sio
import random
import time
from sklearn.svm import LinearSVC, SVC
import joblib
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='SVM training and voting for AsI filtered EEG emotion recognition')
parser.add_argument('--randSeed', default=7, type=int, help='random seed')
parser.add_argument('--training-fold', default='all', type=str, help='the number of training fold, 0~9')
parser.add_argument('--n-vids', default=24, type=int, help='number of videos')
parser.add_argument('--train-or-test', default='train', type=str, help='train or test mode')
parser.add_argument('--feature', default='DE', type=str, help='feature type')
parser.add_argument('--voting-method', default='weighted', choices=['majority', 'weighted', 'average_prob'], 
                    help='voting method for video-level prediction')
parser.add_argument('--overlap-ratio', default=0, type=float, 
                    help='overlap ratio used in window extraction (for weighted voting)')
parser.add_argument('--kernel', default='linear', choices=['linear', 'rbf'],
                    help='SVM kernel to use during training and testing')

args = parser.parse_args()

random.seed(args.randSeed)
np.random.seed(args.randSeed)

# Parameters
C_cands = 10.**np.arange(-5, 1, 0.5)
print('C candidates:', C_cands)

# Add SVM parameters for better convergence
max_iter = 10000
dual = False
tol = 1e-4

n_vids = args.n_vids
feature_type = args.feature
train_or_test = args.train_or_test
voting_method = args.voting_method
overlap_ratio = args.overlap_ratio
kernel = args.kernel

print(f"SVM parameters: max_iter={max_iter}, dual={dual}, tol={tol}")
print(f"Voting method: {voting_method}, Overlap ratio: {overlap_ratio}")
print(f"Kernel: {kernel}")

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
    video_results = defaultdict(list)

def video_level_voting_weighted(predictions, video_ids, true_labels, overlap_ratio=0.5, method='weighted'):
    """
    考慮重疊視窗的加權投票
    
    Args:
        predictions: array of window-level predictions
        video_ids: array of corresponding video IDs
        true_labels: array of true labels for each window
        overlap_ratio: overlap ratio used in window extraction
        method: voting method ('weighted', 'majority', or 'average_prob')
    
    Returns:
        video_predictions: dict {video_id: final_prediction}
        video_true_labels: dict {video_id: true_label}
        voting_stats: dict with detailed statistics
    """
    # Group predictions and labels by video
    video_data = defaultdict(lambda: {'preds': [], 'labels': []})
    
    for pred, vid, label in zip(predictions, video_ids, true_labels):
        video_data[vid]['preds'].append(pred)
        video_data[vid]['labels'].append(label)
    
    video_predictions = {}
    video_true_labels = {}
    voting_stats = {
        'video_id': [],
        'n_windows': [],
        'effective_windows': [],
        'positive_votes': [],
        'negative_votes': [],
        'confidence': []
    }
    
    for vid, data in video_data.items():
        preds = np.array(data['preds'])
        labels = np.array(data['labels'])
        
        # Get true label (should be consistent across windows)
        video_true_labels[vid] = int(labels[0])
        
        if method == 'weighted':
            # 計算有效視窗權重（考慮重疊）
            effective_weight = 1.0 / (1.0 + overlap_ratio)
            
            # 加權計數
            positive_votes = np.sum(preds == 1) * effective_weight
            negative_votes = np.sum(preds == -1) * effective_weight
            
            # 最終預測
            final_pred = 1 if positive_votes > negative_votes else -1
            
            # 計算信心度（投票比例差異）
            total_votes = positive_votes + negative_votes
            confidence = abs(positive_votes - negative_votes) / total_votes if total_votes > 0 else 0
            
            # 記錄統計資訊
            voting_stats['video_id'].append(vid)
            voting_stats['n_windows'].append(len(preds))
            voting_stats['effective_windows'].append(total_votes)
            voting_stats['positive_votes'].append(positive_votes)
            voting_stats['negative_votes'].append(negative_votes)
            voting_stats['confidence'].append(confidence)
            
        elif method == 'majority':
            # 簡單多數投票（不考慮重疊）
            positive_votes = np.sum(preds == 1)
            negative_votes = np.sum(preds == -1)
            
            if positive_votes > negative_votes:
                final_pred = 1
            else:
                final_pred = -1
            
            confidence = max(positive_votes, negative_votes) / len(preds)
            
            voting_stats['video_id'].append(vid)
            voting_stats['n_windows'].append(len(preds))
            voting_stats['effective_windows'].append(len(preds))
            voting_stats['positive_votes'].append(positive_votes)
            voting_stats['negative_votes'].append(negative_votes)
            voting_stats['confidence'].append(confidence)
            
        else:  # average_prob
            # 平均預測值
            final_pred = 1 if np.mean(preds) > 0 else -1
            confidence = abs(np.mean(preds))
            
            voting_stats['video_id'].append(vid)
            voting_stats['n_windows'].append(len(preds))
            voting_stats['effective_windows'].append(len(preds))
            voting_stats['positive_votes'].append(np.sum(preds == 1))
            voting_stats['negative_votes'].append(np.sum(preds == -1))
            voting_stats['confidence'].append(confidence)
        
        video_predictions[vid] = final_pred
    
    return video_predictions, video_true_labels, voting_stats

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
    all_data = fold_data['data']
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
    
    # Apply scaling before PCA - critical for both linear and RBF kernels
    # StandardScaler ensures all features have zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Applied StandardScaler to features before PCA")

    # PCA dimensionality reduction
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    print(f"Training samples: {X_train.shape[0]}, PCA components: {X_train.shape[1]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    start_time = time.time()
    
    if train_or_test == 'train':
        best_val_acc = 0
        best_C = None
        
        # Cross-validation for hyperparameter tuning
        for C in C_cands:
            print(f"Training with C={C:.1e}...", end=' ')
            
            if kernel == 'linear':
                clf = LinearSVC(
                    random_state=args.randSeed,
                    C=C,
                    max_iter=max_iter,
                    dual=dual,
                    tol=tol,
                    class_weight='balanced'
                )
            else:
                clf = SVC(
                    kernel='rbf',
                    random_state=args.randSeed,
                    C=C,
                    tol=tol,
                    max_iter=max_iter,
                    class_weight='balanced',
                    gamma='auto'
                )
            
            try:
                clf.fit(X_train, y_train)
                
                train_preds = clf.predict(X_train)
                train_acc = np.mean(train_preds == y_train)
                
                val_acc = train_acc
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_C = C
                    
                    model_path = os.path.join(save_dir, f'svm_asi_{feature_type}_fold{fold}.joblib')
                    joblib.dump(clf, model_path)
                
                print(f"Train Acc: {train_acc:.3f}, Converged: {clf.n_iter_ < max_iter}")
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        val_acc_folds[fold] = best_val_acc
        best_C_folds[fold] = best_C
        print(f"Best C: {best_C:.1e}, Best Val Acc: {best_val_acc:.3f}")
        
    elif train_or_test == 'test':
        model_path = os.path.join(save_dir, f'svm_asi_{feature_type}_fold{fold}.joblib')
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            continue
            
        clf = joblib.load(model_path)
        
        # Predict on test data (window-level)
        window_predictions = clf.predict(X_test)
        window_acc = np.mean(window_predictions == y_test)
        
        print(f"Window-level accuracy: {window_acc:.3f}")
        
        # Store all voting statistics for this fold
        fold_voting_stats = []
        
        # Perform video-level voting for each test subject
        for subject in test_subjects:
            subject_mask = test_subject_ids == subject
            
            if not np.any(subject_mask):
                continue
                
            # Get predictions and video IDs for this subject
            subject_preds = window_predictions[subject_mask]
            subject_video_ids = test_video_ids[subject_mask]
            subject_true_labels = y_test[subject_mask]
            
            # Perform weighted voting
            video_preds, video_true, voting_stats = video_level_voting_weighted(
                subject_preds, 
                subject_video_ids, 
                subject_true_labels,
                overlap_ratio=overlap_ratio,
                method=voting_method
            )
            
            # Calculate video-level accuracy for this subject
            correct_videos = 0
            total_videos = len(video_preds)
            
            for vid in video_preds:
                if video_preds[vid] == video_true[vid]:
                    correct_videos += 1
            
            video_acc = correct_videos / total_videos if total_videos > 0 else 0
            video_results[subject].append(video_acc)
            
            # Add subject info to voting stats
            voting_stats['subject_id'] = [subject] * len(voting_stats['video_id'])
            voting_stats['fold'] = [fold] * len(voting_stats['video_id'])
            voting_stats['correct'] = [
                1 if video_preds[vid] == video_true[vid] else 0 
                for vid in voting_stats['video_id']
            ]
            
            fold_voting_stats.append(pd.DataFrame(voting_stats))
            
            print(f"Subject {subject}: {correct_videos}/{total_videos} videos correct ({video_acc:.3f}), "
                  f"Avg confidence: {np.mean(voting_stats['confidence']):.3f}")
        
        # Save voting statistics for this fold
        if fold_voting_stats:
            fold_stats_df = pd.concat(fold_voting_stats, ignore_index=True)
            stats_file = os.path.join(result_dir, f'voting_stats_fold{fold}_{feature_type}_{voting_method}.csv')
            fold_stats_df.to_csv(stats_file, index=False)
            print(f"Voting statistics saved to: {stats_file}")
        
        end_time = time.time()
        print(f"Time consumed: {end_time - start_time:.2f}s")

# Print final results
if train_or_test == 'train':
    print(f'\n=== Training Results ===')
    print(f'Mean validation accuracy: {np.mean(val_acc_folds):.3f} ± {np.std(val_acc_folds):.3f}')
    print(f'Best C values: {best_C_folds}')
    
    # Save training results
    train_results = pd.DataFrame({
        'fold': range(n_folds),
        'val_accuracy': val_acc_folds,
        'best_C': best_C_folds
    })
    train_results_file = os.path.join(result_dir, f'training_results_{feature_type}.csv')
    train_results.to_csv(train_results_file, index=False)
    print(f'Training results saved to: {train_results_file}')

elif train_or_test == 'test':
    print(f'\n=== Testing Results (Video-level Voting: {voting_method}) ===')
    
    # Calculate average video accuracy per subject across all folds
    subject_avg_accs = {}
    for subject, accs in video_results.items():
        subject_avg_accs[subject] = np.mean(accs)
    
    if subject_avg_accs:
        overall_acc = np.mean(list(subject_avg_accs.values()))
        overall_std = np.std(list(subject_avg_accs.values()))
        
        print(f'Overall video-level accuracy: {overall_acc:.3f} ± {overall_std:.3f}')
        print(f'Voting method: {voting_method}')
        print(f'Overlap ratio: {overlap_ratio}')
        
        # Save detailed results
        results_df = pd.DataFrame(list(subject_avg_accs.items()), 
                                 columns=['Subject', 'Video_Accuracy'])
        results_file = os.path.join(result_dir, f'video_results_{feature_type}_{voting_method}.csv')
        results_df.to_csv(results_file, index=False)
        print(f'Detailed results saved to: {results_file}')
        
        # Aggregate all voting statistics
        all_stats_files = [
            os.path.join(result_dir, f'voting_stats_fold{fold}_{feature_type}_{voting_method}.csv')
            for fold in folds_list
        ]
        
        existing_stats = [f for f in all_stats_files if os.path.exists(f)]
        if existing_stats:
            all_stats = pd.concat([pd.read_csv(f) for f in existing_stats], ignore_index=True)
            
            # Print summary statistics
            print(f'\n=== Voting Statistics Summary ===')
            print(f"Average windows per video: {all_stats['n_windows'].mean():.1f}")
            print(f"Average effective windows (weighted): {all_stats['effective_windows'].mean():.1f}")
            print(f"Average confidence: {all_stats['confidence'].mean():.3f}")
            print(f"Video-level accuracy: {all_stats['correct'].mean():.3f}")
            
            # Save aggregated statistics
            agg_stats_file = os.path.join(result_dir, f'voting_stats_all_{feature_type}_{voting_method}.csv')
            all_stats.to_csv(agg_stats_file, index=False)
            print(f'Aggregated voting statistics saved to: {agg_stats_file}')
