import joblib
import os
import numpy as np

# Directory where your 10-second model weights are stored
previous_weights_dir = './svm_weights'  # Adjust path as needed
n_folds = 10

# Arrays to store parameters
best_C_values = []
best_gamma_values = []

# Load parameters from each fold
for fold in range(n_folds):
    model_path = os.path.join(previous_weights_dir, 
                              f'subject_cross_vids_24_fold_{fold}_valid_10-folds.joblib')
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        best_C_values.append(np.log10(model.C))
        
        # For RBF kernel models, extract gamma
        if hasattr(model, 'gamma'):
            best_gamma_values.append(np.log10(model.gamma))

print("Best C values from each fold:", best_C_values)
if best_gamma_values:
    print("Best gamma values from each fold:", best_gamma_values)
