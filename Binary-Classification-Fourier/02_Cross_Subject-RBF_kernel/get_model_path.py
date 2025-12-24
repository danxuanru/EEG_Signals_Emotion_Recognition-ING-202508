import os
import joblib

def get_model_path(subjects_type, n_vids, fold, valid_method, kernel_type):
    model_path = os.path.join('./svm_weights', f'subject_{subjects_type}_vids_{n_vids}_fold_{fold}_valid_{valid_method}_kernel_{kernel_type}.joblib')
    clf2 = joblib.load(model_path)
    return clf2

if __name__ == '__main__':
    # Example usage
    subjects_type = 'cross'
    n_vids = 24
    fold = 10
    valid_method = 'valid'
    kernel_type = 'rbf'
    
    model = get_model_path(subjects_type, n_vids, fold, valid_method, kernel_type)
    print("Model loaded successfully:", model)