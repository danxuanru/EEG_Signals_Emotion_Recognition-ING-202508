import argparse
import os
import sys
import scipy.io as sio

def check_file_exists(filepath, step_name):
    if not os.path.exists(filepath):
        print(f"Error: Required file {filepath} not found for {step_name}")
        sys.exit(1)

parser = argparse.ArgumentParser(description='Run all steps for EEG feature extraction and classification')
parser.add_argument('--session-length', default=30, type=int, help='Length of each video session in seconds')
parser.add_argument('--feature', default='DE', choices=['DE', 'DE_PSD', 'DE_PSD_H'], help='Feature combination: DE, DE_PSD, or DE_PSD_H')
args = parser.parse_args()
session_length = args.session_length
feature = args.feature

# None: Run all bands
print("Running all bands!")
os.system(f"python run_all.py --session-length {session_length} --feature {feature}")

# Delta band
print("Running Delta band!")
os.system(f"python run_all.py --session-length {session_length} --remove-band 1 --feature {feature}")

# Theta band
print("Running Theta band!")
os.system(f"python run_all.py --session-length {session_length} --remove-band 2 --feature {feature}")

# Alpha band
print("Running Alpha band!")
os.system(f"python run_all.py --session-length {session_length} --remove-band 3 --feature {feature}")

# Beta band
print("Running Beta band!")
os.system(f"python run_all.py --session-length {session_length} --remove-band 4 --feature {feature}")

# Gamma band
print("Running Gamma band!")
os.system(f"python run_all.py --session-length {session_length} --remove-band 5 --feature {feature}")

