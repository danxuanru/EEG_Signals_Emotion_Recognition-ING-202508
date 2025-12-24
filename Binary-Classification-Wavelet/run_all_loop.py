import argparse
import os
import sys
import scipy.io as sio

def check_file_exists(filepath, step_name):
    if not os.path.exists(filepath):
        print(f"Error: Required file {filepath} not found for {step_name}")
        sys.exit(1)

def delete_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Deleted file: {filepath}")

parser = argparse.ArgumentParser(description='Run all steps for EEG feature extraction and classification')
parser.add_argument('--session-length', default=30, type=int, help='Length of each video session in seconds')
parser.add_argument('--feature', default='DE', choices=['DE', 'DE_PSD', 'DE_PSD_H'], help='Feature combination: DE, DE_PSD, or DE_PSD_H')
args = parser.parse_args()
session_length = args.session_length
feature = args.feature

# define the file/folder names to be deleted after each step
del_file = ["data.mat", "data_binary.mat", "vid_order_binary.mat", "running_norm_24", "running_norm_28", "smooth_24", "smooth_28"]

# None: Run all bands
print("Running all bands!", flush=True)
os.system(f"python run_all.py --session-length {session_length} --feature {feature}")

# check and delete intermediate files to avoid interference
target_dictory = "result/00_None"
check_file_exists(target_dictory, "None band")
for file in del_file:
    delete_file(file)


# Delta band
print("Running Delta band!", flush=True)
os.system(f"python run_all.py --session-length {session_length} --remove-band 1 --feature {feature}")

target_dictory = "result/01_Delta"
check_file_exists(target_dictory, "Delta band")
for file in del_file:
    delete_file(file)


# Theta band
print("Running Theta band!", flush=True)
os.system(f"python run_all.py --session-length {session_length} --remove-band 2 --feature {feature}")

target_dictory = "result/02_Theta"
check_file_exists(target_dictory, "Theta band")
for file in del_file:
    delete_file(file)


# Alpha band
print("Running Alpha band!", flush=True)
os.system(f"python run_all.py --session-length {session_length} --remove-band 3 --feature {feature}")

target_dictory = "result/00_Alpha"
check_file_exists(target_dictory, "Alpha band")
for file in del_file:
    delete_file(file)


# Beta band
print("Running Beta band!", flush=True)
os.system(f"python run_all.py --session-length {session_length} --remove-band 4 --feature {feature}")

target_dictory = "result/00_Beta"
check_file_exists(target_dictory, "Beta band")
for file in del_file:
    delete_file(file)


# Gamma band
print("Running Gamma band!", flush=True)
os.system(f"python run_all.py --session-length {session_length} --remove-band 5 --feature {feature}")

target_dictory = "result/00_Gamma"
check_file_exists(target_dictory, "Gamma band")
for file in del_file:
    delete_file(file)