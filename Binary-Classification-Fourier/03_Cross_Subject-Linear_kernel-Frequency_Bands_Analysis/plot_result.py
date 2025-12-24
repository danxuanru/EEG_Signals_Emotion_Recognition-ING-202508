import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os  # Import os module

# Add command line arguments
parser = argparse.ArgumentParser(description='Plot SVM classification accuracy results')
parser.add_argument('--subject-type', default='cross', type=str,
                    help='Subject type (cross or intra)')
parser.add_argument('--n-vids', default=28, type=int,
                    help='Number of videos (28 for 9-class, 24 for binary)')
parser.add_argument('--kernel', default='', type=str,
                    help='Kernel type (rbf, linear, etc.)')
parser.add_argument('--session-length', default=30, type=int,
                    help='Length of each video session in seconds')
parser.add_argument('--remove-band', default=None, type=int, help='Remove specific frequency band (1-5) or None to include all bands')
args = parser.parse_args()

# Add frequency band mapping
band_names = {0: '0_None', 1: '1_Delta', 2: '2_Theta', 3: '3_Alpha', 4: '4_Beta', 5: '5_Gamma'}
if args.remove_band is not None and args.remove_band in band_names:
    result_subdir = os.path.join('result', band_names[args.remove_band])
else:
    result_subdir = 'result'

# Construct the filename dynamically
filename = f'subject_{args.subject_type}_vids_{args.n_vids}_valid_10-folds'
if args.kernel == 'rbf':
    filename += f'_kernel_rbf'
# if args.session_length != 30:  # Only add session length to filename if it's not the default
#     filename += f'_sec_{args.session_length}'
filename += '.csv'

# Update file path to include result subdirectory
file_path = os.path.join(result_subdir, filename)
print(f"Loading results from: {file_path}")

# Read accuracy data from CSV
accuracy_data = pd.read_csv(file_path, delimiter=',')
accuracy_data = accuracy_data.iloc[:, 1:]  # Skip the first column if it's an index
print(accuracy_data.head())

# Calculate mean and standard deviation
mean_accuracy = accuracy_data.mean(axis=0)
std_accuracy = accuracy_data.iloc[:123].std(axis=0)
print(f"Mean Accuracy: {mean_accuracy[0]:.4f}")
print(f"Standard Deviation: {std_accuracy[0]:.4f}")

accuracy_data.index = accuracy_data.index + 1  # Shift index
accuracy_data.loc[124] = mean_accuracy  # Add mean accuracy to the last row
print(accuracy_data['0'].values)
print(list(accuracy_data.index))

all_subjects = list(accuracy_data.index)
all_value = accuracy_data['0'].values 
colors = ['lightblue'] * 123 + ['red']

# plot bar chart
plt.figure(figsize=(14, 6))
plt.bar(all_subjects, all_value, color=colors)
plt.xlabel('All Subjects')
plt.ylabel('Accuracy')
plt.title(f'Bar Plot of SVM Classification Accuracy ({args.subject_type}, {args.n_vids} videos)')

# Add text annotation for std directly on the graph
plt.axhline(y=mean_accuracy[0], color='black', linestyle='--', alpha=0.5)
plt.axhline(y=mean_accuracy[0]+std_accuracy[0], color='black', linestyle=':', alpha=0.3)
plt.axhline(y=mean_accuracy[0]-std_accuracy[0], color='black', linestyle=':', alpha=0.3)

# Add std dev annotation text box in upper right
plt.annotate(f'σ = {std_accuracy[0]:.4f}', 
             xy=(0.95, 0.95), 
             xycoords='axes fraction',
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
             ha='right', va='top')

# 使用 Patch 建立自定義圖例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', label='Accuracy'),
    Patch(facecolor='red', label=f'Mean Accuracy: {mean_accuracy[0]:.4f}'),
    Patch(facecolor='white', ec='black', alpha=0.5, hatch='--', label=f'Std Dev: ±{std_accuracy[0]:.4f}')
]
plt.legend(handles=legend_elements, loc='upper left')

# Save the plot
output_filename = f'accuracy_{args.subject_type}_{args.n_vids}'
if args.kernel:
    output_filename += f'_{args.kernel}'
if args.session_length != 30:  # Only add session length to filename if it's not the default
    output_filename += f'_sec_{args.session_length}'
output_filename += '.png'

# Update output path to include result subdirectory
output_path = os.path.join(result_subdir, output_filename)
# Ensure result subdirectory exists
if not os.path.exists(result_subdir):
    os.makedirs(result_subdir, exist_ok=True)
    
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as {output_path}")
plt.close()
