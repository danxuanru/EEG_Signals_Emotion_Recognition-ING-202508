import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read accuracy data from CSV
accuracy_data = pd.read_csv('subject_cross_vids_24_valid_10-folds_kernel_rbf.csv', delimiter=',')
accuracy_data = accuracy_data.iloc[:, 1:]  # Skip the first column if it's an index
print(accuracy_data.head())

mean_accuracy = accuracy_data.mean(axis=0)
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
plt.title('Bar Plot of SVM Classification Accuracy')
# 使用 Patch 建立自定義圖例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', label='Accuracy'),
    Patch(facecolor='red', label=f'Mean Accuracy: {mean_accuracy[0]:.4f}')
]
plt.legend(handles=legend_elements)

# plt.legend(['Accuracy'])
# plt.legend(all_value.loc[-1], label=f'Mean Accuracy: {mean_accuracy:.2f}')

# order the x-axis labels


# # differentiate to the original data
# original_data = pd.read_csv('original.csv', delimiter=',')
# print(original_data)
# different = original_data['0'].values - accuracy_data.iloc[0:123, 0].values
# print(different)
# sub = list(original_data.index)
# plt.figure(figsize=(14, 6))
# plt.bar(sub, different, color=colors)
# plt.xlabel('All Subjects')
# plt.ylabel('Accuracy Difference')
# plt.title('Bar Plot of SVM Classification Accuracy Difference')
# plt.legend(['Accuracy Difference'])

# # Create the plot
# plt.figure(figsize=(12, 6))
# plt.boxplot(accuracy_data, whis=1.5)

# # Customize the plot
# plt.xlabel('All Subjects')
# plt.ylabel('Accuracy')
# plt.title('Box Plot of SVM Classification Accuracy')
# plt.grid(True, alpha=0.3)

# # Add mean accuracy line
# mean_accuracy = np.mean(accuracy_data)
# plt.axhline(y=mean_accuracy, color='r', linestyle='--', 
#             label=f'Mean Accuracy: {mean_accuracy:.2f}')
# plt.legend()

# Save the plot
plt.savefig('accuracy_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()
