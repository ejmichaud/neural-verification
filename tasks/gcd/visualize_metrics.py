"""
PREREQUISITE: Run tasks/gcd/train.py to generate metrics.pt
"""


import matplotlib.pyplot as plt
import numpy as np
import torch

# Load your metrics.pt
metrics = torch.load('metrics.pt')

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Smoothing window size
window_size = 2000

# Compute moving averages
train_losses_smooth = moving_average(metrics['train_losses'], window_size)
test_losses_smooth = moving_average(metrics['test_losses'], window_size)
train_accuracies_smooth = moving_average(metrics['train_accuracies'], window_size)
test_accuracies_smooth = moving_average(metrics['test_accuracies'], window_size)

# Compute x-axis positions for test data
test_indices = np.arange(0, len(test_losses_smooth) * 5, 5)

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot losses
axes[0].set_title('Losses')
axes[0].set_yscale('log')
# label y axis
axes[0].set_ylabel('Loss (weighted BCE)')
axes[0].plot(train_losses_smooth, label='Train Loss')
axes[0].plot(test_indices, test_losses_smooth, label='Test Loss')
axes[0].legend()

# Plot accuracies
axes[1].set_title('Accuracies')
axes[1].set_ylim(0.93, 1)  # Setting y-axis limits
axes[1].set_ylabel('Accuracy')
axes[1].plot(train_accuracies_smooth, label='Train Accuracy')
axes[1].plot(test_indices, test_accuracies_smooth, label='Test Accuracy')
axes[1].legend()

plt.show()
