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
window_size = 500

# Compute moving averages
train_losses_smooth = moving_average(metrics['train_losses'], window_size)
test_losses_smooth = moving_average(metrics['test_losses'], window_size)
train_accuracies_smooth = moving_average(metrics['train_accuracies'], window_size)
test_accuracies_smooth = moving_average(metrics['test_accuracies'], window_size)

# Interpolate test data to have the same length as train data
x_new = np.linspace(0, len(test_losses_smooth)-1, len(train_losses_smooth))
test_losses_interp = np.interp(x_new, np.arange(len(test_losses_smooth)), test_losses_smooth)
test_accuracies_interp = np.interp(x_new, np.arange(len(test_accuracies_smooth)), test_accuracies_smooth)

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot losses
axes[0].set_title('Losses')
axes[0].set_ylim(0, 0.05)
axes[0].set_ylabel('Loss (Weighted BCE))')
axes[0].set_xlabel('Steps')
axes[0].plot(train_losses_smooth, label='Train Loss')
axes[0].plot(test_losses_interp, label='Test Loss')
axes[0].legend()

# Plot accuracies
axes[1].set_title('Accuracies')
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('Accuracy')
axes[1].set_xlabel('Steps')
axes[1].plot(train_accuracies_smooth, label='Train Accuracy')
axes[1].plot(test_accuracies_interp, label='Test Accuracy')
axes[1].legend()

plt.show()