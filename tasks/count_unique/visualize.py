import torch
import matplotlib.pyplot as plt

# Load the file
directory = "2"
data = torch.load(f"results/{directory}/metrics.pt")

# Make sure the loaded data contains 'train_losses'
if 'train_losses' in data:
    train_losses = data['train_losses']
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Index')
    plt.ylabel('Train Loss')
    plt.title('Train Losses over Time')
    plt.grid(True)
    plt.savefig('train_losses.png')
else:
    print("'train_losses' not found in the loaded data.")

if 'test_accuracies' in data:
    test_accuracies = data['test_accuracies']
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(test_accuracies)
    plt.xlabel('Index')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy over Time')
    plt.grid(True)
    plt.savefig('test_accuracy.png')
else:
    print("'test_accuracy' not found in the loaded data.")