import numpy as np

# Define the lists of x and the corresponding lists of y
list_x = [
    [18, 21, 6, 4, 4, 11, 22, 20, 18, 7],
    [4, 2, 18, 24, 9, 22, 21, 1, 15, 6],
    [25, 14, 23, 1, 1, 5, 21, 25, 5, 25],
    [14, 12, 5, 1, 17, 24, 13, 20, 10, 8],
    [14, 18, 15, 19, 1, 1, 21, 2, 13, 5]
]

list_y_expected = [
    [0, 0, 0, 1, 2, 2, 2, 3, 3, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 2, 2, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Compute list y
list_y_computed = []
for x in list_x:
    min_val = min(x)
    max_val = max(x)
    n = len(set(x))
    y = np.floor((np.array(x) - min_val) / (max_val - min_val + 1) * (n - 1))
    list_y_computed.append(y.astype(int).tolist())

# Check if the computed list y matches the expected list y
success = all(np.array_equal(y1, y2) for y1, y2 in zip(list_y_computed, list_y_expected))

# Print the result
if success:
    print('Success')
else:
    print('Failure')

# Output the computed list y
print("Computed List Y:")
for y in list_y_computed:
    print(y)