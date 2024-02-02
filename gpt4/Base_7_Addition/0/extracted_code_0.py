list_x = [
    [1, 2], [4, 5], [4, 3], [5, 5], [0, 0], # Row 1
    [1, 0], [0, 3], [3, 5], [1, 1], [2, 2], # Row 2
    [0, 1], [5, 5], [6, 4], [0, 2], [5, 5], # Row 3
    [5, 0], [2, 5], [5, 2], [4, 2], [6, 0], # Row 4
    [0, 1], [6, 3], [4, 2], [6, 5], [4, 5], # Row 5
]
list_y = [
    3, 2, 1, 4, 1, # Row 1
    1, 3, 1, 3, 4, # Row 2
    1, 3, 4, 3, 3, # Row 3
    5, 0, 1, 0, 0, # Row 4
    1, 2, 0, 5, 3, # Row 5
]

def compute_y(x):
    # Placeholder for the actual function to calculate y from x
    return None

success = True
for i, x_pair in enumerate(list_x):
    y_computed = compute_y(x_pair)
    if y_computed != list_y[i]:
        success = False
        break

if success:
    print("Success")
else:
    print("Failure")