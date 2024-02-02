# Function to compute y from x
def compute_y_from_x(x):
    return [0 if i < 3 else x[i-3] for i in range(len(x))]

# Input lists
x_lists = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

# Expected y lists
y_lists = [
    [0, 0, 0, 42, 67, 76, 14, 26, 35, 20],
    [0, 0, 0, 78, 14, 10, 54, 31, 72, 15],
    [0, 0, 0, 49, 76, 73, 11, 99, 13, 41],
    [0, 0, 0, 72, 80, 75, 29, 33, 64, 39],
    [0, 0, 0, 86, 22, 77, 19, 7, 23, 43]
]

# Loop through the first 5 rows
for idx, x in enumerate(x_lists):
    y_computed = compute_y_from_x(x)
    if y_computed == y_lists[idx]:
        print(f"Row {idx+1}: Success")
    else:
        print(f"Row {idx+1}: Failure")