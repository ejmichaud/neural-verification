# Define the lists of x and y
x_lists = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

y_lists = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

# Function to compute y from x
def compute_y(x_list):
    return x_list

# Check if the computed y matches the provided y
for i in range(5):
    computed_y = compute_y(x_lists[i])
    if computed_y == y_lists[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")

# Or simply:
# for i in range(5):
#     if x_lists[i] == y_lists[i]:
#         print(f"Row {i+1} Success")
#     else:
#         print(f"Row {i+1} Failure")