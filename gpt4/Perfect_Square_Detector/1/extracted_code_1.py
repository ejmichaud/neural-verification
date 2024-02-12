def compute_y(x_list):
    # An example heuristic: return 1 if the number is odd, 0 if it's even.
    # This is just an example and may not match the actual pattern of y.
    return [x % 2 for x in x_list]

# Input data
x_data = [
    [2,7,16,14,6,15,0,4,10,13],
    [18,14,10,14,11,12,15,15,7,6],
    [9,16,13,11,19,13,1,9,7,19],
    [12,0,15,9,13,4,19,16,12,10],
    [6,2,17,19,7,3,3,14,13,17]
]

y_data = [
    [0,0,1,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,1,1,0,0],
    [0,1,0,1,0,1,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0]
]

# Check if the computed y matches the given y
for i, x_list in enumerate(x_data):
    computed_y = compute_y(x_list)
    if computed_y == y_data[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")