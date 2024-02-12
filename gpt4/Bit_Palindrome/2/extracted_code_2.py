def compute_y_from_x(x_list):
    # Initialize y with the first element set to 1
    y = [1] + x_list[1:]
    
    # Check the last three elements and process the pattern observed
    if x_list[-3:] in ([1, 0, 1], [1, 1, 1]):
        y[-3:] = [0, 0, 0]
    
    return y

# Define the first 5 rows of x and y
x_lists = [
    [1,0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,1,0,0],
    [1,1,1,1,1,1,0,1,0,0],
    [1,1,1,0,0,1,0,1,1,1],
    [1,1,1,0,1,1,1,0,1,0],
]
y_lists = [
    [1,0,0,0,0,0,0,1,0,0],
    [1,1,1,1,1,1,1,0,0,0],
    [1,1,1,1,1,1,0,0,0,0],
    [1,1,1,0,0,0,0,0,0,0],
    [1,1,1,0,0,0,1,0,0,0],
]

# Test the function with the provided lists
success = True
for i in range(5):
    computed_y = compute_y_from_x(x_lists[i])
    if computed_y != y_lists[i]:
        success = False
        print(f"Failure on row {i+1}")
        print(f"Computed y: {computed_y}")
        print(f"Expected y: {y_lists[i]}")
        break

if success:
    print("Success")