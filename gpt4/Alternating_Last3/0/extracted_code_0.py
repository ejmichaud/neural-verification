def transform_x_to_y(x):
    y = [0] * len(x)  # Initialize y with zeros
    
    for i in range(1, len(x) - 1):
        # Check if there is a '1' followed by at least one '0' before and after
        if x[i] == 1 and x[i - 1] == 0:
            y[i] = 1
    
    return y

# Define the first five rows
input_lists_x = [
    [0,0,0,1,0,0,0,1,0,0],
    [1,0,1,1,1,0,1,0,1,1],
    [1,1,1,1,0,0,1,1,1,0],
    [0,0,0,0,1,1,1,1,1,0],
    [0,1,0,1,0,1,1,0,0,0]
]

output_lists_y = [
    [0,0,0,0,1,0,0,0,1,0],
    [0,1,1,0,0,0,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,0,0,0,0]
]

# Check each list
for i in range(len(input_lists_x)):
    x = input_lists_x[i]
    y = output_lists_y[i]
    computed_y = transform_x_to_y(x)
    
    if computed_y == y:
        print(f"Success for row {i+1}")
    else:
        print(f"Failure for row {i+1}: expected {y}, got {computed_y}")