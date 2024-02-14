def compute_y_from_x(x):
    y = []
    for i in range(len(x)):
        # Check for the rule at each position, with bounds checking
        if i > 0 and i < len(x)-1 and x[i-1] == 1 and x[i] == 0 and x[i+1] == 1:
            y.append(1)
        else:
            y.append(0)
    return y

# Test the function with the provided dataset
x_lists = [
    [0,0,1,0,0,0,1,0,0,0],
    [1,1,1,0,1,0,1,1,1,1],
    [1,0,0,1,1,1,0,1,0,0],
    [1,1,1,1,1,0,1,1,0,1],
    [1,1,0,0,0,0,0,0,0,0],
]

y_lists = [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
]

# Function to check if two lists are equal
def lists_are_equal(list1, list2):
    return list1 == list2

# Iterate over the dataset and test our function
for i in range(len(x_lists)):
    computed_y = compute_y_from_x(x_lists[i])
    if lists_are_equal(computed_y, y_lists[i]):
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")