# Define the lists of x values
list_of_x = [
    [2,7,6,4,6,5,0,4,0,3],
    [8,4,0,4,1,2,5,5,7,6],
    [9,6,3,1,9,3,1,9,7,9],
    [2,0,5,9,3,4,9,6,2,0],
    [6,2,7,9,7,3,3,4,3,7]
]

# Define the corresponding lists of y values
list_of_y = [
    [2,7,7,7,7,7,7,7,7,7],
    [8,8,8,8,8,8,8,8,8,8],
    [9,9,9,9,9,9,9,9,9,9],
    [2,2,5,9,9,9,9,9,9,9],
    [6,6,7,9,9,9,9,9,9,9]
]

# Function to compute y from x using the rule specified
def compute_y_from_x(x):
    # Initialize y with the first element of x
    y = [x[0]]
    # Iterate through the list x starting from the second element
    for i in range(1, len(x)):
        # y[i] is the max of y[i-1] and x[i]
        y.append(max(y[i-1], x[i]))
    return y

# Iterate through the first 5 rows
for i in range(5):
    # Compute y from x
    computed_y = compute_y_from_x(list_of_x[i])
    # Check if the computed y matches the given y
    if computed_y == list_of_y[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")

        # Optionally print the expected and actual to help debug
        print("Expected y:", list_of_y[i])
        print("Actual y:", computed_y)