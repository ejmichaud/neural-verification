# Define the function to compute y using the provided rule.
def compute_y(x):
    y = [0] * len(x)  # create a list of zeros with the same length as x
    for i in range(len(x) - 1):  # iterate through x except the last element
        if x[i] == 1 and x[i+1] == 0:  # check the rule for each position
            y[i] = 1
    if x[-1] == 1:  # check the rule for the last element
        y[-1] = 1
    return y

# List of x and y pairs to be tested
data = [
    ([0,1,0,0,0,1,0,0,0,1], [0,1,0,0,0,0,0,0,0,0]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,0,0,0,0,0,0]),
    ([1,0,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0,0]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,0,1,0,0,0,0,0,0]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,0,1,0,0,0,0,0,0]),
]

# Check if the computed y matches the given y and print the result.
for i, (x, expected_y) in enumerate(data):
    computed_y = compute_y(x)
    if computed_y == expected_y:
        print(f"Row {i+1} - Success")
    else:
        print(f"Row {i+1} - Failure")