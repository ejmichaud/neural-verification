def compute_y(x):
    # If the input list x is all zeros, return a list of ones.
    if all(v == 0 for v in x):
        return [1] * len(x)
    
    y = x.copy() # Start with a copy of x to preserve the length and leading 1s

    # Reset trailing numbers to 0 after the first 0 is encountered
    first_zero_found = False
    for i in range(len(x)):
        if first_zero_found:
            y[i] = 0
        elif x[i] == 0:
            first_zero_found = True

    # Apply overriding rules
    # If x[7] is 1, then y[7] to y[9] are set to 0.
    if x[7] == 1:
        for i in range(7, len(x)):
            y[i] = 0

    # Override the last value of y to be 0
    y[-1] = 0

    return y


# Test the function with the first five rows provided

# Define the test cases (first five rows)
test_cases_x = [
    [1,0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,1,0,0],
    [1,1,1,1,1,1,0,1,0,0],
    [1,1,1,0,0,1,0,1,1,1],
    [1,1,1,0,1,1,1,0,1,0]
]

test_cases_y = [
    [1,0,0,0,0,0,0,1,0,0],
    [1,1,1,1,1,1,1,0,0,0],
    [1,1,1,1,1,1,0,0,0,0],
    [1,1,1,0,0,0,0,0,0,0],
    [1,1,1,0,0,0,1,0,0,0]
]

# Check the function and print results
for i, x in enumerate(test_cases_x):
    result = compute_y(x)
    expected = test_cases_y[i]
    if result == expected:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure: Expected {expected}, got {result}")