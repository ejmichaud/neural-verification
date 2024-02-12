def compute_y(x):
    y = []
    current = x[0]
    for value in x:
        if value <= current:
            current = value
        y.append(current)
    return y

# Define the test cases based on the first five rows provided
test_cases_x = [
    [2,7,6,4,6,5,0,4,0,3],
    [8,4,0,4,1,2,5,5,7,6],
    [9,6,3,1,9,3,1,9,7,9],
    [2,0,5,9,3,4,9,6,2,0],
    [6,2,7,9,7,3,3,4,3,7]
]

test_cases_y = [
    [2,2,2,2,2,2,0,0,0,0],
    [8,4,0,0,0,0,0,0,0,0],
    [9,6,3,1,1,1,1,1,1,1],
    [2,0,0,0,0,0,0,0,0,0],
    [6,2,2,2,2,2,2,2,2,2]
]

# Run the test cases
for i, x in enumerate(test_cases_x):
    y_computed = compute_y(x)
    if y_computed == test_cases_y[i]:
        print(f"Test Case {i+1}: Success")
    else:
        print(f"Test Case {i+1}: Failure")
        print("Computed y:", y_computed)
        print("Expected y:", test_cases_y[i])