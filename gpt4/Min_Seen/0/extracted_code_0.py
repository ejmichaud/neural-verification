# Define the function to compute y from x
def compute_y(x):
    y = []
    max_val = -float('inf')  # Initialize to negative infinity
    for val in x:
        if val == 0:
            # Reset max_val if the current element is 0
            max_val = -float('inf')
            y.append(0)
        else:
            max_val = max(max_val, val)
            y.append(max_val)
    return y

# Test the function with the first five rows
test_cases = [
    ([2,7,6,4,6,5,0,4,0,3], [2,2,2,2,2,2,0,0,0,0]),
    ([8,4,0,4,1,2,5,5,7,6], [8,4,0,0,0,0,0,0,0,0]),
    ([9,6,3,1,9,3,1,9,7,9], [9,6,3,1,1,1,1,1,1,1]),
    ([2,0,5,9,3,4,9,6,2,0], [2,0,0,0,0,0,0,0,0,0]),
    ([6,2,7,9,7,3,3,4,3,7], [6,2,2,2,2,2,2,2,2,2]),
]

# Loop through each test case, compute y, and check if the result matches expected y
for i, (x, expected_y) in enumerate(test_cases):
    calculated_y = compute_y(x)
    if calculated_y == expected_y:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")