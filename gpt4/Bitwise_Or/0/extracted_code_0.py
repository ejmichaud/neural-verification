# Define the input data for x and the expected output for y
list_of_x = [
    [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]],
    [[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0]],
    [[0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0]],
    [[0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1]],
    [[0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0]],
]
expected_y = [
    [1,0,1,0,1,0,0,1,1,1],
    [1,1,1,1,1,0,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,1,1],
    [1,1,0,1,1,1,1,1,1,1],
]

# Function to compute y from each element of x
def compute_y(x):
    return [max(a, b) for a, b in x]

# Main program to compute y and check against the expected y
success = True

for idx, x in enumerate(list_of_x):
    computed_y = compute_y(x)
    if computed_y == expected_y[idx]:
        print(f"Row {idx+1}: Success")
    else:
        print(f"Row {idx+1}: Failure")
        success &= False

if success:
    print("All computations are Successful!")
else:
    print("Some computations Failed.")