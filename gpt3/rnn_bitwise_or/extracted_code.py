# Function to compute y from x
def compute_y(x):
    return [a | b for a, b in x]

# Input list x
list_x = [
    [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]],
    [[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0]],
    [[0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0]],
    [[0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1]],
    [[0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0]]
]

# Expected output list y
expected_y = [
    [1,0,1,0,1,0,0,1,1,1],
    [1,1,1,1,1,0,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,1,1],
    [1,1,0,1,1,1,1,1,1,1]
]

# Compute list y and check if it matches expected output list y
success = True
for i in range(5):
    computed_y = compute_y(list_x[i])
    if computed_y != expected_y[i]:
        print("Failure")
        success = False
        break

if success:
    print("Success")