def compute_y_heuristic(x_pairs):
    # Heuristic: if either element of the pair is 1, output 1, else 0
    return [max(pair) for pair in x_pairs]

def check_success(actual_y, computed_y):
    return 'Success' if actual_y == computed_y else 'Failure'

# Input data (first 5 rows)
x_lists = [
    [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]],
    [[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0]],
    [[0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0]],
    [[0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1]],
    [[0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0]],
]

# Expected y outputs for the first 5 rows
y_lists = [
    [1,0,1,0,1,0,0,1,1,1],
    [1,1,1,1,1,0,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,1,1],
    [1,1,0,1,1,1,1,1,1,1],
]

# Compute y for the first 5 rows and check if output matches expected y
for i, x_pairs in enumerate(x_lists):
    computed_y = compute_y_heuristic(x_pairs)
    result = check_success(y_lists[i], computed_y)
    print(f"Row {i+1}: {result}")

# We can also directly check the overall success by comparing all computed y's with all expected y's
all_success = all(check_success(y_lists[i], compute_y_heuristic(x_lists[i])) == 'Success' for i in range(5))
print("Overall:", "Success" if all_success else "Failure")