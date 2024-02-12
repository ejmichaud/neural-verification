def compute_y(x):
    y = []
    sum_so_far = 0
    for num in x:
        sum_so_far += num[0]
        y.append(sum_so_far)
    return y

# Define the input and expected output for the first 5 rows
data = [
    ([5],[-8],[-6],[9],[8],[7],[9],[-5],[-3],[4], [5],[-3],[-14],[-2],[20],[29],[18],[-16],[-37],[-17]),
    ([-3],[9],[10],[8],[5],[6],[10],[3],[-1],[0], [-3],[6],[19],[21],[7],[-8],[-5],[6],[10],[4]),
    ([-9],[4],[-3],[7],[7],[9],[-2],[5],[6],[6], [-9],[-5],[1],[13],[19],[15],[-6],[-16],[-4],[18]),
    ([-1],[2],[-3],[-3],[-5],[-3],[1],[7],[3],[5], [-1],[1],[-1],[-5],[-9],[-7],[3],[17],[17],[5]),
    ([4],[5],[9],[2],[-4],[1],[4],[6],[2],[-5], [4],[9],[14],[7],[-11],[-17],[-2],[21],[25],[-1])
]

# Process each row, compute y, and check against expected y
for i, (x_list, expected_y_list) in enumerate(data):
    computed_y_list = compute_y(x_list)
    expected_y_values = [pair[0] for pair in expected_y_list]  # Extract numbers from single-element lists
    # Check if computed y matches expected y
    if computed_y_list == expected_y_values:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")

        # Print details on failure for debugging purposes
        print(f"Computed y: {computed_y_list}")
        print(f"Expected y: {expected_y_values}")