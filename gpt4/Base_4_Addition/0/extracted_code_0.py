import itertools

# Define a function to try various operations
def compute_y(x):
    # Try simple arithmetic operations between the two elements of x
    a, b = x
    return [
        a + b,     # Sum of elements
        a - b,     # Difference of elements
        b - a,     # Difference in reverse
        a * b,     # Product of elements
        a ** b,    # a to the power of b
        b ** a,    # b to the power of a
        a // b if b != 0 else 0,  # Integer division a by b
        b // a if a != 0 else 0,  # Integer division b by a
    ]

# List of x and y values for the first 5 rows
x_lists = [
    [[2, 3], [0, 2], [2, 3], [0, 0], [2, 1], [2, 2], [2, 2], [3, 0], [3, 3], [3, 2]],
    [[1, 0], [1, 3], [3, 1], [1, 1], [3, 3], [0, 0], [3, 1], [1, 0], [3, 0], [0, 2]],
    [[2, 2], [1, 3], [3, 3], [3, 2], [1, 1], [2, 1], [2, 3], [2, 3], [3, 0], [2, 0]],
    [[2, 2], [0, 0], [2, 1], [3, 0], [3, 1], [1, 1], [0, 1], [0, 1], [3, 3], [2, 3]],
    [[2, 3], [0, 3], [2, 2], [1, 0], [3, 1], [3, 3], [1, 1], [1, 1], [1, 3], [1, 0]]
]

y_lists = [
    [1, 3, 1, 1, 3, 0, 1, 0, 3, 2],
    [1, 0, 1, 3, 2, 1, 0, 2, 3, 2],
    [0, 1, 3, 2, 3, 3, 1, 2, 0, 3],
    [0, 1, 3, 3, 0, 3, 1, 1, 2, 2],
    [1, 0, 1, 2, 0, 3, 3, 2, 0, 2]
]

# Brute-force search for a correct formula using operations in compute_y function
# Store the potential operations that match the y values
potential_operations = []

# Go through each set of x and y values
for i in range(5):
    # Get the corresponding y list for each x list
    y_list = y_lists[i]
    operation_matches = []
    # Try all operations and find the one that matches the given y list
    for index, x in enumerate(x_lists[i]):
        computed_values = compute_y(x)
        # Check if any of the computed values match the corresponding y value
        operation_index = [j for j, val in enumerate(computed_values) if val == y_list[index]]
        operation_matches.append(operation_index)

    # Find the common operation index among all elements of x that matches the y values
    common_operations = set(operation_matches[0])
    for match_set in operation_matches[1:]:
        # Find intersection of all sets to get common operation index
        common_operations.intersection_update(match_set)

    # Collect all potential operations from the first row that could match further
    if i == 0:
        potential_operations = list(common_operations)
    else:
        # Retain only the operations that continue to produce the correct output
        potential_operations = [op for op in potential_operations if op in common_operations]

    # Print the result for each row
    if common_operations:
        print(f"Row {i+1} Success: Possible operations are {common_operations}.")
    else:
        print(f"Row {i+1} Failure: No matching operations found.")
    
# Printing the results of the final potential operations
print("\nAfter checking all 5 rows, potential operation indices are:", potential_operations)