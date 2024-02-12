# Define the input lists 'x' and the expected lists 'y'
input_x = [
    [[2, 2], [1, 4], [1, 0], [0, 4], [0, 3], [3, 4], [0, 4], [1, 2], [0, 0], [2, 1]],
    [[4, 1], [3, 1], [4, 3], [1, 4], [2, 4], [2, 0], [0, 4], [3, 4], [4, 1], [2, 0]],
    [[1, 2], [2, 4], [2, 3], [3, 4], [3, 2], [0, 4], [0, 4], [1, 4], [0, 4], [3, 3]],
    [[1, 0], [0, 0], [0, 1], [3, 0], [1, 1], [2, 4], [4, 3], [3, 4], [3, 2], [3, 1]],
    [[4, 1], [1, 3], [2, 0], [4, 3], [0, 3], [2, 2], [0, 4], [1, 0], [1, 4], [1, 4]],
]

expected_y = [
    [4, 0, 2, 4, 3, 2, 0, 4, 0, 3],
    [0, 0, 3, 1, 2, 3, 4, 2, 1, 3],
    [3, 1, 1, 3, 1, 0, 0, 1, 0, 2],
    [1, 0, 1, 3, 2, 1, 3, 3, 1, 0],
    [0, 0, 3, 2, 4, 4, 4, 1, 0, 1],
]

# Write a function to guess y based on x
def guess_y(x_list):
    guessed_y = []
    for x in x_list:
        # Trying a simple operation: sum of the elements
        # This is just a guess, replace with your own logic as necessary.
        result = sum(x) % 5  # Modulo operation is applied to get a result within a range
        guessed_y.append(result)
    return guessed_y

# Main program logic
success = True
for i, single_x in enumerate(input_x):
    computed_y = guess_y(single_x)
    if computed_y == expected_y[i]:
        print(f"Row {i+1}: Success")
    else:
        success = False
        print(f"Row {i+1}: Failure, expected {expected_y[i]} but got {computed_y}")

# Print overall success
if success:
    print("Success")
else:
    print("Failure")