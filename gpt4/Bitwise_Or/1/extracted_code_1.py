def compute_y(x_row):
    return [1 if x[0] == 1 or x[1] == 1 else 0 for x in x_row]

# Define list x and list y for the first 5 rows
list_x = [
    [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]],
    [[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0]],
    [[0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0]],
    [[0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1]],
    [[0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0]]
]

list_y = [
    [1,0,1,0,1,0,0,1,1,1],
    [1,1,1,1,1,0,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,1,1],
    [1,1,0,1,1,1,1,1,1,1]
]

# Apply the hypothesis function to each row and check if it matches the corresponding y
for i in range(len(list_x)):
    computed_y = compute_y(list_x[i])
    if computed_y == list_y[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")

        # For debugging purposes, you might want to print the expected vs. actual if there's a failure:
        print("Expected:", list_y[i])
        print("Computed:", computed_y)