def compute_y(x_list):
    # Create a list of y's with the same length as x, initialized with 0's
    y_list = [0 for _ in x_list]
    # Loop through the positions in x, except for the first and last element
    for i in range(1, len(x_list) - 1):
        if x_list[i] == 1 and (x_list[i - 1] == 1 or x_list[i + 1] == 1):
            y_list[i] = 1
        elif x_list[i] == 0 and x_list[i - 1] == 1 and x_list[i + 1] == 1:
            y_list[i] = 1
    # Exceptionally set the last element based on the second to last element
    y_list[-1] = x_list[-2]
    return y_list

# Define the first 5 rows of x and y as provided in the question
x_input = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]
y_actual = [
    [0,0,0,0,0,1,0,0,1,1],
    [0,0,0,0,0,0,0,0,1,1],
    [0,0,0,0,1,1,0,1,1,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,1]
]

# Check each row
for index in range(len(x_input)):
    y_computed = compute_y(x_input[index])
    print(f"Row {index+1}:", "Success" if y_computed == y_actual[index] else "Failure", "- Computed:", y_computed, "Actual:", y_actual[index])