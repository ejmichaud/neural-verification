# Lists x
list_x = [
    [0,2,1,1,0,2,1,2,1,2],
    [1,1,2,0,0,1,2,1,0,1],
    [1,2,1,2,2,1,2,0,1,1],
    [0,0,1,1,2,1,2,2,1,0],
    [2,0,1,0,0,2,2,1,0,2]
]

# Expected lists y
expected_y = [
    [0,0,0,1,0,0,1,1,1,1],
    [1,1,1,1,0,1,1,1,1,1],
    [1,1,1,1,2,1,2,2,1,1],
    [0,0,0,0,0,1,1,1,1,1],
    [2,0,0,0,0,0,0,0,0,0]
]

def compute_y(x_list):
    # Placeholder function just to begin with
    return [0] * len(x_list)

# Initial hypothesis: y values are dependent on the presence of consecutive identical values in list x
def compute_y_based_on_hypothesis(x_list):
    y_list = []
    for i in range(len(x_list)):
        if i == 0:  # there's nothing before the first element
            y_list.append(0)
        else:
            # Check if current value matches the previous one
            if x_list[i] == x_list[i-1]:
                y_list.append(1)  # Hypothetical case: it generates '1' if values match
            else:
                y_list.append(0)  # Hypothetical case: it generates '0' if values don't match
    return y_list

# Compare computed y with the given y
success = True
for i in range(len(list_x)):
    computed_y = compute_y_based_on_hypothesis(list_x[i])
    if computed_y == expected_y[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")
        print(f"Computed: {computed_y}, Expected: {expected_y[i]}")
        success = False

# Final output
if success:
    print("Success: All computed lists match the expected outputs!")
else:
    print("Failure: Not all computed lists match the expected outputs.")