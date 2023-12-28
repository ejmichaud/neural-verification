def compute_y_from_x(x_list):
    # Since we don't have a clear rule, let's start with a simple placeholder function
    # that tries to mirror the observed patterns.
    y_list = []
    
    # Hypothetical rule: Let's check if the number of unique numbers up to the
    # current point affects y.
    unique_counts = 0
    unique_numbers = set()

    for x in x_list:
        unique_numbers.add(x)
        unique_counts = len(unique_numbers)
        # We found that in most rows, the y value seems to increase based on
        # the number of unique numbers we've seen so far, let's use it as our rule
        if unique_counts <= 4:
            y_list.append(0)
        elif unique_counts <= 6:
            y_list.append(1)
        elif unique_counts <= 8:
            y_list.append(2)
        else:
            y_list.append(3)

    return y_list

# The provided dataset
x_data = [
    [18,21,6,4,4,11,22,20,18,7],
    [4,2,18,24,9,22,21,1,15,6],
    [25,14,23,1,1,5,21,25,5,25],
    [14,12,5,1,17,24,13,20,10,8],
    [14,18,15,19,1,1,21,2,13,5]
]

y_data = [
    [0,0,0,1,2,2,2,3,3,3],
    [1,1,1,1,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,2,2,3],
    [1,1,1,1,1,1,1,1,1,1]
]

# Let's test our function
for idx, x_list in enumerate(x_data):
    computed_y = compute_y_from_x(x_list)
    if computed_y == y_data[idx]:
        print(f"Success for row {idx + 1}")
    else:
        print(f"Failure for row {idx + 1}")