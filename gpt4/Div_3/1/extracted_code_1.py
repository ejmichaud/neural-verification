def generate_y_from_x(x):
    '''
    This function attempts to generate list y from list x based on observed rules.
    '''
    y = []
    
    # Keep track of the count of consecutive 1s in x
    count_1s = 0
    last_value = 0

    for item in x:
        if item == 1:
            count_1s += 1
            result = 0
        else:
            # Append 1 to y if the current item is 0 and it breaks a streak of 1s in x
            if count_1s >= 1:
                result = 1
            else:
                result = 0
            count_1s = 0
        
        y.append(result)
        last_value = item
    
    return y

# Test the first five rows
x_lists = [
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
]

y_lists = [
    [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 1]
]

# Check if the function generates the correct output
for i in range(len(x_lists)):
    y_generated = generate_y_from_x(x_lists[i])
    if y_generated == y_lists[i]:
        print(f'Row {i+1} - Success')
    else:
        print(f'Row {i+1} - Failure')
        print(f'Expected: {y_lists[i]}')
        print(f'Received: {y_generated}')