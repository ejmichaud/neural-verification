def compute_y(x):
    frequency = {}
    y = []
    max_frequency_value = 0
    max_frequency = 0
    
    for value in x:
        # Update frequency for the current value
        frequency[value] = frequency.get(value, 0) + 1
        
        # Update the max_frequency_value if needed
        if frequency[value] > max_frequency:
            max_frequency = frequency[value]
            max_frequency_value = value
        elif frequency[value] == max_frequency:
            # Tie breaker logic (may require additional rules)
            max_frequency_value = min(max_frequency_value, value)
        
        # Update y according to the observed rules
        if frequency[value] == 1 and max_frequency == 1:
            y.append(0)
        else:
            y.append(max_frequency_value)
    
    return y

# Define the lists of x and y
lists_x = [
    [0,5,4,4,0,5,4,2,4,5],
    [4,4,2,0,3,4,5,1,3,4],
    [1,2,1,5,5,1,5,3,1,1],
    [0,0,1,1,5,4,5,2,4,0],
    [2,0,1,3,3,5,5,4,3,5]
]

lists_y = [
    [0,0,0,4,0,0,4,4,4,4],
    [4,4,4,4,4,4,4,4,4,4],
    [1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0],
    [2,0,0,0,3,3,3,3,3,3]
]

# Iterate through the first five rows and compute y from x
for i, x in enumerate(lists_x):
    # Compute list y from list x
    computed_y = compute_y(x)
    
    # Check if the computed y matches the given y for the row
    if computed_y == lists_y[i]:
        print(f"Row {i+1} matches: Success")
    else:
        print(f"Row {i+1} does not match: Failure")
        print(f"Expected: {lists_y[i]}")
        print(f"Computed: {computed_y}")