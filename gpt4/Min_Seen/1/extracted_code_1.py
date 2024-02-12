def generate_y(x):
    y = []
    current_max = -float('inf')  # Starts with the lowest possible value
    zero_encountered = False
    
    for value in x:
        if value == 0:
            zero_encountered = True
        if zero_encountered:
            y.append(0)
        else:
            current_max = max(current_max, value)
            y.append(current_max)
    
    return y

# Data from the first five rows
data = [
    ([2,7,6,4,6,5,0,4,0,3], [2,2,2,2,2,2,0,0,0,0]),
    ([8,4,0,4,1,2,5,5,7,6], [8,4,0,0,0,0,0,0,0,0]),
    ([9,6,3,1,9,3,1,9,7,9], [9,6,3,1,1,1,1,1,1,1]),
    ([2,0,5,9,3,4,9,6,2,0], [2,0,0,0,0,0,0,0,0,0]),
    ([6,2,7,9,7,3,3,4,3,7], [6,2,2,2,2,2,2,2,2,2]),
]

# Check for each row if the generated y matches the provided y
for i, (x, expected_y) in enumerate(data):
    calculated_y = generate_y(x)
    if calculated_y == expected_y:
        print(f"Row {i + 1}: Success")
    else:
        print(f"Row {i + 1}: Failure")
        print(f"Expected: {expected_y}")
        print(f"Calculated: {calculated_y}")