# Define the function to compute list y from list x
def compute_y(x):
    y = []

    # Start by checking the first non-zero number in x
    y.append(1 if x[0] % 2 == 0 else 0)

    for i in range(1, len(x)):
        if x[i] == 0 and x[i - 1] != 0:
            y.append(y[-1])  # If current is 0 and previous is not, y stays the same
        elif x[i] == 0 and x[i - 1] == 0:
            y.append(y[-1] + 1)  # If current and previous are 0, y increments
        else:
            parity_change = (x[i] % 2) != (x[i - 1] % 2)
            y.append(y[-1] + 1 if parity_change else y[-1])  # Increment y if parity changes, else stay

    return y

# Given data
data = [
    ([2,7,6,4,6,5,0,4,0,3], [1,1,2,3,4,4,5,6,7,7]),
    ([8,4,0,4,1,2,5,5,7,6], [1,2,3,4,4,5,5,5,5,6]),
    ([9,6,3,1,9,3,1,9,7,9], [0,1,1,1,1,1,1,1,1,1]),
    ([2,0,5,9,3,4,9,6,2,0], [1,2,2,2,2,3,3,4,5,6]),
    ([6,2,7,9,7,3,3,4,3,7], [1,2,2,2,2,2,2,3,3,3]),
]

# Check the program for the first 5 rows
for i, (x, expected_y) in enumerate(data):
    computed_y = compute_y(x)
    if computed_y == expected_y:
        print(f'Row {i+1} Success: {computed_y}')
    else:
        print(f'Row {i+1} Failure: {computed_y}')