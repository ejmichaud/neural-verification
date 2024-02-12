def transform(x):
    # Compute y based on an empirical rule derived from the first five rows:
    y = []
    for i in range(len(x)):
        if i == 0:  # Handling the first element edge case
            y.append(0 if x[i] == 1 else 1)
            continue
        # Rule: Toggle when the previous element is 1 and the current element is 0
        if x[i - 1] == 1 and x[i] == 0:
            y.append(1)
        else:
            y.append(x[i])
    return y

# Define the input and expected output for the first 5 rows
inputs = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

expected_outputs = [
    [0,1,1,1,1,0,0,0,0,1],
    [0,0,0,0,1,1,0,1,0,0],
    [1,1,0,1,0,1,0,1,0,1],
    [0,0,1,0,1,1,0,0,0,0],
    [0,0,1,0,1,0,1,1,0,1]
]

# Perform the check for the first five rows
success = True
for x, expected_y in zip(inputs, expected_outputs):
    y = transform(x)
    if y != expected_y:
        success = False
        break

if success:
    print("Success")
else:
    print("Failure")