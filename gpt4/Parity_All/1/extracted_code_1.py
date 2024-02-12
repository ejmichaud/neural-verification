# Function to compute y from x
def compute_y(x):
    # Implement the observed pattern
    y = []
    for i in range(len(x)):
        # Tentative rule based on pattern inspection
        if i < len(x)-1 and x[i] == 0 and x[i+1] == 1:
            y.append(1)
        elif i < len(x)-1 and x[i] == 1 and x[i+1] == 0:
            y.append(0)
        elif i == len(x)-1:  # Handle the last element
            y.append(x[i])
        else:
            y.append(x[i])
    return y

# Define the dataset for the first 5 rows of x and y
dataset = [
    ([0,1,0,0,0,1,0,0,0,1], [0,1,1,1,1,0,0,0,0,1]), 
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,1,1,0,1,0,0]), 
    ([1,0,1,1,1,1,1,1,1,1], [1,1,0,1,0,1,0,1,0,1]), 
    ([0,0,1,1,1,0,1,0,0,0], [0,0,1,0,1,1,0,0,0,0]), 
    ([0,0,1,1,1,1,1,0,1,1], [0,0,1,0,1,0,1,1,0,1])
]

# Process each row and validate the results
for row in dataset:
    x, expected_y = row
    computed_y = compute_y(x)
    # Check if the computed y matches the expected y
    if computed_y == expected_y:
        print('Success')
    else:
        print('Failure')
        print(f"x: {x}")
        print(f"Expected y: {expected_y}")
        print(f"Computed y: {computed_y}")