def compute_y(x):
    # Speculative function: this is a random guess and likely incorrect,
    # but we can start with this and compare the output
    return [(xi + 1) % 5 for xi in x]  # Adding 1 and then modulo 5 - purely a guess

# Given data for the first five rows
list_x = [
    [2,2,1,4,1,0,0,4,0,3],
    [3,4,0,4,1,2,0,0,2,1],
    [4,1,3,1,4,3,1,4,2,4],
    [2,0,0,4,3,4,4,1,2,0],
    [1,2,2,4,2,3,3,4,3,2]
]

list_y = [
    [2,4,0,4,0,0,0,4,4,2],
    [3,2,2,1,2,4,4,4,1,2],
    [4,0,3,4,3,1,2,1,3,2],
    [2,2,2,1,4,3,2,3,0,0],
    [1,3,0,4,1,4,2,1,4,1]
]

# Iterate over each row to compute y from x and compare with the given y
for i in range(5):
    computed_y = compute_y(list_x[i])
    if computed_y == list_y[i]:
        print("Row", i+1, "Success")
    else:
        print("Row", i+1, "Failure")