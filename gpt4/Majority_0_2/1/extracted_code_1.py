def compute_y(x):
    y = []
    for i in range(len(x)):
        # Rule for when the element is 0.
        if x[i] == 0:
            y.append(0)
        # Rule for when the element is 2.
        elif x[i] == 2:
            y.append(2)
        # Rule for when the element is 1 (assuming based on trends).
        else:
            # Check if the previous element in list y was 0 or 1, and if element in list x is 1.
            # This is based on observation; it does not guarantee correctness outside provided data.
            if i > 0 and (y[i-1] == 1 or y[i-1] == 0) and x[i] == 1:
                y.append(1)
            else:
                y.append(0)
    return y

# Given lists
list_x = [
    [0,2,1,1,0,2,1,2,1,2],
    [1,1,2,0,0,1,2,1,0,1],
    [1,2,1,2,2,1,2,0,1,1],
    [0,0,1,1,2,1,2,2,1,0],
    [2,0,1,0,0,2,2,1,0,2]
]

list_y = [
    [0,0,0,1,0,0,1,1,1,1],
    [1,1,1,1,0,1,1,1,1,1],
    [1,1,1,1,2,1,2,2,1,1],
    [0,0,0,0,0,1,1,1,1,1],
    [2,0,0,0,0,0,0,0,0,0]
]

# Check if the program computes y correctly from x and print the result
for i in range(len(list_x)):
    computed_y = compute_y(list_x[i])
    if computed_y == list_y[i]:
        print(f"Success for row {i+1}")
    else:
        print(f"Failure for row {i+1}")