# Define the first five rows for list x and list y
lists_x = [
    [2,7,6,4,6,5,0,4,0,3],
    [8,4,0,4,1,2,5,5,7,6],
    [9,6,3,1,9,3,1,9,7,9],
    [2,0,5,9,3,4,9,6,2,0],
    [6,2,7,9,7,3,3,4,3,7],
]

given_ys = [
    [1,1,2,3,4,4,5,6,7,7],
    [1,2,3,4,4,5,5,5,5,6],
    [0,1,1,1,1,1,1,1,1,1],
    [1,2,2,2,2,3,3,4,5,6],
    [1,2,2,2,2,2,2,3,3,3],
]

def compute_y_from_x(x):
    unique_elements = set()
    y = []
    for element in x:
        unique_elements.add(element)
        y.append(len(unique_elements))
    return y

# Process and check the lists
for i in range(len(lists_x)):
    computed_y = compute_y_from_x(lists_x[i])
    if computed_y == given_ys[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")