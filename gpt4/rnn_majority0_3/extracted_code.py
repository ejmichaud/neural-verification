def compute_y_from_x(x):
    y = []
    
    # A heuristic to attempt replicating the given pattern
    for i in range(len(x)):
        # Check the number of occurrences of x[i] in the entire list x up to index i
        count = x[:i+1].count(x[i])
        
        # If the current number hasn't appeared before or has appeared exactly twice, 
        # use the same number for list y
        if count == 1 or count == 2:
            y.append(x[i])
        # Otherwise, if the number has appeared more than twice, turn it into a 2 if it is not already 0
        elif count > 2 and x[i] != 0:
            y.append(2)
        # Otherwise, keep the number as is
        else:
            y.append(0)
    
    return y

# The first 5 rows to process
list_of_x = [
    [2,3,0,2,2,3,0,0,2,1],
    [2,2,2,2,3,0,3,3,3,2],
    [1,0,1,3,3,1,1,1,3,3],
    [0,0,3,1,1,0,3,0,0,2],
    [2,2,1,3,3,3,3,2,1,1]
]

# The correct list y we are trying to match
correct_list_of_y = [
    [2,2,0,2,2,2,2,0,2,2],
    [2,2,2,2,2,2,2,2,2,2],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0],
    [2,2,2,2,2,3,3,3,3,3]
]

# Test the function and print the results
success = True
for i, x in enumerate(list_of_x):
    y = compute_y_from_x(x)
    if y == correct_list_of_y[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")
        success = False

if success:
    print("All tests passed!")
else:
    print("Some tests failed.")