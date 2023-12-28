# Define the lists
list_x = [
    [2,3,0,2,2,3,0,0,2,1],
    [2,2,2,2,3,0,3,3,3,2],
    [1,0,1,3,3,1,1,1,3,3],
    [0,0,3,1,1,0,3,0,0,2],
    [2,2,1,3,3,3,3,2,1,1]
]
list_y_expected = [
    [2,2,0,2,2,2,2,0,2,2],
    [2,2,2,2,2,2,2,2,2,2],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0],
    [2,2,2,2,2,3,3,3,3,3]
]

# Compute list y and check for matching with list y_expected
for i in range(5):
    list_y_computed = [max(x) for x in list_x[i]]
    if list_y_computed == list_y_expected[i]:
        print("Success")
    else:
        print("Failure")