# Define the first 5 rows of list x
list_of_x = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

# Define the first 5 rows of list y for comparison
list_of_y = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

# Function to compute y from x
def compute_y(x):
    return x

# Check and print result for each row
for i in range(len(list_of_x)):
    computed_y = compute_y(list_of_x[i])
    if computed_y == list_of_y[i]:
        print('Success')
    else:
        print('Failure')

# Alternatively, you could check the entire list at once:
# computed_list_of_y = [compute_y(x) for x in list_of_x]
# if computed_list_of_y == list_of_y:
#     print('Success')
# else:
#     print('Failure')