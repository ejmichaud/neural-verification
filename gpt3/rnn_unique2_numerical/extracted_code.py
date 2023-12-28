# Define the list x
list_x = [
    [0, 7, 7, 4, 7, 7, 2, 2, 6, 0],
    [0, 6, 7, 5, 4, 4, 2, 6, 3, 0],
    [6, 7, 1, 5, 3, 3, 1, 6, 3, 3],
    [3, 6, 0, 6, 5, 6, 4, 7, 6, 6],
    [6, 0, 3, 5, 3, 3, 1, 3, 1, 3]
]

# Define the expected list y
list_y_expected = [
    [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
]

# Compute list y
list_y = []
for x in list_x:
    y = [1 if i % 2 == 0 else 0 for i in x]
    list_y.append(y)

# Check if the computed list y matches the expected list y
if list_y == list_y_expected:
    print('Success')
else:
    print('Failure')

# Print the computed list y
print(list_y)