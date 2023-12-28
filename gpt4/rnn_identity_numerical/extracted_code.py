# Define the test data for the first 5 rows
data = [
    ([42,67,76,14,26,35,20,24,50,13], [42,67,76,14,26,35,20,24,50,13]),
    ([78,14,10,54,31,72,15,95,67,6], [78,14,10,54,31,72,15,95,67,6]),
    ([49,76,73,11,99,13,41,69,87,19], [49,76,73,11,99,13,41,69,87,19]),
    ([72,80,75,29,33,64,39,76,32,10], [72,80,75,29,33,64,39,76,32,10]),
    ([86,22,77,19,7,23,43,94,93,77], [86,22,77,19,7,23,43,94,93,77])
]

# Function to compute list y from list x
def compute_y(x):
    return x.copy()

# Process each row and check if y computed from x matches the given y
success = True
for x_list, y_list in data:
    computed_y = compute_y(x_list)
    if computed_y != y_list:
        success = False
        break

# Print 'Success' if all y lists match, otherwise print 'Failure'
if success:
    print('Success')
else:
    print('Failure')