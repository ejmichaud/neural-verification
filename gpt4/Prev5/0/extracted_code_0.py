# Define a function to compute list y from list x
def transform_x_to_y(x):
    # Append five zeros at the beginning and remove the first five elements
    return [0] * 5 + x[5:]

# Define the given input lists and expected outputs for the first five rows
input_data = [
    (42,67,76,14,26,35,20,24,50,13,78,14,10,54,31,72,15,95,67,6),
    (49,76,73,11,99,13,41,69,87,19,72,80,75,29,33,64,39,76,32,10),
    (86,22,77,19,7,23,43,94,93,77,70,9,70,39,86,99,15,84,78,8),
    (66,30,40,60,70,61,23,20,11,61,77,89,84,53,48,9,83,7,58,91),
    (14,91,36,3,82,90,89,28,55,33,27,47,65,89,41,45,61,39,61,64)
]

output_data = [
    (0,0,0,0,0,35,20,24,50,13,78,14,10,54,31,72,15,95,67,6),
    (0,0,0,0,0,49,76,73,11,99,13,41,69,87,19,72,80,75,29,33),
    (0,0,0,0,0,86,22,77,19,7,23,43,94,93,77,70,9,70,39,86),
    (0,0,0,0,0,66,30,40,60,70,61,23,20,11,61,77,89,84,53,48),
    (0,0,0,0,0,14,91,36,3,82,90,89,28,55,33,27,47,65,89,41)
]

# Check each test case and print 'Success' or 'Failure'
for i in range(len(input_data)):
    x = input_data[i]
    expected_y = output_data[i]
    
    # Compute list y from list x
    y = transform_x_to_y(x)
    
    # Compare the computed list y with the expected list y and print the result
    if tuple(y) == expected_y:
        print('Success')
    else:
        print('Failure')
        print('Computed y:', y)
        print('Expected y:', expected_y)