# The given data for the first 5 rows (list x and list y)
data = [
    ([0,1,0,0,0,1,0,0,0,1], [0,1,0,1,0,1,0,1,0,1]),
    ([0,0,0,0,1,0,1,1,1,0], [0,1,0,1,0,1,0,1,0,1]),
    ([1,0,1,1,1,1,1,1,1,1], [0,1,0,1,0,1,0,1,0,1]),
    ([0,0,1,1,1,0,1,0,0,0], [0,1,0,1,0,1,0,1,0,1]),
    ([0,0,1,1,1,1,1,0,1,1], [0,1,0,1,0,1,0,1,0,1])
]

# Function to generate list y from list x, according to the observed rule
def compute_y_from_x(x_list):
    return [i % 2 for i in range(len(x_list))]

# Check function that compares the computed y with the actual y and prints the result
def check_and_print_result(data):
    for x_list, expected_y_list in data:
        # Generate y based on the length of x
        computed_y_list = compute_y_from_x(x_list)
        # Check if the generated y matches the expected y
        if computed_y_list == expected_y_list:
            print('Success')
        else:
            print('Failure')

# Run the check function on the provided data
check_and_print_result(data)