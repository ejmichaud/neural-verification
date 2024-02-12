# Lists of x and y provided
list_of_x = [
    [6,3,4,6,2,7,4,4,6,1],
    [2,6,2,2,7,4,3,7,7,2],
    [5,4,1,7,3,5,5,1,7,3],
    [4,0,3,1,5,4,3,0,0,2],
    [2,6,1,7,3,3,7,6,5,5]
]

list_of_y = [
    [6,1,5,3,5,4,0,4,2,3],
    [2,0,2,4,3,7,2,1,0,2],
    [5,1,2,1,4,1,6,7,6,1],
    [4,4,7,0,5,1,4,4,4,6],
    [2,0,1,0,3,6,5,3,0,5]
]

def compute_y_from_x(x, i):
    """Compute y using each value of x minus its index, modulo 8."""
    return (x - i) % 8

# Function to check the success of the computed y
def check_success(x_lists, y_expected_lists):
    for i, x in enumerate(x_lists):
        computed_y = [compute_y_from_x(x_value, idx) for idx, x_value in enumerate(x)]
        if computed_y == y_expected_lists[i]:
            print(f"Row {i+1}: Success")
        else:
            print(f"Row {i+1}: Failure")

# Run the check for the first 5 rows
check_success(list_of_x[:5], list_of_y[:5])