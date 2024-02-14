# Define lists x and corresponding lists y from the table for the first 5 rows
data = [
    ([42,67,76,14,26,35,20,24,50,13,78,14,10,54,31,72,15,95,67,6], [0,0,0,0,0,42,67,76,14,26,35,20,24,50,13,78,14,10,54,31]),
    ([49,76,73,11,99,13,41,69,87,19,72,80,75,29,33,64,39,76,32,10], [0,0,0,0,0,49,76,73,11,99,13,41,69,87,19,72,80,75,29,33]),
    ([86,22,77,19,7,23,43,94,93,77,70,9,70,39,86,99,15,84,78,8], [0,0,0,0,0,86,22,77,19,7,23,43,94,93,77,70,9,70,39,86]),
    ([66,30,40,60,70,61,23,20,11,61,77,89,84,53,48,9,83,7,58,91], [0,0,0,0,0,66,30,40,60,70,61,23,20,11,61,77,89,84,53,48]),
    ([14,91,36,3,82,90,89,28,55,33,27,47,65,89,41,45,61,39,61,64], [0,0,0,0,0,14,91,36,3,82,90,89,28,55,33,27,47,65,89,41]),
]

def compute_y_from_x(x):
    return [0] * 5 + x[5:]

# Function to check if calculated y matches the expected y
def check_success(calculated_y, expected_y):
    return 'Success' if calculated_y == expected_y else 'Failure'

# Loop through the first 5 rows and check if computed y matches the expected y
for x, expected_y in data:
    calculated_y = compute_y_from_x(x)
    result = check_success(calculated_y, expected_y)
    print(result)