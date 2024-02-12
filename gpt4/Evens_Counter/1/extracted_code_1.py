def compute_y(x):
    y = []
    for i in range(len(x)):
        count = sum(1 for el in x[:i] if el < x[i])
        y.append(count)
    return y

# Given data for the first 5 rows
input_data_x = [
    [2,7,6,4,6,5,0,4,0,3],
    [8,4,0,4,1,2,5,5,7,6],
    [9,6,3,1,9,3,1,9,7,9],
    [2,0,5,9,3,4,9,6,2,0],
    [6,2,7,9,7,3,3,4,3,7]
]
output_data_y = [
    [1,1,2,3,4,4,5,6,7,7],
    [1,2,3,4,4,5,5,5,5,6],
    [0,1,1,1,1,1,1,1,1,1],
    [1,2,2,2,2,3,3,4,5,6],
    [1,2,2,2,2,2,2,3,3,3]
]

# Process each row and check if the calculated y matches the given y
for i in range(5):
    calculated_y = compute_y(input_data_x[i])
    if calculated_y == output_data_y[i]:
        print(f"Success for row {i+1}")
    else:
        print(f"Failure for row {i+1}")