# Define the function to compute cumulative sum
def cumulative_sum(x):
    y = []
    total = 0
    for value in x:
        total += value
        y.append(total)
    return y

# Given data for the first five rows
data = [
    ([42,67,76,14,26,35,20,24,50,13], [42,109,185,199,225,260,280,304,354,367]),
    ([78,14,10,54,31,72,15,95,67,6], [78,92,102,156,187,259,274,369,436,442]),
    ([49,76,73,11,99,13,41,69,87,19], [49,125,198,209,308,321,362,431,518,537]),
    ([72,80,75,29,33,64,39,76,32,10], [72,152,227,256,289,353,392,468,500,510]),
    ([86,22,77,19,7,23,43,94,93,77], [86,108,185,204,211,234,277,371,464,541]),
]

# Process each row, compute y from x, and check if it matches the expected y
for i, (x, expected_y) in enumerate(data):
    # Calculate y based on x
    computed_y = cumulative_sum(x)
    
    # Check if the computed y matches the expected y and print the result
    if computed_y == expected_y:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")