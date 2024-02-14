import sympy

def compute_y(x):
    # Placeholder for our guessed function
    y = []
    for xi in x:
        if sympy.isprime(xi):  # Check if it's a prime number
            y.append(1)
        else:
            y.append(0)
    return y

rows_x = [
    [2, 7, 16, 14, 6, 15, 0, 4, 10, 13],
    [18, 14, 10, 14, 11, 12, 15, 15, 7, 6],
    [9, 16, 13, 11, 19, 13, 1, 9, 7, 19],
    [12, 0, 15, 9, 13, 4, 19, 16, 12, 10],
    [6, 2, 17, 19, 7, 3, 3, 14, 13, 17]
]

rows_y = [
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

success = True
for i, x in enumerate(rows_x):
    y_computed = compute_y(x)
    if y_computed != rows_y[i]:
        success = False
        print(f'Failure on row {i+1}')
        break

if success:
    print('Success')