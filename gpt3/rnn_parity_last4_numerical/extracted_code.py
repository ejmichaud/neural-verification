def compute_y(x):
    y = [sum(subarray) for subarray in x]
    return y

x = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

y_expected = [
    [0,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,1,1,0,1,1,1],
    [1,1,0,1,1,0,0,0,0,0],
    [0,0,1,0,1,1,1,0,1,1],
    [0,0,1,0,1,0,0,1,1,1]
]

y_computed = compute_y(x)

if y_computed == y_expected:
    print("Success")
else:
    print("Failure")