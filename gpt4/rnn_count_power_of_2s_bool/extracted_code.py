import numpy as np

def predict_y(x_list):
    # A simple heuristic: Let's say y is 1 if the element is less than 
    # the mean of the list minus the standard deviation 
    # and greater than the mean of the list plus the standard deviation.
    x_array = np.array(x_list)
    mean = x_array.mean()
    std = x_array.std()
    threshold_low = mean - std
    threshold_high = mean + std
    y_predicted = [1 if x < threshold_low or x > threshold_high else 0 for x in x_list]
    return y_predicted

# Provided lists x and their corresponding true lists y
x_lists = [
    [42,32,32,14,26,35,20,24,50,13],
    [78,14,1,1,31,1,15,95,67,6],
    [49,1,1,11,99,13,41,1,87,1],
    [72,80,75,29,32,32,39,76,32,32],
    [86,4,4,4,4,23,43,94,93,4]
]

y_true = [
    [0,1,1,0,0,0,0,0,0,0],
    [0,0,1,1,0,1,0,0,0,0],
    [0,1,1,0,0,0,0,1,0,1],
    [0,0,0,0,1,1,0,0,1,1],
    [0,1,1,1,1,0,0,0,0,1]
]

success = True

# Attempting to predict y from x and checking if predictions match the true y
for i, x_list in enumerate(x_lists):
    y_pred = predict_y(x_list)
    if y_pred != y_true[i]:
        print(f"Failure on row {i+1}: Predicted {y_pred}, Actual {y_true[i]}")
        success = False
        break

if success:
    print("Success")