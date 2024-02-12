from sklearn.linear_model import LinearRegression
import numpy as np

# Provided lists x and y
list_x = [
    [1, 2, 4, 5, 4, 3, 5, 5, 0, 0],
    [0, 5, 6, 4, 1, 2, 6, 6, 2, 3],
    [1, 0, 0, 3, 3, 5, 1, 1, 2, 2],
    [2, 5, 0, 0, 5, 0, 4, 3, 6, 1],
    [0, 1, 5, 5, 6, 4, 0, 2, 5, 5]
]
list_y = [
    [1, 3, 0, 5, 2, 5, 3, 1, 1, 1],
    [0, 5, 4, 1, 2, 4, 3, 2, 4, 0],
    [1, 1, 1, 4, 0, 5, 6, 0, 2, 4],
    [2, 0, 0, 0, 5, 5, 2, 5, 4, 5],
    [0, 1, 6, 4, 3, 0, 0, 2, 0, 5]
]

# Convert the lists to numpy arrays
array_x = np.array(list_x)
array_y = np.array(list_y)

success = True  # Variable to track success or failure

# Process each row
for i in range(len(array_x)):
    # Prepare the data for regression (x values need to be 2D for sklearn)
    X = array_x[i].reshape(-1, 1)
    y = array_y[i]
    
    # Perform linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    
    # Predict y values using the model
    y_pred = lin_reg.predict(X)
    
    # Check if the predicted values match the actual y values using a tolerance
    if not np.allclose(y, y_pred.round(), atol=1):
        success = False
        break  # If any prediction is wrong, we can stop checking

# Print the result
if success:
    print('Success')
else:
    print('Failure')