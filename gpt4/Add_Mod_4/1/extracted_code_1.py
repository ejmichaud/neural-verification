# Given data
list_x = [
    [2,3,0,2,2,3,0,0,2,1],
    [2,2,2,2,3,0,3,3,3,2],
    [1,0,1,3,3,1,1,1,3,3],
    [0,0,3,1,1,0,3,0,0,2],
    [2,2,1,3,3,3,3,2,1,1]
]

list_y = [
    [2,1,1,3,1,0,0,0,2,3],
    [2,0,2,0,3,3,2,1,0,2],
    [1,1,2,1,0,1,2,3,2,1],
    [0,0,3,0,1,1,0,0,0,2],
    [2,0,1,0,3,2,1,3,0,1]
]

# The function to test a potential mapping from x to y
def predict_y(list_x_row):
    # Change the prediction function here to test different hypotheses
    prediction = list_x_row[:]  # Using the 'same as x' hypothesis as an example
    return prediction

# Check if the prediction matches the actual y values
def check_prediction(idx, row_x, row_y):
    predicted_y = predict_y(row_x)
    return "Success" if predicted_y == row_y else "Failure", predicted_y

# Loop through the first five rows and check the predictions
for i in range(5):
    result, predicted_y = check_prediction(i, list_x[i], list_y[i])
    print(f"Row {i + 1} Prediction: {predicted_y} - {result}")