from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Define the input data x and the corresponding outputs y for the first 5 rows
x_data = np.array([
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
])

y_data = np.array([
    [1,1,0,1,0,0,1,0,1,1],
    [1,0,1,0,0,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,1,1,0,1,0],
    [1,0,0,0,0,0,0,1,1,1]
])

# Train a decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(x_data.reshape(-1, 10), y_data.reshape(-1, 10))

# Predict y values using the trained model
predicted_y = model.predict(x_data.reshape(-1, 10))

# Convert predictions to the same shape as y
predicted_y = predicted_y.reshape(-1, 10)

# Check if the predicted y matches the actual y
correct_predictions = np.all(predicted_y == y_data, axis=1)
for idx, correct in enumerate(correct_predictions):
    print(f"Row {idx+1}, Predicted: {predicted_y[idx]}, Actual: {y_data[idx]}, Result: {'Success' if correct else 'Failure'}")