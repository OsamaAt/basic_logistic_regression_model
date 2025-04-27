# # Initialize dataset
import numpy as np
x_train = np.array([1, 2, 3, 4, 5, 6])
y_train = np.array([0, 0, 0, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Add bias term (x_b)
x_train = x_train.reshape(-1, 1)  # Reshape to column vector
x_b = np.c_[np.ones((x_train.shape[0], 1)), x_train]  # Add bias term

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (log loss)
def cost_function(x, y, theta):
    m = len(y)
    predictions = sigmoid(x.dot(theta))
    cost = (-1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

# Gradient descent
def gradient_descent(x, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history=[]
    for _ in range(iterations):
        predictions = sigmoid(x.dot(theta))
        errors = predictions - y
        gradients = (1/m) * x.T.dot(errors)
        theta -= learning_rate * gradients
        cost_history.append(cost_function(x,y,theta))
    return theta , cost_history

# Training parameters
theta = np.zeros((x_b.shape[1], 1))
learning_rate = 0.1
iterations = 1000

# Train model
theta , cost = gradient_descent(x_b, y_train.reshape(-1, 1), theta, learning_rate, iterations)

print(f"Trained Parameters: {theta.flatten()}")
print(f'Final Cost: {cost[-1]} ')

# Predict probability for 3.5 hours of study
x_test = np.array([1, 3.5])  # 1 for bias, 3.5 study hours
probability = sigmoid(np.dot(x_test, theta))
if probability >= 0.5:
    print(f"Prediction: Pass ✅{probability}")
else:
    print(f"Prediction: Fail ❌{probability}")
