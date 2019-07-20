import tensorflow as tf
import numpy as np

# ------------------ Built-In Estimator Function ---------------------
print(" \n-------------------------------------------------\n ")
# Sample Input/Training Data
X_train = np.asarray([1.0, 2.0, 3.0, 4.0])
y_train = np.asarray([-1.0, -2.0, -3.0, -4.0])

# Sample Evaluation/Test Data
X_eval = np.asarray([5.0, 10.0, 15.0, 20.0])
y_eval = np.asarray([-5.0, -10.0, -15.0, -20.0])

# Test values to make prediction on
X_predict = np.asarray([50.0, 100.0])

feature_column = tf.feature_column.numeric_column(key="x", shape=[1], dtype=tf.float32)
feature_columns = [feature_column]

# Define the Linear Regression Estimator Model
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# Data to train the model
input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_train},
                                              y=y_train,
                                              batch_size=4,
                                              num_epochs=None,
                                              shuffle=True)

# Data to validate the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_train},
                                              y=y_train,
                                              batch_size=4,
                                              num_epochs=1000,
                                              shuffle=True)

# Data to test the trained model
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_eval},
                                              y=y_eval,
                                              batch_size=4,
                                              num_epochs=1000,
                                              shuffle=False)

# Data to Make prediction on
predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_predict},
                                              num_epochs=1,
                                              shuffle=False)

# Train the Estimator Model
estimator.train(input_fn=input_fn,
                steps=1000)

# Validation
print("Validation: ", estimator.evaluate(input_fn=train_input_fn))

# Testing
print("Test: ", estimator.evaluate(input_fn=eval_input_fn))

# Prediction
print("Predictions: ", list(estimator.predict(input_fn=predict_input_fn)))

print(" \n-------------------------------------------------\n ")


# ------------------ Custom Estimator Function ---------------------
# Define the Custom Linear Regression Estimator Model
# y = W * x + b

def model_fn(features, labels, mode):
    # If model in prediction mode, provide a default value for label
    if mode == 'infer':
        labels = np.asarray([0,0])

    # Initialize Weights
    W = tf.get_variable(name="W", shape=[1], dtype=tf.float64)
    # Initialize Bias
    b = tf.get_variable(name="b", shape=[1], dtype=tf.float64)
    # Define Linear Regression Equation
    y_hat = W * features["x"] + b
    # Loss Function
    loss = tf.reduce_sum(input_tensor=tf.square(x=(y_hat - labels)))
    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    # Progress to next step for training and testing
    global_step = tf.train.get_global_step()
    # Training Step
    train_step = tf.group(optimizer.minimize(loss=loss), tf.assign_add(global_step, 1))
    # Return the estimator
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=y_hat,
                                      loss=loss,
                                      train_op=train_step)


# Custom Estimator
custom_estimator = tf.estimator.Estimator(model_fn=model_fn)

# Train the Custom Estimator Model
custom_estimator.train(input_fn=input_fn,
                       steps=1000)

# Validation
print("Validation (Custom Estimator): ", custom_estimator.evaluate(input_fn=train_input_fn))

# Testing
print("Test (Custom Estimator): ", custom_estimator.evaluate(input_fn=eval_input_fn))

# Prediction
print("Predictions (Custom Estimator): ", list(custom_estimator.predict(input_fn=predict_input_fn)))

print(" \n-------------------------------------------------\n ")