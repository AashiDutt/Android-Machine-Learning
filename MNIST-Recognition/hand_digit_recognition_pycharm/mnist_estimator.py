import tensorflow as tf
import numpy as np

# getting dataset and training ,Testing data
mnist_data = tf.contrib.learn.datasets.load_dataset('mnist')
x_train = mnist_data.train.images
y_train = np.asarray(mnist_data.train.labels, dtype=np.int32)
x_eval = mnist_data.test.images
y_eval = np.asarray(mnist_data.test.labels, dtype=np.int32)

x_predict =x_eval[:1]


# creating linear regression model y = Wx+b
def model_fn(features, labels, mode):
    x = tf.reshape(features['x'],[-1,784])
    W = tf.get_variable(name='W', shape=[784, 10], dtype=tf.float32)
    b = tf.get_variable(name='b', shape=[10], dtype=tf.float32)
    y = tf.add(tf.matmul(x, W), b)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=tf.nn.softmax(logits=y))

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=y)

#training
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_step = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_step)

#testing and evaluation
    eval_metric_ops={'accuracy':tf.metrics.accuracy(labels=labels,predictions=tf.argmax(y, 1))}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)





estimator= tf.estimator.Estimator(model_fn= model_fn)

train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':x_train}, y=y_train, batch_size=100, num_epochs=None, shuffle=True)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':x_eval}, y=y_eval, num_epochs=1, shuffle=False)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':x_predict},  num_epochs=1, shuffle=False)

estimator.train(input_fn=train_input_fn, steps=20000)

print(estimator.evaluate(input_fn=eval_input_fn))
print(list(estimator.predict(input_fn=predict_input_fn)))
