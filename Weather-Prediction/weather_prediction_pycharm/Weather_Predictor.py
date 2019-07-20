import tensorflow as tf
from Data_Retriever import build_data_subset

def measure_accuracy(actual, expected):
    num_correct =0
    for i in range(len(actual)):
        actual_value =actual[i]
        expected_value = expected[i]
        if actual_value[0] >= actual_value[1] and expected_value[0] >= expected_value[1]:
            num_correct += 1
        elif actual_value[0] <= actual_value[1] and expected_value[0] <= expected_value[1]:
            num_correct += 1

        return (num_correct /len(actual))* 100



input_shape =4

x_train, y_train =build_data_subset('weather_2019.csv', 1, 37)  # skip 1st row first 130 datapoints for trainin
x_test, y_test =build_data_subset('weather_2019.csv', 38, 7 ) # skip first 138 rows and get rest 66 datapoints

#y =Wx+b  computational graph
x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_shape], name='x_input')
y_input = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y_input')

W = tf.Variable(initial_value=tf.ones(shape=[input_shape, 2]), name='W')
b = tf.Variable(initial_value=tf.ones(shape=[2]), name='b')

y_output= tf.add(tf.matmul(x_input,W), b, name='y_output')

# calculating loss and minimizing loss with optimizer
loss = tf.reduce_sum(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_output)))
optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)

# saving file
saver = tf.train.Saver()

session=tf.Session()
session.run(tf.global_variables_initializer())

# training and testing
tf.train.write_graph(session.graph_def, './save_model/', 'weather_prediction.pbtxt', False)

for _ in range(10000):
    session.run(optimizer, feed_dict={x_input:x_train, y_input:y_train})

# saving checkpoint after training and testing
saver.save(session, './save_model/weather_prediction.ckpt')

# print accuracy for training and testing
print(measure_accuracy(session.run(y_output, feed_dict={x_input:x_train}), y_train))

print(measure_accuracy(session.run(y_output, feed_dict={x_input:x_test}), y_test))
