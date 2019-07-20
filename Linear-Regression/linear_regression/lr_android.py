# code for linear regression model for android application
# y = Wx+b

import tensorflow as tf

x_train = [1.0, 2.0, 3.0, 4.0]
y_train =[-1.0, -2.0, -3.0, -4.0]

# graph construction
# weight and bias
W = tf.Variable([1.0], tf.float32, name='W')
b = tf.Variable([1.0], tf.float32, name='b')

# input x as placeholder
x = tf.placeholder(tf.float32,name='x')
y_input = tf.placeholder(tf.float32,name='y_input')

#y_output = W * x +b
y_output = tf.add(x=tf.multiply(W,x,'multiply'),y=b,name='y_output')

# loss function
loss = tf.reduce_sum(tf.square(y_output-y_input),name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01,name ='optimizer')
train_step = optimizer.minimize(loss,name ='train_step')
saver = tf.train.Saver()

session = tf.Session()
session.run(tf.global_variables_initializer())

tf.train.write_graph(graph_or_graph_def=session.graph_def,logdir='.',name ='lr_android.pbtxt',as_text=False)
# total loss before training
print(session.run(loss,feed_dict={x: x_train,y_input: y_train}))

# training phase
for _ in range(1000):
    session.run(train_step, feed_dict={x:x_train,y_input:y_train})

#checkpoint file
saver.save(sess=session,save_path='lr_android.ckpt')
# total loss and modified weight and bias after training
print(session.run([loss,W,b],feed_dict={x:x_train,y_input:y_train}))

# test the model on new values
print(session.run(y_output,feed_dict={x:[5.0,10.0,15.0]}))