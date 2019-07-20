# freezing graph
# and write optimized version to import to android studio

import tensorflow as tf
from tensorflow.python.tools import freeze_graph,optimize_for_inference_lib

freeze_graph.freeze_graph(input_graph='mnist_model.pbtxt',input_saver='',input_binary=True,input_checkpoint='./saved_model/mnist_model.ckpt',output_node_names='y_actual',restore_op_name='save/restore_all',filename_tensor_name='save/Const:0',output_graph='frozen_mnist_model.pb',clear_devices=True,initializer_nodes='')

# creating optimized version

input_graph_def= tf.GraphDef()
with tf.gfile.Open('frozen_mnist_model.pb','rb') as f:
    data=f.read()
    input_graph_def.ParseFromString(data)  # reading data and converting it into a string

output_graph_def= optimize_for_inference_lib.optimize_for_inference(input_graph_def=input_graph_def, input_node_names=['x_input'], output_node_names=['y_actual'], placeholder_type_enum=tf.float32.as_datatype_enum)

f = tf.gfile.FastGFile(name='optimized_frozen_mnist_model.pb', mode='w')
f.write(file_content=output_graph_def.SerializeToString())