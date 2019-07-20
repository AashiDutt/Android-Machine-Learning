import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

freeze_graph.freeze_graph(input_graph='./saved_model/text_predictor.pbtxt',
                          input_saver='',
                          input_binary=True,
                          input_checkpoint='./saved_model/text_predictor.ckpt',
                          output_node_names='y_output',
                          restore_op_name='save/restore_all',
                          filename_tensor_name='save/Const:0',
                          output_graph='frozen_text_predictor.pb',
                          clear_devices=True,
                          initializer_nodes='')


input_graph_def = tf.GraphDef()
with tf.gfile.Open('frozen_text_predictor.pb', 'rb') as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def,
                                                                     ['x_input'],
                                                                     ['y_output'],
                                                                     tf.float32.as_datatype_enum)
f = tf.gfile.FastGFile('optimized_text_predictor.pb', 'w')
f.write(output_graph_def.SerializeToString())
