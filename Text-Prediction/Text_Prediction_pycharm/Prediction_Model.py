import tensorflow as tf
import nltk
import collections
import numpy as np
from tensorflow.contrib import rnn

TEXT_FILE_NAME= 'atom_text'
NUM_WORDS_FOR_PREDICTION =3  # no. of words required for prediction

# making a dictionary of all words in text file taking common words just once
def embed_words(word_list):
    vocab = collections.Counter(word_list).most_common()
    vocab_dict = dict()
    for word, _ in vocab:
        vocab_dict[word] = len(vocab_dict)  # assigning keys to words
    reverse_vocab_dict = dict(zip(vocab_dict.values(),vocab_dict.keys())) # assigning words to keys
    return vocab_dict, reverse_vocab_dict


# building dataset using 3 words at a time and using next words as labels
def build_data_set(word_list, vocab_dict):
    X = []
    Y = []
    sample = []
    for index in range(0, len(word_list)- NUM_WORDS_FOR_PREDICTION):
        for i in range(0, NUM_WORDS_FOR_PREDICTION):
            sample.append((vocab_dict[word_list[index + i]]))
            if (i+1) % NUM_WORDS_FOR_PREDICTION == 0:
                X.append(sample)
                Y.append(vocab_dict[word_list[index + i + 1]])
                sample =[]
    return X, Y


def interpret_results(results):
    list_of_predictions =[]
    for result in results:
        max_index = np.argmax(result, 0)
        predicted_word = reverse_vocab_dict[max_index]
        list_of_predictions.append(predicted_word)
    return list_of_predictions


# creating RNN model
# takes input and outputs the output after running input through the network
def RNN(x_input):
    # flatten input for better analysis
    x = tf.unstack(x_input, NUM_WORDS_FOR_PREDICTION, 1)
    # create lstm cell basic unit of RNN - there are 512 cells in our network
    lstm_cell = rnn.BasicLSTMCell(512)
    # creates a network made of basic lstm cells(dense layer)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    weights = tf.Variable(tf.random_normal([512, len(vocab_dict)]))
    biases = tf.Variable(tf.random_normal([ len(vocab_dict)]))
    op = tf.matmul(outputs[-1], weights) + biases
    return op

# read text file
with open(TEXT_FILE_NAME) as f:
    text = f.read()

# turn text into array strings
word_list = nltk.tokenize.word_tokenize(text)
# create vocab dictionary
vocab_dict, reverse_vocab_dict = embed_words(word_list)

# creating training testing datasets
x_train, y_train = build_data_set(word_list, vocab_dict)
#test_string = 'element liquid neutral billiard nucleus protons '
#tokenized_test_string = nltk.tokenize.word_tokenize(test_string)
#x_test, y_test = build_data_set(tokenized_test_string, vocab_dict)

# inputs
x_input = tf.placeholder(tf.float32, [None, NUM_WORDS_FOR_PREDICTION, 1], 'x_input')
y_input = tf.placeholder(tf.float32, [None, len(vocab_dict)])

# outputs
logits = RNN(x_input)
y_output = tf.nn.softmax(logits, name='y_output')

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=logits))
train_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# saving model
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 10
    batch_size = 20

    tf.train.write_graph(sess.graph_def,'./saved_model/', 'text_predictor.pbtxt', False)
    for i in range(epochs):  # dividing training data into batches
        last_batch = len(x_train) % batch_size
        num_train_steps = (len(x_train)/ batch_size) + 1
        for step in range(int(num_train_steps)):   # running training on individual batch
            x_batch = x_train[(step * batch_size): ((step + 1) * batch_size)]
            y_batch = y_train[(step * batch_size): ((step + 1) * batch_size)]
            y_batch_encoded =[]

            for y in y_batch:
                one_hot = np.zeros([len(vocab_dict)], dtype= float)
                one_hot[y] = 1.0
                y_batch_encoded = np.concatenate((y_batch_encoded, one_hot))

            x_batch = np.array(x_batch)
            y_batch_encoded = np.array(y_batch_encoded)

            if len(x_batch) < batch_size:
                x_batch = x_batch.reshape(last_batch, NUM_WORDS_FOR_PREDICTION, 1)
                y_batch_encoded = y_batch_encoded.reshape(last_batch, len(vocab_dict))
            else:
                x_batch = x_batch.reshape(batch_size, NUM_WORDS_FOR_PREDICTION, 1)
                y_batch_encoded = y_batch_encoded.reshape(batch_size, len(vocab_dict))

            _, acc, loss, prediction = sess.run([train_step, accuracy, loss_op, y_output],
                                                feed_dict={x_input: x_batch, y_input: y_batch_encoded})
            print("Step: " + str(i) + ", loss: {:.4f}".format(loss) + ", accuracy: {:.2f}".format(acc * 100))
            print("Prediction: ", interpret_results(prediction))

# checkpoint file
    saver.save(sess, './saved_model/text_predictor.ckpt')



