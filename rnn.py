import tensorflow as tf
from tensorflow.contrib import rnn
# no. of frames in a question video
time_steps = 40
# No. of features for every frame
input_size = 378
# Lstm units in a Lstm cell
num_units = 113
learning_rate = 0.001
# binary classification
n_classes = 2
# 113 frames from 300-303, input all frames for now
batch_size = 113
#batch_size = num_units for matrix mul
# weights
out_weights = tf.Variable(tf.random_normal([batch_size, n_classes]))
# bias
out_bias = tf.Variable(tf.random_normal([n_classes]))

# input image
x = tf.placeholder("float", [None, time_steps, input_size])
# input label
y = tf.placeholder("float", [None, n_classes])
# convert input[batch_size, time_steps, input_size] to [batch_size, input_size]
input = tf.unstack(x, time_steps, 1)
# one layer of lstm
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias = 1)
outputs, _  = rnn.static_rnn(lstm_layer, input, dtype = "float32")
# output of lstm = [batch_size, num_units], convert to [batch_size, num_class]
prediction = tf.matmul(outputs[-1], out_weights)+out_bias
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
# compare the predicted and actual labels
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
filename_queue = tf.train.string_input_producer(["./dataset/USC/frames/frames/300_P1.csv"])
reader = tf.TextLineReader(skip_header_lines = 1)
key, value = reader.read(filename_queue)
record_defaults = [[""],[""],[0]]
for i in range(0, 379):
    record_defaults.append([0.5])
# print (record_defaults)
all_columns = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.pack()
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     iter = 1
#     while iter<100:
        
#     iter += 1
