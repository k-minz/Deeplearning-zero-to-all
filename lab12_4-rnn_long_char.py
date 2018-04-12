'''
1) logit
2) RNN의 depth
문제로 인해 잘 작동이 안되는 case
--> 해결해보자
'''
from __future__ import print_function #파이썬2와 3 호환을 위해

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)  # reproducibility

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10  # Any arbitrary number 임의로 지정
learning_rate = 0.1

dataX = []
dataY = []
#문장이 너무 기니까 잘라가면서 x, y data를 뽑아냄-> window를 옮겨가면서 정해진 sequence만큼 잘라
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length] #10 길이로 문장들이 겹쳐서 생김
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX) #180 - 10 = 170 문장이 많아 졌으니 batch size가 많아져버림

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)  # check out the shape[?, 10, 25]


# Make a lstm cell with hidden_size (each unit output vector size)
# Vanishing gradient or exploding을 방지하과 LSTM  or GRM 사용
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

#outputs.shape = (?, 10, 25)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size]) #(?, 25)
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None) #softmax

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes]) #170*10*25

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500): #500*170
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)

# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')

'''
0 167 tttttttttt 3.23111
0 168 tttttttttt 3.23111
0 169 tttttttttt 3.23111
…
499 167  of the se 0.229616
499 168 tf the sea 0.229616
499 169   the sea. 0.229616

g you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.

'''


#stacked RNN
#make a lstm cell with hidden_size(each unit output vector size)
cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
cell = rnn.MultiRNNCell([cell] * 2, state_is_tuple=True) #100층을 원하면 *100

#결과값 y들을 각각 softmax취해주면 메모리와 시간 측면에서 비효율. 따라서 한꺼번에 묶어주고 softmax한번 취해주고 결과값을 다시 펼쳐주는 단계 - 데이터가 과정속에서 섞이지 않음.
X_for softmax = tf.reshape(outputs, [-1, hidden_size]) #output의 dimension에만 맞게 알아서 쌓아라
outputs= tf.reshape(outputs,[batch_size], seq_length, num_classes]) #output처럼 다시 펼쳐라
