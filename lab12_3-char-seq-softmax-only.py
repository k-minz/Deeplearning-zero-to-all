# Lab 12 Character Sequence Softmax only
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility
#sess = tf.InteractiveSession()
#import pprint
#pp = pprint.PrettyPrinter(indent=4)

sample = " if you want you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

'''
#결과값 y들을 각각 softmax취해주면 메모리와 시간 측면에서 비효율. 
따라서 한꺼번에 묶어주고 softmax한번 취해주고 결과값을 다시 펼쳐주는 단계 
-> 데이터가 과정속에서 섞이지 않음.
X_for softmax = tf.reshape(outputs, [-1, hidden_size]) 
    #output의 dimension에만 맞게 알아서 쌓아라
outputs= tf.reshape(outputs,[batch_size], seq_length, num_classes]) 
    #output처럼 다시 펼쳐라
'''

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

# flatten the data (ignore batches for now). No effect if the batch size is 1
X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
X_for_softmax = tf.reshape(X_one_hot, [-1, rnn_hidden_size]) #softmax를 위해 reshape된 input

### softmax layer (rnn_hidden_size -> num_classes)
softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, num_classes]) #사실 같은 수. rnn_hidden_size = input, num_classes = 예측하고자 하는 것의 class
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b #softmax output

# expend the data (revive the batches)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes]) #RNN의 output과 같은 shape 
weights = tf.ones([batch_size, sequence_length])

# Compute sequence cost/loss
'''
사실 RNN을 통해 갓 나온 y값은 activation function이 적용되어 있어서 loss를 계산하는 데 부적합. 하지만 softmax layer를 통해 activation function이 적용되지 않고 softmax 취해진 결과값을 loss계산에 적용하는 게 적절. 좋은 성능.
'''
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights) #softmax function
loss = tf.reduce_mean(sequence_loss)  # mean all sequence loss
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", ''.join(result_str)) #output으로 나온 문자 출력

'''
0 loss: 2.29513 Prediction: yu yny y y oyny
1 loss: 2.10156 Prediction: yu ynu y y oynu
2 loss: 1.92344 Prediction: yu you y u  you

..

2997 loss: 0.277323 Prediction: yf you yant you
2998 loss: 0.277323 Prediction: yf you yant you
2999 loss: 0.277323 Prediction: yf you yant you
'''
#%%
# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: x_data})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')
