'''
<데이터 입력 자동화 단계>
기존에 직접 데이터를 넣고, one-hot encoding 한 것과 달리 이번엔 자동으로 해보자
'''
# Lab 12 Character Sequence RNN
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility
#import pprint
#pp = pprint.PrettyPrinter(indent=4)


sample = "if you want you"
idx2char = list(set(sample))  # index -> char 유니크한 charcter를 뽑아내고 index화
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex to make one-hot encoding

#input dim, output dim, sequence length, batch size
# hyper parameters #input, output 데이터를 one hot 으로 표현하기 때문에 len = 5
dic_size = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  # char to index 샘플 데이터를 자동으로 index화
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell 처음부터 마지막 한개까지
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello 두번째부터 마지막까지

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label


x_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0 dimension 주의
    #shape: None * 22 * 13
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_size, state_is_tuple=True) #num_units = dictionary size = hidden size
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)
#output:[batch_size * sequence_length * hidden_size] = [1*22*13]
#_state : [1*13]


# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size]) #[1*22*13] -> [22*13]
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes]) #[1*22*13]

weights = tf.ones([batch_size, sequence_length]) #모두 1. shape=[1*22] 각 sequence별 하나의 w
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights) #outputs = [0.01, 0.002, 0.012, 0.837] -> 3번째 정답
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2) #dim = [1, 22]. outputs = [0.01, 0.002, 0.012, 0.837] 최대가 되는 x out
#%%
#Training and Results

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        #x_data, y_data를 넘겨주면서 train을 시켜주고 그때마다 loss를 보고
        result = sess.run(prediction, feed_dict={X: x_data})
        #그때마다 prediction을 해서 결과값을 출력해내
        '''
        result:
        [[6 6 0 7 2 6 0 1 9 8 6 0 7 2]]
        [[6 6 0 7 2 6 0 1 9 8 6 0 7 2]]
        [[5 6 0 7 2 6 4 1 9 8 6 0 7 2]]
        [[5 6 0 7 2 6 4 1 9 8 6 0 7 2]]
        '''
        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        '''
        np.squeeze: Remove single-dimensional entries from the shape of an array. [1, 3, 1]->[3,]
        np.squeeze(result)
        array([5, 6, 0, 7, 2, 6, 4, 1, 9, 8, 6, 0, 7, 2], dtype=int64)
        
        print(result_str)
        ['f', ' ', 'y', 'o', 'u', ' ', 'w', 'a', 'n', 't', ' ', 'y', 'o', 'u']
        '''

        print(i, "loss:", l, "Prediction:", ''.join(result_str))

#%%
'''
0 loss: 2.35377 Prediction: uuuuuuuuuuuuuuu
1 loss: 2.21383 Prediction: yy you y    you
2 loss: 2.04317 Prediction: yy yoo       ou
3 loss: 1.85869 Prediction: yy  ou      uou
4 loss: 1.65096 Prediction: yy you  a   you
5 loss: 1.40243 Prediction: yy you yan  you
6 loss: 1.12986 Prediction: yy you wann you
7 loss: 0.907699 Prediction: yy you want you
8 loss: 0.687401 Prediction: yf you want you
9 loss: 0.508868 Prediction: yf you want you
10 loss: 0.379423 Prediction: yf you want you
11 loss: 0.282956 Prediction: if you want you
12 loss: 0.208561 Prediction: if you want you

...

'''
