'''
lab10_01_mnist_softmax.py
    single Weight and bias

Accuracy: 0.9035
'''
import numpy as np
import tensorflow as tf
import random
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
#%%
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# parameter
learning_rate = 0.001
training_epochs = 15
batch_size = 100 #전체 데이터에서 100개 씩만 가져와서 train(다 가져오면 메모리 낭비)
# input place holders & Weight, bias
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))
#
#hypothesis
hypothesis = tf.matmul(X, W) + b #predicted y
#loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
#Gradient_cost가 최저일 때의 최적의 parameter 찾기
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#train the model
for epoch in range(training_epochs): #15번(epoch) 반복된다
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size) #550 = 55000/100

    for i in range(total_batch): #1 epoch
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) #batch size만큼 데이터 불러옴(알아서 다음꺼)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict) #train은 train data로
        avg_cost += c / total_batch #average cost #cost 점점 감소

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
                    #4숫자                            #소수9자리
print('Learning Finished')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy,
      feed_dict={X: mnist.test.images, Y: mnist.test.labels})) #accuracy는 test로
#print('Accuracy', accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1) #0부터 mnist.test.num_examples-1개 중 한 수
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r: r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r: r + 1]}))

#
plt.imshow(mnist.test.images[r:r + 1].
          reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
#%%
'''
lab10_02_mnist_nn.py
    Multiple Weight and bias
    
Accuracy: 0.9455
'''
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)
#
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

#parameter
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

#Weight and bias for nn layers
W1 = tf.Variable(tf.random_normal([784, 256]), name='Weight1')
b1 = tf.Variable(tf.random_normal([256]), name='bias1')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]), name='Weight2')
b2 = tf.Variable(tf.random_normal([256]), name='bias2')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 10]), name='Weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
hypothesis = tf.matmul(L2, W3) + b3

#Define cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#train the model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for batch in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost:', '{:.9f}'.format(avg_cost) )

print('Learning Finished')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run([accuracy], feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

r = random.randint(0, mnist.test.num_examples - 1)
print('Label: ', sess.run(tf.argmax(mnist.test.labels[[r]], 1)))
print('Prediction: ', sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[[r]]}))

plt.imshow(mnist.test.images[[r]].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()

#%%
'''
Lab10_03 MNIST and Xavier(initalize method)

Accuracy: 0.9783
'''
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

#Weight & bias for nn layers
#get_variable이 initializer 설정 가능
W1 = tf.get_variable('W1', shape=[784, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)


W2 = tf.get_variable('W2', shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable('W3', shape=[256, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3

'''
lab_10_04_mnist_nn_deep.py
Multiple Weight and bias using xavier_initializer

Accuracy: 0.9742

Accuracy doesn't predict well even though it has more deep layers
'''
# X = tf.placeholder(tf.float32, [None, 784])
# Y = tf.placeholder(tf.float32, [None, 10])
#
# W1 = tf.get_variable("W1", shape=[784, 512],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b1 = tf.Variable(tf.random_normal([512]))
# L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
#
# W2 = tf.get_variable("W2", shape=[512, 512],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b2 = tf.Variable(tf.random_normal([512]))
# L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
#
# W3 = tf.get_variable("W3", shape=[512, 512],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b3 = tf.Variable(tf.random_normal([512]))
# L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
#
# W4 = tf.get_variable("W4", shape=[512, 512],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b4 = tf.Variable(tf.random_normal([512]))
# L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
#
# W5 = tf.get_variable("W5", shape=[512, 10],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b5 = tf.Variable(tf.random_normal([10]))
# hypothesis = tf.matmul(L4, W5) + b5

# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)

#initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
    X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()
#%%
'''
lab_10_05_mnint_nn_dropout.py
Multiple Weight and bias using xavier_initializer
and do dropout & ensemble

Accuracy: 0.9742
'''
# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7} #train에서는 node들의 70%만을 남겨둠
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})) #sess.run할 때 test시에는 모두 사용하니까(ensemble) prob: 1

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))
#sess.run할 때 test하면 모든 노드 사용: 1

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

