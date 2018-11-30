import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle



mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)
x1 = mnist.test.images
y1 = mnist.test.labels
x = tf.placeholder(tf.float32,shape = [None,784])
y_ = tf.placeholder(tf.float32,shape = [None,10])
keep_drop = tf.placeholder(tf.float32)
w1 = tf.Variable(tf.truncated_normal([784,128],stddev=0.1))
b1 = tf.Variable(tf.zeros([1,128]))
w2 = tf.Variable(tf.zeros([128,10]))
b2 = tf.Variable(tf.zeros([1,10]))
temp = tf.nn.relu(tf.matmul(x,w1)+b1)
h1_drop = tf.nn.dropout(temp,keep_drop)
y = tf.nn.softmax(tf.matmul(temp,w2)+b2)
loss = -tf.reduce_mean(y_*tf.log(y))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)
grad = tf.gradients(ys = loss,xs = x)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _ = sess.run(train,feed_dict={x:x1,y_:y1,keep_drop:1})
    g = sess.run(grad,feed_dict={x:x1,y_:y1,keep_drop:1})
    print(g)
    with open (r'C:\Myprogram\mnistdata'+'\\'+"111"+'advsampel.pkl', 'wb') as f:
        pickle.dump((0.2*np.sign(g)+x1).reshape((10000,784)),f)
    with open(r'C:\Myprogram\mnistdata' + '\\' + "111" + 'advlabel.pkl', 'wb') as f:
        pickle.dump(y1,f)