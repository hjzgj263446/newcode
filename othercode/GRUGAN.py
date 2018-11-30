import tensorflow as tf
import numpy as np

x = (np.random.rand(300))
y = np.sin(x)

x1 = tf.placeholder(tf.float32,shape=[300,1])
y1 = tf.placeholder(tf.float32,shape=[1])
w1 = tf.Variable(tf.truncated_normal([1,10],stddev=0.1),dtype=tf.float32)
b1 = tf.Variable(tf.zeros([1,10]),tf.float32)
w2 = tf.Variable(tf.truncated_normal([10,1],stddev=0.1),dtype=tf.float32)
b2 = tf.Variable(tf.zeros([1]))

h = tf.matmul(x,w1)+b1

lo = tf.reduce_mean(tf.square(h-y))
tra = tf.train.AdamOptimizer(1e-4).minimize(lo)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(10000):
        tro,l = sess.run([tra,lo],feed_dict={x1:x,y1:y})
        if i %100 == 0:
            print(l)