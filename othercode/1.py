import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)
x = tf.placeholder(tf.float32,shape = [None,784])
y_ = tf.placeholder(tf.float32,shape = [None,10])
keep_drop = tf.placeholder(tf.float32)
w1 = tf.Variable(tf.truncated_normal([784,1280],stddev=0.1))
b1 = tf.Variable(tf.zeros([1,1280]))
w2 = tf.Variable(tf.zeros([1280,10]))
b2 = tf.Variable(tf.zeros([1,10]))
temp = tf.nn.relu(tf.matmul(x,w1)+b1)
h1_drop = tf.nn.dropout(temp,keep_drop)
y = tf.nn.softmax(tf.matmul(temp,w2)+b2)
loss = -tf.reduce_mean(y_*tf.log(y))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30000):
        x1,y1 = mnist.train.next_batch(50)
        _,lo = sess.run([train,loss],feed_dict={x:x1,y_:y1,keep_drop:1})
        if i%3000 == 0:
            t_loss = sess.run(loss,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_drop:0.5})
            print(i,lo,t_loss)
            acc = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
            acc1 = tf.reduce_mean(tf.cast(acc, tf.float32))
            print("test",acc1.eval({x: mnist.test.images, y_: mnist.test.labels, keep_drop: 1}))
            print("tairn",acc1.eval({x:x1,y_:y1,keep_drop:1}))
            with open(r'C:\Myprogram\mnistdata'+'\\'+"111"+'advsampel.pkl', 'rb') as f:
                sample = pickle.load(f)
                with open(r'C:\Myprogram\mnistdata' + '\\' + "111" + 'advlabel.pkl', 'rb') as f:
                    label = pickle.load(f)
                    print("adv", acc1.eval({x: sample, y_: label, keep_drop: 1.0}))
            with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+'sampleswgan.pkl', 'rb') as f:
                sample = pickle.load(f)
                with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+'y_label.pkl', 'rb') as f:
                    label = pickle.load(f)
                    print("shengc", acc1.eval({x: sample, y_: label, keep_drop: 1.0}))
    print("test", acc1.eval({x: mnist.test.images, y_: mnist.test.labels, keep_drop: 1.0}))


