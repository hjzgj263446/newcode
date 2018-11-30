import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle
import time


def g(fake_imgs,keep_prob,reuse=False):
    with tf.variable_scope('g',reuse=reuse):
        w1 = tf.get_variable('w1',shape=[fake_imgs.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,200],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,200],initializer=tf.zeros_initializer)
        h1 = tf.maximum(0.01*(tf.matmul(fake_imgs,w1)+b1),tf.matmul(fake_imgs,w1)+b1)
        h1_drop = tf.nn.dropout(h1,keep_prob)
        l2 = tf.matmul(h1_drop,w2)+b2
        h2 = tf.nn.relu(l2)
    return h2

def g2(fake_imgs,keep_prob,reuse=False):
    with tf.variable_scope('g2',reuse=reuse):
        w1 = tf.get_variable('w1',shape=[fake_imgs.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,300],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,300],initializer=tf.zeros_initializer)
        h1 = tf.maximum(0.01*(tf.matmul(fake_imgs,w1)+b1),tf.matmul(fake_imgs,w1)+b1)
        h1_drop = tf.nn.dropout(h1,keep_prob)
        l2 = tf.matmul(h1_drop,w2)+b2
        h2 = tf.tanh(l2)
    return h2

def d(raf_img,reuse=False):
    with tf.variable_scope('d',reuse=reuse):
        w1 = tf.get_variable('w1',shape=[raf_img.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,1],initializer=tf.zeros_initializer)
        h3 = tf.maximum(0.01*(tf.matmul(raf_img,w1)+b1),tf.matmul(raf_img,w1)+b1)
        l3 = tf.matmul(h3, w2) + b2
    return l3

def d2(raf_img,reuse=False):
    with tf.variable_scope('d2',reuse=reuse):
        w1 = tf.get_variable('w1',shape=[raf_img.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,1],initializer=tf.zeros_initializer)
        h3 = tf.maximum(0.01*(tf.matmul(raf_img,w1)+b1),tf.matmul(raf_img,w1)+b1)
        l3 = tf.matmul(h3, w2) + b2
        h3 = tf.sigmoid(l3)
    return h3

C = []
batch_size = 50
learn_rate = 1e-4
echo = 50000
t1 = time.time()
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)
real_img = tf.placeholder(tf.float32,shape=[batch_size,28*28])
fake_img = tf.placeholder(tf.float32,shape=[batch_size,28*28])
keep_prob = tf.placeholder(tf.float32)

real_g = g(real_img,keep_prob)
fake_g = g(fake_img,keep_prob,True)

real_d2 = d2(real_g)
fake_d2 = d2(fake_g,True)

fake_g2 = g2(fake_g,keep_prob)
real_d = d(real_img)
fake_d = d(fake_g2,True)

loss_d = -tf.reduce_mean(tf.log(real_d)+tf.log(1-fake_d))
loss_d2 = -tf.reduce_mean(tf.log(real_d2)+tf.log(1-fake_d2))
loss_g2 = -tf.reduce_mean(tf.log(fake_d))
loss_g = -tf.reduce_mean(tf.log(fake_d2))

var_g2 = [var for var in tf.trainable_variables() if var.name.startswith('g2')]
var_d2 = [var for var in tf.trainable_variables() if var.name.startswith('d2')]
var_g = [var for var in tf.trainable_variables() if var.name.startswith('g')]
var_d = [var for var in tf.trainable_variables() if var.name.startswith('d')]
#d1_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in var_d]
train_g = tf.train.AdamOptimizer(learn_rate).minimize(loss_g,var_list=var_g)
train_g2 = tf.train.AdamOptimizer(learn_rate).minimize(loss_g2,var_list=var_g2)
train_d = tf.train.AdamOptimizer(learn_rate).minimize(loss_d,var_list=var_d)
train_d2 = tf.train.AdamOptimizer(learn_rate).minimize(loss_d2,var_list=var_d2)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(echo+1):
        batch,y_label = mnist.train.next_batch(batch_size)
        batch = batch*2-1
        g_z = np.random.uniform(-1, 1, size=(batch_size, 28*28))
        _,_,lossd,lossg2 = sess.run([train_d,train_g2,loss_d,loss_g2],feed_dict={fake_img:g_z,real_img:batch,keep_prob:0.5})
        for j in range(1):
            _,_,lossg,lossd2 = sess.run([train_g,train_d2,loss_g,loss_d2],feed_dict={real_img:batch,fake_img:g_z,keep_prob:0.5})
        if i % 1000 == 0:
            print("epoch:%d"%i,"loss_d:%s"%lossd,"loss_g:%s"%lossg,"loss_d2:%s"%lossd2,"loss_g2:%s"%lossg2)
            g1 = sess.run(fake_g2, feed_dict={fake_img: g_z,keep_prob:1.0})
            with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+'sampleswgan.pkl', 'wb') as f:
                pickle.dump(g1, f)
    g1 = sess.run(fake_g,feed_dict={fake_img:g_z,keep_prob:1.0})
    print(g1.shape)
    t2 = time.time()
    print(t2-t1)
    fake = g1[:1,:].reshape([28,28])
    plt.imshow(fake)
    fake = g1[1:2,:].reshape([28,28])
    plt.imshow(fake)
    fake = g1[2:3,:].reshape([28,28])
    plt.imshow(fake)
    plt.show()