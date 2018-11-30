import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle
import time


def g(fake_imgs,y,keep_prob):
    with tf.variable_scope('g'):
        fake_imgs = tf.concat([fake_imgs,y],1)
        w1 = tf.get_variable('w1',shape=[fake_imgs.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,28*28],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,28*28],initializer=tf.zeros_initializer)
        h1 = tf.maximum(0.01*(tf.matmul(fake_imgs,w1)+b1),tf.matmul(fake_imgs,w1)+b1)
        h1_drop = tf.nn.dropout(h1,keep_prob)
        l2 = tf.matmul(h1_drop,w2)+b2
        h2 = tf.tanh(l2)
    return h2,l2

def d(raf_img,y,reuse=False):
    with tf.variable_scope('d',reuse=reuse):
        raf_img = tf.concat([raf_img,y],1)
        w1 = tf.get_variable('w1',shape=[raf_img.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,1],initializer=tf.zeros_initializer)
        h3 = tf.maximum(0.01*(tf.matmul(raf_img,w1)+b1),tf.matmul(raf_img,w1)+b1)
        l3 = tf.matmul(h3, w2) + b2
    return l3


batch_size = 50
learn_rate = 1e-3
echo = 50000
t1 = time.time()
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)
real_img = tf.placeholder(tf.float32,shape=[batch_size,28*28])
fake_img = tf.placeholder(tf.float32,shape=[batch_size,28*28])
g_y = tf.placeholder(tf.float32,shape=[batch_size,10])
r_y = tf.placeholder(tf.float32,shape=[batch_size,10])
keep_prob = tf.placeholder(tf.float32)
fake_g,l2 = g(fake_img,g_y,keep_prob)
real_d = d(real_img,r_y)
fake_d = d(fake_g,g_y,True)
loss_g = -tf.reduce_mean(fake_d)
loss_d = -tf.reduce_mean(real_d-fake_d)
var_g = [var for var in tf.trainable_variables() if var.name.startswith('g')]
var_d = [var for var in tf.trainable_variables() if var.name.startswith('d')]
#d1_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in var_d]
train_g = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss_g,var_list=var_g)
train_d = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss_d,var_list=var_d)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(echo+1):
        batch,y_label = mnist.train.next_batch(batch_size)
        batch = batch*2-1
        g_y1 = np.zeros([batch_size, 10])
        g_y1[:, 3] = 1
        g_z = np.random.uniform(-1, 1, size=(batch_size, 28*28))
        _= sess.run([train_g],feed_dict={fake_img:g_z,g_y:g_y1,keep_prob:0.5})
        for j in range(5):
            _,lossd,lossg = sess.run([train_d,loss_d,loss_g],feed_dict={real_img:batch,fake_img:g_z,r_y:y_label,g_y:g_y1,keep_prob:0.5})
            sess.run(d1_clip)
        if i % 4000 == 0:
            print("epoch:%d"%i,"loss_d:%s"%lossd,"loss_g:%s"%lossg)
            g1 = sess.run(fake_g, feed_dict={fake_img: g_z,g_y:g_y1,keep_prob:1.0})
            with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+'sampleswgan.pkl', 'wb') as f:
                pickle.dump(g1, f)
    g1 = sess.run(fake_g,feed_dict={fake_img:g_z,g_y:g_y1})
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