import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)



def g(fake_imgs,keep_prob,reuse=False):
    with tf.variable_scope('g',reuse=reuse):
        w1 = tf.get_variable('w1',shape=[fake_imgs.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.01))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,28*28],initializer=
        tf.truncated_normal_initializer(stddev=0.01))
        b2 = tf.get_variable('b2',shape=[1,28*28],initializer=tf.zeros_initializer)
        h1 = tf.maximum(0.01*(tf.matmul(fake_imgs,w1)+b1),tf.matmul(fake_imgs,w1)+b1)
        h1_drop = tf.nn.dropout(h1,keep_prob)
        l2 = tf.matmul(h1_drop,w2)+b2
        h2 = tf.tanh(l2)
    return h2

def d(raf_img,d_dim,reuse=False):
    with tf.variable_scope('d'+str(d_dim),reuse=reuse):
        w1 = tf.get_variable('w1',shape=[raf_img.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,1],initializer=tf.truncated_normal_initializer(stddev=0.01))
        b2 = tf.get_variable('b2',shape=[1,1],initializer=tf.zeros_initializer)
        h3 = tf.maximum(0.01*(tf.matmul(raf_img,w1)+b1),tf.matmul(raf_img,w1)+b1)
        l3 = tf.matmul(h3, w2) + b2
    return l3

def openfile(name):
    with open(r'C:\Myprogram\mnistdata' + '\\' + name+'.pkl', 'rb') as f:
        x = pickle.load(f)
    return x

def save(file_name,name):
    with open(r'C:\Myprogram\mnistdata' + '\\' + name + '.pkl', 'wb') as f:
        pickle.dump(file_name,f)


batch_size = 50
learn_rate = 1e-3
epoch = 50000


fake = tf.placeholder(tf.float32,shape=[batch_size,28*28],name="fake")
real = tf.placeholder(tf.float32,shape=[batch_size,28*28],name="real")
keep_drop = tf.placeholder(tf.float32,name="keep_drop")

g_fake = g(fake,keep_drop)
tf.add_to_collection("pre_img",g_fake)
d_real1 = d(real,1)
d_fake1 = d(g_fake,1,True)


d_loss1 = -tf.reduce_mean(d_real1-d_fake1)
g_loss = -tf.reduce_mean(d_fake1)

var_g = [var for var in tf.trainable_variables() if var.name.startswith('g')]
var_d1 = [var for var in tf.trainable_variables() if var.name.startswith('d1')]
d1_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in var_d1]


train_g = tf.train.GradientDescentOptimizer(learn_rate).minimize(g_loss,var_list=var_g)
train_d1 = tf.train.GradientDescentOptimizer(learn_rate).minimize(d_loss1,var_list=var_d1)


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        for j in range(20):
            fake_img = np.random.random([batch_size, 28 * 28])
            real_img, _ = mnist.train.next_batch(batch_size)
            _, loss_d1 ,loss_g= sess.run([train_d1, d_loss1,g_loss],
                                      feed_dict={fake: fake_img, real: real_img, keep_drop: 0.5})
            sess.run(d1_clip)
        _ = sess.run([train_g],
                     feed_dict={fake:fake_img,real:real_img,keep_drop:0.5})


        if i%5000 == 0:
            print("epoch:%d"%i,"loss_g:%s"%loss_g,"loss_d1:%s"%loss_d1)
            fake_img = np.random.random((batch_size,28*28))
            g_fake1 = sess.run(g_fake,feed_dict={fake:fake_img,keep_drop:1})
            save(g_fake1,"g_fake"+str(i))
            save(real_img, "realimg" + str(i))

    saver.save(sess,r'C:\Myprogram\mnistdata' + '\\' +'model1.ckpt')