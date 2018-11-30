import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle
import time


def g(fake_imgs,keep_prob):
    with tf.variable_scope('g'):
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
    return h2

def d(raf_img,reuse=False):
    with tf.variable_scope('d',reuse=reuse):
        w1 = tf.get_variable('w1',shape=[raf_img.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,10],initializer=tf.zeros_initializer)
        h3 = tf.maximum(0.01*(tf.matmul(raf_img,w1)+b1),tf.matmul(raf_img,w1)+b1)
        l3 = tf.matmul(h3, w2) + b2
        h3 = tf.nn.softmax(l3)
    return h3


batch_size = 50
learn_rate = 1e-4
epoch = 30000
t1 = time.time()
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)
real_img = tf.placeholder(tf.float32,shape=[None,28*28])
fake_img = tf.placeholder(tf.float32,shape=[None,28*28])
real_y = tf.placeholder(tf.float32,shape=[None,10])
fake_y = tf.placeholder(tf.float32,shape=[None,10])
keep_prob = tf.placeholder(tf.float32)

g_fake = g(fake_img,keep_prob)
#g_fake1 = tf.concat([g_fake,real_y],1)
#real_img1 = tf.concat([real_img,real_y],1)
d_real = d(real_img)
d_fake = d(g_fake,True)


loss_d = -tf.reduce_mean(real_y*tf.log(d_real)+real_y*tf.log(d_fake))
#loss_d = -tf.reduce_mean(real_y*tf.log(d_real))
loss_g = -tf.reduce_mean(real_y*tf.log(d_fake)+(1-real_y)*tf.log(1-d_fake))

var_g = [var for var in tf.trainable_variables() if var.name.startswith('g')]
var_d = [var for var in tf.trainable_variables() if var.name.startswith('d')]
d1_clip = [var.assign(tf.clip_by_value(var, -0.1, 0.1)) for var in var_d]
train_g = tf.train.AdamOptimizer(learn_rate).minimize(loss_g,var_list=var_g)
train_d = tf.train.AdamOptimizer(learn_rate).minimize(loss_d,var_list=var_d)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    '''a = np.zeros([batch_size,1])
    b = np.zeros([batch_size,11])
    b[:,10] = 1
    c = np.zeros([10000,1])'''
    for i in range(epoch+1):
        g_z = np.random.uniform(-1, 1, size=(batch_size, 28*28))
        for j in range(5):
            batch_x, y_label = mnist.train.next_batch(batch_size)
            #y_label = np.concatenate((y_label,a),1)
            _,d_loss = sess.run([train_d,loss_d],feed_dict={fake_img:g_z,real_img:batch_x,real_y:y_label,keep_prob:0.5})
            sess.run([d1_clip])
        _,g_loss = sess.run([train_g,loss_g],feed_dict={fake_img:g_z,real_img:batch_x,real_y:y_label,keep_prob:0.5})
        if i%3000 == 0:
            print("epoch:{0},loss_d:{1},loss_g:{2}".format(i,d_loss,g_loss))
            g1 = sess.run(g_fake, feed_dict={fake_img:g_z,real_img:batch_x,real_y:y_label,keep_prob:1})
            acc = tf.equal(tf.argmax(real_y, 1), tf.argmax(d_real, 1))
            accg = tf.equal(tf.argmax(real_y, 1), tf.argmax(d_fake, 1))
            acc1 = tf.reduce_mean(tf.cast(acc, tf.float32))
            acc1g = tf.reduce_mean(tf.cast(accg, tf.float32))
            print("test",acc1.eval({real_img: mnist.test.images, real_y: mnist.test.labels}))
            print("train", acc1.eval({real_img:batch_x, real_y:y_label, }))
            print("gfake", acc1g.eval({fake_img:g_z,real_img:batch_x,real_y:y_label,keep_prob:1}))
            '''with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+'sampleswgan.pkl', 'wb') as f:
                pickle.dump(g1, f)

            with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+'y_label.pkl', 'wb') as f:
                pickle.dump(y_label, f)'''