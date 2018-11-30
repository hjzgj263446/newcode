import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)

def w_vari(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def b_vari(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def g_graph(fake_image,input_size,out_size,alpha,keep_drop):
    w1 = w_vari([input_size,128])
    b1 = b_vari([128])
    l1 = tf.matmul(fake_image,w1)+b1
    h1 = tf.maximum(alpha*(l1),l1)
    h1_drop = tf.nn.dropout(h1,keep_drop)
    w2 = w_vari([128,out_size])
    b2 = b_vari([out_size])
    l3 = tf.matmul(h1_drop,w2)+b2
    g_reu = tf.tanh(l3)
    return l3,g_reu

def d_graph(image,input_size,out_size,alpha):
    w3 = w_vari([input_size,128])
    b3= b_vari([128])
    l2 = tf.matmul(image,w3)+b3
    h2 = tf.maximum(alpha*l2,l2)
    w4 = w_vari([128,out_size])
    b4 = b_vari([out_size])
    l4 = tf.matmul(h2,w4)+b4
    d_reu = tf.sigmoid(l4)
    return l4,d_reu

learn_rate = 1e-3
img_size = 28*28
fake_size = 28*28
batch_size = 50
real_img = tf.placeholder(tf.float32,shape=[None,img_size])
fake_img = tf.placeholder(tf.float32,shape = [None,fake_size])
g_logit,g_reu1 = g_graph(fake_img,28*28,fake_size,0.01,0.8)
d_logit,d_reu1 = d_graph(tf.concat([real_img,g_reu1],0),img_size,10,0.01)
d_reu = d_reu1[:50,:]
g_reu = d_reu1[50:,:]
d_loss = -tf.reduce_mean(tf.log(d_reu) + tf.log(1 - g_reu))
g_loss = -tf.reduce_mean(tf.log(g_reu))
g_train = tf.train.AdamOptimizer(learn_rate).minimize(g_loss)
d_train = tf.train.AdamOptimizer(learn_rate).minimize(d_loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        real_batch,_ = mnist.train.next_batch(50)
        g_z = np.random.uniform(-1, 1, size=(batch_size, fake_size))
        _,loss2 = sess.run([g_train,g_loss],feed_dict={fake_img:g_z,real_img:real_batch})
        _,loss = sess.run([d_train,d_loss],feed_dict={fake_img:g_z,real_img:real_batch})
        if i%500 == 0:
            print(i,loss,loss2)
    fake = sess.run(g_reu1,feed_dict={fake_img:g_z})
    with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+'gan.pkl', 'wb') as f:
        pickle.dump(fake,f)



