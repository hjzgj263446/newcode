import tensorflow as tf
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)

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
        w2 = tf.get_variable('w2',shape=[128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,1],initializer=tf.zeros_initializer)
        h3 = tf.maximum(0.01*(tf.matmul(raf_img,w1)+b1),tf.matmul(raf_img,w1)+b1)
        l3 = tf.matmul(h3, w2) + b2
        h4 = tf.sigmoid(l3)
    return h4

def save(i,name,file):
    with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+name+'.pkl', 'wb') as f:
        pickle.dump(file,f)

batch_size = 100
lr = 1e-4
epoch = 30000
x = tf.placeholder(tf.float32,shape = [None,784])
y = tf.placeholder(tf.float32,shape = [None,10])
z = tf.placeholder(tf.float32,shape = [None,784])
keep_pro = tf.placeholder(tf.float32)

inputimg = tf.concat([z,y],1)
drinput = tf.concat([x,y],1)
gfake = g(inputimg,keep_pro)
gfakein = tf.concat([gfake,y],1)
doutreal = d(drinput)
doutfake = d(gfakein,True)
lossd = -tf.reduce_mean(tf.log(1-doutfake)+tf.log(doutreal))
lossg = -tf.reduce_mean(tf.log(doutfake))
var_g = [var for var in tf.trainable_variables() if var.name.startswith('g')]
var_d = [var for var in tf.trainable_variables() if var.name.startswith('d')]
train_g = tf.train.AdamOptimizer(lr).minimize(lossg,var_list=var_g)
train_d = tf.train.AdamOptimizer(lr).minimize(lossd,var_list=var_d)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch+1):
        gz = np.random.uniform(-1, 1, size=(batch_size,784))
        x_train,y_trian = mnist.train.next_batch(batch_size)
        for j in range(5):
            _,loss_d = sess.run([train_d,lossd],feed_dict={x:x_train,y:y_trian,z:gz,keep_pro:0.5})
        _,loss_g = sess.run([train_g,lossg],feed_dict={x:x_train,y:y_trian,z:gz,keep_pro:0.5})
        if i % 3000 == 0:
            print("epoch:{0}   loss_d:{1}     loss_g:{2}".format(i,loss_d,loss_g))
            gfake1 = sess.run(gfake,feed_dict={x:x_train,y:y_trian,z:gz,keep_pro:1})
            save(i,"cganfake",gfake1)
