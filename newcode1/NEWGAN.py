import tensorflow as tf
import numpy as np
import pickle
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)


pathread = r'C:\Myprogram\mnistdata'
pathwrite = r'C:\Myprogram\mnistdata'
def g(fake_imgs,label_y,keep_prob):
    with tf.variable_scope('g'):
        fake_imgs = tf.concat([fake_imgs,label_y],1)
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
        h2 = tf.reshape(h2,(-1,28,28,1))
    return h2

def d(raf_img,label_y,keep_drop,reuse=False):
    with tf.variable_scope('d',reuse=reuse):
        h1 = con2d(raf_img, w_shape=[5, 5, 1, 12], b_shape=[12], h_dim=1, reuse=False)
        h2 = con2d(h1, w_shape=[5, 5, 12, 24], b_shape=[24], h_dim=2, reuse=False)
        h2 = tf.reshape(h2, [-1, 7 * 7 * 24])
        h_concat = tf.concat([h2,label_y],1)
        f2 = fcont(h_concat, keep_drop=keep_drop)
    return h2,f2

def save(i,name,file):
    with open(pathwrite+'\\'+str(i)+name+'.pkl', 'wb') as f:
        pickle.dump(file,f)

def con2d(x,w_shape,b_shape,h_dim,reuse):
    with tf.variable_scope('h'+str(h_dim),reuse=reuse):
        w = tf.get_variable('w',shape=w_shape,dtype=tf.float32,initializer=
        tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b',shape=b_shape,dtype=tf.float32,initializer=
        tf.zeros_initializer)
        h = tf.nn.relu(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")+b)
        h_pool = tf.nn.max_pool(h,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        return h_pool

def fcont(xin,keep_drop=1):
    with tf.variable_scope('f',reuse=tf.AUTO_REUSE):
        w = tf.get_variable('w',shape=[xin.shape[1],128],dtype=tf.float32,initializer=
                            tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b',shape=[1,128],initializer=
                            tf.constant_initializer(0.1))
        w1 = tf.get_variable('w1',shape=[128,1],dtype=tf.float32,initializer=
                            tf.truncated_normal_initializer(stddev=0.01))
        b1 = tf.get_variable('b1',shape=[1,1],initializer=
                            tf.constant_initializer(0.1))
        h1 = tf.nn.relu(tf.matmul(xin,w)+b)
        h1 = tf.nn.dropout(h1,keep_drop)
        h2 = tf.nn.sigmoid(tf.matmul(h1,w1)+b1)
        return h2

batch_size = 50
lr = 1e-4
epoch = 100000
LAMBDA = 10
x = tf.placeholder(tf.float32,shape = [None,784])
y = tf.placeholder(tf.float32,shape = [None,10])
z = tf.placeholder(tf.float32,shape = [None,784])
keep_pro = tf.placeholder(tf.float32)

x_ = tf.reshape(x,(-1,28,28,1))
g_fake = g(z,y,keep_pro)
featurereal,doutreal = d(x_,y,keep_pro)
featurefake,doutfake = d(g_fake,y,keep_pro,True)
lossd = tf.reduce_mean(doutfake)-tf.reduce_mean(doutreal)
lossg = -tf.reduce_mean((doutfake))+tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(featurereal-featurefake))))
alpha = tf.random_uniform(shape=[batch_size, 1,1,1], minval=0., maxval=1.)
interpolates = alpha * x_ + (1 - alpha) * g_fake
grad = tf.gradients(d(interpolates,y,keep_pro,reuse=True), [interpolates])[0]
slop = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
gp = tf.reduce_mean((slop - 1.) ** 2)
lossd += LAMBDA * gp
var_g = [var for var in tf.trainable_variables() if var.name.startswith('g')]
var_d = [var for var in tf.trainable_variables() if var.name.startswith('d')]
train_g = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(lossg,var_list=var_g)
train_d = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(lossd,var_list=var_d)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t1 = time.time()
    for i in range(epoch+1):
        gz = np.random.normal(0, 1, size=(batch_size, 28 * 28))
        x_train,y_train = mnist.train.next_batch(batch_size)
        for k in range(2):
            _,loss_d = sess.run([train_d,lossd],feed_dict={x:x_train,y:y_train,z:gz,keep_pro:0.5})
        _,loss_g = sess.run([train_g,lossg],feed_dict={x:x_train,y:y_train,z:gz,keep_pro:0.5})
        if i % 5000 == 0:
            t2 = time.time()
            print("epoch:{0}   loss_d:{1}     loss_g:{2}   time:{3}".format(i,loss_d,loss_g,t2-t1))
            t1 = t2
        if i % 10000 == 0:
            gfake1 = sess.run(g_fake,feed_dict={x:x_train,y:y_train,z:gz,keep_pro:1})
            save(i,"cganfake",gfake1)
            save(i,"ylabel",y_train)
