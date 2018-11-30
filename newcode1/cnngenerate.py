import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)


def con2d(x,reuse):
    with tf.variable_scope('h',reuse=reuse):
        w = tf.get_variable('w',shape=[5,5,1,32],dtype=tf.float32,initializer=
        tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b',shape=[32],dtype=tf.float32,initializer=
        tf.zeros_initializer)
        w1 = tf.get_variable('w1',shape=[5,5,32,32],dtype=tf.float32,initializer=
        tf.truncated_normal_initializer(stddev=0.01))
        b1 = tf.get_variable('b1',shape=[32],dtype=tf.float32,initializer=
        tf.zeros_initializer)
        h = tf.nn.relu(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")+b)
        h_pool = tf.nn.max_pool(h,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        h1 = tf.nn.relu(tf.nn.conv2d(h_pool,w1,strides=[1,1,1,1],padding="SAME")+b1)
        h_pool1 = tf.nn.max_pool(h1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        return h_pool1


epoch = 500
batch_size = 20
x = tf.placeholder(tf.float32,shape=[None,28*28])
y = tf.placeholder(tf.float32,shape=[None,10])
keep_drop = tf.placeholder(tf.float32)
x_ = tf.reshape(x,shape=[-1,28,28,1])