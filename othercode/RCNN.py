import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)

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
        w = tf.get_variable('w',shape=[xin.shape[1],300],dtype=tf.float32,initializer=
                            tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b',shape=[1,300],initializer=
                            tf.constant_initializer(0.1))
        w1 = tf.get_variable('w1',shape=[300,10],dtype=tf.float32,initializer=
                            tf.truncated_normal_initializer(stddev=0.01))
        b1 = tf.get_variable('b1',shape=[1,10],initializer=
                            tf.constant_initializer(0.1))
        h1 = tf.nn.relu(tf.matmul(xin,w)+b)
        h2 = tf.nn.softmax(tf.matmul(h1,w1)+b1)
        return h2

def f2cont(xin):
    with tf.variable_scope("f2",reuse=tf.AUTO_REUSE):
        w = tf.get_variable("w",shape=[xin.shape[1],1000],dtype=tf.float32,initializer=
                            tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b',shape=[1,1000],initializer=
                            tf.constant_initializer(0.1))
        w1 = tf.get_variable("w1",shape=[1000,64],dtype=tf.float32,initializer=
                            tf.truncated_normal_initializer(stddev=0.01))
        b1 = tf.get_variable('b1',shape=[1,64],initializer=
                            tf.constant_initializer(0.1))
        h1 = tf.nn.relu(tf.matmul(xin,w)+b)
        h2 = tf.nn.softmax(tf.matmul(h1,w1)+b1)
        return h2

n = 50
epoch = 5000
batch_size = 50
j,p = 0,0
temp = np.zeros((64,49))
temp2 = tf.Variable(tf.zeros([batch_size,64,64]), name="v1")
temp3 = tf.Variable(tf.zeros([batch_size,64,49]))
x = tf.placeholder(tf.float32,shape=[None,28*28])
y = tf.placeholder(tf.float32,shape=[None,10])
keep_drop = tf.placeholder(tf.float32)
x_ = tf.reshape(x,shape=[-1,28,28,1])
h1 = con2d(x_,w_shape=[5,5,1,32],b_shape=[32],h_dim=1,reuse=False)
h2 = con2d(h1,w_shape=[5,5,32,64],b_shape=[64],h_dim=2,reuse=False)
h2 = tf.reshape(h2,(-1,7*7,64))
for i in range(n):
    k = tf.transpose(h2[i])
    for j in range(k.shape[0]):
        m = k[j] + temp
        temp1 = tf.concat([m,k],1)
        temp1 = tf.reshape(temp1,(1,-1))
        out = f2cont(temp1)
        out = tf.reshape(out,(1,64))
        temp2[i,j].assign(out)

print(temp2[0].shape)
i,j = 0,0
for i in range(n):
    k = temp2[i]
    for j in range(k.shape[0]):
        p = tf.reshape(tf.transpose(k[j]),(64,1))
        m = tf.transpose(h2[i])
        temp3[i,j].assign(tf.reduce_sum(tf.multiply(m,p),reduction_indices=[0]))

temp3 = tf.reshape(temp3,(-1,64*49))
h3 = fcont(temp3)
loss = -tf.reduce_mean(y*tf.log(h3))
train = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i=0
    for i in range(epoch):
        x1,y1 = mnist.train.next_batch(batch_size)
        _,loss1 = sess.run([train,loss],feed_dict={x:x1,y:y1,keep_drop:0.5})
        if i % 200 == 0:
            print("epoch:%d"%i,' ',"loss:%s"%loss1)


    acc = tf.equal(tf.argmax(y, 1), tf.argmax(h3, 1))
    acc1 = tf.reduce_mean(tf.cast(acc, tf.float32))
    print(acc1.eval({x: mnist.test.images[:n], y: mnist.test.labels[:n], keep_drop: 1.0}))