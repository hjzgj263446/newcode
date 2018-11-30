import tensorflow as tf
import matplotlib.pyplot as plt
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

x = tf.placeholder(tf.float32,shape=[None,28*28])
y = tf.placeholder(tf.float32,shape=[None,10])
keep_drop = tf.placeholder(tf.float32)
epoch = 1000
batch_size = 50
l1 = []
l2 = []
item = []
x_ = tf.reshape(x,shape=[-1,28,28,1])
h1 = con2d(x_,w_shape=[5,5,1,32],b_shape=[32],h_dim=1,reuse=False)
h2 = con2d(h1,w_shape=[5,5,32,64],b_shape=[64],h_dim=2,reuse=False)
h2 = tf.reshape(h2,(-1,7*7*64))
h2 = fcont(h2,keep_drop)
loss = -tf.reduce_mean(y*tf.log(h2))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        x1,y1 = mnist.train.next_batch(batch_size)
        _,loss1 = sess.run([train,loss],feed_dict={x:x1,y:y1,keep_drop:0.5})
        if i % 50 == 0:
            l1.append(sess.run(loss,feed_dict={x:x1,y:y1, keep_drop: 1.0}))
            l2.append(sess.run(loss,feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_drop: 1.0}))
            item.append(i)
            print("epoch:%d"%i,' ',"loss:%s"%loss1)

    acc = tf.equal(tf.argmax(y, 1), tf.argmax(h2, 1))
    acc1 = tf.reduce_mean(tf.cast(acc, tf.float32))
    print(acc1.eval({x: mnist.test.images, y: mnist.test.labels, keep_drop: 1.0}))

plt.plot(item,l1,"r-.",label="T")
plt.plot(item,l2,'b-',label="r")
plt.legend()
plt.show()