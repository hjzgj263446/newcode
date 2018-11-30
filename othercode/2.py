import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)

def w_vari(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def b_vari(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(w,x):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])
x_ = tf.reshape(x,shape=[-1,28,28,1])
w_conv1 = w_vari([5,5,1,32])
b_conv1 = b_vari([32])
h1 = tf.nn.relu(conv2d(w_conv1,x_)+b_conv1)
h1_p = max_pooling(h1)
w_conv2 = w_vari([5,5,32,64])
b_conv2 = b_vari([64])
h2 = tf.nn.relu(conv2d(w_conv2,h1_p)+b_conv2)
h2_pool = max_pooling(h2)
h2_re = tf.reshape(h2_pool,[-1,7*7*64])
w1 = w_vari([7*7*64,300])
b1 = b_vari([300])
h1_c = tf.nn.relu(tf.matmul(h2_re,w1)+b1)
keep_drop = tf.placeholder(tf.float32)
temp = tf.nn.dropout(h1_c,keep_drop)
w2 = w_vari([300,10])
b2 = b_vari([10])
y = tf.nn.softmax(tf.matmul(temp,w2)+b2)
loss = -tf.reduce_mean(y_*tf.log(y))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)
tf.global_variables_initializer().run()
loss_train = []
loss_test = []
item = []
for i in range(1000):
    x1,y1 = mnist.train.next_batch(50)
    trian_,lossc=sess.run([train,loss],feed_dict={x:x1,y_:y1,keep_drop:0.5})
    if i%40==0:
        loss_train.append(sess.run(loss,feed_dict={x:x1,y_:y1,keep_drop:1}))
        loss_test.append(sess.run(loss,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_drop:1.0}))
        item.append(i)
        print(i,lossc)
acc = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
acc1 = tf.reduce_mean(tf.cast(acc,tf.float32))
print(acc1.eval({x:mnist.test.images,y_:mnist.test.labels,keep_drop:1.0}))

plt.plot(item,loss_train,"r-.",label="T")
plt.plot(item,loss_test,'b-',label="r")
plt.legend()
plt.show()




