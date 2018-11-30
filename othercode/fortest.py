import tensorflow as tf
import time
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = tf.placeholder(tf.float32)
y = (x**2+x)
z = y**2+y
zy = tf.gradients(ys=z,xs=y)
zx = tf.gradients(ys=z,xs=x)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x1 = 2
    t1 = time.time()
    for i in range(10000):
#sess.run([y,g],feed_dict={x:x1})
        sess.run([zx],feed_dict={x:x1})
    t2 = time.time()
    print(t2-t1)