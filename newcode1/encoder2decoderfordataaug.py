import tensorflow as tf
import numpy as np
import pickle
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)

def encoderModel(inputx,keep_pro):
    with tf.variable_scope("encoder"):
        w1 = tf.get_variable('w1',shape=[inputx.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,128],initializer=tf.zeros_initializer)
        h1 = tf.nn.relu(tf.matmul(x,w1)+b1)
        h1 = tf.nn.dropout(h1,keep_pro)
        h2 = tf.tanh(tf.matmul(h1,w2)+b2)
        return h2

def decoderModel(encoderinput,keep_pro,reuse=False):
    with tf.variable_scope("decoder",reuse=reuse):
        w1 = tf.get_variable('w1',shape=[encoderinput.shape[1],300],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,300],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[300,28*28],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,28*28],initializer=tf.zeros_initializer)
        h1 = tf.nn.relu(tf.matmul(encoderinput,w1)+b1)
        h1 = tf.nn.dropout(h1,keep_pro)
        h2 = tf.tanh(tf.matmul(h1,w2)+b2)
        return h2

def g(fake_imgs,label_y,keep_prob):
    with tf.variable_scope('g'):
        fake_imgs = tf.concat([fake_imgs,label_y],1)
        w1 = tf.get_variable('w1',shape=[fake_imgs.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,128],initializer=tf.zeros_initializer)
        h1 = tf.maximum(0.01*(tf.matmul(fake_imgs,w1)+b1),tf.matmul(fake_imgs,w1)+b1)
        h1_drop = tf.nn.dropout(h1,keep_prob)
        l2 = tf.matmul(h1_drop,w2)+b2
        h2 = tf.tanh(l2)
    return h2

def d(raf_img,label_y,reuse=False):
    with tf.variable_scope('d',reuse=reuse):
        raf_img = tf.concat([raf_img,label_y],1)
        w1 = tf.get_variable('w1',shape=[raf_img.shape[1],128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,128],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,1],initializer=tf.zeros_initializer)
        h3 = tf.maximum(0.01*(tf.matmul(raf_img,w1)+b1),tf.matmul(raf_img,w1)+b1)
        l3 = tf.matmul(h3, w2) + b2
    return l3

def save(i,name,file):
    with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+name+'.pkl', 'wb') as f:
        pickle.dump(file,f)




batch_size = 100
lr = 1e-4
epoch = 50000
LAMBDA = 10
x = tf.placeholder(tf.float32,shape = [None,784])
y = tf.placeholder(tf.float32,shape = [None,10])
z = tf.placeholder(tf.float32,shape = [None,128])
encoder2d = tf.placeholder(tf.float32,shape = [None,128])
keep_pro = tf.placeholder(tf.float32)


encoder_x = encoderModel(x,keep_pro)
recon_x = decoderModel(encoder_x,keep_pro)
g_fake = g(z,y,keep_pro)
doutreal = d(encoder_x,y)
doutfake = d(g_fake,y,True)
lossd = tf.reduce_mean(doutfake)-tf.reduce_mean(doutreal)
lossg = -tf.reduce_mean((doutfake))
lossen2de = tf.reduce_mean(tf.sqrt(tf.reduce_sum((tf.square(x-recon_x)))))

alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
interpolates = alpha * encoder_x + (1 - alpha) * g_fake
grad = tf.gradients(d(interpolates,y,reuse=True), [interpolates])[0]
slop = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
gp = tf.reduce_mean((slop - 1.) ** 2)
lossd += LAMBDA * gp
var_g = [var for var in tf.trainable_variables() if var.name.startswith('g')]
var_d = [var for var in tf.trainable_variables() if var.name.startswith('d')]
var_enc = [var for var in tf.trainable_variables() if var.name.startswith('encoder')]
var_dec = [var for var in tf.trainable_variables() if var.name.startswith('decoder')]
train_g = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(lossg,var_list=var_g)
train_d = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(lossd,var_list=var_d)
train_e2d = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(lossen2de,var_list=[var_enc,var_dec])
reconfake = decoderModel(encoder2d,keep_pro,True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    j = 0
    t1 = time.time()
    for i in range(epoch+1):
        gz = np.random.normal(0, 1, size=(batch_size, 128))
        x_train,y_train = mnist.train.next_batch(batch_size)
        for j in range(2):
            _,loss_en2de = sess.run([train_e2d,lossen2de],feed_dict={x:x_train,y:y_train,z:gz,keep_pro:0.5})
        for k in range(2):
            _,loss_d = sess.run([train_d,lossd],feed_dict={x:x_train,y:y_train,z:gz,keep_pro:0.5})
        _,loss_g = sess.run([train_g,lossg],feed_dict={x:x_train,y:y_train,z:gz,keep_pro:0.5})
        if i % 500 == 0:
            t2 = time.time()
            print("epoch:{0}  loss_en2d:{4} loss_d:{1}     loss_g:{2}   time:{3}".format(i,loss_d,loss_g,t2-t1,loss_en2de))
            t1 = t2
        if i % 5000 == 0:
            gfake1 = sess.run(g_fake, feed_dict={x: x_train, y: y_train, z: gz, keep_pro: 1})
            gfake2 = sess.run(reconfake,feed_dict={x:x_train,y:y_train,encoder2d:gfake1,z:gz,keep_pro:1})
            gfake3 = sess.run(recon_x,feed_dict={x: x_train, y: y_train, z: gz, keep_pro: 1})
            save(i,"caganreal",x_train)
            save(i,"cganfake",gfake2)
            save(i, "cganfake3", gfake3)
            save(i,"ylabel",y_train)
