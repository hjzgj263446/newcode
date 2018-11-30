import tensorflow as tf
from sklearn.externals import joblib
import numpy as np
import pickle
import time
from PIL import Image

def get1():
    a = Image.open(r"C:\Users\62271\Desktop\4.jpg")
    a = a.resize((120, 120))
    a = np.array(a)
    a = a.reshape((-1,120*120,3))
    return a/255

def con2d(x,w_shape,b_shape,h_dim,reuse,keep_drop):
    with tf.variable_scope('h'+str(h_dim),reuse=reuse):
        w = tf.get_variable('w',shape=w_shape,dtype=tf.float32,initializer=
        tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b',shape=b_shape,dtype=tf.float32,initializer=
        tf.zeros_initializer)
        h = tf.nn.relu(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")+b)
        h = tf.nn.dropout(h,keep_drop)
        h_pool = tf.nn.max_pool(h,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        return h_pool


def fcont(x,f_dim,out_size,reuse,last_dim,keep_drop=1):
    with tf.variable_scope('f'+str(f_dim),reuse=reuse):
        w = tf.get_variable('w',shape=[x.shape[1],out_size],dtype=tf.float32,initializer=
                            tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b',shape=[1,out_size],initializer=
                            tf.constant_initializer(0.1))
        if last_dim == 2:
            return tf.matmul(x,w)+b
        a = tf.nn.relu(tf.matmul(x,w)+b)
        a = tf.nn.dropout(a,keep_prob=keep_drop)
        return a

def g(fake_img,keep_drop,reuse=False):
    with tf.variable_scope("g",reuse=reuse):
        h1 = con2d(fake_img, w_shape=[5, 5, 3, 12], b_shape=[12], h_dim=1, reuse=False,keep_drop=keep_drop)
        h2 = con2d(h1, w_shape=[5, 5, 12, 24], b_shape=[24], h_dim=2, reuse=False,keep_drop=keep_drop)
        w1 = tf.get_variable('w1',shape=[5,5,12,24],dtype=tf.float32,initializer=
                            tf.truncated_normal_initializer(stddev=0.01))
        h3 = tf.nn.conv2d_transpose(h2,w1,output_shape=[batch_size,60,60,12], strides=[1,2,2,1],padding="SAME")
        w2 = tf.get_variable('w2',shape=[5,5,3,12],dtype=tf.float32,initializer=
                            tf.truncated_normal_initializer(stddev=0.01))
        h4 = tf.nn.conv2d_transpose(h3,w2,output_shape=[batch_size,120,120,3], strides=[1,2,2,1],padding="SAME")
        h4 = tf.nn.tanh(h4)
        return h4

def d(img,keep_drop,dim,reuse=False):
    with tf.variable_scope("d"+str(dim),reuse=reuse):
        h1 = con2d(img, w_shape=[5, 5, 3, 12], b_shape=[12], h_dim=1, reuse=reuse,keep_drop=keep_drop)
        h2 = con2d(h1, w_shape=[5, 5, 12, 24], b_shape=[24], h_dim=2, reuse=reuse,keep_drop=keep_drop)
        h2 = tf.reshape(h2, [-1, 30 * 30 * 4])
        f1 = fcont(h2, 1, 4000, reuse, 0,keep_drop=keep_drop)
        f2 = fcont(f1, 2, 1, reuse, 2, keep_drop=keep_drop)
        return f2

def getXY():
    path1 = r"C:\迅雷下载\kaggle\train\test\dog120.m"
    x = joblib.load(path1)
    return x

def save(file_name,name):
    with open(r'C:\Myprogram\mnistdata' + '\\' + name + '.pkl', 'wb') as f:
        pickle.dump(file_name,f)



batch_size = 10
learn_rate = 1e-4
epoch = 5000
start = 0
end = 10
img5 = getXY()
img5 = img5[:1000,:,:]
img5 = img5/255
print(img5.shape)
a = img5.shape[0]-100
fake = tf.placeholder(tf.float32,shape=[batch_size,120*120,3],name="fake")
fake_ = tf.reshape(fake,shape=[-1,120,120,3])
real = tf.placeholder(tf.float32,shape=[batch_size,120*120,3],name="real")
real_ = tf.reshape(real,shape=[-1,120,120,3])
keep_drop = tf.placeholder(tf.float32,name="keep_drop")

g_fake = g(fake_,keep_drop)
tf.add_to_collection("pre_img",g_fake)
d_real1 = d(real_,keep_drop,1)
d_fake1 = d(g_fake,keep_drop,1,True)
d_real2 = d(fake_,keep_drop,2)
d_fake2 = d(g_fake,keep_drop,2,True)

d_loss1 = -tf.reduce_mean(d_real1-d_fake1)
d_loss2 = -tf.reduce_mean(d_real2-d_fake2)
g_loss = -tf.reduce_mean(d_fake1+d_fake2)

var_g = [var for var in tf.trainable_variables() if var.name.startswith('g')]
var_d1 = [var for var in tf.trainable_variables() if var.name.startswith('d1')]
var_d2 = [var for var in tf.trainable_variables() if var.name.startswith('d2')]
d1_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in var_d1]
d2_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in var_d2]

train_g = tf.train.GradientDescentOptimizer(learn_rate).minimize(g_loss,var_list=var_g)
train_d1 = tf.train.GradientDescentOptimizer(learn_rate).minimize(d_loss1,var_list=var_d1)
train_d2 = tf.train.GradientDescentOptimizer(learn_rate).minimize(d_loss2,var_list=var_d2)
saver = tf.train.Saver()
with tf.Session() as sess:
    t1 = time.time()
    sess.run(tf.global_variables_initializer())
    fake1 = get1()
    # real_img = img6[start:end,:]
    real_img = np.zeros((batch_size, 120 * 120, 3))
    real_img += fake1
    for i in range(epoch+1):
        fake_img = img5[start:end,:,:]
        start += 10
        end += 10
        if end>a:
            start = 0
            end = 10
        for j in range(20):
            _, __, loss_d1, loss_d2 = sess.run([train_d1, train_d2, d_loss1, d_loss2],
                                               feed_dict={fake: fake_img, real: real_img, keep_drop: 0.5})
            sess.run([d1_clip,d2_clip])
        _,loss_g= sess.run([train_g,g_loss],feed_dict={fake:fake_img,real:real_img,keep_drop:0.5})

        if i %50 == 0:
            g_fake1 = sess.run(g_fake,feed_dict={fake:fake_img,real:real_img,keep_drop:1})
            save(g_fake1,"g_fakef"+str(i))
            save(fake_img,"fake_img"+str(i))
            t2 = time.time()
            print("epoch:%d" % i, "loss_g:%s" % loss_g, "loss_d1:%s" % loss_d1, "loss_d2:%s" % loss_d2,"time:%d"%(t2-t1))
            t1 = t2

    saver.save(sess,r'C:\Myprogram\mnistdata' + '\\' +'model1.ckpt')