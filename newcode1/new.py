import tensorflow as tf
import numpy as np
import pickle
import time

pathread = r"D:\tjj\\"
pathwrite = r'D:\tjj\generadata'

def g(fake_imgs,label_y,keep_prob=1,reuse=False):
    with tf.variable_scope('g',reuse=reuse):
        fake_imgs = tf.concat([fake_imgs,label_y],1)
        w1 = tf.get_variable('w1',shape=[fake_imgs.shape[1],300],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,300],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[300,128],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,128],initializer=tf.zeros_initializer)
        h1 = tf.nn.relu(tf.matmul(fake_imgs,w1)+b1)
        h1_drop = tf.nn.dropout(h1,keep_prob)
        l2 = tf.matmul(h1_drop,w2)+b2
    return l2

def d(raf_img,label_y,label,reuse=False):
    with tf.variable_scope('d',reuse=reuse):
        raf_img = tf.concat([raf_img,label_y,label],1)
        w1 = tf.get_variable('w1',shape=[raf_img.shape[1],300],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',shape=[1,300],initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2',shape=[300,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',shape=[1,1],initializer=tf.zeros_initializer)
        h3 = tf.nn.relu(tf.matmul(raf_img,w1)+w2)
        l3 = tf.matmul(h3, w2) + b2
    return l3

def openfile(name):
    with open(pathread+name+'.pkl','rb') as f:
        a = pickle.load(f)
        return a
def save(i,name,file):
    with open(pathwrite+'\\'+str(i)+name+'.pkl', 'wb') as f:
        pickle.dump(file,f)

batch_size = 50
lr = 1e-4
epoch = 400000
LAMBDA = 10
x = tf.placeholder(tf.float32,shape = [None,18])
y = tf.placeholder(tf.float32,shape = [None,18])
z1 = tf.placeholder(tf.float32,shape = [None,128])
z2 = tf.placeholder(tf.float32,shape = [None,128])
keep_pro = tf.placeholder(tf.float32)

g_fake = g(z1,x,keep_pro)
g_real = g(z2,y,reuse=True)
doutreal = d(g_real,y,x)
doutfake = d(g_fake,x,x,True)
# lossd = -tf.reduce_mean(tf.log(doutreal)+tf.log(1-doutfake))
# lossg = -tf.reduce_mean(tf.log(doutfake))
lossd = tf.reduce_mean(doutfake)-tf.reduce_mean(doutreal)
lossg = -tf.reduce_mean((doutfake))
alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
interpolates = alpha * g_real + (1 - alpha) * g_fake
grad = tf.gradients(d(interpolates,x,reuse=True), [interpolates])[0]
slop = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
gp = tf.reduce_mean((slop - 1.) ** 2)
lossd += LAMBDA * gp
var_g = [var for var in tf.trainable_variables() if var.name.startswith('g')]
var_d = [var for var in tf.trainable_variables() if var.name.startswith('d')]
train_g = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(lossg,var_list=var_g)
train_d = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(lossd,var_list=var_d)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    j = 0
    jj = batch_size
    trainsample = openfile("trainsample")
    labelsample = openfile("labelsample")
    length = len(trainsample)
    flag = True
    for i in range(epoch+1):
        t1 = time.time()
        while(flag):
            gz1 = np.random.normal(0, 1, size=(batch_size, 128))
            gz2 = np.random.normal(0, 1, size=(batch_size, 128))
            if j+1 > length:
                j = 0
                flag = False
            x_train,y_label = trainsample[j],labelsample[j]
            j += 1
            for k in range(2):
                _,loss_d = sess.run([train_d,lossd],feed_dict={x:x_train,y:y_label,z1:gz1,z2:gz2,keep_pro:0.5})
            _,loss_g = sess.run([train_g,lossg],feed_dict={x:x_train,y:y_label,z1:gz1,z2:gz2,keep_pro:0.5})
        flag = True
        fake = sess.run([g_fake],feed_dict={x:x_train,y:y_label,z1:gz1,z2:gz2,keep_pro:1})
        save(i,"fake",fake)
        save(i,"label",x_train)
        di1 = openfile("sougoudic2")
        di2 = openfile("sougoudic1")
        tem = np.zeros((10000,18))
        gz3 = np.random.normal(0, 1, size=(10000, 128))
        for kk in range(1,10001):
            tem[kk-1] = di2[di1[kk]]
        fake = sess.run([g_fake],feed_dict={x:tem,z1:gz3,keep_pro:1.0})
        save(i,"allfake",fake)
        save(i,"alllabel",tem)
        t2 = time.time()
        print("epoch:{0}   loss_d:{1}     loss_g:{2}   time:{3}".format(i,loss_d,loss_g,t2-t1))
        t1 = t2

