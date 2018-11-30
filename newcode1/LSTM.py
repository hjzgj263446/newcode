import tensorflow as tf
import newcode1.dataget as nd
import pickle
import numpy as np
import time

def g(fake_imgs,keep_prob,reuse=False):
    with tf.variable_scope('g',reuse=reuse):
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
        h3 = tf.sigmoid(l3)
    return h3

def Lstm(c_t,a_t,x,size,num,reuse):
    ac = tf.concat([c_t,a_t],1)
    ax = tf.concat([a_t,x],1)
    with tf.variable_scope("Lstm"+str(num),reuse=reuse):
        wc = tf.get_variable('wc',shape=[ac.shape[1],size],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        bc = tf.get_variable('bc',shape=[1,size],initializer=tf.zeros_initializer)
        wu = tf.get_variable('wu',shape=[ax.shape[1],size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bu = tf.get_variable('bu',shape=[1,size],initializer=tf.zeros_initializer)
        wf = tf.get_variable('wf',shape=[ax.shape[1],size],initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        bf = tf.get_variable('bf',shape=[1,size],initializer=tf.zeros_initializer)
        wo = tf.get_variable('wo',shape=[ax.shape[0],size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bo = tf.get_variable('bo',shape=[1,size],initializer=tf.zeros_initializer)
        cnt = tf.tanh(tf.matmul(ac,wc)+bc)
        Fu = tf.sigmoid(tf.matmul(ax,wu)+bu)
        Ff = tf.sigmoid(tf.matmul(ax,wf)+bf)
        Fo = tf.sigmoid(tf.matmul(ax,wo)+bo)
        ct = Fu*cnt + Ff*c_t
        at = Fo*tf.tanh(ct)
    return ct,at

def save(i,name,file):
    with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+name+'.pkl', 'wb') as f:
        pickle.dump(file,f)

x = tf.placeholder(tf.float32,[None,50,28*28])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
layer_num = 3
hide_size = 300
batch_size = 50
timestep_size = 4
lr = 1e-4
epoch = 1000
stacked_rnn = []
for iiLyr in range(layer_num):
    stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(num_units=hide_size, state_is_tuple=True))
mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = mlstm_cell(x[timestep], state)
        outputs.append(cell_output)

gfakelist = []
lossglist = []
lossdlist = []
start = 0
end = 50
t = 1
for i in outputs[:-1]:
    if i == outputs[0]:
        gfake = g(i,keep_prob)
        dreal = d(x[t])
        t += 1
    else:
        gfake = g(i,keep_prob,True)
        dreal = d(x[t],True)
    dfake = d(gfake,True)
    gfakelist.append(gfake)
    lossdlist.append(-tf.reduce_mean(tf.log(1-dfake)+tf.log(dreal)))
    lossglist.append(-tf.reduce_mean(tf.log(dfake)))
lossgall = tf.reduce_sum(lossglist)
lossdall = tf.reduce_sum(lossdlist)
train_g = tf.train.AdamOptimizer(lr).minimize(lossgall)
train_d = tf.train.AdamOptimizer(lr).minimize(lossdall)

cellpreout = [x[0]]
with tf.variable_scope('RNN',reuse=True):
    for timestep in range(timestep_size):
        (cellout, state) = mlstm_cell(cellpreout[-1], state)
        prefake = g(cellout,keep_prob,tf.AUTO_REUSE)
        cellpreout.append(prefake)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(epoch+1):
        if end<=55000:
            x1, y1 = nd.getData(start, end)
            start += batch_size
            end += batch_size
        else:
            start = 0
            end = batch_size
            x1, y1 = nd.getData(start, end)
        for j in range(5):
            _,lossd = sess.run([train_d,lossdall],feed_dict={x:x1,y:y1,keep_prob:0.5})
        _,lossg = sess.run([train_g,lossgall],feed_dict={x:x1,y:y1,keep_prob:0.5})
        if i % 10 == 0:
            print("epoch:{0}   lossd:{1}   lossg:{2}".format(i,lossd,lossg))
        if i % 200 == 0:
            save(i,"gfake",sess.run(gfakelist,feed_dict={x:x1,y:y1,keep_prob:1}))
            save(i, "prefake", sess.run(cellpreout,feed_dict={x: x1, y: y1, keep_prob: 1}))



