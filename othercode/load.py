import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np

def openfile(name):
    with open(r'C:\Myprogram\mnistdata' + '\\' + name+'.pkl', 'rb') as f:
        x = pickle.load(f)
    return x

x = openfile("image2")

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(r'C:\Myprogram\mnistdata' + '\\' + 'model.ckpt.meta')
    saver.restore(sess,r'C:\Myprogram\mnistdata' + '\\' +'model.ckpt')
    pre = tf.get_collection("pre_img")
    graph = tf.get_default_graph()
    fake = graph.get_operation_by_name("fake").outputs[0]
    keep_drop = graph.get_operation_by_name("keep_drop").outputs[0]
    p = sess.run(pre,feed_dict={fake:x[:50,:],keep_drop:1})
    p = np.squeeze(np.array(p))
    for i in range(20):
        plt.imshow(p[i].reshape([28,28]))
        plt.show()
