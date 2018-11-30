import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)
x = mnist.train.images
y = mnist.train.labels
t = x.reshape((55000,28,28))
t1 = np.zeros((55000,28,28))
t2 = np.zeros((55000,28,28))
t3 = np.zeros((55000,28,28))
for i in range(t.shape[0]):
    t1[i] = np.rot90(t[i])
    t2[i] = np.rot90(t1[i])
    t3[i] = np.rot90((t2[i]))


np.savez(r'C:\Myprogram\mnistdata'+'\\'+'transdata.npz',t,t1,t2,t3,y)
