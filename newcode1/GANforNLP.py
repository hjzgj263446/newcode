import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)

dic1 = {0:[0,0,0,1],1:[0,0,1,0],2:[0,0,1,1],3:[0,1,0,0],4:[0,1,0,1],5:[0,1,1,0],6:[0,1,1,1],
        7:[1,0,0,0],8:[1,0,0,1],9:[1,0,1,0]}
x_train,y_train = mnist.train.images,mnist.train.labels
x_test,y_test = mnist.test.images,mnist.test.labels
temp1 = np.zeros((y_train.shape[0],4))
temp2 = np.zeros((y_test.shape[0],4))
j = 0
for i in y_train:
    a = np.argmax(i)
    temp1[j] = dic1[a]
    j += 1
j = 0
for i in y_test:
    a = np.argmax(i)
    temp2[j] = dic1[a]
    j += 1
np.savez(r"C:\Myprogram\DATA"+"\\"+"train.npz",x_train,temp1)
np.savez(r"C:\Myprogram\DATA"+"\\"+"test.npz",x_test,temp2)