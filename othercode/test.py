import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def save(file_name,file):
    with open(r'C:\Myprogram\mnistdata' + '\\' + file_name+'.pkl', 'wb') as f:
        pickle.dump(file, f)

mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)
print(mnist.train.labels[:20,:])
l = np.argmax(mnist.train.labels[:50000,:],1)
l = list(l)
l_5 = []
l_6 = []
l_2 = []
for i in range(len(l)):
    if l[i] == 8:
        l_6.append(i)
    elif l[i] == 1:
        l_5.append(i)
    elif l[i] == 2:
        l_2.append(i)
imgas8 = mnist.train.images[l_6]
imgas1 = mnist.train.images[l_5]
imgas2 = mnist.train.images[l_2]
save('image8',imgas8)
save('image1',imgas1)
save("image2",imgas2)