import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sklearn.externals import joblib
import time

#a = Image.open(r"C:\Users\62271\Desktop\4.jpg")
'''a = a.resize((120, 120))
a = np.array(a)
a = a.reshape((120*120,3))'''
#b = Image.open(r"C:\Users\62271\Desktop\3.jpg")
'''b = b.resize((120, 120))
b = np.array(b)
b = b.reshape((120*120,3))
c = [a,b]
print(np.array(c).shape)
plt.imshow(a)
plt.show()'''

def getimage(file_dir):
    c = []
    for file in os.listdir(file_dir):
        path = file_dir + '\\' + file
        a = Image.open(path)
        a = a.resize((120,120))
        a = np.array(a)
        a = a.reshape((120 * 120, 3))
        c.append(a)
    return np.array(c)

def dumpimage(sub):
    path = r"C:\迅雷下载\kaggle\train"+'\\'+sub
    t = time.time()
    image = getimage(path)
    joblib.dump(image,r"C:\迅雷下载\kaggle\train\test\cat120.m")
    t1 = time.time()
    print(t1-t)

def getXY():
    path1 = r"C:\迅雷下载\kaggle\train\test\dog120.m"
    x = joblib.load(path1)
    return x

def show(img):
    x = img[3:4,:]
    plt.imshow(x.reshape((120,120,3)))
    plt.show()
#dumpimage("dog")
dumpimage("cat")