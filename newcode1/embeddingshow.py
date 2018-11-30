import numpy as np
import pickle


pathread = r"C:\Myprogram\DATA\\"
pathwrite = r'C:\Myprogram\mnistdata\\'

'''from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Myprogram\mnistdata',one_hot=True)

pathwrite = r'C:\Myprogram\mnistdata\\'
def getMnist():
    x,y = mnist.test.images,mnist.test.labels
    temp = np.zeros((y.shape[0],1))
    j = 0
    print(y.shape)
    temp = np.argmax(y,1)
    np.savetxt(pathwrite+"mx.tsv",x,delimiter="\t")
    np.savetxt(pathwrite+"my.tsv",temp,delimiter="\t")'''


def turn(num):
    # with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+"label"+str(i)+'.pkl', 'rb') as f:
    with open(r"C:\Myprogram\DATA\sougoudic2.pkl", "rb") as f1:
        b = pickle.load(f1)
        # a = pickle.load(f)
        t = ""
        li = []
        for i in num:
            # for j in i:
            t += str(int(i))
        li.append(b[int(t, 2)])
        t = ""
        print(li)


def openfile(i):
    with open(r'C:\Myprogram\mnistdata' + '\\' + i + '.pkl', 'rb') as f:
        a = pickle.load(f)
        return a


def trainsample():
    with open(r"C:\Myprogram\DATA" + "\\" + "trainsample" + ".pkl", "rb") as f:
        a = pickle.load(f)
        for i, j in zip(a[0], a[1]):
            print(i - j)
        print(len(a))
        l1 = a[10]
        turn(l1)


def savetsv(name, i):
    if i == 0:
        s = openfile(name)[0]
    else:
        s = openfile(name)
        print(s.shape, type(s))
    return s


# trainsample()
def ll():
    f = (openfile("100000allfake")[0])[0:50, :]
    f1 = (openfile("100000alllabel")[0:50, :])
    i, j = 3, 29
    turn(f1[i, :])
    turn(f1[j, :])
    print(f[i, :], f[j, :])
    f = np.sum(np.square(f[i, :] - f[j, :]))
    print(f)


def zhuan(num):
    with open(r"C:\Myprogram\DATA\sougoudic2.pkl", "rb") as f1:
        b = pickle.load(f1)
        t = ""
        li = []
        for i in num:
            for j in i:
                t += str(int(j))
            li.append(b[int(t, 2)])
            t = ""
        return np.array(li)


# ll()
# getMnist()
# print(openfile("20000fake"))
np.savetxt(pathwrite+"ww.tsv",savetsv("200000allfake",0),delimiter="\t")
np.savetxt(pathwrite+"1ww.tsv",zhuan(savetsv("200000alllabel",1)),fmt="%s",encoding="utf-8")
