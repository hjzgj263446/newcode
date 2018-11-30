import matplotlib.pyplot as plt
import pickle
import numpy as np


def show2(i):
    # 展示图片
    with open(r'C:\Myprogram\mnistdata'+'\\'+i+'2cganfake.pkl', 'rb') as f:
        with open(r'C:\Myprogram\mnistdata' + '\\' +i+ '2caganreal.pkl', 'rb') as f1:
            with open(r'C:\Myprogram\mnistdata' + '\\' + i + '2canfake3.pkl', 'rb') as f2:
                samples = pickle.load(f)
                samples1 = pickle.load(f1)
                samples2 = pickle.load(f2)
                fig = plt.figure(figsize=(6,6))
                for j in range(0,27):
                    a = fig.add_subplot(9,9,j+1)
                    image = samples[j:j+1,:].reshape((28,28))
                    a.axis("off")
                    a.imshow(image)
                    a = fig.add_subplot(9,9,j+28)
                    a.axis("off")
                    image1 = samples1[j:j+1, :].reshape((28, 28))
                    a.imshow(image1)
                    a = fig.add_subplot(9,9,j+55)
                    image2 = samples2[j:j+1, :].reshape((28, 28))
                    a.axis("off")
                    a.imshow(image2)
                plt.show()

def show(name):
    # 展示图片
    with open(r'C:\Myprogram\mnistdata'+'\\'+name+'.pkl', 'rb') as f:
        samples = pickle.load(f)
    fig = plt.figure(figsize=(5,5))
    for j in range(0,20):
        a = fig.add_subplot(5,5,j+1)
        a.axis("off")
        image = samples[j:j+1,:].reshape((28,28))
        a.imshow(image)
    plt.show()

def pltshow(a):
    plt.imshow(a)
    plt.show()

def openfile(name):
    with open(r'C:\Myprogram\mnistdata' + '\\' + name+'.pkl', 'rb') as f:
        x = pickle.load(f)
    for j in range(1,20):
        plt.imshow(x[j-1:j,:].reshape((120,120)))
        plt.show()

def juli(name):
    with open(r'C:\Myprogram\mnistdata'+'\\'+name+'.pkl', 'rb') as f:
        samples = pickle.load(f)
    i = 1
    a = 0
    count = 0
    while(i<20):
        a += np.sqrt(np.sum(np.multiply(samples[i-1:i,:] - samples[i:i+1,:],samples[i-1:i,:] - samples[i:i+1,:])))
        count += 1
        i += 1
        print(a)
    return a/count

def run(num,name=0):
    if num == 1:
        show("g_fake10000")
    elif num == 3:
        show(name)
    elif num == 2:
        a = juli(name)
    return a

def l2l(name1,name2):
    with open(r'C:\Myprogram\mnistdata'+'\\'+name1+'.pkl', 'rb') as f:
        samples1 = pickle.load(f)
    with open(r'C:\Myprogram\mnistdata'+'\\'+name2+'.pkl', 'rb') as f:
        samples2 = pickle.load(f)
    i = 1
    count = 0
    a = 0
    while(i<20):
        a1 = samples1[i-1:i,:]
        a2 = samples2[i-1:i,:]
        a += np.sqrt(np.sum(np.multiply(a1-a2,a1-a2)))
        count += 1
        i += 1
    return a/count

def showing(name):
    with open(r'C:\Myprogram\mnistdata'+'\\'+name+'.pkl', 'rb') as f:
        samples = pickle.load(f)
    fig = plt.figure(figsize=(5,5))
    count = 0
    for i in samples:
        for j in range(1,6):
            a = fig.add_subplot(5,5,j+count)#add_sbuplot 是将画布分为5行5列的块，图片展示在第i块
            a.axis('off')
            image = i[j-1:j,:].reshape((28,28))
            a.imshow(image)
        count += j
    plt.show()

def loady(name):
    with open (r'C:\Myprogram\mnistdata'+'\\'+name+'.pkl', 'rb') as f:
        y = pickle.load(f)
        #print((np.argmax(y[0:21,:],1)))
        print(y[0:20,:])
def loady2(name):
    dic1 = {0: [0, 0, 0, 1], 1: [0, 0, 1, 0], 2: [0, 0, 1, 1], 3: [0, 1, 0, 0], 4: [0, 1, 0, 1], 5: [0, 1, 1, 0],
            6: [0, 1, 1, 1],
            7: [1, 0, 0, 0], 8: [1, 0, 0, 1], 9: [1, 0, 1, 0]}
    with open (r'C:\Myprogram\mnistdata'+'\\'+name+'.pkl', 'rb') as f:
        y = pickle.load(f)
        a = y[0:20,:]

if __name__ == '__main__':
    #a = run(2,"g_fake10000")
    #b = run(3,"image8")
    #c= l2l("image8","image2")
    #print(a,b,c)
    #show("g_fake10000")
    #show("realimg10000")
    show2("25000")
    #show("1500cganfake")
    #openfile("image1")
    #show("image8")
    #pltshow(-np.ones([28,28]))
    #show("advc1")
    #show("111advsampel")
    #show("50000sampleswgan")