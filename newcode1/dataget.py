import numpy as np
import pickle

def getData(i,j):
    path = r"C:\Myprogram\mnistdata\transdata.npz"
    with np.load(path) as f:
        a0 = f["arr_0"][i:j,:,:].reshape((-1,28*28))
        a1 = f["arr_1"][i:j,:,:].reshape((-1,28*28))
        a2 = f["arr_2"][i:j,:,:].reshape((-1,28*28))
        a3 = f["arr_3"][i:j,:,:].reshape((-1,28*28))
        y = f["arr_4"][i:j,:]
    cp = np.array([a0,a1,a2,a3])
    return cp,y


def save(i,name,file):
    with open(r'C:\Myprogram\mnistdata'+'\\'+str(i)+name+'.pkl', 'wb') as f:
        pickle.dump(file,f)

def getD(i,j,name):
    with np.load(r"C:\Myprogram\DATA"+"\\"+name+".npz") as f:
        x = f["arr_0"][i:j,:]
        y = f["arr_1"][i:j,:]
        return x,y


