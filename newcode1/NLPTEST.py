import pickle
import numpy as np


path = r"C:\Myprogram\DATA\sogouword.txt"
def savef(file,name):
    with open(r"C:\Myprogram\DATA"+"\\"+name+".pkl","wb") as f:
        pickle.dump(file,f)
def conver(i):
    b = np.zeros((1,18))
    i = bin(i)
    i = i[2:].zfill(18)
    t = 0
    for j in i:
        b[:,t] = int(j)
        t += 1
    return b
def w():
    with open(path,'r',encoding="utf-8") as f:
        dic1 = {}
        dic2 = {}
        count = 0
        line = f.read().split(" ")
        for i in line:
            if i not in dic1:
                count += 1
                dic1[i] = conver(count)
                dic2[count] = i
    savef(dic1,"sougoudic1")
    savef(dic2,"sougoudic2")

def re():
    with open(path, 'r', encoding="utf-8") as f:
        print(f.read())

def wree():
    with open(path, 'r', encoding="utf-8") as f:
        with open(r"C:\Myprogram\DATA\sougoudic1.pkl","rb") as f1:
            a = f.read().split(" ")
            length = len(a)
            dic1 = pickle.load(f1)
            list1 = list()
            list2 = list()
            batch_size = 50
            temp1 = np.zeros((batch_size,18))
            temp2 = np.zeros((batch_size,18))
            tadd1 = np.zeros((4,18))
            matrix = dic1.get(a[0])
            temp2[0],temp2[1] = dic1.get(a[1]),dic1.get(a[2])
            temp1[0],temp1[1] = matrix,matrix
            matrix = dic1.get(a[1])
            temp1[2],temp1[3],temp1[4] = matrix,matrix,matrix
            temp2[2],temp2[3],temp2[4] = dic1.get(a[0]),dic1.get(a[2]),dic1.get(a[3])
            i,j,k,m = 2,5,5,0
            while((i+3)<=length):
                matrix = dic1.get(a[i])
                if (k+4)>=batch_size:
                    m += 1
                    temp1[k:] = matrix
                    temp2[k:] = dic1.get(a[i+1])
                    list1.append(temp1)
                    list2.append(temp2)
                    temp1 = np.zeros((batch_size, 18))
                    temp2 = np.zeros((batch_size, 18))
                    k = 0
                else:
                    j += 1
                    temp1[k:k+4] = matrix
                    r1,r2 = dic1.get(a[i+1]),dic1.get(a[i+2])
                    l1,l2 = dic1.get(a[i-1]),dic1.get(a[i-2])
                    temp2[k],temp2[k+1],temp2[k+2],temp2[k+3] = r1,r2,l1,l2
                    i += 1
                    k += 4
            temp1[k:] = matrix
            temp2[k:] = dic1.get(a[i + 1])
            list1.append(temp1)
            list2.append(temp2)
            savef(list1,"trainsample")
            savef(list2, "labelsample")

def reversedic():
    with open(r"C:\Myprogram\DATA\sougoudic2.pkl", "rb") as f1:
        a = pickle.load(f1)
        print(a[1])
wree()