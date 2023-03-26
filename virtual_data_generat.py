import numpy as np
import csv

def noise():
    return np.round(np.random.uniform(low=-1.0, high=1.0, size=(1,288)),2)

def re_data():
    return np.round(np.random.uniform(low=1.0, high=50.0, size=(1,288)),2)

def label(j):
    c = np.random.uniform(low=j,high=j,size=(50,1)) 
    return c

def generate():
    x = re_data()
    a = x 
    for _ in range(1, 50):
        a_ = x + noise()
        a = np.concatenate((a,a_),axis=0)
    return a

def full_data():
    tt = np.concatenate((generate(),label(1)), axis=1)
    for i in range (2, 8):
        a = generate()
        b = np.concatenate((a,label(i)), axis=1)
        tt = np.concatenate((tt,b), axis=0)
    with open('full.csv', 'w',newline='') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerows(tt)

if __name__=="__main__":
    full_data()
