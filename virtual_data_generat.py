import numpy as np
import csv

def noise():
    return np.round(np.random.uniform(low=-5.0, high=5.0, size=(1,288)),2)

def re_data():
    return np.round(np.random.uniform(low=1.0, high=50.0, size=(1,288)),2)
def generate(num):
    x = re_data()
    a = x 

    for _ in range(1, 51):
        a_ = x + noise()
        a = np.concatenate((a,a_),axis=0)

    with open('{n}.csv'.format(n=num), 'w',newline='') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerows(a)

if __name__=="__main__":
    for i in range(1,8):
        generate(i)