import numpy as np
import csv
import matplotlib.pyplot as plt

data = np.genfromtxt('full.csv', dtype=None, delimiter=',',skip_header=0)
# rng = np.random.default_rng()
# rng.shuffle(data, axis=0)
# # print(data.shape)

X = data[:,:288]
y = data[:,288]


ones = np.ones((X.shape[0],1))
X_b = np.concatenate((X,ones), axis=1)

mean = 0.000001
epochs = 500000

theta = np.random.uniform(low=-2.0,high=2.0,size=(289,1))
print(theta.shape)

losses = []

for _ in range(epochs):
    for i in range(0,X.shape[0]):
        xi = X_b[i:i+1]
        yi = y[i:i+1]

        y_hat = np.dot(xi,theta)
        # print(y_hat)
        # print(_)
        lossi = (y_hat - yi)*(y_hat - yi)
        
        l = 2*xi*(y_hat-yi)
        l_reshape = np.reshape(l,(289,1))
        theta = theta - mean * l_reshape

        losses.append(lossi)

with open('result.csv', 'w',newline='') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerows(np.round((np.reshape(theta,(1,289))),2))
# print(losses)
