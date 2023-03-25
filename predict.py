import numpy as np

theta = np.genfromtxt('result.csv', dtype=None, delimiter=',',skip_header=0)



data = np.genfromtxt('full.csv', dtype=None, delimiter=',',skip_header=0)

print(data[0:1,0:288].shape)
x = np.reshape(data[0:1,0:288],(288,))
x = np.concatenate((x,np.array([1])),axis=0)

print(x.shape)

predict = np.dot(theta,x)
print(predict)
print(np.array([1,2])*np.array([5,6]))