import numpy as np
import sys, os
sys.path.append(os.pardir)

def cross_entropy_error(y, t):
    delta = 1e-7    # minus inf 방지용
    return -np.sum(t*np.log(y+delta))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)   # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

net = simpleNet()
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print("predict :",p)
print("max idx :", np.argmax(p))

t = np.array([0, 0, 1])
print("loss :", net.loss(x,t))