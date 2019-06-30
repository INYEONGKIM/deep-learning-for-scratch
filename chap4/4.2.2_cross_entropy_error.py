import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7    # minus inf 방지용
    return -np.sum(t*np.log(y+delta))

t = [0]*10
t[2] = 1    # one-hot encoding

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]    # 2
print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]    # 7
print(cross_entropy_error(np.array(y), np.array(t)))