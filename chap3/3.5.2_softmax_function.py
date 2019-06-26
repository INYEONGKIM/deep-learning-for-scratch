import numpy as np

def softmax(a):
    # softmax == probability
    c = np.max(a) # to prevent overflow
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = softmax(np.array([0.3, 2.9, 4.0]))
print(a)
print(np.sum(a))    # characteristic of softmax