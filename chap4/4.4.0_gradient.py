import numpy as np
import matplotlib.pylab as plt

def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열, 값은 모두 0
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def function_2(x):
    return x[0]**2 + x[1]**2

# print(numerical_gradient(function_2, np.array([3.0, 4.0])))
# print(numerical_gradient(function_2, np.array([0.0, 2.0])))
# print(numerical_gradient(function_2, np.array([3.0, 0.0])))

def gradient_descent(f, init_x, lr, step_num):
    # lr = learning rate
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr*grad
    return x, np.array(x_history)
# print(gradient_descent(function_2, np.array([-3.0, 4.0]), lr=0.1, step_num=100))    # res = almost [0, 0]
init_x = np.array([-3.0, -4.0])

lr = 0.1    # 학습률이 결과에 영향을 미침(너무 크면 발산, 너무 작으면 진행 x), =hyper parameter
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()