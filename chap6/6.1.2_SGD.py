# 확률적 경사 하강법, 앞에서 계속 다룬방식
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]