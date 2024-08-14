"""Optimization module"""
import needle as ndl
import numpy as np
from collections import defaultdict

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params) # params是包含多个需要进行多次迭代计算的Param
        self.lr = lr
        self.momentum = momentum
        self.u = {} # 字典，key是Param，value是该Param上一次计算时的梯度
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            grad  = self.u.get(param, 0) * self.momentum + (1-self.momentum) * (param.grad.data + self.weight_decay * param.data)
            grad = ndl.Tensor(grad, dtype = param.dtype)
            self.u[param] = grad
            param.data =  param.data - self.lr * grad
        

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION




class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is not None:
                grad_with_L2rgl = param.grad.data + self.weight_decay * param.data
            else:
                grad_with_L2rgl = self.weight_decay * param.data
            new_m = self.beta1 * self.m.get(param, 0) + (1-self.beta1) * grad_with_L2rgl.data
            new_v = self.beta2 * self.v.get(param, 0) + (1-self.beta2) * grad_with_L2rgl.data * grad_with_L2rgl.data
            self.m[param] = new_m
            self.v[param] = new_v
            m_hat = new_m.data / (1 - self.beta1 ** self.t)
            v_hat = new_v.data / (1 - self.beta2 ** self.t)
            out = param.data - self.lr * m_hat / (ndl.ops.power_scalar(v_hat, 1/2) + self.eps)
            out = ndl.Tensor(out, dtype=param.dtype)
            param.data = out