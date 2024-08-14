from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z): # 再次记住，在进行compute的时候输入输出是array，要使用array_api中的函数
        max_z_f = array_api.max(Z, axis=self.axes, keepdims=False) #和Z的维度一致，每一行中每个元素都是这一行的最大值
        max_z_t = array_api.max(Z, axis=self.axes, keepdims=True) # 公式最后的那个maxz
        tmp = array_api.sum(array_api.exp(Z - max_z_t),axis=self.axes, keepdims=False)
        out = array_api.log(tmp) + max_z_f
        return out

    # 疑问：maxz对z的梯度为0吗，为什么不计算
    def gradient(self, out_grad, node):
        input = node.inputs[0].cached_data
        max_val_t = array_api.max(input, axis=self.axes, keepdims=True)
        exp_val = array_api.exp(input - max_val_t)
        sum_val = array_api.sum(exp_val, axis=self.axes, keepdims=False)
        # log_val = array_api.log(sum_val)
        
        #把log中看成整体，先求对log的梯度
        grad_log = out_grad.cached_data / sum_val 
        
        #求sum的梯度，和summation一样
        shape = list(node.inputs[0].shape)
        axes = self.axes
        if axes is None:
           axes = list(range(len(shape))) #axes为None是对所有维度求和，相当于axes=(0,1,..)
        for _ in axes:
            shape[_] = 1
        grad_sum = array_api.broadcast_to(array_api.reshape(grad_log, shape), node.inputs[0].shape)
        # broadcast_to(reshape(grad_log, shape), node.inputs[0].shape)
        
        #求exp的梯度
        grad_exp = grad_sum * exp_val
        
        return Tensor(grad_exp)



def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

