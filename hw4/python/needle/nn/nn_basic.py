"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.have_bias = bias
        
        # weight和bias需要定义成Parameter类型
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features,device = device, dtype = dtype))
        if self.have_bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device = device, dtype = dtype).reshape((1, out_features)))

    def forward(self, X: Tensor) -> Tensor:
        if self.have_bias:
            bias = ops.broadcast_to(self.bias, (X.shape[0], self.out_features))
            return ops.matmul(X, self.weight) + bias
        else:
            return ops.matmul(X, self.weight)


class Flatten(Module):
    def forward(self, X):
        batch_size = X.shape[0]
        dim = 1
        for i in range(1, len(X.shape)):
            dim *= X.shape[i]
        return ops.reshape(X, (batch_size, dim))



class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        input = x
        for module in self.modules:
            input = module(input)
        return input


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        softmax = ops.logsumexp(logits, axes=(1,))
        
        shape =  logits.shape
        batch_size = shape[0]
        num_class = shape[1]
        y_one_hot = init.one_hot(num_class, y)
        I = ops.summation(logits * y_one_hot, axes=(1,))
        # print(I)
        return ops.summation(softmax - I) / batch_size


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(1, dim, device=device, dtype=dtype)
        self.running_var = init.ones(1, dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        broadcast_weight = ops.broadcast_to(self.weight, x.shape)
        broadcast_bias = ops.broadcast_to(self.bias, x.shape)
        if self.training:
            # E[x]
            mean_x_batch = ops.summation(x, axes=0) / batch_size
            mean_x_batch = ops.reshape(mean_x_batch, (1, mean_x_batch.shape[0])) # 这里（1，-1）是因为按列求平均了，下面的 LayerNorm1d 是按行求平均
            broadcast_mean_x_batch = ops.broadcast_to(mean_x_batch, x.shape)

            # Var[x]
            var_batch = (x - broadcast_mean_x_batch) ** 2
            var_batch = ops.summation(var_batch, axes=0) / batch_size
            var_batch = ops.reshape(var_batch, (1, var_batch.shape[0]))
            std_dev_batch = (var_batch + self.eps) ** (0.5)
            broadcast_std_dev_batch = ops.broadcast_to(std_dev_batch, x.shape)

            # Normalize
            x_norm = broadcast_weight * (x - broadcast_mean_x_batch) / broadcast_std_dev_batch + broadcast_bias

            # Update running mean and var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x_batch
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_batch
        else:
            broadcast_running_mean = ops.broadcast_to(self.running_mean, x.shape)
            broadcast_running_var = ops.broadcast_to(self.running_var, x.shape)
            x_norm = broadcast_weight * (x - broadcast_running_mean) / (broadcast_running_var + self.eps) ** (0.5) + broadcast_bias
        return x_norm

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # 初始化参数
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        features = x.shape[1]
        
        # 计算均值
        sum_x = ops.summation(x, 1) #计算完后维度降低了,原来是(batch_size, features)现在是(batch_size,)
        mean_x = ops.divide_scalar(sum_x, features)
        mean_x_reshape = ops.reshape(mean_x, (-1,1))

        E = ops.broadcast_to(mean_x_reshape, x.shape)

        # 计算标准差
        V_inner = ops.power_scalar(x-E, 2)
        V_sum = ops.summation(V_inner, 1)   # (batch_size,)
        V_mean = ops.divide_scalar(V_sum, features)
        V = ops.add_scalar(V_mean, self.eps)
        sqrt_Var = ops.power_scalar(V, 1/2)    # (batch_size,)
        sqrt_Var_reshape = ops.reshape(sqrt_Var, (-1,1))
        sqrt_Var_reshape_brocst = ops.broadcast_to(sqrt_Var_reshape, x.shape) # (batch_size,feature)
        
        # 计算X
        X  = (x - E) / sqrt_Var_reshape_brocst
        
        # weight和bias的维度都是dim即features
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1,-1)), x.shape)
        
        out = broadcast_weight * X + broadcast_bias
        return out


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p = 1- self.p) / (1 - self.p)
            x = x * mask
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
