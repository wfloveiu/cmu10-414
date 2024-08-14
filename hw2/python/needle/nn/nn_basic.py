"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


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
    # 通过对象名.(输入X)来进行前向传播
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
        # print(type(modules))

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
        
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        """
        下边两个并不是Parameter，因为他们不是权重，在最后一个测试时发现
        """
        self.running_mean = init.zeros(dim, device=device, dtype=dtype) #运行时每一dim上的均值方差
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1,-1)), x.shape)
        
        if self.training:    
        # 计算当前batch的均值
            sum_x = ops.summation(x, 0) #(batch_size, features)->(features)
            mean_x = ops.divide_scalar(sum_x, batch_size)
            broadcast_mean = ops.broadcast_to(mean_x, x.shape)

            # 计算当前batch的标准差
            pow_Var = ops.power_scalar(x-broadcast_mean, 2)
            sum_Var = ops.summation(pow_Var, 0) #(batch_size, features)->(features)
            Var = ops.divide_scalar(sum_Var, batch_size)
            Var_add_eps = ops.power_scalar(ops.add_scalar(Var, self.eps), 1/2)
            broadcast_var = ops.broadcast_to(Var_add_eps, x.shape)
            
            out = broadcast_weight * (x - broadcast_mean) / broadcast_var + broadcast_bias
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * Var
        else:
            broadcast_running_mean = ops.broadcast_to(ops.reshape(self.running_mean, (1,-1)), x.shape)
            running_var_add_eps = ops.power_scalar(ops.add_scalar(self.running_var, self.eps), 1/2)
            broadcast_running_var = ops.broadcast_to(ops.reshape(running_var_add_eps, (1,-1)), x.shape)
            out = broadcast_weight * (x-broadcast_running_mean) / broadcast_running_var + broadcast_bias
        
        return out
"""
开始时计算E时，想直接使用.data来计算，不想创建复杂的计算图。后来意识到这计算过程就是需要创建计算图的。乐
"""
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
