"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api

"""
注意
1. compute的输入输出是NDArray, gradient的输入输出是Tensor
2. 
"""
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)


    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return self.scalar * (power_scalar(a, self.scalar-1)) * out_grad


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a/b

    def gradient(self, out_grad, node):
        a,b = node.inputs
        return out_grad/b, -a*out_grad/(b*b)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a/self.scalar

    def gradient(self, out_grad, node):
        return out_grad/self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        else:
            return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)

def reshape(a, shape):
    return Reshape(shape)(a)

"""
广播机制的条件
(1). 如果两个数组的维度相同，对应位置上轴的长度相同或其中一个的轴长度为1,广播兼容，可在轴长度为1的轴上进行广播机制处理。

(2). 如果两个数组的维度不同，那么给低维度的数组前扩展提升一维，扩展维的轴长度为1,然后在扩展出的维上进行广播机制处理。

假设输入是input,输出是output
input:[0 1 2 3]
output:[[[0 1 2 3]
  [0 1 2 3]
  [0 1 2 3]
  [0 1 2 3]]

 [[0 1 2 3]
  [0 1 2 3]
  [0 1 2 3]
  [0 1 2 3]]]
"""
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape #如(1)
        # 将延伸出的维度进行加和,out_grad由(2,3,4)->(4)
        ret = summation(out_grad,tuple(range(len(out_grad.shape) - len(input_shape))))
        # 可能出现某一维度上由1复制的情况
        """
        out_grad.shape[i]!=1 多加这个判断条件的原因：
        刚开始不加这个判断条件依然通过了测试。但是在做hw2的Layernorm1d时
        在ret = summation(ret, axes=i)时报错：axes=1 but array_size = 1
        后来发现存在这样一种情况
        Broadcastto将(1,1)广播成(10,1),在处理这个循环时，第一轮i=0且dim=1，将第0维进行summation，ret.shape = (1,)
        当第二轮i=1且dim=1,依旧会进行summation,但是此时ret.shape(1,)，不存在第二维，因此报错
        所以需要再加一个限制条件，判断这个维度上的原始长度是不是1
        """
        for i, dim in enumerate(input_shape):
            if dim == 1 and out_grad.shape[i]!=1:
                    ret = summation(ret, axes=i)
        return reshape(ret, input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        shape = list(a.shape)
        axes = self.axes
        
        if axes is None:
            axes = list(range(len(shape)))
            for _ in axes:
                shape[_] = 1
        elif isinstance(axes, int):
            shape[axes] = 1
        else:
            for _ in axes:
                shape[_] = 1
        
        return broadcast_to(reshape(out_grad, shape), a.shape)
"""
输入为的shape为3*2*4*5,在第 0 和 2 的维度上做 summation,输出的shape为2*5
self.axes = (0,2) 
执行
    for _ in axes:
            shape[_] = 1
将输入的shape在0和2维度上的长度变为1,目的是让out_grad可以reshape为(1,2,1,5)的形状
然后通过broadcast,在第0维和第2维上进行复制,
在某个维度上sum，相当于y=x1+x2+...+xk,那么对z对x1的导数和对y的导数一样,因此复制out_grad就行
"""

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a@b

    def gradient(self, out_grad, node):
        a,b = node.inputs
        # 在a,b维度完全一致时
        adjoint1 = out_grad @ transpose(b)
        adjoint2 = transpose(a) @ out_grad
        # 出现广播,将多余的维度加和
        adjoint1 = summation(adjoint1, axes=tuple(range(len(adjoint1.shape) - len(a.shape))))
        adjoint2 = summation(adjoint2, axes=tuple(range(len(adjoint2.shape) - len(b.shape))))
        return adjoint1, adjoint2
        
        


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)

# 操作的实际还是array来计算
class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(0, a)

    def gradient(self, out_grad, node):
        a = node.realize_cached_data()
        return Tensor(array_api.where(a>0, 1, 0)) * out_grad


def relu(a):
    return ReLU()(a)
