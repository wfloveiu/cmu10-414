import math
from .init_basic import *

# 均匀分布 U(-a, a)
def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    out = rand(fan_in, fan_out, low=-a, high=a)
    return out
# 正太分布 N(0, std)
def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    out = randn(fan_in, fan_out, mean=0, std=std)
    return out

# 均匀分布 U(-bound, bound)
def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    out = rand(fan_in, fan_out, low=-bound, high=bound)
    return out


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    out = randn(fan_in, fan_out, mean=0, std=std)
    return out
