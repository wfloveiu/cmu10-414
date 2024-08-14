import sys

sys.path.append("./python")
sys.path.append("./apps")
from simple_ml import *
import numdifftools as nd

import numpy as np
import mugrade
import needle as ndl

a = ndl.Tensor([[4.0], [4.55]])
b = ndl.log(a)