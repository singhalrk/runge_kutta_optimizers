from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export
import tensorflow as tf


"""
Runge Kutta 2nd Order - Heun's Method
x(t + 1) = x(t) - (lr / 2) * (k1 + k2)
k1 = grad L(x(t))
k2 = grad L(x(t) - lr * k1)
"""


class RK2heunOptimizer(optimizer.Optimizer):
    def __init__(self, lr=0.1, loss, use_locking=False, name="RK2heunoptimizer"):

        super(RK2heunOptimizer, self).__init__(use_locking, name)
        self._lr = lr
        self._lr_tensor = None
        # self._opt = optimizer.SGD()
        # self.loss = loss

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, 'k1', self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)


        #calculate k1
        k1 = self.get_slot(var, "k1")
        k1_t = optimizer.
