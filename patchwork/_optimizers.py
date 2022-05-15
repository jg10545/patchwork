# -*- coding: utf-8 -*-
import math
import re
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op


class CosineDecayWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps,
                  alpha=0.01, warmup_steps=2000, name=None):
        
        """
        Cosine decay with linear warmup.
        
        :initial_learning_rate: A scalar `float32` or `float64` Tensor or a
             Python number. The initial learning rate.
        :decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps to decay over.
        :alpha: A scalar `float32` or `float64` Tensor or a Python number.
             Minimum learning rate value as a fraction of initial_learning_rate.
        :warmup_steps: number of steps to warm up over- starts at 0 and scales
            linearly to initial_learning_rate
        :name: String. Optional name of the operation.  Defaults to 'CosineDecay'.
        """
    
        super(CosineDecayWarmup, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.name = name

    def _warmupstep(self, step):
        return self.initial_learning_rate*step/self.warmup_steps
    
    def _decaystep(self, step):
        step -= self.warmup_steps
        initial_learning_rate = ops.convert_to_tensor_v2_with_dispatch(
                    self.initial_learning_rate, name="initial_learning_rate")
        dtype = initial_learning_rate.dtype
        decay_steps = math_ops.cast(self.decay_steps, dtype)
            
        global_step_recomp = math_ops.cast(step, dtype)
        global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
        completed_fraction = global_step_recomp / decay_steps
        cosine_decayed = 0.5 * (1.0 + math_ops.cos(
            constant_op.constant(math.pi) * completed_fraction))
        
        decayed = (1 - self.alpha) * cosine_decayed + self.alpha
        return math_ops.multiply(initial_learning_rate, decayed)
    
    def __call__(self, step):
        with ops.name_scope_v2(self.name or "CosineDecay"):
            return tf.cond(step <= self.warmup_steps,
                           lambda: self._warmupstep(step),
                           lambda: self._decaystep(step))
        

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "warmup_steps":self.warmup_steps,
            "name": self.name}






class LARSOptimizer(tf.keras.optimizers.Optimizer):
    """Basic LARS implementation, based on the one at
    https://github.com/google-research/simclr/blob/master/tf2/lars_optimizer.py
    """

    def __init__(self,
               learning_rate,
               momentum=0.9,
               weight_decay=0.0,
               eeta=0.001,
               name="LARSOptimizer"):
        """
        learning_rate: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """
        super(LARSOptimizer, self).__init__(name)

        self._set_hyper("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eeta = eeta
        self.exclude = ['batch_normalization', 'bias']
        
    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, "Momentum")

    def _resource_apply_dense(self, grad, param, apply_state=None):
        if grad is None or param is None:
            return tf.no_op()

        var_device, var_dtype = param.device, param.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
        learning_rate = coefficients["lr_t"]

        param_name = param.name

        v = self.get_slot(param, "Momentum")

        if self._use_weight_decay(param_name):
            grad += self.weight_decay * param

        trust_ratio = 1.0
        if self._do_layer_adaptation(param_name):
            w_norm = tf.norm(param, ord=2)
            g_norm = tf.norm(grad, ord=2)
            trust_ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (self.eeta * w_norm / g_norm), 1.0),
                1.0)
        scaled_lr = learning_rate * trust_ratio

        next_v = tf.multiply(self.momentum, v) + scaled_lr * grad
        update = next_v
        next_param = param - update



        return tf.group(*[
            param.assign(next_param, use_locking=False),
            v.assign(next_v, use_locking=False)
            ])

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        for r in self.exclude:
            if re.search(r, param_name) is not None:
                return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        for r in self.exclude:
            if re.search(r, param_name) is not None:
                return False
        return True

    def get_config(self):
        config = super(LARSOptimizer, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "eeta": self.eeta
        })
        return config