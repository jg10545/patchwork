# -*- coding: utf-8 -*-
import math
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

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "CosineDecay"):
            if step < self.warmup_steps:
                return self.initial_learning_rate*step/self.warmup_steps
            else:
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

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "warmup_steps":self.warmup_steps,
            "name": self.name}
