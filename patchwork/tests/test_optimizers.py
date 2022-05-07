# -*- coding: utf-8 -*-
from patchwork._optimizers import CosineDecayWarmup


def test_cosinedecaywarmup():
    lr = 0.01
    decay_steps = 100000
    warmup_steps = 1000
    alpha = 0.01
    schedule = CosineDecayWarmup(lr, decay_steps,
                                 warmup_steps=warmup_steps,
                                 alpha=alpha)
    # start at 0
    assert schedule(0) == 0
    # linear warmup
    assert schedule(warmup_steps/2) == lr/2
    assert schedule(warmup_steps) == lr
    # decay to alpha*lr
    assert schedule(10*decay_steps) == lr*alpha
