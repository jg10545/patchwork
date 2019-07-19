# -*- coding: utf-8 -*-
import numpy as np
from patchwork._util import shannon_entropy





def test_shannon_entropy():
    maxent = np.array([[0.5,0.5]])
    assert shannon_entropy(maxent)[0] == 1.0