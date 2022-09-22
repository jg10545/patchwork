import numpy as np
import pandas as pd

from patchwork._eval import _add_label_noise, _split_domains, _get_accuracy

def test_add_label_noise():
    X = np.random.randint(0, 2, size=(7 ,5))

    assert (X == _add_label_noise(X ,0)).all()
    assert (X != _add_label_noise(X ,1)).all()
    assert (X != _add_label_noise(X, 0.5)).any()


def test_split_domains():
    N = 100
    # generate random subset data
    subset = np.random.choice(["foo", "bar", "foobar"], size=100, replace=True)

    subA, subB, domainA, domainB = _split_domains(subset)

    assert len(domainA & domainB) == 0
    assert 2 in [len(domainA), len(domainB)]
    assert 1 in [len(domainA), len(domainB)]
    assert subA.sum() + subB.sum() == len(subset)
    assert (subA * subB).sum() == 0


def test_get_accuracy():
    N = 10000
    d = 3
    X = np.random.normal(0, 1, size=(N, d))
    Y = (X[:, :2] > 0).astype(int)

    train_acc, test_acc = _get_accuracy(X, Y, X, Y)
    assert test_acc > 0.99
