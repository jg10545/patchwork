# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from patchwork._sample import find_unlabeled, find_fully_labeled
from patchwork._sample import find_partially_labeled, find_subset
from patchwork._sample import stratified_sample, unlabeled_sample


testdf = pd.DataFrame({
        "filepath":["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"],
        "exclude":[True,  False, False, False, False],
        "validation":[False,  False, False, False, False],
        "class1":[None, 1, 0, 1, 1],
        "class2":[None, 0, 1, None, None],
        "subset":[0,0,1,1,1]
        })


def test_find_unlabeled():
    unlab = find_unlabeled(testdf)
    assert unlab.sum() == 1
    assert "a.jpg" in testdf["filepath"][unlab].values

def test_find_fully_labeled():
    flab = find_fully_labeled(testdf)
    assert flab.sum() == 2
    assert "b.jpg" in testdf["filepath"][flab].values


def test_partially_unlabeled():
    plab = find_partially_labeled(testdf)
    assert plab.sum() == 2
    assert "d.jpg" in testdf["filepath"][plab].values

def test_find_subset_subset_column():
    s0 = find_subset(testdf, "subset: 0")
    assert s0.sum() == 1
    assert "b.jpg" in testdf["filepath"][s0].values # a.jpg is excluded
    
    s1 = find_subset(testdf, "subset: 1")
    assert s1.sum() == 3
    for f in ["c.jpg", "d.jpg", "e.jpg"]:
        assert f in testdf["filepath"][s1].values
        


def test_stratified_sampler():
    N = 100
    outlist, ys = stratified_sample(testdf, N=N, return_indices=False)
    
    assert len(outlist) == N
    assert ys.shape[0] == N
    assert ys.shape[1] == 2
    #assert isinstance(outlist[0], str)
    #assert False, "this should definitely be tested"