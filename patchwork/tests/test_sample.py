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


def test_find_subset_unlabeled_all():
    s0 = find_subset(testdf, "unlabeled", "not excluded", "all")
    assert s0.sum() == 0


def test_find_subset_labeled_all():
    s0 = find_subset(testdf, "labeled", "not excluded", "all")
    assert s0.sum() == 2
    
    

def test_find_subset_unlabeled_excluded_all():
    s0 = find_subset(testdf, "unlabeled", "excluded", "all")
    assert s0.sum() == 1
    assert "a.jpg" in testdf[s0].filepath.values
    

def test_find_subset_partial_all():
    s0 = find_subset(testdf, "partial", "not excluded", "all")
    assert s0.sum() == 2
    for f in ["d.jpg", "e.jpg"]:
        assert f in testdf[s0].filepath.values

def test_find_subset_labeled_subset():
    s0 = find_subset(testdf, "labeled", "not excluded", "subset: 0")
    assert s0.sum() == 1
    assert "b.jpg" in testdf[s0].filepath.values
    

def test_find_subset_partial_subset():
    s0 = find_subset(testdf, "partial", "not excluded", "subset: 1")
    assert s0.sum() == 2
    assert "d.jpg" in testdf[s0].filepath.values
    assert "e.jpg" in testdf[s0].filepath.values
    

def test_find_subset_partial_contains():
    s0 = find_subset(testdf, "partial", "not excluded", "contains: class1")
    assert s0.sum() == 2
    assert "d.jpg" in testdf[s0].filepath.values
    assert "e.jpg" in testdf[s0].filepath.values
    
    
def test_find_subset_labeled_doesnt_contain():
    s0 = find_subset(testdf, "labeled", "not excluded", "doesn't contain: class1")
    assert s0.sum() == 1
    assert "c.jpg" in testdf[s0].filepath.values
    
        


def test_stratified_sampler():
    N = 100
    outlist, ys = stratified_sample(testdf, N=N, return_indices=False)
    
    assert len(outlist) == N
    assert ys.shape[0] == N
    assert ys.shape[1] == 2
    #assert isinstance(outlist[0], str)
    #assert False, "this should definitely be tested"
    

def test_stratified_sampler_instance_sampling():
    N = 100
    outlist, ys = stratified_sample(testdf, N=N, return_indices=False,
                                    sampling="instance")
    
    assert len(outlist) == N
    assert ys.shape[0] == N
    assert ys.shape[1] == 2
    

def test_stratified_sampler_sqrt_sampling():
    N = 100
    outlist, ys = stratified_sample(testdf, N=N, return_indices=False,
                                    sampling="squareroot")
        
    assert len(outlist) == N
    assert ys.shape[0] == N
    assert ys.shape[1] == 2