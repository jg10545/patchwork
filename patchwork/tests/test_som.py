import numpy as np

from patchwork._som import _find_BMU, _update_weights, train_SOM


def test_find_BMU():
    d = 7
    SOM = np.random.normal(0, 1, size=(3,5,d))
    
    assert _find_BMU(SOM, SOM[1,3]) == (1,3)
    
    
def test_update_weights():
    d = 7
    coord = (1,3)
    SOM = np.random.normal(0, 1, size=(3,5,d))
    train_ex = SOM[coord]
    SOM_old = SOM.copy()

    SOM = _update_weights(SOM, train_ex, 0.1, 1, coord, 2)

    assert SOM.shape == SOM_old.shape
    assert np.sum((SOM-SOM_old)**2) > 0
    
    
def test_train_SOM():
    N = 23
    d = 7
    g = 5
    train_data = np.random.normal(0, 1, size=(N,d))
    SOM = np.random.normal(0, 1, size=(g,g,7))

    SOM_new = train_SOM(SOM, train_data)
    assert SOM_new.shape == SOM.shape