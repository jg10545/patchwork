import numpy as np
import tensorflow as tf
from patchwork._fine_tuning_models import GlobalPooling, ConvNet



def test_GlobalPooling_with_maxpool():
    pool = GlobalPooling()
    model = pool.build(2)
    
    testarr = np.zeros((1,5,7,2))
    testarr[0,:,:,1] = 1
    
    test_out = model(testarr).numpy()
    
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 2
    assert (test_out == np.array([[0.,1.]])).all()
    
    
    
def test_GlobalPooling_with_avpool():
    pool = GlobalPooling()
    pool.pooling_type = "average pool"
    model = pool.build(2)
    
    testarr = np.zeros((1,5,4,2))
    testarr[0,:,:2,1] = 1
    
    test_out = model(testarr).numpy()
    
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 2
    assert (test_out == np.array([[0.,0.5]])).all()
    
    
def test_ConvNet():
    c = ConvNet()
    
    c.dropout_rate = 0
    #c.number_of_layers = 2
    #c.filters_per_layer = 5
    c.layers = "10,p,11"
    model = c.build(7)
    
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 5
    assert model.output_shape[1] == 11
    
    
def test_ConvNet_with_dropout():
    c = ConvNet()
    
    c.dropout_rate = 0.5
    #c.number_of_layers = 2
    #c.filters_per_layer = 5
    c.layers = "7,p,11,p,13"
    model = c.build(7)
    
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 10
    assert model.output_shape[1] == 13
    
def test_ConvNet_with_separable_convolutions():
    c = ConvNet()
    
    c.dropout_rate = 0
    #c.number_of_layers = 2
    #c.filters_per_layer = 5
    c.layers = "5,13"
    c.separable_convolutions = True
    model = c.build(7)
    
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 4
    assert model.output_shape[1] == 13
    