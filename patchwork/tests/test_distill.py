# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork._distill import _build_student_model, distill


def test_student_model_with_premade_model():
    inpt = tf.keras.layers.Input((None, None, 3))
    net = tf.keras.layers.GlobalMaxPool2D()(inpt)
    net = tf.keras.layers.Dense(5, activation="sigmoid")(net)
    student0 = tf.keras.Model(inpt, net)
    
    student1 = _build_student_model(student0, 5)
    
    assert isinstance(student1, tf.keras.Model)
    assert len(student0.layers) == len(student1.layers)
    assert student0.output_shape == student1.output_shape


def test_student_model_without_premade_model():    
    student = _build_student_model("VGG16", 5, imshape=(32,32))
    
    assert isinstance(student, tf.keras.Model)
    assert student.output_shape[-1] == 5
    
    
def test_distill(test_png_path):
    inpt = tf.keras.layers.Input((None, None, 3))
    net = tf.keras.layers.GlobalMaxPool2D()(inpt)
    net = tf.keras.layers.Dense(5, activation="sigmoid")(net)
    student0 = tf.keras.Model(inpt, net)
    
    filepaths = [test_png_path, test_png_path]
    ys = 0.5*np.ones((2,5), dtype=np.float32)
    
    student1, trainloss = distill(filepaths, ys, student0, epochs=1, 
                                  imshape=(32,32), batch_size=1,
                                  augment=False) 
    
    assert isinstance(student1, tf.keras.Model)
    assert len(student0.layers) == len(student1.layers)
    assert student0.output_shape == student1.output_shape
    
    assert isinstance(trainloss, dict)
    assert isinstance(trainloss["train_loss"][0], np.float32)