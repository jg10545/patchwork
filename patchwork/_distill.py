# -*- coding: utf-8 -*-
from tqdm import tqdm
import tensorflow as tf

from patchwork._losses import multilabel_distillation_loss


def distill(student, ds, epochs=5, lr=1e-3, optimizer="momentum"):
    """
    
    """
    if optimizer.lower() == "momentum":
        opt = tf.keras.optimizers.SGD(lr, momentum=0.9)
    elif optimizer.lower() == "adam":
        opt = tf.keras.optimizers.Adam(lr)
    else:
        assert False, "dont know what optimizer %s is"%optimizer
        
    @tf.function
    def train_step(x,y):
        with tf.GradientTape() as tape:
            student_pred = student(x, training=True)
            loss = multilabel_distillation_loss(y, student_pred, 1.)
        
        gradients = tape.gradient(loss, student.trainable_variables)
        opt.apply_gradients(zip(gradients, student.trainable_variables))
        return loss
    
    train_loss = []
    for e in tqdm(range(epochs)):
        for x, y in ds:
            train_loss.append(train_step(x,y).numpy())
            
    return train_loss