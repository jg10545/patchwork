import tensorflow as tf



class CosineDense(tf.keras.layers.Layer):
    """
    Retuning layer for the "baseline++" model in "A CLOSER LOOK AT FEW-SHOT CLASSIFICATION"
    by Chen et al
    """
    
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CosineDense, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                  shape=shape,
                                  initializer='uniform',
                                  trainable=True)
        # Be sure to call this at the end
        super(CosineDense, self).build(input_shape)

    def call(self, inputs):
        inputs_norm = tf.keras.backend.l2_normalize(inputs, -1)
        kernel_norm = tf.keras.backend.l2_normalize(self.kernel, 0)
        return tf.keras.activations.softmax(tf.matmul(inputs_norm, kernel_norm))

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)
