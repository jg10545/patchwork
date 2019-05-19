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

    
class ChannelWiseLayer(tf.keras.layers.Layer):
    """
    Channel-wise dense layer as described in "Context Encoders: feature
    learning by inpainting" by Pathak et al
    """
    
    def __init__(self, **kwargs):
        super(ChannelWiseLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert len(input_shape)==4, "don't know what to do with this"
        h = input_shape[1]
        w = input_shape[2]
        c = input_shape[3]
        
        self.h = h
        self.w = w
        self.c = c
        
        self.kernel = self.add_weight(
            name="kernel",
            shape=tf.TensorShape((c, h*w, h*w)),
            initializer="uniform",
            trainable=True
            )
        super(ChannelWiseLayer, self).build(input_shape)
        
    def call(self, inputs):
        # reshape to (-1, wh, c)
        unwound = tf.reshape(inputs, [-1, self.h*self.w, self.c])
        # reorder to (c, -1, wh)
        transposed = tf.transpose(unwound, [2,0,1])
        # do the matrix multiplication, to (c, -1, wh)
        prod = tf.matmul(transposed, self.kernel)
        # swap back to batch first, (-1, wh, c)
        transposed_back = tf.transpose(prod, (1,2,0))
        # and back to original size
        rewound = tf.reshape(transposed_back, 
                             [-1, self.h, self.w, self.c])
        return tf.keras.activations.relu(rewound)
        
        
    def compute_output_shape(self, input_shape):
        return input_shape