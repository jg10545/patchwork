import tensorflow as tf
import tensorflow.keras.backend as K


class CosineDense(tf.keras.layers.Layer):
    """
    Dense layer for multilabel cosine similarity. Expects a [None,d] tensor of
    features and returns a [None,num_classes] tensor of class probabilities.
    """
    
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CosineDense, self).__init__(**kwargs)

    def build(self, input_shape):
        # input shape (None, d)
        # matrix to multiply by should be (d, 2*output_dim)
        kernel_shape = (input_shape[1], 2*self.output_dim)
        self.kernel = self.add_weight(name="kernel",
                                     shape=kernel_shape,
                                     initializer="uniform",
                                     trainable=True)
        # class embeddings should be (1, num_classes, 2)
        embed_shape = (1, self.output_dim, 2)
        self.class_embeds = self.add_weight(name="class_embeds",
                                           shape=embed_shape,
                                           initializer="uniform",
                                           trainable=True)
        # Be sure to call this at the end
        super(CosineDense, self).build(input_shape)

    def call(self, inputs):
        # multiply inputs by weight kernel. (None, d)*(d, 2*output_dim)
        projected = K.dot(inputs, self.kernel)
        # reshape
        reshaped = K.reshape(projected, (-1, self.output_dim, 2))
        # normalize along final axis
        projected_norm = K.l2_normalize(reshaped, -1)
        # normalize embeddings
        embeds_norm = K.l2_normalize(self.class_embeds, -1)
        # dot product
        cosine_similarity = K.sum(projected_norm*embeds_norm, axis=-1)
        # shift to unit interval
        return 0.5*(cosine_similarity+1)


    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape((shape[0], self.output_dim))


    
class ChannelWiseDense(tf.keras.layers.Layer):
    """
    Channel-wise dense layer as described in "Context Encoders: feature
    learning by inpainting" by Pathak et al.
    
    ReLU activation hard-coded.
    """
    
    def __init__(self, **kwargs):
        super(ChannelWiseDense, self).__init__(**kwargs)
        
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
        super(ChannelWiseDense, self).build(input_shape)
        
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