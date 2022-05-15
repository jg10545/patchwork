import tensorflow as tf

#BN = tf.keras.layers.experimental.SyncBatchNormalization


def ResNet(stack_fn,input_shape=None):
    """
    """
    if input_shape is None:
        input_shape = (None, None, 3)

    inputs = tf.keras.layers.Input(input_shape)
    x = inputs
    x = tf.keras.layers.Conv2D(64, 7, strides=2)(x)

    x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

    x = stack_fn(x)

    model = tf.keras.Model(inputs,x) 
    return model



def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, final=False,
           BN=tf.keras.layers.BatchNormalization):
    """
    A residual block.
    """

    if conv_shortcut:
      shortcut = tf.keras.layers.Conv2D(
          4 * filters, 1, strides=stride)(x)
      shortcut = BN()(shortcut)
    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(filters, 1, strides=stride)(x)
    x = BN()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(x)
    x = BN()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(4 * filters, 1)(x)
    x = BN()(x)

    if final:
        x = tf.keras.layers.Add(dtype="float32")([shortcut, x])
        x = tf.keras.layers.Activation('relu', dtype="float32")(x)
    else:
        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.Activation('relu')(x)
    return x

def stack1(x, filters, blocks, stride1=2, final=False,
           BN=tf.keras.layers.BatchNormalization):
    """
    A set of stacked residual blocks.
    """
    x = block1(x, filters, stride=stride1, BN=BN)
    for i in range(2, blocks + 1):
        if i == blocks and final:
            x = block1(x, filters, conv_shortcut=False, final=True, BN=BN)
        else:
            x = block1(x, filters, conv_shortcut=False, BN=BN)
    return x

def build_resnet50(input_shape=None, syncbn=True, width_multiplier=1):
    """
    Build a ResNet50
    
    :input_shape:
    :syncbn: whether to use synchronized batch normalization
    :width_multiplier: scale filters by this factor
    """
    if syncbn:
        BN = tf.keras.layers.experimental.SyncBatchNormalization
    else:
        BN = tf.keras.layers.BatchNormalization


    def stack_fn(x):
        x = stack1(x, int(width_multiplier*64), 3, stride1=1, BN=BN)
        x = stack1(x, int(width_multiplier*128), 4, BN=BN)
        x = stack1(x, int(width_multiplier*256), 6, BN=BN)
        return stack1(x, int(width_multiplier*512), 3, final=True, BN=BN)

    return ResNet(stack_fn, input_shape)


