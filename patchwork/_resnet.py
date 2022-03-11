import tensorflow as tf

BN = tf.keras.layers.experimental.SyncBatchNormalization


from tensorflow.keras import backend
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.util.tf_export import keras_export


def ResNet(stack_fn,
           input_shape=None):
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



def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True):
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

  x = tf.keras.layers.Add()([shortcut, x])
  x = tf.keras.layers.Activation('relu')(x)
  return x

def stack1(x, filters, blocks, stride1=2):
  """
  A set of stacked residual blocks.
  """
  x = block1(x, filters, stride=stride1)
  for i in range(2, blocks + 1):
    x = block1(x, filters, conv_shortcut=False)
  return x

def build_resnet50(input_shape=None):
  """
  Build a ResNet50 with synched batchnorm
  """

  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1)
    x = stack1(x, 128, 4)
    x = stack1(x, 256, 6)
    return stack1(x, 512, 3)

  return ResNet(stack_fn, input_shape)


