import tensorflow as tf
from enum import Enum

class NORM_TYPE(Enum):
  BATCH = "batch"
  LAYER = "layer"
  NONE = "none"

  def __str__(self):
    return self.value


class BasicFFC:
  def get_model():
    initializer = tf.keras.initializers.GlorotNormal()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu',kernel_initializer=initializer, use_bias=False),
        tf.keras.layers.Dense(10,kernel_initializer=initializer, use_bias=False)
    ])

    return model

# from: https://arxiv.org/pdf/2005.05955.pdf
class RSO_PAPER_MNIST_MODEL:
  def get_model(norm:NORM_TYPE=NORM_TYPE.BATCH):
    initializer = tf.keras.initializers.GlorotNormal()

    def norm_func():
      if norm == NORM_TYPE.BATCH:
        return tf.keras.layers.BatchNormalization()
      elif norm == NORM_TYPE.LAYER:
        return tf.keras.layers.LayerNormalization()
      elif norm == NORM_TYPE.NONE:
        # defaults to an identify function (noop)
        return tf.keras.layers.Layer()
      else:
        raise NotImplementedError("Norm type not implemented")

    model = tf.keras.Sequential([
      tf.keras.Input(shape=(28,28)),
      tf.keras.layers.Reshape((28,28,1)),
      tf.keras.layers.Conv2D(16, 3, kernel_initializer=initializer, padding="same"),
      # norm_func(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Conv2D(16, 3, kernel_initializer=initializer, padding="same"),
      # norm_func(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(16, 3, kernel_initializer=initializer, padding="same"),
      # norm_func(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Conv2D(16, 3, kernel_initializer=initializer, padding="same"),
      # norm_func(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(16, 3, kernel_initializer=initializer, padding="same"),
      # norm_func(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Conv2D(16, 3, kernel_initializer=initializer, padding="same"),
      # norm_func(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10,kernel_initializer=initializer, use_bias=False)
    ])
    return model



