import tensorflow as tf

class BasicFFC:
  def get_model(sparse=False):
    initializer = tf.keras.initializers.GlorotNormal()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu',kernel_initializer=initializer, use_bias=False),
        tf.keras.layers.Dense(10,kernel_initializer=initializer, use_bias=False)
    ])

    return model

class MNIST_RSO:
  def get_model(sparse=False):
      initializer = tf.keras.initializers.GlorotNormal()

      model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu',kernel_initializer=initializer, use_bias=False),
        tf.keras.layers.Dense(10,kernel_initializer=initializer, use_bias=False)
    ])



