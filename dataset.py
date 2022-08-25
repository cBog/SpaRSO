import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

BUFFER_SIZE = 1024

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  label = tf.cast(label, tf.int32)

  return image, label

def get_fashion_mnist(batch_size):
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_dataset = train_dataset.map(scale).shuffle(BUFFER_SIZE).batch(batch_size)
    test_dataset = test_dataset.map(scale).batch(batch_size)

    return train_dataset, test_dataset, class_names
