# very loosely based on https://github.com/google-research/lottery-ticket-hypothesis/blob/1f17279d282e729ee29e80a2f750cfbffc4b8500/mnist_fc/constants.py
# and https://www.tensorflow.org/tutorials/keras/classification
import tensorflow as tf

# import tensorflow_datasets as tfds
from tqdm import tqdm
import random
import numpy as np
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# datasets, info = tfds.load(name='fashion_mnist', with_info=True, as_supervised=True)
# dataset_train, dataset_test = datasets['train'], datasets['test']
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

BUFFER_SIZE = 10 # Use a much larger value for real code
BATCH_SIZE = 64
NUM_EPOCHS = 10


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  # label = tf.cast(label, tf.int32)

  return image, label

train_dataset = train_dataset.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.map(scale).batch(BATCH_SIZE)

# STEPS_PER_EPOCH = 5

# train_data = train_data.take(STEPS_PER_EPOCH)
# test_data = test_data.take(STEPS_PER_EPOCH)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# train_images = train_images / 255.0

# test_images = test_images / 255.0
initializer = tf.keras.initializers.GlorotNormal()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu',kernel_initializer=initializer),
    tf.keras.layers.Dense(10,kernel_initializer=initializer)
])

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# for x,y in tqdm(zip(train_images, train_labels)):
#   print(y)
#   if y ==3:
#     break
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    prediction = model(x, training=True)
    loss = loss_fn(y, prediction)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_acc_metric.update_state(y, prediction)
  return loss

# def get_rand_weight(val):
#   return val + random.randrange(-1,1)

def get_delta_weight(layer_std_dev):
  return np.random.normal(0.0, layer_std_dev)

@tf.function
def forward_pass(x,y):
  prediction = model(x, training=True)
  loss = loss_fn(y, prediction)
  return loss, prediction

layer_std_devs = {}


def train_step_rso(x, y):
  # best_prediction = model(x, training=True)
  # current_loss = loss_fn(y, best_prediction)
  # current_loss, best_prediction = forward_pass(x, y)
  init_loss, init_prediction = forward_pass(x, y)
  # best_loss = init_loss
  # best_prediction = init_prediction
  for layer in reversed(model.layers):
    if layer.get_weights():
      if layer not in layer_std_devs:
        # import pdb; pdb.set_trace()
        for weights in layer.get_weights():
          layer_std_devs[layer] = np.std(weights)
          if layer_std_devs[layer] > 0:
            break
        assert(layer_std_devs[layer] > 0)
      layer_weights_list = layer.get_weights()
      for w_idx in range(len(layer_weights_list)-1, 0, -1):
        layer_weights = layer_weights_list[w_idx]
        weights_shape = layer_weights.shape
        flattened_weights = layer_weights.flatten()
        new_weights = np.zeros(flattened_weights.shape, dtype=flattened_weights.dtype)
        for i in range(len(flattened_weights)):
          current_loss = init_loss
          # current_prediction = init_prediction
          og_val = flattened_weights[i]
          new_val = og_val
          
          delta_weight = get_delta_weight(layer_std_devs[layer])

          try_val = og_val + delta_weight
          flattened_weights[i] = try_val
          layer_weights_list[w_idx] = flattened_weights.reshape(weights_shape)
          layer.set_weights(layer_weights_list)

          # new_prediction = model(x, training=True)
          # new_loss = loss_fn(y, new_prediction)
          new_loss, new_prediction = forward_pass(x, y)

          if new_loss < current_loss:
            new_val = try_val
            current_loss = new_loss
            # current_prediction = new_prediction

          try_val = og_val - delta_weight
          flattened_weights[i] = try_val
          layer_weights_list[w_idx] = flattened_weights.reshape(weights_shape)
          layer.set_weights(layer_weights_list)

          # new_prediction = model(x, training=True)
          # new_loss = loss_fn(y, new_prediction)
          new_loss, new_prediction = forward_pass(x, y)

          if new_loss < current_loss:
            new_val = try_val
            # current_loss = new_loss
            # current_prediction = new_prediction
          
          flattened_weights[i] = og_val
          layer_weights_list[w_idx] = flattened_weights.reshape(weights_shape)
          layer.set_weights(layer_weights_list)


          new_weights[i] = new_val

        layer_weights_list[w_idx] = new_weights.reshape(weights_shape)
        layer.set_weights(layer_weights_list)
        init_loss, init_prediction = forward_pass(x, y)

  
  best_loss, best_prediction = forward_pass(x, y)

  train_acc_metric.update_state(y, best_prediction)
  return current_loss



  # import pdb; pdb.set_trace()



def train(model, dataset, optimizer, loss_fn, epochs):
  for epoch in range(epochs):
    print(f"Epoch {epoch}")
    for step, (x, y) in tqdm(enumerate(dataset)):
      # import pdb; pdb.set_trace()
      # loss = train_step(x, y)
      loss = train_step_rso(x,y)

    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# for i in range(10):
#     print(f"Epoch {i}")
    # import pdb; pdb.set_trace()
train(model, train_dataset, optimizer, loss_fn, epochs=10)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10)

# TODO: get this working with the tfds dataset
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print('\nTest accuracy:', test_acc)