# very loosely based on https://github.com/google-research/lottery-ticket-hypothesis/blob/1f17279d282e729ee29e80a2f750cfbffc4b8500/mnist_fc/constants.py
# and https://www.tensorflow.org/tutorials/keras/classification
import tensorflow as tf

# import tensorflow_datasets as tfds
from tqdm import tqdm
import random
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
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
    tf.keras.layers.Dense(128, activation='relu',kernel_initializer=initializer, use_bias=False),
    tf.keras.layers.Dense(10,kernel_initializer=initializer, use_bias=False)
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

# @tf.function
# def n_forward_pass(x_n, y_n):
  

layer_std_devs = {}

MAX_WEIGHT_PER_ITER_PROPORTION=0.15
LOGGING = False

class StatefulWeightLooper:
  def __init__(self, model):
    self.model = model
    self.num_model_layers = len(model.layers)
    self.layer_std_devs = {}
    self.init_loop_state()

  def init_loop_state(self):
    self.layers_idx = self.num_model_layers - 1
    self.w_idx = len(self.model.layers[self.layers_idx].get_weights()) - 1 # counts down

    weight_len = len(self.model.layers[self.layers_idx].get_weights()[self.w_idx].flatten())
    if weight_len:
      self.idx = weight_len - 1
      self.permutation = [n for n in range(0, weight_len, 1)]
      # self.permutation = np.random.permutation(weight_len)
    else:
      self.idx = -1
      self.permutation = None
  
  def print_loop_state(self):
    if LOGGING:
      print(f"LOOP STATE: layer: {self.layers_idx}, weights: {self.w_idx}, idx: {self.idx}")
      if self.permutation is not None:
        print(f"permutations: [{self.permutation[0]}, {self.permutation[1]},{self.permutation[2]}, ...)")

  def loop_state_step(self):
    # decrement indices
    if self.idx <= 0:
      if self.w_idx <= 0:
        if self.layers_idx <= 0:
          self.layers_idx = self.num_model_layers - 1
          print(f"reset layer index to {self.layers_idx}")
        else:
          self.layers_idx -= 1
        self.w_idx = len(self.model.layers[self.layers_idx].get_weights()) - 1
        print(f"reset weight index to {self.w_idx}")
      else:
        self.w_idx -= 1
      weight_len = len(self.model.layers[self.layers_idx].get_weights()[self.w_idx].flatten()) if self.w_idx >= 0 else 0
      if weight_len:
        self.idx = weight_len - 1
        # self.permutation = np.random.permutation(weight_len)
        self.permutation = [n for n in range(0, weight_len, 1)]
        print(f"reset index to {self.idx}")
      else:
        self.idx = -1
        self.permutation = None
    else:
      self.idx -= 1
    
    self.print_loop_state()

  def train_loop_step(self, x, y):
    current_loss, best_prediction = forward_pass(x, y)
    layer = model.layers[self.layers_idx]
    if layer.get_weights():
      if layer not in self.layer_std_devs:
          for weights in layer.get_weights():
            self.layer_std_devs[layer] = np.std(weights)
            if self.layer_std_devs[layer] > 0:
              print(f"layer std dev set to {self.layer_std_devs[layer]}")
              break
          assert(self.layer_std_devs[layer] > 0)
      layer_weights_list = layer.get_weights()
      layer_weights = layer_weights_list[self.w_idx]
      weights_shape = layer_weights.shape
      flattened_weights = layer_weights.flatten()

      i = self.permutation[self.idx]
      
      og_val = flattened_weights[i]
      new_val = og_val
      
      delta_weight = get_delta_weight(self.layer_std_devs[layer])

      try_val = og_val + delta_weight
      flattened_weights[i] = try_val
      layer_weights_list[self.w_idx] = flattened_weights.reshape(weights_shape)
      layer.set_weights(layer_weights_list)

      # new_prediction = model(x, training=True)
      # new_loss = loss_fn(y, new_prediction)
      new_loss, new_prediction = forward_pass(x, y)

      if new_loss < current_loss:
        new_val = try_val
        current_loss = new_loss
        best_prediction = new_prediction

      try_val = og_val - delta_weight
      flattened_weights[i] = try_val
      layer_weights_list[self.w_idx] = flattened_weights.reshape(weights_shape)
      layer.set_weights(layer_weights_list)

      # new_prediction = model(x, training=True)
      # new_loss = loss_fn(y, new_prediction)
      new_loss, new_prediction = forward_pass(x, y)

      if new_loss < current_loss:
        new_val = try_val
        current_loss = new_loss
        best_prediction = new_prediction
      
      flattened_weights[i] = new_val
      layer_weights_list[self.w_idx] = flattened_weights.reshape(weights_shape)
      layer.set_weights(layer_weights_list)

    train_acc_metric.update_state(y, best_prediction)
      
    self.loop_state_step()
    return current_loss
    



def train_step_rso(x, y):
  # best_prediction = model(x, training=True)
  # current_loss = loss_fn(y, best_prediction)
  current_loss, best_prediction = forward_pass(x, y)
  # import pdb; pdb.set_trace()
  for layer in reversed(model.layers):
    if layer.get_weights():
      # print("HERE")
      if layer not in layer_std_devs:
        # import pdb; pdb.set_trace()
        for weights in layer.get_weights():
          layer_std_devs[layer] = np.std(weights)
          if layer_std_devs[layer] > 0:
            print(f"layer std dev set to {layer_std_devs[layer]}")
            break
        assert(layer_std_devs[layer] > 0)
      layer_weights_list = layer.get_weights()
      # import pdb; pdb.set_trace()
      for w_idx in range(len(layer_weights_list), 0, -1):
        w_idx = w_idx - 1
        layer_weights = layer_weights_list[w_idx]
        weights_shape = layer_weights.shape
        flattened_weights = layer_weights.flatten()
        # print(f"setting {len(flattened_weights)} weights")
        permutation = np.random.permutation(len(flattened_weights))
        MAX_WEIGHT_PER_ITER = 200# MAX_WEIGHT_PER_ITER_PROPORTION * len(flattened_weights)
        for idx in range(min(len(flattened_weights),MAX_WEIGHT_PER_ITER)):
          # i = idx
          i = permutation[idx]
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
            best_prediction = new_prediction

          try_val = og_val - delta_weight
          flattened_weights[i] = try_val
          layer_weights_list[w_idx] = flattened_weights.reshape(weights_shape)
          layer.set_weights(layer_weights_list)

          # new_prediction = model(x, training=True)
          # new_loss = loss_fn(y, new_prediction)
          new_loss, new_prediction = forward_pass(x, y)

          if new_loss < current_loss:
            new_val = try_val
            current_loss = new_loss
            best_prediction = new_prediction
          
          flattened_weights[i] = new_val
          layer_weights_list[w_idx] = flattened_weights.reshape(weights_shape)
          layer.set_weights(layer_weights_list)
  train_acc_metric.update_state(y, best_prediction)
  return current_loss



  # import pdb; pdb.set_trace()



def train(model, dataset, optimizer, loss_fn, epochs):
  train_step_looper = StatefulWeightLooper(model)

  training_acc_log = []

  for epoch in range(epochs):
    print(f"Epoch {epoch}")
    for step, (x, y) in tqdm(enumerate(dataset)):
      # import pdb; pdb.set_trace()
      # loss = train_step(x, y)
      loss = train_step_rso(x,y)

      # loss = train_step_looper.train_loop_step(x, y)


    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    training_acc_log.append(train_acc)

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
  
  train_log_np = np.array(training_acc_log)
  datetimestr = datetime.now().strftime("%m%d%Y_%H%M%S")
  np.save(f"{datetimestr}_training_acc_results", train_log_np)

  fig = plt.figure()
  plt.title("Training accuracy every epoch of data") 
  plt.xlabel("Epochs") 
  plt.ylabel("Training accuracy") 
  plt.plot(train_log_np)
  fig.savefig(f"{datetimestr}_training_acc_plot.png")






optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# for i in range(10):
#     print(f"Epoch {i}")
    # import pdb; pdb.set_trace()
train(model, train_dataset, optimizer, loss_fn, epochs=4)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10)

# TODO: get this working with the tfds dataset
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print('\nTest accuracy:', test_acc)