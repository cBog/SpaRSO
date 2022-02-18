import tensorflow as tf
from keras.utils.layer_utils import count_params

import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod

class Optimiser(ABC):
  def __init__(self, model):
    self.LOGGING = False
    self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.model = model

  @tf.function
  def forward_pass(self, x, y):
    prediction = self.model(x, training=True)
    loss = self.loss_fn(y, prediction)
    return loss, prediction

  def get_delta_weight(self, layer_std_dev):
    return np.random.normal(0.0, layer_std_dev)

  @abstractmethod
  def run_training(self, dataset):
    raise NotImplementedError()

class StandardSGD(Optimiser):
  def __init__(self, model, epochs):
    super(StandardSGD, self).__init__(model)
    self.epochs = epochs
    self.sgd_optimiser = tf.keras.optimizers.Adam(0.001)

  @tf.function
  def train_step_gradients(self, x, y):
    with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      loss, prediction = self.forward_pass(x,y)
      # prediction = self.model(x, training=True)
      # loss = self.loss_fn(y, prediction)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.sgd_optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))
    self.train_acc_metric.update_state(y, prediction)
    return loss

  def run_training(self, dataset):
    training_acc_log = []

    for epoch in range(self.epochs):
      print(f"Epoch {epoch}")
      for step, (x, y) in tqdm(enumerate(dataset)):
        loss = self.train_step_gradients(x, y)

      train_acc = self.train_acc_metric.result()
      print("Training acc over epoch: %.4f" % (float(train_acc),))

      training_acc_log.append(train_acc)

      # Reset training metrics at the end of each epoch
      self.train_acc_metric.reset_states()
    return training_acc_log

class WeightPerBatchRSO(Optimiser):
  def __init__(self, model, number_of_weight_updates, random_update_order=False):
    super(WeightPerBatchRSO, self).__init__(model)
    # import pdb; pdb.set_trace()
    self.trainable_weight_count = count_params(model.trainable_weights)
    self.number_of_weight_updates = number_of_weight_updates
    self.number_of_batches = self.number_of_weight_updates * self.trainable_weight_count
    # TODO: this should be 'number_of_weight_updates'
    # self.epochs = int()
    self.random_update_order = random_update_order
    self.num_model_layers = len(model.layers)
    self.layer_std_devs = {}
    self.init_loop_state()

  def init_loop_state(self):
    self.layers_idx = self.num_model_layers - 1
    self.w_idx = len(self.model.layers[self.layers_idx].get_weights()) - 1 # counts down

    weight_len = len(self.model.layers[self.layers_idx].get_weights()[self.w_idx].flatten())
    if weight_len:
      self.idx = weight_len - 1
      if self.random_update_order:
        self.permutation = np.random.permutation(weight_len)
      else:
        self.permutation = [n for n in range(0, weight_len, 1)]
    else:
      self.idx = -1
      self.permutation = None
  
  def print_loop_state(self):
    if self.LOGGING:
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
    current_loss, best_prediction = self.forward_pass(x, y)
    layer = self.model.layers[self.layers_idx]
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
      
      delta_weight = self.get_delta_weight(self.layer_std_devs[layer])

      try_val = og_val + delta_weight
      flattened_weights[i] = try_val
      layer_weights_list[self.w_idx] = flattened_weights.reshape(weights_shape)
      layer.set_weights(layer_weights_list)

      # new_prediction = model(x, training=True)
      # new_loss = loss_fn(y, new_prediction)
      new_loss, new_prediction = self.forward_pass(x, y)

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
      new_loss, new_prediction = self.forward_pass(x, y)

      if new_loss < current_loss:
        new_val = try_val
        current_loss = new_loss
        best_prediction = new_prediction
      
      flattened_weights[i] = new_val
      layer_weights_list[self.w_idx] = flattened_weights.reshape(weights_shape)
      layer.set_weights(layer_weights_list)

    self.train_acc_metric.update_state(y, best_prediction)
      
    self.loop_state_step()
    return current_loss

  def run_training(self, dataset):
    training_acc_log = []

    total_steps = 0
    epochs = 0
    # TODO: tidy this up to loop through weights predominantly and call next on a data iterator manually (i.e. swap the loop around)
    while total_steps < self.number_of_batches:
      print(f"Epoch {epochs}")
      for step, (x, y) in tqdm(enumerate(dataset)):
        if total_steps == self.number_of_batches:
          break
        loss = self.train_loop_step(x, y)
      
      total_steps += step
      epochs += 1

      # TODO: make this every nth or after every weight iter?
      train_acc = self.train_acc_metric.result()
      print("Training acc over epoch: %.4f" % (float(train_acc),))

      training_acc_log.append(train_acc)

      # Reset training metrics at the end of each epoch
      self.train_acc_metric.reset_states()
    return training_acc_log

class WeightsPerBatchRSO(Optimiser):
  def __init__(self, model, epochs, max_weight_per_iter=np.Inf, random_update_order=False):
    # TODO: distinguish between permute per batch and permute per weight iteration?
    super(WeightsPerBatchRSO, self).__init__(model)
    self.epochs = epochs
    self.layer_std_devs = {}
    self.max_weight_per_iter = max_weight_per_iter
    self.random_update_order = random_update_order

  def train_step_rso(self, x, y):
    # best_prediction = model(x, training=True)
    # current_loss = loss_fn(y, best_prediction)
    current_loss, best_prediction = self.forward_pass(x, y)
    # import pdb; pdb.set_trace()
    for layer in reversed(self.model.layers):
      if layer.get_weights():
        # print("HERE")
        if layer not in self.layer_std_devs:
          # import pdb; pdb.set_trace()
          for weights in layer.get_weights():
            self.layer_std_devs[layer] = np.std(weights)
            if self.layer_std_devs[layer] > 0:
              print(f"layer std dev set to {self.layer_std_devs[layer]}")
              break
          assert(self.layer_std_devs[layer] > 0)
        layer_weights_list = layer.get_weights()
        # import pdb; pdb.set_trace()
        for w_idx in range(len(layer_weights_list), 0, -1):
          w_idx = w_idx - 1
          layer_weights = layer_weights_list[w_idx]
          weights_shape = layer_weights.shape
          flattened_weights = layer_weights.flatten()
          # print(f"setting {len(flattened_weights)} weights")
          if self.random_update_order:
            permutation = np.random.permutation(len(flattened_weights))
          else:
            permutation = [n for n in range(0, len(flattened_weights), 1)]
          # MAX_WEIGHT_PER_ITER = self.200# MAX_WEIGHT_PER_ITER_PROPORTION * len(flattened_weights)
          for idx in range(min(len(flattened_weights),self.max_weight_per_iter)):
            # i = idx
            i = permutation[idx]
            og_val = flattened_weights[i]
            new_val = og_val
            
            delta_weight = self.get_delta_weight(self.layer_std_devs[layer])

            try_val = og_val + delta_weight
            flattened_weights[i] = try_val
            layer_weights_list[w_idx] = flattened_weights.reshape(weights_shape)
            layer.set_weights(layer_weights_list)

            # new_prediction = model(x, training=True)
            # new_loss = loss_fn(y, new_prediction)
            new_loss, new_prediction = self.forward_pass(x, y)

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
            new_loss, new_prediction = self.forward_pass(x, y)

            if new_loss < current_loss:
              new_val = try_val
              current_loss = new_loss
              best_prediction = new_prediction
            
            flattened_weights[i] = new_val
            layer_weights_list[w_idx] = flattened_weights.reshape(weights_shape)
            layer.set_weights(layer_weights_list)
    self.train_acc_metric.update_state(y, best_prediction)
    return current_loss

  def run_training(self, dataset):
    training_acc_log = []

    for epoch in range(self.epochs):
      print(f"Epoch {epoch}")
      for step, (x, y) in tqdm(enumerate(dataset)):
        loss = self.train_step_rso(x,y)

      # TODO: make this every nth
      train_acc = self.train_acc_metric.result()
      print("Training acc over epoch: %.4f" % (float(train_acc),))

      training_acc_log.append(train_acc)

      # Reset training metrics at the end of each epoch
      self.train_acc_metric.reset_states()
    return training_acc_log

class SpaRSO(Optimiser):
# SpaRSO
# algorithm steps as follows:
# 	- initiate random sparsity mask for all weights (defined init_sparsity and max_sparsity)
# 	- Loop through algorithm phases:
# 		- Improve phase:
# 			- loop through all non-zero weights and consider +/- norm value as per RSO algorithm and also zeroed weight
# 			- if weight is zeroed, update the mask accordingly
# 		- Regrow phase:
# 			- loop through g random masked weights with a randomly generate value 
# 				- value based on layer norm?
# 			- decide whether to keep with same philosophy
# 			- g is decided by either:
# 				- max_weights - current_count 
# 				- max_weights - max(current_count, initial)
# 					- this will always consider a fixed number of growths as current_count reduces
# 			- update the mask if new weights chosen
# 		- Replace/swap phase:
# 			- for r randomly masked weights, try swapping with random unmasked weights, decide whether to replace acording to RSO principles
# 			- update the mask if replacements are chosen

# 	TODO: need to keep logs of when events happen to measure frequencies of improvements/regrows/swaps
# 	- there should be an equilibrium reached where loosing too many weights will cause more regrowths but number of regrowths is fixed… could play about with reducing max_sparsity as alg trains? or encouraging zeroed weights somehow…?

# TODO: need to decide whether to include bias tensors


  def __init__(self, model, initial_sparsity, maximum_sparsity, swap_proportion, update_iterations):
    super(SpaRSO, self).__init__(model)
    self.initial_sparsity = initial_sparsity
    self.maximum_sparsity = maximum_sparsity
    self.swap_proportion = swap_proportion
    self.update_iterations = update_iterations

    # define sparse masks for all layers
    # set init proportion of all weights to initial sparsity
    # multiply each weight tensor by its mask
    self.count_weights = 0

  def get_batch(self):
    # return new batch each time or same batch based on some count or conditions
    ...

  def improve_phase(self):
    # loop over non zero indices calling get_batch for each adjustment
    ...

  def regrow_phase(self):
    # try random values for some g non-masked weights calling get_batch for each
    ...

  def replace_phase(self):
    # for swap proportion number of weights, try a random masked value swapping for a non-masked one calling get_batch for each
    ...

  def run_training(self, dataset):
    self.dataset = dataset

    for _ in tqdm(range(self.update_iterations)):
      self.improve_phase()
      self.regrow_phase()
      self.replace_phase()

    # TODO: add logs and training metrics
    


