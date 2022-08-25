from typing import NamedTuple
import tensorflow as tf
from keras.utils.layer_utils import count_params

import numpy as np
from tqdm import tqdm
import sys

from abc import ABC, abstractmethod
from enum import Enum
from collections import namedtuple

from experiment_logging import get_logger

class LOG_LEVEL(Enum):
  INFO = "info"
  TRACE = "trace"

  def __str__(self):
    return self.value

class Optimiser(ABC):
  def __init__(self, model):
    self.LOGGING = True
    self.LOGGER = get_logger()
    self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.model = model
    self.forward_count = 0

  @tf.function
  def forward_pass(self, x, y):
    prediction = self.model(x, training=True)
    loss = self.loss_fn(y, prediction)
    self.forward_count += 1
    return loss, prediction

  def get_delta_weight(self, layer_std_dev):
    return np.random.normal(0.0, layer_std_dev)

  @abstractmethod
  def run_training(self, dataset):
    raise NotImplementedError()

  def log(self, message: str, level=LOG_LEVEL.INFO, flush=False):
    # if level != LOG_LEVEL.TRACE:
      # self.log(message,flush=True)
    self.LOGGER.log(message, level, flush=flush)

  def save_model_state(self, label, state_dict_in):
    self.log(f"saving model status at {label}")
    self.LOGGER.save(self.model, f"{label}_model")

    state_dict = {}
    state_dict["forward_count"] = self.forward_count
    state_dict.update(state_dict_in)

    self.LOGGER.save(state_dict, f"{label}_statedict")


class StandardSGD(Optimiser):
  def __init__(self, model, epochs):
    super(StandardSGD, self).__init__(model)
    self.epochs = epochs
    # self.sgd_optimiser = tf.keras.optimizers.Adam(0.001)
    self.sgd_optimiser = tf.keras.optimizers.Adam(0.03)

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
    # TODO: call update state more regularly!
    self.train_acc_metric.update_state(y, prediction)
    return loss

  def run_training(self, dataset):
    training_acc_log = []

    for epoch in range(self.epochs):
      self.log(f"Epoch {epoch}")
      for step, (x, y) in tqdm(enumerate(dataset),file=self.LOGGER.tqdm_logger,mininterval=30):
        loss = self.train_step_gradients(x, y)

      self.save_model_state(f"state_{epoch}", {"epoch": epoch})
      train_acc = self.train_acc_metric.result()
      self.log("Training acc over epoch: %.4f" % (float(train_acc),))

      training_acc_log.append(train_acc)

      # Reset training metrics at the end of each epoch
      self.train_acc_metric.reset_states()
    # import pdb; pdb.set_trace()
    return training_acc_log, []

class WeightPerBatchRSO(Optimiser):
  def __init__(self, model, number_of_weight_updates, random_update_order=False):
    super(WeightPerBatchRSO, self).__init__(model)
    # import pdb; pdb.set_trace()
    self.trainable_weight_count = count_params(model.trainable_weights)
    self.number_of_weight_updates = number_of_weight_updates
    self.number_of_batches = self.number_of_weight_updates * self.trainable_weight_count
    self.random_update_order = random_update_order
    self.num_model_layers = len(model.layers)
    self.layer_std_devs = {}
    self.init_loop_state()
    self.training_acc_log = []
    self.training_forwards_log = []
    self.count_weight_iters = 0

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
      self.log(f"LOOP STATE: layer: {self.layers_idx}, weights: {self.w_idx}, idx: {self.idx}")
      if self.permutation is not None:
        self.log(f"permutations: [{self.permutation[0]}, {self.permutation[1]},{self.permutation[2]}, ...)")

  def loop_state_step(self):
    # decrement indices
    if self.idx <= 0:
      if self.w_idx <= 0:
        if self.layers_idx <= 0:
          self.layers_idx = self.num_model_layers - 1
          self.log(f"reset layer index to {self.layers_idx}")
          train_acc = self.train_acc_metric.result()
          self.log("Training acc over epoch: %.4f" % (float(train_acc),))

          self.training_acc_log.append(train_acc)
          self.training_forwards_log.append(self.forward_count)
          self.save_model_state(f"state_{self.count_weight_iters}", 
                                {"count_weight_iters" : self.count_weight_iters, 
                                 "total_steps" : self.total_steps, 
                                 "epochs" : self.epochs})
          self.count_weight_iters += 1

          # Reset training metrics at the end of each epoch
          self.train_acc_metric.reset_states()
        else:
          self.layers_idx -= 1
        self.w_idx = len(self.model.layers[self.layers_idx].get_weights()) - 1
        self.log(f"reset weight index to {self.w_idx}")
      else:
        self.w_idx -= 1
      weight_len = len(self.model.layers[self.layers_idx].get_weights()[self.w_idx].flatten()) if self.w_idx >= 0 else 0
      if weight_len:
        self.idx = weight_len - 1
        # self.permutation = np.random.permutation(weight_len)
        self.permutation = [n for n in range(0, weight_len, 1)]
        self.log(f"reset index to {self.idx}")
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
          # import pdb; pdb.set_trace()
          for weights in layer.get_weights():
            self.layer_std_devs[layer] = np.std(weights)
            if self.layer_std_devs[layer] > 0:
              self.log(f"layer std dev set to {self.layer_std_devs[layer]}")
              break
          assert(self.layer_std_devs[layer] > 0)
      layer_weights_list = layer.get_weights()
      layer_weights = layer_weights_list[self.w_idx]
      weights_shape = layer_weights.shape
      flattened_weights = layer_weights.flatten()

      i = self.permutation[self.idx]
      
      og_val = flattened_weights[i]
      new_val = og_val
      
      # as per paper: for stdevs, they "linearly anneal the standard deviation σcdat a cycle
      # c of the sampling distribution for layer d, such that thestandard deviation at the 
      # final cycle C is σCd = σ1d/10."
      # equivalent to between σCd and σCd - (0.9 * σCd)
      stddev = self.layer_std_devs[layer] - (0.9 * self.layer_std_devs[layer] * (self.count_weight_iters/(self.number_of_weight_updates-1)))
      delta_weight = self.get_delta_weight(stddev)

      try_val = og_val + delta_weight
      flattened_weights[i] = try_val
      # layer_weights_list[self.w_idx] = flattened_weights.reshape(weights_shape)
      # layer.set_weights(layer_weights_list)
      layer_weights.assign(flattened_weights.reshape(weights_shape))

      # new_prediction = model(x, training=True)
      # new_loss = loss_fn(y, new_prediction)
      new_loss, new_prediction = self.forward_pass(x, y)

      if new_loss < current_loss:
        new_val = try_val
        current_loss = new_loss
        best_prediction = new_prediction

      try_val = og_val - delta_weight
      flattened_weights[i] = try_val
      # layer_weights_list[self.w_idx] = flattened_weights.reshape(weights_shape)
      # layer.set_weights(layer_weights_list)
      layer_weights.assign(flattened_weights.reshape(weights_shape))

      # new_prediction = model(x, training=True)
      # new_loss = loss_fn(y, new_prediction)
      new_loss, new_prediction = self.forward_pass(x, y)

      if new_loss < current_loss:
        new_val = try_val
        current_loss = new_loss
        best_prediction = new_prediction
      
      flattened_weights[i] = new_val
      # layer_weights_list[self.w_idx] = flattened_weights.reshape(weights_shape)
      # layer.set_weights(layer_weights_list)
      layer_weights.assign(flattened_weights.reshape(weights_shape))

    self.train_acc_metric.update_state(y, best_prediction)
      
    self.loop_state_step()
    return current_loss

  def run_training(self, dataset):
    self.total_steps = 0
    self.epochs = 0
    # TODO: tidy this up to loop through weights predominantly and call next on a data iterator manually (i.e. swap the loop around)
    while self.total_steps < self.number_of_batches:
      self.log(f"Epoch {self.epochs}")
      for step, (x, y) in tqdm(enumerate(dataset),file=self.LOGGER.tqdm_logger,mininterval=30):
        if (self.total_steps + step) == self.number_of_batches:
          break
        loss = self.train_loop_step(x, y)
      
      self.total_steps += step
      self.epochs += 1

      # TODO: make this every nth or after every weight iter?
      # train_acc = self.train_acc_metric.result()
      # self.log("Training acc over epoch: %.4f" % (float(train_acc),))

      # training_acc_log.append(train_acc)
      # training_forwards_log.append(self.forward_count)

      # # Reset training metrics at the end of each epoch
      # self.train_acc_metric.reset_states()
    return self.training_acc_log, self.training_forwards_log

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
        # self.log("HERE")
        if layer not in self.layer_std_devs:
          # import pdb; pdb.set_trace()
          for weights in layer.get_weights():
            self.layer_std_devs[layer] = np.std(weights)
            if self.layer_std_devs[layer] > 0:
              self.log(f"layer std dev set to {self.layer_std_devs[layer]}")
              break
          assert(self.layer_std_devs[layer] > 0)
        layer_weights_list = layer.get_weights()
        # import pdb; pdb.set_trace()
        for w_idx in range(len(layer_weights_list), 0, -1):
          w_idx = w_idx - 1
          layer_weights = layer_weights_list[w_idx]
          weights_shape = layer_weights.shape
          flattened_weights = layer_weights.flatten()
          # self.log(f"setting {len(flattened_weights)} weights")
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
    training_forwards_log = []

    for epoch in range(self.epochs):
      self.log(f"Epoch {epoch}")
      for step, (x, y) in tqdm(enumerate(dataset),file=self.LOGGER.tqdm_logger,mininterval=30):
        loss = self.train_step_rso(x,y)

      # TODO: make this every nth
      train_acc = self.train_acc_metric.result()
      self.log("Training acc over epoch: %.4f" % (float(train_acc),))

      training_acc_log.append(train_acc)
      training_forwards_log.append(self.forward_count)

      # Reset training metrics at the end of each epoch
      self.train_acc_metric.reset_states()
    return training_acc_log, training_forwards_log

class BATCH_MODE(Enum):
  EVERY_WEIGHT = "every_weight"
  EVERY_PHASE = "every_phase"
  EVERY_ITER = "every_iteration"

  def __str__(self):
    return self.value

class WEIGHT_CHOICE(Enum):
  SAME = 0
  PLUS_DELTA = 1
  MINUS_DELTA = 2
  ZERO = 3
  ACCIDENTAL_ZERO = 4

class PHASE_TYPE(Enum):
  IMPROVE = "improve"
  PRUNE = "prune"
  REGROW = "regrow"
  REPLACE = "replace"

  def __str__(self):
    return self.value

class SliceInfo(NamedTuple):
  slice_idx: int
  layer: tf.keras.layers.Layer
  weight_idx: int
  weights: np.ndarray
  start_slice_idx: int
  end_slice_idx: int
  weight_shape: list
  og_val: float
  local_idx: int

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
#     - Prune phase (?)

# 	TODO: need to keep logs of when events happen to measure frequencies of improvements/regrows/swaps
# 	- there should be an equilibrium reached where loosing too many weights will cause more regrowths but number of regrowths is fixed… could play about with reducing max_sparsity as alg trains? or encouraging zeroed weights somehow…?

# TODO: need to decide whether to include bias tensors

# TODO: handle batch norm....

  def __init__(self, model, initial_density, maximum_density, initial_prune_factor, swap_proportion, update_iterations, phases, const_norm_weights=False, consider_zero_improve=True, batch_mode=BATCH_MODE.EVERY_ITER,):
    super(SpaRSO, self).__init__(model)
    self.batch_mode = batch_mode
    self.initial_density = initial_density
    self.maximum_density = maximum_density
    self.initial_prune_factor = initial_prune_factor
    self.swap_proportion = swap_proportion
    self.update_iterations = update_iterations
    self.phases = phases
    self.const_norm_weights = const_norm_weights
    self.consider_zero_improve = consider_zero_improve
    # TODO: some sort of maximum sparsity schedule

    # used to store the full parameter space (concatenated and flattened weights of all layers)
    self.flattened_params = np.empty((0), dtype=np.float32)
    # slice_index is the index into the per-slice arrays of the full parameter array
    # search using [slice_index] => global_start or global_end or shape
    self.weight_slice_starts = []
    self.weight_slice_ends = []
    self.weight_shapes = []

    # map from layer, weight_idx => slice_indices
    # search using [layer][layer_weight_idx] => slice_index
    self.layer_weight_index_map = {}

    # map from slice_index to layer and layer_weight index
    self.global_slice_layer_map = []
    self.global_slice_weight_idx_map = []

    # index from global index to slice_index
    self.global_index_global_slice_map = []

    # layer to std dev for layer map
    self.layer_std_devs = {}

    # create a giant flattened array of all parameters
    # takes each layer, flattens it, concatenate it also record a map from each layer index to the start and end index
    # store also each index to the layer and weight index!
    for layer in self.model.layers:
      if self.const_norm_weights and (isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.LayerNormalization)):
        continue
      if layer.trainable_weights:
        # get the std devs for each layer for making random perturbations
        if layer not in self.layer_std_devs:
            if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.LayerNormalization):
              self.layer_std_devs[layer] = 0.1 # TODO: work out what to do for these..
            else:
              # for weights in layer.trainable_weights:
              #   if np.any(weights.numpy()==0):
              #     import pdb; pdb.set_trace()
              for weights in layer.trainable_weights:
                # if np.any(weights.numpy()==0):
                #   import pdb; pdb.set_trace()
                self.layer_std_devs[layer] = np.std(weights)
                if self.layer_std_devs[layer] > 0:
                  self.log(f"layer {layer} std dev set to {self.layer_std_devs[layer]}")
                  break
            assert self.layer_std_devs[layer] > 0, f"layer std dev <= 0 {self.layer_std_devs[layer]}"

        # add all weights to flattened array with slice maps
        for i, weights in enumerate(layer.trainable_weights):
          # include biases for now.. TODO: add a mechanism later to set bias masks first and prevent zero being considered for bias

          current_slice_index = len(self.weight_slice_starts)
          if layer not in self.layer_weight_index_map:
            self.layer_weight_index_map[layer] = []
          self.layer_weight_index_map[layer].append(current_slice_index)
          self.weight_shapes.append(weights.shape)
          self.global_slice_layer_map.append(layer)
          self.global_slice_weight_idx_map.append(i)

          self.weight_slice_starts.append(self.flattened_params.shape[0])
          # import pdb; pdb.set_trace()
          flattened_weights = weights.numpy().flatten()
          # TODO: find a way to set values using assign/tf-ops so that tf can optimise things

          self.global_index_global_slice_map.extend([current_slice_index]*flattened_weights.shape[0])

          self.flattened_params = np.concatenate((self.flattened_params, flattened_weights))

          self.weight_slice_ends.append(self.flattened_params.shape[0])

    # define global mask and masked params
    # define a uniformly random mask over the whole parameter space and multiply by the parameter space
    # do it using an array of non-masked indices in the whole parameter space (can reason on global indicies list rather than lots of local ones..)
    self.total_params = self.flattened_params.shape[0]
    self.sparse_mask = np.zeros([self.total_params],dtype=np.float32)
    init_num_params = int(self.initial_density * self.total_params)
    self.active_params = init_num_params
    self.unmasked_indices = np.sort(np.random.choice(self.total_params,init_num_params,replace=False))
    self.sparse_mask[self.unmasked_indices] = 1
    self.masked_flattened_params = self.flattened_params * self.sparse_mask


    # set masked params per layer
    # set all weights looping through again using the start/end maps back into the full array
    # also some index map integrity checks
    for layer in self.model.layers:
      if self.const_norm_weights and (isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.LayerNormalization)):
        continue
      if layer.trainable_weights:
        # new_weights = []
        for i, weights in enumerate(layer.trainable_weights):
          slice_index = self.layer_weight_index_map[layer][i]
          shape = weights.shape
          assert weights.shape == self.weight_shapes[slice_index], f"weight shapes don't match {weights.shape} and {self.weight_shapes[slice_index]}"
          assert self.global_slice_layer_map[slice_index] == layer
          assert self.global_slice_weight_idx_map[slice_index] == i, f"slice index for weight array map does not equal expected: {slice_index}=>{self.global_slice_weight_idx_map[slice_index]}<>{i}"
          assert self.global_slice_layer_map[slice_index] == layer, f"slice index to layer map does not equal layer for layer {slice_index} => {self.global_slice_layer_map[slice_index]}<>{layer}"
          slice_start = self.weight_slice_starts[slice_index]
          slice_end = self.weight_slice_ends[slice_index]
          weights.assign(self.masked_flattened_params[slice_start:slice_end].reshape(shape))
          # new_weights.append(self.masked_flattened_params[slice_start:slice_end].reshape(shape))
        # if isinstance(layer, tf.keras.layers.BatchNormalization):
        #   import pdb; pdb.set_trace()
        # layer.set_weights(new_weights)
    
    self.every_iter_batch_flag = True
    self.every_phase_batch_flag = True

    assert (self.active_params == (self.sparse_mask>0).sum()), "active params and sparse mask count not equal"

  # def get_global_slice(layer, weight_idx):
  #   ...
  
  def get_slice_info_from_global_index(self, global_idx):
    slice_idx = self.global_index_global_slice_map[global_idx]
    layer = self.global_slice_layer_map[slice_idx]
    weight_idx = self.global_slice_weight_idx_map[slice_idx]
    weights = layer.trainable_weights[weight_idx]
    start_slice_idx = self.weight_slice_starts[slice_idx]
    end_slice_idx = self.weight_slice_ends[slice_idx]
    weight_shape = self.weight_shapes[slice_idx]
    assert weight_shape == weights.shape, f"shapes do not match {weight_shape} and {layer.trainable_weights[weight_idx].shape}"
    assert start_slice_idx <= global_idx <= end_slice_idx, f"indices out of whack {start_slice_idx} <= {global_idx} <= {end_slice_idx}"
    if not ((self.masked_flattened_params[start_slice_idx:end_slice_idx] == weights.numpy().flatten()).all()):
      import pdb; pdb.set_trace()
    assert (self.masked_flattened_params[start_slice_idx:end_slice_idx] == weights.numpy().flatten()).all(), f"masked global and layer weights do not match\n{self.masked_flattened_params[start_slice_idx:end_slice_idx]}\n{weights.numpy().flatten()}\n{layer}"
    og_val = self.masked_flattened_params[global_idx]
    if self.sparse_mask[global_idx] == 0:
    #   self.log(og_val)
    #   if abs(og_val) <= 0:
    #     import pdb; pdb.set_trace()
    #   assert abs(og_val) > 0, f"value {og_val} not greater than zero"
    # else:
      assert og_val == 0, f"value {og_val} expected to be zero"

    local_idx = global_idx - start_slice_idx
    assert weights.numpy().flatten()[local_idx] == self.masked_flattened_params[global_idx], f"local index {local_idx} into weights does not equal global index {global_idx} into global weights"
    return SliceInfo(slice_idx, layer, weight_idx, weights, start_slice_idx, end_slice_idx, weight_shape, og_val, local_idx)

  def set_dataset(self, dataset):
    self.dataset = dataset
    self.datasetiter = iter(self.dataset)
    self.current_batch = None
    self.num_epochs = 0

  def reset_iter(self):
    self.datasetiter = iter(self.dataset)
    self.current_batch = next(self.datasetiter)
    self.num_epochs +=1
    self.log(f"New data epoch reached: {self.num_epochs}")
  
  def save_model_state(self, label):
    # self.log(f"saving model status at {label}")
    # self.LOGGER.save(self.model, f"{label}_model")

    state_dict = {}
    state_dict["flattened_params"] = self.flattened_params
    state_dict["sparse_mask"] = self.sparse_mask
    state_dict["active_params"] = self.active_params
    state_dict["unmasked_indices"] = self.unmasked_indices
    state_dict["masked_flattened_params"] = self.masked_flattened_params
    super().save_model_state(label, state_dict)
    # self.LOGGER.save(state_dict, f"{label}_statedict")




  # TODO: IDEA batch mode with different behaviour for each phase..? or changes during training?
  def get_batch(self):
    # return new batch each time or same batch based on some count or conditions
    if self.batch_mode == BATCH_MODE.EVERY_WEIGHT:
      # always call next
      self.current_batch = next(self.datasetiter)
    elif self.batch_mode == BATCH_MODE.EVERY_PHASE:
      # if every_phase flag set, call next and set flag false
      if self.every_phase_batch_flag:
        self.current_batch = next(self.datasetiter)
        self.every_phase_batch_flag = False
    elif self.batch_mode == BATCH_MODE.EVERY_ITER:
      # if every_iter flag set, call next and set flag false
      if self.every_iter_batch_flag:
        self.current_batch = next(self.datasetiter)
        self.every_iter_batch_flag = False
    else:
      raise Exception(f"Invalid batch mode set: {self.batch_mode}")
    assert self.current_batch, "Current batch is None after get batch call"
    return self.current_batch
  
  def next_batch_phase_mode(self):
    self.every_phase_batch_flag = True
  
  def next_batch_iter_mode(self):
    self.every_iter_batch_flag = True

  def improve_phase(self):
    # loop over non zero indices calling get_batch for each adjustment

    # get batch
    self.next_batch_phase_mode()
    x, y = self.get_batch()
    # get current loss
    current_loss, best_prediction = self.forward_pass(x, y)

    remove_indices_indices_list = []

    # loop through indices in reverse order
    # TODO: add controls for this to permute or change the order, maybe add an order-as-appended mode
    for i, index in tqdm(enumerate(reversed(self.unmasked_indices)),desc="IMPROVE PHASE",file=self.LOGGER.tqdm_logger,mininterval=30):
      assert (self.active_params == (self.sparse_mask>0).sum()), "active params and sparse mask count not equal"
      x, y = self.get_batch()
      choice = WEIGHT_CHOICE.SAME

      # get slice info (layer, weight index/array, og val, etc) for the parameter index
      slice_info = self.get_slice_info_from_global_index(index)
      # layer_weights_list = slice_info.layer.trainable_weights

      flattened_weights = slice_info.weights.numpy().flatten()

      # retest + and - perturbance as per RSO
      new_val = slice_info.og_val
      delta_weight = self.get_delta_weight(self.layer_std_devs[slice_info.layer])

      # TRY + DELTA
      try_val = slice_info.og_val + delta_weight
      flattened_weights[slice_info.local_idx] = try_val
      slice_info.weights.assign(flattened_weights.reshape(slice_info.weight_shape))
      # slice_info.layer.set_weights(layer_weights_list)

      new_loss, new_prediction = self.forward_pass(x, y)

      if new_loss < current_loss:
        self.log(f"CHOSEN PLUS loss={new_loss}, previous_loss={current_loss}, old_val={new_val}, zero_val={try_val}",level=LOG_LEVEL.TRACE)
        choice = WEIGHT_CHOICE.PLUS_DELTA
        new_val = try_val
        current_loss = new_loss
        best_prediction = new_prediction

      # TRY - DELTA
      try_val = slice_info.og_val - delta_weight
      flattened_weights[slice_info.local_idx] = try_val
      slice_info.weights.assign(flattened_weights.reshape(slice_info.weight_shape))
      # slice_info.layer.set_weights(layer_weights_list)

      new_loss, new_prediction = self.forward_pass(x, y)

      if new_loss < current_loss:
        self.log(f"CHOSEN MINUS loss={new_loss}, previous_loss={current_loss}, old_val={new_val}, zero_val={try_val}",level=LOG_LEVEL.TRACE)
        choice = WEIGHT_CHOICE.MINUS_DELTA
        new_val = try_val
        current_loss = new_loss
        best_prediction = new_prediction

      if self.consider_zero_improve:
        # TRY ZERO
        try_val = 0
        flattened_weights[slice_info.local_idx] = try_val
        slice_info.weights.assign(flattened_weights.reshape(slice_info.weight_shape))
        # slice_info.layer.set_weights(layer_weights_list)

        new_loss, new_prediction = self.forward_pass(x, y)

        # check for zero means that already zero values will be removed here
        # TODO: wanted to do this with <= check instead but the loss is the same for all of the values in the first iterations!
        if new_loss < current_loss or new_val==0:
          self.log(f"CHOSEN ZERO loss={new_loss}, previous_loss={current_loss}, old_val={new_val}, zero_val={try_val}",level=LOG_LEVEL.TRACE)
          choice = WEIGHT_CHOICE.ZERO
          new_val = try_val
          current_loss = new_loss
          best_prediction = new_prediction

      # if zero chosen then update the mask and indices list and parameter count
      if new_val == 0:
        if choice != WEIGHT_CHOICE.ZERO:
          # self.log("ALREADY ZERO")
          choice = WEIGHT_CHOICE.ACCIDENTAL_ZERO
          #  shouldn't get here when considering zero to improve as new_loss <= current_loss check should be always be true when value is zero
        # TODO: possibly add logic to not do this when masking turned off (if I add that mode)
        else:
          self.sparse_mask[index] = 0
          self.active_params -= 1
          remove_indices_indices_list.append(len(self.unmasked_indices)-i-1)
          assert self.unmasked_indices[remove_indices_indices_list[-1]] == index, f"index to delete from unmasked nparray {remove_indices_indices_list[-1]} doesn't match index {index}: {self.unmasked_indices[remove_indices_indices_list[-1]]}"
      
      # choose the best and set it in the layer and in the flattened weights and masked weights
      flattened_weights[slice_info.local_idx] = new_val
      self.flattened_params[index] = new_val
      self.masked_flattened_params[index] = new_val
      slice_info.weights.assign(flattened_weights.reshape(slice_info.weight_shape))
      # slice_info.layer.set_weights(layer_weights_list)
      self.train_acc_metric.update_state(y, best_prediction)
    
      # log the choice somewhere
      self.log(f"weight choice {choice} for index {index}",level=LOG_LEVEL.TRACE)

      slice_info = self.get_slice_info_from_global_index(index)
    
    self.unmasked_indices = np.delete(self.unmasked_indices, remove_indices_indices_list)
    assert len(self.unmasked_indices) == self.active_params, f"length of unmasked {self.unmasked_indices} does not equal active count {self.active_params}"
    return
  
  def prune_phase(self):
    # prune either with lowest magnitude
    # TODO: use a smarter metric to decide this
    
    # get batch
    self.next_batch_phase_mode()

    x, y = self.get_batch()
    # get current loss
    current_loss, best_prediction = self.forward_pass(x, y)
    
    sorted_indices_desc = np.argsort(np.abs(self.masked_flattened_params))[::-1]

    cosine_decay = 0.5 * (1 + np.cos(np.pi * self.iteration_count / self.update_iterations))
    pruned_param_factor = self.initial_prune_factor * cosine_decay
    num_pruned = int(self.active_params * pruned_param_factor)

    indices_to_prune = sorted_indices_desc[self.active_params-num_pruned:self.active_params]

    # remove each indice and run assert checks
    for remove_index in tqdm(indices_to_prune, desc="PRUNE PHASE",file=self.LOGGER.tqdm_logger,mininterval=30):
      remove_slice_info = self.get_slice_info_from_global_index(remove_index)
      remove_flattened_weights = remove_slice_info.weights.numpy().flatten()
      remove_try_val = 0
      remove_flattened_weights[remove_slice_info.local_idx] = remove_try_val
      remove_slice_info.weights.assign(remove_flattened_weights.reshape(remove_slice_info.weight_shape))
      self.sparse_mask[remove_index] = 0
      self.flattened_params[remove_index] = 0
      self.masked_flattened_params[remove_index] = 0
      self.unmasked_indices = self.unmasked_indices[self.unmasked_indices != remove_index]
      self.active_params -= 1
      self.log(f"index {remove_index} removed", level=LOG_LEVEL.TRACE)
      
      # TODO: remove calls that run these assert checks after update
      slice_info = self.get_slice_info_from_global_index(remove_index)
    
    new_loss, new_prediction = self.forward_pass(x, y)

    # TODO: save weights
    self.train_acc_metric.update_state(y, new_prediction)
    self.log(f"REMOVE PHASE: removed {num_pruned} weights; loss before = {current_loss}, loss after = {new_loss}")

    self.unmasked_indices = np.sort(self.unmasked_indices)

    return


  def regrow_phase(self):
    # try random values for some number of non-masked weights specified by max_params calling get_batch for each

    # get batch
    self.next_batch_phase_mode()
    x, y = self.get_batch()
    # get current loss
    current_loss, best_prediction = self.forward_pass(x, y)

    # use p = np.ones(total) - self.mask
    assert (self.active_params == (self.sparse_mask>0).sum()), "active params and sparse mask count not equal"
    prob_dist = (np.ones(self.total_params) - self.sparse_mask)/(self.total_params-self.active_params)
    if not np.isclose(np.sum(prob_dist),1):
      import pdb; pdb.set_trace()
    assert np.isclose(np.sum(prob_dist),1), "probability mask does not equal 1"
    # TODO: use a "smart" probability distribution (could I use this for the cross over experiments somehow)
    # TODO: IDEA: could save a probability distribution that keeps track of which params were the least useful..?
    max_num_new_params = int(self.maximum_density*self.total_params) - self.active_params
    self.log(f"max new params = {max_num_new_params} = int({self.maximum_density}*{self.total_params}) - {self.active_params}")
    new_indices = np.random.choice(self.total_params, max_num_new_params, p=prob_dist,replace=False)

    for index in tqdm(new_indices,desc="REGROW PHASE",file=self.LOGGER.tqdm_logger,mininterval=30):
      if not (self.active_params == (self.sparse_mask>0).sum()):
        import pdb; pdb.set_trace()
      assert (self.active_params == (self.sparse_mask>0).sum()), "active params and sparse mask count not equal"
      x, y = self.get_batch()

      # get slice info (layer, weight index/array, og val, etc) for the parameter index
      slice_info = self.get_slice_info_from_global_index(index)
      # layer_weights_list = slice_info.layer.trainable_weights

      flattened_weights = slice_info.weights.numpy().flatten()

      # retest with random - or + perturbance
      new_val = slice_info.og_val
      new_weight = np.random.choice([-1,1],1) * self.get_delta_weight(self.layer_std_devs[slice_info.layer])

      # TRY WITH WEIGHT
      try_val = new_weight
      flattened_weights[slice_info.local_idx] = try_val
      slice_info.weights.assign(flattened_weights.reshape(slice_info.weight_shape))
      # slice_info.layer.set_weights(layer_weights_list)

      new_loss, new_prediction = self.forward_pass(x, y)

      if new_loss < current_loss:
        self.log(f"CHOSEN GROW loss={new_loss}, previous_loss={current_loss}, old_val={new_val}, zero_val={try_val}",level=LOG_LEVEL.TRACE)
        new_val = try_val
        current_loss = new_loss
        best_prediction = new_prediction
        self.sparse_mask[index] = 1
        self.active_params += 1
        self.unmasked_indices = np.append(self.unmasked_indices, index)
        self.log(f"index weight added [{index}]={new_val}", level=LOG_LEVEL.TRACE)
        # self.log(f"growth: {self.active_params}, {(self.sparse_mask>0).sum()}")
      else:
        self.log(f"index weight not added {index}", level=LOG_LEVEL.TRACE)

      flattened_weights[slice_info.local_idx] = new_val
      self.flattened_params[index] = new_val
      self.masked_flattened_params[index] = new_val
      slice_info.weights.assign(flattened_weights.reshape(slice_info.weight_shape))
      # slice_info.layer.set_weights(layer_weights_list)
      self.train_acc_metric.update_state(y, best_prediction)

      slice_info = self.get_slice_info_from_global_index(index)

    self.unmasked_indices = np.sort(self.unmasked_indices)
    return

  def replace_phase(self):
    # for swap proportion number of weights, try a random masked value swapping for a non-masked one calling get_batch for each
    
    # get batch
    self.next_batch_phase_mode()
    x, y = self.get_batch()
    # get current loss
    current_loss, best_prediction = self.forward_pass(x, y)

    weights_to_swap = int(self.swap_proportion * self.active_params)

    # TODO: IDEA: again, could save a probability distribution to make this "smarter"

    # CHOOSE WEIGHTS TO ADD
    # use p = np.ones(total) - self.mask
    assert (self.active_params == (self.sparse_mask>0).sum()), "active params and sparse mask count not equal"
    prob_dist = (np.ones(self.total_params) - self.sparse_mask)/(self.total_params-self.active_params)
    assert np.isclose(np.sum(prob_dist),1), "expecting swap phas prob dist sum to equal 1"
    add_indices = np.random.choice(self.total_params, weights_to_swap, p=prob_dist,replace=False)

    # CHOOSE WEIGHTS TO REMOVE
    # use p = self.mask
    prob_dist = self.sparse_mask/self.active_params
    assert np.isclose(np.sum(prob_dist),1), "probability mask does not equal 1"
    remove_indices = np.random.choice(self.total_params, weights_to_swap, p=prob_dist,replace=False)

    for i in tqdm(range(weights_to_swap),desc="REPLACE PHASE",file=self.LOGGER.tqdm_logger,mininterval=30):
      x, y = self.get_batch()

      remove_index = remove_indices[i]
      add_index = add_indices[i]

      # get slice info for the add parameter index
      add_slice_info = self.get_slice_info_from_global_index(add_index)
      # add_layer_weights_list = add_slice_info.layer.trainable_weights
      add_flattened_weights = add_slice_info.weights.numpy().flatten()
      assert add_slice_info.og_val == 0, f"swap add value not zero {add_index}=>{add_slice_info.og_val}"


      # get slice info for the remove parameter index
      remove_slice_info = self.get_slice_info_from_global_index(remove_index)
      # remove_layer_weights_list = remove_slice_info.layer.trainable_weights
      if remove_slice_info.slice_idx != add_slice_info.slice_idx:
        remove_flattened_weights = remove_slice_info.weights.numpy().flatten()
      else:
        remove_flattened_weights = add_flattened_weights
      # assert abs(remove_slice_info.og_val) > 0, f"swap remove value not > zero {remove_index}=>{remove_slice_info.og_val}" TODO: this assert doesn't work for biases


      # new_val = slice_info.og_val
      new_weight = np.random.choice([-1,1],1) * self.get_delta_weight(self.layer_std_devs[add_slice_info.layer])

      # SET ADD WEIGHT
      add_try_val = new_weight
      add_flattened_weights[add_slice_info.local_idx] = add_try_val
      add_slice_info.weights.assign(add_flattened_weights.reshape(add_slice_info.weight_shape))
      # add_slice_info.layer.set_weights(add_layer_weights_list)


      # REMOVE WEIGHT
      remove_try_val = 0
      remove_flattened_weights[remove_slice_info.local_idx] = remove_try_val
      remove_slice_info.weights.assign(remove_flattened_weights.reshape(remove_slice_info.weight_shape))
      # remove_slice_info.layer.set_weights(remove_layer_weights_list)

      new_loss, new_prediction = self.forward_pass(x, y)

      if new_loss < current_loss:
        # new_val = try_val
        current_loss = new_loss
        best_prediction = new_prediction
        self.sparse_mask[add_index] = 1
        self.flattened_params[add_index] = add_try_val
        self.masked_flattened_params[add_index] = add_try_val
        self.unmasked_indices = np.append(self.unmasked_indices, add_index)
        self.sparse_mask[remove_index] = 0
        self.flattened_params[remove_index] = 0
        self.masked_flattened_params[remove_index] = 0
        self.unmasked_indices = self.unmasked_indices[self.unmasked_indices != remove_index]
        self.log(f"index {remove_index} swapped with new value for {add_index}={add_try_val}", level=LOG_LEVEL.TRACE)
      else:
        # set weights back
        add_flattened_weights[add_slice_info.local_idx] = 0
        remove_flattened_weights[remove_slice_info.local_idx] = remove_slice_info.og_val
        remove_slice_info.weights.assign(remove_flattened_weights.reshape(remove_slice_info.weight_shape))
        # does add slice assign at the end in case remove and add indices are from the same layer, in which case the weights are assined twice
        add_slice_info.weights.assign(add_flattened_weights.reshape(add_slice_info.weight_shape))
        self.log(f"index {remove_index} not swapped with new value for {add_index}", level=LOG_LEVEL.TRACE)

      # TODO: remove calls that run these assert checks after update
      slice_info = self.get_slice_info_from_global_index(add_index)
      slice_info = self.get_slice_info_from_global_index(remove_index)

      # TODO: save weights
      self.train_acc_metric.update_state(y, best_prediction)

    self.unmasked_indices = np.sort(self.unmasked_indices)

    return

  def run_training(self, dataset):
    self.set_dataset(dataset)
    training_acc_log = []
    training_forwards_log = []

    for self.iteration_count in tqdm(range(self.update_iterations),file=sys.stdout):
      self.next_batch_iter_mode()

      for i,phase in enumerate(self.phases):
        assert (self.active_params == (self.sparse_mask>0).sum()), f"active params and sparse mask count not equal at phase {i}:{phase}"
        
        if phase == PHASE_TYPE.IMPROVE:
          self.improve_phase()
        elif phase == PHASE_TYPE.PRUNE:
          self.prune_phase()
        elif phase == PHASE_TYPE.REGROW:
          self.regrow_phase()
        elif phase == PHASE_TYPE.REPLACE:
          self.replace_phase()
        else:
          raise NotImplementedError(f"phase {phase} is not implemented")
        
        self.save_model_state(f"state_{self.iteration_count}_{i}_{phase}")
      
      assert (self.active_params == (self.sparse_mask>0).sum()), "active params and sparse mask count not equal"
      train_acc = self.train_acc_metric.result()
      self.log("Training acc over iteration: %.4f" % (float(train_acc),))
      self.log(f"Number of forward passes {self.forward_count}")

      training_acc_log.append(train_acc)
      training_forwards_log.append(self.forward_count)

      # Reset training metrics at the end of each full iterations
      self.train_acc_metric.reset_states()
    return training_acc_log, training_forwards_log

    # TODO: add logs and training metrics
    


