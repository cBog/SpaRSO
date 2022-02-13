# very loosely based on https://github.com/google-research/lottery-ticket-hypothesis/blob/1f17279d282e729ee29e80a2f750cfbffc4b8500/mnist_fc/constants.py
# and https://www.tensorflow.org/tutorials/keras/classification
import tensorflow as tf

# import tensorflow_datasets as tfds
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from dataset import get_fashion_mnist
from models import BasicFFC, RSO_PAPER_MNIST_MODEL
from optimisers import StandardSGD, WeightPerBatchRSO, WeightsPerBatchRSO, Optimiser

import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

BATCH_SIZE = 64#1024#64
NUM_EPOCHS = 10

train_dataset, test_dataset, class_names = get_fashion_mnist(BATCH_SIZE)

# model = BasicFFC.get_model()
model = RSO_PAPER_MNIST_MODEL.get_model()
model.summary()
# import pdb; pdb.set_trace()

def train(optimiser: Optimiser, dataset):
  training_acc_log = optimiser.run_training(dataset)
  
  train_log_np = np.array(training_acc_log)
  datetimestr = datetime.now().strftime("%m%d%Y_%H%M%S")
  np.save(f"{datetimestr}_training_acc_results", train_log_np)

  fig = plt.figure()
  plt.title("Training accuracy every epoch of data") 
  plt.xlabel("Epochs") 
  plt.ylabel("Training accuracy") 
  plt.plot(train_log_np)
  fig.savefig(f"{datetimestr}_training_acc_plot.png")

sgdOptimiser = StandardSGD(model, epochs=10)
# TODO: add all options for these to constructor (e.g. random_order)
rso1weightOptimiser = WeightPerBatchRSO(model, number_of_weight_updates=50, random_update_order=False) # (50 updates in the paper with batch size 5000?)
rsoManyWeightsOptimiser = WeightsPerBatchRSO(model, 1, max_weight_per_iter=200, random_update_order=True)

train(sgdOptimiser, train_dataset)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

is_pruning = True

if is_pruning:
  print("PRUNING TIME")
  pruning_epochs = 2
  num_batches = len(train_dataset)
  end_step = np.ceil(num_batches).astype(np.int32) * pruning_epochs

  # Define model for pruning.
  pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.80,
                                                                begin_step=0,
                                                                end_step=end_step)
  }

  model_for_pruning = prune_low_magnitude(model, **pruning_params)

  # `prune_low_magnitude` requires a recompile.
  model_for_pruning.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  model_for_pruning.summary()
  callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    # tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
  ]

  model_for_pruning.fit(train_dataset,
                    batch_size=BATCH_SIZE, epochs=pruning_epochs,
                    callbacks=callbacks)

  test_loss, test_acc = model_for_pruning.evaluate(test_dataset, verbose=2)

