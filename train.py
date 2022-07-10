# very loosely based on https://github.com/google-research/lottery-ticket-hypothesis/blob/1f17279d282e729ee29e80a2f750cfbffc4b8500/mnist_fc/constants.py
# and https://www.tensorflow.org/tutorials/keras/classification
import tensorflow as tf

# import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
from dataset import get_fashion_mnist
from models import BasicFFC, RSO_PAPER_MNIST_MODEL
from optimisers import SpaRSO, StandardSGD, WeightPerBatchRSO, WeightsPerBatchRSO, Optimiser, BATCH_MODE
from experiment_logging import create_logger
from args import parse_args

import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

args = parse_args()
LOGGER = create_logger(args.run_description, args)

# TODO:
# - add args for everything
#  - go through each of the optimisers and make sure args are settable and passed through
#  - handle batch norm ops and biases better
#  - args and checks for running the pruning version with sgd (maybe incorporate into the run_training function)
#  - replace all prints with log lines

# BATCH_SIZE = args.batch_size#64#1024#64
# NUM_EPOCHS = args.epochs

train_dataset, test_dataset, class_names = get_fashion_mnist(args.batch_size)

if args.model == "BASIC_FFC":
  model = BasicFFC.get_model()
elif args.model == "RSO_MNIST":
  model = RSO_PAPER_MNIST_MODEL.get_model()
else:
  LOGGER.log("model not implemented")
  exit(1)

LOGGER.log_model_summary(model)

def train(optimiser: Optimiser, dataset):
  training_acc_log = optimiser.run_training(dataset)
  
  train_log_np = np.array(training_acc_log)
  LOGGER.save(train_log_np, f"training_acc_results")

  fig = plt.figure()
  plt.title("Training accuracy every epoch of data") 
  plt.xlabel("Epochs") 
  plt.ylabel("Training accuracy") 
  plt.plot(train_log_np)
  LOGGER.save(fig ,f"training_acc_plot")

# ["SGD","WsPB_RSO","WPB_RSO","spaRSO"]
if args.optimiser == "SGD":
  optimiser = StandardSGD(model, epochs=args.epochs)
elif args.optimiser == "WsPB_RSO":
  optimiser = WeightsPerBatchRSO(model, args.epochs, max_weight_per_iter=args.max_weight_per_iter, random_update_order=args.random_update_order)
elif args.optimiser == "WPB_RSO":
  optimiser = WeightPerBatchRSO(model, number_of_weight_updates=args.opt_iters, random_update_order=args.random_update_order)
elif args.optimiser == "spaRSO":
  optimiser = SpaRSO(model, args.initial_density, args.maximum_density, args.initial_prune_factor, args.swap_proportion, args.opt_iters, consider_zero_improve=args.consider_zero_improve, batch_mode=args.batch_mode)
else:
  LOGGER.log("optimiser not implemented")
  exit(1)

train(optimiser, train_dataset)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

LOGGER.log_eval_results(test_loss, test_acc)

is_pruning = False

if is_pruning:
  LOGGER.log("PRUNING TIME")
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
                    batch_size=args.batch_size, epochs=pruning_epochs,
                    callbacks=callbacks)

  test_loss, test_acc = model_for_pruning.evaluate(test_dataset, verbose=2)

