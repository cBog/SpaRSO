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
from numpy.random import seed
import tensorflow_model_optimization as tfmot
import random

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

args = parse_args()

random.seed(args.seed)
seed(args.seed)
tf.random.set_seed(args.seed)

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
  model = RSO_PAPER_MNIST_MODEL.get_model(args.norm_type)
else:
  LOGGER.log("model not implemented")
  exit(1)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

LOGGER.log_model_summary(model)

def train(optimiser: Optimiser, dataset, test_dataset):
  training_acc_log, training_forwards_log, val_acc_log = optimiser.run_training(dataset, test_dataset)
  
  train_log_np = np.array(training_acc_log)
  LOGGER.save(train_log_np, f"training_acc_results")
  val_log_np = np.array(val_acc_log)
  LOGGER.save(val_log_np, f"val_acc_results")
  train_fwds_log_np = np.array(training_forwards_log)
  LOGGER.save(train_fwds_log_np, f"training_forwards_counts")

  fig = plt.figure()
  plt.title("Training accuracy every epoch of data") 
  plt.xlabel("Epochs") 
  plt.ylabel("Training accuracy") 
  plt.plot(train_log_np)
  LOGGER.save(fig ,f"training_acc_plot")

  fig = plt.figure()
  plt.title("Validation accuracy every epoch/cycle") 
  plt.xlabel("Epochs/Cycles") 
  plt.ylabel("Validation Accuracy") 
  plt.plot(val_log_np)
  LOGGER.save(fig ,f"validation_acc_plot")
  return optimiser.model

# ["SGD","WsPB_RSO","WPB_RSO","spaRSO"]
if args.optimiser == "SGD":
  optimiser = StandardSGD(model, epochs=args.epochs)
elif args.optimiser == "WsPB_RSO":
  optimiser = WeightsPerBatchRSO(model, args.epochs, max_weight_per_iter=args.max_weight_per_iter, random_update_order=args.random_update_order)
elif args.optimiser == "WPB_RSO":
  optimiser = WeightPerBatchRSO(model, number_of_weight_updates=args.opt_iters, random_update_order=args.random_update_order)
elif args.optimiser == "spaRSO":
  optimiser = SpaRSO(model=model, 
                     initial_density=args.initial_density, 
                     maximum_density=args.maximum_density, 
                     initial_prune_factor=args.initial_prune_factor, 
                     swap_proportion=args.swap_proportion, 
                     update_iterations=args.opt_iters, 
                     phases=args.phases,
                     warm_up_replace_phases=args.warm_up_replace_phases,
                     const_norm_weights=args.const_norm_weights,
                     consider_zero_improve=args.consider_zero_improve, 
                     batch_mode=args.batch_mode)
else:
  LOGGER.log("optimiser not implemented")
  exit(1)

model = train(optimiser, train_dataset, test_dataset)
# import pdb; pdb.set_trace()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)


LOGGER.log_eval_results(test_loss, test_acc)

LOGGER.log(f"Number of forward passes: {optimiser.forward_count.numpy()}")

is_pruning = args.run_classic_pruning

if is_pruning:
  LOGGER.log("PRUNING TIME")
  pruning_epochs = 2
  num_batches = len(train_dataset)
  end_step = np.ceil(num_batches).astype(np.int32) * pruning_epochs

  # Define model for pruning.
  # from https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
  pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=args.pruning_min,
                                                                final_sparsity=args.pruning_max,
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
    tfmot.sparsity.keras.PruningSummaries(log_dir=LOGGER.id_dir_path),
  ]

  model_for_pruning.fit(train_dataset,
                    batch_size=args.batch_size, epochs=pruning_epochs,
                    callbacks=callbacks)

  test_loss, test_acc = model_for_pruning.evaluate(test_dataset, verbose=2)

  LOGGER.log(f"EVAL AFTER PRUNING FROM {args.pruning_min} TO {args.pruning_max}")
  LOGGER.log_eval_results(test_loss, test_acc)

