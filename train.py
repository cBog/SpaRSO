# very loosely based on https://github.com/google-research/lottery-ticket-hypothesis/blob/1f17279d282e729ee29e80a2f750cfbffc4b8500/mnist_fc/constants.py
# and https://www.tensorflow.org/tutorials/keras/classification
import tensorflow as tf

# import tensorflow_datasets as tfds
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from dataset import get_fashion_mnist
from models import BasicFFC
from optimisers import StandardSGD, WeightPerBatchRSO, WeightsPerBatchRSO, Optimiser

BATCH_SIZE = 64
NUM_EPOCHS = 10

train_dataset, test_dataset, class_names = get_fashion_mnist(BATCH_SIZE)

model = BasicFFC.get_model()

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
rso1weightOptimiser = WeightPerBatchRSO(model, epochs=1)
rsoManyWeightsOptimiser = WeightsPerBatchRSO(model, 1, max_weight_per_iter=200)

train(rso1weightOptimiser, train_dataset)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
