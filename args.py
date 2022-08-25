import argparse
from xmlrpc.client import boolean
import numpy as np

from optimisers import BATCH_MODE, PHASE_TYPE
from models import NORM_TYPE

def parse_args():
    # define experiment configuration args
    parser = argparse.ArgumentParser()

    # MAIN ARGS
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int
    )
    parser.add_argument(
        "--model",
        default="RSO_MNIST",
        choices=["RSO_MNIST", "BASIC_FFC"],
        type=str
    )
    parser.add_argument(
        "--optimiser",
        default="SGD",
        choices=["SGD","WsPB_RSO","WPB_RSO","spaRSO"]
    )
    parser.add_argument(
        "--run-classic-pruning",
        action='store_true',
        help="Run low magnitude pruning training phase with a PolynomialDecay schedule"
    )
    parser.add_argument(
        "--pruning-min",
        default=0.5,
        type=float,
        help="Initialised pruning factor (% weights pruned) at start of polynomial decay low magnitude pruning phase"
    )
    parser.add_argument(
        "--pruning-max",
        default=0.8,
        type=float,
        help="Max pruning factor (% weights pruned) at end of polynomial decay low magnitude pruning phase"
    )



    # SGD ARGS
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Number of training epochs (where applicable)"
    )

    # RSO ARGS
    parser.add_argument(
        "--opt-iters",
        default=50,
        type=int,
        help="Number of optimiser iterations (where applicable)"
    )
    parser.add_argument(
        "--max-weight-per-iter",
        default=np.Inf,
        type=int,
        help="Maximum number of weights to be updated per batch iteration"
    )
    parser.add_argument(
        "--random-update-order",
        action='store_true',
        help="Set the weight update order to be randomised"
    )

    # SpaRSO
    parser.add_argument(
        "--initial-density",
        default=0.1,
        type=float,
        help="Fraction of non-zero params at the beginning of training"
    )
    parser.add_argument(
        "--maximum-density",
        default=0.2,
        type=float,
        help="Maximum possible fraction of non-zero params"
    )
    parser.add_argument(
        "--initial-prune-factor",
        default=0.2,
        type=float,
        help="Number of smallest values to prune at start of training (with cosine decayed values to zero afterwards)"
    )
    parser.add_argument(
        "--swap-proportion",
        default=0.2,
        type=float,
        help="Fraction of non-weights to swap with zero weights in the replace phase"
    )
    parser.add_argument(
        "--consider-zero-improve",
        action='store_true',
        help="Consider the value zero during improve phase (testing if removing the param improves the loss)"
    )
    parser.add_argument(
        "--batch-mode",
        type=BATCH_MODE,
        choices=list(BATCH_MODE),
        default=BATCH_MODE.EVERY_ITER,
        help="Choose whether to sample a new batch of data after every weight update, after every phase or after every full iteration"
    )
    parser.add_argument(
        "--norm-type",
        type=NORM_TYPE,
        choices=list(NORM_TYPE),
        default=NORM_TYPE.BATCH,
        help="Choose what sort of normalisation layer to use: batch, layer or none"
    )
    parser.add_argument(
        "--phases",
        type=PHASE_TYPE,
        choices=list(PHASE_TYPE),
        default=[PHASE_TYPE.IMPROVE, PHASE_TYPE.PRUNE, PHASE_TYPE.REGROW, PHASE_TYPE.REPLACE],
        nargs='+',
        help="List of phase types per iteration"
    )
    parser.add_argument(
        "--const-norm-weights",
        action='store_true',
        help="Don't mask and adjust the norm weights and biases (leave as ones and zeros)"
    )



    # LOGGING ARGS
    parser.add_argument(
        "--run-description",
        type=str,
        help="provide a string description of the experiment for future Chris"
    )
    return parser.parse_args()