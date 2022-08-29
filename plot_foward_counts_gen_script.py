import numpy as np
from matplotlib import pyplot as plt
import sys

# usage: call this script from top of repo with a space separated list of log ids

for log_id in sys.argv[1:]:
    fwd_cnts = np.load(f"logs/{log_id}/training_forwards_counts.npy")
    train_accs = np.load(f"logs/{log_id}/training_acc_results.npy")
    val_accs = np.load(f"logs/{log_id}/val_acc_results.npy")
    fig = plt.figure()
    plt.title("training and validation accuracy against forward execution counts")
    plt.xlabel("Count of calls to forward function")
    plt.ylabel("accuracy")
    plt.plot(fwd_cnts, train_accs, label="training accuracy")
    plt.plot(fwd_cnts, val_accs, label="validation accuracy")
    plt.legend(loc="lower right")
    fig.savefig(f"logs/{log_id}/plot_accs_fwd_cnts.png")