import numpy as np
from matplotlib import pyplot as plt
import sys, pathlib

# usage: call this script from top of repo with a space separated list of log ids
log_dict = {
    "1A":"20220826_1621_a634e5b000",
    "1B":"20220827_1048_45b9eb8795",
    "1C":"20220827_2120_dc1ed85beb",
    "1D":"20220828_0606_dc1ed85beb",
    "1E":"20220828_1309_ad8d4348f8",
    # "1F":"20220828_1833_ad8d4348f8",
    # "1G":"",
    # "2A":"20220827_1111_76f8cde6b8",
    # "2B":"20220827_1524_9cb22be385",
    # "2C":"20220827_1929_9cb22be385",
    # "2D":"20220827_2338_dc1ed85beb",
    # "2E":"20220828_0345_dc1ed85beb",
    # "3A":"20220828_1215_ad8d4348f8",
    # "3B":"20220828_1456_ad8d4348f8",
    # "3C":"20220828_1834_ad8d4348f8",
    # "3D":"",
    # "3E":"",
}

fig = plt.figure()
plt.title("validation accuracy against compute cost")
plt.xlabel("compute cost")
plt.ylabel("accuracy")

for (run_id, log_id) in log_dict.items():
    # fig = plt.figure()
    # plt.title("validation accuracy against compute cost")
    # plt.xlabel("compute cost")
    # plt.ylabel("accuracy")
    fwd_cnts = np.load(f"logs/{log_id}/training_forwards_counts.npy")
    train_accs = np.load(f"logs/{log_id}/training_acc_results.npy")
    val_accs = np.load(f"logs/{log_id}/val_acc_results.npy")

    file = pathlib.Path(f"logs/{log_id}/training_sparsity_log.npy")
    if file.exists():
        sparsity_log = np.load(f"logs/{log_id}/training_sparsity_log.npy")
    else:
        sparsity_log = np.array([0.2]*len(val_accs))

    fwd_cnts = fwd_cnts * sparsity_log
    
    # plt.plot(fwd_cnts, train_accs, label="training accuracy")
    plt.plot(fwd_cnts, val_accs, label=f"experiment {run_id}")
    # plt.legend(loc="upper left")
    # fig.savefig(f"logs/{log_id}/plot_val_acc_comp_cost.png")
plt.legend(loc="upper left")
fig.savefig(f"logs/plot_val_acc_comp_cost.png")