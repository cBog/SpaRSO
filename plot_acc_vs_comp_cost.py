import numpy as np
from matplotlib import pyplot as plt
import sys, pathlib

log_dict = {
    "1A":("20220826_1621_a634e5b000","S_init=0.3",0.7),
    "1B":("20220827_1048_45b9eb8795","S_init=0.4",0.6),
    "1C":("20220827_2120_dc1ed85beb","S_init=0.5",0.5),
    "1D":("20220828_0606_dc1ed85beb","S_init=0.6",0.4),
    "1E":("20220828_1309_ad8d4348f8","S_init=0.7",0.3),
    "1F":("20220828_1833_ad8d4348f8","S_init=0.8",0.2),
    "1G":("20220829_0044_9c5aebed3c","S_init=0.9",0.1),
    "2A":("20220828_2020_82bf403a2f","S_replace=0.2",0.2), # all s_init=0.8
    "2B":("20220828_2240_79afd14a6b","S_replace=0.3",0.2),
    "2C":("20220829_0101_9c5aebed3c","S_replace=0.4",0.2),
    "2D":("20220829_0340_9c5aebed3c","S_replace=0.5",0.2),
    "2E":("20220829_0606_9c5aebed3c","S_replace=0.6",0.2),
    "3A":("20220828_1215_ad8d4348f8","S_max=0.3,S_prune=0.1",0.2), # all s_init=0.8
    "3B":("20220828_1456_ad8d4348f8","S_max=0.4,S_prune=0.1",0.2),
    "3C":("20220828_1834_ad8d4348f8","S_max=0.5,S_prune=0.1",0.2),
    "3D":("20220828_2316_79afd14a6b","S_max=0.3,S_prune=0.2",0.2),
    "3E":("20220829_0224_9c5aebed3c","S_max=0.3,S_prune=0.3",0.2),
    "4A":("20220828_2048_118aace4a3","S_max=0.3,S_prune=0.0,zero_improve",0.2),  # all s_init=0.8
    "4B":("20220829_0046_9c5aebed3c","S_max=0.3,S_prune=0.1,zero_improve",0.2),
    "4C":("20220829_0431_9c5aebed3c","S_replace=0.3,batch_every_phase",0.2),
    "4D":("20220829_0552_9c5aebed3c","S_replace=0.3,batch_every_iter",0.2),
    "5A":("20220828_2056_366016aff5","combined_phases,S_init=0.8,S_replace=0.3,S_max=0.3,S_prune=0.1",0.2), # needs to be compared against the non combined somehow
    "6A":("20220828_2103_79afd14a6b","warm_up=1,S_replace=0.3",0.2), # compare against the replace only run 0.3
    "6B":("20220828_2340_79afd14a6b","warm_up=2,S_replace=0.3",0.2),
    "6C":("20220829_0223_9c5aebed3c","warm_up=3,S_replace=0.3",0.2),
}

def plot(run_list, plot_name, is_compute_cost, is_plot_train, is_plot_val):

    fig = plt.figure()
    plt.title(plot_name)
    plt.xlabel("compute cost" if is_compute_cost else "count forwards")
    plt.ylabel("accuracy")

    for run_id in run_list:
        (log_id, run_name, default_density) = log_dict[run_id]
        
        fwd_cnts = np.load(f"logs/{log_id}/training_forwards_counts.npy")
        train_accs = np.load(f"logs/{log_id}/training_acc_results.npy")
        val_accs = np.load(f"logs/{log_id}/val_acc_results.npy")

        file = pathlib.Path(f"logs/{log_id}/training_sparsity_log.npy")
        if file.exists():
            sparsity_log = np.load(f"logs/{log_id}/training_sparsity_log.npy")
        else:
            sparsity_log = np.array([default_density]*len(val_accs))

        if (is_compute_cost):
            fwd_cnts = fwd_cnts * sparsity_log
        
        if is_plot_train:
            plt.plot(fwd_cnts, train_accs, label=f"{run_name} - train")
        if is_plot_val:
            plt.plot(fwd_cnts, val_accs, label=f"{run_name} - validation")
    plt.legend(loc="lower right", fontsize="x-small")
    fig.savefig(f"logs/{plot_name}.png")

plot(["1A","1B","1C","1D","1E","1F","1G"], 
     "Training accuracy with increasing S_init against forward count",
     is_compute_cost=False,
     is_plot_train=True,
     is_plot_val=False)
plot(["1A","1B","1C","1D","1E","1F","1G"], 
     "Validation accuracy with increasing S_init against forward count",
     is_compute_cost=False,
     is_plot_train=False,
     is_plot_val=True)
plot(["1A","1B","1C","1D","1E","1F","1G"], 
     "Training accuracy with increasing S_init against compute cost",
     is_compute_cost=True,
     is_plot_train=True,
     is_plot_val=False)
plot(["1A","1B","1C","1D","1E","1F","1G"], 
     "Validation accuracy with increasing S_init against compute cost",
     is_compute_cost=True,
     is_plot_train=False,
     is_plot_val=True)