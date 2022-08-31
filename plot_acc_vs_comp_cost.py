import numpy as np
from matplotlib import pyplot as plt
import sys, pathlib

batch_size=1024

log_dict = {
    "0A":("...","SGD",1.0),
    "0B":("20220829_1935_faa2d8b2ba","SGD_PRUNED 80",1.0),
    "0C":("20220829_1952_faa2d8b2ba","SGD_PRUNED 90",1.0),
    "0D":("20220826_1400_c6ab1f8864","RSO",1.0),
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
    "3A":("20220828_1215_ad8d4348f8","S_min=0.7,S_prune=0.1",0.2), # all s_init=0.8
    "3B":("20220828_1456_ad8d4348f8","S_min=0.6,S_prune=0.1",0.2),
    "3C":("20220828_1834_ad8d4348f8","S_min=0.5,S_prune=0.1",0.2),
    "3D":("20220828_2316_79afd14a6b","S_min=0.7,S_prune=0.2",0.2),
    "3E":("20220829_0224_9c5aebed3c","S_min=0.7,S_prune=0.3",0.2),
    "4A":("20220828_2048_118aace4a3","S_min=0.7,S_prune=0.0,zero_improve",0.2),  # all s_init=0.8
    "4B":("20220829_0046_9c5aebed3c","S_min=0.7,S_prune=0.1,zero_improve",0.2),
    "4C":("20220829_0431_9c5aebed3c","S_replace=0.3,batch_every_phase",0.2),
    "4D":("20220829_0552_9c5aebed3c","S_replace=0.3,batch_every_iter",0.2),
    "5A":("20220829_1301_0891ac381b","combined_phases,S_init=0.8,S_replace=0.3,S_min=0.7,S_prune=0.3",0.2), # needs to be compared against the non combined somehow (previous with 0.3 pruning is 20220828_2056_366016aff5)
    "6A":("20220828_2103_79afd14a6b","warm_up=1,S_replace=0.3",0.2), # compare against the replace only run 0.3
    "6B":("20220828_2340_79afd14a6b","warm_up=2,S_replace=0.3",0.2),
    "6C":("20220829_0223_9c5aebed3c","warm_up=3,S_replace=0.3",0.2),
    "FINALA":("20220829_1318_1f4d3bea6b","Final run replace only",0.2),
    "FINALB":("20220829_1323_1f4d3bea6b","Final run all phases",0.2),
    "FINALC":("20220830_1102_010b5183c8","Final run with bs32, all phases, 10 warm up phases and 100 iters",0.2),
    "FINALD":("20220831_0953_e15020ca50","Final run with bs64, all phases, 10 warm up phases and 100 iters",0.2),
    "FINALE":("...","Final run replace only different config",0.2),
}

def plot(run_list, plot_name, experiment, is_compute_cost, is_plot_train, is_plot_val, baseline_name=None):

    fig = plt.figure()
    plt.title(plot_name)
    plt.xlabel("compute cost" if is_compute_cost else "count forwards")
    plt.ylabel("accuracy")

    for i, run_id in enumerate(run_list):
        (log_id, run_name, default_density) = log_dict[run_id]

        if baseline_name and i==0:
            run_name = baseline_name
        
        fwd_cnts = np.load(f"logs/{log_id}/training_forwards_counts.npy")
        train_accs = np.load(f"logs/{log_id}/training_acc_results.npy")
        val_accs = np.load(f"logs/{log_id}/val_acc_results.npy")

        file = pathlib.Path(f"logs/{log_id}/training_sparsity_log.npy")
        if file.exists():
            sparsity_log = np.load(f"logs/{log_id}/training_sparsity_log.npy")
        else:
            sparsity_log = np.array([default_density]*len(val_accs))

        if (is_compute_cost):
          import pdb; pdb.set_trace()
          fwd_cnts = fwd_cnts * sparsity_log * batch_size
        
        if is_plot_train:
            plt.plot(fwd_cnts, train_accs, label=f"{run_name} - train")
        if is_plot_val:
            plt.plot(fwd_cnts, val_accs, label=f"{run_name} - validation")
    plt.legend(loc="lower right", fontsize="x-small")
    fig.savefig(f"logs/{experiment}/{plot_name}.png")

# EXPERIMENT 1 increasing sparsity analysis
plot(["1A","1B","1C","1D","1E","1F","1G"], 
     "Training accuracy with increasing S_init against forward count",
     experiment="Exp1",
     is_compute_cost=False,
     is_plot_train=True,
     is_plot_val=False)
plot(["1A","1B","1C","1D","1E","1F","1G"], 
     "Validation accuracy with increasing S_init against forward count",
     experiment="Exp1",
     is_compute_cost=False,
     is_plot_train=False,
     is_plot_val=True)
plot(["1A","1B","1C","1D","1E","1F","1G"], 
     "Training accuracy with increasing S_init against compute cost",
     experiment="Exp1",
     is_compute_cost=True,
     is_plot_train=True,
     is_plot_val=False)
plot(["1A","1B","1C","1D","1E","1F","1G"], 
     "Validation accuracy with increasing S_init against compute cost",
     experiment="Exp1",
     is_compute_cost=True,
     is_plot_train=False,
     is_plot_val=True)


# EXPERIMENT 2 replace
# The forward counts look identical because it is just scaled in this case!
plot(["1F","2A","2B","2C","2D","2E"], 
     "Training accuracy with increasing S_replace against forward count",
     experiment="Exp2",
     is_compute_cost=False,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="S_replace=0.0")
plot(["1F","2A","2B","2C","2D","2E"], 
     "Validation accuracy with increasing S_replace against forward count",
     experiment="Exp2",
     is_compute_cost=False,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="S_replace=0.0")
plot(["1F","2A","2B","2C","2D","2E"], 
     "Training accuracy with increasing S_replace against compute cost",
     experiment="Exp2",
     is_compute_cost=True,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="S_replace=0.0")
plot(["1F","2A","2B","2C","2D","2E"], 
     "Validation accuracy with increasing S_replace against compute cost",
     experiment="Exp2",
     is_compute_cost=True,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="S_replace=0.0")
# winner is 0.3


# EXPERIMENT 3 prune and regrow
plot(["1F","3A","3B","3C"], 
     "Training accuracy with increasing S_min against forward count",
     experiment="Exp3",
     is_compute_cost=False,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="S_min=0.8,S_prune=0.0")
plot(["1F","3A","3B","3C"], 
     "Validation accuracy with increasing S_min against forward count",
     experiment="Exp3",
     is_compute_cost=False,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="S_min=0.8,S_prune=0.0")
plot(["1F","3A","3B","3C"], 
     "Training accuracy with increasing S_min against compute cost",
     experiment="Exp3",
     is_compute_cost=True,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="S_min=0.8,S_prune=0.0")
plot(["1F","3A","3B","3C"], 
     "Validation accuracy with increasing S_min against compute cost",
     experiment="Exp3",
     is_compute_cost=True,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="S_min=0.8,S_prune=0.0")

plot(["1F","3A","3D","3E"], 
     "Training accuracy with increasing S_prune against forward count",
     experiment="Exp3",
     is_compute_cost=False,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="S_min=0.8,S_prune=0.0")
plot(["1F","3A","3D","3E"], 
     "Validation accuracy with increasing S_prune against forward count",
     experiment="Exp3",
     is_compute_cost=False,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="S_min=0.8,S_prune=0.0")
plot(["1F","3A","3D","3E"], 
     "Training accuracy with increasing S_prune against compute cost",
     experiment="Exp3",
     is_compute_cost=True,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="S_min=0.8,S_prune=0.0")
plot(["1F","3A","3D","3E"], 
     "Validation accuracy with increasing S_prune against compute cost",
     experiment="Exp3",
     is_compute_cost=True,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="S_min=0.8,S_prune=0.0")

# winner is 0.7 and 0.3
# Plotting best of both on the same chart..? Not needed as winner from first is also plotted in the second

# EXPERIMENT 4_1 - zero improve pruning
plot(["3A","4A","4B"], 
     "Training accuracy with zero_improve_pruning enabled against forward count",
     experiment="Exp4_1",
     is_compute_cost=False,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="S_min=0.7,S_prune=0.1")
plot(["3A","4A","4B"], 
     "Validation accuracy with zero_improve_pruning enabled against forward count",
     experiment="Exp4_1",
     is_compute_cost=False,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="S_min=0.7,S_prune=0.1")
plot(["3A","4A","4B"], 
     "Training accuracy with zero_improve_pruning enabled against compute cost",
     experiment="Exp4_1",
     is_compute_cost=True,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="S_min=0.7,S_prune=0.1")
plot(["3A","4A","4B"], 
     "Validation accuracy with zero_improve_pruning enabled against compute cost",
     experiment="Exp4_1",
     is_compute_cost=True,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="S_min=0.7,S_prune=0.1")
# zero improve is good!

# EXPERIMENT 4_2 - batch modes
plot(["2B","4C","4D"], 
     "Training accuracy with different batch modes against forward count",
     experiment="Exp4_2",
     is_compute_cost=False,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="S_replace=0.3,batch_every_weight")
plot(["2B","4C","4D"], 
     "Validation accuracy with different batch modes against forward count",
     experiment="Exp4_2",
     is_compute_cost=False,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="S_replace=0.3,batch_every_weight")
plot(["2B","4C","4D"], 
     "Training accuracy with different batch modes against compute cost",
     experiment="Exp4_2",
     is_compute_cost=True,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="S_replace=0.3,batch_every_weight")
plot(["2B","4C","4D"], 
     "Validation accuracy with different batch modes against compute cost",
     experiment="Exp4_2",
     is_compute_cost=True,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="S_replace=0.3,batch_every_weight")
# Every phase is the winner (based on validation... just about) but maybe not much in it so leave it as weight to compare with RSO?

# EXPERIMENT 5 - combining grow strategies
plot(["2B","3E","5A"], 
     "Training accuracy with combined phases against forward count",
     experiment="Exp5",
     is_compute_cost=False,
     is_plot_train=True,
     is_plot_val=False,)
plot(["2B","3E","5A"], 
     "Validation accuracy with combined phases against forward count",
     experiment="Exp5",
     is_compute_cost=False,
     is_plot_train=False,
     is_plot_val=True,)
plot(["2B","3E","5A"], 
     "Training accuracy with combined phasess against compute cost",
     experiment="Exp5",
     is_compute_cost=True,
     is_plot_train=True,
     is_plot_val=False,)
plot(["2B","3E","5A"], 
     "Validation accuracy with combined phases against compute cost",
     experiment="Exp5",
     is_compute_cost=True,
     is_plot_train=False,
     is_plot_val=True,)
# replace is the best.. but not taking the best of each! need to rerun with the best of each (only difference is prune factor 0.1=>0.3)
# if adding warm up will need to compare against the warm up version of replace in exp6

# EXPERIMENT 6 warm up phases
plot(["2B","6A","6B","6C"], 
     "Training accuracy with increasing warm up phases against forward count",
     experiment="Exp6",
     is_compute_cost=False,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="warm_up=0,S_replace=0.3")
plot(["2B","6A","6B","6C"], 
     "Validation accuracy with increasing warm up phases against forward count",
     experiment="Exp6",
     is_compute_cost=False,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="warm_up=0,S_replace=0.3")
plot(["2B","6A","6B","6C"], 
     "Training accuracy with increasing warm up phases against compute cost",
     experiment="Exp6",
     is_compute_cost=True,
     is_plot_train=True,
     is_plot_val=False,
     baseline_name="warm_up=0,S_replace=0.3")
plot(["2B","6A","6B","6C"], 
     "Validation accuracy with increasing warm up phases against compute cost",
     experiment="Exp6",
     is_compute_cost=True,
     is_plot_train=False,
     is_plot_val=True,
     baseline_name="warm_up=0,S_replace=0.3")

# TODO: plot param count for prune regrow experiments