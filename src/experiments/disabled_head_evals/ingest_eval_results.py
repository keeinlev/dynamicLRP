import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

results_path = os.path.join("output", "dnabert2", "results")

# Config for DNABERT-2
NUM_LAYERS = 12
NUM_HEADS = 12

# Uncomment the targets you want to generate heatmaps for
tasks = {
    ## The keys here are TASKS
    "EMP": [
        ## The strings in here are TARGETS
    #     "H3",
    #     "H3K14ac",
    #     "H3K36me3",
    #     "H3K4me1",
    #     "H3K4me2",
    #     "H3K4me3",
    #     "H3K79me3",
    #     "H3K9ac",
    #     "H4",
    #     "H4ac",
    ],
    "prom": [
    #     "prom_core_all",
    #     "prom_core_notata",
    #     "prom_core_tata",
    #     "prom_300_all",
    #     "prom_300_notata",
    #     "prom_300_tata",
    ],
    "splice": [
    #     "reconstructed",
    ],
    "virus": [
    #     "covid",
    ],
    "tf": [
    #     "mouse0",
    #     "mouse1",
    #     "mouse2",
    #     "mouse3",
    #     "mouse4",
    #     "human0",
    #     "human1",
    #     "human2",
    #     "human3",
    #     "human4",
    ],
}

# How many maps we see on each PDF, corresponds to the metric keys returned by the HF trainer.evaluate() in train.py under evaluate()
metrics = ["eval_loss", "eval_accuracy", "eval_f1", "eval_matthews_correlation", "eval_precision", "eval_recall"]

metric_matrices = {
    task : {
        target : {
            metric : [ [] for _ in range(NUM_LAYERS) ] for metric in metrics
        } for target in tasks[task]
    } for task in tasks
}

sns.set_style("white")
for task, targets in list(tasks.items()):
    for target in targets:
        fig, axs = plt.subplots((len(metrics) + 1) // 2, 2, figsize=(10,10))
        fig.subplots_adjust(top=1.0)
        
        with open(os.path.join(results_path, f"DNABERT2_{task}_{target}", "disabled_head_results.json"), "r") as fileIn:
            results = json.load(fileIn)

        # Parse the results JSON file, organize into the metric_matrices map
        for layer_ind in range(NUM_LAYERS):
            layer_result = results[layer_ind]
            for head_ind in range(NUM_HEADS):
                head_result = layer_result["head_results"][head_ind]
                for metric in metrics:
                    metric_matrices[task][target][metric][layer_ind].append(head_result[metric])
        
        # Plotting and saving as PDF
        for i, metric in enumerate(metrics):
            sns.heatmap(metric_matrices[task][target][metric], vmin=0.0, square=True, ax=axs[i // 2, i % 2])
            axs[i // 2, i % 2].set_title(metric)
            axs[i // 2, i % 2].set_xlabel("head index")
            axs[i // 2, i % 2].set_ylabel("layer index")
        fig.suptitle(f"{task}-{target} task single head disabling evaluation results")
        fig.tight_layout()

        fig_save_path = os.path.join("eval_heatmaps", task)
        if not os.path.isdir(fig_save_path):
            os.mkdir(fig_save_path)
        fig.savefig(os.path.join(fig_save_path, f"{task}_{target}.png"))
        plt.close(fig)

        del metric_matrices[task][target] # save memory
            


