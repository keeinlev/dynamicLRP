import json
import os
import numpy as np
from matplotlib import pyplot as plt

os.chdir("C:\\Users\\Kevin\\Desktop\\Programming\\research\\lrp\\src\\experiments\\text")

attnlrp_version = ""
dynamiclrp_version = "_attn_gamma"
task = "wiki"

with open(f"results/attnlrp_llama_{task}_results{attnlrp_version}.json", "r") as f:
    attnlrp_results = json.load(f)

attnlrp_logits = attnlrp_results["diffs"]
attnlrp_diffs = [ [ lerf - morf for (lerf, morf) in x ] for x in attnlrp_logits ]

with open(f"results/dynamiclrp_llama_{task}_results{dynamiclrp_version}.json", "r") as f:
    dynamiclrp_results = json.load(f)

dynamiclrp_logits = dynamiclrp_results["diffs"]
dynamiclrp_diffs = [ [ lerf - morf for (lerf, morf) in x ] for x in dynamiclrp_logits ]


attnlrp_sums = [ sum(c) for c in attnlrp_diffs ]
dynamiclrp_sums = [ sum(c) for c in dynamiclrp_diffs ]

num_samples = 60
num_occlusion_iters = 80

plt.plot(range(num_samples), attnlrp_sums)
fig = plt.gcf()
fig.set_figwidth(12)
plt.savefig(f"results/sums_attnlrp_{task}{attnlrp_version}.png")
plt.cla()

plt.plot(range(num_samples), dynamiclrp_sums)
plt.savefig(f"results/sums_dynamiclrp_{task}{dynamiclrp_version}.png")
plt.cla()

offset = 0
for i in range(10):
    fig, ax = plt.subplots(2,2)
    ar = np.array(attnlrp_diffs[i + offset])
    dr = np.array(dynamiclrp_diffs[i + offset])
    ax[0][0].plot(range(num_occlusion_iters), ar)
    ax[0][1].plot(range(num_occlusion_iters), dr)
    ax[1][0].plot(range(num_occlusion_iters), ar.cumsum())
    ax[1][1].plot(range(num_occlusion_iters), dr.cumsum())
    fig.savefig(f"results/llama_comparison_{task}_{i}.png")

ar = np.array(attnlrp_diffs)
dr = np.array(dynamiclrp_diffs)

plt.clf()
plt.plot(range(num_occlusion_iters), ar.transpose((1,0)).mean(-1), label="attnlrp")
plt.plot(range(num_occlusion_iters), dr.transpose((1,0)).mean(-1), label="dynamiclrp")
plt.ylabel("LeRF - MoRF confidence")
plt.xlabel("Occlusion %")
plt.legend()
plt.savefig(f"results/llama_avg_curves_{task}{dynamiclrp_version}{attnlrp_version}.png")
