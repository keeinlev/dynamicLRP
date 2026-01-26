import json
import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from util import run_occlusion_text

module_path = os.path.join(os.getcwd(), '../..')
sys.path.append(module_path)
from lrp_engine import LRPEngine

lrp = LRPEngine(dtype=torch.bfloat16, use_gamma=True)


def run_llama_morf_lerf_dynamiclrp(model, tokenizer, dataset, occlusion_iters=100, num_samples=1000):
    iter_range = range(1, occlusion_iters + 1)
    all_diffs = []
    for example in (pbar := tqdm(dataset["test"].take(num_samples))):
        context = example["text"]
        label = example["label"]
        input_ids = tokenizer(context, return_tensors="pt")["input_ids"]
        output = model(input_ids.to(device))
        if output.logits.argmax() != label:
            continue
        relevance = lrp.run(output.logits)[1][0]
        diffs = []
        
        diffs = run_occlusion_text(model, tokenizer, device, input_ids, label, relevance, occlusion_iters, "class")

        all_diffs.append(diffs)
    
    return all_diffs



if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load Model and tokenizers
    tokenizer = AutoTokenizer.from_pretrained("yash3056/Llama-3.2-1B-imdb")
    model = AutoModelForSequenceClassification.from_pretrained("yash3056/Llama-3.2-1B-imdb", torch_dtype=torch.bfloat16, num_labels=2) #n is the number of labels in the code
    model.model.config._attn_implementation = "sdpa"
    model.to(device)

    diffs = run_llama_morf_lerf_dynamiclrp(model, tokenizer, dataset)

    with open("results/dynamiclrp_llama_imdb_results_attn_rand_tokens.json", "w") as f:
        json.dump({'diffs': diffs}, f)

