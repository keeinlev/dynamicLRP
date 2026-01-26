import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from datasets import load_dataset

module_path = os.path.join(os.getcwd(), '../..')
sys.path.append(module_path)

from lrp_engine import LRPEngine, checkpoint_hook

lrp = LRPEngine(use_attn_lrp=True, dtype=torch.bfloat16)
dataset = load_dataset("stanfordnlp/imdb")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load Model and tokenizers
tokenizer = AutoTokenizer.from_pretrained("yash3056/Llama-3.2-1B-imdb")
model = AutoModelForSequenceClassification.from_pretrained("yash3056/Llama-3.2-1B-imdb", torch_dtype=torch.bfloat16, num_labels=2) #n is the number of labels in the code
model.model.config._attn_implementation = "sdpa"

model.to(device)

attrs = []
correct_examples = []

occlusion_iters = 256
iter_range = range(occlusion_iters + 1) if attrs == [] else range(1, occlusion_iters + 1)
accuracies = []
logits = []
for occlusion_iter in tqdm(iter_range):
    correct = 0
    incorrect = 0
    for i, example in enumerate(dataset["test"]):
        if i > 1000:
            break
        if occlusion_iter > 0 and correct_examples[i] == False:
            continue
        context = example["text"]
        label = example["label"]
    
        input_ids = tokenizer(context, return_tensors="pt")["input_ids"]

        if occlusion_iter > 0:
            if occlusion_iter > attrs[i].shape[-1]:
                continue
            mask_inds = attrs[i].flatten().abs().topk(occlusion_iter).indices.cpu()
            input_ids[0][mask_inds] = tokenizer.pad_token_id

        output = model(input_ids.to(device))
        logits.append(output.logits.detach().cpu())

        if output.logits.argmax() == label:
            if occlusion_iter == 0:
                correct_examples.append(True)
            correct += 1
        else:
            if occlusion_iter == 0:
                correct_examples.append(False)
            incorrect += 1

        if occlusion_iter == 0:
            lrp_output = lrp.run(output.logits)
            relevance = lrp_output[1][0].detach().cpu()
            attrs.append(relevance)

    accuracies.append((correct, incorrect))
    print(accuracies[-1])