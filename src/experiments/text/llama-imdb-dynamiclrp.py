import json
import os
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from util import run_occlusion_text

module_path = os.path.join(os.getcwd(), '../..')
sys.path.append(module_path)
from lrp_engine import LRPEngine

lrp = LRPEngine(dtype=torch.bfloat16, use_gamma=True)


run_name = "gamma_attn_sample"

def run_llama_morf_lerf_dynamiclrp(model, tokenizer, dataset, occlusion_type="random", occlusion_iters=100, num_samples=11):
    all_logits = []
    all_confidences = []
    for example in (tqdm(dataset["test"].take(num_samples))):
        context = example["text"]
        label = example["label"]
        input_ids = tokenizer(context, return_tensors="pt")["input_ids"]
        output = model(input_ids.to(device))
        if output.logits.argmax() != label:
            continue
        relevance = lrp.run(output.logits)[1][0]

        with torch.no_grad():
            logits, confidences = run_occlusion_text(model, tokenizer, device, input_ids, label, relevance, occlusion_iters, "class", occlusion_type)

        all_logits.append(logits)
        all_confidences.append(confidences)
        torch.cuda.empty_cache()
    
    return all_logits, all_confidences



if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load Model and tokenizers
    tokenizer = AutoTokenizer.from_pretrained("yash3056/Llama-3.2-1B-imdb")
    model = AutoModelForSequenceClassification.from_pretrained("yash3056/Llama-3.2-1B-imdb", torch_dtype=torch.bfloat16, num_labels=2) #n is the number of labels in the code
    model.model.config._attn_implementation = "sdpa"
    model.to(device)

    logits, confs = run_llama_morf_lerf_dynamiclrp(model, tokenizer, dataset)

    with open(f"results/dynamiclrp_llama_imdb_results_{run_name}_random.json", "w") as f:
        json.dump({'logits': logits, 'confs': confs}, f)

    logits2, confs2 = run_llama_morf_lerf_dynamiclrp(model, tokenizer, dataset, occlusion_type="zero")

    with open(f"results/dynamiclrp_llama_imdb_results_{run_name}_zero.json", "w") as f:
        json.dump({'logits': logits2, 'confs': confs2}, f)

