import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from datasets.utils.info_utils import VerificationMode
from util import run_occlusion_text, sample_positions_for_eval

module_path = os.path.join(os.getcwd(), '../..') # Path to src
sys.path.append(module_path)
from lrp_engine import LRPEngine

lrp = LRPEngine(dtype=torch.bfloat16, use_gamma=True)
model_name = "meta-llama/Llama-3.2-1B"


run_name = "gamma_1000_100"


def run_causal_lm_morf_lerf(model, tokenizer, dataset,
                             occlusion_type="random",
                             num_samples=1000,
                            #  positions_per_sample=2,
                             occlusion_iters=100):
    all_logits = []
    all_confidences = []
    
    for example in tqdm(dataset.shuffle(seed=42).take(num_samples)):
        text = example["text"]
        input_ids = tokenizer(text, return_tensors="pt", 
                             max_length=512, truncation=True)["input_ids"]
        
        # Sample positions to evaluate
        # eval_positions = sample_positions_for_eval(input_ids, positions_per_sample)

        # for pos in eval_positions:
            # Get context (everything before pos)
            # context_ids = input_ids[:, :pos+1]
            # true_next_token = input_ids[0, pos+1].item()

        # Just do last token prediction
        pos = input_ids.shape[-1] - 1
        context_ids = input_ids[:, :pos]
        true_next_token = input_ids[0, pos].item()

        # Get prediction
        output = model(context_ids.to(device))
        # predicted_token = output.logits[0, -1].argmax().item()
        
        # Skip if wrong prediction
        # if predicted_token != true_next_token:
        #     continue

        # Compute LRP relevances for this position
        relevance = lrp.run(output.logits[0, -1])[1][0]
        
        # Run MoRF/LeRF occlusion
        with torch.no_grad():
            logits, confidences = run_occlusion_text(
                model, tokenizer, device, context_ids, true_next_token,
                relevance, occlusion_iters, mode="causal", occlusion_type=occlusion_type
            )
        
        all_logits.append(logits)
        all_confidences.append(confidences)

        torch.cuda.empty_cache()
    
    return all_logits, all_confidences



if __name__ == "__main__":
    # Only take a manageable subset of the dataset
    data_files = [ f"20231101.en/train-0000{i}-of-00041.parquet" for i in range(7) ]
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", data_files=data_files, verification_mode=VerificationMode.NO_CHECKS) # Need to bypass the checks for split size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load Model and tokenizers
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.model.config._attn_implementation = "sdpa"
    model.to(device)

    logits1, confs1 = run_causal_lm_morf_lerf(model, tokenizer, dataset["train"])

    os.makedirs("results", exist_ok=True)
    with open(f"results/dynamiclrp_llama_wiki_results_{run_name}_random.json", "w") as f:
        json.dump({'logits': logits1, 'confs': confs1}, f)

    logits2, confs2 = run_causal_lm_morf_lerf(model, tokenizer, dataset["train"], occlusion_type="zero")

    with open(f"results/dynamiclrp_llama_wiki_results_{run_name}_zero.json", "w") as f:
        json.dump({'logits': logits2, 'confs': confs2}, f)

