import json
import numpy as np
import os
import sys
import torch
from transformers.models.llama import modeling_llama
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from datasets.utils.info_utils import VerificationMode
from util import run_occlusion_text

module_path = os.path.join(os.getcwd(), '../../../external/LRP-eXplains-Transformers')
sys.path.append(module_path)
from lxt.efficient import monkey_patch

model_name = "meta-llama/Llama-3.2-1B"


run_name = "500_5_100"


def sample_positions_for_eval(input_ids, num_positions=5):
    """Sample positions to evaluate, avoiding very early positions"""
    seq_len = input_ids.shape[-1]
    
    # Avoid first few tokens (not enough context) and last token (no prediction)
    valid_range = range(10, seq_len - 1)
    
    if len(valid_range) <= num_positions:
        return list(valid_range)
    
    # Sample uniformly across the sequence
    indices = np.linspace(10, seq_len - 2, num_positions, dtype=int)
    return indices.tolist()


def run_causal_lm_morf_lerf(model, tokenizer, dataset, 
                             num_samples=500,
                             positions_per_sample=5,
                             occlusion_iters=100):
    all_confidences = []
    
    for example in tqdm(dataset.shuffle(seed=42).take(num_samples)):
        text = example["text"]
        input_ids = tokenizer(text, return_tensors="pt", 
                             max_length=512, truncation=True)["input_ids"]
        
        # Sample positions to evaluate
        eval_positions = sample_positions_for_eval(input_ids, positions_per_sample)

        for pos in eval_positions:
            # Get context (everything before pos)
            context_ids = input_ids[:, :pos+1]
            true_next_token = input_ids[0, pos+1].item()
            
            # Get prediction and setup for AttnLRP
            input_embeds = model.model.embed_tokens(context_ids.to(device)).requires_grad_()
            input_embeds.retain_grad()
            output = model(inputs_embeds=input_embeds)
            # predicted_token = output.logits[0, -1].argmax().item()
            
            # Skip if wrong prediction
            # if predicted_token != true_next_token:
            #     continue

            # Compute LRP relevances for this position
            output.logits[0, -1].max().backward()
            relevance = (input_embeds * input_embeds.grad).sum(-1).detach().cpu()
            
            # Run MoRF/LeRF occlusion
            confidences = run_occlusion_text(
                model, tokenizer, device, context_ids, true_next_token,
                relevance, occlusion_iters, mode="causal"
            )
            
            all_confidences.append(confidences)

            torch.cuda.empty_cache()
    
    return all_confidences



if __name__ == "__main__":
    # Only take a manageable subset of the dataset
    data_files = [ f"20231101.en/train-0000{i}-of-00041.parquet" for i in range(7) ]
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", data_files=data_files, verification_mode=VerificationMode.NO_CHECKS) # Need to bypass the checks for split size
    monkey_patch(modeling_llama)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load Model and tokenizers
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = modeling_llama.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.model.config._attn_implementation = "sdpa"
    model.to(device)

    diffs = run_causal_lm_morf_lerf(model, tokenizer, dataset["train"])

    os.makedirs("results", exist_ok=True)
    with open(f"results/attnlrp_llama_wiki_results_{run_name}.json", "w") as f:
        json.dump({'diffs': diffs}, f)

