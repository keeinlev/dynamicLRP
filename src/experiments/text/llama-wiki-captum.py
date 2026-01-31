import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets.utils.info_utils import VerificationMode
from captum.attr import IntegratedGradients, GradientShap#, LayerIntegratedGradients, LayerGradientShap
from sklearn.metrics import auc
from util import run_occlusion_text

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_ig(model, input_ids, attention_mask, target, embedding_layer, target_idx=-1):
    # Integrated Gradients on Embeddings
    # We wrap the model to handle dtype and output selection
    def forward_wrapper(inputs_embeds):
        # inputs_embeds might be float32 from Captum, model might be bfloat16
        if inputs_embeds.dtype != model.dtype:
            inputs_embeds = inputs_embeds.to(model.dtype)
        
        outputs = model(inputs_embeds=inputs_embeds.to(device), attention_mask=attention_mask)
        logits = outputs.logits
        
        # Select the target logit
        if logits.dim() == 2: # Classification [batch, num_labels]
            return logits[:, target]
        else: # NWP [batch, seq, vocab]
            # Captum expands batch dim, so we use : for batch
            return logits[:, target_idx, target]

    inputs_embeds = embedding_layer(input_ids.to(device)).detach()
    # Run in float32 for Captum stability
    inputs_embeds_float = inputs_embeds.float()
    inputs_embeds_float.requires_grad = True
    
    ig = IntegratedGradients(forward_wrapper)
    attributions = ig.attribute(inputs_embeds_float, n_steps=50)
    return attributions.sum(dim=-1).detach().cpu()

def run_gradshap(model, input_ids, attention_mask, target, embedding_layer, target_idx=-1):
    # GradientSHAP on Embeddings
    def forward_wrapper(inputs_embeds):
        if inputs_embeds.dtype != model.dtype:
            inputs_embeds = inputs_embeds.to(model.dtype)
        
        outputs = model(inputs_embeds=inputs_embeds.to(device), attention_mask=attention_mask)
        logits = outputs.logits
        
        if logits.dim() == 2:
            return logits[:, target]
        else:
            return logits[:, target_idx, target]

    inputs_embeds = embedding_layer(input_ids.to(device)).detach()
    inputs_embeds_float = inputs_embeds.float()
    inputs_embeds_float.requires_grad = True
    
    baseline = torch.zeros_like(inputs_embeds_float)
    
    gs = GradientShap(forward_wrapper)
    attributions = gs.attribute(inputs_embeds_float, baselines=baseline, n_samples=50)
    return attributions.sum(dim=-1).detach().cpu()

def run_llama_morf_lerf(model, tokenizer, dataset, method, occlusion_type="random", occlusion_iters=100, num_samples=1000):
    all_logits = []
    all_confidences = []
    for example in tqdm(dataset.shuffle(seed=42).take(num_samples)):
        context = example["text"]
        input_ids = tokenizer(context, max_length=512, truncation=True, return_tensors="pt")
        pos = input_ids["input_ids"].shape[-1] - 1
        context_ids = input_ids["input_ids"][:, :pos]
        attn_mask = input_ids["attention_mask"][:, :pos]
        true_next_token = input_ids["input_ids"][0, pos].item()

        if method == "ig":
            relevance = run_ig(model, context_ids, attn_mask, true_next_token, model.model.embed_tokens)
        elif method == "gradshap":
            relevance = run_gradshap(model, context_ids, attn_mask, true_next_token, model.model.embed_tokens)

        with torch.no_grad():
            logits, confidences = run_occlusion_text(model, tokenizer, device, input_ids, true_next_token, relevance, occlusion_iters, "causal", occlusion_type)

        all_logits.append(logits)
        all_confidences.append(confidences)
        torch.cuda.empty_cache()
    
    return all_logits, all_confidences


if __name__ == "__main__":
    # Only take a manageable subset of the dataset
    data_files = [ f"20231101.en/train-0000{i}-of-00041.parquet" for i in range(7) ]
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", data_files=data_files, verification_mode=VerificationMode.NO_CHECKS) # Need to bypass the checks for split size
    model_name = "meta-llama/Llama-3.2-1B"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load Model and tokenizers
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.model.config._attn_implementation = "sdpa"
    model.to(device)

    logits, confs = run_llama_morf_lerf(model, tokenizer, dataset["train"], "ig")

    with open(f"results/captum_ig_llama_wiki_results_random.json", "w") as f:
        json.dump({'logits': logits, 'confs': confs}, f)

    logits2, confs2 = run_llama_morf_lerf(model, tokenizer, dataset["train"], "ig", occlusion_type="zero")

    with open(f"results/captum_ig_llama_wiki_results_zero.json", "w") as f:
        json.dump({'logits': logits2, 'confs': confs2}, f)

    logits3, confs3 = run_llama_morf_lerf(model, tokenizer, dataset["train"], "gradshap")

    with open(f"results/captum_gradshap_llama_wiki_results_random.json", "w") as f:
        json.dump({'logits': logits3, 'confs': confs3}, f)

    logits4, confs4 = run_llama_morf_lerf(model, tokenizer, dataset["train"], "gradshap", occlusion_type="zero")

    with open(f"results/captum_gradshap_llama_wiki_results_zero.json", "w") as f:
        json.dump({'logits': logits4, 'confs': confs4}, f)
