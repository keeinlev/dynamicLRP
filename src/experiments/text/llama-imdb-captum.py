import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
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
    attributions = ig.attribute(inputs_embeds_float, n_steps=50, internal_batch_size=10)
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
    
    n_samples = 50
    batch_size = 5
    total_attr = None
    remaining = n_samples
    
    while remaining > 0:
        current_batch = min(remaining, batch_size)
        attr_batch = gs.attribute(inputs_embeds_float, baselines=baseline, n_samples=current_batch, stdevs=0.0)
        
        if total_attr is None:
            total_attr = attr_batch * current_batch
        else:
            total_attr += attr_batch * current_batch
            
        remaining -= current_batch
        
    attributions = total_attr / n_samples
    return attributions.sum(dim=-1).detach().cpu()

def run_llama_morf_lerf(model, tokenizer, dataset, method, occlusion_type="random", occlusion_iters=100, num_samples=1000):
    all_logits = []
    all_confidences = []
    for example in (tqdm(dataset["test"].take(num_samples))):
        context = example["text"]
        label = example["label"]
        input_ids = tokenizer(context, return_tensors="pt")
        with torch.no_grad():
            output = model(input_ids["input_ids"].to(device))
        if output.logits.argmax() != label:
            continue

        if method == "ig":
            relevance = run_ig(model, input_ids["input_ids"], input_ids["attention_mask"], label, model.model.embed_tokens)
        elif method == "gradshap":
            relevance = run_gradshap(model, input_ids["input_ids"], input_ids["attention_mask"], label, model.model.embed_tokens)

        with torch.no_grad():
            logits, confidences = run_occlusion_text(model, tokenizer, device, input_ids["input_ids"], label, relevance, occlusion_iters, "class", occlusion_type)

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

    logits, confs = run_llama_morf_lerf(model, tokenizer, dataset, "gradshap")

    # with open(f"results/captum_ig_llama_imdb_results_random.json", "w") as f:
    #     json.dump({'logits': logits, 'confs': confs}, f)

    # logits2, confs2 = run_llama_morf_lerf(model, tokenizer, dataset, "ig", occlusion_type="zero")

    # with open(f"results/captum_ig_llama_imdb_results_zero.json", "w") as f:
    #     json.dump({'logits': logits2, 'confs': confs2}, f)

    logits3, confs3 = run_llama_morf_lerf(model, tokenizer, dataset, "gradshap")

    with open(f"results/captum_gradshap_llama_imdb_results_random.json", "w") as f:
        json.dump({'logits': logits3, 'confs': confs3}, f)

    logits4, confs4 = run_llama_morf_lerf(model, tokenizer, dataset, "gradshap", occlusion_type="zero")

    with open(f"results/captum_gradshap_llama_imdb_results_zero.json", "w") as f:
        json.dump({'logits': logits4, 'confs': confs4}, f)
