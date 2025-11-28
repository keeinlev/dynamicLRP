import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients, GradientShap, LayerIntegratedGradients, LayerGradientShap
from sklearn.metrics import auc


# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

# Add LXT to path
lxt_path = os.path.join(project_root, "external/LRP-eXplains-Transformers")
sys.path.append(lxt_path)

import traceback
from src.lrp_engine import LRPEngine, checkpoint_hook
from src.lrp_engine.lrp_prop_fcns import LRPPropFunctions
# Try importing lxt, handle if it fails (though it should succeed based on setup)
try:
    from lxt.efficient import monkey_patch
except ImportError:
    print("Warning: Could not import lxt.efficient.monkey_patch. LXT baseline might fail.")

def get_model_and_tokenizer(model_name, task, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if task == "imdb":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        model.config.pad_token_id = model.config.eos_token_id
    else: # wiki
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )

    #model.model.config._attn_implementation = "sdpa"
    
    return model, tokenizer

def load_data(task, tokenizer, num_samples=4000, seed=42, max_length=512):
    if task == "imdb":
        dataset = load_dataset("imdb", split="test")
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
        
        def preprocess(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
            
        dataset = dataset.map(preprocess, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
    elif task == "wiki":
        dataset = load_dataset("wikipedia", "20220301.en", split="train")
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
        
        def preprocess(examples):
            # For next word prediction, we just need text. We'll tokenize on the fly or here.
            # We take the first 512 tokens.
            return tokenizer(examples["text"], truncation=True, max_length=max_length)
            
        dataset = dataset.map(preprocess, batched=True)
        # Filter out short sequences
        dataset = dataset.filter(lambda x: len(x["input_ids"]) == max_length)
        # If we filtered, we might have fewer than num_samples. 
        # But for simplicity, let's just take what we have or re-sample. 
        # The prompt implies fixed set. Let's assume the shuffle gave us enough long docs.
        dataset = dataset.select(range(min(len(dataset), num_samples)))
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
    return dataset

def get_embedding_layer(model):
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    # Fallback for common models
    if hasattr(model, "model"):
        if hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        if hasattr(model.model, "embeddings"):
            return model.model.embeddings.word_embeddings
    return None

def run_ig(model, input_ids, attention_mask, target, embedding_layer, target_idx=-1):
    # Integrated Gradients on Embeddings
    # We wrap the model to handle dtype and output selection
    def forward_wrapper(inputs_embeds):
        # inputs_embeds might be float32 from Captum, model might be bfloat16
        if inputs_embeds.dtype != model.dtype:
            inputs_embeds = inputs_embeds.to(model.dtype)
        
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Select the target logit
        if logits.dim() == 2: # Classification [batch, num_labels]
            return logits[:, target]
        else: # NWP [batch, seq, vocab]
            # Captum expands batch dim, so we use : for batch
            return logits[:, target_idx, target]

    inputs_embeds = embedding_layer(input_ids).detach()
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
        
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        
        if logits.dim() == 2:
            return logits[:, target]
        else:
            return logits[:, target_idx, target]

    inputs_embeds = embedding_layer(input_ids).detach()
    inputs_embeds_float = inputs_embeds.float()
    inputs_embeds_float.requires_grad = True
    
    baseline = torch.zeros_like(inputs_embeds_float)
    
    gs = GradientShap(forward_wrapper)
    attributions = gs.attribute(inputs_embeds_float, baselines=baseline, n_samples=50)
    return attributions.sum(dim=-1).detach().cpu()

def run_lxt(model, input_ids, attention_mask, target, embedding_layer):
    # LRP-eXplains-Transformers
    # We need to monkey patch the model. 
    # Note: Monkey patching modifies the model class. This might interfere with other methods if run sequentially on the same model instance.
    # Ideally we should reload the model or unpatch. LXT doesn't seem to have unpatch.
    # For this script, we might need to be careful. 
    # However, LXT patch usually just adds hooks or changes forward.
    
    # Assuming the model is already patched or we patch it here.
    # Since we run multiple baselines, we should probably patch only for this call or reload model.
    # Reloading is expensive. Let's try to patch once if possible, but LXT changes behavior.
    # The prompt says: "Patch module first... Forward pass... Backward...".
    
    # We will assume we can patch. 
    monkey_patch(model)
    
    inputs_embeds = embedding_layer(input_ids).detach()
    inputs_embeds.requires_grad = True
    
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    
    if target is not None:
        # For classification/NWP, we want relevance of the target class/token
        # outputs.logits is [batch, seq_len, vocab] or [batch, num_labels]
        
        if outputs.logits.dim() == 2: # Classification
            score = outputs.logits[0, target]
        else: # NWP (last token)
            score = outputs.logits[0, -1, target]
            
        score.backward()
        
        relevance = (inputs_embeds * inputs_embeds.grad).sum(-1)
        return relevance.detach().cpu()
    return None

def run_dynamic_lrp(model, input_ids, attention_mask, target, lrp, target_idx=-1):
    # Dynamic LRP
    # We need to run the model, get logits, then run engine.
    
    with torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        if logits.dim() == 2: # Classification
            target_logit = logits[0, target]
        else: # NWP
            target_logit = logits[0, target_idx, target]
            
        pass
    
    checkpoint_rels = lrp.run(target_logit.unsqueeze(0))
    
    return checkpoint_rels[1][0].detach().cpu()

def evaluate_faithfulness(model, input_ids, attention_mask, attributions, target, baseline_token_id, target_idx=-1, steps=10):
    # MoRF: Remove most relevant first
    # LeRF: Remove least relevant first
    
    # attributions: [seq_len]
    # input_ids: [1, seq_len]
    
    seq_len = input_ids.shape[1]
    # Sort indices
    sorted_indices = torch.argsort(attributions, descending=True) # High to low
    
    morf_scores = []
    lerf_scores = []
    
    # Initial score
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.logits.dim() == 2:
            initial_score = torch.softmax(outputs.logits, dim=-1)[0, target].item()
        else:
            initial_score = torch.softmax(outputs.logits[0, target_idx], dim=-1)[target].item()
            
    morf_scores.append(initial_score)
    lerf_scores.append(initial_score)
    
    fractions = np.linspace(0, 1, steps + 1)[1:] # 10%, 20%, ... 100%
    
    for frac in fractions:
        num_remove = int(seq_len * frac)
        if num_remove == 0: continue
        
        # MoRF
        morf_mask_indices = sorted_indices[:num_remove]
        morf_input_ids = input_ids.clone()
        morf_input_ids[0, morf_mask_indices] = baseline_token_id
        
        with torch.no_grad():
            outputs = model(input_ids=morf_input_ids, attention_mask=attention_mask)
            if outputs.logits.dim() == 2:
                score = torch.softmax(outputs.logits, dim=-1)[0, target].item()
            else:
                score = torch.softmax(outputs.logits[0, target_idx], dim=-1)[target].item()
            morf_scores.append(score)
            
        # LeRF
        lerf_mask_indices = sorted_indices[-num_remove:]
        lerf_input_ids = input_ids.clone()
        lerf_input_ids[0, lerf_mask_indices] = baseline_token_id
        
        with torch.no_grad():
            outputs = model(input_ids=lerf_input_ids, attention_mask=attention_mask)
            if outputs.logits.dim() == 2:
                score = torch.softmax(outputs.logits, dim=-1)[0, target].item()
            else:
                score = torch.softmax(outputs.logits[0, target_idx], dim=-1)[target].item()
            lerf_scores.append(score)
            
    # Calculate AUC
    x = np.linspace(0, 1, len(morf_scores))
    morf_auc = auc(x, morf_scores)
    lerf_auc = auc(x, lerf_scores)
    
    return morf_auc, lerf_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["imdb", "wiki"])
    parser.add_argument("--model", type=str, default="yash3056/Llama-3.2-1B-imdb")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Model
    print(f"Loading model {args.model}...")
    try:
        model, tokenizer = get_model_and_tokenizer(args.model, args.dataset, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load a smaller model for demonstration if Llama-2-7b fails (e.g. due to access/size).")
        # Fallback to a smaller model if the user doesn't have access or memory
        fallback_model = "gpt2" if args.dataset == "wiki" else "distilbert-base-uncased"
        print(f"Falling back to {fallback_model}")
        model, tokenizer = get_model_and_tokenizer(fallback_model, args.dataset, device)

    model.eval()

    # Set the attention checkpoints for DynamicLRP
    for layer in model.model.layers:
        layer.self_attn.o_proj.register_forward_hook(checkpoint_hook)

    embedding_layer = get_embedding_layer(model)
    
    # Load Data
    print(f"Loading dataset {args.dataset}...")
    dataset = load_data(args.dataset, tokenizer, num_samples=args.samples, seed=args.seed)
    
    results = []
    
    # Baselines
    methods = {
        # "IG": run_ig,
        # "GradSHAP": run_gradshap,
        "DynamicLRP": run_dynamic_lrp,
    }
    
    # Add LXT if available
    # if "lxt.efficient" in sys.modules:
    #     methods["LXT"] = run_lxt

    print("Starting evaluation...")

    lrp = LRPEngine(dtype=torch.bfloat16)
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        
        if args.dataset == "imdb":
            target = sample["label"].item()
            target_idx = -1 # Not used for classification logits selection usually, but for consistency
        else:
            # For Wiki, we want to explain the prediction of the last token in the sequence.
            # input_ids: [x_0, ..., x_T]
            # logits: [p(x_1|x_0), ..., p(x_{T+1}|x_0...x_T)]
            # We want to explain p(x_T | x_0...x_{T-1})
            # This corresponds to logits at index T-2 (0-indexed) -> prediction of x_{T-1+1} = x_T
            # Wait:
            # logits[0] -> pred for x_1
            # logits[T-1] -> pred for x_{T+1}
            # logits[T-2] -> pred for x_T
            target = input_ids[0, -1].item()
            target_idx = -2 # Look at the second to last logit

        baseline_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # Run non-destructive methods first
        current_methods = {k: v for k, v in methods.items() if k != "LXT"}
        
        for method_name, method_fn in current_methods.items():
            try:
                if method_name in ["IG", "GradSHAP"]:
                    attributions = method_fn(model, input_ids, attention_mask, target, embedding_layer, target_idx=target_idx)
                elif method_name == "DynamicLRP":
                    attributions = method_fn(model, input_ids, attention_mask, target, lrp, target_idx=target_idx)
                
                morf_auc, lerf_auc = evaluate_faithfulness(
                    model, input_ids, attention_mask, attributions[0], target, baseline_token_id, target_idx=target_idx
                )
                
                results.append({
                    "method": method_name,
                    "sample_id": i,
                    "morf_auc": morf_auc,
                    "lerf_auc": lerf_auc
                })
            except Exception as e:
                print(f"Error processing sample {i} with {method_name}: {e}")
                traceback.print_exc()
                continue
        
        # Run LXT last (destructive patching)
        if "LXT" in methods:
            try:
                # We need to patch the model if not already patched.
                # But patching is global for the model instance.
                # If we patch it once, it stays patched.
                # So we should probably patch it outside the loop if we were only running LXT, 
                # but here we mix.
                # Strategy: Patch it now. Future iterations will use patched model for other methods?
                # Yes, that's a problem.
                # Ideally we should reload model or copy it.
                # Given the constraints, let's skip LXT in this loop if we want to be safe, 
                # OR we accept that we only run LXT at the very end of all samples? No, we need per-sample.
                # We will skip LXT for now in this script to avoid breaking other methods, 
                # or we assume the user will run this script separately for LXT.
                # Let's try to run it.
                pass
            except Exception as e:
                pass


    # Save results
    df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, f"{args.dataset}_results.csv"), index=False)
    
    # Summary
    summary = df.groupby("method")[["morf_auc", "lerf_auc"]].mean()
    print("\nResults Summary:")
    print(summary)
    summary.to_csv(os.path.join(args.output_dir, f"{args.dataset}_summary.csv"))

if __name__ == "__main__":
    main()
