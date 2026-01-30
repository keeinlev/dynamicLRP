import argparse
import json
import pickle
import torch
import sys
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from captum.attr import IntegratedGradients, GradientShap, InputXGradient, KernelShap

# Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_and_tokenizer():
    model_name = "deepset/roberta-large-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device=device, dtype=torch.bfloat16)
    # RoBERTa's embedding layer
    return model, tokenizer, model.roberta.embeddings.word_embeddings

# --- Captum Wrapper Functions to handle Memory ---

def forward_func(inputs_embeds, model, attention_mask, start_ind, end_ind):
    # wrapper to compute sum of logit(start) and logit(end)
    # inputs_embeds: [batch, seq, dim]
    
    # Ensure dtype match
    if inputs_embeds.dtype != model.dtype:
        inputs_embeds = inputs_embeds.to(model.dtype)
        
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    
    batch_size = inputs_embeds.shape[0]
    
    # outputs.start_logits: [batch, seq_len]
    start_logits = outputs.start_logits[torch.arange(batch_size), start_ind]
    end_logits = outputs.end_logits[torch.arange(batch_size), end_ind]
    
    return start_logits + end_logits

def run_captum_attribution(method, model, inputs_embeds, attention_mask, start_ind, end_ind):
    inputs_embeds_float = inputs_embeds.detach().float()
    inputs_embeds_float.requires_grad = True
    
    def wrapper(inputs):
        return forward_func(inputs, model, attention_mask, start_ind, end_ind)

    baseline = torch.zeros_like(inputs_embeds_float)

    if method == "ig":
        ign = IntegratedGradients(wrapper)
        attributions = ign.attribute(inputs_embeds_float, baselines=baseline, n_steps=50, internal_batch_size=5)
        
    elif method == "input_x_gradient":
        ixg = InputXGradient(wrapper)
        attributions = ixg.attribute(inputs_embeds_float)
        
    elif method == "gradshap":
        gs = GradientShap(wrapper)
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
            torch.cuda.empty_cache()
            
        attributions = total_attr / n_samples
    
    elif method == "kernelshap":
        ks = KernelShap(wrapper)
        seq_len = inputs_embeds.shape[1]
        feature_mask = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(-1).expand_as(inputs_embeds_float)
        attributions = ks.attribute(inputs_embeds_float, baselines=baseline, feature_mask=feature_mask, n_samples=100)
    
    else:
        raise ValueError(f"Unknown method: {method}")

    return attributions.sum(dim=-1).detach().cpu()


def run_roberta_squad_benchmark(model, tokenizer, embedding_layer, dataset, run_name, method, question_first=True, model_topk=2):
    results = []
    top1_label_hits = 0
    top1_model_hits = 0
    total_examples = 0
    total_strict_intersect = 0
    total_span_intersect = 0
    total_strict_union = 0
    total_span_union = 0
    total_strict_iou = 0
    total_span_iou = 0
    model_exact_matches = 0
    lrp_model_exact_matches = 0
    precision_denom = 0
    recall_denom = 0
    total_overlap = 0
    total_overlap_ratio = 0
    total_start_end_skips = 0
    total_empty_answer_skips = 0
    total_unanswerable = 0
    lrp_label_exact_matches = 0
    total_strict_label_intersect = 0
    total_strict_label_union = 0
    total_span_label_intersect = 0
    total_span_label_union = 0
    total_span_label_union = 0
    total_strict_label_iou = 0
    total_span_label_iou = 0
    
    print(f"Starting Benchmark: {run_name} using {method}")

    for example in tqdm(dataset["validation"]):
        question = example["question"]
        context = example["context"]
        answers = example["answers"]["text"]
        answer_start_inds = example["answers"]["answer_start"]
    
        if not answers:
            total_unanswerable += 1
            continue
    
        answer_char_ranges = [ (i, i + len(a)) for (a, i) in zip(answers, answer_start_inds) ]
    
        if question_first:
            # RoBERTa notebook logic: direct tokenizer call without prepending cls_token manually
            tokenized = tokenizer(question, context, return_tensors="pt", return_offsets_mapping=True)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            offset_mapping = tokenized["offset_mapping"]
        else:
            tokenized = tokenizer(context, question, return_tensors="pt", return_offsets_mapping=True)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            offset_mapping = tokenized["offset_mapping"]
        if input_ids.shape[-1] > 512:
            continue
    
        with torch.no_grad():
            output = model(input_ids.to(device))
    
        sep_token_ind = list(input_ids[0]).index(tokenizer.eos_token_id)
        mask = torch.zeros_like(output.start_logits).bool()
        if question_first:
            mask_slice = [slice(None, None), slice(sep_token_ind + 1, None)]
        else:
            mask_slice = [slice(None, None), slice(None, sep_token_ind + 1)]
        mask[mask_slice] = True
    
        masked_start_logits = output.start_logits.masked_fill(~mask, -float("inf"))
        masked_end_logits = output.end_logits.masked_fill(~mask, -float("inf"))
        start = torch.argmax(masked_start_logits)
        end = torch.argmax(masked_end_logits)
    
        if start > end:
            total_start_end_skips += 1
            continue

        model_answer = tokenizer.decode(input_ids[0][start:end + 1], skip_special_tokens=False).replace("<unk>", "x").strip()
        if model_answer == "":
            total_empty_answer_skips += 1
            continue
    
        char_start = offset_mapping[0][start][0]
        if context[char_start] == " ":
            char_start += 1
        char_end = offset_mapping[0][end][1]

        best_overlap, best_answer_len, best_overlap_ratio = max([ (ans_overlap := (max(0, min(char_end, answer_end) - max(char_start, answer_start))), ans_len := (answer_end - answer_start), ans_overlap / (ans_len + len(model_answer) - ans_overlap)) for (answer_start, answer_end) in answer_char_ranges ], key =lambda x: x[2])
        total_overlap += best_overlap
        total_overlap_ratio += best_overlap_ratio
        precision_denom += len(model_answer)
        recall_denom += best_answer_len
    
        if model_exact_match := (best_overlap_ratio == 1):
            model_exact_matches += 1

        # --- CAPTUM ATTRIBUTION ---
        inputs_embeds = embedding_layer(input_ids)
        relevance = run_captum_attribution(method, model, inputs_embeds, attention_mask, start, end)
        param_vals_2 = relevance

        lrp_max = param_vals_2.masked_fill(~mask.cpu(), -float("inf")).flatten().argmax()
        lrp_top_token = tokenizer.decode([input_ids[0][lrp_max]], skip_special_tokens=True).strip().replace(chr(9601), "")

        if start <= lrp_max <= end:
            top1_model_hits += 1

        strict_intersect = 0
        span_intersect = 0
        strict_union = model_topk
        span_union = model_topk
        found_start = False
        found_end = False
        for top_ind in param_vals_2.flatten().topk(model_topk).indices:
            if found_start and found_end:
                break
            if start == top_ind:
                found_start = True
                strict_intersect += 1
                span_intersect += 1

            if end == top_ind:   
                found_end = True
                strict_intersect += 1
                span_intersect += 1

            if start < top_ind < end:
                strict_union += 1
                span_intersect += 1

            elif top_ind < start or end < top_ind:
                strict_union += 1
                span_union += 1

        if lrp_model_exact_match := (strict_intersect == strict_union):
            lrp_model_exact_matches += 1
        total_strict_intersect += strict_intersect
        total_strict_union += strict_union
        total_span_intersect += span_intersect
        total_span_union += span_union

        strict_label_intersect = None
        span_label_intersect = None
        strict_label_union = None
        span_label_union = None
        lrp_label_exact_match = None
        if model_exact_match:
            flat_attr = param_vals_2.flatten()
            max_val = flat_attr.max()
            if max_val == 0: max_val = 1e-9
            pos_inds = torch.where((2 * flat_attr / max_val - 1) > 0)[0]
            
            strict_label_intersect = 0
            span_label_intersect = 0
            strict_label_union = 2
            span_label_union = int(end - start + 1)
            for top_ind in pos_inds:
                if start == top_ind or end == top_ind:
                    span_label_intersect += 1
                    strict_label_intersect += 1
                    if start == top_ind and end == top_ind:
                        strict_label_intersect += 1
    
                if start < top_ind < end:
                    strict_label_union += 1
                    span_label_intersect += 1

                elif top_ind < start or end < top_ind:
                    strict_label_union += 1
                    span_label_union += 1
                    
            if lrp_label_exact_match := (strict_label_intersect == strict_label_union):
                lrp_label_exact_matches += 1
            total_strict_label_intersect += strict_label_intersect
            total_strict_label_union += strict_label_union
            total_span_label_intersect += span_label_intersect
            total_span_label_union += span_label_union
            
            if strict_label_union > 0:
                total_strict_label_iou += strict_label_intersect / strict_label_union
            if span_label_union > 0:
                total_span_label_iou += span_label_intersect / span_label_union

        if any(lrp_top_token in ans for ans in answers):
            top1_label_hits += 1

        lrp_top5 = param_vals_2.flatten().topk(k=5)
        lrp_top5_tokens = tokenizer.decode(input_ids[0][lrp_top5.indices.cpu()])
        results.append({
            "example": example,
            "model_exact_match": bool(model_exact_match),
            "lrp_top1_ind": lrp_max.tolist(),
            "model_start_end": (start.detach().cpu().tolist(), end.detach().cpu().tolist()),
            "lrp_top1_token": lrp_top_token,
            "lrp_top5_tokens": lrp_top5_tokens,
            "lrp_top5_relevances": lrp_top5.values.detach().cpu().tolist(),
            "lrp_model_strict_intersect": strict_intersect,
            "lrp_model_strict_union": strict_union,
            "lrp_model_span_intersect": span_intersect,
            "lrp_model_span_union": span_union,
            "lrp_model_exact_match": lrp_model_exact_match,
            "lrp_label_strict_intersect": strict_label_intersect,
            "lrp_label_strict_union": strict_label_union,
            "lrp_label_span_intersect": span_label_intersect,
            "lrp_label_span_union": span_label_union,
            "lrp_label_exact_match": lrp_label_exact_match,
        })

        total_examples += 1
        if not (total_examples % 100):
            precision = float(total_overlap / precision_denom) if precision_denom > 0 else 0
            recall = float(total_overlap / recall_denom) if recall_denom > 0 else 0
            f1 = float(2 / ((1 / precision) + (1 / recall))) if (precision > 0 and recall > 0) else 0
            print(f"Stats @ {total_examples}: F1={f1:.4f}, ModelEM={model_exact_matches/total_examples:.4f}, AttrTop1Model={top1_model_hits/total_examples:.4f}")
            sys.stdout.flush()

    precision = float(total_overlap / precision_denom) if precision_denom > 0 else 0
    recall = float(total_overlap / recall_denom) if recall_denom > 0 else 0
    f1 = float(2 / ((1 / precision) + (1 / recall))) if (precision > 0 and recall > 0) else 0
    data = {
        "summary": {
            "total_examples": total_examples,
            "model_exact_matches": model_exact_matches,
            "model_exact_match_pct": model_exact_matches / total_examples if total_examples > 0 else 0,
            "model_precision": precision,
            "model_recall": recall,
            "model_f1": f1,
            "model_overlap_ratio": float(total_overlap_ratio / total_examples) if total_examples > 0 else 0,
            "lrp_top1_model_answer_hits": top1_model_hits,
            "lrp_top1_model_answer_hit_pct": top1_model_hits / total_examples if total_examples > 0 else 0,
            "lrp_top1_label_answer_hits": top1_label_hits,
            "lrp_top1_label_answer_hit_pct": top1_label_hits / total_examples if total_examples > 0 else 0,
            f"lrp_top{model_topk}_strict_accuracy": total_strict_intersect / (model_topk * total_examples) if total_examples > 0 else 0,
            f"lrp_top{model_topk}_span_accuracy": total_span_intersect / (model_topk * total_examples) if total_examples > 0 else 0,
            "lrp_model_exact_matches": lrp_model_exact_matches,
            "lrp_model_strict_iou": total_strict_intersect / total_strict_union if total_strict_union > 0 else 0,
            "lrp_model_span_iou": total_span_intersect / total_span_union if total_span_union > 0 else 0,
            "lrp_model_avg_strict_iou": total_strict_iou / total_examples if total_examples > 0 else 0,
            "lrp_model_avg_span_iou": total_span_iou / total_examples if total_examples > 0 else 0,
            "lrp_label_exact_matches": lrp_label_exact_matches,
            "lrp_label_strict_iou": total_strict_label_intersect / total_strict_label_union if total_strict_label_union and total_strict_label_union > 0 else 0,
            "lrp_label_span_iou": total_span_label_intersect / total_span_label_union if total_span_label_union and total_span_label_union > 0 else 0,
            "lrp_label_avg_strict_iou": total_strict_label_iou / model_exact_matches if model_exact_matches > 0 else 0,
            "lrp_label_avg_span_iou": total_span_label_iou / model_exact_matches if model_exact_matches > 0 else 0,
            "total_start_end_skips" : total_start_end_skips,
            "total_empty_answer_skips" : total_empty_answer_skips,
            "total_unanswerable" : total_unanswerable,
        },
        "results": results
    }

    try:
        with open(f"roberta_squadv2_results_{run_name}.json", "w") as f:
            json.dump(data, f, indent=4)
        print(f"Results saved to roberta_squadv2_results_{run_name}.json")
    except TypeError:
        print("Failed JSON write, attempting pickle...")
        try:
            with open(f"roberta_squadv2_results_{run_name}.pkl", "wb") as f:
                pickle.dump(data, f)
            print(f"Results saved to roberta_squadv2_results_{run_name}.pkl")
        except Exception as e:
            print("Encountered exception while pickling, will return (data, exception)...")
            return data, e
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["ig", "gradshap", "input_x_gradient", "kernelshap"], required=True)
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to run from dataset (for debugging)")
    args = parser.parse_args()

    model, tokenizer, embedding_layer = get_model_and_tokenizer()
    dataset = load_dataset("squad_v2")
    
    target_dataset = dataset
    if args.num_samples:
        target_dataset = {"validation": dataset["validation"].select(range(args.num_samples))}

    run_name = f"captum_{args.method}"
    run_roberta_squad_benchmark(model, tokenizer, embedding_layer, target_dataset, run_name, args.method)
