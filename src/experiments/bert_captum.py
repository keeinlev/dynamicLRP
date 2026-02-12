import argparse
import json
import pickle
import torch
import sys
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from roberta_squad_captum import run_captum_attribution

model_name = "MrKite/bert-large-squadv2"
device = "cuda" if torch.cuda.is_available() else "cpu"


# QA version

def run_bert_squad_benchmark(model, tokenizer, embedding_layer, dataset, run_name, method, question_first=True, model_topk=2):
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
            tokenized = tokenizer(question, context, return_tensors="pt", return_offsets_mapping=True)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"]
            offset_mapping = tokenized["offset_mapping"]
        else:
            tokenized = tokenizer(context, question, return_tensors="pt", return_offsets_mapping=True)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"]
            offset_mapping = tokenized["offset_mapping"]
        if input_ids.shape[-1] > 512:
            continue
    
        output = model(input_ids.to(device))
    
        # Tokenizer outputs [CLS]<question>[SEP]<context>[SEP]
        # We need to mask the question tokens in the start/end logits, this is standard in QA evals
        sep_token_ind = list(input_ids[0]).index(tokenizer.sep_token_id)
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

        # Get the decoded answer per the model
        model_answer = tokenizer.decode(input_ids[0][start:end + 1], skip_special_tokens=False).replace("<unk>", "x").strip()
        if model_answer == "":
            total_empty_answer_skips += 1
            continue
    
        # Convert from token inds to char inds since that's how the dataset tracks the labels
        char_start = offset_mapping[0][start][0]
        if context[char_start] == " ":
            char_start += 1
        char_end = offset_mapping[0][end][1]

        # Take the answer where the prediction gets the best # chars intersecting over the model and label ranges divided by total # chars in label
        # best_overlap_ratio is the Jaccard similarity
        best_overlap, best_answer_len, best_overlap_ratio = max([ (ans_overlap := (max(0, min(char_end, answer_end) - max(char_start, answer_start))), ans_len := (answer_end - answer_start), ans_overlap / (ans_len + len(model_answer) - ans_overlap)) for (answer_start, answer_end) in answer_char_ranges ], key =lambda x: x[2])
        total_overlap += best_overlap
        total_overlap_ratio += best_overlap_ratio
        precision_denom += len(model_answer)
        recall_denom += best_answer_len
    
        if model_exact_match := (best_overlap_ratio == 1):
            model_exact_matches += 1

        # --- CAPTUM ATTRIBUTION ---
        inputs_embeds = embedding_layer(input_ids)
        relevance = run_captum_attribution(method, model, inputs_embeds, attention_mask, start, end).masked_fill(~mask.cpu(), -float("inf"))

        lrp_max = relevance.flatten().argmax()
        lrp_top_token = tokenizer.decode([input_ids[0][lrp_max]], skip_special_tokens=True).strip().replace(chr(9601), "")

        # Do model answer-based accuracy, i.e. is the attribution aligned with the model prediction
        if start <= lrp_max <= end:
            top1_model_hits += 1

        # Do IoU with the model prediction
        # We split into two ways to compute:
        # For Strict IoU we do not count tokens IN BETWEEN start and end because the model is meant to only output 2 signals for start and end.
        # So, strict_intersect = 2, strict_union = 2 if LRP top-2 and the model prediction indices matched exactly
        # strict_intersect = 1, strict_union = 3 if LRP only 1 of the LRP top-2 was a model prediction index
        # strict_intersect = 0, strict_union = 4 if neither LRP top-2 were a model prediction index
        # We double count in the case of single word answers (start = end), and do not consider the second place attributed token if the first
        #   place token was the answer.
        # For Span IoU, we DO count tokens in between start and end
        strict_intersect = 0
        span_intersect = 0
        strict_union = model_topk
        span_union = model_topk
        found_start = False
        found_end = False
        for top_ind in relevance.flatten().topk(model_topk).indices:
            if found_start and found_end:
                break
            if start == top_ind:
                found_start = True
                strict_intersect += 1
                span_intersect += 1

            if end == top_ind:   # Crucial to keep these two as separate ifs for single-word answers
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
        total_strict_iou += strict_intersect / strict_union
        total_span_iou += span_intersect / span_union

        # LRP-Label IoU using normalized relevance and positive filtering, like what we assume happens in the AttnLRP paper
        # Also described in AttnLRP paper, we only do this for the model-label exact matches
        strict_label_intersect = None
        span_label_intersect = None
        strict_label_union = None
        span_label_union = None
        lrp_label_exact_match = None
        if model_exact_match:
            # Check for non-zero max to avoid division by zero
            max_val = relevance.flatten().max()
            if max_val == 0: max_val = 1e-9
            
            # Using relevance itself since it's already [1, seq]
            pos_inds = torch.where((2 * relevance.flatten() / max_val - 1).cpu() > 0)[0]
            strict_label_intersect = 0
            span_label_intersect = 0
            strict_label_union = 2
            span_label_union = int(end - start + 1)
            for top_ind in pos_inds:
                if found_start and found_end:
                    break
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

            if strict_label_intersect / strict_label_union > 1 or span_label_intersect / span_label_union > 1:
                print("ERROR ", strict_label_intersect, strict_label_union, span_label_intersect, span_label_union)
    
            if lrp_label_exact_match := (strict_label_intersect == strict_label_union):
                lrp_label_exact_matches += 1
            total_strict_label_intersect += strict_label_intersect
            total_strict_label_union += strict_label_union
            total_span_label_intersect += span_label_intersect
            total_span_label_union += span_label_union
            total_strict_label_iou += strict_label_intersect / strict_label_union
            total_span_label_iou += span_label_intersect / span_label_union

        # Do label-based accuracy
        if any(lrp_top_token in ans for ans in answers):
            # Is the attribution aligned with the ground truth label
            top1_label_hits += 1

        lrp_top5 = relevance.flatten().topk(k=5)
        lrp_top5_tokens = tokenizer.decode(input_ids[0][lrp_top5.indices.cpu()])
        results.append({
            "example_id": example["id"],
            "example_answers": example["answers"],
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
            precision = float(total_overlap / precision_denom)
            recall = float(total_overlap / recall_denom)
            f1 = float(2 / ((1 / precision) + (1 / recall)))
            print("Total examples: ", total_examples)
            print("Model exact matches: ", model_exact_matches)
            print("Model exact match %: ", model_exact_matches / total_examples)
            print("Model precision: ", precision)
            print("Model recall: ", recall)
            print("Model F1: ", f1)
            print("Model overlap ratio: ", float(total_overlap_ratio / total_examples))
            print("LRP Top-1 model answer hits: ", top1_model_hits)
            print("LRP Top-1 model answer hit %: ", top1_model_hits / total_examples)
            print("LRP Top-1 label answer hits: ", top1_label_hits)
            print("LRP Top-1 label answer hit %: ", top1_label_hits / total_examples)
            print(f"LRP Top-{model_topk} Strict Accuracy: ", total_strict_intersect / (model_topk * total_examples))
            print(f"LRP Top-{model_topk} Span Accuracy: ", total_span_intersect / (model_topk * total_examples))
            print("LRP-Model exact matches: ", lrp_model_exact_matches)
            print("LRP-Model Strict IoU: ", total_strict_intersect / total_strict_union)
            print("LRP-Model Span IoU: ", total_span_intersect / total_span_union)
            print("LRP-Model Avg Strict IoU: ", total_strict_iou / total_examples)
            print("LRP-Model Avg Span IoU: ", total_span_iou / total_examples)
            print("LRP-Label exact matches: ", lrp_label_exact_matches)
            print("LRP-Label Strict IoU: ", total_strict_label_intersect / total_strict_label_union)
            print("LRP-Label Span IoU: ", total_span_label_intersect / total_span_label_union)
            print("LRP-Label Avg Strict IoU: ", total_strict_label_iou / model_exact_matches)
            print("LRP-Label Avg Span IoU: ", total_span_label_iou / model_exact_matches)
            print("Total start-end skips: ", total_start_end_skips)
            print("Total empty model answer skips: ", total_empty_answer_skips)
            print("Total unanswerable: ", total_unanswerable)

    precision = float(total_overlap / precision_denom)
    recall = float(total_overlap / recall_denom)
    f1 = float(2 / ((1 / precision) + (1 / recall)))
    data = {
        "summary": {
            "total_examples": total_examples,
            "model_exact_matches": model_exact_matches,
            "model_exact_match_pct": model_exact_matches / total_examples,
            "model_precision": precision,
            "model_recall": recall,
            "model_f1": f1,
            "model_overlap_ratio": float(total_overlap_ratio / total_examples),
            "lrp_top1_model_answer_hits": top1_model_hits,
            "lrp_top1_model_answer_hit_pct": top1_model_hits / total_examples,
            "lrp_top1_label_answer_hits": top1_label_hits,
            "lrp_top1_label_answer_hit_pct": top1_label_hits / total_examples,
            f"lrp_top{model_topk}_strict_accuracy": total_strict_intersect / (model_topk * total_examples),
            f"lrp_top{model_topk}_span_accuracy": total_span_intersect / (model_topk * total_examples),
            "lrp_model_exact_matches": lrp_model_exact_matches,
            "lrp_model_strict_iou": total_strict_intersect / total_strict_union,
            "lrp_model_span_iou": total_span_intersect / total_span_union,
            "lrp_model_avg_strict_iou": total_strict_iou / total_examples,
            "lrp_model_avg_span_iou": total_span_iou / total_examples,
            "lrp_label_exact_matches": lrp_label_exact_matches,
            "lrp_label_strict_iou": total_strict_label_intersect / total_strict_label_union,
            "lrp_label_span_iou": total_span_label_intersect / total_span_label_union,
            "lrp_label_avg_strict_iou": total_strict_label_iou / model_exact_matches,
            "lrp_label_avg_span_iou": total_span_label_iou / model_exact_matches,
            "total_start_end_skips" : total_start_end_skips,
            "total_empty_answer_skips" : total_empty_answer_skips,
            "total_unanswerable" : total_unanswerable,
        },
        "results": results
    }

    try:
        with open(f"bert_squadv2_results_{run_name}.json", "w") as f:
            json.dump(data, f, indent=4)
    except TypeError:
        print("Failed JSON write, attempting pickle...")
        try:
            with open(f"bert_squadv2_results_{run_name}.pkl", "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print("Encountered exception while pickling, will return (data, exception)...")
            return data, e
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["ig", "gradshap", "input_x_gradient", "kernelshap"], required=True)
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to run from dataset (for debugging)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device=device, dtype=torch.bfloat16)
    print(model)
    embedding_layer = model.bert.embeddings.word_embeddings
    dataset = load_dataset("squad_v2")
    
    target_dataset = dataset
    if args.num_samples:
        target_dataset = {"validation": dataset["validation"].select(range(args.num_samples))}

    run_name = f"captum_{args.method}"
    run_bert_squad_benchmark(model, tokenizer, embedding_layer, target_dataset, run_name, args.method)
