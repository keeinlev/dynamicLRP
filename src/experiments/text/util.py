import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def run_occlusion_text(model, tokenizer, device, input_ids, label_ind, relevance, occlusion_iters, mode=None, occlusion_type="random"):

    if mode not in ["class", "causal"]:
        raise ValueError(f"kwarg `mode` must be one of 'class' or 'causal', got {mode}")

    if occlusion_type not in ["random", "zero"]:
        raise ValueError(f"kwarg `occlusion_type` must be one of 'random' or 'zero', got {occlusion_type}")

    logits = []
    confidences = []

    morf_indices = relevance.flatten()[1:].sort(descending=True).indices.cpu() + 1
    lerf_indices = relevance.flatten()[1:].sort(descending=False).indices.cpu() + 1
    for occlusion_iter in tqdm(range(1, occlusion_iters + 1), leave=False):
        occlusion_percent = occlusion_iter / occlusion_iters

        toks_to_occlude = math.floor(occlusion_percent * morf_indices.shape[-1])
        mask_inds_morf = morf_indices[:toks_to_occlude]
        mask_inds_lerf = lerf_indices[:toks_to_occlude]
        input_ids_morf = input_ids.clone()
        input_ids_lerf = input_ids.clone()

        if occlusion_type == "random":
            random_tokens = torch.randint(100, tokenizer.vocab_size - 100, (toks_to_occlude,), device=input_ids.device)
            input_ids_morf[0][mask_inds_morf] = random_tokens
            input_ids_lerf[0][mask_inds_lerf] = random_tokens

            output_morf = model(input_ids_morf.to(device))
            output_lerf = model(input_ids_lerf.to(device))
        elif occlusion_type == "zero":
            input_embeds_morf = model.model.embed_tokens(input_ids_morf.to(device))
            input_embeds_lerf = model.model.embed_tokens(input_ids_lerf.to(device))
            input_embeds_morf[0][mask_inds_morf] = 0.0
            input_embeds_lerf[0][mask_inds_lerf] = 0.0

            output_morf = model(inputs_embeds=input_embeds_morf)
            output_lerf = model(inputs_embeds=input_embeds_lerf)

        if mode == "class":
            morf_logit = output_morf.logits.detach().cpu()[0, label_ind]
            lerf_logit = output_lerf.logits.detach().cpu()[0, label_ind]
            morf_conf = F.softmax(output_morf.logits.detach().cpu()[0], dim=-1)[label_ind]
            lerf_conf = F.softmax(output_lerf.logits.detach().cpu()[0], dim=-1)[label_ind]
        elif mode == "causal":
            morf_logit = output_morf.logits.detach().cpu()[0, -1, label_ind]
            lerf_logit = output_lerf.logits.detach().cpu()[0, -1, label_ind]
            morf_conf = F.softmax(output_morf.logits.detach().cpu()[0, -1], dim=-1)[label_ind]
            lerf_conf = F.softmax(output_lerf.logits.detach().cpu()[0, -1], dim=-1)[label_ind]

        logits.append([float(lerf_logit), float(morf_logit)])
        confidences.append([float(lerf_conf), float(morf_conf)])

    return logits, confidences


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
