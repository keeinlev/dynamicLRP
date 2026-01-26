import math
import torch
from tqdm import tqdm

def run_occlusion_text(model, tokenizer, device, input_ids, label_ind, relevance, occlusion_iters, mode=None):

    if mode is None:
        raise ValueError("must give kwarg `mode` = (one of 'class' or 'causal')")

    elif mode not in ["class", "causal"]:
        raise ValueError(" kwarg `mode` must be one of 'class' or 'causal'")
    logits = []

    morf_indices = relevance.flatten().sort(descending=True).indices.cpu()
    lerf_indices = relevance.flatten().sort(descending=False).indices.cpu()
    for occlusion_iter in tqdm(range(1, occlusion_iters + 1), leave=False):
        occlusion_percent = occlusion_iter / occlusion_iters

        toks_to_occlude = math.floor(occlusion_percent * input_ids.shape[-1])
        mask_inds_morf = morf_indices[:toks_to_occlude]
        mask_inds_lerf = lerf_indices[:toks_to_occlude]
        input_ids_morf = input_ids.clone()
        input_ids_lerf = input_ids.clone()
        random_tokens = torch.randint(100, tokenizer.vocab_size - 100, (toks_to_occlude,), device=input_ids.device)
        input_ids_morf[0][mask_inds_morf] = random_tokens
        input_ids_lerf[0][mask_inds_lerf] = random_tokens

        output_morf = model(input_ids_morf.to(device))
        output_lerf = model(input_ids_lerf.to(device))

        if mode == "class":
            morf_logit = output_morf.logits.detach().cpu()[0, label_ind]
            lerf_logit = output_lerf.logits.detach().cpu()[0, label_ind]
        elif mode == "causal":
            morf_logit = output_morf.logits.detach().cpu()[0, -1, label_ind]
            lerf_logit = output_lerf.logits.detach().cpu()[0, -1, label_ind]

        logits.append([float(lerf_logit), float(morf_logit)])

    return logits
