import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_attention_layer_relevance(relevance_values, layer_ind=0):
    # Visualizing relevance of attention mechanism

    # Use pooling with kernel size and stride set to d_head = d_model / num_heads = 64 (for DNABERT2)
    max_pool = torch.nn.MaxPool1d(kernel_size=64, stride=64)
    min_pool = lambda x: -max_pool(-x)
    avg_pool = torch.nn.AvgPool1d(kernel_size=64, stride=64)
    sum_pool = torch.nn.LPPool1d(kernel_size=64, stride=64, norm_type=1)

    pools = {
        "Max Pooling": max_pool,
        "Min Pooling": min_pool,
        "Average Pooling": avg_pool,
        "Sum Pooling": sum_pool,
    }

    fig, axs = plt.subplots((len(pools) + 1) // 2, 2, figsize=(10,8))
    fig.subplots_adjust(top=1.0)

    for i, (pool_name, pool_layer) in enumerate(list(pools.items())):
        # Make a heatmap for each type of pooling
        checkpoint_output = pool_layer(relevance_values[layer_ind])
        lrp_shape = checkpoint_output.shape
        raw_heatmap = checkpoint_output.view((lrp_shape[-2], lrp_shape[-1])).nan_to_num(0.0)
        normed = raw_heatmap.sign() * torch.log1p(raw_heatmap.abs())
        normed /= torch.max(normed.abs())
        sns.heatmap(normed.cpu(), ax=axs[i // 2, i % 2])
        axs[i // 2, i % 2].set_title(pool_name)
        axs[i // 2, i % 2].set_xlabel("head index")
        axs[i // 2, i % 2].set_ylabel("token index")

    # Remove overflow subplot
    if i < ((len(pools) + 1) // 2) * 2 - 1:
        axs[i // 2, 1].set_axis_off()
    fig.suptitle(f"Post-Attention Relevance at layer {layer_ind}, pooled by head, using signed-log scaling.")
    fig.tight_layout()

    plt.show()


# AI-generated, needed quick tooling
# Basically adding two matrices together if dim 0 is variable, but dim 1 is constant.
def accumulate_variable_matrix(accumulator, new_matrix):
    # Get current shapes
    acc_rows, acc_cols = accumulator.shape
    new_rows, new_cols = new_matrix.shape

    # Determine target size (assume row counts always match, or adapt as needed)
    target_rows = max(acc_rows, new_rows)
    target_cols = max(acc_cols, new_cols)

    # Expand accumulator if needed
    if acc_rows < target_rows:
        pad_rows = target_rows - acc_rows
        accumulator = torch.cat([accumulator, torch.zeros(pad_rows, acc_cols, device=accumulator.device, dtype=accumulator.dtype)], dim=0)

    # Expand new_matrix if needed
    if new_rows < target_rows:
        pad_rows = target_rows - new_rows
        new_matrix = torch.cat([new_matrix, torch.zeros(pad_rows, new_cols, device=accumulator.device, dtype=accumulator.dtype)], dim=0)

    # Expand accumulator if needed
    if acc_cols < target_cols:
        pad_cols = target_cols - acc_cols
        accumulator = torch.cat([accumulator, torch.zeros(target_rows, pad_cols, device=accumulator.device, dtype=accumulator.dtype)], dim=1)

    # Expand new_matrix if needed
    if new_cols < target_cols:
        pad_cols = target_cols - new_cols
        new_matrix = torch.cat([new_matrix, torch.zeros(target_rows, pad_cols, device=accumulator.device, dtype=accumulator.dtype)], dim=1)

    # Now both matrices have same shape, so they can be added
    accumulator += new_matrix.nan_to_num()
    return accumulator


def LRPEmbeddingModelEval(model, tokenizer, engine, examples, labels, num_examples=200, masked_inds=None, run_lrp=True):
    """Runs LRP with the given model, tokenizer, and examples.
    By default will run on a subset of 200 examples of the given data, can be changed using the num_examples argument.
    Optionally can be given a list of masks, where each input must correspond to a unique mask in the masked_inds list by the same index.
    Optionally can run without LRP, to only get the T/F P/N values.
    
    Returns:
    - agg_checkpoint_vals : The summed relevances at each LRPCheckpoint, input dimension agnostic.
    - param_vals : The relevances at each Embedding layer, from each individual example. Not aggregated.
    - tpnfpn : A tuple containing (true_pos, true_neg, false_pos, false_neg)"""

    num_examples = min(len(examples), num_examples)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agg_checkpoint_vals = None
    param_vals = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in tqdm(range(num_examples)):
        inputs = tokenizer(examples[i], return_tensors = 'pt')["input_ids"]
        if masked_inds:
            inputs[0][masked_inds[i]] = tokenizer.pad_token_id
        outputs : torch.Tensor = model(inputs.to(device))
        logits = outputs[0]
        pred = logits.argmax()
        if pred == labels[i]:
            if pred:
                tp += 1
            else:
                tn += 1
        else:
            if pred:
                fp += 1
            else:
                fn += 1
        hidden_states = outputs.hidden_states

        if run_lrp:
            relevance_outputs = engine.run(hidden_states)

            param_vals.append(relevance_outputs[1][0])
            checkpoint_vals = relevance_outputs[0]
            if agg_checkpoint_vals is None:
                agg_checkpoint_vals = checkpoint_vals
            else:
                for j in range(len(agg_checkpoint_vals)):
                    accumulate_variable_matrix(agg_checkpoint_vals[j], checkpoint_vals[j])
    return agg_checkpoint_vals, param_vals, (tp, tn, fp, fn)