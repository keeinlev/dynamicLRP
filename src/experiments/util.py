import torch
import seaborn as sns
import matplotlib.pyplot as plt

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