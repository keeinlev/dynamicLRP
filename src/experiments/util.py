import numpy as np
import torch
import torchvision.transforms as T
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc
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

def compute_abpc_metrics(x_axis, y_axis_morf, y_axis_lerf, occlusion_iters):
    # ABC
    abc = auc(x_axis, y_axis_lerf) - auc(x_axis, y_axis_morf)
    
    # Comprehensiveness
    baseline = y_axis_morf[0]
    comprehensiveness = auc(x_axis, np.full_like(y_axis_morf, baseline)) - auc(x_axis, y_axis_morf)
    
    # Sufficiency  
    sufficiency = auc(x_axis, np.full_like(y_axis_lerf, baseline)) - auc(x_axis, y_axis_lerf)
    print(abc / occlusion_iters, comprehensiveness / occlusion_iters, sufficiency / occlusion_iters)

def visualize_abpc(baseline, morf_preds, lerf_preds, patch_size, occlusion_iters, attr_name):
    x_axis = torch.arange(0, occlusion_iters + 1, 1) * 4
    y_axis_morf = [ p[0] / (p[0] + p[1]) for p in [baseline] + morf_preds ]
    y_axis_lerf = [ p[0] / (p[0] + p[1]) for p in [baseline] + lerf_preds ]
    
    plt.plot(x_axis, y_axis_morf, color="red")
    plt.plot(x_axis, y_axis_lerf, color="blue")
    plt.title(f"MoRF and LeRF perturbation curves of {attr_name} Attribution on VGG16")
    plt.xlabel(f"# {patch_size}x{patch_size} patches occluded")
    plt.ylabel("Accuracy")
    plt.show()

    compute_abpc_metrics(x_axis, y_axis_morf, y_axis_lerf, occlusion_iters)

def run_morf_lerf_occlusion(model, heatmaps, patch_size, dims, occlusion_step, num_samples, attr_name, eval_fcn, image_gen_fcn, baseline=None):
    print("Warning: dims expects (dimx, dimy) if giving a tuple")
    if isinstance(dims, tuple):
        assert len(dims == 2) and all(isinstance(d, int) for d in dims), "Expected dims to be either an int or tuple of two ints"
        dimx = dims[0]
        dimy = dims[1]
    elif isinstance(dims, int):
        dimx = dims
        dimy = dims
    else:
        raise TypeError("dims must be either an int or a tuple of 2 ints.")
    assert dimx % patch_size == 0 and dimy % patch_size == 0, f"boths dims x {dimx} and y {dimy} must be divisible by {patch_size}"

    num_patchesx = dimx // patch_size
    num_patchesy = dimy // patch_size
    max_patch_pool = torch.nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)
    # min_patch_pool = lambda x: -torch.nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)(x)
    # heatmap_pos_patches = [ max_patch_pool(hm.sum(dim=1)).squeeze(0) for hm in heatmaps ]
    heatmap_pos_patches = [ max_patch_pool(hm) for hm in heatmaps ]
    # heatmap_pos_patches = [ max_patch_pool(hm).squeeze(0) for hm in heatmaps ]
    # heatmap_pos_patches = [ max_patch_pool(hm.sum(dim=1)).squeeze(0) for _, hm in heatmaps ]
    # heatmap_neg_patches = [ min_patch_pool(hm.sum(dim=1)).squeeze(0) for _, hm in heatmaps ]
    patch_mask_morf = [ patches.flatten().topk(occlusion_step).indices.cpu() for patches in heatmap_pos_patches ]
    patch_mask_lerf = [ patches.flatten().topk(occlusion_step, largest=False).indices.cpu() for patches in heatmap_pos_patches ]

    occlusion_iters = ((dimx*dimy) / (patch_size**2)) // occlusion_step
    occlusion_iters = int(occlusion_iters)
    print(f"Running {occlusion_iters} occlusion iterations with {patch_size}x{patch_size} patches")

    # Load fresh images for MoRF occlusion
    morf_imgs_list, labels_list = image_gen_fcn(num_samples=num_samples)
    morf_preds = []

    if baseline is None:
        print("Computing baseline...")
        _, _, baseline = eval_fcn(model, None, morf_imgs_list, labels_list, run_lrp=False)
    print(f"Baseline correct/incorrect ratio is {baseline}")

    print("Generating blurred images")
    blur = T.GaussianBlur(kernel_size=51, sigma=20)
    blurred_imgs = [ blur(img) for img in morf_imgs_list ]

    # mean_pixels = [ img.mean(dim=[-1, -2]) for img in imgs_list ]
    for i in range(occlusion_iters):
        # Do occlusion
        for j in range(len(morf_imgs_list)):
            patch_inds = patch_mask_morf[j]
            for patch in patch_inds[-occlusion_step:]:
                x = (patch // num_patchesx) * patch_size
                y = (patch % num_patchesy) * patch_size
                # samplex = random.randint(0,223)
                # sampley = random.randint(0,223)
                # morf_imgs_list[j][:,x:x+patch_size,y:y+patch_size] =  morf_imgs_list[j][:,samplex,sampley].unsqueeze(1).unsqueeze(1)
                morf_imgs_list[j][:,x:x+patch_size,y:y+patch_size] = blurred_imgs[j][:,x:x+patch_size,y:y+patch_size]
    
        _, _, preds = eval_fcn(model, None, morf_imgs_list, labels_list, run_lrp=False)
        morf_preds.append(preds)
        print(i, preds)
        if i == occlusion_iters - 1:
            break
        patch_mask_morf = [ patches.flatten().topk((i + 2) * occlusion_step).indices.cpu() for patches in heatmap_pos_patches ]

    # Do everything again for LeRF
    lerf_imgs_list, _ = image_gen_fcn(num_samples=num_samples)
    lerf_preds = []
    for i in range(occlusion_iters):
        # Do occlusion
        for j in range(len(lerf_imgs_list)):
            patch_inds = patch_mask_lerf[j]
            for patch in patch_inds[-occlusion_step:]:
                x = (patch // num_patchesx) * patch_size
                y = (patch % num_patchesy) * patch_size
                # samplex = random.randint(0,223)
                # sampley = random.randint(0,223)
                # lerf_imgs_list[j][:,x:x+patch_size,y:y+patch_size] =  lerf_imgs_list[j][:,samplex,sampley].unsqueeze(1).unsqueeze(1)
                lerf_imgs_list[j][:,x:x+patch_size,y:y+patch_size] = blurred_imgs[j][:,x:x+patch_size,y:y+patch_size]
    
        _, _, preds = eval_fcn(model, None, lerf_imgs_list, labels_list, run_lrp=False)
        lerf_preds.append(preds)
        print(i, preds)
        if i == occlusion_iters - 1:
            break
        patch_mask_lerf = [ patches.flatten().topk((i + 2) * occlusion_step, largest=False).indices.cpu() for patches in heatmap_pos_patches ]

    visualize_abpc(baseline, morf_preds, lerf_preds, 16, 49, attr_name)
    return morf_preds, lerf_preds
