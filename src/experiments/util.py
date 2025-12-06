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
        logits = outputs.logits
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

        if run_lrp:
            relevance_outputs = engine.run(logits)

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

def default_model_eval_fcn(model, examples : list[torch.Tensor], labels : list[torch.Tensor]):
    preds = [0, 0]

    for i in tqdm(range(len(examples))):
        img_tensor = examples[i]
        true_label = labels[i]
        output = model(img_tensor)

        if isinstance(output, torch.Tensor):
            pred_idx = output.argmax(dim=-1)
        else:
            pred_idx = output.logits.argmax(dim=-1)
        if pred_idx == true_label:
            preds[0] += 1
        else:
            preds[1] += 1

    return preds


def run_morf_lerf_occlusion_patches(model, heatmaps, patch_size, dims, occlusion_step, num_samples, attr_name, image_gen_fcn, eval_fcn=default_model_eval_fcn, occlusion_iters=None, baseline=None, blurred_imgs=None):
    """Runs the MoRF/LeRF attribution evaluation and visualizations for pixel-wise image classification attributions, occluding in square patches.
    Requires that all images/heatmaps are of the same shape

    Returns:
    - morf_preds: a list of list[int, int], each element represents the accuracy of the model's predictions on the dataset at each occlusion iteration for MoRF
    - lerf_preds: same as above but for LeRF
    - blurred_imgs: the list of blurred images, either given or generated at call time, for future re-use

    Arguments:
    - model: the image model
    - heatmaps: iterable of attribution tensors for each example
        accepts shapes of [1, num_channels, dim0, dim1], [1 OR num_channels, dim0, dim1], [dim0, dim1] at EACH attribution tensor, where dim0 = dim1 = dims if dims is an int
    - patch_size: integer dimension for how big each occluded square patch should be
    - dims: integer or tuple of 2 integers representing the image dimensions (all images must be the same dimension)
    - occlusion_step: integer representing how many patches to occlude per iteration
    - num_samples: how many examples there are
    - attr_name: string name which will be outputted in the visualizations and filenames for results files
    - image_gen_fcn: a callable which takes no arguments, and returns:
        - an iterable of the examples that correspond to the attributions,
        - an iterable of the true labels associated with those examples
    - eval_fcn: a callable whose first return value is an tuple/list containing 2 integers: (# correct predictions, # incorrect predictions)
        eval_fcn must take in 3 positional arguments: model, examples : list[tensor], labels : list[int]
    - baseline: an iterable of two integers, same format as the returned value of eval_fcn, for the model prediction results of the unoccluded dataset
    - blurred_imgs: a list of images that have been blurred or occluded in a desired fashion"""

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

    if len(heatmaps[0].shape) == 2:
        heatmap_pos_patches = [ max_patch_pool(hm) for hm in heatmaps ]
    elif len(heatmaps[0].shape) == 3:
        heatmap_pos_patches = [ max_patch_pool(hm).sum(dim=0) for hm in heatmaps ]
    elif len(heatmaps[0].shape) == 4:
        heatmap_pos_patches = [ max_patch_pool(hm.sum(dim=1)).squeeze(0) for hm in heatmaps ]
    patch_mask_morf = [ patches.flatten().topk(occlusion_step).indices.cpu() for patches in heatmap_pos_patches ]
    patch_mask_lerf = [ patches.flatten().topk(occlusion_step, largest=False).indices.cpu() for patches in heatmap_pos_patches ]

    if occlusion_iters is None:
        occlusion_iters = ((dimx*dimy) / (patch_size**2)) // occlusion_step
        occlusion_iters = int(occlusion_iters)
    print(f"Running {occlusion_iters} occlusion iterations with {patch_size}x{patch_size} patches")

    # Load fresh images for MoRF occlusion
    morf_imgs_list, labels_list = image_gen_fcn(num_samples=num_samples)
    morf_preds = []

    if baseline is None:
        print("Computing baseline...")
        eval_output = eval_fcn(model, morf_imgs_list, labels_list)
        if isinstance(eval_output, tuple) and isinstance(eval_output[0], (tuple, list)):
            baseline = eval_output[0]
        else:
            baseline = eval_output
    print(f"Baseline correct/incorrect ratio is {baseline}")

    if blurred_imgs is None:
        print("Generating blurred images")
        blur = T.GaussianBlur(kernel_size=51, sigma=20)
        blurred_imgs = [ blur(img) for img in morf_imgs_list ]
    else:
        print("Blurred images provided!")

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

        eval_output = eval_fcn(model, morf_imgs_list, labels_list)
        if isinstance(eval_output, tuple) and isinstance(eval_output[0], (tuple, list)):
            preds = eval_output[0]
        else:
            preds = eval_output

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

        eval_output = eval_fcn(model, lerf_imgs_list, labels_list)
        if isinstance(eval_output, tuple) and isinstance(eval_output[0], (tuple, list)):
            preds = eval_output[0]
        else:
            preds = eval_output

        lerf_preds.append(preds)
        print(i, preds)
        if i == occlusion_iters - 1:
            break
        patch_mask_lerf = [ patches.flatten().topk((i + 2) * occlusion_step, largest=False).indices.cpu() for patches in heatmap_pos_patches ]

    visualize_abpc(baseline, morf_preds, lerf_preds, 16, 49, attr_name)
    return morf_preds, lerf_preds, blurred_imgs

def run_morf_lerf_occlusion_pixels(model, heatmaps, dims, occlusion_step, num_samples, attr_name, image_gen_fcn, eval_fcn=default_model_eval_fcn, occlusion_iters=None, baseline=None, blurred_imgs=None):
    """Runs the MoRF/LeRF attribution evaluation and visualizations for pixel-wise image classification attributions, occluding <occlusion_step> pixels each iteration.
    Requires that all images/heatmaps are of the same shape

    Returns:
    - morf_preds: a list of list[int, int], each element represents the accuracy of the model's predictions on the dataset at each occlusion iteration for MoRF
    - lerf_preds: same as above but for LeRF
    - blurred_imgs: the list of blurred images, either given or generated at call time, for future re-use

    Arguments:
    - model: the image model
    - heatmaps: iterable of attribution tensors for each example
        accepts shapes of [1, num_channels, dim0, dim1], [1 OR num_channels, dim0, dim1], [dim0, dim1] at EACH attribution tensor, where dim0 = dim1 = dims if dims is an int
    - dims: integer or tuple of 2 integers representing the image dimensions (all images must be the same dimension)
    - occlusion_step: integer representing how many pixels to occlude per iteration
    - num_samples: how many examples there are
    - attr_name: string name which will be outputted in the visualizations and filenames for results files
    - image_gen_fcn: a callable which takes no arguments, and returns:
        - an iterable of the examples that correspond to the attributions,
        - an iterable of the true labels associated with those examples
    - eval_fcn: a callable whose first return value is an iterable containing 2 integers: (# correct predictions, # incorrect predictions)
        eval_fcn must take in 3 positional arguments: model, examples : list[tensor], labels : list[int]
    - occlusion_iters: integer for how many occlusion iterations to run. If none given, occlusion will run to completion, i.e. until all the images are completely occluded
    - baseline: an iterable of two integers, same format as the returned value of eval_fcn, for the model prediction results of the unoccluded dataset
    - blurred_imgs: a list of images that have been blurred or occluded in a desired fashion"""

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


    if len(heatmaps[0].shape) == 2:
        heatmaps = [ hm for hm in heatmaps ]
    elif len(heatmaps[0].shape) == 3:
        heatmaps = [ hm.sum(dim=0) for hm in heatmaps ]
    elif len(heatmaps[0].shape) == 4:
        heatmaps = [ hm.sum(dim=1).squeeze(0) for hm in heatmaps ]
    else:
        raise ValueError(f"Expected one of [1, num_channels, dim0, dim1], [1 OR num_channels, dim0, dim1], or [dim0, dim1] for the shape of the elements in heatmaps, got {heatmaps[0].shape}")
    pixel_mask_morf = [ pixels.flatten().topk(occlusion_step).indices.cpu() for pixels in heatmaps ]
    pixel_mask_lerf = [ pixels.flatten().topk(occlusion_step, largest=False).indices.cpu() for pixels in heatmaps ]

    if occlusion_iters is None:
        occlusion_iters = (dimx*dimy) // occlusion_step
        occlusion_iters = int(occlusion_iters)
    print(f"Running {occlusion_iters} occlusion iterations with {occlusion_step} pixels each iteration.")

    # Load fresh images for MoRF occlusion
    morf_imgs_list, labels_list = image_gen_fcn(num_samples=num_samples)
    morf_preds = []

    if baseline is None:
        print("Computing baseline...")
        eval_output = eval_fcn(model, morf_imgs_list, labels_list)
        if isinstance(eval_output, tuple) and isinstance(eval_output[0], (tuple, list)):
            baseline = eval_output[0]
        else:
            baseline = eval_output
    print(f"Baseline correct/incorrect ratio is {baseline}")

    if blurred_imgs is None:
        print("Generating blurred images")
        blur = T.GaussianBlur(kernel_size=51, sigma=20)
        blurred_imgs = [ blur(img) for img in morf_imgs_list ]
    else:
        print("Blurred images provided!")

    for i in range(occlusion_iters):
        # Do occlusion
        for j in range(len(morf_imgs_list)):
            pixel_inds = pixel_mask_morf[j]
            x_inds = []
            y_inds = []
            for pixel in pixel_inds[-occlusion_step:]:
                x_inds.append(pixel // dimx)
                y_inds.append(pixel % dimy)
            morf_imgs_list[j][:, :,x_inds, y_inds] = blurred_imgs[j][:, :,x_inds, y_inds]

        eval_output = eval_fcn(model, morf_imgs_list, labels_list)
        if isinstance(eval_output, tuple) and isinstance(eval_output[0], (tuple, list)):
            preds = eval_output[0]
        else:
            preds = eval_output

        morf_preds.append(preds)
        print(i, preds)
        if i == occlusion_iters - 1:
            break
        pixel_mask_morf = [ pixels.flatten().topk((i + 2) * occlusion_step).indices.cpu() for pixels in heatmaps ]

    # Do everything again for LeRF
    lerf_imgs_list, _ = image_gen_fcn(num_samples=num_samples)
    lerf_preds = []
    for i in range(occlusion_iters):
        # Do occlusion
        for j in range(len(lerf_imgs_list)):
            pixel_inds = pixel_mask_lerf[j]
            x_inds = []
            y_inds = []
            for pixel in pixel_inds[-occlusion_step:]:
                x_inds.append(pixel // dimx)
                y_inds.append(pixel % dimy)
            lerf_imgs_list[j][:, :,x_inds, y_inds] = blurred_imgs[j][:, :,x_inds, y_inds]

        eval_output = eval_fcn(model, lerf_imgs_list, labels_list)
        if isinstance(eval_output, tuple) and isinstance(eval_output[0], (tuple, list)):
            preds = eval_output[0]
        else:
            preds = eval_output

        lerf_preds.append(preds)
        print(i, preds)
        if i == occlusion_iters - 1:
            break
        pixel_mask_lerf = [ pixels.flatten().topk((i + 2) * occlusion_step, largest=False).indices.cpu() for pixels in heatmaps ]

    visualize_abpc(baseline, morf_preds, lerf_preds, 1, occlusion_iters, attr_name)
    return morf_preds, lerf_preds, blurred_imgs
