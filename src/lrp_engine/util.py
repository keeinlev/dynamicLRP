import torch
from functools import reduce
from typing import Tuple, Callable
from .relevance_filter import relevance_filter

DEBUG = False

# For denominator stability in relevance distribution
epsilon = 10e-12

# def renormalize_epsilon(rz, rx, ry):
#     """Renormalizes output relevances after dividing by a denominator with epsilon added to preserve conservation"""
#     scale = rz / (rx + ry)
#     return torch.nan_to_num(rx * scale, nan=0.0), torch.nan_to_num(ry * scale, nan=0.0)

# def renormalize_epsilon_scalar(rz, rx, ry):
#     """Renormalizes output relevances which have different shapes by using a scalar renormalization factor based on sums"""
#     scale = rz.nansum() / (rx.nansum() + ry.nansum())
#     return rx * scale, ry * scale

# def shift_and_renormalize(rz, rx, alpha=0.5):
#     """Shifts rx by alpha in the positive direction, then renormalizes rx to rz via scalar sums"""
#     # if rx.max() - rx.min() <= rz.sum():
#     #   return rx
#     rx = rx + alpha
#     scale = rz.nansum() / rx.nansum()
#     return rx * scale

def handle_neg_index(ind, seqlen):
    if ind < seqlen:
        return ind
    return ind - (1 << ind.bit_length())

# Handling AddMmBackward0 is not the exact same as just AddBackward0. The calculation of the in-relevances
# is slightly different because you need to consider the matmul after the addition is propagated.
# AddMmBackward0 also has 3 inputs, rather than 2 as in AddBackward0, so they will not be easily
# compatible with the current promise tree structure.
# I created the promise and promise tree structure with the aim to not have to re-visit the promise
# origin nodes after sending out the promises (we re-visit the promise tree nodes, but not the grad_fn
# nodes themselves). AddMmBackward0 would add a good amount of extra work if we were to handle it with
# promises, it would require a completely different promise completion handling sequence.
# It would be much easier to simply decompose an AddMmBackward0 into an AddBackward0 and a MmBackward0
# in our graph, then traverse using the normal AddBackward0 promises, where we can fill in the Mm side first.

# Since the autograd Nodes are code-generated and not exposed to the torch API, we just redefine shell
# classes for the ones we need to instantiate, with the fields we need according to the dir() of the
# original classes.

class InputMetadata():
    def __init__(self, shape):
        self.shape = shape

class DecomposedNode:
    def __init__(self, next_functions, sequence_nr, shapes):
        self.next_functions = next_functions
        self.metadata = {}
        self.sequence_nr = sequence_nr
        self.grad_fn = None

        if isinstance(shapes, (tuple, list)):
            if isinstance(shapes[0], int):
                shapes = [shapes]
        elif isinstance(shapes, torch.Size):
            shapes = [shapes]
        else:
            raise TypeError(f"Decomposed Node with sequence number {sequence_nr} was given an invalid shapes type {type(shapes)}. Either supply one tensor shape as tuple/torch.Size or a list/tuple of those types.")

        self._input_metadata = tuple(InputMetadata(shape) for shape in shapes)

    def __call__(self, *args):
        # grad_fn needs to be set by child class init, if we expect to use the grad_fn instead of custom functionality
        return self.grad_fn(*args)

    def _sequence_nr(self):
        return self.sequence_nr

class AddBackward0(DecomposedNode):
    def __init__(self, next_functions, sequence_nr, shape):
        super().__init__(next_functions, sequence_nr, shape)
        self.name = "AddBackward0"

class MmBackward0(DecomposedNode):
    def __init__(self, next_functions, sequence_nr, shape, mat1, mat2):
        super().__init__(next_functions, sequence_nr, shape)
        self.name = "MmBackward0"
        self._saved_self = mat1
        self._saved_self_sym_sizes = mat1.shape
        self._saved_mat2 = mat2
        self._saved_mat2_sym_sizes = mat2.shape

class DecomposedConvolutionBackward0(DecomposedNode):
    def __init__(self, next_functions, sequence_nr, shape, x, weight, stride, padding, dilation, groups):
        super().__init__(next_functions, sequence_nr, shape)
        self.name = "ConvolutionBackward0"
        self._saved_input = x
        self._saved_weight = weight
        self._saved_stride = stride
        self._saved_padding = padding
        self._saved_dilation = dilation
        self._saved_groups = groups

class AccumulateGrad(DecomposedNode):
    def __init__(self, sequence_nr, shape, variable):
        super().__init__([], sequence_nr, shape)
        self.name = "AccumulateGrad"
        self.variable = variable

class MulBackward0(DecomposedNode):
    def __init__(self, next_functions, sequence_nr, shape, t1, t2):
        super().__init__(next_functions, sequence_nr, shape)
        self.name = "MulBackward0"
        self._saved_self = t1
        self._saved_other = t2

class LRPCheckpoint(torch.autograd.Function):
    """Identity autograd fcn for marking where to capture relevance."""
    num_checkpoints_reached = 0

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        return grad_output, None, None

    @classmethod
    def save_val(cls, grad_fn, val):
        grad_fn.metadata["relevance"] = val
        grad_fn.metadata["checkpoint_ind"] = cls.num_checkpoints_reached
        cls.num_checkpoints_reached += 1

create_checkpoint = LRPCheckpoint.apply


def decompose_addmmbackward(grad_fn):
    """Assuming grad_fn is an instance of AddMmBackward, returns an AddBackward0 instance that is the parent
    of a new MmBackward0 instance and the first function in grad_fn.next_functions.
    The MmBackward0 is then the parent of the last two functions in grad_fn.next_functions."""
    result_shape = grad_fn._input_metadata[0].shape
    mm_fn = MmBackward0(grad_fn.next_functions[1:], grad_fn._sequence_nr(), result_shape, grad_fn._saved_mat1, grad_fn._saved_mat2)
    add_fn = AddBackward0((grad_fn.next_functions[0], (mm_fn, 0)), grad_fn._sequence_nr(), result_shape)

    return add_fn


def decompose_convbackward(grad_fn):
    """Assuming grad_fn is an instance of ConvolutionBackward0, returns an AddBackward0 instance that is the parent
    of a new ConvolutionBackward0 instance and the third function in grad_fn.next_functions.
    The MmBackward0 is then the parent of the first two functions in grad_fn.next_functions."""
    result_shape = grad_fn._input_metadata[0].shape
    bias_shape = (result_shape[1], *[ 1 for _ in result_shape[2:] ])
    conv_fn = DecomposedConvolutionBackward0((*grad_fn.next_functions[:2], (None, 0)), grad_fn._sequence_nr(), result_shape, grad_fn._saved_input, grad_fn._saved_weight, grad_fn._saved_stride, grad_fn._saved_padding, grad_fn._saved_dilation, grad_fn._saved_groups)
    bias_accumulate = AccumulateGrad(grad_fn._sequence_nr(), bias_shape, grad_fn.next_functions[2][0].variable.reshape((bias_shape)))
    add_fn = AddBackward0(((bias_accumulate, 0), (conv_fn, 0)), grad_fn._sequence_nr(), result_shape)

    return add_fn


def decompose_addcmulbackward(grad_fn):
    """Assuming grad_fn is an instance of AddcmulBackward, returns an AddBackward0 instance that is the parent
    of a new MulBackward0 instance and the first function in grad_fn.next_functions.
    The MulBackward0 is then the parent of the last two functions in grad_fn.next_functions."""
    result_shape = max((im.shape for im in grad_fn._input_metadata), key=lambda shape: (len(shape), tuple(shape))) # Largest dimensionality and largest dimensions, to account for broadcasting
    mul_fn = MulBackward0(grad_fn.next_functions[1:], grad_fn._sequence_nr(), result_shape, grad_fn._saved_tensor1, grad_fn._saved_tensor2)
    add_fn = AddBackward0((grad_fn.next_functions[0], (mul_fn, 0)), grad_fn._sequence_nr(), result_shape)

    return add_fn


def merge_input_shapes(grad_fn):
    shapes = [ im.shape for im in grad_fn._input_metadata ]
    fwd_shape = grad_fn._input_metadata[0].shape
    dim = handle_neg_index(grad_fn._saved_dim, len(fwd_shape))
    out_shape = list(shapes[0])
    out_shape = out_shape[:dim] + [len(shapes)] + out_shape[dim:]

    return out_shape

def epsilon_lrp_matmul(x: torch.Tensor, w: torch.Tensor, z: torch.Tensor, r: torch.Tensor, w_transposed=True, bilinear=False):
    sign = ((z == 0.).to(z) + z.sign())

    if bilinear:
        z_stabilized = 2 * z + epsilon * sign
    else:
        z_stabilized = z + epsilon * sign
    tmp = r / z_stabilized

    if w_transposed:
        # If the operation is x @ w, i.e. w is already transposed for matmul
        rin_input = x * (tmp @ w.transpose(-2, -1))
        rin_weight = w * (x.transpose(-2,-1) @ tmp)
    else:
        # If the operation is x @ w.T
        rin_input = x * (tmp @ w)
        rin_weight = w * (tmp.transpose(-2,-1) @ x)
    
    return rin_input, rin_weight

def gammma_lrp_matmul_grad(x: torch.Tensor, w: torch.Tensor, r: torch.Tensor, filter_val=1.0):
    """Analytical Linear LRP for a single (x, w, z, r) tuple"""
    rin_input = x * (r @ w.t()) # s d
    rin_input = rin_input

    rin_weight = w * (x.t() @ r) # s o
    rin_weight = rin_weight

    return relevance_filter(rin_input, filter_val), rin_weight

def gamma_lrp_conv2d_grad(conv_T: Callable, module_kwargs: dict, x: torch.Tensor, w: torch.Tensor, r: torch.Tensor):
    """Analytical Conv2d LRP for a single (x, w, z, r) tuple"""
    # r is already r_out / z_out from the caller
    c = conv_T(r, w, None, **module_kwargs)
    
    r_input = x * c
    
    grad_w = torch.nn.grad.conv2d_weight(x, w.shape, r, **{k:v for k,v in list(module_kwargs.items()) if k != "output_padding"})
    r_weight = w * grad_w

    return r_input, r_weight

def gamma_lrp_general(module_op: Callable, module_T_op: Callable, module_kwargs: dict, x: torch.Tensor, w: torch.Tensor, z: torch.Tensor, r: torch.Tensor, gamma, filter_val=1.0):
    """Analytical Gamma-LRP using clamped input and weights"""
    if not r.requires_grad:
        x = x.detach()
        w = w.detach()
        z = z.detach()
        r = r.detach()

    # Create clamped components, track each one individually for correct gamma-LRP
    x_pos = x.clamp(min=0.0)
    x_neg = x.clamp(max=0.0)
    w_pos = w + w.clamp(min=0.0) * gamma
    w_neg = w + w.clamp(max=0.0) * gamma

    # matmuls/convs
    z_pp = module_op(x_pos, w_pos, **module_kwargs)
    z_pn = module_op(x_pos, w_neg, **module_kwargs)
    z_np = module_op(x_neg, w_pos, **module_kwargs)
    z_nn = module_op(x_neg, w_neg, **module_kwargs)

    z_pos, z_neg = z_pp + z_nn, z_pn + z_np

    z_mask = (z > 0)
    z_neg_sign = ((z_neg == 0.).to(z_neg) + z_neg.sign())
    den1 = z_pos + epsilon
    den2 = z_neg + z_neg_sign * epsilon

    # Only divide within pos/neg groups, as defined by Gamma-LRP
    r_pos1 = r * z_mask / den1
    r_pos2 = r * z_mask / den1
    r_neg1 = r * ~z_mask / den2
    r_neg2 = r * ~z_mask / den2

    # Prep inputs
    xs = [x_pos, x_neg, x_pos, x_neg]
    ws = [w_pos, w_neg, w_neg, w_pos]
    rs = [r_pos1, r_pos2, r_neg1, r_neg2]
    if module_T_op is None:
        rins_x_w = [ gammma_lrp_matmul_grad(*args) for args in zip(xs, ws, rs) ]
    else:
        # Calculate output_padding to match input size (AI code, check here if things start to not work)
        stride = module_kwargs.get('stride', 1)
        padding = module_kwargs.get('padding', 0)
        dilation = module_kwargs.get('dilation', 1)
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        _, _, H_out, W_out = z.shape
        _, _, H_in, W_in = x.shape
        _, _, kH, kW = w.shape
        
        # Calculate expected input size from conv_transpose2d formula
        # H_in_expected = (H_out - 1) * stride + dilation * (kH - 1) + 1 - 2 * padding + output_padding
        output_padding_h = H_in - ((H_out - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kH - 1) + 1)
        output_padding_w = W_in - ((W_out - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kW - 1) + 1)
        module_kwargs["output_padding"] = (output_padding_h, output_padding_w)

        # Do the equivalent of grad calculations, without the autograd overhead
        rins_x_w = [ gamma_lrp_conv2d_grad(module_T_op, module_kwargs, *args) for args in zip(xs, ws, rs) ]

    # Sum over input/weights relevances respectively
    rins_x_w = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), rins_x_w)

    if module_T_op is None:
        return relevance_filter(rins_x_w[0], filter_val), rins_x_w[1]
    else:
        return relevance_filter(rins_x_w[0], filter_val), rins_x_w[1], 0.0

    # THE ABOVE IS MEANT TO BE EQUIVALENT TO THE FOLLOWING:
    # gradients = torch.autograd.grad(
    #     zs,
    #     xs,
    #     grad_outputs=rs,
    # )

    # r_x = sum(input * gradient for (input, gradient) in zip(xs, gradients)) / 2

    # w_gradients = torch.autograd.grad(
    #     zs,
    #     ws,
    #     grad_outputs=rs,
    # )

    # r_w = sum(weight * gradient for (weight, gradient) in zip(ws, w_gradients)) / 2

    # return relevance_filter(r_x, filter_val), r_w, 0.0
