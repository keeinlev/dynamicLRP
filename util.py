from typing import Tuple
import torch

DEBUG = False

# For denominator stability in relevance distribution
epsilon = 10e-6

def renormalize_epsilon(rz, rx, ry):
    """Renormalizes output relevances after dividing by a denominator with epsilon added to preserve conservation"""
    scale = rz / (rx + ry)
    return torch.nan_to_num(rx * scale, nan=0.0), torch.nan_to_num(ry * scale, nan=0.0)

def renormalize_epsilon_scalar(rz, rx, ry):
    """Renormalizes output relevances which have different shapes by using a scalar renormalization factor based on sums"""
    scale = rz.nansum() / (rx.nansum() + ry.nansum())
    return rx * scale, ry * scale

def shift_and_renormalize(rz, rx, alpha=0.5):
    """Shifts rx by alpha in the positive direction, then renormalizes rx to rz via scalar sums"""
    # if rx.max() - rx.min() <= rz.sum():
    #   return rx
    rx = rx + alpha
    scale = rz.nansum() / rx.nansum()
    return rx * scale


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


def merge_input_shapes(grad_fn):
    shapes = [ im.shape for im in grad_fn._input_metadata ]
    dim = grad_fn._saved_dim
    out_shape = list(shapes[0])
    out_shape = out_shape[:dim] + [len(shapes)] + out_shape[dim:]

    return out_shape

