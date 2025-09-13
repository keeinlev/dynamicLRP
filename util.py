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

class AddBackward0:
    def __init__(self, next_functions, sequence_nr, shape):
        self.name = "AddBackward0"
        self.next_functions = next_functions
        self.metadata = {}
        self.sequence_nr = sequence_nr
        self._input_metadata = (InputMetadata(shape),)
    
    def _sequence_nr(self):
        return self.sequence_nr
class MmBackward0:
    def __init__(self, next_functions, mat1, mat2, sequence_nr, shape):
        self.name = "MmBackward0"
        self.next_functions = next_functions
        self._saved_self = mat1
        self._saved_self_sym_sizes = mat1.shape
        self._saved_mat2 = mat2
        self._saved_mat2_sym_sizes = mat2.shape
        self.metadata = {}
        self.sequence_nr = sequence_nr
        self._input_metadata = (InputMetadata(shape),)
    
    def _sequence_nr(self):
        return self.sequence_nr


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
        grad_fn.metadata["checkpoint_relevance"] = val
        grad_fn.metadata["checkpoint_ind"] = cls.num_checkpoints_reached
        cls.num_checkpoints_reached += 1

create_checkpoint = LRPCheckpoint.apply


def decompose_addmmbackward(grad_fn):
    """Assuming grad_fn is an instance of AddMmBackward, returns an AddBackward0 instance that is the parent
    of an MmBackward0 instance and the first function in grad_fn.next_functions.
    The MmBackward0 is then the parent of the last two functions in grad_fn.next_functions."""
    mat1_shape = list(grad_fn._saved_mat1.shape)
    mat2_shape = list(grad_fn._saved_mat2.shape)
    result_shape = mat1_shape[:-2] + [mat1_shape[-2]] + [mat2_shape[-1]]
    mm_fn = MmBackward0(grad_fn.next_functions[1:], grad_fn._saved_mat1, grad_fn._saved_mat2, grad_fn._sequence_nr(), result_shape)
    add_fn = AddBackward0((grad_fn.next_functions[0], (mm_fn, 0)), grad_fn._sequence_nr(), result_shape)

    return add_fn
