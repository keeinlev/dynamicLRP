# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Taken from https://github.com/mosaicml/examples/blob/main/examples/benchmarks/bert/src/bert_padding.py
# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
# Which was adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py

"""Helper functions for padding and unpadding batches.

These functions are used extensively throughout the Mosaic BERT implementation
in `bert_layers.py`.
"""

from typing import Tuple, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(input: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """Get just the values of `input` which are at `indices`.

        Arguments:
            ctx: the autograd context object
            input: (b, ...) 2+ dimensional tensor
            indices: (num_idx) 1D tensor
        """
        assert input.ndim >= 2
        other_shape = input.shape[1:]
        second_dim = other_shape.numel(
        )  # product of sizes of all but first dimension
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        return torch.gather(
            rearrange(input, 'b ... -> b (...)'),  # (b, ...) -> (b, second_dim)
            0,
            repeat(indices, 'z -> z d',
                   d=second_dim)  # (indices,) -> (indices, second_dim)
        ).reshape(-1, *other_shape)  # (num_idx, ...)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        indices, = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, 'b ... -> b (...)')
        grad_input = torch.zeros([ctx.first_axis_dim, grad_output.shape[1]],
                                 device=grad_output.device,
                                 dtype=grad_output.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0,
                            repeat(indices, 'z -> z d', d=grad_output.shape[1]),
                            grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape)


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(values: torch.Tensor, indices: torch.Tensor,
                first_axis_dim) -> torch.Tensor:
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim,
                             *values.shape[1:],
                             device=values.device,
                             dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor) -> torch.Tensor:
        indices, = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values


index_put_first_axis = IndexPutFirstAxis.apply