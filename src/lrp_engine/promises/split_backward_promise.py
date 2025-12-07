import torch
from .dummy_promise import DummyPromise
from ..util import handle_neg_index

class SplitBackwardPromise(DummyPromise):

    def __init__(self, promise, traversal_ind, bucket, saved_dim, saved_slice_size):
        assert saved_dim is not None, f"SplitBackwardPromise at topo-ind {traversal_ind} got None for dim"
        assert saved_slice_size is not None, f"SplitBackwardPromise at topo-ind {traversal_ind} got None for slice size"
        super().__init__(promise, traversal_ind, bucket)
        self.dim = handle_neg_index(saved_dim, len(self.fwd_shape))
        self.slice_size = saved_slice_size

    def op_result(self):
        return torch.split(self.arg, self.slice_size, self.dim)
    
    def accumulate_rout(self, new_rout, parent_idx=None):
        """Split-specific rout accumulation"""
        assert parent_idx is not None, "SplitBackwardPromise rout accumulation requires a parent idx to be given"

        idx = [slice(None)] * self.rout.shape
        idx[self.dim] = slice(slice_start := (parent_idx * self.slice_size), slice_start + self.slice_size)
        self.rout[idx] = new_rout
