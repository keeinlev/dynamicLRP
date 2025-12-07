import torch
from .dummy_promise import DummyPromise
from ..util import handle_neg_index

class UnbindBackwardPromise(DummyPromise):

    def __init__(self, promise, traversal_ind, bucket, saved_dim):
        super().__init__(promise, traversal_ind, bucket)
        self.dim = handle_neg_index(saved_dim, len(self.fwd_shape))

    def op_result(self):
        return torch.unbind(self.arg, self.dim)
    
    def accumulate_rout(self, new_rout, parent_idx=None):
        """Unbind-specific rout accumulation"""
        assert parent_idx is not None, "UnbindBackwardPromise rout accumulation requires a parent idx to be given"
        idx = [slice(None)] * self.rout.shape
        idx[self.dim] = parent_idx
        self.rout[idx] = new_rout
