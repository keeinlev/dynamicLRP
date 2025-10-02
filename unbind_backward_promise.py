import torch
from dummy_promise import DummyPromise

class UnbindBackwardPromise(DummyPromise):

    def __init__(self, promise, traversal_ind, saved_dim):
        super().__init__(promise, traversal_ind)
        
        # Negative indices -i get saved as 2**32 - i
        if saved_dim > len(self.fwd_shape) - 1:
            saved_dim -= 2**32

        self.dim = saved_dim

    def op_result(self):
        return torch.unbind(self.arg, self.dim)
    
    def accumulate_rout(self, new_rout, parent_idx=None):
        """Unbind-specific rout accumulation"""
        assert parent_idx is not None, "UnbindBackwardPromise rout accumulation requires a parent idx to be given"
        idx = [slice(None)] * self.rout.shape
        idx[self.dim] = parent_idx
        self.rout[idx] = new_rout
