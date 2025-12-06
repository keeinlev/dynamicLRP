import torch
from ..util import (
    epsilon,
    # renormalize_epsilon_scalar,
)
from .dummy_promise import DummyPromise

class SumBackwardPromise(DummyPromise):
    def __init__(self, promise, traversal_ind, bucket, saved_dim, keepdim, is_mean=False):
        super().__init__(promise, traversal_ind, bucket)

        if saved_dim is not None:
            if isinstance(saved_dim, int):
                saved_dim = [saved_dim]
            elif isinstance(saved_dim, tuple):
                saved_dim = list(saved_dim)
            
            for i in range(len(saved_dim)):
                # Negative indices -i get saved as 2**32 - i
                if saved_dim[i] > len(self.fwd_shape) - 1:
                    saved_dim[i] -= 2**32

            self.dim = tuple(saved_dim)
            self.sum_type = 1
        else:
            self.dim = None
            self.sum_type = 0
        self.keepdim = keepdim
        self.is_mean = is_mean
    
    @property
    def op_result(self):
        if self.sum_type == 1:
            if self.is_mean:
                return self.arg.mean(dim=self.dim, keepdim=self.keepdim)
            return self.arg.sum(dim=self.dim, keepdim=self.keepdim)
        else:
            if self.is_mean:
                return self.arg.mean()
            return self.arg.sum()

    def compute_rins(self):
        """Compute base branch relevances based on sum of squares ratios."""
        assert self.ready and self.pending_parents == 0, f"Expected Promise {self.id}, {self} to be ready and have 0 pending parents, instead the following state was found: ready: {self.ready}, pending_parents: {self.pending_parents}"
        argabs = self.arg.abs()
        if self.sum_type == 0:
            ratios = argabs / (argabs.sum() + epsilon)
            contribs = ratios * self.rout
        else:
            ratios = argabs / (argabs.sum(dim=self.dim, keepdim=True) + epsilon)
            if self.keepdim:
                contribs = ratios * self.rout
            else:
                unsqueezed_shape = [1 if i in self.dim else s for i, s in enumerate(self.arg.shape)]
                contribs = ratios * self.rout.reshape(unsqueezed_shape)
        
        self.set_rin(contribs)#renormalize_epsilon_scalar(self.rout, contribs, torch.zeros_like(contribs))[0])
