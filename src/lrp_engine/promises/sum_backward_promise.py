import torch
from ..util import (
    epsilon,
    renormalize_epsilon_scalar,
)
from .promise import Promise

class SumBackwardPromise(Promise):
    def __init__(self, promise, traversal_ind, bucket, saved_dim, keepdim):
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

    @property
    def arg(self):
        return self.promise["args"][0]
    
    @property
    def op_result(self):
        if self.sum_type == 1:
            return self.arg.sum(dim=self.dim, keepdim=self.keepdim)
        else:
            return self.arg.sum()

    @property
    def rin(self):
        return self.promise["rins"][0]

    def set_rout(self, new_rout):
        self.promise["rout"] = new_rout

    def set_rin(self, new_rin):
        self.promise["rins"][0] = new_rin

    def compute_rins(self):
        """Compute base branch relevances based on sum of squares ratios."""
        assert self.ready and self.pending_parents == 0, f"Expected Promise {self.id}, {self} to be ready and have 0 pending parents, instead the following state was found: ready: {self.ready}, pending_parents: {self.pending_parents}"
        argsquared = self.arg ** 2
        if self.sum_type == 0:
            ratios = argsquared / (argsquared.sum() + epsilon)
            contribs = ratios * self.rout
        else:
            ratios = argsquared / (argsquared.sum(dim=self.dim, keepdim=True) + epsilon)
            if self.keepdim:
                contribs = ratios * self.rout
            else:
                unsqueezed_shape = [1 if i in self.dim else s for i, s in enumerate(self.arg.shape)]
                contribs = ratios * self.rout.reshape(unsqueezed_shape)
        
        self.set_rin(renormalize_epsilon_scalar(self.rout, contribs, torch.zeros_like(contribs))[0])

    def _setarg(self, value):
        self.promise["args"][0] = value
