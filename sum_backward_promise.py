import torch
from util import (
    epsilon,
    renormalize_epsilon_scalar,
)
from promise import Promise

class SumBackwardPromise(Promise):
    def __init__(self, promise, traversal_ind, saved_dim, keepdim):
        super().__init__(promise, traversal_ind)

        #### TODO: Need to handle negative indices (also in CatBackwardPromise)
        self.dim = saved_dim
        if isinstance(saved_dim, int):
            self.dim = (saved_dim,)
        self.keepdim = keepdim
        self.sum_type = int(saved_dim is not None)

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

    def accumulate_rout(self, new_rout):
        assert type(new_rout) == torch.Tensor, f"New rout was not a tensor, but {type(new_rout)}"
        self.promise["rout"] = self.rout + new_rout

    def set_rin(self, new_rin):
        self.promise["rins"][0] = new_rin

    def compute_rins(self):
        """Compute base branch relevances based on sum of squares ratios."""
        assert self.ready and self.pending_parents == 0
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
        
        self.set_rin(renormalize_epsilon_scalar(self.rout, contribs, 0.0)[0])
        
        # arg1, arg2 = self.promise["args"]
        # r = self.promise["rout"]
        # denom = arg1 ** 2 + arg2 ** 2 + epsilon
        # r1 = (arg1 ** 2 / denom) * r
        # r2 = (arg2 ** 2 / denom) * r
        # self.promise["rins"][0], self.promise["rins"][1] = renormalize_epsilon(r, r1, r2)

    def _setarg(self, value):
        self.promise["args"][0] = value
