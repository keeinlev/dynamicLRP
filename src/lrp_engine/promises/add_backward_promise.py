import torch
from ..util import (
    epsilon,
    # renormalize_epsilon,
)
from .promise import Promise

class AddBackwardPromise(Promise):
    """
    idx: specifies which argument/operand the branch is looking for.
    """
    def __init__(self, promise, traversal_ind, bucket, idx):
        super().__init__(promise, traversal_ind, bucket)
        self.idx = idx

    @property
    def arg1(self):
        return self.promise["args"][0]

    @property
    def arg2(self):
        return self.promise["args"][1]
    
    @property
    def arg(self):
        return self.promise["args"][self.idx]
    
    @property
    def op_result(self):
        """Returns the forward result of the operation when applied on the promise args"""
        return self.arg1 + self.arg2

    @property
    def rin(self):
        return self.promise["rins"][self.idx]

    def set_rout(self, new_rout):
        self.promise["rout"] = new_rout

    def set_rin(self, new_rin):
        self.promise["rins"][self.idx] = new_rin

    def compute_rins(self):
        """Compute base branch relevances based on sum of squares ratios."""
        assert self.ready and self.pending_parents == 0, f"Expected Promise {self.id}, {self} to be ready and have 0 pending parents, instead the following state was found: ready: {self.ready}, pending_parents: {self.pending_parents}"
        if (non_residual_idx := self.promise.get("non_residual_idx")) is not None:
            self.promise["rins"][non_residual_idx] = self.promise["rout"]
            self.promise["rins"][1 - non_residual_idx] = torch.zeros_like(self.promise["args"][1 - non_residual_idx])
        arg1, arg2 = self.promise["args"]
        r = self.promise["rout"]
        if not isinstance(arg1, float):
            arg1 = arg1.abs()
        if not isinstance(arg2, float):
            arg2 = arg2.abs()
        denom = arg1 + arg2 + epsilon
        r1 = (arg1 / denom) * r
        r2 = (arg2 / denom) * r
        self.promise["rins"][0], self.promise["rins"][1] = r1, r2#renormalize_epsilon(r, r1, r2)

    def _setarg(self, value):
        self.promise["args"][self.idx] = value
