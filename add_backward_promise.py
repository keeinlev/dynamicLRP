import torch
from util import (
    epsilon,
    renormalize_epsilon,
    DEBUG,
)
from promise import Promise

class AddBackwardPromise(Promise):
    """
    idx: specifies which argument/operand the branch is looking for.
    """
    all_promises = []
    def __init__(self, promise, traversal_ind, idx):
        super().__init__(promise, traversal_ind)
        self.idx = idx

        if DEBUG:
            AddBackwardPromise.all_promises.append(self)

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
        assert self.ready and self.pending_parents == 0
        arg1, arg2 = self.promise["args"]
        r = self.promise["rout"]
        denom = arg1 ** 2 + arg2 ** 2 + epsilon
        r1 = (arg1 ** 2 / denom) * r
        r2 = (arg2 ** 2 / denom) * r
        self.promise["rins"][0], self.promise["rins"][1] = renormalize_epsilon(r, r1, r2)

    def _setarg(self, value):
        self.promise["args"][self.idx] = value
