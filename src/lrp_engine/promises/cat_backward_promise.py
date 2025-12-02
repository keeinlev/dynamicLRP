import torch
from .promise import (
    Promise,
    ensure_dtype
)
from ..util import handle_neg_index

class CatBackwardPromise(Promise):
    """
    idx: specifies which argument/operand the branch is looking for.
    """
    def __init__(self, promise, traversal_ind, bucket, saved_dim, idx):
        super().__init__(promise, traversal_ind, bucket)
        self.dim = handle_neg_index(saved_dim, len(self.fwd_shape))
        self.idx = idx

    @property
    def arg(self):
        return self.promise["args"][self.idx]
    
    @property
    def op_result(self):
        return torch.cat(self.promise["args"], dim=self.dim)

    @property
    def rin(self):
        return self.promise["rins"][self.idx]

    @ensure_dtype
    def set_rin(self, new_rin):
        self.promise["rins"][self.idx] = new_rin

    def compute_rins(self):
        """Compute using the internal grad_fn at the CatBackward Node."""
        assert self.ready and self.pending_parents == 0, f"Expected Promise {self.id}, {self} to be ready and have 0 pending parents, instead the following state was found: ready: {self.ready}, pending_parents: {self.pending_parents}"
        split_fcn = self.bucket.ind_to_node[self.start_ind]
        for i, split in enumerate(split_fcn(self.rout)):
            self.promise["rins"][i] = split

    @ensure_dtype
    def _setarg(self, value):
        self.promise["args"][self.idx] = value
