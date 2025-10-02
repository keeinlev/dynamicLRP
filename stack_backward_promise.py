import torch
from cat_backward_promise import CatBackwardPromise

class StackBackwardPromise(CatBackwardPromise):
    """
    idx: specifies which argument/operand the branch is looking for.
    """
    def __init__(self, promise, traversal_ind, saved_dim, idx):
        super().__init__(promise, traversal_ind, saved_dim, idx)
    
    @property
    def op_result(self):
        return torch.stack(self.promise["args"], dim=self.dim)
