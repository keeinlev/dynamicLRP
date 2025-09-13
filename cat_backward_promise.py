import torch
from promise import Promise

class CatBackwardPromise(Promise):
    """
    idx: specifies which argument/operand the branch is looking for.
    """
    def __init__(self, promise, traversal_ind, saved_dim, idx):
        super().__init__(promise, traversal_ind)
        self.dim = saved_dim
        if isinstance(saved_dim, int):
            self.dim = (saved_dim,)
        self.idx = idx

    @property
    def arg(self):
        return self.promise["args"][self.idx]
    
    @property
    def op_result(self):
        return torch.cat(self.promise["args"])

    @property
    def rin(self):
        return self.promise["rins"][self.idx]

    def set_rout(self, new_rout):
        self.promise["rout"] = new_rout

    def accumulate_rout(self, new_rout):
        assert type(new_rout) == torch.Tensor, f"New rout was not a tensor, but {type(new_rout)}"
        self.promise["rout"] = self.rout + new_rout

    def set_rin(self, new_rin):
        self.promise["rins"][self.idx] = new_rin

    def compute_rins(self):
        """Compute base branch relevances based on sum of squares ratios."""
        assert self.ready and self.pending_parents == 0
        self.set_rin(Promise.ind_to_node[self.start_ind](self.rout)[self.idx])

    def _setarg(self, value):
        self.promise["args"][self.idx] = value
