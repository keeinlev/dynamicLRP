import torch
from promise import Promise

class CatBackwardPromise(Promise):
    """
    idx: specifies which argument/operand the branch is looking for.
    """
    def __init__(self, promise, traversal_ind, saved_dim, idx):
        super().__init__(promise, traversal_ind)
        
        # Negative indices -i get saved as 2**32 - i
        if saved_dim > len(self.fwd_shape) - 1:
            saved_dim -= 2**32

        self.dim = saved_dim
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

    def set_rout(self, new_rout):
        self.promise["rout"] = new_rout

    def set_rin(self, new_rin):
        self.promise["rins"][self.idx] = new_rin

    def compute_rins(self):
        """Compute using the internal grad_fn at the CatBackward Node."""
        assert self.ready and self.pending_parents == 0, f"Expected Promise {self.id}, {self} to be ready and have 0 pending parents, instead the following state was found: ready: {self.ready}, pending_parents: {self.pending_parents}"
        split_fcn = Promise.ind_to_node[self.start_ind]
        for i, split in enumerate(split_fcn(self.rout)):
            self.promise["rins"][i] = split

    def _setarg(self, value):
        self.promise["args"][self.idx] = value
